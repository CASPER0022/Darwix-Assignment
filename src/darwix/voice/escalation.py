"""Human escalation and the mock CRM business action.

Escalation triggers are code, not model judgement, because every one of them is
a case where the model has an incentive to keep talking:

  explicit_request     - the customer asked for a person. Non-negotiable.
  repeated_low_confidence - the KB could not answer 3 turns in a row.
  distress             - frustration or complaint language, or the ombudsman.
  repeated_refusal     - two refusals; a third attempt is harassment.
  repeated_unclear     - ASR failed 3 turns running; the line is the problem.

The business action (the assessment's optional item) is implemented as a mock
CRM write plus a callback booking plus an escalation webhook, so a completed
call produces something a sales desk could actually act on.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from ..common.config import settings
from ..common.logging import log

# Deliberately narrow. Suspicion that the call itself is a scam is NOT distress:
# in the Philippines and Indonesia it is the single most common opening
# objection, and both locale packs carry a documented trust-repair response for
# it (OBJ-PH-01 / the no-payment disclosure). Treating it as distress escalated
# a Taglish test call on turn three and the bot never got to answer the thing it
# was built to answer.
#
# What stays here is escalation the agent genuinely must not attempt to handle:
# a formal complaint, a regulator, legal action, or harassment.
DISTRESS_PATTERNS = re.compile(
    r"\b(complaint|complain|ombudsman|consumer court|legal notice|lawyer|harass|"
    r"sue|rbi|nodal officer|stop calling|do not call|reklamo|demanda|"
    r"pengaduan|lapor polisi)\b",
    re.I,
)

# Suspicion the call is fraudulent -> handled as an objection, from the KB.
SCAM_SUSPICION = re.compile(
    r"\b(scam|fraud|fake|is this real|prank|penipuan|tipu|palsu|manloloko)\b", re.I)


@dataclass
class Escalation:
    reason: str
    detail: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict:
        return {"reason": self.reason, "detail": self.detail, "created_at": self.created_at}


def check_triggers(state, *, intent: str, sentiment: str, utterance: str) -> Escalation | None:
    if intent == "request_human":
        return Escalation("explicit_request", "Customer asked to speak to a person.")
    if DISTRESS_PATTERNS.search(utterance or ""):
        return Escalation("distress", "Complaint or escalation language detected in the utterance.")
    if sentiment == "frustrated" and state.low_confidence_streak >= 2:
        # Two unanswered questions, not one. A customer saying "berat banget"
        # (money is tight) reads as frustrated and is exactly who the payment
        # -support path exists for - handing them straight to a human on the
        # first stumble abandons the conversation the bot is built to have.
        return Escalation("distress", "Frustration alongside repeated unanswered questions.")
    if state.low_confidence_streak >= 3:
        return Escalation("repeated_low_confidence",
                          "Three consecutive questions the knowledge base could not answer.")
    if state.refusal_count >= 2:
        return Escalation("repeated_refusal", "Customer declined twice; no third attempt.")
    if state.unclear_streak >= 3:
        return Escalation("repeated_unclear", "Three consecutive unintelligible turns.")
    return None


def next_callback_slot(now: datetime | None = None) -> str:
    """Respect the collections-conduct window in the credit policy: contact is
    permitted 08:00-19:00 local time only."""
    now = now or datetime.now()
    candidate = now + timedelta(hours=2)
    if candidate.hour >= 19:
        candidate = (candidate + timedelta(days=1)).replace(hour=10, minute=0)
    elif candidate.hour < 8:
        candidate = candidate.replace(hour=10, minute=0)
    return candidate.strftime("%A at %-I %p") if hasattr(candidate, "strftime") else str(candidate)


def _fmt_slot(dt: datetime) -> str:
    hour = dt.hour % 12 or 12
    ampm = "am" if dt.hour < 12 else "pm"
    return dt.strftime("%A") + " at " + str(hour) + " " + ampm


def suggest_callback(now: datetime | None = None) -> str:
    now = now or datetime.now()
    candidate = now + timedelta(hours=2)
    if candidate.hour >= 19:
        candidate = (candidate + timedelta(days=1)).replace(hour=10, minute=0)
    elif candidate.hour < 8:
        candidate = candidate.replace(hour=10, minute=0)
    return _fmt_slot(candidate)


def write_lead(state, decision: dict | None, *, escalation: Escalation | None = None,
               callback_slot: str = "") -> dict:
    """Mock CRM write. One JSON line per call, in the shape a real lead API
    would take.

    Note what is NOT written: no raw audio path in the CRM row, and the
    transcript reference is a file id rather than the text, so the CRM copy
    never becomes a second uncontrolled store of what the customer said.
    """
    settings.crm_dir.mkdir(parents=True, exist_ok=True)
    lead = {
        "lead_id": "LD-" + state.call_id[-8:].upper(),
        "call_id": state.call_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "channel": "ai_voice_agent",
        "locale": state.locale,
        "consent_captured": state.consent,
        "disclosures_given": sorted(state.disclosures_given),
        "slots": state.slots,
        "qualification": decision or {},
        "disposition": (decision or {}).get("disposition", "incomplete"),
        "escalation": escalation.as_dict() if escalation else None,
        "callback_slot": callback_slot,
        "transcript_ref": "data/transcripts/" + state.call_id + ".json",
        "turns": len(state.turns),
    }
    path = settings.crm_dir / "leads.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(lead, ensure_ascii=False, default=str) + "\n")
    log("crm.lead_written", lead_id=lead["lead_id"], disposition=lead["disposition"],
        escalated=bool(escalation))
    return lead


async def post_webhook(payload: dict) -> bool:
    """Fire the escalation webhook if one is configured. A failure here must
    never break the call - the lead is already durably written to the CRM file
    before this runs."""
    url = settings.escalation_webhook_url
    if not url:
        log("escalation.webhook_skipped", reason="no ESCALATION_WEBHOOK_URL configured")
        return False
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, json=payload)
        log("escalation.webhook_sent", status=r.status_code)
        return r.status_code < 400
    except Exception as exc:  # noqa: BLE001
        log("escalation.webhook_failed", error=str(exc)[:200])
        return False
