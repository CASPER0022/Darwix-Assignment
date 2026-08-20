"""Conversation flow and control.

The flow is an explicit state machine, not an instruction to a model. What the
LLM is allowed to decide is narrow and auditable:

    LLM decides   : what the customer meant, and which values they gave
    CODE decides  : what happens next, what gets asked, what gets said

Why: a prompt-driven agent will skip a mandatory disclosure under conversational
pressure, re-ask a question the customer already answered, or quietly continue
after being asked for a human. Each of those is a compliance or CX failure that
you cannot fix by adding another sentence to the prompt. Here, the disclosure
gate, the escalation trigger and the qualification arithmetic are code.

Phases:
    greeting -> consent -> disclosure -> qualification -> decision -> next_step -> closed

Any phase can be interrupted by a question, an objection, or a request for a
human; the interrupt is handled and the flow resumes where it left off.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

LOCALE_DIR = Path(__file__).parent / "locales"


class Phase(str, Enum):
    GREETING = "greeting"
    CONSENT = "consent"
    DISCLOSURE = "disclosure"
    QUALIFICATION = "qualification"
    DECISION = "decision"
    NEXT_STEP = "next_step"
    ESCALATED = "escalated"
    CLOSED = "closed"


class Intent(str, Enum):
    ANSWER = "answer"                 # giving information / answering the question asked
    QUESTION = "question"             # asking something factual
    OBJECTION = "objection"           # pushing back on price, trust, timing
    REQUEST_HUMAN = "request_human"
    REFUSAL = "refusal"               # not interested / stop calling
    OUT_OF_SCOPE = "out_of_scope"
    CONFIRM = "confirm"
    DENY = "deny"
    SMALLTALK = "smalltalk"
    UNCLEAR = "unclear"


def load_pack(locale_dir: str) -> dict:
    path = LOCALE_DIR / locale_dir / "pack.yaml"
    if not path.exists():
        raise FileNotFoundError("No locale pack at " + str(path))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass
class Turn:
    speaker: str            # "agent" | "customer"
    text: str
    phase: str = ""
    intent: str = ""
    latency_ms: float = 0.0
    grounded: dict | None = None
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        row = {"speaker": self.speaker, "text": self.text, "phase": self.phase}
        if self.intent:
            row["intent"] = self.intent
        if self.latency_ms:
            row["latency_ms"] = round(self.latency_ms, 1)
        if self.grounded:
            row["grounded"] = self.grounded
        if self.meta:
            row["meta"] = self.meta
        return row


@dataclass
class CallState:
    call_id: str
    locale: str
    phase: Phase = Phase.GREETING
    slots: dict[str, Any] = field(default_factory=dict)
    disclosures_given: set[str] = field(default_factory=set)
    consent: bool | None = None
    turns: list[Turn] = field(default_factory=list)
    decision: dict | None = None
    escalation: dict | None = None
    low_confidence_streak: int = 0
    refusal_count: int = 0
    unclear_streak: int = 0
    asked_slots: list[str] = field(default_factory=list)
    pending_slot: str | None = None
    questions_asked: list[str] = field(default_factory=list)
    conflicts_probed: list[str] = field(default_factory=list)
    # Indirect refusals ("nanti saya kabari") and whether the customer ever
    # named a real date. Kept apart from `slots` because this is not a fact
    # about the customer - it is what the call is allowed to claim it achieved.
    soft_refusals: list[str] = field(default_factory=list)
    commitment: str = ""
    commitment_probed: bool = False
    awaiting_commitment: bool = False
    ended_reason: str = ""

    @property
    def ended(self) -> bool:
        return self.phase in (Phase.CLOSED, Phase.ESCALATED)


UNDERSTAND_SYSTEM = """You are the understanding layer of a phone agent for {brand} ({sector}, {market}).
You do NOT reply to the customer. You only report what they said.

The agent had just said: {last_agent_utterance}
It was expecting an answer about: {expected_slot}

Return JSON only:
{{"intent": "answer|question|objection|request_human|refusal|out_of_scope|confirm|deny|smalltalk|unclear",
  "language": "<bcp47 tag of what the customer actually spoke, e.g. en, fil, tl, id>",
  "slots": {{}},
  "kb_query": "<a search query if they asked something factual, else empty>",
  "sentiment": "positive|neutral|negative|frustrated",
  "verbatim_concern": "<their objection in their own words, else empty>"}}

Slot keys you may fill, only when the customer actually stated them. Put the value in
verbatim as spoken - the units are converted later, so "6 years" and "72 lakhs" are
correct values to return exactly as written:
{slot_keys}

Rules:
- "intent": "answer" when they are giving information the agent asked for.
- "intent": "objection" for price/trust/timing pushback, even if phrased as a question.
- "intent": "request_human" whenever they ask for a person, agent, officer or branch.
- "intent": "out_of_scope" when the topic is not this product or this company.
- Do not invent slot values. Do not convert units. Copy what they said."""


# Bare key names lost slots in testing: the model saw `business_vintage_months`,
# was told not to convert units, and silently dropped "we've been running about
# six years" rather than return years under a months-named key. Describing each
# slot in words fixed it.
SLOT_DESCRIPTIONS = {
    "business_name": 'name of the business, e.g. "Sharada Textiles"',
    "entity_type": 'legal structure as they said it, e.g. "proprietorship", "pvt ltd"',
    "industry": 'what the business does, e.g. "textile trading"',
    "business_vintage_months": 'how long the business has been running, AS SPOKEN, e.g. "6 years", "18 months", "since 2019"',
    "annual_turnover_inr": 'last year turnover AS SPOKEN, e.g. "72 lakhs", "1.2 crore"',
    "loan_amount_inr": 'amount they want to borrow AS SPOKEN, e.g. "15 lakh"',
    "loan_purpose": 'what the money is for, e.g. "working capital"',
    "existing_emi_inr": 'current monthly loan instalment AS SPOKEN, e.g. "42 thousand"',
    "gst_registered": 'whether GST registered - "yes" or "no"',
    "city": "city the business operates in",
    "applicant_age": 'their age in years, e.g. "41"',
    "credit_score": 'CIBIL / bureau score, e.g. "760"',
}


def flow_slots(pack: dict) -> list[dict]:
    """The slot sequence for this locale, declared in the pack.

    The markets ask genuinely different questions: India qualifies a business
    loan, the Philippines verifies a policy and its premium, Indonesia verifies
    a financing contract and its instalment. Hardcoding one slot list and
    relabelling it per market is how a "multilingual" bot ends up asking an
    insurance customer for their GST registration.
    """
    return pack.get("flow", {}).get("slots", [])


def build_understand_prompt(pack: dict, state: CallState, expected_slot: str | None) -> str:
    last_agent = next((t.text for t in reversed(state.turns) if t.speaker == "agent"), "(nothing yet)")
    known = sorted(state.slots.keys())
    declared = flow_slots(pack)
    if declared:
        slot_lines = "\n".join(
            "  " + s["key"] + ": " + s.get("describe", s.get("ask", ""))
            for s in declared if s["key"] not in known
        ) or "  (all slots already collected)"
    else:
        slot_lines = "\n".join(
            "  " + k + ": " + v for k, v in SLOT_DESCRIPTIONS.items() if k not in known
        ) or "  (all slots already collected)"
    prompt = UNDERSTAND_SYSTEM.format(
        brand=pack["brand"]["display_name"],
        sector=pack["sector"],
        market=pack["market"],
        last_agent_utterance=last_agent[:240],
        expected_slot=expected_slot or "(no specific slot)",
        slot_keys=slot_lines,
    )
    if known:
        # Already-known slots are removed from the list above AND named here,
        # so the model stops re-reporting them and the agent stops re-asking.
        prompt += "\n\nAlready known, do not ask about or re-report: " + ", ".join(known) + "."
    return prompt


def pending_disclosures(pack: dict, state: CallState, before: str) -> list[dict]:
    return [
        d for d in pack.get("disclosures", [])
        if d.get("required_before") == before and d["id"] not in state.disclosures_given
    ]


def render(template: str, pack: dict, **extra: Any) -> str:
    values = {
        "agent_name": pack["brand"]["agent_name"],
        "brand": pack["brand"]["display_name"],
        "callback_slot": extra.pop("callback_slot", "tomorrow morning"),
        "phone": extra.pop("phone", "this number"),
    }
    values.update(extra)
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", str(value))
    return " ".join(out.split())
