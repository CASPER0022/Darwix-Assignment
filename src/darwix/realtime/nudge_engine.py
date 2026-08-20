"""Signal -> nudge, with the suppression that makes it usable.

The assessment lists "excessive low-value alerts" as a rejection condition, and
that is the correct instinct: a real agent on a live call can read maybe one
short prompt every 20-30 seconds. A system that fires eleven is worse than one
that fires none, because the agent learns to ignore the panel entirely.

So the engine's job is mostly *refusal*. Every control below exists to drop
something:

  threshold      - below `NUDGE_MIN_CONFIDENCE`, never shown.
  dedupe         - the same signal kind + near-identical text fires once.
  cooldown       - a topic that fired cannot fire again for `NUDGE_COOLDOWN_SECONDS`,
                   however many times it is detected.
  grouping       - kinds that mean the same thing to an agent share one cooldown
                   bucket ("payment_difficulty" and "cannot pay now" are one nudge).
  max active     - only `NUDGE_MAX_ACTIVE` on screen; a new high-priority nudge
                   evicts the lowest-priority active one rather than stacking.
  expiry         - a nudge about something said 90 seconds ago is noise; it
                   disappears on its own.
  end-of-call    - nudges are worthless after the call, so anything not shown
                   in time is dropped, not queued.

Priority is fixed by business consequence, not by model confidence: a missed
disclosure outranks a cross-sell hint even when the cross-sell is more certain.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field

from ..common.config import settings
from ..common.logging import log
from .signals.rules import Signal

# lower number = more important
PRIORITY = {
    "compliance_gap": 1,
    "risky_statement": 1,
    "risk": 1,
    "human_request": 2,
    "payment_difficulty": 2,
    "rising_frustration": 2,
    "frustration": 2,
    "repeated_question": 3,
    "soft_refusal": 3,
    "competitor_mention": 3,
    "confusion": 3,
    "buying_signal": 4,
    "missed_cross_sell": 4,
    "missed_opportunity": 4,
    "callback_request": 4,
    "topic_shift": 5,
}
DEFAULT_PRIORITY = 5

# Per-kind confidence floors, above the global threshold.
#
# Added after a measured false positive: on the deliberately uneventful test
# call, the LLM layer reported "customer needs to check with their accountant"
# as a missed opportunity at 0.80 confidence and it was shown. It is not wrong
# exactly - it is just not worth interrupting an agent for.
#
# The principle: the bar scales with the cost of being wrong in the *other*
# direction. Missing a compliance gap is expensive, so it keeps the low global
# floor. Missing a speculative opportunity costs almost nothing, so it has to be
# nearly certain before it earns a slot on the agent's screen.
MIN_CONFIDENCE_BY_KIND = {
    "missed_opportunity": 0.88,
    "topic_shift": 0.88,
    "confusion": 0.85,
    "buying_signal": 0.75,
}

# Kinds that an agent would experience as "the same nudge" share a bucket, so
# the rules layer and the LLM layer cannot both fire for one situation.
TOPIC_GROUPS = {
    # Deliberately two buckets, not one: "you skipped a disclosure" and "you
    # just promised something you cannot" require different corrective actions,
    # so one must not silence the other via the shared cooldown.
    "compliance_gap": "compliance_disclosure",
    "risky_statement": "compliance_conduct",
    "risk": "compliance_conduct",
    "frustration": "sentiment",
    "rising_frustration": "sentiment",
    "repeated_question": "sentiment",
    "payment_difficulty": "payment",
    "soft_refusal": "payment",
    "missed_cross_sell": "opportunity",
    "missed_opportunity": "opportunity",
    "buying_signal": "opportunity",
    "competitor_mention": "objection",
    "callback_request": "logistics",
    "human_request": "logistics",
    "topic_shift": "context",
    "confusion": "context",
}

# The words the agent actually reads. Short imperatives - an agent mid-sentence
# cannot parse a paragraph.
TEMPLATES = {
    "compliance_gap": "Required disclosure missing: {detail} Say it before continuing.",
    "risky_statement": "Careful - that sounded like a guarantee. Re-frame: decision rests with credit.",
    "risk": "Compliance risk: {detail}",
    "missed_cross_sell": "Opportunity: {detail} Mention the relevant product before closing.",
    "missed_opportunity": "Opportunity: {detail}",
    "buying_signal": "Buying signal: {detail} Move to next steps now.",
    "payment_difficulty": "Payment difficulty. Offer the approved support path - do not promise a waiver.",
    "frustration": "Frustration rising. Acknowledge the concern before asking anything else.",
    "rising_frustration": "Frustration building. Acknowledge it, then slow down.",
    "repeated_question": "Third time asked. Answer it directly or escalate to a human.",
    "soft_refusal": "That was an indirect no. Pin a specific date or log it as no commitment.",
    "competitor_mention": "Competitor named. Compare on total cost and turnaround, do not disparage.",
    "callback_request": "Callback requested. Confirm a specific slot before ending.",
    "human_request": "Customer asked for a human. Stop qualifying and hand off.",
    "confusion": "Customer seems lost. Re-explain in one plain sentence.",
    "topic_shift": "Topic moved on: {detail} Follow the customer.",
}


@dataclass
class Nudge:
    id: str
    kind: str
    group: str
    text: str
    priority: int
    confidence: float
    created_at: float
    expires_at: float
    call_time_s: float
    evidence: str = ""
    source: str = "rules"
    latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "group": self.group,
            "text": self.text,
            "priority": self.priority,
            "confidence": round(self.confidence, 2),
            "call_time_s": round(self.call_time_s, 1),
            "evidence": self.evidence[:160],
            "source": self.source,
            "latency_ms": round(self.latency_ms, 1),
            "ttl_s": round(self.expires_at - self.created_at, 1),
        }


@dataclass
class SuppressionStats:
    below_threshold: int = 0
    duplicate: int = 0
    cooldown: int = 0
    evicted: int = 0
    expired: int = 0
    emitted: int = 0

    def as_dict(self) -> dict:
        total_in = (self.below_threshold + self.duplicate + self.cooldown + self.emitted)
        return {
            "signals_considered": total_in,
            "emitted": self.emitted,
            "suppressed_below_threshold": self.below_threshold,
            "suppressed_duplicate": self.duplicate,
            "suppressed_cooldown": self.cooldown,
            "evicted_for_priority": self.evicted,
            "expired_unread": self.expired,
            "suppression_rate_pct": round(
                100.0 * (total_in - self.emitted) / max(1, total_in), 1
            ),
        }


class NudgeEngine:
    def __init__(
        self,
        *,
        min_confidence: float | None = None,
        cooldown_s: float | None = None,
        ttl_s: float | None = None,
        max_active: int | None = None,
    ) -> None:
        self.min_confidence = min_confidence if min_confidence is not None else settings.nudge_min_confidence
        self.cooldown_s = cooldown_s if cooldown_s is not None else settings.nudge_cooldown_seconds
        self.ttl_s = ttl_s if ttl_s is not None else settings.nudge_ttl_seconds
        self.max_active = max_active if max_active is not None else settings.nudge_max_active

        self.active: list[Nudge] = []
        self.history: list[Nudge] = []
        self.stats = SuppressionStats()
        self._last_group_at: dict[str, float] = {}
        self._seen_hashes: set[str] = set()

    # ------------------------------------------------------------------ core
    def consider(self, signal: Signal, *, now: float | None = None,
                 detection_latency_ms: float = 0.0) -> Nudge | None:
        now = now if now is not None else time.perf_counter()
        self.expire(now)

        floor = max(self.min_confidence, MIN_CONFIDENCE_BY_KIND.get(signal.kind, 0.0))
        if signal.confidence < floor:
            self.stats.below_threshold += 1
            log("nudge.suppressed", reason="below_threshold", kind=signal.kind,
                confidence=round(signal.confidence, 2), floor=floor)
            return None

        group = TOPIC_GROUPS.get(signal.kind, signal.kind)
        text = self._render(signal)
        fingerprint = hashlib.md5(
            (signal.kind + "|" + _norm(text)).encode("utf-8")
        ).hexdigest()[:12]

        if fingerprint in self._seen_hashes:
            self.stats.duplicate += 1
            log("nudge.suppressed", reason="duplicate", kind=signal.kind)
            return None

        last = self._last_group_at.get(group)
        if last is not None and (now - last) < self.cooldown_s:
            self.stats.cooldown += 1
            log("nudge.suppressed", reason="cooldown", group=group,
                since_s=round(now - last, 1))
            return None

        priority = PRIORITY.get(signal.kind, DEFAULT_PRIORITY)
        nudge = Nudge(
            id=fingerprint,
            kind=signal.kind,
            group=group,
            text=text,
            priority=priority,
            confidence=signal.confidence,
            created_at=now,
            expires_at=now + self.ttl_s,
            call_time_s=signal.at_call_time_s,
            evidence=signal.evidence,
            source=signal.source,
            latency_ms=detection_latency_ms,
        )

        if len(self.active) >= self.max_active:
            weakest = max(self.active, key=lambda n: (n.priority, -n.created_at))
            if weakest.priority <= nudge.priority:
                # nothing on screen is less important than this one
                self.stats.cooldown += 1
                log("nudge.suppressed", reason="screen_full", kind=signal.kind)
                return None
            self.active.remove(weakest)
            self.stats.evicted += 1
            log("nudge.evicted", evicted=weakest.kind, for_=nudge.kind)

        self._seen_hashes.add(fingerprint)
        self._last_group_at[group] = now
        self.active.append(nudge)
        self.active.sort(key=lambda n: (n.priority, n.created_at))
        self.history.append(nudge)
        self.stats.emitted += 1
        log("nudge.emitted", kind=nudge.kind, priority=nudge.priority,
            confidence=round(nudge.confidence, 2), text=nudge.text[:90])
        return nudge

    def expire(self, now: float | None = None) -> list[Nudge]:
        now = now if now is not None else time.perf_counter()
        expired = [n for n in self.active if n.expires_at <= now]
        if expired:
            self.active = [n for n in self.active if n.expires_at > now]
            self.stats.expired += len(expired)
        return expired

    def _render(self, signal: Signal) -> str:
        template = TEMPLATES.get(signal.kind, "{detail}")
        detail = signal.detail.strip()
        if detail and not detail.endswith((".", "!", "?")):
            detail += "."
        return " ".join(template.format(detail=detail).split())

    # ----------------------------------------------------------------- report
    def report(self) -> dict:
        return {
            "nudges": [n.as_dict() for n in self.history],
            "suppression": self.stats.as_dict(),
            "config": {
                "min_confidence": self.min_confidence,
                "min_confidence_by_kind": MIN_CONFIDENCE_BY_KIND,
                "cooldown_s": self.cooldown_s,
                "ttl_s": self.ttl_s,
                "max_active": self.max_active,
            },
        }


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()
