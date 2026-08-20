"""Deterministic signal detection.

Two tiers exist because they answer different questions:

* This tier is **immediate** (microseconds) and **certain**. A disclosure either
  was or was not said. A customer either did or did not say the word "second
  vehicle". No model, no latency, no false confidence.
* The LLM tier (signals/llm.py) handles what rules cannot: frustration building
  over three turns, an implied buying signal, a missed opportunity nobody named
  explicitly.

Running rules first also means the most compliance-critical signals - the ones
where a miss is a regulatory problem rather than a lost sale - never depend on
a model call that might time out.

Compliance is modelled as a *checklist with a deadline*, not a keyword search:
each required disclosure must appear in the agent's speech before a given point
in the call, and the signal fires when the deadline passes without it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from ...common.logging import log


@dataclass
class Signal:
    kind: str                 # compliance_gap | missed_cross_sell | frustration | ...
    detail: str
    confidence: float
    speaker: str = ""
    evidence: str = ""
    at_call_time_s: float = 0.0
    source: str = "rules"

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "confidence": round(self.confidence, 2),
            "speaker": self.speaker,
            "evidence": self.evidence[:200],
            "at_call_time_s": round(self.at_call_time_s, 1),
            "source": self.source,
        }


# --------------------------------------------------------------------------
# keyword families
# --------------------------------------------------------------------------
def _rx(*words: str) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(words) + r")\b", re.I)


CROSS_SELL = {
    "second_vehicle": _rx("second (car|vehicle|bike|truck)", "another (car|vehicle|truck|bike)",
                          "two (cars|vehicles|trucks)", "mobil kedua", "kendaraan lain"),
    "second_property": _rx("another (shop|property|godown|warehouse|outlet)",
                           "second (shop|property|branch)", "new branch", "another location"),
    "expansion": _rx("expand", "expansion", "new machine", "another machine", "scaling up",
                     "grow the business", "buka cabang", "mag-expand"),
    "family_cover": _rx("my (wife|husband|son|daughter|family)", "for my kids",
                        "asawa ko", "anak ko", "istri saya", "anak saya"),
}

PAYMENT_DIFFICULTY = _rx(
    "can'?t pay", "cannot pay", "no money", "short of (funds|cash)", "tight (this )?month",
    "lost my job", "business is down", "slow (season|month)", "delay the payment",
    "wala akong pambayad", "walang pera", "hindi ko kaya", "belum ada uang",
    "belum ada dana", "tidak sanggup", "berat bulan ini", "nggak ada uang",
)

FRUSTRATION = _rx(
    "third time", "again and again", "already told you", "not listening", "waste of time",
    "ridiculous", "fed up", "annoyed", "frustrated", "keeps? happening", "no one helps",
    "pang-ilang beses", "nakakainis", "sawa na ako", "sudah berkali-kali", "capek",
    "kesal", "gimana sih",
)

BUYING_SIGNAL = _rx(
    "how (soon|fast|quickly)", "soon can i", "when can i get", "when do i get", "what documents", "how do i apply",
    "send me the", "what'?s the next step", "i'?m interested", "let'?s do it",
    "kailan po", "anong requirements", "kapan bisa", "syaratnya apa", "gimana caranya",
)

COMPETITOR = _rx(
    "bajaj", "hdfc", "icici", "axis", "kotak", "tata capital", "lendingkart", "indifi",
    "another (lender|bank|company)", "other (lender|bank)", "competitor",
    "ibang bangko", "leasing lain", "finance lain",
)

CALLBACK_REQUEST = _rx(
    "call me (back|later)", "call me tomorrow", "ring me", "later i will call",
    "tawagan mo ako", "tawag na lang", "telepon lagi", "hubungi lagi", "nanti saya kabari",
)

HUMAN_REQUEST = _rx(
    "real person", "speak to (a|an) (human|person|agent|officer)", "talk to someone",
    "transfer me", "tao naman", "sa tao", "bicara dengan orang", "petugas manusia",
)

RISKY_STATEMENT = _rx(
    # Phrasing matters more than vocabulary here: the first version of this
    # pattern only matched "guaranteed approval" and missed the agent actually
    # saying "your approval is guaranteed, I can promise you that".
    "guarantee[ds]?", "100 ?% approval", "definitely (approved|get it|approve)",
    "i (can )?promise", "promise you", "assured (approval|sanction)",
    "no interest at all", "free money", "(i|we)('ll| will)? waive",
    "waive the (penalty|charge|fee|denda)", "no processing fee for you",
    "pasti (disetujui|cair)", "dijamin cair", "sigurado(ng)? approved",
)

SOFT_REFUSAL_ID = _rx(
    "nanti saya kabari", "nanti deh", "iya nanti", "belum ada kabar", "lihat nanti",
    "diusahakan", "kalau ada rezeki",
)


@dataclass
class ComplianceItem:
    id: str
    description: str
    pattern: re.Pattern
    deadline_s: float           # must be said by this point in the call
    speaker: str = "agent"


# Deadlines are conversation-design decisions: an AI disclosure after 60 seconds
# of questioning is not a disclosure, it is a confession.
COMPLIANCE_CHECKLIST = [
    ComplianceItem(
        "ai_disclosure",
        "Agent must disclose it is an AI before collecting information",
        _rx("ai assistant", "not a human", "hindi po tao", "asisten ai", "bukan petugas manusia",
            "automated assistant", "virtual assistant"),
        deadline_s=45.0,
    ),
    ComplianceItem(
        "recording_disclosure",
        "Agent must disclose the call is recorded",
        _rx("recorded", "recording", "naka-record", "direkam"),
        deadline_s=45.0,
    ),
    ComplianceItem(
        "no_payment_disclosure",
        "Agent must state it will not ask for OTP or payment on the call",
        _rx("will not ask for", "hindi po ako hihingi", "tidak akan meminta", "otp"),
        deadline_s=60.0,
    ),
]


class RuleSignals:
    """Stateful across a call: compliance deadlines and repetition need history."""

    def __init__(self, *, checklist: list[ComplianceItem] | None = None,
                 require_payment_disclosure: bool = False) -> None:
        items = checklist if checklist is not None else list(COMPLIANCE_CHECKLIST)
        if not require_payment_disclosure:
            items = [i for i in items if i.id != "no_payment_disclosure"]
        self.checklist = items
        self.satisfied: set[str] = set()
        self.fired_gaps: set[str] = set()
        self.customer_turns: list[str] = []
        self.agent_turns: list[str] = []

    def observe(self, speaker: str, text: str, call_time_s: float) -> list[Signal]:
        signals: list[Signal] = []
        low = text.lower()

        if speaker == "agent":
            self.agent_turns.append(text)
            for item in self.checklist:
                if item.id not in self.satisfied and item.pattern.search(low):
                    self.satisfied.add(item.id)
                    log("signal.compliance_met", item=item.id, at=round(call_time_s, 1))
            for m in RISKY_STATEMENT.finditer(low):
                signals.append(Signal(
                    "risky_statement",
                    "Agent may have promised an outcome it cannot guarantee.",
                    0.9, speaker, m.group(0), call_time_s,
                ))
            return signals

        self.customer_turns.append(text)

        for name, pattern in CROSS_SELL.items():
            m = pattern.search(low)
            if m:
                signals.append(Signal(
                    "missed_cross_sell",
                    "Customer mentioned " + name.replace("_", " ") + ".",
                    0.75, speaker, m.group(0), call_time_s,
                ))
        for pattern, kind, detail, conf in (
            (PAYMENT_DIFFICULTY, "payment_difficulty",
             "Customer signalled they cannot pay right now.", 0.85),
            (FRUSTRATION, "frustration", "Customer used frustration language.", 0.7),
            (BUYING_SIGNAL, "buying_signal",
             "Customer asked about next steps or requirements.", 0.7),
            (COMPETITOR, "competitor_mention", "Customer named another lender.", 0.8),
            (CALLBACK_REQUEST, "callback_request", "Customer asked to be called back.", 0.8),
            (HUMAN_REQUEST, "human_request", "Customer asked for a human.", 0.95),
            (SOFT_REFUSAL_ID, "soft_refusal",
             "Indirect Indonesian refusal - not a promise to pay.", 0.75),
        ):
            m = pattern.search(low)
            if m:
                signals.append(Signal(kind, detail, conf, speaker, m.group(0), call_time_s))

        # Repetition: the same question three times means the agent is failing.
        if len(self.customer_turns) >= 3:
            recent = [_norm(t) for t in self.customer_turns[-3:]]
            if len(set(recent)) == 1 and recent[0]:
                signals.append(Signal(
                    "repeated_question",
                    "Customer has asked the same thing three times.",
                    0.85, speaker, text, call_time_s,
                ))
        return signals

    def check_deadlines(self, call_time_s: float) -> list[Signal]:
        out: list[Signal] = []
        for item in self.checklist:
            if item.id in self.satisfied or item.id in self.fired_gaps:
                continue
            if call_time_s >= item.deadline_s:
                self.fired_gaps.add(item.id)
                out.append(Signal(
                    "compliance_gap",
                    item.description,
                    0.95, "agent",
                    "not detected by " + str(int(item.deadline_s)) + "s",
                    call_time_s,
                ))
        return out

    def summary(self) -> dict:
        return {
            "compliance_satisfied": sorted(self.satisfied),
            "compliance_missed": sorted(self.fired_gaps),
            "customer_turns": len(self.customer_turns),
            "agent_turns": len(self.agent_turns),
        }


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()
