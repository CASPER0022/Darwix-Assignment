"""Qualification logic.

The rules are *data*, loaded from the qualification matrix that is itself a KB
source document. They are not in the system prompt and not in this file.

That matters for two reasons the assessment calls out directly:

* "do not hardcode all FAQs, objections, or policies in the system prompt" -
  changing a turnover cut-off is a CSV edit and a KB rebuild, with a version
  bump and an audit trail, not a prompt edit.
* An LLM asked to compare "forty five lakhs" against a policy threshold will
  usually get it right and will occasionally not. Eligibility arithmetic is
  deterministic here; the model's only job is to collect the values and speak
  the outcome.

Dispositions: pass / refer / reject. `refer` exists because a real credit desk
does not answer every borderline file with "no" - and a bot that says "no" to a
qualified-with-review customer destroys more value than one that hedges.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..common.config import settings
from ..common.logging import log

RULES_PATH = settings.raw_dir / "internal" / "qualification_rules.csv"

# Slots the agent collects, in the order it asks for them. Order is a
# conversation-design decision: cheap, non-sensitive questions first, so a
# caller who drops out early has still told us something useful.
SLOT_ORDER = [
    "business_name",
    "entity_type",
    "industry",
    "business_vintage_months",
    "annual_turnover_inr",
    "loan_amount_inr",
    "loan_purpose",
    "existing_emi_inr",
    "gst_registered",
    "city",
    "applicant_age",
    "credit_score",
]

REQUIRED_FOR_DECISION = [
    "entity_type",
    "business_vintage_months",
    "annual_turnover_inr",
    "loan_amount_inr",
]

# What the agent actually walks through on the call. Shorter than SLOT_ORDER on
# purpose: a qualification call that asks twelve questions gets hung up on.
# `city`, `applicant_age` and `credit_score` are still captured if the customer
# volunteers them (they are in the normalisers), and the rules that depend on
# them simply do not fire otherwise.
ASK_SEQUENCE = [
    "business_name",
    "entity_type",
    "industry",
    "business_vintage_months",
    "annual_turnover_inr",
    "loan_amount_inr",
    "loan_purpose",
    "existing_emi_inr",
    "gst_registered",
]

NUMERIC_SLOTS = {
    "business_vintage_months", "annual_turnover_inr", "loan_amount_inr",
    "existing_emi_inr", "applicant_age", "credit_score",
}


@dataclass
class Rule:
    rule_id: str
    product: str
    slot: str
    operator: str
    value: str
    disposition: str
    reason_code: str
    customer_message: str

    def evaluate(self, actual: Any) -> bool:
        if actual is None or actual == "":
            return False
        op = self.operator
        try:
            if op == "in":
                return str(actual).strip().lower() in {v.strip().lower() for v in self.value.split("|")}
            if op == "eq":
                return str(actual).strip().lower() == self.value.strip().lower()
            if op in {"gte", "gt", "lt", "lte"}:
                a, b = float(actual), float(self.value)
                return {"gte": a >= b, "gt": a > b, "lt": a < b, "lte": a <= b}[op]
            if op in {"between", "outside"}:
                lo, hi = (float(x) for x in self.value.split("|"))
                inside = lo <= float(actual) <= hi
                return inside if op == "between" else not inside
        except (TypeError, ValueError):
            return False
        return False


@dataclass
class Decision:
    disposition: str  # pass | refer | reject | incomplete
    reason_codes: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    rules_fired: list[str] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "disposition": self.disposition,
            "reason_codes": self.reason_codes,
            "rules_fired": self.rules_fired,
            "missing_slots": self.missing_slots,
            "conflicts": self.conflicts,
            "customer_messages": self.messages,
        }


class QualificationEngine:
    def __init__(self, product: str = "unsecured_business_loan", rules_path: Path | None = None) -> None:
        self.product = product
        self.rules = self._load(rules_path or RULES_PATH)
        log("qualification.loaded", rules=len(self.rules), product=product)

    @staticmethod
    def _load(path: Path) -> list[Rule]:
        if not path.exists():
            raise FileNotFoundError(
                "Qualification rules not found at " + str(path)
                + ". Run: python -m darwix.kb.ingest.seed_internal_docs"
            )
        with path.open("r", encoding="utf-8", newline="") as fh:
            return [Rule(**row) for row in csv.DictReader(fh)]

    def slots_for_product(self) -> list[str]:
        return [s for s in SLOT_ORDER]

    def next_question_slot(self, slots: dict[str, Any]) -> str | None:
        """Next slot to ask about.

        A slot explicitly present with value None has been asked and abandoned;
        it is skipped rather than asked again. A slot that is simply absent has
        never been asked.
        """
        for slot in ASK_SEQUENCE:
            if slot not in slots:
                return slot
        return None

    def evaluate(self, slots: dict[str, Any]) -> Decision:
        """Three states, not two.

        * a required slot ABSENT from the dict  -> not asked yet -> incomplete
        * a required slot present as None       -> asked, customer could not or
          would not answer -> the file is decidable, but never as a `pass`
        * a required slot with a value          -> evaluate the rules
        """
        not_asked = [s for s in REQUIRED_FOR_DECISION if s not in slots]
        if not_asked:
            return Decision(disposition="incomplete", missing_slots=not_asked)
        unknown = [s for s in REQUIRED_FOR_DECISION if slots.get(s) in (None, "")]

        fired_reject: list[Rule] = []
        fired_refer: list[Rule] = []
        fired_pass: list[Rule] = []
        for rule in self.rules:
            if rule.product != self.product:
                continue
            actual = slots.get(rule.slot)
            if actual in (None, ""):
                continue
            if not rule.evaluate(actual):
                continue
            {"reject": fired_reject, "refer": fired_refer, "pass": fired_pass}[rule.disposition].append(rule)

        # A reject anywhere is decisive; otherwise any refer makes the file a
        # refer. This ordering is the credit policy's, not the model's.
        if fired_reject:
            chosen, disposition = fired_reject, "reject"
        elif fired_refer:
            chosen, disposition = fired_refer, "refer"
        else:
            chosen, disposition = fired_pass, "pass"

        reason_codes = [r.reason_code for r in chosen]
        # A required value the customer never gave cannot be scored as a pass.
        # Downgrading to `refer` is the conservative direction: a human looks at
        # it, rather than the bot implying an outcome it has no basis for.
        if unknown and disposition == "pass":
            disposition = "refer"
            reason_codes = reason_codes + ["UNVERIFIED_" + s.upper() for s in unknown]

        return Decision(
            disposition=disposition,
            reason_codes=reason_codes,
            messages=[r.customer_message for r in chosen],
            rules_fired=[r.rule_id for r in chosen],
            missing_slots=unknown,
            conflicts=detect_conflicts(slots),
        )


def detect_conflicts(slots: dict[str, Any]) -> list[str]:
    """Catch internally inconsistent answers.

    The assessment requires an "incomplete or conflicting details" test case.
    A caller who says the business is three years old and then that it started
    last year is not lying - they are usually distinguishing the trade from the
    registration - but the agent has to notice and ask, rather than silently
    scoring one of the two answers.
    """
    out: list[str] = []
    turnover = _num(slots.get("annual_turnover_inr"))
    amount = _num(slots.get("loan_amount_inr"))
    vintage = _num(slots.get("business_vintage_months"))
    emi = _num(slots.get("existing_emi_inr"))

    if turnover and amount and amount > turnover:
        out.append("requested_amount_exceeds_annual_turnover")
    if vintage is not None and vintage > 600:
        out.append("implausible_business_vintage")
    if turnover and emi and (emi * 12) > turnover:
        out.append("existing_obligations_exceed_turnover")
    if slots.get("gst_registered") is False and turnover and turnover > 4_000_000:
        out.append("turnover_above_gst_threshold_but_not_registered")
    return out


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarise_for_agent(decision: Decision) -> str:
    """One line the dialogue model is allowed to paraphrase - and nothing more.

    The model never sees the raw rules, so it cannot invent a threshold.
    """
    if decision.disposition == "incomplete":
        return "Decision not possible yet. Still missing: " + ", ".join(decision.missing_slots) + "."
    head = {
        "pass": "The customer meets the documented criteria for an in-principle review.",
        "refer": "The file is borderline and must be reviewed by a credit officer.",
        "reject": "The file does not meet the documented criteria.",
    }[decision.disposition]
    body = " ".join(decision.messages[:3])
    conflict = ""
    if decision.conflicts:
        conflict = (" Note these inconsistencies in what the customer said, and ask about them "
                    "before closing: " + ", ".join(decision.conflicts) + ".")
    return head + " " + body + conflict
