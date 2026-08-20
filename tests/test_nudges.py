"""Nudge suppression and qualification rules.

The suppression logic is what stops Q4 becoming alert spam, and the
qualification engine is what stops the LLM inventing an eligibility decision.
Both are pure functions of their inputs, so both are tested exactly.
"""
import pytest

from darwix.realtime.nudge_engine import MIN_CONFIDENCE_BY_KIND, NudgeEngine
from darwix.realtime.signals.rules import RuleSignals, Signal
from darwix.voice.qualification import QualificationEngine, detect_conflicts


def sig(kind="payment_difficulty", conf=0.9, detail="Customer cannot pay.", t=10.0):
    return Signal(kind=kind, detail=detail, confidence=conf, speaker="customer",
                  evidence="no money", at_call_time_s=t)


# ---------------------------------------------------------------- threshold
def test_low_confidence_is_suppressed():
    e = NudgeEngine(min_confidence=0.6)
    assert e.consider(sig(conf=0.4)) is None
    assert e.stats.below_threshold == 1


def test_per_kind_floor_is_stricter_than_global():
    # A speculative opportunity has to be nearly certain; a compliance gap does
    # not. Same engine, same confidence, different outcome.
    e = NudgeEngine(min_confidence=0.55)
    assert MIN_CONFIDENCE_BY_KIND["missed_opportunity"] > 0.55
    assert e.consider(sig(kind="missed_opportunity", conf=0.8)) is None
    assert e.consider(sig(kind="compliance_gap", conf=0.8)) is not None


# ------------------------------------------------------------------- dedupe
def test_identical_signal_fires_once():
    e = NudgeEngine()
    assert e.consider(sig()) is not None
    assert e.consider(sig()) is None
    assert e.stats.duplicate == 1


def test_cooldown_blocks_same_topic_group():
    e = NudgeEngine(cooldown_s=45)
    assert e.consider(sig(kind="payment_difficulty")) is not None
    # different kind, same group -> still suppressed
    assert e.consider(sig(kind="soft_refusal", detail="Indirect no.")) is None
    assert e.stats.cooldown >= 1


def test_disclosure_and_conduct_are_separate_groups():
    # A skipped disclosure and a risky promise need different corrective
    # actions, so one must not silence the other.
    e = NudgeEngine()
    assert e.consider(sig(kind="risky_statement", conf=0.9, detail="Promised approval.")) is not None
    assert e.consider(sig(kind="compliance_gap", conf=0.95, detail="No AI disclosure.")) is not None


# ----------------------------------------------------------------- capacity
def test_screen_cap_evicts_only_lower_priority():
    e = NudgeEngine(max_active=2, cooldown_s=0)
    assert e.consider(sig(kind="missed_cross_sell", conf=0.9, detail="Second vehicle.")) is not None
    assert e.consider(sig(kind="callback_request", conf=0.9, detail="Call back.")) is not None
    # a P1 must displace a P4 rather than be dropped
    high = e.consider(sig(kind="compliance_gap", conf=0.95, detail="Missing disclosure."))
    assert high is not None
    assert len(e.active) <= 2
    assert e.stats.evicted == 1


def test_low_priority_does_not_evict_when_screen_is_full():
    e = NudgeEngine(max_active=1, cooldown_s=0)
    e.consider(sig(kind="compliance_gap", conf=0.95, detail="Missing disclosure."))
    assert e.consider(sig(kind="callback_request", conf=0.9, detail="Call back.")) is None


def test_nudges_expire():
    e = NudgeEngine(ttl_s=5)
    e.consider(sig(), now=100.0)
    assert len(e.active) == 1
    e.expire(now=106.0)
    assert e.active == []
    assert e.stats.expired == 1


# -------------------------------------------------------------- rule layer
def test_compliance_gap_fires_only_after_the_deadline():
    r = RuleSignals()
    assert r.check_deadlines(10.0) == []
    gaps = r.check_deadlines(60.0)
    assert {g.detail for g in gaps}
    assert all(g.kind == "compliance_gap" for g in gaps)


def test_disclosure_satisfied_prevents_the_gap():
    r = RuleSignals()
    r.observe("agent", "Just so you know, I'm an AI assistant, and this call is recorded.", 5.0)
    assert r.check_deadlines(60.0) == []


def test_risky_promise_is_detected_in_natural_phrasing():
    r = RuleSignals()
    signals = r.observe("agent", "Your approval is guaranteed, I can promise you that.", 30.0)
    assert any(s.kind == "risky_statement" for s in signals)


def test_cross_sell_cue_detected():
    r = RuleSignals()
    signals = r.observe("customer", "We just bought a second truck last month.", 20.0)
    assert any(s.kind == "missed_cross_sell" for s in signals)


def test_indonesian_soft_refusal_is_not_agreement():
    r = RuleSignals()
    signals = r.observe("customer", "Iya nanti saya kabari deh kalau sudah ada.", 40.0)
    assert any(s.kind == "soft_refusal" for s in signals)


def test_quiet_customer_turn_produces_nothing():
    r = RuleSignals()
    assert r.observe("customer", "Yes, that's right, we've been doing it a while.", 15.0) == []


# ----------------------------------------------------------- qualification
@pytest.fixture(scope="module")
def qualifier():
    return QualificationEngine()


def test_clean_file_passes(qualifier):
    d = qualifier.evaluate({
        "entity_type": "proprietorship", "business_vintage_months": 72,
        "annual_turnover_inr": 7_200_000, "loan_amount_inr": 1_500_000,
    })
    assert d.disposition == "pass"
    assert "QR001" in d.rules_fired


def test_short_vintage_is_rejected(qualifier):
    d = qualifier.evaluate({
        "entity_type": "proprietorship", "business_vintage_months": 12,
        "annual_turnover_inr": 7_200_000, "loan_amount_inr": 1_500_000,
    })
    assert d.disposition == "reject"
    assert "VINTAGE_SHORT" in d.reason_codes


def test_borderline_turnover_is_referred_not_rejected(qualifier):
    d = qualifier.evaluate({
        "entity_type": "proprietorship", "business_vintage_months": 60,
        "annual_turnover_inr": 3_000_000, "loan_amount_inr": 800_000,
    })
    assert d.disposition == "refer"


def test_excluded_entity_type_is_rejected(qualifier):
    d = qualifier.evaluate({
        "entity_type": "trust", "business_vintage_months": 120,
        "annual_turnover_inr": 20_000_000, "loan_amount_inr": 2_000_000,
    })
    assert d.disposition == "reject"


def test_unasked_slot_means_incomplete_not_a_decision(qualifier):
    d = qualifier.evaluate({"entity_type": "proprietorship"})
    assert d.disposition == "incomplete"
    assert "annual_turnover_inr" in d.missing_slots


def test_asked_but_unknown_required_value_cannot_pass(qualifier):
    # Present-as-None means "asked, customer would not say". That is decidable,
    # but never as a pass - a human reviews it instead.
    d = qualifier.evaluate({
        "entity_type": "proprietorship", "business_vintage_months": 72,
        "annual_turnover_inr": None, "loan_amount_inr": 1_500_000,
    })
    assert d.disposition == "refer"
    assert any(c.startswith("UNVERIFIED") for c in d.reason_codes)


def test_conflict_detection():
    conflicts = detect_conflicts({"annual_turnover_inr": 3_900_000,
                                  "loan_amount_inr": 4_500_000})
    assert "requested_amount_exceeds_annual_turnover" in conflicts
