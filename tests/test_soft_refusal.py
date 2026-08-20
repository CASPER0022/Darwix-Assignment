"""Indirect refusal handling (Q3, Indonesia).

Indonesian politeness avoids a flat no. "Nanti saya kabari deh" is a refusal
dressed as a promise, and a bot that hears agreement books a promise-to-pay
that never arrives. The locale pack documents this as the hard part of the
market, so the behaviour is tested rather than asserted in a comment.
"""
import pytest

from darwix.voice.dialog_policy import Phase
from darwix.voice.session import CallSession


@pytest.fixture()
def session():
    return CallSession(locale="id-ID", call_id="test_soft_refusal")


def test_pack_declares_both_marker_lists(session):
    assert session.soft_refusal_markers, "id-ID pack must declare soft_refusal_markers"
    assert session.commitment_markers, "id-ID pack must declare commitment_markers"


@pytest.mark.parametrize("utterance", [
    "nanti saya kabari deh kalau sudah ada",
    "iya nanti ya kak",
    "diusahakan bulan ini",
    "kalau ada rezeki saya bayar",
])
def test_indirect_refusal_is_recorded(session, utterance):
    session._note_commitment(utterance)
    assert session.state.soft_refusals, utterance
    assert not session.state.commitment


@pytest.mark.parametrize("utterance", [
    "tanggal 25 saya transfer",
    "besok saya bayar",
    "minggu depan ya bu",
    "setelah gajian saya lunasi",
])
def test_a_named_date_resolves_a_pending_deferral(session, utterance):
    session._note_commitment("nanti saya kabari deh")
    session.state.awaiting_commitment = True          # the probe was just asked
    session._note_commitment(utterance)
    assert session.state.commitment


def test_a_due_date_is_not_a_promise_to_pay(session):
    """The flow asks "jatuh temponya tanggal berapa?" in its own right. The
    customer naming the date the money is owed - before or after a deferral -
    must never be recorded as a date they promised to send it."""
    session._note_commitment("jatuh temponya tanggal 5 kak")
    assert session.state.commitment == ""

    session._note_commitment("nanti saya kabari deh")
    session._note_commitment("itu temponya tanggal 5, kak")
    assert session.state.commitment == ""


def test_a_date_taken_back_in_the_same_breath_is_not_a_commitment(session):
    """Verbatim from a test call, in reply to the probe: a date, a "but", and a
    deferral. The deferral wins - it is the part that decides whether money
    arrives."""
    session._note_commitment("nanti saya kabari deh")
    session.state.awaiting_commitment = True
    session._note_commitment(
        "eh, tanggal 5 ya, tapi bulan ini cashnya masih belum ada, "
        "jadi transfernya nanti saya kabari ya")
    assert session.state.commitment == ""
    assert session.state.soft_refusals == ["nanti saya kabari"]


def test_the_probe_is_only_answered_once(session):
    """A date in the turn after the probe counts. A date two turns later does
    not - by then the agent has asked something else."""
    session._note_commitment("nanti saya kabari deh")
    session.state.awaiting_commitment = True
    session._note_commitment("belum tahu kak")        # consumes the probe
    assert session.state.commitment == ""
    session._note_commitment("tanggal 20 lah")
    assert session.state.commitment == ""


def test_a_date_in_answer_to_the_probe_cancels_the_deferral(session):
    session._note_commitment("nanti saya kabari deh")
    assert session.state.soft_refusals and not session.state.commitment
    session.state.awaiting_commitment = True          # the probe was just asked
    session._note_commitment("oke, tanggal 20 saya transfer")
    assert session.state.commitment


def test_the_same_marker_is_only_recorded_once(session):
    session._note_commitment("nanti saya kabari kalau sudah ada")
    session._note_commitment("nanti saya kabari lagi ya")
    assert session.state.soft_refusals == ["nanti saya kabari"]


def test_a_deferral_without_a_date_cannot_close_as_pass(session):
    """The whole point: the call must not record a commitment it never got."""
    session.state.phase = Phase.QUALIFICATION
    session.state.slots = {spec["key"]: "x" for spec in session.flow}
    session._note_commitment("nanti saya kabari deh")

    session._qualification_step()

    assert session.state.decision["disposition"] == "no_commitment"
    assert session.state.decision["reason_codes"] == ["NO_COMMITMENT"]
    assert session.state.decision["soft_refusals"] == ["nanti saya kabari"]


def test_a_deferral_that_resolves_into_a_date_closes_normally(session):
    session.state.phase = Phase.QUALIFICATION
    session.state.slots = {spec["key"]: "x" for spec in session.flow}
    session._note_commitment("nanti saya kabari deh")
    session.state.awaiting_commitment = True
    session._note_commitment("tanggal 20 saya transfer")

    session._qualification_step()

    assert session.state.decision["disposition"] == "pass"
    assert session.state.decision["commitment"] == "tanggal"


def test_markets_without_markers_are_unaffected():
    """A pack that declares nothing keeps the previous behaviour exactly."""
    en = CallSession(locale="en-IN", call_id="test_no_markers")
    assert en.soft_refusal_markers == []
    en._note_commitment("nanti saya kabari deh")
    assert en.state.soft_refusals == []
