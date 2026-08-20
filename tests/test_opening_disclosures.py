"""The opening is now spoken in two parts, so the caller can interrupt it.

That is a latency change with a compliance edge: `required_before:
qualification` must still hold when the second part never gets said. These
tests pin the rule rather than the wording, so the copy can be edited without
them going stale.

No API key, no network.
"""
from __future__ import annotations

import pytest

from darwix.voice.dialog_policy import Phase, pending_disclosures
from darwix.voice.session import CallSession

LOCALES = ["en-IN", "fil-PH", "id-ID"]


def session(locale: str) -> CallSession:
    return CallSession(locale, retriever=None)


def required(s: CallSession) -> set[str]:
    return {d["id"] for d in s.pack.get("disclosures", [])
            if d.get("required_before") == "qualification"}


@pytest.mark.parametrize("locale", LOCALES)
def test_opening_is_split_into_interruptible_parts(locale: str) -> None:
    parts = session(locale).opening_parts()
    assert len(parts) >= 2, "the opening is still one uninterruptible block"
    assert all(text.strip() for text, _ in parts)


@pytest.mark.parametrize("locale", LOCALES)
def test_every_required_disclosure_is_delivered_across_the_parts(locale: str) -> None:
    s = session(locale)
    delivered = {i for _, ids in s.opening_parts() for i in ids}
    assert required(s) == delivered, "splitting dropped a disclosure"


@pytest.mark.parametrize("locale", LOCALES)
def test_the_ai_disclosure_lands_in_the_first_part(locale: str) -> None:
    """Of the three, this is the one a caller is entitled to hear before
    deciding whether to keep talking to a machine."""
    parts = session(locale).opening_parts()
    assert "ai_disclosure" in parts[0][1]


@pytest.mark.parametrize("locale", LOCALES)
def test_nothing_is_marked_given_until_the_audio_is_sent(locale: str) -> None:
    """The transcript is a compliance record. It used to be written when the
    text was assembled, so a TTS failure produced a call that claimed to have
    disclosed things the caller never heard."""
    s = session(locale)
    s.opening_parts()
    assert s.state.disclosures_given == set()


@pytest.mark.parametrize("locale", LOCALES)
def test_marking_a_part_records_only_that_part(locale: str) -> None:
    s = session(locale)
    parts = s.opening_parts()
    s.mark_disclosed(parts[0][1], parts[0][0])
    assert s.state.disclosures_given == set(parts[0][1])
    outstanding = {d["id"] for d in
                   pending_disclosures(s.pack, s.state, before="qualification")}
    assert outstanding == required(s) - set(parts[0][1])


@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.asyncio
async def test_an_interrupted_opening_still_discloses_before_qualification(
        locale: str) -> None:
    """The whole point of the safety net.

    Simulate the caller cutting in after part one: only the first part was
    spoken. By the time the agent asks its first qualification question, the
    outstanding disclosures must have been said.
    """
    s = session(locale)
    parts = s.opening_parts()
    s.mark_disclosed(parts[0][1], parts[0][0])          # part two never spoken
    missing_before = required(s) - s.state.disclosures_given
    assert missing_before, "test set-up is wrong: nothing was outstanding"

    s.state.phase = Phase.CONSENT
    result = await s.handle("Yes, go ahead.")

    assert s.state.disclosures_given >= required(s), (
        f"{locale}: entered qualification still owing {required(s) - s.state.disclosures_given}"
    )
    for d in s.pack["disclosures"]:
        if d["id"] in missing_before:
            assert d["text"] in result.text, (
                f"{locale}: {d['id']} was marked given but never spoken"
            )


@pytest.mark.parametrize("locale", LOCALES)
def test_the_unsplit_opening_still_works_and_agrees(locale: str) -> None:
    """`opening()` remains for callers that do not stream; it must deliver the
    same disclosures as the split form."""
    whole = session(locale)
    text = whole.opening()
    assert whole.state.disclosures_given == required(whole)
    for d in whole.pack["disclosures"]:
        if d["id"] in required(whole):
            assert d["text"] in text
