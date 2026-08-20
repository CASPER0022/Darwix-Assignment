"""Endpointing against the committed call recordings.

The rest of the VAD tests drive synthetic Gaussian noise at a constant level.
That is useful for the threshold arithmetic and useless for everything a real
conversation does - onsets, breaths, sentence pauses, a caller who is already
talking when the recording opens. A regression that only shows up on real audio
would pass every one of them, which is exactly what happened: the warm-up spike
fixed here closed an utterance 2.8 s in, mid-sentence, with the synthetic suite
fully green.

These run offline against files already in the repo. No API key, no network.
"""
from __future__ import annotations

import array
import wave
from pathlib import Path

import pytest

from darwix.common.audio import (SPEECH_RMS_FLOOR, EnergyVAD, duration_seconds,
                                 has_speech, rms)

REC = Path("data/recordings")
CHUNK = 4096 * 2
MIN_SEGMENT_S = 0.35

pytestmark = pytest.mark.skipif(
    not list(REC.glob("q4_*.wav")), reason="scenario recordings not present"
)


def scenarios() -> list[Path]:
    return sorted(REC.glob("q4_*.wav"))


def channel(path: Path, idx: int) -> bytes:
    with wave.open(str(path)) as wf:
        ch, n = wf.getnchannels(), wf.getnframes()
        pcm = wf.readframes(n)
    if ch == 1:
        return pcm
    a = array.array("h", pcm)
    return array.array("h", a[idx::ch]).tobytes()


def segment(pcm: bytes, *, hangover_ms: float = 800.0) -> list[bytes]:
    return [s for s, _ in segment_with_offsets(pcm, hangover_ms=hangover_ms)]


def segment_with_offsets(pcm: bytes, *, hangover_ms: float = 800.0
                         ) -> list[tuple[bytes, int]]:
    """Segments plus the byte offset in `pcm` where each one ended."""
    vad = EnergyVAD(hangover_ms=hangover_ms)
    out: list[tuple[bytes, int]] = []
    buf = bytearray()
    capturing = False
    for i in range(0, len(pcm), CHUNK):
        c = pcm[i:i + CHUNK]
        event = vad.push(c)
        if event == "speech_start":
            capturing, buf = True, bytearray()
        if capturing:
            buf.extend(c)
        if event == "speech_end" and capturing:
            capturing = False
            if duration_seconds(bytes(buf)) >= MIN_SEGMENT_S:
                out.append((bytes(buf), i + len(c)))
            buf = bytearray()
    if capturing and duration_seconds(bytes(buf)) >= MIN_SEGMENT_S:
        out.append((bytes(buf), len(pcm)))
    return out


@pytest.mark.parametrize("wav", scenarios(), ids=lambda p: p.stem)
def test_every_channel_yields_utterances(wav: Path) -> None:
    """Both sides of a two-party call must produce speech."""
    for idx, who in ((0, "agent"), (1, "customer")):
        segs = segment(channel(wav, idx))
        assert segs, f"{wav.stem}/{who}: the VAD heard nothing at all"


@pytest.mark.parametrize("wav", scenarios(), ids=lambda p: p.stem)
def test_no_utterance_is_closed_while_the_speaker_is_still_talking(wav: Path) -> None:
    """`speech_end` is a claim that the speaker stopped. Check it against the
    tape: the audio just past the boundary must actually be quiet.

    This is the assertion that catches the warm-up spike. There, the floor
    estimate climbed into the speech it was measuring, the threshold went above
    the talker, and the utterance was closed 2.8 s in with the speaker still
    mid-sentence - while a duration-based check ("longer than a second") sailed
    straight past it.
    """
    # `speech_end` fires only after a hangover of continuous silence, so that
    # hangover must really be silent on the tape. Looking *after* the boundary
    # proves nothing - the next utterance may legitimately start immediately.
    hangover_bytes = int(0.8 * 16000) * 2
    for idx, who in ((0, "agent"), (1, "customer")):
        pcm = channel(wav, idx)
        for seg, end in segment_with_offsets(pcm):
            if end >= len(pcm):
                continue                      # closed by end of file, not by VAD
            window = pcm[max(0, end - hangover_bytes):end]
            if len(window) < hangover_bytes // 2:
                continue
            assert rms(window) < SPEECH_RMS_FLOOR, (
                f"{wav.stem}/{who}: utterance closed at {end / 32000:.2f}s, but the "
                f"{len(window) / 32000:.2f}s of hangover before it is speech "
                f"(rms {rms(window):.4f}) - the VAD stopped hearing a talker who "
                f"was still talking"
            )


@pytest.mark.parametrize("wav", scenarios(), ids=lambda p: p.stem)
def test_captured_segments_all_contain_speech(wav: Path) -> None:
    """Whatever the VAD opens, the energy gate must agree is speech - otherwise
    the two guards disagree and one of them is spending ASR calls on noise."""
    for idx, who in ((0, "agent"), (1, "customer")):
        for s in segment(channel(wav, idx)):
            assert has_speech(s), (
                f"{wav.stem}/{who}: {duration_seconds(s):.2f}s segment "
                f"(rms {rms(s):.4f}) opened by the VAD but rejected by has_speech"
            )


@pytest.mark.parametrize("wav", scenarios(), ids=lambda p: p.stem)
def test_segmentation_is_stable_run_to_run(wav: Path) -> None:
    """The VAD is deterministic; only the network above it is not. This is what
    distinguishes a real regression from ordinary run variance."""
    a = [round(duration_seconds(s), 3) for s in segment(channel(wav, 1))]
    b = [round(duration_seconds(s), 3) for s in segment(channel(wav, 1))]
    assert a == b


def test_total_segment_count_across_the_scenarios() -> None:
    """A canary on the whole corpus.

    Pinned to the measured value with room either side. It is not a claim that
    46 is correct - it is a tripwire: a change that moves this materially has
    changed turn taking on real audio and needs looking at, not waving through.
    """
    total = sum(len(segment(channel(w, i)))
                for w in scenarios() for i in (0, 1))
    assert 40 <= total <= 52, f"segment count moved to {total} (was 46)"


def test_agent_and_customer_channels_are_separable() -> None:
    """Q4's speaker attribution depends on the channels being genuinely
    distinct; if a recording were mixed to both, every nudge would misattribute.
    """
    for wav in scenarios():
        left, right = channel(wav, 0), channel(wav, 1)
        assert left != right, f"{wav.stem}: both channels are identical"
        assert rms(left) > 0 and rms(right) > 0, f"{wav.stem}: a channel is silent"
