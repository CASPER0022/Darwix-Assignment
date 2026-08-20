"""Endpointing and silence-rejection tests.

These cover the failure that made every web-call turn come back as "Thank you.":
Whisper answers silence with a confident subtitle phrase rather than an empty
string, so anything that lets silence reach it produces a fabricated customer
turn. Two independent guards are tested here - the VAD must not open on
non-speech, and text that did come back must be recognised as an artefact - so
neither one silently becomes the only thing holding the line.

No API key and no network required.
"""
from __future__ import annotations

import array
from pathlib import Path
import random

import pytest

from darwix.common.asr import Transcript, looks_like_silence
from darwix.common.audio import (SPEECH_RMS_FLOOR, EnergyVAD, duration_seconds,
                                 has_speech, peak, rms)

SR = 16_000


def tone(amplitude: float, seconds: float = 0.256, seed: int = 7) -> bytes:
    """Gaussian noise at a given RMS amplitude - a stand-in for room tone."""
    rng = random.Random(seed)
    n = int(SR * seconds)
    return array.array(
        "h", [int(max(-32768, min(32767, rng.gauss(0, amplitude * 32768)))) for _ in range(n)]
    ).tobytes()


def feed(vad: EnergyVAD, pcm: bytes, chunk_s: float = 0.256) -> list[str]:
    step = int(SR * chunk_s) * 2
    return [e for e in (vad.push(pcm[i:i + step]) for i in range(0, len(pcm), step)) if e]


# ----------------------------------------------------------------- the VAD
def test_noise_floor_never_decays_into_the_speech_range() -> None:
    """The bug: an unclamped floor converges on the room tone it measures.

    Measured before the clamp, the start threshold fell from 0.0229 to 0.0012
    after 51 s of quiet - a 19x collapse - after which ambient noise opened a
    turn and 3 s of near-silence went to Whisper.
    """
    vad = EnergyVAD()
    quiet = tone(0.0004, seconds=60.0)
    feed(vad, quiet)
    assert vad.threshold >= SPEECH_RMS_FLOOR, (
        f"threshold decayed to {vad.threshold:.5f}, below the speech floor"
    )


def test_ambient_noise_does_not_open_a_turn_after_a_long_silence() -> None:
    vad = EnergyVAD()
    feed(vad, tone(0.0004, seconds=60.0))            # a quiet room
    events = feed(vad, tone(0.004, seconds=2.0))     # a fan swell / distant tap
    assert "speech_start" not in events


def test_real_speech_levels_still_open_a_turn() -> None:
    """The guard must not make the agent deaf.

    0.023 RMS is the quietest utterance measured across the committed call
    recordings, so anything at or above it has to get through.
    """
    for level in (0.023, 0.05, 0.15):
        vad = EnergyVAD()
        feed(vad, tone(0.0004, seconds=60.0))        # decay the floor first
        events = feed(vad, tone(level, seconds=1.5))
        assert "speech_start" in events, f"missed real speech at rms {level}"


def test_speech_end_fires_after_the_hangover() -> None:
    vad = EnergyVAD()
    feed(vad, tone(0.05, seconds=1.5))
    events = feed(vad, tone(0.0002, seconds=2.0))
    assert "speech_end" in events


def test_floor_tracks_a_noisy_room_without_going_deaf() -> None:
    """The clamp must not stop the floor adapting upward in a loud room."""
    vad = EnergyVAD()
    feed(vad, tone(0.02, seconds=60.0))
    assert vad.noise_floor > 0.004, "floor failed to adapt upward"
    assert vad.noise_floor <= vad.floor_max


# --------------------------------------------------------- the energy gate
def test_has_speech_rejects_silence_and_near_silence() -> None:
    assert not has_speech(b"\x00\x00" * int(SR * 2.8))
    assert not has_speech(tone(0.004, seconds=2.8))
    assert not has_speech(b"")


def test_has_speech_accepts_real_speech_levels() -> None:
    assert has_speech(tone(0.023, seconds=1.0))
    assert has_speech(tone(0.08, seconds=1.0))


def test_has_speech_ignores_a_lone_transient() -> None:
    """200 ms of a door closing inside 3 s of quiet is not an utterance."""
    seg = tone(0.0004, seconds=1.4) + tone(0.09, seconds=0.1) + tone(0.0004, seconds=1.4)
    assert not has_speech(seg)


def test_has_speech_accepts_speech_with_pauses() -> None:
    seg = tone(0.06, seconds=0.9) + tone(0.0004, seconds=0.6) + tone(0.06, seconds=0.9)
    assert has_speech(seg)


# ------------------------------------------------- the transcript backstop
@pytest.mark.parametrize("text", [
    "Thank you.", "Thank you", "thanks for watching!", "Thank you for watching",
    "Please subscribe", "[BLANK_AUDIO]", "(soft music)", "*sighs*", "...", ".",
    "", "   ", "you", "Terima kasih", "Subtitles by the amara.org community",
])
def test_known_whisper_artefacts_are_rejected(text: str) -> None:
    assert looks_like_silence(text)
    assert Transcript(text=text).is_empty


@pytest.mark.parametrize("text", [
    "Yes", "No", "Okay", "Yeah", "Correct", "Sure", "Go ahead", "Not right now",
    "Yes, thank you", "Thank you for explaining the rate",
    "I run a textile business in Surat", "My turnover is about 80 lakhs",
    "Ya", "Opo", "Hindi po", "Can I speak to a human?",
])
def test_real_customer_answers_are_not_rejected(text: str) -> None:
    """A bare "yes" is a whole answer to a consent question.

    The energy gate is the primary defence, so this list stays narrow on
    purpose - swallowing "yes" would break qualification far more often than a
    stray artefact ever did.
    """
    assert not looks_like_silence(text)
    assert not Transcript(text=text).is_empty


# -------------------------------------------------------------- end to end
def test_silence_never_reaches_the_transcriber() -> None:
    """The whole path, as it runs on a quiet line: nothing should be emitted."""
    vad = EnergyVAD()
    captured: list[bytes] = []
    buf = bytearray()
    capturing = False

    for _ in range(400):                      # ~100 s of an open, quiet line
        chunk = tone(0.0004 + random.random() * 0.003)
        event = vad.push(chunk)
        if event == "speech_start":
            capturing, buf = True, bytearray()
        if capturing:
            buf.extend(chunk)
        if event == "speech_end" and capturing:
            capturing = False
            seg = bytes(buf)
            if duration_seconds(seg) >= 0.35 and has_speech(seg):
                captured.append(seg)

    assert captured == [], f"{len(captured)} silent segments would have hit Whisper"


def test_helpers_agree_on_scale() -> None:
    loud = tone(0.1, seconds=0.5)
    assert 0.09 < rms(loud) < 0.11
    assert peak(loud) > rms(loud)
    assert peak(b"\x00\x00" * 100) == 0.0


# ------------------------------------------------- multilingual artefacts
@pytest.mark.parametrize("text", [
    "\u3054\u8996\u8074\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f",   # ja - seen live on an en-IN call
    "\u3042\u308a\u304c\u3068\u3046\u3054\u3056\u3044\u307e\u3057\u305f",
    "\u8c22\u8c22\u5927\u5bb6",                                                                # zh
    "\uc2dc\uccad\ud574\uc8fc\uc154\uc11c \uac10\uc0ac\ud569\ub2c8\ub2e4",                      # ko
    "\u0421\u043f\u0430\u0441\u0438\u0431\u043e \u0437\u0430 \u043f\u0440\u043e\u0441\u043c\u043e\u0442\u0440",  # ru
    "Terima kasih", "Salamat po",
])
def test_artefacts_in_other_languages_are_rejected(text: str) -> None:
    """Whisper picks a language for silence too, so the artefact is not English.

    Language is deliberately left unset for Taglish, which is exactly what
    lets this happen.
    """
    assert looks_like_silence(text, script="latin")
    assert Transcript(text=text).is_empty


def test_non_latin_script_guard_is_length_bounded() -> None:
    """A long non-Latin passage is not assumed to be an artefact."""
    long_ja = "\u79c1\u306f\u5927\u962a\u3067\u5c0f\u3055\u306a\u88fd\u9020\u4f1a\u793e\u3092\u7d4c\u55b6\u3057\u3066\u3044\u307e\u3059\u3002" * 3
    assert not looks_like_silence(long_ja, script="latin")


def test_script_guard_is_opt_in() -> None:
    """Without a declared script nothing is rejected on script alone."""
    assert not looks_like_silence("\u79c1\u306f\u5143\u6c17\u3067\u3059")


# ------------------------------------------------------------ echo guard
def test_echo_guard_threshold_sits_above_normal_speech() -> None:
    """Barge-in must be harder than merely speaking.

    While the agent talks the microphone also hears the agent, and what echo
    cancellation leaves through looks exactly like a turn - an 11.4 s "utterance"
    was captured mid-sentence before this guard existed.
    """
    from darwix.common.config import settings
    assert settings.barge_in_rms > SPEECH_RMS_FLOOR
    assert settings.barge_in_rms > 0.023          # the quietest measured speech
    assert settings.echo_guard_tail_ms > 0


@pytest.mark.asyncio
async def test_agent_does_not_capture_its_own_playback() -> None:
    """Speaker bleed at ordinary speech level must not open a turn."""
    from darwix.common.config import settings
    from darwix.voice.turn_manager import AudioCall

    class _State:
        call_id = "test"

    class _Session:
        state = _State()

    call = AudioCall.__new__(AudioCall)          # no real session/ASR needed
    call.session = _Session()
    call.vad = EnergyVAD()
    call.buffer = bytearray()
    call.preroll = bytearray()
    call.customer_track = bytearray()
    call.agent_track = bytearray()
    call.capturing = False
    call.agent_speaking = True
    call.processing = False
    call._barged = False
    call._echo_guard_until = 0.0

    events: list[dict] = []

    async def on_event(p):
        events.append(p)

    call.on_event = on_event

    bleed = tone(0.03, seconds=0.256)            # leaked agent audio
    for _ in range(20):
        await call.push_audio(bleed)

    assert not call.capturing, "agent captured its own voice"
    assert not any(e.get("type") == "barge_in" for e in events)

    # A real interruption is louder and must still get through.
    call.vad = EnergyVAD()
    loud = tone(0.12, seconds=0.256)
    for _ in range(6):
        await call.push_audio(loud)
    assert any(e.get("type") == "barge_in" for e in events), "real barge-in was blocked"


# ------------------------------------------------ minimum-statistics floor
@pytest.mark.parametrize("ambient", [0.0004, 0.008, 0.013, 0.020])
def test_ambient_noise_never_opens_a_turn_at_any_room_level(ambient: float) -> None:
    """The failure this replaced: once room tone rose above the threshold every
    chunk counted as speech, so the branch that adapted the floor stopped
    running and the VAD stayed permanently triggered. Seen live at 0.013 RMS,
    which Whisper rendered as "in the end, Pernando, a room because you're not
    happy with your home."
    """
    vad = EnergyVAD()
    feed(vad, tone(ambient, seconds=30.0))
    events = feed(vad, tone(ambient, seconds=4.0))
    assert "speech_start" not in events, (
        f"ambient {ambient} opened a turn; floor={vad.noise_floor:.5f} "
        f"threshold={vad.threshold:.5f}"
    )


@pytest.mark.parametrize("ambient", [0.0004, 0.008, 0.013, 0.020])
def test_speech_is_still_heard_over_each_ambient_level(ambient: float) -> None:
    vad = EnergyVAD()
    feed(vad, tone(ambient, seconds=30.0))
    assert "speech_start" in feed(vad, tone(0.08, seconds=1.5))


def test_floor_rises_to_meet_a_noisy_room() -> None:
    """A percentile keeps adapting whatever the speech decision was."""
    vad = EnergyVAD()
    feed(vad, tone(0.013, seconds=30.0))
    assert vad.noise_floor > 0.010, "floor did not track the noise"
    assert vad.threshold > 0.013, "threshold did not clear the room tone"


def test_floor_recovers_when_a_noisy_room_goes_quiet() -> None:
    vad = EnergyVAD()
    feed(vad, tone(0.02, seconds=30.0))
    noisy = vad.threshold
    feed(vad, tone(0.0004, seconds=30.0))
    assert vad.threshold < noisy, "floor stayed high after the noise stopped"
    assert vad.threshold >= SPEECH_RMS_FLOOR


# ------------------------------------------------------ language allowlist
@pytest.mark.parametrize("locale,allowed,rejected", [
    ("en_IN",  ["English", "Hindi", "Tamil"],      ["Japanese", "Spanish", "Russian"]),
    ("fil_PH", ["Tagalog", "English", "Cebuano"],  ["Japanese", "German"]),
    ("id_ID",  ["Indonesian", "Javanese", "Sundanese", "English"], ["Spanish", "Korean"]),
])
def test_locale_language_allowlists(locale: str, allowed: list, rejected: list) -> None:
    """Auto-detect is required for code-switching, and is also what lets Whisper
    label noise as an unrelated language. Each market therefore declares which
    detections are plausible; Javanese and Sundanese stay in for the Q3 accent
    tests, and Taglish keeps both of its halves.
    """
    import yaml

    pack = yaml.safe_load(
        (Path("src/darwix/voice/locales") / locale / "pack.yaml").read_text(encoding="utf-8")
    )
    langs = {x.strip().lower() for x in pack["voice"]["asr_languages"]}
    for lang in allowed:
        assert lang.lower() in langs, f"{lang} should be accepted on {locale}"
    for lang in rejected:
        assert lang.lower() not in langs, f"{lang} should be rejected on {locale}"


def test_every_locale_declares_an_allowlist_including_english() -> None:
    """English is on every list: all three markets code-switch into it."""
    import yaml

    for locale in ("en_IN", "fil_PH", "id_ID"):
        pack = yaml.safe_load(
            (Path("src/darwix/voice/locales") / locale / "pack.yaml").read_text(encoding="utf-8")
        )
        langs = pack["voice"].get("asr_languages")
        assert langs, f"{locale} declares no asr_languages"
        assert "english" in {x.strip().lower() for x in langs}


# ------------------------------------------------ barge-in during synthesis
def _bare_call(monkeypatch, tts_pcm: bytes, *, on_synth=None):
    """An AudioCall with TTS and the session stubbed out."""
    from darwix.common import tts as tts_mod
    from darwix.voice.turn_manager import AudioCall

    class _State:
        call_id = "test"
        phase = type("P", (), {"value": "greeting"})()

    class _Latency:
        def record(self, *a, **k): pass

    class _Session:
        state = _State()
        latency = _Latency()
        pack = {"voice": {"tts_voice": "agent_en_in_f"}}

    class _TTS:
        async def synthesize(self, text, voice):
            if on_synth:
                await on_synth()
            return tts_pcm

    monkeypatch.setattr(tts_mod, "get_tts", lambda: _TTS())
    import darwix.voice.turn_manager as tm
    monkeypatch.setattr(tm, "get_tts", lambda: _TTS())

    call = AudioCall.__new__(AudioCall)
    call.session = _Session()
    call.vad = EnergyVAD()
    call.buffer = bytearray(); call.preroll = bytearray()
    call.customer_track = bytearray(); call.agent_track = bytearray()
    call.capturing = False; call.agent_speaking = False; call.processing = False
    call._barged = False; call._echo_guard_until = 0.0; call._pending = None
    call.events = []; call.audio_out = bytearray()

    async def on_event(p): call.events.append(p)
    async def on_audio(b): call.audio_out.extend(b)
    call.on_event = on_event
    call.on_audio = on_audio
    return call


@pytest.mark.asyncio
async def test_interruption_during_synthesis_cancels_the_reply(monkeypatch) -> None:
    """Synthesis is the longest part of a turn - seconds, and ~11 s where the
    provider's TLS is being reset. The agent used to be deaf for all of it and
    would then talk over a caller who had already started.
    """
    speech = tone(0.12, seconds=2.0)

    async def interrupt():
        # The caller starts talking while the words are still being made.
        for _ in range(6):
            await call.push_audio(speech[:4096 * 2])

    call = _bare_call(monkeypatch, tone(0.08, seconds=3.0), on_synth=lambda: interrupt())
    await call.speak("A reply nobody will hear.")

    assert call._barged, "the interruption was not registered during synthesis"
    assert call.audio_out == bytearray(), "the agent talked over the caller"
    assert any(e.get("type") == "barge_in" for e in call.events)


@pytest.mark.asyncio
async def test_a_normal_reply_is_played_in_full(monkeypatch) -> None:
    pcm = tone(0.08, seconds=1.0)
    call = _bare_call(monkeypatch, pcm)
    await call.speak("A reply the caller does hear.")
    assert len(call.audio_out) == len(pcm)
    assert not call.agent_speaking
    assert call.events[-1]["type"] == "idle"


@pytest.mark.asyncio
async def test_barge_state_does_not_leak_between_utterances(monkeypatch) -> None:
    """The greeting is spoken outside _flush(), whose `finally` used to be the
    only place _barged was cleared - so an interruption there silenced every
    later utterance for the rest of the call."""
    pcm = tone(0.08, seconds=1.0)
    call = _bare_call(monkeypatch, pcm)
    call._barged = True                      # left over from a previous turn
    await call.speak("This must still be spoken.")
    assert len(call.audio_out) == len(pcm), "a stale barge flag silenced the agent"


@pytest.mark.asyncio
async def test_recording_channels_advance_at_the_same_rate(monkeypatch) -> None:
    """Both channels are driven by the microphone clock. Previously push_audio
    appended live mic *and* speak() appended an equal run of zeros, so the
    customer channel ran at double rate whenever the agent talked."""
    mic = tone(0.001, seconds=0.256)
    call = _bare_call(monkeypatch, tone(0.08, seconds=2.0))
    for _ in range(20):                      # 5.1 s of call before the agent talks
        await call.push_audio(mic)
    before = len(call.customer_track)
    await call.speak("Two seconds of agent audio.")
    assert len(call.customer_track) == before, (
        "speak() advanced the customer channel; only the microphone may do that"
    )
    for _ in range(8):
        await call.push_audio(mic)
    skew = abs(len(call.customer_track) - len(call.agent_track)) / (2 * 16000)
    assert skew < 6.0, f"channels drifted {skew:.1f}s apart"
