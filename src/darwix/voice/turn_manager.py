"""Audio turn management: the layer between a byte stream and a conversation.

Responsibilities, in order:

1. **Endpointing.** Decide when the customer has stopped talking. This is the
   single biggest driver of perceived responsiveness on a voice call - too
   eager and you interrupt them, too patient and the bot feels dead. Energy VAD
   with a minimum-statistics noise floor and a 1100 ms hangover (see
   common/audio.py).

2. **Barge-in.** If the customer starts speaking while the agent is talking,
   the agent stops. A bot that talks over you is the most common complaint
   about voice agents, and it is a transport-level fix, not a prompt.

3. **Segment ASR.** Each captured utterance goes to Whisper as one segment,
   with a locale-specific vocabulary prompt.

4. **Recording.** Both directions are captured to separate buffers and mixed
   to stereo at the end - agent left, customer right - which gives the Q4
   analysis perfect speaker separation for free.

Every stage is timed into a `LatencyCollector`, so the per-call latency
breakdown in the transcript is measured rather than estimated.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from ..common.asr import get_asr
from ..common.audio import (SAMPLE_RATE, EnergyVAD, duration_seconds, has_speech,
                            mix_stereo, rms, to_mp3, write_wav)
from ..common.config import settings
from ..common.logging import log
from ..common.tts import get_tts
from .session import CallSession

MAX_UTTERANCE_SECONDS = 30.0
MIN_UTTERANCE_SECONDS = 0.35

# The VAD needs ~250 ms of energy before it will declare speech, so by the time
# capture opens the first word is already gone. Q4's stream.py has carried this
# ring buffer since "Who is this?" kept arriving as "is this?"; the Q1 path was
# left without it and clipped every utterance the same way.
PREROLL_MS = 300


@dataclass
class AudioCall:
    """One audio-backed call: VAD + ASR + session + TTS + recording."""

    session: CallSession
    on_event: Callable[[dict], Awaitable[None]]
    on_audio: Callable[[bytes], Awaitable[None]]

    vad: EnergyVAD = field(default_factory=EnergyVAD)
    buffer: bytearray = field(default_factory=bytearray)
    preroll: bytearray = field(default_factory=bytearray)
    agent_track: bytearray = field(default_factory=bytearray)
    customer_track: bytearray = field(default_factory=bytearray)
    capturing: bool = False
    agent_speaking: bool = False
    processing: bool = False
    _barged: bool = False
    _echo_guard_until: float = 0.0
    _pending: bytes | None = None

    @property
    def locale_voice(self) -> str:
        return self.session.pack["voice"]["tts_voice"]

    @property
    def asr_language(self) -> str | None:
        return self.session.pack["voice"].get("asr_language")

    @property
    def asr_prompt(self) -> str:
        return self.session.pack["voice"].get("asr_prompt", "")

    @property
    def asr_languages(self) -> set[str]:
        """Languages a caller on this market plausibly speaks, lowercased."""
        return {str(x).strip().lower()
                for x in (self.session.pack["voice"].get("asr_languages") or [])}

    # ------------------------------------------------------------------ audio
    async def push_audio(self, pcm: bytes) -> None:
        """Feed one chunk of customer audio (16 kHz mono PCM16)."""
        # The customer track records continuously, including silence, so the
        # recording stays time-aligned with the agent track.
        self.customer_track.extend(pcm)

        # While the agent is speaking, the microphone hears the agent too. The
        # browser's echo canceller reduces that but does not remove it, and
        # what leaks through is continuous energy that the VAD reads as a turn:
        # the agent transcribes its own voice, answers it, and talks to itself
        # until the call is hung up. Observed directly - an 11.4 s "utterance"
        # captured while the agent was mid-sentence.
        #
        # So during playback (and briefly after, for audio already queued in
        # the browser) only clearly louder audio counts, which is what a real
        # interruption sounds like. Quieter bleed is not fed to the VAD at all,
        # so it also cannot drag the noise floor around.
        if self.agent_speaking or time.perf_counter() < self._echo_guard_until:
            if rms(pcm) < settings.barge_in_rms:
                return

        event = self.vad.push(pcm)

        if event == "speech_start":
            if self.agent_speaking and not self._barged:
                self._barged = True
                await self.on_event({"type": "barge_in"})
                log("call.barge_in", call_id=self.session.state.call_id)
            self.capturing = True
            self.buffer.clear()
            self.buffer.extend(self.preroll)   # recover the utterance onset
            await self.on_event({"type": "listening"})

        if self.capturing:
            self.buffer.extend(pcm)
            if duration_seconds(bytes(self.buffer)) > MAX_UTTERANCE_SECONDS:
                await self._flush()
                return
        else:
            self.preroll.extend(pcm)
            keep = int(SAMPLE_RATE * (PREROLL_MS / 1000.0)) * 2
            if len(self.preroll) > keep:
                del self.preroll[:len(self.preroll) - keep]

        if event == "speech_end" and self.capturing:
            await self._flush()

    async def _flush(self) -> None:
        pcm = bytes(self.buffer)
        self.buffer.clear()
        self.preroll.clear()
        self.capturing = False
        if duration_seconds(pcm) < MIN_UTTERANCE_SECONDS:
            return
        # Never hand Whisper a segment with no speech in it. It does not answer
        # silence with "" - it answers with "Thank you.", confidently, and the
        # agent then replies to a customer who never said anything. Dropping
        # here also saves the ASR round trip.
        if not has_speech(pcm):
            log("call.segment_dropped", call_id=self.session.state.call_id,
                reason="no_speech_energy", audio_s=round(duration_seconds(pcm), 2),
                rms=round(rms(pcm), 5))
            await self.on_event({"type": "idle"})
            return
        if self.processing:
            # Held, not discarded. A turn takes seconds - retrieval, the model,
            # then synthesis - and a caller who answers inside that window used
            # to be thrown away with nothing said and nothing shown, which on a
            # slow network is most of the call. One slot only: if they speak
            # again the newer utterance replaces the older, because answering a
            # question two turns late is worse than not answering it.
            if self._pending is not None:
                log("call.pending_replaced", call_id=self.session.state.call_id)
            self._pending = pcm
            log("call.segment_queued", call_id=self.session.state.call_id,
                audio_s=round(duration_seconds(pcm), 2))
            return
        self.processing = True
        try:
            await self._handle_utterance(pcm)
            # Drain anything that arrived while the turn was running, so the
            # caller's own pacing decides the conversation rather than ours.
            while self._pending is not None:
                queued, self._pending = self._pending, None
                await self._handle_utterance(queued)
        finally:
            self.processing = False
            self._pending = None
            self._barged = False

    async def _handle_utterance(self, pcm: bytes) -> None:
        state = self.session.state
        await self.on_event({"type": "thinking"})

        t_asr = time.perf_counter()
        try:
            tr = await get_asr().transcribe_pcm(
                pcm, language=self.asr_language, prompt=self.asr_prompt
            )
        except Exception as exc:  # noqa: BLE001
            log("call.asr_failed", error=str(exc)[:200])
            await self.speak(self.session.pack["fallback"]["asr_failure"])
            return
        asr_ms = (time.perf_counter() - t_asr) * 1000
        self.session.latency.record("asr_ms", asr_ms)

        if tr.is_empty:
            # Reached only when the segment *did* carry speech energy, so this
            # is a real failure to understand rather than noise, and saying so
            # is the right response.
            log("call.asr_empty", call_id=state.call_id,
                audio_s=round(duration_seconds(pcm), 2), raw_text=tr.text[:80])
            await self.speak(self.session.pack["fallback"]["asr_failure"])
            return

        # Whisper labels every transcription with a language, including the
        # ones it invents for noise. On a market whose callers speak a known
        # set, a detection outside that set is the strongest signal available
        # that nobody actually spoke: observed as a Japanese turn and then a
        # Spanish one, both produced from room noise on an en-IN call.
        #
        # Dropped silently rather than answered with "I didn't catch that",
        # because there was no utterance to have missed - and a caller who did
        # speak simply says it again.
        allowed = self.asr_languages
        if allowed and tr.language and tr.language.strip().lower() not in allowed:
            log("call.language_rejected", call_id=state.call_id,
                detected=tr.language, text=tr.text[:80],
                audio_s=round(duration_seconds(pcm), 2))
            await self.on_event({"type": "idle"})
            return

        await self.on_event({
            "type": "transcript", "speaker": "customer", "text": tr.text,
            "language": tr.language, "asr_ms": round(asr_ms),
        })

        result = await self.session.handle(tr.text)
        if result.text:
            await self.speak(result.text, meta={
                "intent": result.intent,
                "phase": state.phase.value,
                "grounded": result.grounded,
                "think_ms": round(result.latency_ms),
                "asr_ms": round(asr_ms),
            })
        if result.ended:
            await self.on_event({"type": "call_ended", "reason": state.ended_reason,
                                 "summary": self.session.transcript()})

    async def speak(self, text: str, *, meta: dict | None = None) -> None:
        if not text.strip():
            return
        await self.on_event({"type": "transcript", "speaker": "agent", "text": text,
                             **(meta or {})})
        # Each utterance starts its own barge-in state. Without this reset an
        # interruption during the greeting - which is spoken by open(), outside
        # the _flush() whose `finally` used to do the clearing - left _barged
        # set for the rest of the call, so every later utterance was cut off
        # after its first frame.
        self._barged = False
        t_tts = time.perf_counter()
        # The agent counts as speaking from here, not from the moment audio
        # starts flowing. Synthesis is the longest part of the turn (seconds,
        # and ~11 s on a network where the provider's TLS is being reset), and
        # for all of it the old code left agent_speaking False: the echo guard
        # was off and an interruption could not be registered at all.
        self.agent_speaking = True
        try:
            pcm = await get_tts().synthesize(text, self.locale_voice)
        except Exception as exc:  # noqa: BLE001
            log("call.tts_failed", error=str(exc)[:200])
            self.agent_speaking = False
            self._echo_guard_until = (time.perf_counter()
                                      + settings.echo_guard_tail_ms / 1000.0)
            return
        if self._barged:
            # Interrupted while the words were still being synthesised. Playing
            # them now would talk over a caller who has already started.
            log("call.playback_skipped", call_id=self.session.state.call_id,
                reason="barged_during_synthesis")
            self.agent_speaking = False
            await self.on_event({"type": "idle"})
            return
        tts_ms = (time.perf_counter() - t_tts) * 1000
        self.session.latency.record("tts_ms", tts_ms)
        await self.on_event({"type": "speaking", "tts_ms": round(tts_ms)})

        # The microphone stream is the call's clock: it arrives continuously, in
        # real time, for as long as the call is up. So the customer track is
        # written only by push_audio, and the agent track is padded up to
        # wherever that clock has reached before its own audio is appended.
        #
        # Previously both were advanced during playback - push_audio appended
        # the live microphone *and* speak() appended an equal run of zeros - so
        # the customer channel ran at roughly double rate whenever the agent
        # talked. Measured on a real call: agent audio ended at 58.3 s while the
        # customer channel ran to 142.5 s of the same recording, which makes the
        # stereo file useless as evidence and unusable by the Q4 analysis.
        self._align_agent_track()
        self.agent_track.extend(pcm)
        try:
            # Stream in ~200 ms frames so barge-in can cut it off mid-sentence.
            frame = 3200 * 2
            for i in range(0, len(pcm), frame):
                if self._barged:
                    log("call.playback_stopped", call_id=self.session.state.call_id)
                    break
                await self.on_audio(pcm[i:i + frame])
                await asyncio.sleep(0.01)
        finally:
            self.agent_speaking = False
            self._echo_guard_until = (time.perf_counter()
                                      + settings.echo_guard_tail_ms / 1000.0)
        await self.on_event({"type": "idle"})

    async def open(self) -> None:
        """Speak the opening in parts, leaving the caller room between them."""
        for i, (text, ids) in enumerate(self.session.opening_parts()):
            if not text.strip():
                continue
            if self._barged:
                # They cut in during the first part. The rest is not abandoned:
                # session.handle() re-delivers any outstanding disclosure before
                # qualification, which is where the rule actually applies.
                log("call.opening_interrupted", call_id=self.session.state.call_id,
                    delivered_parts=i)
                break
            await self.speak(text, meta={"phase": "greeting"})
            # Recorded after the audio has gone out, so the transcript's
            # disclosure list is a record of what was heard, not what was built.
            self.session.mark_disclosed(ids, text)

    # -------------------------------------------------------------- artefacts
    def _align_agent_track(self) -> None:
        """Pad the agent channel with silence up to the microphone's position."""
        gap = len(self.customer_track) - len(self.agent_track)
        if gap > 0:
            self.agent_track.extend(b"\x00" * gap)

    def save_recording(self) -> str | None:
        if not self.agent_track and not self.customer_track:
            return None
        # One last pad, so a call that ends while the customer is talking does
        # not leave the agent channel short.
        self._align_agent_track()
        settings.recordings_dir.mkdir(parents=True, exist_ok=True)
        path = settings.recordings_dir / (self.session.state.call_id + ".wav")
        stereo = mix_stereo(bytes(self.agent_track), bytes(self.customer_track))
        write_wav(path, stereo, channels=2)
        # The WAV is gitignored and the MP3 is committed, so the MP3 is written
        # here rather than by a separate step someone has to remember.
        mp3 = to_mp3(path)
        log("call.recording_saved", path=str(path), mp3=bool(mp3),
            seconds=round(len(stereo) / (2 * 2 * 16000), 1))
        return str(path)

    def finalise(self) -> dict:
        recording = self.save_recording()
        transcript_path = self.session.save_transcript()
        return {"recording": recording, "transcript": transcript_path}
