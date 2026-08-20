"""Streaming audio -> speaker-attributed transcript segments, in real time.

Two input modes, one code path:

* `LiveSource`  - audio pushed in as it arrives (a real call).
* `ReplaySource` - a recording played back **at true wall-clock speed**. The
  assessment is explicit that analysing a completed recording after upload does
  not qualify, so replay sleeps between chunks and the pipeline genuinely
  cannot see the future. A 90-second call takes 90 seconds to analyse.

Speaker separation comes from the channel layout, not a diarisation model:
recordings are written agent-left / customer-right (see common/audio.mix_stereo),
which is how contact-centre recorders work. Each channel gets its own VAD, so a
segment is emitted when *that speaker* stops talking - which also means only the
channel that actually contains speech is sent to ASR. In a normal call only one
party talks at a time, so this costs roughly one ASR call per utterance rather
than two per window.

Every segment carries the timestamps needed for the latency report:
`audio_end_at` (when the audio arrived) and `transcribed_at` (when text was
ready). Nothing downstream has to guess.
"""
from __future__ import annotations

import array
import asyncio
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable

from ..common.asr import Transcript, get_asr
from ..common.audio import SAMPLE_RATE, EnergyVAD, duration_seconds, has_speech
from ..common.logging import log

CHUNK_MS = 200
MIN_SEGMENT_S = 0.4
MAX_SEGMENT_S = 20.0


@dataclass
class Segment:
    speaker: str            # "agent" | "customer"
    text: str
    started_at: float       # monotonic seconds, when the utterance began
    audio_end_at: float     # when the last audio sample of it arrived
    transcribed_at: float   # when the text was available
    call_time_s: float      # position within the call, for the transcript
    duration_s: float
    language: str = ""

    @property
    def asr_latency_ms(self) -> float:
        return (self.transcribed_at - self.audio_end_at) * 1000.0

    def as_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "text": self.text,
            "call_time_s": round(self.call_time_s, 2),
            "duration_s": round(self.duration_s, 2),
            "asr_latency_ms": round(self.asr_latency_ms, 1),
            "language": self.language,
        }


class ChannelTranscriber:
    """One VAD + ASR pipeline for a single speaker channel."""

    # 300 ms of pre-roll. The VAD needs ~250 ms of energy before it declares
    # speech, so by the time capture starts the first word is already past.
    # Transcripts came back as "is this?" for "Who is this?" until this ring
    # buffer was added, and a truncated first word breaks keyword signals.
    PREROLL_MS = 300

    def __init__(self, speaker: str, *, asr_prompt: str = "", language: str | None = None,
                 languages: set[str] | None = None, script: str = "latin") -> None:
        self.speaker = speaker
        # Q1 gained both of these when a turn arrived as Japanese and then as
        # Spanish on an en-IN call. Q4 runs the same Whisper against the same
        # kind of audio, so it needs the same two guards - here a fabricated
        # segment does not just print a wrong line, it can fire a nudge at an
        # agent about something nobody said.
        self.languages = languages or set()
        self.script = script
        self.vad = EnergyVAD(hangover_ms=800.0)
        self.preroll = bytearray()
        self.buffer = bytearray()
        self.capturing = False
        self.started_at = 0.0
        self.call_time_s = 0.0
        self.asr_prompt = asr_prompt
        self.language = language
        self._elapsed = 0.0

    async def push(self, pcm: bytes) -> bytes | None:
        """Feed a chunk; return a completed utterance's PCM when one closes."""
        event = self.vad.push(pcm)
        self._elapsed += duration_seconds(pcm)

        if event == "speech_start":
            self.capturing = True
            self.started_at = time.perf_counter()
            self.call_time_s = max(0.0, self._elapsed - self.PREROLL_MS / 1000.0)
            self.buffer.clear()
            self.buffer.extend(self.preroll)   # recover the utterance onset

        if self.capturing:
            self.buffer.extend(pcm)
            if duration_seconds(bytes(self.buffer)) >= MAX_SEGMENT_S:
                return self._take()
        else:
            self.preroll.extend(pcm)
            keep = int(SAMPLE_RATE * (self.PREROLL_MS / 1000.0)) * 2
            if len(self.preroll) > keep:
                del self.preroll[:len(self.preroll) - keep]

        if event == "speech_end" and self.capturing:
            return self._take()
        return None

    def _take(self) -> bytes | None:
        pcm = bytes(self.buffer)
        self.buffer.clear()
        self.preroll.clear()
        self.capturing = False
        if duration_seconds(pcm) < MIN_SEGMENT_S:
            return None
        # Same guard as the Q1 path: silence sent to Whisper comes back as
        # "Thank you.", which on this side would become a spurious transcript
        # line and could fire a nudge on a customer who never spoke.
        if not has_speech(pcm):
            log("stream.segment_dropped", speaker=self.speaker,
                reason="no_speech_energy", audio_s=round(duration_seconds(pcm), 2))
            return None
        return pcm

    async def transcribe(self, pcm: bytes, audio_end_at: float) -> Segment | None:
        try:
            tr: Transcript = await get_asr().transcribe_pcm(
                pcm, language=self.language, prompt=self.asr_prompt
            )
        except Exception as exc:  # noqa: BLE001 - a dropped segment must not kill the stream
            log("stream.asr_failed", speaker=self.speaker, error=str(exc)[:200])
            return None
        tr.script = self.script
        if tr.is_empty:
            return None
        if self.languages and tr.language and tr.language.strip().lower() not in self.languages:
            log("stream.language_rejected", speaker=self.speaker,
                detected=tr.language, text=tr.text[:80])
            return None
        return Segment(
            speaker=self.speaker,
            text=tr.text,
            started_at=self.started_at,
            audio_end_at=audio_end_at,
            transcribed_at=time.perf_counter(),
            call_time_s=self.call_time_s,
            duration_s=duration_seconds(pcm),
            language=tr.language,
        )


class ReplaySource:
    """Replays a WAV at real-time speed, yielding (agent_pcm, customer_pcm)."""

    def __init__(self, path: Path, *, speed: float = 1.0) -> None:
        self.path = Path(path)
        self.speed = speed

    async def stream(self) -> AsyncIterator[tuple[bytes, bytes]]:
        with wave.open(str(self.path), "rb") as wf:
            channels, rate = wf.getnchannels(), wf.getframerate()
            frames_per_chunk = int(rate * (CHUNK_MS / 1000.0))
            total = wf.getnframes()
            log("replay.start", path=str(self.path), channels=channels, rate=rate,
                seconds=round(total / rate, 1), speed=self.speed)
            started = time.perf_counter()
            sent = 0
            while True:
                raw = wf.readframes(frames_per_chunk)
                if not raw:
                    break
                if channels == 2:
                    samples = array.array("h", raw)
                    left = array.array("h", samples[0::2]).tobytes()
                    right = array.array("h", samples[1::2]).tobytes()
                else:
                    # Mono input: no channel separation available. Everything is
                    # attributed to the customer, and the write-up says so.
                    left, right = b"", raw
                sent += frames_per_chunk
                yield left, right
                # This sleep is the whole point: it forces the analysis to run
                # at the speed the conversation actually happens.
                target = started + (sent / rate) / self.speed
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)


class TranscriptStream:
    """Drives both channels and emits segments as soon as they are ready."""

    def __init__(self, *, asr_prompt: str = "", language: str | None = None,
                 languages: set[str] | None = None, script: str = "latin") -> None:
        self.agent = ChannelTranscriber("agent", asr_prompt=asr_prompt, language=language,
                                        languages=languages, script=script)
        self.customer = ChannelTranscriber("customer", asr_prompt=asr_prompt, language=language,
                                           languages=languages, script=script)
        self.segments: list[Segment] = []
        self._tasks: set[asyncio.Task] = set()

    async def run(self, source: ReplaySource, on_segment: Callable) -> list[Segment]:
        queue: asyncio.Queue = asyncio.Queue()

        async def transcribe_and_emit(ch: ChannelTranscriber, pcm: bytes, at: float) -> None:
            seg = await ch.transcribe(pcm, at)
            if seg:
                await queue.put(seg)

        async def pump() -> None:
            async for left, right in source.stream():
                now = time.perf_counter()
                for ch, pcm in ((self.agent, left), (self.customer, right)):
                    if not pcm:
                        continue
                    done = await ch.push(pcm)
                    if done:
                        # Transcribe concurrently so the stream never stalls
                        # waiting on the network.
                        task = asyncio.create_task(transcribe_and_emit(ch, done, now))
                        self._tasks.add(task)
                        task.add_done_callback(self._tasks.discard)
            # flush whatever is still buffered at end of call
            for ch in (self.agent, self.customer):
                pcm = ch._take()
                if pcm:
                    task = asyncio.create_task(transcribe_and_emit(ch, pcm, time.perf_counter()))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
            while self._tasks:
                await asyncio.gather(*list(self._tasks), return_exceptions=True)
            await queue.put(None)

        pump_task = asyncio.create_task(pump())
        while True:
            seg = await queue.get()
            if seg is None:
                break
            self.segments.append(seg)
            log("stream.segment", speaker=seg.speaker, t=round(seg.call_time_s, 1),
                asr_ms=round(seg.asr_latency_ms), text=seg.text[:90])
            await on_segment(seg)
        await pump_task
        return self.segments
