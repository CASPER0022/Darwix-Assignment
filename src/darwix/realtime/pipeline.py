"""The Q4 pipeline: live audio -> transcript -> signals -> nudges -> delivery.

    audio chunk --> VAD --> ASR --> [rules layer]    --> nudge engine --> sink
                                 \-> [LLM layer]    /

End-to-end latency is defined as: the moment the last audio sample of an
utterance arrives, to the moment a nudge derived from it is handed to the
delivery sink. Every stage stamps itself, so the numbers in
evaluation/latency_report.md come from real runs rather than estimates.

The rules layer runs synchronously on every segment (microseconds). The LLM
layer runs debounced and concurrently, so a slow model call delays only its own
nudges, never transcription and never the rule-based compliance alerts.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from ..common.config import settings
from ..common.latency import LatencyCollector
from ..common.logging import log
from .nudge_engine import Nudge, NudgeEngine
from .signals.llm import LLMSignalExtractor
from .signals.rules import RuleSignals, Signal
from .stream import ReplaySource, Segment, TranscriptStream

Sink = Callable[[dict], Awaitable[None]]


def _locale_languages(locale: str) -> set[str]:
    """The languages a caller on `locale` plausibly speaks.

    Q4 listens to the same Whisper as Q1 and inherits the same failure: with no
    language forced, silence and noise come back labelled as some unrelated
    language. Reading the allowlist from the locale pack keeps the two paths
    agreeing without a second copy of the list.
    """
    from ..voice.dialog_policy import load_pack

    try:
        pack = load_pack(locale.replace("-", "_"))
    except FileNotFoundError:
        return set()
    return {str(x).strip().lower()
            for x in (pack.get("voice", {}).get("asr_languages") or [])}


@dataclass
class LiveAnalysis:
    call_id: str
    locale: str = "en-IN"
    asr_prompt: str = ""
    language: str | None = None
    use_llm_layer: bool = True
    require_payment_disclosure: bool = False

    engine: NudgeEngine = field(default_factory=NudgeEngine)
    rules: RuleSignals = field(default=None)  # type: ignore[assignment]
    llm: LLMSignalExtractor = field(default_factory=LLMSignalExtractor)
    latency: LatencyCollector = field(default_factory=LatencyCollector)
    turns: list[dict] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    started_at: float = 0.0
    _sinks: list[Sink] = field(default_factory=list)
    _llm_tasks: set = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.rules is None:
            self.rules = RuleSignals(require_payment_disclosure=self.require_payment_disclosure)

    def add_sink(self, sink: Sink) -> None:
        self._sinks.append(sink)

    async def _emit(self, payload: dict) -> None:
        for sink in self._sinks:
            try:
                await sink(payload)
            except Exception as exc:  # noqa: BLE001 - a dead dashboard must not stop analysis
                log("pipeline.sink_failed", error=str(exc)[:160])

    # --------------------------------------------------------------- segments
    async def on_segment(self, seg: Segment) -> None:
        self.segments.append(seg)
        self.turns.append({"speaker": seg.speaker, "text": seg.text,
                           "call_time_s": seg.call_time_s})
        self.latency.record("asr_ms", seg.asr_latency_ms)
        await self._emit({"type": "transcript", **seg.as_dict()})

        # ---- rules layer: synchronous, microseconds ----
        t0 = time.perf_counter()
        signals = self.rules.observe(seg.speaker, seg.text, seg.call_time_s)
        signals += self.rules.check_deadlines(seg.call_time_s)
        rules_ms = (time.perf_counter() - t0) * 1000.0
        self.latency.record("signal_rules_ms", rules_ms)

        for signal in signals:
            await self._consider(signal, seg, extra_ms=rules_ms)

        # ---- LLM layer: debounced, concurrent ----
        if self.use_llm_layer and seg.speaker == "customer" and self.llm.due():
            task = asyncio.create_task(self._llm_pass(seg))
            self._llm_tasks.add(task)
            task.add_done_callback(self._llm_tasks.discard)

    async def _llm_pass(self, seg: Segment) -> None:
        known = {n.kind for n in self.engine.active}
        signals, llm_ms = await self.llm.extract(
            self.turns, known_kinds=known, call_time_s=seg.call_time_s
        )
        self.latency.record("signal_llm_ms", llm_ms)
        for signal in signals:
            await self._consider(signal, seg, extra_ms=llm_ms)

    async def _consider(self, signal: Signal, seg: Segment, *, extra_ms: float = 0.0) -> None:
        await self._emit({"type": "signal", **signal.as_dict()})
        # End-to-end: from the last audio sample of the utterance to now.
        detection_ms = (time.perf_counter() - seg.audio_end_at) * 1000.0
        nudge = self.engine.consider(signal, detection_latency_ms=detection_ms)
        if nudge is None:
            return
        t0 = time.perf_counter()
        await self._emit({"type": "nudge", **nudge.as_dict()})
        delivery_ms = (time.perf_counter() - t0) * 1000.0
        self.latency.record("delivery_ms", delivery_ms)
        self.latency.record("end_to_end_ms", detection_ms + delivery_ms)
        log("pipeline.nudge_delivered", kind=nudge.kind,
            end_to_end_ms=round(detection_ms + delivery_ms))

    # ------------------------------------------------------------------- runs
    async def run_replay(self, wav_path: Path, *, speed: float = 1.0) -> dict:
        self.started_at = time.perf_counter()
        await self._emit({"type": "call_started", "call_id": self.call_id,
                          "source": str(wav_path), "speed": speed})
        stream = TranscriptStream(asr_prompt=self.asr_prompt, language=self.language,
                                  languages=_locale_languages(self.locale))
        await stream.run(ReplaySource(wav_path, speed=speed), self.on_segment)

        while self._llm_tasks:
            await asyncio.gather(*list(self._llm_tasks), return_exceptions=True)

        # Anything still unsaid at the end of the call is a missed disclosure.
        final_time = self.segments[-1].call_time_s if self.segments else 0.0
        for signal in self.rules.check_deadlines(max(final_time, 999.0)):
            await self._consider(signal, self.segments[-1] if self.segments else
                                 Segment("agent", "", 0, time.perf_counter(),
                                         time.perf_counter(), 0, 0))

        report = self.report(wall_clock_s=time.perf_counter() - self.started_at)
        await self._emit({"type": "call_ended", "report": report})
        return report

    def report(self, *, wall_clock_s: float = 0.0) -> dict:
        return {
            "call_id": self.call_id,
            "locale": self.locale,
            "wall_clock_s": round(wall_clock_s, 1),
            "segments": len(self.segments),
            "transcript": [s.as_dict() for s in self.segments],
            "compliance": self.rules.summary(),
            "latency": self.latency.summary(),
            **self.engine.report(),
        }

    def save(self, path: Path | None = None) -> Path:
        path = path or (settings.transcripts_dir / ("live_" + self.call_id + ".json"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report(), indent=2, ensure_ascii=False),
                        encoding="utf-8")
        log("pipeline.saved", path=str(path))
        return path


async def analyse_file(
    wav_path: Path,
    *,
    call_id: str | None = None,
    speed: float = 1.0,
    use_llm: bool = True,
    sinks: list[Sink] | None = None,
    locale: str = "en-IN",
) -> dict:
    analysis = LiveAnalysis(
        call_id=call_id or Path(wav_path).stem,
        locale=locale,
        use_llm_layer=use_llm,
    )
    for sink in (sinks or []):
        analysis.add_sink(sink)
    report = await analysis.run_replay(Path(wav_path), speed=speed)
    analysis.save()
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Analyse a call in real time and emit nudges while it plays")
    ap.add_argument("wav", help="path to a call recording (stereo: agent L, customer R)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="1.0 = true real time (default). Higher only for debugging.")
    ap.add_argument("--no-llm", action="store_true", help="rules layer only")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    async def cli_sink(payload: dict) -> None:
        if args.quiet:
            return
        kind = payload.get("type")
        if kind == "transcript":
            print(f"  [{payload['call_time_s']:6.1f}s] {payload['speaker']:8s} "
                  f"{payload['text'][:96]}")
        elif kind == "nudge":
            print(f"  >>> NUDGE  p{payload['priority']} [{payload['kind']}] "
                  f"{payload['text']}  ({payload['latency_ms']:.0f} ms)")

    report = asyncio.run(analyse_file(Path(args.wav), speed=args.speed,
                                      use_llm=not args.no_llm, sinks=[cli_sink]))
    print()
    print("wall clock:", report["wall_clock_s"], "s |", report["segments"], "segments")
    print("suppression:", json.dumps(report["suppression"], indent=2))
    print("latency:", json.dumps(report["latency"], indent=2))
