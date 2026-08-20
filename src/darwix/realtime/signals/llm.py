"""LLM signal extraction over a rolling window.

Scope is deliberately narrow. The rules layer already catches everything that
can be caught by a keyword, and it catches it instantly. This layer is asked
only for the judgements a keyword cannot make:

    - is frustration *building* across turns, or was that one sharp sentence?
    - did the customer imply an opportunity without naming it?
    - has the topic shifted away from what the agent is still pursuing?
    - is there a risk in what the agent said, in context?

Design constraints that come from Q4 being a latency exercise:

* **Small model, low reasoning effort.** ~800 ms on Groq's 20B. The 120B is
  better at nuance and roughly 40% slower; on a live call that trade is wrong.
* **Rolling window, not the whole call.** Sending the full transcript every few
  seconds re-pays for tokens already analysed and grows without bound. The
  window is the last N turns, and Groq's free tier caps at 8k tokens/minute -
  which is a real constraint that forces the right design anyway.
* **Debounced.** It runs at most every `min_interval_s`, and only after a
  customer turn. Analysing the agent's own greeting produces nothing.
* **Fails silently.** A timeout drops one analysis pass. The rules layer keeps
  running, so the call is never left without signal detection.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from ...common.llm import get_llm
from ...common.logging import log
from .rules import Signal

SYSTEM = """You watch a live sales/service phone call and report signals a rule cannot catch.

Report ONLY things clearly evidenced in the transcript window. Reporting nothing is
a correct and common answer. Do not repeat a signal that is already listed as known.

Signal kinds you may report:
  rising_frustration   - irritation building across turns, not one sharp phrase
  buying_signal        - readiness to proceed, implied rather than stated
  missed_opportunity   - a need the agent has not responded to
  topic_shift          - the customer has moved on and the agent has not
  payment_difficulty   - inability to pay, implied
  risk                 - something the agent said that could be a compliance or mis-selling problem
  confusion            - the customer has not understood and has not said so

Return JSON only:
{"signals": [{"kind": "...", "detail": "<one short sentence>", "confidence": 0.0-1.0,
              "evidence": "<a short quote from the window>"}]}

confidence must reflect real certainty: 0.9 only when the evidence is explicit,
0.5-0.7 when it is an inference, and do not report below 0.5."""


@dataclass
class LLMSignalExtractor:
    window_turns: int = 8
    min_interval_s: float = 6.0
    _last_run: float = 0.0
    _known: tuple = ()

    def due(self, now: float | None = None) -> bool:
        now = now if now is not None else time.perf_counter()
        return (now - self._last_run) >= self.min_interval_s

    async def extract(
        self,
        turns: list[dict],
        *,
        known_kinds: set[str] | None = None,
        call_time_s: float = 0.0,
    ) -> tuple[list[Signal], float]:
        """Returns (signals, llm_latency_ms)."""
        self._last_run = time.perf_counter()
        window = turns[-self.window_turns:]
        if not window:
            return [], 0.0

        transcript = "\n".join(t["speaker"].upper() + ": " + t["text"] for t in window)
        known = sorted(known_kinds or set())
        user = "TRANSCRIPT WINDOW:\n" + transcript
        if known:
            user += "\n\nALREADY KNOWN, do not repeat: " + ", ".join(known)

        t0 = time.perf_counter()
        try:
            data = await get_llm().chat_json(
                SYSTEM, [{"role": "user", "content": user}],
                provider="groq_fast", temperature=0.1, max_tokens=400,
            )
        except Exception as exc:  # noqa: BLE001 - never let this stall the call
            log("signal.llm_failed", error=str(exc)[:180])
            return [], (time.perf_counter() - t0) * 1000.0
        latency_ms = (time.perf_counter() - t0) * 1000.0

        out: list[Signal] = []
        for row in (data.get("signals") or [])[:5]:
            kind = str(row.get("kind", "")).strip()
            if not kind:
                continue
            try:
                confidence = float(row.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < 0.5:
                continue
            out.append(Signal(
                kind=kind,
                detail=str(row.get("detail", ""))[:200],
                confidence=min(1.0, confidence),
                speaker="customer",
                evidence=str(row.get("evidence", ""))[:200],
                at_call_time_s=call_time_s,
                source="llm",
            ))
        log("signal.llm", found=len(out), latency_ms=round(latency_ms),
            kinds=[s.kind for s in out])
        return out, latency_ms
