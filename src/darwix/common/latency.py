"""Latency instrumentation.

Q4 requires P50/P95 for the whole chain *and* per component. Rather than
timing things by hand at report time, every stage is stamped as it runs and
the report is derived from the collected samples.
"""
from __future__ import annotations

import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class LatencyCollector:
    """Collects per-stage durations in milliseconds."""

    samples: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    @contextmanager
    def stage(self, name: str) -> Iterator[dict]:
        meta: dict = {}
        t0 = time.perf_counter()
        try:
            yield meta
        finally:
            self.record(name, (time.perf_counter() - t0) * 1000.0)

    def record(self, name: str, millis: float) -> None:
        self.samples[name].append(millis)

    def merge(self, other: "LatencyCollector") -> None:
        for k, v in other.samples.items():
            self.samples[k].extend(v)

    @staticmethod
    def _pct(values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        k = (len(ordered) - 1) * pct
        lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, values in self.samples.items():
            if not values:
                continue
            out[name] = {
                "n": len(values),
                "min_ms": round(min(values), 1),
                "p50_ms": round(self._pct(values, 0.50), 1),
                "p95_ms": round(self._pct(values, 0.95), 1),
                "max_ms": round(max(values), 1),
                "mean_ms": round(statistics.fmean(values), 1),
            }
        return out


def now_ms() -> float:
    return time.perf_counter() * 1000.0
