"""Structured logging. One JSON object per line so runs can be replayed and
measured after the fact -- the latency report is built from these events."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

_LOG_FILE: Path | None = None

# Windows consoles default to cp1252, which raises UnicodeEncodeError the first
# time a Tagalog, Indonesian or Devanagari string is printed. Every stream this
# project writes to is forced to UTF-8 at import time.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover - non-standard streams
        pass


def set_log_file(path: Path | None) -> None:
    global _LOG_FILE
    _LOG_FILE = path
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)


def log(event: str, **fields: Any) -> None:
    record = {"ts": round(time.time(), 4), "event": event, **fields}
    line = json.dumps(record, ensure_ascii=False, default=str)
    print(line, file=sys.stderr, flush=True)
    if _LOG_FILE:
        with _LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
