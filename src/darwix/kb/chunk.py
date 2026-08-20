"""Chunking strategy.

Chunk boundaries decide what the bot can and cannot say, so they follow the
document's own structure rather than a character count:

* **Split on headings first.** The cleaner emits `## heading` lines, including
  FAQ questions promoted from accordion buttons. A section becomes a chunk, so
  a question and its answer stay together and the heading becomes the
  `source_locator` in the citation.

* **Never split a rules table row.** Rows in the qualification matrix and the
  schedule of charges are atomic: half a rule is worse than no rule, because it
  retrieves confidently and answers wrongly.

* **Target 250-400 tokens with ~60 token overlap.** Small enough that a spoken
  answer is grounded in something the agent can actually paraphrase in one
  breath; large enough to keep an eligibility clause with its exceptions.

* **Oversized sections split on sentence boundaries**, and every part keeps the
  parent heading so context is never orphaned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_TOKENS = 320
MAX_TOKENS = 420
MIN_TOKENS = 25
OVERLAP_TOKENS = 60
ATOMIC_PREFIXES = ("| ", "- rule_id", "| rule_id")


def estimate_tokens(text: str) -> int:
    # ~4 chars/token is close enough for chunk sizing and needs no tokenizer
    # dependency that must match the serving model.
    return max(1, len(text) // 4)


@dataclass
class Chunk:
    heading: str
    text: str
    index: int
    token_estimate: int
    atomic: bool = False


def _split_sections(text: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    heading = ""
    buffer: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if buffer or heading:
                sections.append((heading, buffer))
            heading = line[3:].strip()
            buffer = []
        else:
            buffer.append(line)
    if buffer or heading:
        sections.append((heading, buffer))
    return [(h, b) for h, b in sections if h or any(l.strip() for l in b)]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


def _pack(lines: list[str], heading: str, start_index: int) -> list[Chunk]:
    """Pack lines into chunks, keeping atomic rows whole."""
    chunks: list[Chunk] = []
    current: list[str] = []
    idx = start_index

    def flush(atomic: bool = False) -> None:
        nonlocal current, idx
        body = "\n".join(l for l in current if l.strip())
        if not body.strip():
            current = []
            return
        chunks.append(Chunk(heading, body, idx, estimate_tokens(body), atomic))
        idx += 1
        current = []

    for line in lines:
        is_atomic = line.strip().startswith(ATOMIC_PREFIXES)
        candidate = current + [line]
        if estimate_tokens("\n".join(candidate)) > MAX_TOKENS and current:
            flush(atomic=is_atomic)
            # overlap: carry the tail of the previous chunk for continuity,
            # but never carry an atomic row into another chunk
            if chunks and not chunks[-1].atomic:
                tail = chunks[-1].text[-OVERLAP_TOKENS * 4:]
                current = [tail]
        if estimate_tokens(line) > MAX_TOKENS and not is_atomic:
            for sent in _sentences(line):
                if estimate_tokens("\n".join(current + [sent])) > MAX_TOKENS and current:
                    flush()
                current.append(sent)
            continue
        current.append(line)
    flush()
    return chunks


def chunk_document(text: str, *, doc_title: str = "") -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for heading, lines in _split_sections(text):
        heading = heading or doc_title
        body = "\n".join(lines).strip()
        if not body:
            continue
        if estimate_tokens(body) <= MAX_TOKENS:
            atomic = body.lstrip().startswith(ATOMIC_PREFIXES)
            chunks.append(Chunk(heading, body, idx, estimate_tokens(body), atomic))
            idx += 1
        else:
            packed = _pack(lines, heading, idx)
            chunks.extend(packed)
            idx += len(packed)

    # merge runt chunks into their neighbour: a 12-token fragment retrieves
    # noisily and cites nothing useful
    # Runts are merged into the PRECEDING chunk only when they belong to the
    # same section. Merging across a heading boundary would put text under a
    # heading it did not come from, and the heading is the source locator the
    # citation is built from - so the citation would then be wrong.
    merged: list[Chunk] = []
    for ch in chunks:
        if (
            merged
            and ch.token_estimate < MIN_TOKENS
            and ch.heading == merged[-1].heading
            and not ch.atomic
            and not merged[-1].atomic
            and merged[-1].token_estimate + ch.token_estimate <= MAX_TOKENS
        ):
            prev = merged[-1]
            prev.text = prev.text + "\n" + ch.text
            prev.token_estimate = estimate_tokens(prev.text)
        else:
            merged.append(ch)
    for i, ch in enumerate(merged):
        ch.index = i
    return merged
