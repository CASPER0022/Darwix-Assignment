"""Duplicate and near-duplicate removal.

Three distinct problems, three mechanisms, because one technique does not cover
them:

1. **Exact duplicates** - the same page reachable at two URLs
   (`/loan-against-property` and `/secured-loan-against-property` are
   byte-identical after cleaning). Caught by content hash.

2. **Near duplicates** - a paragraph reused across a policy PDF and a partner
   FAQ with small edits. Caught by SimHash over token shingles with a Hamming
   distance threshold. SimHash is chosen over MinHash/LSH because the corpus is
   small enough that an O(n^2) comparison is instant, and SimHash needs no
   tuning of band/row parameters to explain.

3. **Translations** - the same Fair Practices Code published in ten languages.
   These share no tokens, so neither hash catches them; they are resolved by
   the language tag from `normalize.detect_language` and kept out of the
   English agent's retrieval scope.

Nothing is deleted. The survivor records `merged_from`, so every merged source
is still traceable - which is what "source tracking" has to mean if a citation
is going to be defensible.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

SHINGLE = 4
SIMHASH_BITS = 64
NEAR_DUP_MAX_DISTANCE = 6

# SimHash is designed for long documents. On a short paragraph there are too
# few shingles for the signature to be stable: measured, a single added word
# moved 8 of 64 bits, well past the threshold, so a genuine near-duplicate was
# kept. Below this length the comparison switches to exact shingle Jaccard,
# which is affordable at this corpus size and has no such blind spot.
SHORT_DOC_TOKENS = 80
JACCARD_NEAR_DUP = 0.82


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def content_hash(text: str) -> str:
    normalised = " ".join(_tokens(text))
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:20]


def simhash(text: str, bits: int = SIMHASH_BITS) -> int:
    toks = _tokens(text)
    if len(toks) < SHINGLE:
        shingles = [" ".join(toks)] if toks else []
    else:
        shingles = [" ".join(toks[i:i + SHINGLE]) for i in range(len(toks) - SHINGLE + 1)]
    if not shingles:
        return 0
    vector = [0] * bits
    for sh in shingles:
        h = int(hashlib.md5(sh.encode("utf-8")).hexdigest(), 16)
        for b in range(bits):
            vector[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b in range(bits):
        if vector[b] > 0:
            out |= 1 << b
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def shingles(text: str, n: int = SHINGLE) -> set[str]:
    toks = _tokens(text)
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: str, b: str) -> float:
    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_near_duplicate(a: str, b: str, sig_a: int, sig_b: int) -> tuple[bool, int]:
    """Returns (is_duplicate, reported_distance)."""
    if min(len(_tokens(a)), len(_tokens(b))) < SHORT_DOC_TOKENS:
        score = jaccard(a, b)
        return score >= JACCARD_NEAR_DUP, int(round((1 - score) * 100))
    dist = hamming(sig_a, sig_b)
    return dist <= NEAR_DUP_MAX_DISTANCE, dist


@dataclass
class DedupeReport:
    total_in: int = 0
    exact_removed: int = 0
    near_removed: int = 0
    kept: int = 0
    pairs: list[tuple[str, str, int]] = field(default_factory=list)  # (kept, dropped, distance)

    def as_dict(self) -> dict:
        return {
            "total_in": self.total_in,
            "exact_duplicates_merged": self.exact_removed,
            "near_duplicates_merged": self.near_removed,
            "kept": self.kept,
            "reduction_pct": round(
                100.0 * (self.exact_removed + self.near_removed) / max(1, self.total_in), 1
            ),
            "examples": [
                {"kept": k, "merged": d, "hamming_distance": dist} for k, d, dist in self.pairs[:12]
            ],
        }


def deduplicate(records: list) -> tuple[list, DedupeReport]:
    """Merge duplicates among KBRecord-like objects (needs .content, .record_id,
    .merged_from, .title, .language)."""
    report = DedupeReport(total_in=len(records))
    by_hash: dict[str, object] = {}
    survivors: list = []

    # pass 1: exact
    for rec in records:
        h = content_hash(rec.content)
        existing = by_hash.get(h)
        if existing is None:
            by_hash[h] = rec
            survivors.append(rec)
        else:
            existing.merged_from.append(rec.record_id)  # type: ignore[attr-defined]
            report.exact_removed += 1
            report.pairs.append((existing.record_id, rec.record_id, 0))  # type: ignore[attr-defined]

    # pass 2: near, within the same language only
    kept: list = []
    signatures: list[tuple[int, object]] = []
    for rec in survivors:
        sig = simhash(rec.content)
        match = None
        for other_sig, other in signatures:
            if getattr(other, "language", "en") != getattr(rec, "language", "en"):
                continue
            duplicate, dist = is_near_duplicate(rec.content, other.content, sig, other_sig)
            if duplicate:
                match = (other, dist)
                break
        if match is None:
            signatures.append((sig, rec))
            kept.append(rec)
        else:
            other, dist = match
            # keep the longer, more informative version
            if len(rec.content) > len(other.content) * 1.25:
                other.content, rec.content = rec.content, other.content
                other.title = rec.title or other.title
            other.merged_from.append(rec.record_id)  # type: ignore[attr-defined]
            report.near_removed += 1
            report.pairs.append((other.record_id, rec.record_id, dist))  # type: ignore[attr-defined]

    report.kept = len(kept)
    return kept, report
