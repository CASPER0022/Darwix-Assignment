"""Hybrid retrieval with citations.

Ranking = BM25 (lexical) + cosine (dense), fused with Reciprocal Rank Fusion.

Why both, and why RRF:

* Dense-only fails the questions this bot is actually asked. "What is the
  bounce charge?" needs the record containing the literal string "bounce
  charge"; embeddings happily return three semantically adjacent fee records
  instead.
* Lexical-only fails paraphrase, which is most of what a caller says out loud
  ("how much do I pay every month" -> "EMI").
* RRF fuses by *rank*, not score, so it needs no score normalisation between
  two systems whose scales are unrelated - and it degrades gracefully to
  whichever system is available. With no API key there are no vectors and the
  same code path returns pure BM25 results.

Two hard filters run before ranking, in SQL rather than in a prompt:
  1. `retrieval_allowed = 1` - PII records can never be retrieved.
  2. language - the corpus holds the same policy in six scripts; an English
     call must not be answered from the Tamil edition.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..common.config import settings
from ..common.logging import log
from .index import connect, load_embeddings

RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "to", "in", "on",
    "for", "and", "or", "if", "it", "this", "that", "with", "as", "at", "by", "from",
    "can", "do", "does", "i", "you", "we", "my", "your", "me", "what", "how", "much",
    "many", "please", "tell", "about", "there", "any", "have", "has",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


@dataclass
class Hit:
    record_id: str
    title: str
    content: str
    category: str
    source: str
    source_url: str
    source_locator: str
    version: str
    product: str
    language: str
    market: str
    score: float
    lexical_rank: int | None = None
    dense_rank: int | None = None
    debug: dict = field(default_factory=dict)

    def citation(self) -> str:
        bits = [self.title]
        if self.source_locator:
            bits.append(self.source_locator)
        if self.source_url:
            bits.append(self.source_url)
        elif self.source:
            bits.append(self.source)
        return " | ".join(b for b in bits if b)

    def as_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "title": self.title,
            "category": self.category,
            "version": self.version,
            "score": round(self.score, 4),
            "citation": self.citation(),
            "content": self.content,
        }


class Retriever:
    """Loads once, answers many. Cheap enough to hold in the call server."""

    def __init__(self, *, language: str = "en", market: str = "IN") -> None:
        self.language = language
        self.market = market
        self.conn: sqlite3.Connection = connect()
        self.rows = [dict(r) for r in self.conn.execute(
            "SELECT * FROM records WHERE retrieval_allowed = 1"
        )]
        self.embeddings = load_embeddings()
        self._docs = [tokenize(r["title"] + " " + r["content"] + " " + (r["tags"] or "")) for r in self.rows]
        self._build_bm25()
        log("retriever.ready", records=len(self.rows),
            dense=self.embeddings is not None, languages=len({r["language"] for r in self.rows}))

    # ---------------------------------------------------------------- BM25
    def _build_bm25(self) -> None:
        self.doc_len = [len(d) for d in self._docs]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        self.tf: list[Counter] = [Counter(d) for d in self._docs]
        df: Counter = Counter()
        for d in self._docs:
            for term in set(d):
                df[term] += 1
        n = max(1, len(self._docs))
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def _bm25_scores(self, query_terms: Sequence[str], candidates: list[int]) -> list[tuple[int, float]]:
        out: list[tuple[int, float]] = []
        for i in candidates:
            tf, dl = self.tf[i], self.doc_len[i]
            score = 0.0
            for term in query_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                score += idf * (f * (BM25_K1 + 1)) / (f + BM25_K1 * (1 - BM25_B + BM25_B * dl / max(1e-9, self.avgdl)))
            if score > 0:
                out.append((i, score))
        out.sort(key=lambda kv: -kv[1])
        return out

    # --------------------------------------------------------------- dense
    async def _dense_scores(self, query: str, candidates: list[int]) -> list[tuple[int, float]]:
        if self.embeddings is None or not settings.gemini_api_key:
            return []
        from ..common.llm import get_llm

        try:
            vec = (await get_llm().embed([query], task="RETRIEVAL_QUERY"))[0]
        except Exception as exc:  # noqa: BLE001 - lexical results are better than none
            log("retrieve.embed_failed", error=str(exc)[:200])
            return []
        q = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm:
            q = q / norm
        rows = np.array([self.rows[i]["embedding_row"] for i in candidates], dtype=int)
        valid = rows >= 0
        if not valid.any():
            return []
        sims = self.embeddings[rows[valid]] @ q
        pairs = [(candidates[j], float(s)) for j, s in zip(np.nonzero(valid)[0], sims)]
        pairs.sort(key=lambda kv: -kv[1])
        return pairs

    # ------------------------------------------------------------- filters
    def _candidates(self, *, language: str | None, categories: Sequence[str] | None,
                    product: str | None, market: str | None = None) -> list[int]:
        lang = language or self.language
        mkt = market or self.market
        out: list[int] = []
        for i, r in enumerate(self.rows):
            if lang and r["language"] != lang:
                continue
            if mkt and (r["market"] or "IN") != mkt:
                continue
            if categories and r["category"] not in categories:
                continue
            if product and r["product"] and r["product"] != product:
                continue
            out.append(i)
        return out

    # -------------------------------------------------------------- search
    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        language: str | None = None,
        categories: Sequence[str] | None = None,
        product: str | None = None,
        market: str | None = None,
    ) -> list[Hit]:
        top_k = top_k or settings.retrieval_top_k
        candidates = self._candidates(language=language, categories=categories,
                                      product=product, market=market)
        if not candidates:
            return []
        terms = tokenize(query)
        lexical = self._bm25_scores(terms, candidates)[:50]
        dense = (await self._dense_scores(query, candidates))[:50]

        fused: dict[int, float] = {}
        lex_rank = {idx: r for r, (idx, _) in enumerate(lexical, start=1)}
        den_rank = {idx: r for r, (idx, _) in enumerate(dense, start=1)}
        for idx, rank in lex_rank.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        for idx, rank in den_rank.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]

        # RRF decides the ORDER. It must not decide confidence.
        #
        # Rank-based fusion says nothing about absolute relevance: whatever
        # comes back first is rank 1 in both systems and normalises to ~1.0
        # even when the corpus contains no answer at all. Measured: the query
        # "what is the current price of gold in Dubai" scored 1.0 against a
        # lending knowledge base and sailed through the confidence gate, which
        # is exactly the failure that produces a confident hallucination.
        #
        # So `score` is an ABSOLUTE relevance estimate, computed per hit:
        #   * dense cosine similarity - does this record mean the same thing?
        #   * term coverage - how much of the query does it actually contain?
        # Both are 0..1 and neither depends on what else was retrieved.
        dense_by_idx = dict(dense)
        # Coverage is IDF-weighted, and terms the corpus has never seen carry
        # the MAXIMUM weight rather than being discarded.
        #
        # The first version filtered out-of-vocabulary terms before computing
        # coverage, which inverted the signal: "current price of gold in Dubai"
        # scored 100% coverage, because "gold" and "dubai" were dropped and the
        # two survivors ("current", "price") matched a pricing record. An
        # unmatched rare word is the strongest evidence there is that this
        # corpus cannot answer the question, so it has to count against the
        # score, not disappear from it.
        max_idf = max(self.idf.values(), default=1.0)
        term_weights = {t: self.idf.get(t, max_idf) for t in terms}
        total_weight = sum(term_weights.values()) or 1.0

        hits: list[Hit] = []
        for idx, raw in ranked:
            r = self.rows[idx]
            matched = [t for t in terms if self.tf[idx].get(t)]
            coverage = sum(term_weights[t] for t in matched) / total_weight
            cosine = dense_by_idx.get(idx)
            if cosine is None:
                score = coverage
            else:
                # Cosine over Gemini embeddings sits in a narrow band (~0.5-0.9
                # for this corpus), so it is rescaled to spread that band across
                # 0..1 before being blended with coverage.
                spread = max(0.0, min(1.0, (cosine - 0.45) / 0.35))
                score = 0.6 * spread + 0.4 * coverage
            hits.append(Hit(
                record_id=r["record_id"], title=r["title"], content=r["content"],
                category=r["category"], source=r["source"], source_url=r["source_url"],
                source_locator=r["source_locator"], version=r["version"], product=r["product"],
                language=r["language"], market=r["market"], score=score,
                lexical_rank=lex_rank.get(idx), dense_rank=den_rank.get(idx),
                debug={"rrf_raw": raw, "matched_terms": matched,
                       "cosine": round(cosine, 3) if cosine is not None else None,
                       "term_coverage": round(coverage, 3)},
            ))
        log("retrieve.search", query=query[:120], candidates=len(candidates),
            lexical=len(lexical), dense=len(dense), returned=len(hits),
            top_score=round(hits[0].score, 3) if hits else 0.0)
        return hits

    def is_confident(self, hits: list[Hit]) -> bool:
        """The single gate that decides whether the agent may make a factual
        claim at all. Below it, the agent must say it does not know."""
        return bool(hits) and hits[0].score >= settings.retrieval_min_score

    def get(self, record_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM records WHERE record_id = ?", (record_id,)).fetchone()
        return dict(row) if row else None


_shared: Retriever | None = None


def get_retriever() -> Retriever:
    global _shared
    if _shared is None:
        _shared = Retriever()
    return _shared


if __name__ == "__main__":
    import argparse
    import asyncio

    ap = argparse.ArgumentParser(description="Query the knowledge base")
    ap.add_argument("query", nargs="+")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    async def main() -> None:
        r = Retriever(language=args.lang)
        hits = await r.search(" ".join(args.query), top_k=args.k)
        if not hits:
            print("no results")
            return
        print("confident:", r.is_confident(hits))
        for h in hits:
            print("-" * 78)
            print(f"[{h.score:.3f}] {h.title}  ({h.category}, v{h.version})")
            print("  cite:", h.citation())
            print("  ", h.content[:260].replace("\n", " "))

    asyncio.run(main())
