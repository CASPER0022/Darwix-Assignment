"""Indexing: SQLite + a NumPy embedding matrix.

Why not a vector database: the corpus is a few hundred records. A brute-force
cosine over a (n x 768) float32 matrix takes well under a millisecond, which is
three orders of magnitude below the ASR latency it sits behind. Adding Chroma,
FAISS or Qdrant here would add a service to run, a build dependency on Windows,
and no measurable retrieval benefit. SQLite holds the records and their
metadata; a .npy file holds the vectors; both are rebuilt by one command.

The scaling point at which this stops being true is stated in
docs/limitations_and_production_plan.md rather than pretended away.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from ..common.config import settings
from ..common.logging import log
from .schema import KBRecord, read_records

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    record_id       TEXT PRIMARY KEY,
    title           TEXT,
    content         TEXT,
    category        TEXT,
    source          TEXT,
    version         TEXT,
    pii             INTEGER,
    doc_id          TEXT,
    source_type     TEXT,
    source_url      TEXT,
    source_locator  TEXT,
    product         TEXT,
    market          TEXT,
    language        TEXT,
    effective_date  TEXT,
    tags            TEXT,
    checksum        TEXT,
    token_estimate  INTEGER,
    quality_flag    TEXT,
    retrieval_allowed INTEGER,
    merged_from     TEXT,
    pii_types       TEXT,
    embedding_row   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_category ON records(category);
CREATE INDEX IF NOT EXISTS idx_language ON records(language);
CREATE INDEX IF NOT EXISTS idx_allowed  ON records(retrieval_allowed);
CREATE INDEX IF NOT EXISTS idx_product  ON records(product);
CREATE INDEX IF NOT EXISTS idx_market   ON records(market);
"""


def embeddings_path() -> Path:
    return settings.kb_dir / "embeddings.npy"


def connect(path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or settings.index_path))
    conn.row_factory = sqlite3.Row
    return conn


def build_index(records: list[KBRecord] | None = None, *, embeddings: np.ndarray | None = None) -> int:
    records = records if records is not None else read_records(settings.records_path)
    settings.kb_dir.mkdir(parents=True, exist_ok=True)
    conn = connect()
    # DROP rather than delete the file. Deleting fails silently-ish on Windows
    # when another process (the dev server) holds the database open, and the
    # old schema then survives a rebuild - which cost an hour when a new column
    # simply never appeared. Dropping the table works through the connection.
    conn.execute("DROP TABLE IF EXISTS records")
    conn.commit()
    conn.executescript(SCHEMA)
    rows = []
    for i, r in enumerate(records):
        rows.append((
            r.record_id, r.title, r.content, r.category, r.source, r.version, int(r.pii),
            r.doc_id, r.source_type, r.source_url, r.source_locator, r.product, r.market, r.language,
            r.effective_date, json.dumps(r.tags), r.checksum, r.token_estimate, r.quality_flag,
            int(r.retrieval_allowed), json.dumps(r.merged_from), json.dumps(r.pii_types),
            i if embeddings is not None else -1,
        ))
    conn.executemany(
        "INSERT INTO records VALUES (" + ",".join(["?"] * 23) + ")", rows
    )
    conn.commit()
    conn.close()
    if embeddings is not None:
        np.save(embeddings_path(), embeddings.astype(np.float32))
    log("index.built", records=len(rows), embedded=embeddings is not None,
        path=str(settings.index_path))
    return len(rows)


def cache_path() -> Path:
    return settings.kb_dir / "embedding_cache.jsonl"


def _load_cache() -> dict[str, list[float]]:
    path = cache_path()
    if not path.exists():
        return {}
    cache: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                cache[row["checksum"]] = row["vector"]
    return cache


async def embed_records(records: list[KBRecord], *, chunk_size: int = 25) -> np.ndarray:
    """Embed the records the bot is allowed to retrieve.

    Two design points that matter more than they look:

    * **Keyed by content checksum, cached on disk.** A rebuild after editing
      one source document re-embeds only what changed, which is what makes the
      versioning story real and keeps the build inside a free-tier quota. It
      also means a run interrupted by a 429 resumes instead of restarting.

    * **PII records still occupy a row, as a zero vector.** The matrix stays
      aligned with the SQLite table, and a blocked record is unreachable by
      similarity as well as by the SQL filter - two independent guarantees
      rather than one.
    """
    from ..common.llm import get_llm

    llm = get_llm()
    cache = _load_cache()
    targets: list[tuple[int, str, str]] = []  # (row, checksum, text)
    for i, r in enumerate(records):
        if not r.retrieval_allowed:
            continue
        targets.append((i, r.checksum, (r.title + "\n" + r.content)[:8000]))

    missing = [(i, c, t) for i, c, t in targets if c not in cache]
    log("index.embedding", total=len(records), retrievable=len(targets),
        cached=len(targets) - len(missing), to_embed=len(missing))

    # Small sequential batches with bounded concurrency inside each: the free
    # tier rate-limits per minute, and a 250-way gather trips it every time.
    cache_file = cache_path()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(missing), chunk_size):
        part = missing[start:start + chunk_size]
        vectors = await llm.embed([t for _, _, t in part], task="RETRIEVAL_DOCUMENT",
                                  concurrency=3)
        with cache_file.open("a", encoding="utf-8") as fh:
            for (_, checksum, _), vec in zip(part, vectors):
                cache[checksum] = vec
                fh.write(json.dumps({"checksum": checksum, "vector": vec}) + "\n")
        log("index.embedded_batch", done=min(start + chunk_size, len(missing)), of=len(missing))

    dim = settings.embed_dimensions
    matrix = np.zeros((len(records), dim), dtype=np.float32)
    for row, checksum, _ in targets:
        vec = cache.get(checksum)
        if not vec:
            continue
        v = np.asarray(vec, dtype=np.float32)
        n = np.linalg.norm(v)
        matrix[row] = v / n if n else v
    embedded = int((np.linalg.norm(matrix, axis=1) > 0).sum())
    log("index.embeddings_ready", embedded_rows=embedded, total_rows=len(records))
    return matrix


def load_embeddings() -> np.ndarray | None:
    path = embeddings_path()
    if not path.exists():
        return None
    return np.load(path)


if __name__ == "__main__":
    import argparse
    import asyncio

    ap = argparse.ArgumentParser(description="Build the retrieval index")
    ap.add_argument("--no-embeddings", action="store_true",
                    help="lexical-only index (no API key needed)")
    args = ap.parse_args()

    recs = read_records(settings.records_path)
    if args.no_embeddings or not settings.gemini_api_key:
        if not settings.gemini_api_key and not args.no_embeddings:
            print("GEMINI_API_KEY not set - building a lexical-only index.")
        build_index(recs)
    else:
        mat = asyncio.run(embed_records(recs))
        build_index(recs, embeddings=mat)
    print("indexed", len(recs), "records")
