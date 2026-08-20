"""Knowledge-base record schema.

The assessment specifies the fields it wants to see
(record_id / title / content / category / source / version / pii). Those are
kept verbatim. Everything else exists because a voice agent needs it at
runtime:

* `source_url` + `source_locator`  -> the citation the bot speaks/logs. Without
  a locator, "cite your source" degrades into "name the website".
* `retrieval_allowed`              -> PII-bearing records stay in the KB for
  audit but are never retrievable by the bot.
* `quality_flag`                   -> extraction problems survive into the KB
  instead of being silently indexed as fact.
* `checksum` + `version`           -> change detection across rebuilds, which
  is what makes versioning real rather than a field name.
* `merged_from`                    -> dedup provenance; a merged record can
  still be traced to every page it came from.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

from pydantic import BaseModel, Field

Category = Literal[
    "product",
    "eligibility",
    "policy",
    "pricing",
    "process",
    "faq",
    "objection",
    "partnership",
    "compliance",
    "company",
    "lead_record",
    "unclassified",
]

SourceType = Literal["website", "website_pdf", "internal_pdf", "internal_table",
                     "internal_handbook", "crm_export"]


class Document(BaseModel):
    """A cleaned source document, before chunking."""

    doc_id: str
    title: str
    text: str
    source_type: SourceType
    source_url: str = ""
    source_path: str = ""
    fetched_at: str = ""
    market: str = "IN"
    lang: str = "en"
    quality_flag: str = ""  # "", "soft_404", "thin_content", "extraction_failed", ...
    notes: str = ""
    raw_chars: int = 0
    clean_chars: int = 0


class KBRecord(BaseModel):
    """One retrievable, citable unit of knowledge."""

    # --- fields named by the assessment ---------------------------------
    record_id: str
    title: str
    content: str
    category: Category = "unclassified"
    source: str = ""  # human-readable "where this came from"
    version: str = "1.0"
    pii: bool = False

    # --- operational fields ---------------------------------------------
    doc_id: str = ""
    source_type: SourceType = "website"
    source_url: str = ""
    source_locator: str = ""  # heading / page / row that pins the claim
    product: str = ""
    # Which market this record serves. The Philippines bot must never answer
    # from the Indian lending corpus, and vice versa - retrieval filters on it.
    market: str = "IN"
    language: str = "en"
    effective_date: str = Field(default_factory=lambda: date.today().isoformat())
    tags: list[str] = Field(default_factory=list)
    checksum: str = ""
    token_estimate: int = 0
    quality_flag: str = ""
    retrieval_allowed: bool = True
    merged_from: list[str] = Field(default_factory=list)
    pii_types: list[str] = Field(default_factory=list)

    def compute_checksum(self) -> str:
        payload = (self.title + "\n" + self.content).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def citation(self) -> str:
        """What the agent says / logs when it uses this record."""
        bits = [self.title]
        if self.source_locator:
            bits.append(self.source_locator)
        bits.append(self.source or self.source_url or self.source_type)
        return " - ".join(b for b in bits if b)

    def finalise(self) -> "KBRecord":
        self.checksum = self.compute_checksum()
        self.token_estimate = max(1, len(self.content) // 4)
        if self.pii:
            self.retrieval_allowed = False
        return self


def write_records(path: Path, records: Iterable[KBRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_records(path: Path) -> list[KBRecord]:
    if not path.exists():
        return []
    out: list[KBRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(KBRecord(**json.loads(line)))
    return out


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n
