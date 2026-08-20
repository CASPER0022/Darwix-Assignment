"""KB build pipeline: interim documents -> versioned, citable records.

    clean -> normalise -> chunk -> classify -> PII -> dedupe -> version -> write

Classification is rules-first and model-second on purpose. Deterministic rules
cover the corpus's obvious cases (a chunk from the objection handbook is an
objection; a chunk with "processing fee" and a percentage is pricing) and cost
nothing to re-run. The LLM is asked only about chunks the rules cannot place,
and its answers are cached to disk, so a rebuild is free and reproducible even
if the model changes underneath.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..common.config import REPO_ROOT, settings
from ..common.logging import log
from . import clean as clean_mod
from . import dedupe as dedupe_mod
from . import normalize as norm
from . import pii as pii_mod
from .chunk import chunk_document
from .schema import Document, KBRecord, write_jsonl, write_records

CACHE_PATH = settings.interim_dir / "classification_cache.json"

CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("objection", ["objection", "obj-0", "rebuttal", "not interested", "competitor"]),
    ("partnership", ["branch partner", "dsa", "direct selling agent", "lending service provider",
                     "connector", "partner payout", "referral program", "gem sahay"]),
    ("compliance", ["fair practices", "grievance", "ombudsman", "nodal officer", "redressal",
                    "privacy policy", "fraud", "kyc", "regulator", "reserve bank"]),
    ("pricing", ["processing fee", "schedule of charges", "rate of interest", "penal",
                 "bounce charge", "foreclosure charge", "reference rate", "per annum",
                 "pricing grid", "% p.a"]),
    ("eligibility", ["eligibility", "eligible", "vintage", "turnover", "bureau", "cibil",
                     "qualification", "rule_id", "criteria", "who can apply", "age"]),
    ("process", ["documents required", "document checklist", "how to apply", "application form",
                 "loan process", "turnaround", "disbursal", "section a -", "form bl-"]),
    ("faq", ["faq", "frequently asked", "?"]),
    ("product", ["loan amount", "tenure", "collateral", "product", "finance", "facility",
                 "up to inr", "loan against property", "machinery"]),
    ("company", ["about us", "branch", "head office", "contact", "customer care", "leadership"]),
]


def classify_by_rules(heading: str, text: str, doc: Document) -> str:
    if doc.source_type == "crm_export":
        return "lead_record"
    blob = (heading + "\n" + text).lower()
    if doc.doc_id == "objection_handbook":
        return "objection"
    if doc.doc_id == "qualification_rules":
        return "eligibility"
    if doc.doc_id == "partner_faq":
        return "partnership"
    if doc.doc_id == "application_form":
        return "process"
    for category, keywords in CATEGORY_RULES:
        hits = sum(1 for k in keywords if k in blob)
        if category == "faq":
            # a question mark alone is weak evidence; require it in the heading
            if "?" in heading:
                return "faq"
            continue
        if hits >= 2 or (hits == 1 and len(blob) < 400):
            return category
    return "unclassified"


def detect_product(text: str, taxonomy: dict) -> str:
    low = text.lower()
    best, best_hits = "", 0
    for key, spec in taxonomy.get("products", {}).items():
        hits = sum(1 for m in spec.get("match", []) if m in low)
        if hits > best_hits:
            best, best_hits = key, hits
    return best


async def classify_with_llm(pending: list[tuple[str, str, str]]) -> dict[str, str]:
    """pending: [(record_id, heading, text_excerpt)] -> {record_id: category}."""
    if not pending:
        return {}
    from ..common.llm import get_llm

    cache: dict[str, str] = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    todo = [p for p in pending if p[0] not in cache]
    if not todo:
        return {rid: cache[rid] for rid, _, _ in pending if rid in cache}

    if not settings.gemini_api_key:
        log("build.classify_skipped", reason="no GEMINI_API_KEY", pending=len(todo))
        return {rid: cache.get(rid, "unclassified") for rid, _, _ in pending}

    llm = get_llm()
    categories = ["product", "eligibility", "policy", "pricing", "process", "faq", "objection",
                  "partnership", "compliance", "company", "unclassified"]
    system = (
        "You classify chunks of a lending company's knowledge base. Reply with JSON only: "
        '{"assignments": [{"id": "...", "category": "..."}]}. '
        "Allowed categories: " + ", ".join(categories) + ". "
        "Choose 'unclassified' only when the chunk carries no usable business meaning."
    )
    out: dict[str, str] = {}
    # Gemini, explicitly: classification is offline work where its 2-3 s turn
    # time is irrelevant, and Groq's free tier caps at 8k tokens/minute - a
    # batch of chunk text blows straight through that with a 413.
    batch = 12
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        payload = [{"id": rid, "heading": h, "text": t[:600]} for rid, h, t in part]
        try:
            data = await llm.chat_json(
                system,
                [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                provider="gemini",
                temperature=0.0,
                max_tokens=1500,
            )
            for row in data.get("assignments", []):
                if row.get("category") in categories:
                    out[row["id"]] = row["category"]
        except Exception as exc:  # noqa: BLE001 - classification must not break the build
            log("build.classify_failed", batch=i, error=str(exc)[:200])
    cache.update(out)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return {rid: cache.get(rid, "unclassified") for rid, _, _ in pending}


def _load_documents() -> list[Document]:
    path = settings.interim_dir / "documents.jsonl"
    if not path.exists():
        return clean_mod.clean_all()
    return [Document(**json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _source_label(doc: Document) -> str:
    if doc.source_type == "website":
        return "website: " + doc.source_url
    if doc.source_type == "website_pdf":
        return "published PDF: " + (doc.source_url or doc.source_path)
    return "internal document (synthetic, authored for this assessment): " + doc.title


def build(*, use_llm: bool = True) -> dict:
    taxonomy = norm.load_taxonomy()
    documents = _load_documents()
    log("build.documents_loaded", count=len(documents))

    records: list[KBRecord] = []
    norm_stats = {"dates_normalized": 0, "amounts_annotated": 0}
    skipped_docs: list[dict] = []

    for doc in documents:
        if doc.quality_flag in ("soft_404", "extraction_failed") or not doc.text.strip():
            skipped_docs.append({"doc_id": doc.doc_id, "url": doc.source_url,
                                 "reason": doc.quality_flag or "empty",
                                 "note": doc.notes})
            continue

        text, stats = norm.normalize_document_text(doc.text)
        for k in norm_stats:
            norm_stats[k] += stats.get(k, 0)
        language = norm.detect_language(text)

        for ch in chunk_document(text, doc_title=doc.title):
            heading = norm.normalize_heading(ch.heading or doc.title)
            # PDF sections are headed "Page 7", which makes a useless title and
            # a useless citation. Keep the page as the *locator* (it is exactly
            # what someone needs to verify the claim) and take the title from
            # the document plus the chunk's own first line.
            page_match = re.fullmatch(r"Page (\d+)", heading)
            if page_match:
                locator = "page " + page_match.group(1)
                first_line = next((l.strip(" #|-") for l in ch.text.splitlines()
                                   if len(l.strip(" #|-")) > 12), "")
                heading = doc.title
                if first_line:
                    heading = doc.title + " - " + norm.normalize_heading(first_line)[:70]
            else:
                locator = ("section: " + heading) if heading else ("chunk " + str(ch.index))
            content, aliases = norm.canonical_terms(ch.text, taxonomy)
            record_id = doc.doc_id + "__" + str(ch.index).zfill(3)

            scan = pii_mod.scan_and_redact(content)
            category = classify_by_rules(heading, ch.text, doc)

            rec = KBRecord(
                record_id=record_id,
                title=(heading or doc.title)[:160],
                content=scan.redacted if scan.has_pii else content,
                category=category,  # type: ignore[arg-type]
                source=_source_label(doc),
                version="1.0",
                pii=scan.has_pii,
                doc_id=doc.doc_id,
                source_type=doc.source_type,
                source_url=doc.source_url,
                source_locator=locator,
                product=detect_product(ch.text, taxonomy),
                market=doc.market,
                language=language,
                tags=aliases,
                quality_flag=doc.quality_flag,
                pii_types=scan.kinds,
            )
            records.append(rec)

    log("build.chunked", records=len(records), skipped_documents=len(skipped_docs))

    # ---- LLM classification for the residue ------------------------------
    pending = [(r.record_id, r.title, r.content) for r in records if r.category == "unclassified"]
    if use_llm and pending:
        assignments = asyncio.run(classify_with_llm(pending))
        for rec in records:
            if rec.category == "unclassified":
                rec.category = assignments.get(rec.record_id, "unclassified")  # type: ignore[assignment]
    log("build.classified",
        unclassified_before=len(pending),
        unclassified_after=sum(1 for r in records if r.category == "unclassified"))

    # ---- dedupe ----------------------------------------------------------
    kept, dedupe_report = dedupe_mod.deduplicate(records)
    log("build.dedupe", **dedupe_report.as_dict())

    # ---- versioning ------------------------------------------------------
    previous = {r.record_id: r for r in _previous_records()}
    changed = new = unchanged = 0
    for rec in kept:
        rec.finalise()
        old = previous.get(rec.record_id)
        if old is None:
            new += 1
        elif old.checksum != rec.checksum:
            major, minor = (old.version.split(".") + ["0"])[:2]
            rec.version = major + "." + str(int(minor) + 1)
            changed += 1
        else:
            rec.version = old.version
            unchanged += 1

    write_records(settings.records_path, kept)
    snapshot = settings.kb_dir / "versions" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".jsonl")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(settings.records_path, snapshot)

    stats = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "documents_in": len(documents),
        "documents_skipped": skipped_docs,
        "chunks_before_dedupe": len(records),
        "records": len(kept),
        "records_new": new,
        "records_changed": changed,
        "records_unchanged": unchanged,
        "pii_records": sum(1 for r in kept if r.pii),
        "retrieval_blocked_records": sum(1 for r in kept if not r.retrieval_allowed),
        "languages": _count(kept, lambda r: r.language),
        "markets": _count(kept, lambda r: r.market),
        "categories": _count(kept, lambda r: r.category),
        "source_types": _count(kept, lambda r: r.source_type),
        "normalisation": norm_stats,
        "dedupe": dedupe_report.as_dict(),
        "snapshot": str(snapshot.relative_to(REPO_ROOT)),
    }
    (settings.kb_dir / "build_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_jsonl(settings.kb_dir / "skipped_sources.jsonl", skipped_docs)
    log("build.done", records=len(kept), pii=stats["pii_records"], snapshot=stats["snapshot"])
    return stats


def _previous_records() -> list[KBRecord]:
    from .schema import read_records

    return read_records(settings.records_path)


def _count(records: list[KBRecord], key) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        k = key(r) or "none"
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the knowledge base from interim documents")
    ap.add_argument("--no-llm", action="store_true", help="rules-only classification")
    ap.add_argument("--reclean", action="store_true", help="re-run cleaning from raw snapshots")
    args = ap.parse_args()
    if args.reclean:
        clean_mod.clean_all()
    stats = build(use_llm=not args.no_llm)
    print(json.dumps({k: v for k, v in stats.items() if k != "documents_skipped"}, indent=2))
