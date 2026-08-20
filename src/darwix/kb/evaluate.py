"""Q2 retrieval evaluation -> evaluation/retrieval_tests.md.

Expectations are declared in config/retrieval_tests.yaml before the run, so a
verdict is computed, not awarded after seeing the output. Two of the twelve
queries are negative tests where the correct result is that retrieval *fails*
the confidence gate: one asks for another customer's PII, one asks something the
corpus has no answer to. A retrieval report without negative tests only proves
the system can find things, not that it knows when it has not.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..common.config import EVAL_DIR, REPO_ROOT, settings
from ..common.logging import log
from .retrieve import Hit, Retriever

TESTS_PATH = REPO_ROOT / "config" / "retrieval_tests.yaml"


def _verdict(spec: dict, hits: list[Hit], confident: bool) -> tuple[str, str]:
    """Returns (verdict, explanation)."""
    expect_confident = spec.get("expect_confident", True)
    must = [m.lower() for m in (spec.get("must_contain_any") or [])]
    forbid = [f.lower() for f in (spec.get("forbid_any") or [])]
    expect_category = spec.get("expect_category")

    blob_all = " ".join((h.title + " " + h.content).lower() for h in hits)
    for bad in forbid:
        if bad in blob_all:
            return "incorrect", ("Leaked forbidden content (" + bad
                                 + ") - PII exclusion failed.")
    forbid_category = spec.get("forbid_category")
    if forbid_category and any(h.category == forbid_category for h in hits):
        return "incorrect", ("Returned a record from the forbidden category '"
                             + forbid_category + "'.")

    # --- negative tests -------------------------------------------------
    if not expect_confident:
        if confident:
            top = hits[0] if hits else None
            return "incorrect", (
                "Retrieval cleared the confidence gate (top score "
                + (str(round(top.score, 3)) if top else "n/a")
                + ") when it should have refused, so the agent would have "
                  "attempted an answer.")
        return "correct", (
            "Retrieval stayed below the confidence threshold as required, so the "
            "agent says it does not have the information and offers a human.")

    # --- positive tests -------------------------------------------------
    if not hits:
        return "incorrect", "No records retrieved at all."
    if not confident:
        return "incorrect", (
            "Top score " + str(round(hits[0].score, 3)) + " is below the "
            + str(settings.retrieval_min_score) + " threshold, so the agent would "
            "refuse to answer a question the knowledge base does contain.")

    top = hits[0]
    top_blob = (top.title + " " + top.content).lower()
    in_top = not must or any(m in top_blob for m in must)
    in_any = not must or any(m in (h.title + " " + h.content).lower()
                             for h in hits for m in must)

    category_ok = (expect_category is None) or (top.category == expect_category)

    if in_top and category_ok:
        # Only claim the category was checked when the query actually declared
        # one. Eight of the queries do not, and saying "evidence and category"
        # for those overstates what was verified - in a file that is submitted
        # as evidence, that is the kind of small untruth worth not writing.
        checked = "evidence and category" if expect_category else "evidence"
        return "correct", (
            "Top result carries the expected " + checked + ", cited to "
            + (top.source_url or top.source) + ".")
    if in_top and not category_ok:
        return "partially correct", (
            "Top result contains the right evidence but is classified as '"
            + top.category + "' rather than '" + str(expect_category)
            + "'. Usable for the agent, but the category filter would have "
              "missed it if the search were narrowed by category.")
    if in_any:
        rank = next(i + 1 for i, h in enumerate(hits)
                    if any(m in (h.title + " " + h.content).lower() for m in must))
        return "partially correct", (
            "The expected evidence is present at rank " + str(rank)
            + ", not rank 1. The agent would still see it (top-k is "
            + str(settings.retrieval_top_k) + ") but the strongest record is not "
              "the most relevant one.")
    return "incorrect", (
        "None of the retrieved records contain the expected evidence ("
        + ", ".join(must) + ").")


async def run(top_k: int | None = None) -> dict:
    spec = yaml.safe_load(TESTS_PATH.read_text(encoding="utf-8"))
    retriever = Retriever()
    top_k = top_k or settings.retrieval_top_k
    results = []

    for q in spec["queries"]:
        hits = await retriever.search(q["question"], top_k=top_k)
        confident = retriever.is_confident(hits)
        verdict, explanation = _verdict(q, hits, confident)
        results.append({
            "id": q["id"],
            "question": q["question"],
            "kind": q["kind"],
            "note": (q.get("note") or "").strip(),
            "confident": confident,
            "top_score": round(hits[0].score, 3) if hits else 0.0,
            "verdict": verdict,
            "explanation": explanation,
            "hits": [
                {
                    "rank": i + 1,
                    "record_id": h.record_id,
                    "title": h.title,
                    "category": h.category,
                    "version": h.version,
                    "score": round(h.score, 3),
                    "citation": h.citation(),
                    "lexical_rank": h.lexical_rank,
                    "dense_rank": h.dense_rank,
                    "excerpt": " ".join(h.content.split())[:420],
                }
                for i, h in enumerate(hits)
            ],
        })
        log("kb.eval", id=q["id"], verdict=verdict, confident=confident,
            top=round(hits[0].score, 3) if hits else 0.0)
        await asyncio.sleep(0.3)  # free-tier embedding rate limit

    counts = {v: sum(1 for r in results if r["verdict"] == v)
              for v in ("correct", "partially correct", "incorrect")}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records_searchable": len(retriever.rows),
        "dense_enabled": retriever.embeddings is not None,
        "threshold": settings.retrieval_min_score,
        "top_k": top_k,
        "counts": counts,
        "results": results,
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "retrieval_tests.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(payload)
    return payload


def write_report(payload: dict) -> Path:
    lines: list[str] = []
    add = lines.append
    stats = json.loads((settings.kb_dir / "build_stats.json").read_text(encoding="utf-8")) \
        if (settings.kb_dir / "build_stats.json").exists() else {}

    add("# Q2 - Retrieval tests")
    add("")
    add("Generated by `python -m darwix.kb.evaluate` on "
        + payload["generated_at"][:19].replace("T", " ") + " UTC.")
    add("")
    add("| | |")
    add("|---|---|")
    add("| Searchable records | " + str(payload["records_searchable"]) + " |")
    if stats:
        add("| Total records (incl. PII-blocked) | " + str(stats.get("records", "?")) + " |")
        add("| Records blocked from retrieval (PII) | "
            + str(stats.get("retrieval_blocked_records", "?")) + " |")
    add("| Dense retrieval | " + ("enabled" if payload["dense_enabled"] else "lexical only") + " |")
    add("| Confidence threshold | " + str(payload["threshold"]) + " |")
    add("| top-k | " + str(payload["top_k"]) + " |")
    add("")
    c = payload["counts"]
    add("**Result: " + str(c["correct"]) + " correct, " + str(c["partially correct"])
        + " partially correct, " + str(c["incorrect"]) + " incorrect, out of "
        + str(len(payload["results"])) + ".**")
    add("")
    add("Expectations for every query are declared in `config/retrieval_tests.yaml` "
        "*before* the run, so these verdicts are computed rather than assigned "
        "after seeing the output. Two queries are negative tests where the correct "
        "outcome is that retrieval **fails** the confidence gate.")
    add("")
    add("## Summary")
    add("")
    add("| ID | Question | Top record | Score | Confident | Verdict |")
    add("|---|---|---|---:|:---:|---|")
    for r in payload["results"]:
        top = r["hits"][0] if r["hits"] else None
        add("| " + r["id"] + " | " + r["question"][:58]
            + " | " + (top["title"][:44] if top else "_none_")
            + " | " + str(r["top_score"])
            + " | " + ("yes" if r["confident"] else "no")
            + " | " + r["verdict"] + " |")
    add("")
    add("## Detail")
    add("")
    for r in payload["results"]:
        add("### " + r["id"] + " - " + r["question"])
        add("")
        if r["note"]:
            add("_Why this query:_ " + r["note"])
            add("")
        add("**Verdict: " + r["verdict"] + "** (top score " + str(r["top_score"])
            + ", confidence gate " + ("passed" if r["confident"] else "not passed") + ")")
        add("")
        add(r["explanation"])
        add("")
        if not r["hits"]:
            add("_No records retrieved._")
            add("")
            continue
        add("| Rank | Record | Category | v | Score | BM25 | Dense | Source |")
        add("|---:|---|---|---|---:|---:|---:|---|")
        for h in r["hits"]:
            add("| " + str(h["rank"]) + " | `" + h["record_id"] + "` | " + h["category"]
                + " | " + h["version"] + " | " + str(h["score"])
                + " | " + (str(h["lexical_rank"]) if h["lexical_rank"] else "-")
                + " | " + (str(h["dense_rank"]) if h["dense_rank"] else "-")
                + " | " + h["citation"][:90].replace("|", "/") + " |")
        add("")
        add("Top record excerpt:")
        add("")
        add("> " + r["hits"][0]["excerpt"])
        add("")
    path = EVAL_DIR / "retrieval_tests.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log("kb.eval_report", path=str(path))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the KB retrieval test set")
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()
    payload = asyncio.run(run(top_k=args.k))
    c = payload["counts"]
    print("correct:", c["correct"], "| partial:", c["partially correct"],
          "| incorrect:", c["incorrect"])
    for r in payload["results"]:
        print(("  " + r["verdict"][:9]).ljust(20), r["id"], r["question"][:60])
