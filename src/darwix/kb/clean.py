"""Cleaning: raw sources -> `Document` objects.

The hard part of web extraction is not getting text out of HTML, it is getting
*only the page's own* text out of HTML. Three mechanisms, in order:

1. **Structural**: drop script/style/nav/header/footer/form-chrome outright and
   prefer <main>/<article> when the page provides one.

2. **Statistical boilerplate removal**: every remaining block is hashed across
   the whole crawl. A block that appears on more than `boilerplate_ratio` of
   pages is site furniture (the mega-menu, the footer address, the cookie
   line), not content, and is removed. This is what makes the cleaner survive
   a site redesign - no per-site CSS selectors to maintain.

3. **Quality gates**: a page that returns HTTP 200 with a "page not found"
   body, or that yields almost no text after cleaning, is *flagged*, not
   dropped. The live sitemap really does contain such a URL, and a KB that
   silently indexes it is worse than one that records the problem.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF
import yaml
from bs4 import BeautifulSoup

from ..common.config import REPO_ROOT, settings
from ..common.logging import log
from .schema import Document

BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "dt", "dd",
              "blockquote", "button", "summary"]
STRIP_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "svg", "iframe",
              "select", "option", "aside"]

# Accordion FAQ *questions* are rendered as <button> / <summary>. Stripping
# those tags wholesale - the obvious thing to do - silently produced a KB of
# answers with no questions, which retrieves badly and reads worse. So buttons
# are kept when they carry a real sentence and dropped when they are UI chrome.
UI_CHROME = {"apply now", "submit", "read more", "know more", "close", "menu", "next",
             "previous", "search", "login", "sign in", "download", "view all", "back",
             "explore", "contact us", "get started", "calculate", "reset", "send otp"}
MIN_BUTTON_CHARS = 15

SOFT_404_MARKERS = ("page not found", "404 not found", "sorry, the page", "does not exist")
MIN_CONTENT_CHARS = 400
BOILERPLATE_RATIO = 0.35


def _norm_ws(text: str) -> str:
    text = text.replace(" ", " ").replace("​", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def _block_key(text: str) -> str:
    return hashlib.md5(_norm_ws(text).lower().encode("utf-8")).hexdigest()


def _html_blocks(html: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (page title, [(tag, text)]) for the page's own content."""
    soup = BeautifulSoup(html, "html.parser")
    title = _norm_ws(soup.title.get_text()) if soup.title else ""
    for tag in soup(STRIP_TAGS):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    blocks: list[tuple[str, str]] = []
    for el in root.find_all(BLOCK_TAGS):
        txt = _norm_ws(el.get_text(" ", strip=True))
        if len(txt) < 3:
            continue
        if el.name in ("button", "summary"):
            if len(txt) < MIN_BUTTON_CHARS or txt.lower().strip(" .:->") in UI_CHROME:
                continue
            # a question is a heading for the answer that follows it
            blocks.append(("h4", txt))
            continue
        blocks.append((el.name, txt))
    return title, blocks


def _boilerplate_keys(pages: dict[str, list[tuple[str, str]]]) -> set[str]:
    """Blocks common to many pages are furniture, not content."""
    counts: Counter[str] = Counter()
    for blocks in pages.values():
        for _, txt in set((t, x) for t, x in blocks):
            counts[_block_key(txt)] += 1
    threshold = max(2, int(len(pages) * BOILERPLATE_RATIO))
    return {k for k, c in counts.items() if c >= threshold}


def _assemble(blocks: list[tuple[str, str]], drop: set[str]) -> str:
    lines: list[str] = []
    seen_local: set[str] = set()
    for tag, txt in blocks:
        key = _block_key(txt)
        if key in drop or key in seen_local:
            continue
        seen_local.add(key)
        if tag.startswith("h"):
            lines.append("")
            lines.append("## " + txt)
        elif tag in ("li", "dt", "dd"):
            lines.append("- " + txt)
        elif tag in ("td", "th"):
            lines.append("| " + txt)
        else:
            lines.append(txt)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def clean_website() -> list[Document]:
    manifest_path = settings.raw_dir / "web" / "manifest.jsonl"
    if not manifest_path.exists():
        log("clean.no_web_manifest", path=str(manifest_path))
        return []

    entries = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    parsed: dict[str, tuple[dict, str, list[tuple[str, str]]]] = {}
    for e in entries:
        if not e.get("html_path"):
            continue
        path = REPO_ROOT / e["html_path"]
        if not path.exists():
            continue
        try:
            html = path.read_text(encoding="utf-8", errors="replace")
            title, blocks = _html_blocks(html)
        except Exception as exc:  # noqa: BLE001
            log("clean.parse_failed", url=e["url"], error=str(exc)[:160])
            continue
        parsed[e["url"]] = (e, title, blocks)

    drop = _boilerplate_keys({u: b for u, (_, _, b) in parsed.items()})
    log("clean.boilerplate", pages=len(parsed), boilerplate_blocks=len(drop))

    docs: list[Document] = []
    for url, (entry, title, blocks) in parsed.items():
        text = _assemble(blocks, drop)
        flag = ""
        note = ""
        low = (title + " " + text[:600]).lower()
        if any(m in low for m in SOFT_404_MARKERS):
            flag = "soft_404"
            note = ("URL is listed in the site's own sitemap but returns a "
                    "'page not found' body with HTTP 200. Flagged, not indexed.")
        elif len(text) < MIN_CONTENT_CHARS:
            flag = "thin_content"
            note = "Less than " + str(MIN_CONTENT_CHARS) + " characters survived cleaning."
        docs.append(
            Document(
                doc_id="web_" + entry["slug"],
                title=title or entry["slug"].replace("__", " / "),
                text=text,
                source_type="website",
                source_url=url,
                fetched_at=entry.get("fetched_at", ""),
                quality_flag=flag,
                notes=note,
                raw_chars=entry.get("html_chars", 0),
                clean_chars=len(text),
            )
        )
    flagged = [d.doc_id for d in docs if d.quality_flag]
    log("clean.website_done", documents=len(docs), flagged=len(flagged), flagged_ids=flagged[:10])
    return docs


# --------------------------------------------------------------------------
# internal documents
# --------------------------------------------------------------------------
def clean_website_pdfs() -> list[Document]:
    """Parse the policy PDFs the site links to (fair practices code, schedule
    of charges). These are the authoritative source for every compliance and
    charges question the bot will be asked."""
    manifest = settings.raw_dir / "web_pdf" / "manifest.jsonl"
    if not manifest.exists():
        return []
    docs: list[Document] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = REPO_ROOT / row["path"]
        if not path.exists():
            continue
        doc_id = "webpdf_" + path.stem[:60]
        doc = _clean_pdf(path, doc_id, path.stem.replace("_", " ").replace("-", " ").title())
        # The download endpoints have opaque names (/download-fair-practice/13),
        # so the document's own first heading is a better title than the URL.
        doc.title = _pdf_title(doc.text) or doc.title
        doc.source_type = "website_pdf"
        doc.source_url = row["url"]
        doc.fetched_at = datetime.now(timezone.utc).isoformat()
        docs.append(doc)
    log("clean.website_pdfs_done", documents=len(docs))
    return docs


def _pdf_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip(" #|-")
        if line.lower().startswith("page "):
            continue
        if 8 <= len(line) <= 90:
            return line
    return ""


def _clean_pdf(path: Path, doc_id: str, label: str) -> Document:
    parts: list[str] = []
    empty_pages: list[int] = []
    with fitz.open(str(path)) as pdf:
        total_pages = pdf.page_count
        for i, page in enumerate(pdf, start=1):
            txt = page.get_text("text")
            if not txt.strip():
                # No text layer: a scanned or image-only page. Record which page
                # and keep the rest - discarding a 15-page policy because page 4
                # is an image would lose every answer it contains.
                empty_pages.append(i)
                continue
            parts.append("## Page " + str(i) + "\n" + _norm_ws_multiline(txt))
    text = "\n\n".join(parts)

    flag = ""
    note = ""
    if empty_pages and not parts:
        flag = "extraction_failed"
        note = ("No text layer on any of " + str(total_pages) + " pages "
                "(scanned document - OCR would be required).")
    elif empty_pages:
        flag = "partial_extraction"
        note = ("No text layer on page(s) " + ", ".join(str(i) for i in empty_pages)
                + " of " + str(total_pages) + "; the remaining pages are indexed.")
    if "[text truncated in source scan" in text:
        flag = flag or "truncated_source"
        note = note or ("Truncation marker present in the source document; the affected "
                        "line is flagged rather than indexed as fact.")
    return Document(
        doc_id=doc_id,
        title=label,
        text=text,
        source_type="internal_pdf",
        source_path=str(path.relative_to(REPO_ROOT)),
        quality_flag=flag,
        notes=note,
        raw_chars=path.stat().st_size,
        clean_chars=len(text),
    )


def _norm_ws_multiline(text: str) -> str:
    lines = [_norm_ws(l) for l in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(l for l in lines if l is not None)).strip()


def _clean_csv(path: Path, doc_id: str, label: str, source_type: str) -> Document:
    import csv as _csv

    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(_csv.reader(fh))
    if not rows:
        return Document(doc_id=doc_id, title=label, text="", source_type=source_type,  # type: ignore[arg-type]
                        source_path=str(path.relative_to(REPO_ROOT)), quality_flag="extraction_failed")
    header, body = rows[0], rows[1:]
    lines = []
    for r in body:
        cells = ["" + header[i] + ": " + v for i, v in enumerate(r) if i < len(header) and v.strip()]
        lines.append("| " + " | ".join(cells))
    text = "## " + label + "\n" + "\n".join(lines)
    return Document(
        doc_id=doc_id,
        title=label,
        text=text,
        source_type=source_type,  # type: ignore[arg-type]
        source_path=str(path.relative_to(REPO_ROOT)),
        raw_chars=path.stat().st_size,
        clean_chars=len(text),
    )


def _clean_markdown(path: Path, doc_id: str, label: str) -> Document:
    text = _norm_ws_multiline(path.read_text(encoding="utf-8"))
    return Document(
        doc_id=doc_id,
        title=label,
        text=text,
        source_type="internal_handbook",
        source_path=str(path.relative_to(REPO_ROOT)),
        raw_chars=path.stat().st_size,
        clean_chars=len(text),
    )


def clean_internal() -> list[Document]:
    cfg = yaml.safe_load((REPO_ROOT / "config" / "sources.yaml").read_text(encoding="utf-8"))
    docs: list[Document] = []
    for spec in cfg.get("internal_documents", []):
        path = settings.raw_dir / spec["file"]
        if not path.exists():
            log("clean.internal_missing", file=spec["file"])
            continue
        kind = spec["kind"]
        if kind == "pdf":
            doc = _clean_pdf(path, spec["id"], spec["label"])
        elif kind == "csv":
            doc = _clean_csv(path, spec["id"], spec["label"], "internal_table")
        elif kind == "csv_pii":
            doc = _clean_csv(path, spec["id"], spec["label"], "crm_export")
        else:
            doc = _clean_markdown(path, spec["id"], spec["label"])
        doc.market = spec.get("market", "IN")
        docs.append(doc)
    log("clean.internal_done", documents=len(docs))
    return docs


def clean_all() -> list[Document]:
    docs = clean_website() + clean_website_pdfs() + clean_internal()
    out = settings.interim_dir / "documents.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d.model_dump(), ensure_ascii=False) + "\n")
    log("clean.done", documents=len(docs), out=str(out),
        total_clean_chars=sum(d.clean_chars for d in docs))
    return docs


if __name__ == "__main__":
    for d in clean_all():
        print(f"{d.doc_id:38s} {d.clean_chars:7d}  {d.quality_flag or 'ok':16s} {d.title[:60]}")
