"""Website extraction (Q2, "Explain website extraction").

Approach and why:

1. **Sitemap first, crawl second.** The sitemap is the site's own statement of
   what it considers a page. It avoids following pagination, tracking and
   infinite calendar links, and it is polite by construction.

2. **Headless Chromium, not plain HTTP.** Verified against the live site: the
   FAQ page returns ~2.8k characters over plain HTTP because the answers are
   injected client-side, and ~5x that after rendering. A plain-requests
   pipeline would have silently indexed the questions without the answers -
   the worst kind of KB bug, because retrieval still "works".

3. **Snapshot to disk, then parse.** Raw HTML is written to data/raw/web/ and
   committed. The KB is therefore rebuildable by a reviewer with no network,
   and re-running the crawl produces a diff instead of an untraceable change.

4. **robots.txt is parsed and obeyed**, requests are serialised with a delay,
   and the User-Agent identifies the crawler rather than impersonating a
   browser.
"""
from __future__ import annotations

import json
import re
import time
import urllib.robotparser as robotparser
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
import yaml

from ...common.config import REPO_ROOT, settings
from ...common.logging import log

USER_AGENT = "darwix-assessment-crawler/1.0 (candidate technical assessment; contact via GitHub issue)"
SOURCES_FILE = REPO_ROOT / "config" / "sources.yaml"


@dataclass
class Snapshot:
    url: str
    slug: str
    status: int
    fetched_at: str
    html_path: str
    html_chars: int
    render: str  # "chromium" | "http"
    error: str = ""


def _slug(url: str) -> str:
    p = urlparse(url)
    s = (p.path or "/").strip("/").replace("/", "__") or "index"
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", s)[:120]


def load_sources() -> dict:
    with SOURCES_FILE.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _robots(base: str) -> robotparser.RobotFileParser:
    rp = robotparser.RobotFileParser()
    rp.set_url(urljoin(base, "/robots.txt"))
    try:
        r = httpx.get(urljoin(base, "/robots.txt"), timeout=20,
                      headers={"User-Agent": USER_AGENT}, follow_redirects=True)
        rp.parse(r.text.splitlines())
    except Exception as exc:  # noqa: BLE001
        log("crawl.robots_unavailable", base=base, error=str(exc)[:120])
        rp.parse(["User-agent: *", "Disallow:"])
    return rp


def discover_urls(site: dict) -> list[str]:
    """Sitemap -> candidate URLs, filtered by the include/exclude rules."""
    base = site["base_url"]
    urls: list[str] = []
    for sm in site.get("sitemaps", ["/sitemap.xml"]):
        try:
            r = httpx.get(urljoin(base, sm), timeout=25,
                          headers={"User-Agent": USER_AGENT}, follow_redirects=True)
            if r.status_code == 200:
                urls.extend(re.findall(r"<loc>\s*(.*?)\s*</loc>", r.text))
        except Exception as exc:  # noqa: BLE001
            log("crawl.sitemap_failed", sitemap=sm, error=str(exc)[:120])
    urls.extend(urljoin(base, p) for p in site.get("extra_paths", []))

    include = [re.compile(p) for p in site.get("include_patterns", [".*"])]
    exclude = [re.compile(p) for p in site.get("exclude_patterns", [])]
    seen: set[str] = set()
    keep: list[str] = []
    for u in urls:
        u = u.split("#")[0].rstrip("/") or u
        if u in seen:
            continue  # the live sitemap really does repeat URLs
        seen.add(u)
        if not any(rx.search(u) for rx in include):
            continue
        if any(rx.search(u) for rx in exclude):
            continue
        keep.append(u)
    limit = site.get("max_pages")
    return keep[:limit] if limit else keep


def links_from_snapshots(site_key: str, site: dict) -> tuple[list[str], list[str]]:
    """Second-pass discovery from pages we already hold.

    A sitemap is a claim about the site, not a description of it. On the live
    source, ten sitemap URLs are dead while several linked pages are missing
    from the sitemap entirely, so both sources are used and unioned.

    Returns (html_urls, pdf_urls).
    """
    from bs4 import BeautifulSoup

    base = site["base_url"]
    host = urlparse(base).netloc
    site_dir = settings.raw_dir / "web" / site_key
    include = [re.compile(p) for p in site.get("include_patterns", [".*"])]
    exclude = [re.compile(p) for p in site.get("exclude_patterns", [])]

    html_urls: set[str] = set()
    pdf_urls: set[str] = set()
    for f in sorted(site_dir.glob("*.html")):
        soup = BeautifulSoup(f.read_text(encoding="utf-8", errors="replace"), "html.parser")
        for a in soup.find_all("a", href=True):
            href = urljoin(base, a["href"].strip()).split("#")[0]
            if urlparse(href).netloc != host:
                continue
            if href.lower().endswith(".pdf"):
                pdf_urls.add(href)
                continue
            href = href.rstrip("/") or href
            if not any(rx.search(href) for rx in include):
                continue
            if any(rx.search(href) for rx in exclude):
                continue
            html_urls.add(href)
    return sorted(html_urls), sorted(pdf_urls)


def download_pdfs(site_key: str | None = None, *, limit: int = 12) -> list[dict]:
    """Fetch the PDFs the site links to.

    The regulatory pages (fair practices code, ombudsman scheme, schedule of
    charges) are HTML shells whose actual content is a linked PDF. Ignoring
    them would mean the bot cannot answer a single grievance question, so the
    PDFs are pulled and parsed like any other source.
    """
    cfg = load_sources()
    sites = cfg["sites"]
    if site_key:
        sites = {site_key: sites[site_key]}
    out: list[dict] = []
    for key, site in sites.items():
        html_urls, pdf_urls = links_from_snapshots(key, site)
        # Some policy documents are served by a download endpoint with no .pdf
        # extension (/download-fair-practice/13). They are still PDFs - the
        # %PDF magic-byte check below is what actually decides.
        endpoints = [u for u in html_urls
                     if re.search(r"/(download|view)-(fair-practice|ombudsman-file|document)", u)]
        wanted = [u for u in pdf_urls if _pdf_is_relevant(u)][:limit] + endpoints[:6]
        dest = settings.raw_dir / "web_pdf" / key
        dest.mkdir(parents=True, exist_ok=True)
        rp = _robots(site["base_url"])
        for url in wanted:
            stem = re.sub(r"[^a-zA-Z0-9_.-]", "-", urlparse(url).path.strip("/").replace("/", "_"))[:110]
            path = dest / (stem if stem.lower().endswith(".pdf") else stem + ".pdf")
            if not path.exists():
                if not rp.can_fetch(USER_AGENT, url):
                    log("crawl.robots_disallow_pdf", url=url)
                    continue
                try:
                    r = httpx.get(url, timeout=60, headers={"User-Agent": USER_AGENT},
                                  follow_redirects=True)
                    if r.status_code != 200 or not r.content[:4] == b"%PDF":
                        log("crawl.pdf_bad", url=url, status=r.status_code)
                        continue
                    path.write_bytes(r.content)
                    time.sleep(1.0)
                except Exception as exc:  # noqa: BLE001
                    log("crawl.pdf_failed", url=url, error=str(exc)[:140])
                    continue
            out.append({"url": url, "path": str(path.relative_to(REPO_ROOT)),
                        "bytes": path.stat().st_size})
            log("crawl.pdf", url=url, bytes=path.stat().st_size)
    manifest = settings.raw_dir / "web_pdf" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


PDF_KEYWORDS = ("fair-practice", "fair_practice", "ombudsman", "grievance", "charges",
                "pricing", "policy", "kyc", "code-of-conduct", "interest-rate", "english")


def _pdf_is_relevant(url: str) -> bool:
    low = url.lower()
    # regional-language duplicates of the same policy add no retrievable value
    for lang in ("hindi", "punjabi", "urdu", "telugu", "tamil", "marathi", "kannada",
                 "bengali", "gujarati", "odia", "assamese", "malayalam"):
        if lang in low:
            return False
    return any(k in low for k in PDF_KEYWORDS)


def crawl(site_key: str | None = None, *, force: bool = False, urls: list[str] | None = None) -> list[Snapshot]:
    """Fetch and snapshot every allowed page. Idempotent unless `force`."""
    from playwright.sync_api import sync_playwright

    cfg = load_sources()
    sites = cfg["sites"]
    if site_key:
        sites = {site_key: sites[site_key]}

    out_root = settings.raw_dir / "web"
    out_root.mkdir(parents=True, exist_ok=True)
    snapshots: list[Snapshot] = []

    for key, site in sites.items():
        base = site["base_url"]
        rp = _robots(base)
        site_urls = list(urls) if urls else discover_urls(site)
        log("crawl.start", site=key, candidate_urls=len(site_urls))
        delay = float(site.get("delay_seconds", 1.5))
        site_dir = out_root / key
        site_dir.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context(user_agent=USER_AGENT, viewport={"width": 1280, "height": 1800})
            page = ctx.new_page()
            # Images/fonts/media are irrelevant to text extraction and are the
            # bulk of the bytes. Blocking them makes the crawl ~4x faster and
            # much lighter on the source site.
            page.route(
                "**/*",
                lambda route: route.abort()
                if route.request.resource_type in {"image", "media", "font"}
                else route.continue_(),
            )
            for url in site_urls:
                slug = _slug(url)
                html_path = site_dir / (slug + ".html")
                if html_path.exists() and not force:
                    snapshots.append(
                        Snapshot(url, slug, 200, "cached", str(html_path.relative_to(REPO_ROOT)),
                                 html_path.stat().st_size, "cached")
                    )
                    continue
                if not rp.can_fetch(USER_AGENT, url):
                    log("crawl.robots_disallow", url=url)
                    continue
                try:
                    resp = page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                    # Accordion FAQs and tabbed policy tables hydrate late.
                    page.wait_for_timeout(1800)
                    _expand_accordions(page)
                    html = page.content()
                    html_path.write_text(html, encoding="utf-8")
                    snap = Snapshot(
                        url=url,
                        slug=slug,
                        status=resp.status if resp else 0,
                        fetched_at=datetime.now(timezone.utc).isoformat(),
                        html_path=str(html_path.relative_to(REPO_ROOT)),
                        html_chars=len(html),
                        render="chromium",
                    )
                except Exception as exc:  # noqa: BLE001 - a dead page must not kill the crawl
                    snap = Snapshot(url, slug, 0, datetime.now(timezone.utc).isoformat(),
                                    "", 0, "chromium", error=str(exc)[:200])
                    log("crawl.page_failed", url=url, error=str(exc)[:160])
                snapshots.append(snap)
                log("crawl.page", url=url, status=snap.status, chars=snap.html_chars)
                time.sleep(delay)
            ctx.close()
            browser.close()

    manifest = out_root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for s in snapshots:
            fh.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    log("crawl.done", pages=len(snapshots), manifest=str(manifest))
    return snapshots


def _expand_accordions(page) -> None:
    """FAQ answers live inside collapsed accordions. Clicking them is the
    difference between indexing questions and indexing answers."""
    selectors = [
        ".accordion-button", ".accordion-header button", "[data-bs-toggle='collapse']",
        ".faq-question", ".panel-title a", "summary",
    ]
    for sel in selectors:
        try:
            handles = page.query_selector_all(sel)
        except Exception:  # noqa: BLE001
            continue
        for h in handles[:60]:
            try:
                h.click(timeout=800)
                page.wait_for_timeout(60)
            except Exception:  # noqa: BLE001
                continue


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Snapshot source websites into data/raw/web/")
    ap.add_argument("--site", default=None)
    ap.add_argument("--force", action="store_true", help="re-fetch even if a snapshot exists")
    ap.add_argument("--discover", action="store_true",
                    help="second pass: crawl same-domain links found in existing snapshots")
    ap.add_argument("--pdfs", action="store_true", help="download policy PDFs linked from snapshots")
    args = ap.parse_args()

    if args.discover:
        cfg = load_sources()
        sites = {args.site: cfg["sites"][args.site]} if args.site else cfg["sites"]
        for key, site in sites.items():
            html_urls, pdf_urls = links_from_snapshots(key, site)
            print("discovered", len(html_urls), "html and", len(pdf_urls), "pdf links for", key)
            crawl(key, urls=html_urls)
    elif args.pdfs:
        rows = download_pdfs(args.site)
        print("downloaded", len(rows), "pdfs")
    else:
        crawl(args.site, force=args.force)
