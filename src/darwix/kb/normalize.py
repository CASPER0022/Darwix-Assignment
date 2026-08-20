"""Standardisation of headings, dates, terminology, amounts and languages.

Why this stage exists at all: retrieval is lexical as well as semantic, and a
customer who asks "what is my monthly instalment" should reach a record that
says "EMI". Standardising the *canonical* term while preserving the original
phrasing as a searchable alias is the difference between a KB that answers and
one that nearly answers.

Everything here is deterministic and reversible-by-inspection - no model is
involved - so a reviewer can see exactly what changed and why.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime

import yaml

from ..common.config import REPO_ROOT

TAXONOMY_PATH = REPO_ROOT / "config" / "taxonomy.yaml"

# --- language detection by script ------------------------------------------
# The source site publishes the same Fair Practices Code in ten languages. Those
# are not duplicates in the textual sense (no shared n-grams) so the near-dup
# detector cannot catch them; script detection can, in three lines.
_SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),  # Devanagari (Hindi/Marathi)
    "bn": (0x0980, 0x09FF),
    "pa": (0x0A00, 0x0A7F),
    "gu": (0x0A80, 0x0AFF),
    "ta": (0x0B80, 0x0BFF),
    "te": (0x0C00, 0x0C7F),
    "kn": (0x0C80, 0x0CFF),
    "ur": (0x0600, 0x06FF),
}


def detect_language(text: str) -> str:
    sample = text[:4000]
    if not sample:
        return "en"
    counts: dict[str, int] = {}
    for ch in sample:
        cp = ord(ch)
        for lang, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
                break
    if counts:
        lang, n = max(counts.items(), key=lambda kv: kv[1])
        if n / max(1, len(sample)) > 0.05:
            return lang
    return "en"


# --- dates ------------------------------------------------------------------
_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}
_MONTHS.update({m[:3].lower(): i for m, i in list(_MONTHS.items())})

_DATE_PATTERNS = [
    (re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b"), "dmy"),
    (re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b"), "ymd"),
    (re.compile(r"\b([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b"), "mdy"),
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9}),?\s+(\d{4})\b"), "dmy_name"),
]


def _iso(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def normalize_dates(text: str) -> tuple[str, int]:
    """Rewrite every recognised date to ISO-8601. Returns (text, count).

    The corpus really does mix `01-04-2026`, `2025/10/15`, `April 1, 2026` and
    `Jun 30th, 2026` - sometimes in the same document.
    """
    changes = 0

    def repl(match: re.Match, kind: str) -> str:
        nonlocal changes
        g = match.groups()
        iso = None
        if kind == "dmy":
            iso = _iso(int(g[2]), int(g[1]), int(g[0]))
        elif kind == "ymd":
            iso = _iso(int(g[0]), int(g[1]), int(g[2]))
        elif kind == "mdy":
            mon = _MONTHS.get(g[0].lower())
            iso = _iso(int(g[2]), mon, int(g[1])) if mon else None
        elif kind == "dmy_name":
            mon = _MONTHS.get(g[1].lower())
            iso = _iso(int(g[2]), mon, int(g[0])) if mon else None
        if not iso:
            return match.group(0)
        changes += 1
        return iso

    for rx, kind in _DATE_PATTERNS:
        text = rx.sub(lambda m, k=kind: repl(m, k), text)
    return text, changes


# --- amounts ----------------------------------------------------------------
_LAKH = re.compile(r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|lac|crore|cr)\b", re.I)
_PLAIN = re.compile(r"(?:rs\.?|inr|₹)\s*([\d][\d,]{2,})", re.I)


def normalize_amounts(text: str) -> tuple[str, int]:
    """Append a machine-comparable value next to Indian-format amounts.

    `Rs. 50 lakh` and `Rs. 50,00,000` are the same number to a human and two
    unrelated strings to a retriever. Both get an explicit `(INR 5000000)`
    annotation so lexical search and the qualification engine agree.
    """
    changes = 0

    def lakh_repl(m: re.Match) -> str:
        nonlocal changes
        try:
            n = float(m.group(1).replace(",", ""))
        except ValueError:
            return m.group(0)
        mult = 10_000_000 if m.group(2).lower().startswith(("cr", "crore")) else 100_000
        changes += 1
        return m.group(0) + " (INR " + str(int(n * mult)) + ")"

    def plain_repl(m: re.Match) -> str:
        nonlocal changes
        digits = m.group(1).replace(",", "")
        if not digits.isdigit():
            return m.group(0)
        changes += 1
        return m.group(0) + " (INR " + digits + ")"

    text = _LAKH.sub(lakh_repl, text)
    text = _PLAIN.sub(plain_repl, text)
    return text, changes


# --- terminology ------------------------------------------------------------
def load_taxonomy() -> dict:
    return yaml.safe_load(TAXONOMY_PATH.read_text(encoding="utf-8"))


def canonical_terms(text: str, taxonomy: dict | None = None) -> tuple[str, list[str]]:
    """Return (annotated_text, aliases_seen).

    Note what this deliberately does *not* do: it does not rewrite the source
    wording. Rewriting a policy document to say "EMI" where it said "monthly
    payment" would corrupt a quotable source. Instead the canonical term is
    appended once per record as a searchable alias line, and the original text
    stays verbatim so the citation remains truthful.
    """
    tax = taxonomy or load_taxonomy()
    seen: list[str] = []
    low = text.lower()
    for canon, spec in tax.get("terminology", {}).items():
        for alias in spec.get("aliases", []):
            if re.search(r"\b" + re.escape(alias.lower()) + r"\b", low):
                if canon not in seen:
                    seen.append(canon)
                break
    if seen:
        text = text + "\n\n[canonical terms: " + ", ".join(seen) + "]"
    return text, seen


# --- headings ---------------------------------------------------------------
def normalize_heading(heading: str) -> str:
    h = unicodedata.normalize("NFKC", heading).strip(" #-|:•\t")
    h = re.sub(r"\s+", " ", h)
    h = re.sub(r"^(faq'?s?|frequently asked questions)\s*[:-]?\s*", "", h, flags=re.I)
    if h.isupper() and len(h) > 4:
        h = h.title()
    return h.strip()


def normalize_document_text(text: str) -> tuple[str, dict[str, int]]:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"[–—]", "-", text)
    # the source encodes its own brand name with a mojibake byte on several
    # pages ("U GRO Capital <?>"); strip replacement characters rather than
    # indexing them
    text = text.replace("�", "")
    text, n_dates = normalize_dates(text)
    text, n_amounts = normalize_amounts(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, {"dates_normalized": n_dates, "amounts_annotated": n_amounts}
