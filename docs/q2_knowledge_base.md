# Q2 — Production-ready knowledge base

**Run it:** `make kb` &nbsp;·&nbsp; **Query it:** `make serve` then <http://127.0.0.1:8000/kb>
&nbsp;·&nbsp; **Test results:** [`evaluation/retrieval_tests.md`](../evaluation/retrieval_tests.md)

---

## 1. Where the content comes from

Two classes of source, deliberately.

| Class | What | Why |
|---|---|---|
| **Real public web** | 33 pages + 7 linked policy PDFs from a listed Indian NBFC's public site | Real extraction problems: JS-rendered FAQs, mega-menu boilerplate, dead sitemap URLs, duplicated pages, policy content buried inside PDFs |
| **Synthetic internal** | Credit policy (PDF), qualification rules (CSV), objection handbook (MD), application form (PDF), CRM lead export (CSV), branch-partner FAQ (MD), and two market KBs for PH/ID | No lender publishes its credit policy, agent handbook or customer list. These exercise the half of the pipeline the public web cannot: PDF parsing, tabular rules, form fields, and **PII** |

Every synthetic file opens with a `SYNTHETIC DOCUMENT` banner, is tagged
`source_type: internal_*`, and says so in the citation the bot speaks. Nothing
in the KB attributes invented policy to the real company.

### Website extraction — what it actually took

1. **Sitemap first, then link discovery.** A sitemap is a claim about a site,
   not a description of it. On the live source, **10 sitemap URLs return HTTP
   200 with a "Page not found" body**, while several real pages are missing from
   the sitemap entirely. Both sources are unioned.
2. **Headless Chromium, not plain HTTP.** Measured on the live FAQ page: ~2.8k
   characters over plain HTTP vs ~5x that after rendering, because the answers
   are injected client-side. A requests-based pipeline would have silently
   indexed questions with no answers — the worst kind of KB bug, because
   retrieval still "works".
3. **Accordions get clicked.** FAQ answers live in collapsed panels.
4. **Linked PDFs get fetched.** The Fair Practices Code and Schedule of Charges
   pages are HTML shells around a PDF — including a download endpoint with no
   `.pdf` extension, so the fetcher decides by magic bytes, not by URL. Without
   this the bot cannot answer a single grievance question.
5. **robots.txt is parsed and obeyed**, requests are serialised at 1.5 s, and
   the User-Agent identifies the crawler rather than impersonating a browser.
6. **Snapshots are written to disk before parsing**, so the KB rebuilds with no
   network and a re-crawl produces a reviewable diff.

### A defect I did not expect, and kept

The first FAQ extraction produced a knowledge base of **answers with no
questions** — because accordion questions are `<button>` elements and I was
stripping `<button>` as UI chrome. Buttons are now kept when they carry a real
sentence and dropped when they are `Apply Now`. It is in
[`clean.py`](../src/darwix/kb/clean.py) with the reasoning attached, because it
is the kind of bug that makes a retrieval system quietly mediocre.

## 2. Cleaning

| Mechanism | What it does |
|---|---|
| Structural | Drops `script`/`style`/`nav`/`header`/`footer`, prefers `<main>`/`<article>` |
| **Statistical boilerplate removal** | Every block is hashed across the whole crawl; blocks appearing on >35% of pages are site furniture. **38 blocks removed** on this crawl. No per-site CSS selectors to maintain |
| Quality gates | Problems are **flagged, not dropped** |

Flags raised on this corpus:

| Flag | Count | Meaning |
|---|---:|---|
| `soft_404` | 10 | HTTP 200 with a "page not found" body — skipped, listed in `data/kb/skipped_sources.jsonl` |
| `thin_content` | 8 | Under 400 chars survived cleaning — kept but flagged |
| `partial_extraction` | 1 | A PDF page with no text layer (a scan); the other pages are indexed and the missing page number is recorded |
| `truncated_source` | 1 | A truncation marker in the source document itself |

## 3. Normalisation

Deterministic, no model, so every change is inspectable:

- **Dates** → ISO-8601. The corpus genuinely mixes `01-04-2026`, `2025/10/15`,
  `April 1, 2026` and `Jun 30th, 2026`, sometimes in one document.
  **106 normalised.**
- **Amounts** → machine-comparable annotation. `Rs. 50 lakh` and `Rs. 50,00,000`
  are the same number to a human and two unrelated strings to a retriever; both
  get `(INR 5000000)` appended. **176 annotated.**
- **Terminology** → canonical aliases appended, *source text left verbatim*.
  Rewriting a policy document to say "EMI" where it said "monthly payment" would
  corrupt a quotable source, so the canonical term is added as a searchable
  alias line instead.
- **Language** → detected by script range. The source publishes the same Fair
  Practices Code in Hindi, Punjabi, Urdu, Telugu and Tamil; those share no
  tokens with the English edition, so no hash can pair them — only the language
  tag can.

## 4. Deduplication — three problems, three mechanisms

| Problem | Mechanism | Found here |
|---|---|---|
| Exact duplicates | Content hash | `/loan-against-property` and `/secured-loan-against-property` are byte-identical after cleaning |
| Near duplicates | SimHash (long docs) / **shingle Jaccard (short docs)** | A paragraph shared between the credit policy and the partner FAQ |
| Translations | Language tag | The same policy in six scripts |

**A measured correction:** SimHash alone missed a real near-duplicate. On a
short paragraph a single added word moved **8 of 64 bits** — past the threshold —
because there are too few shingles for the signature to be stable. Short
documents now use exact shingle Jaccard, which is affordable at this corpus size
and has no such blind spot. (`tests/test_kb.py::test_near_duplicates_merge`)

Nothing is deleted. The survivor keeps `merged_from`, so a merged record is
still traceable to every page it came from — which is what "source tracking" has
to mean if a citation is going to be defensible.

## 5. PII — identified, and actually protected

Identifying PII is the easy half. The protection is the design:

- **PII records stay in the KB** (they are real business data) but carry
  `retrieval_allowed = false`, enforced **in the SQL query**, not in a prompt.
- **They are stored redacted.** The original values are never copied into the
  KB, so a leak of the KB file is not a leak of customer data.
- **They hold zero embedding vectors**, so they are unreachable by similarity as
  well as by the filter — two independent guarantees rather than one.
- **Validation, not just pattern matching.** A 12-digit number is only an
  Aadhaar if it passes the **Verhoeff checksum**; `ABCDE1234F` is only a PAN if
  the 4th character is a valid holder-type code. Without this, every invoice and
  account number is a false positive.

**39 records carry PII; 39 are blocked from retrieval.** Verified as a live
query in the retrieval test set (RT11): asking for a specific customer's mobile
and PAN returns the privacy policy's definition of personal data — and none of
their actual details.

Residual risk, stated: person names are only redacted where the source labels
them (`full_name:`). A general NER pass would catch more and would also redact
"Fair Practices Code" and every city in the corpus.

## 6. Schema

The assessment's fields are kept verbatim; the rest exist because a voice agent
needs them at runtime.

```json
{
  "record_id": "webpdf_download-fair-practice_13__007",
  "title": "Fair Practices Code - Grievance Redressal Mechanism",
  "content": "...",
  "category": "compliance",
  "source": "published PDF: https://www.ugrocapital.com/download-fair-practice/13",
  "version": "1.1",
  "pii": false,

  "source_url": "https://www.ugrocapital.com/download-fair-practice/13",
  "source_locator": "page 7",
  "market": "IN",
  "language": "en",
  "product": "",
  "effective_date": "2026-08-18",
  "checksum": "a3f9c1d2e4b57890",
  "token_estimate": 214,
  "quality_flag": "",
  "retrieval_allowed": true,
  "merged_from": [],
  "pii_types": [],
  "tags": ["grievance", "kyc"]
}
```

`source_locator` is what makes a citation checkable: "page 7", not "the website".

## 7. Chunking

Chunk boundaries decide what the bot can say, so they follow the document's own
structure:

- **Split on headings first** — including FAQ questions promoted from accordion
  buttons, so a question and its answer stay together and the heading becomes
  the citation locator.
- **Never split a rules-table row.** Half a rule is worse than no rule: it
  retrieves confidently and answers wrongly.
- **250–400 tokens, ~60 token overlap.**
- **Runts merge only within their own section.** Merging a fragment under a
  neighbouring heading would make its citation point at the wrong place.

## 8. Indexing, retrieval and the confidence gate

Hybrid: **BM25 + Gemini embeddings, fused with Reciprocal Rank Fusion**.
Dense-only fails "what is the bounce charge" (embeddings return semantically
adjacent fee records); lexical-only fails "how much do I pay every month".

**Filters run in SQL before ranking:** `retrieval_allowed = 1`, language, and
market — the Philippines bot physically cannot answer from the Indian lending
corpus.

### The most important bug I found

RRF ranks well and **says nothing about absolute relevance**. Whatever comes
back first is rank 1 in both systems and normalises to ~1.0 — so *"what is the
current price of gold in Dubai"* scored **1.000** against a lending knowledge
base and sailed through the confidence gate. That is precisely the path to a
confident hallucination.

RRF now decides **order only**. Confidence is a separate, absolute measure:

```
score = 0.6 · rescaled_cosine + 0.4 · idf_weighted_term_coverage
```

with a second correction: query terms the corpus has **never seen** count at
maximum weight *against* the match. The first version filtered them out, which
inverted the signal — "gold" and "dubai" were discarded and the surviving
"current"/"price" matched a pricing record for 100% coverage.

| Query | Before | After |
|---|---:|---:|
| "current price of gold in Dubai" | 1.000 (answered) | **0.169 (refused)** |
| "what is the bounce charge" | 1.000 | 0.586 |
| "what support do branch partners get" | 1.000 | 0.878 |

Threshold `0.35` sits cleanly between the lowest true positive (0.409) and the
negative test (0.169).

## 9. Versioning

- Records are content-addressed by `checksum`; a rebuild bumps `version` only
  for records whose content actually changed (`1.0 → 1.1`).
- Every build snapshots the full record set to `data/kb/versions/<timestamp>.jsonl`.
- Embeddings are **cached by checksum**, so a rebuild re-embeds only what
  changed — which is what makes versioning cheap enough to do on every edit, and
  what let an interrupted run resume after a rate limit.

## 10. Results

```
48 documents → 391 records (IN 358 · PH 17 · ID 16)
352 embedded · 39 PII-blocked · 10 dead sources skipped and listed
106 dates normalised · 176 amounts annotated
32 exact + 5 near duplicates merged, all with provenance
```

**Retrieval tests: 14 correct, 1 partially correct, 0 incorrect** across 15
queries — including two negative tests. Full detail, with the retrieved record,
citation, scores and reasoning for every query, in
[`evaluation/retrieval_tests.md`](../evaluation/retrieval_tests.md).

The five question types the brief names are each covered by a query that
declares its expected evidence before the run:

| Type | Query | Verdict |
|---|---|---|
| Product | RT09 `Can I get a loan against my property?` · RT10 `What is the maximum I can borrow without collateral?` | correct |
| Policy | RT13 `What are the rules your collections agents have to follow?` | correct |
| Qualification | RT14 `What is the minimum bureau score you accept?` | correct |
| FAQ | RT15 `Do I need to give collateral for a working capital loan?` | correct |
| Objection | RT06 `The interest rate you are offering is too high…` | correct |

Two of them were worth adding for what they exposed rather than for the tick:

* **RT13** retrieves the right record and the record's category is wrong. The
  chunk spans sections 9–12 of the credit policy and was labelled from the
  branch-partner section at its tail, so it carries `partnership` while its
  substance is collections conduct. Retrieval is unaffected — the query is not
  category-filtered — but a narrowed search would miss it. Left visible rather
  than papered over by relaxing the assertion.
* **RT15** answers from a website FAQ accordion whose questions were stripped
  during extraction, leaving a record that is a bare list of answers. Dense
  matching still lands it; BM25 alone would not have, because the lexical
  anchors are in the discarded question text.

## 11. Connection to the voice agent

The KB is not a separate artefact. `voice/grounding.py` calls the same
`Retriever`, the same threshold decides whether the agent may speak a fact, and
the same `source_locator` becomes the citation logged against the turn. A live
demonstration of that link is the point of the Q1 web call: ask about hidden
charges and watch the citation appear next to the answer.
