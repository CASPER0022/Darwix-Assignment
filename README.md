# Darwix AI — AI Engineer assessment

Four deliverables, one system: a knowledge-grounded voice agent, a production
knowledge-base pipeline, native-language agents for two more markets, and live
nudges generated during a call.

They share a codebase rather than sitting in four folders because **Q1 and Q3
are the same agent with different locale packs**, and **Q4 reuses Q1's audio
layer** to listen to a call instead of conducting one. Splitting them would have
meant three VADs to tune and three places for a compliance rule to be wrong.

| | Deliverable | Documentation | Evidence |
|---|---|---|---|
| **Q1** | Knowledge-grounded voice agent (NBFC lead qualification, `en-IN`) | [`docs/q1_voice_agent.md`](docs/q1_voice_agent.md) | 5 recorded calls, per-turn grounding audit |
| **Q2** | Production knowledge base from a real public website + internal docs | [`docs/q2_knowledge_base.md`](docs/q2_knowledge_base.md) | [`retrieval_tests.md`](evaluation/retrieval_tests.md) |
| **Q3** | Native-language agents: Philippines (Taglish), Indonesia (+ regional accents) | [`docs/q3_multilingual.md`](docs/q3_multilingual.md) | 4 recorded calls, [`asr_benchmark.md`](evaluation/asr_benchmark.md) |
| **Q4** | Real-time nudges delivered **during** a live call | [`docs/q4_realtime.md`](docs/q4_realtime.md) | [`latency_report.md`](evaluation/latency_report.md), [`false_positive_analysis.md`](evaluation/false_positive_analysis.md) |

**Start here if you are evaluating this submission:**
[`SUBMISSION_OVERVIEW.md`](SUBMISSION_OVERVIEW.md) — the executive submission overview and employer evaluation guide covering all four questions, benchmarks, and architectural decisions.

[`IMPLEMENTATION.md`](IMPLEMENTATION.md) — how the whole system works, in the
order it runs, with the reasoning behind each decision.

**Cross-cutting:** [`docs/architecture.md`](docs/architecture.md) ·
[`docs/limitations_and_production_plan.md`](docs/limitations_and_production_plan.md) ·
[`evaluation/model_selection.md`](evaluation/model_selection.md)

---

## Results at a glance

Every number below is produced by a command in this repo and written to a file
in `evaluation/`. Nothing is hand-entered.

The model and ASR calls are non-deterministic, so re-running a command moves its numbers a little - Q4 end-to-end p50 has been observed between 610 and 860 ms across runs on the same audio. The files in `evaluation/` are the source of truth for whichever run produced them; a scenario changing **verdict** is a regression, a figure shifting by a hundred milliseconds is not.

| Measurement | Result | Source |
|---|---|---|
| Retrieval test set | 14 correct, 1 partial, 0 incorrect of 15 | `make eval` |
| Records indexed | 352 searchable of 391 built, from 48 source documents | `make kb` |
| Q1 agent turn latency | 967 ms p50 to a decision (ASR 335 ms + understand 632 ms); the caller also waits for TTS | simulation transcripts |
| Q3 ASR word error rate | 8.8% en-IN · 9.5% Taglish · 11.9% id-ID · 25.0% Javanese | `make asr-bench` |
| Q4 end-to-end nudge latency | **854 ms p50**, 1,228 ms p95, real-time replay | `make q4-eval` |
| Q4 scenario coverage | 4 of 4 pass; 1 unanticipated nudge in 6 (16.7% upper-bound FP) | `make q4-eval` |
| Unit tests | 207 passing, no API key required | `make test` |

## Quick start

```bash
cp .env.example .env      # add GEMINI_API_KEY and GROQ_API_KEY (both free tier)
make install              # editable install + playwright chromium
make test                 # 207 unit tests, no network, no keys
make serve                # http://127.0.0.1:8000
```

`make serve` hosts three interfaces:

| Path | What |
|---|---|
| `/webcall` | Talk to the agent in the browser — live transcript, and the citation behind every factual claim |
| `/kb` | Search the knowledge base directly, with scores and the confidence gate visible |
| `/dashboard` | The Q4 agent-assist screen: nudges appearing as a call plays |

The knowledge base is committed, so `/webcall` and `/kb` work immediately
without a crawl or a rebuild.

## Everything else

```bash
make setup       # seed the synthetic internal source documents
make crawl       # re-crawl the public website + linked policy PDFs (needs network)
make kb          # clean -> build -> index
make kb-offline  # same, with no API key at all (lexical retrieval only)
make eval        # retrieval test set  -> evaluation/retrieval_tests.md

make sim         # every scripted test call (records audio + transcripts)
make sim-q1      # the five Q1 scenarios
make sim-q3      # the four Q3 scenarios
make asr-bench   # ASR provider comparison -> evaluation/asr_benchmark.md

make q4          # render the Q4 scenario recordings
make live FILE=data/recordings/q4_compliance_gap.wav
make q4-eval     # all Q4 scenarios in real time -> latency + false-positive reports

make check       # tests + KB rebuild + retrieval eval
```

## How to read the evidence

The calls in `data/recordings/` and `data/transcripts/` were **not scripted into
existence**. A synthetic caller persona is generated by an LLM, spoken by TTS in
a native voice for its locale, and fed into the agent **as audio** — so VAD,
Whisper, retrieval and grounding all run exactly as they would for a human
caller. The transcripts record what the system actually heard, not what the
script said, which is why the mistakes in them are real ones.

Each persona declares its expected outcome **before** the run
(`src/darwix/simulator/personas/*.yaml`), and
`data/transcripts/simulation_results.json` records pass/fail against that
declaration. The same discipline applies to Q2 (`config/retrieval_tests.yaml`)
and Q4 (`data/recordings/<id>.truth.json`): expectations are written first and
scored after, rather than assigned once the output is visible.

## What is synthetic, and what is not

| Real | Synthetic |
|---|---|
| 33 pages + 7 linked policy PDFs from a listed Indian NBFC's public site | Credit policy, qualification rules, objection handbook, application form, CRM leads |
| ASR, TTS, LLM and embedding calls to live provider APIs | The PH and ID market knowledge bases |
| Latency, WER, retrieval scores — all measured on real calls | The customers on every recorded call |

Every synthetic file opens with a `SYNTHETIC DOCUMENT` banner, is tagged
`source_type: internal_*`, and says so in the citation the agent speaks. Nothing
in the knowledge base attributes invented policy to the real company.

## Design decisions worth knowing before reading the code

- **The model decides what the customer meant; code decides everything else.**
  Disclosures, eligibility arithmetic, whether a fact may be stated at all, and
  escalation are all in code with tests. A model under conversational pressure
  skips disclosures, and "usually right" is not a lending control.
- **No answer without retrieval, and no retrieval below a confidence floor.**
  Five gates, including a numeric guard that rejects any number not present in
  the retrieved text — the failure that turns "14.5% to 16.0%" into "around 15%".
- **Rules are data.** Changing a turnover cut-off is a CSV edit plus a rebuild,
  with a version bump and an audit trail — not a prompt edit.
- **A second market is a locale pack, not a translation.** Different sector,
  flow, objections and politeness system per market; the agent code is identical.
- **Q4 is real-time by construction.** The replay source sleeps between chunks to
  match wall-clock time, so the pipeline cannot see audio that would not yet
  exist on a live call.
- **No vector database.** 250 records and a NumPy matrix; the reasoning, and the
  point at which that stops being true, are in
  [`architecture.md`](docs/architecture.md#why-there-is-no-vector-database).

## Limitations

The honest version is a document, not a paragraph:
[`docs/limitations_and_production_plan.md`](docs/limitations_and_production_plan.md)
covers what breaks at 10x concurrency, what happens on noisy audio, why
`edge-tts` cannot ship, where the PII redaction stops, and what I would change
first. The short version: this is a browser web call rather than a PSTN number,
the callers are synthetic, and no native speaker has reviewed the Q3 packs yet.

## Requirements

Python 3.11+. Two free-tier API keys (Google AI Studio, Groq) — no card
required. `make test` runs with neither. Secrets live in `.env`, which is
gitignored; `.env.example` documents every variable.
