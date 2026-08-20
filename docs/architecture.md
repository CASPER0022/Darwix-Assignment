# Architecture

Four deliverables, one system. The reason they share a codebase rather than
sitting in four folders is that Q1 and Q3 are the *same* agent with different
locale packs, and Q4 reuses Q1's audio layer to listen to a call instead of
conducting one. Duplicating any of that would have meant three VADs to tune and
three places for a compliance rule to be wrong.

## System

```mermaid
flowchart TB
    subgraph sources["Sources (Q2)"]
        WEB["Public website<br/>33 pages, Playwright"]
        PDFS["Linked policy PDFs<br/>Fair Practices Code, charges"]
        INT["Internal documents (synthetic)<br/>credit policy, rules matrix,<br/>objection handbook, form, leads"]
    end

    subgraph kb["Knowledge base pipeline"]
        CLEAN["clean<br/>boilerplate removal, quality flags"]
        NORM["normalise<br/>dates, amounts, terminology, language"]
        CHUNK["chunk<br/>heading-aware, atomic rules rows"]
        PII["PII scan<br/>Verhoeff / PAN validation"]
        DEDUP["dedupe<br/>hash + SimHash + Jaccard"]
        INDEX[("SQLite records<br/>+ embeddings.npy")]
    end

    subgraph retrieval["Retrieval"]
        BM25["BM25 lexical"]
        DENSE["Gemini embeddings"]
        RRF["RRF fusion (order)<br/>+ absolute relevance (gate)"]
    end

    subgraph agent["Voice agent (Q1 + Q3)"]
        VAD["VAD / endpointing<br/>barge-in"]
        ASR1["Whisper ASR"]
        UND["understand<br/>intent + slots"]
        POL["dialog policy<br/>state machine, disclosures"]
        QUAL["qualification engine<br/>rules from CSV"]
        GRND["grounding<br/>threshold, citation, numeric check"]
        TTS["TTS<br/>locale voice"]
    end

    subgraph live["Live insights (Q4)"]
        SPLIT["channel split<br/>agent L / customer R"]
        ASR2["streaming ASR"]
        RULES["rules signals<br/>microseconds"]
        LLMS["LLM signals<br/>debounced, concurrent"]
        NUDGE["nudge engine<br/>threshold, cooldown, priority, TTL"]
    end

    WEB --> CLEAN
    PDFS --> CLEAN
    INT --> CLEAN
    CLEAN --> NORM --> CHUNK --> PII --> DEDUP --> INDEX
    INDEX --> BM25 & DENSE --> RRF

    VAD --> ASR1 --> UND --> POL
    POL --> QUAL
    POL --> GRND
    RRF --> GRND
    GRND --> TTS
    QUAL --> TTS

    TTS -.recording.-> SPLIT
    SPLIT --> ASR2 --> RULES --> NUDGE
    ASR2 --> LLMS --> NUDGE
    NUDGE --> DASH["dashboard / WebSocket / CLI"]
```

## The one decision that shapes everything

**The model is never the source of truth.**

| Concern | Owned by | Not by |
|---|---|---|
| What the customer meant | LLM | - |
| What the customer said (values) | LLM extracts, code normalises | LLM arithmetic |
| Whether they qualify | `qualification.py` + rules CSV | the prompt |
| What must be disclosed, and when | `dialog_policy.py` | the prompt |
| Whether a factual claim is allowed | `grounding.py` + retrieval threshold | the model's confidence |
| When to escalate to a human | `escalation.py` | the model's judgement |
| Which nudge reaches the agent | `nudge_engine.py` | the signal's own confidence |

Every one of these was a deliberate move *out* of the prompt, because each is a
place where a fluent model will confidently do the wrong thing under
conversational pressure.

## Call flow (Q1 / Q3)

```mermaid
sequenceDiagram
    participant C as Customer
    participant V as VAD
    participant A as ASR (Whisper)
    participant S as Session
    participant K as Knowledge base
    participant T as TTS

    C->>V: speech
    V->>V: 300ms pre-roll + 1100ms hangover
    V->>A: utterance
    A->>S: transcript (~300ms)
    S->>S: understand: intent + slots (~600ms)
    alt escalation trigger
        S->>T: acknowledge + hand off
    else factual question or objection
        S->>K: retrieve
        K-->>S: hits + scores
        alt below threshold
            S->>T: "I don't have that" + offer human
        else confident
            S->>S: generate + verify citations & numbers
            S->>T: grounded answer
        end
    else answering a slot
        S->>S: normalise value, evaluate rules
        S->>T: next question / outcome
    end
    T-->>C: audio (barge-in interruptible)
```

## Model routing, and why

Measured, not assumed — see `evaluation/model_selection.md`.

| Job | Model | Why |
|---|---|---|
| Dialogue (Q1/Q3) | Groq `openai/gpt-oss-120b`, `reasoning_effort=low` | 923 ms median, and it answered in the customer's own language and register in every Taglish and Bahasa test |
| Q4 signal extraction | Groq `openai/gpt-oss-20b` | ~810 ms for short structured JSON; the job is classification, not conversation |
| KB classification, ASR benchmark, customer simulation | Gemini 3.6 Flash | 16-38 s per turn on this free tier, so it is used only where latency does not matter |
| ASR | Groq `whisper-large-v3-turbo` | Handles Taglish and Bahasa code-switching in one pass instead of forcing a language decision per turn |
| Embeddings | Gemini `gemini-embedding-001` (768d) | Free tier, asymmetric document/query task types |
| TTS | `edge-tts` | The only free option with native `fil-PH`, `id-ID`, and regional `jv-ID` / `su-ID` voices |

## Data flow and storage

```
data/
├── raw/                  # crawler snapshots + synthetic internal docs (regenerable)
│   ├── web/              #   33 HTML pages + manifest
│   ├── web_pdf/          #   7 policy PDFs fetched from the site
│   ├── internal/         #   authored: credit policy, rules CSV, handbook, form, leads
│   └── markets/          #   authored: PH life insurance + ID multifinance KBs (Q3)
├── interim/
│   └── documents.jsonl   # cleaned documents, committed - the KB rebuilds offline from here
├── kb/
│   ├── records.jsonl     # the knowledge base itself
│   ├── index.sqlite      # records + metadata, with the PII filter in SQL
│   ├── embeddings.npy    # (n, 768) float32; PII rows are zero vectors
│   ├── embedding_cache.jsonl  # keyed by content checksum -> only changed records re-embed
│   ├── versions/         # content-addressed snapshots per build
│   └── build_stats.json  # what was ingested, merged, flagged, skipped
├── transcripts/          # per-call JSON: turns, slots, decisions, latency
├── recordings/           # stereo calls (agent L / customer R), mp3 committed
└── crm/leads.jsonl       # the business action: one lead row per completed call
```

## Why there is no vector database

The corpus is a few hundred records. A brute-force cosine over an `(n, 768)`
float32 matrix is well under a millisecond — three orders of magnitude below the
ASR call it sits behind. Adding Chroma, FAISS or Qdrant would add a service to
run, a build dependency on Windows, and no measurable retrieval benefit.

The point at which that stops being true, and what replaces it, is in
`limitations_and_production_plan.md` rather than pretended away.
