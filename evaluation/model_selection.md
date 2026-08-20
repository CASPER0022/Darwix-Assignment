# Model selection — measured, not assumed

Both providers are free tier (Google AI Studio, Groq Cloud). The interesting
part is that the obvious choice was wrong twice, and the measurements are what
caught it.

## 1. The intended dialogue model was unusable

The plan was Gemini for dialogue: large context, strong multilingual behaviour,
generous free tier. Then it was measured on real turns.

| Model | English turn | Taglish turn | Bahasa turn | Notes |
|---|---:|---:|---:|---|
| `gemini-3.6-flash` (default) | 37,990 ms | 15,980 ms | 28,260 ms | one turn returned HTTP 503 "high demand"; answers truncated |
| `gemini-3.6-flash` `thinkingLevel: low` | 6,162 ms | 26,013 ms | 36,783 ms | slower, not faster |
| `gemini-3.6-flash` `thinkingLevel: minimal` | ~2,060 ms | — | ~2,300 ms | usable, still far too slow for voice |

Gemini 3.6 Flash is a **thinking model**. It spent 133 reasoning tokens
answering "say ok", and those tokens are charged against `maxOutputTokens`,
which silently truncated replies until the budget was raised. On a phone call,
a 16-second pause is a dropped call.

**Decision: Gemini is used only off the latency path** — knowledge-base
classification, the ASR benchmark, and generating simulated customers. There it
is genuinely useful and its 2-3 s turn time costs nothing.

## 2. The fast model was returning empty strings

Groq's `gpt-oss` models emit reasoning on a separate channel. Without
`reasoning_effort`, reasoning consumes the token budget and the visible
`content` comes back **empty** — a failure that looks like a working call with a
silent agent.

| Model | `reasoning_effort` | English | Taglish | Bahasa | Median |
|---|---|---:|---:|---:|---:|
| `gpt-oss-20b` | (unset) | *empty* | 1,720 ms | 1,028 ms | 1,097 ms |
| `gpt-oss-20b` | `low` | 811 ms | 1,820 ms | 670 ms | **811 ms** |
| `gpt-oss-120b` | (unset) | 1,327 ms | 1,226 ms | 1,333 ms | 1,327 ms |
| `gpt-oss-120b` | `low` | 923 ms | 1,024 ms | 923 ms | **923 ms** |
| `qwen3.6-27b` | n/a | leaks `<think>` into content | — | — | rejected |

Speed alone said `gpt-oss-20b`. Quality said otherwise:

> **Taglish prompt:** *"Ate, sorry po pero wala akong pambayad ngayon sa premium ko, next week na lang po ba?"*
>
> - `gpt-oss-20b` @ low → *"Sure, next week works—just let us know when you're ready to pay."* ← **answered in English**
> - `gpt-oss-120b` @ low → *"Oo, puwede po i-postpone hanggang susunod na linggo."* ← correct language, correct `po` register

For Q3, answering a Taglish speaker in English is a scored failure, not a style
preference. The 112 ms difference is not worth it.

**Decision:**

| Job | Model | Rationale |
|---|---|---|
| Dialogue (Q1, Q3) | `openai/gpt-oss-120b` @ `reasoning_effort=low` | 923 ms median, correct language and register every time |
| Q4 signal extraction | `openai/gpt-oss-20b` @ `reasoning_effort=low` | Classification into fixed JSON — no register to get wrong |
| Offline (KB classify, ASR benchmark, simulated customers) | `gemini-3.6-flash` @ `thinkingLevel: minimal` | Latency irrelevant, quality good, keeps Groq quota for the agent |

## 3. Free-tier limits are a design constraint, not a footnote

Both limits were hit during the build and both changed the architecture:

- **Groq: 8,000 tokens/minute** on `gpt-oss-120b`. A batch of 20 KB chunks sent
  for classification returned HTTP 413 outright. This is also why the Q4 LLM
  layer analyses a **rolling window** rather than the whole transcript — the
  constraint forced the design that was correct anyway.
- **Gemini: daily request quota.** It ran out mid-run while generating simulated
  customers, and every customer turn silently degraded to *"Sorry, could you
  repeat that?"* — a test suite quietly lying about what it had tested. Two
  fixes: the customer simulator moved to Groq, and embeddings became
  **checksum-keyed and cached on disk**, so an interrupted run resumes and a
  rebuild only re-embeds what actually changed.

## Reproducing this

```bash
python -m darwix.common.llm            # imports cleanly, no side effects
python -m darwix.kb.evaluate           # retrieval quality
python -m darwix.realtime.evaluate     # latency + false positives, real time
```

Model ids are environment variables (`.env.example`), so a retired model id is a
one-line change rather than a code change.
