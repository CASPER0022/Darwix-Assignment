# Q4 — Live insights and nudges from call audio

**Run one call:** `make live FILE=data/recordings/q4_compliance_gap.wav`
**Dashboard:** `make serve` → <http://127.0.0.1:8000/dashboard>
**Full evaluation:** `make q4-eval` → [`latency_report.md`](../evaluation/latency_report.md) ·
[`false_positive_analysis.md`](../evaluation/false_positive_analysis.md)

---

## "During the call" is enforced, not claimed

The assessment is explicit that analysing a completed recording after upload
does not qualify. So the replay source **sleeps between chunks to match
wall-clock time**: a 57-second call takes 57 seconds to analyse, and the
pipeline cannot see audio that would not yet exist on a live call.

Measured on the compliance scenario: **57.5 s wall clock for a 57.1 s
recording**, with the first nudge delivered at **42 s** — while there were still
15 seconds of call left to act on it.

## Pipeline

```
audio chunk (200ms)
   → channel split (agent L / customer R)
   → per-channel VAD with 300ms pre-roll
   → Whisper ASR on each completed utterance
   → ├── rules layer      (synchronous, microseconds)
     └── LLM layer        (debounced 6s, concurrent, rolling window)
   → nudge engine  (threshold → dedupe → cooldown → priority → TTL)
   → WebSocket / dashboard / CLI
```

**Speaker separation comes from the channel layout, not a diarisation model.**
Recordings are written agent-left / customer-right, which is how contact-centre
recorders work. That also means only the channel containing speech is sent to
ASR — roughly one ASR call per utterance rather than two per window.

**300 ms of pre-roll** was added after transcripts came back as *"is this?"* for
*"Who is this?"*: the VAD needs ~250 ms of energy before it declares speech, so
by the time capture starts the first word is gone. A truncated first word also
breaks keyword signals, which is how the bug was noticed.

## Two signal layers, because they answer different questions

| | Rules layer | LLM layer |
|---|---|---|
| Latency | microseconds | ~800 ms |
| Certainty | deterministic | probabilistic |
| Catches | disclosures, named cues, competitor mentions, soft refusals, risky promises, repetition | frustration *building* across turns, implied opportunities, topic drift, confusion |
| Fails how | misses paraphrase | occasionally over-reads |

Every **compliance-critical** signal is in the rules layer, so it never depends
on a model call that might time out. Compliance is modelled as a **checklist
with a deadline** — each required disclosure must appear in the agent's speech
before a point in the call, and the signal fires when the deadline passes
without it. An AI disclosure at 60 seconds is not a disclosure, it is a
confession.

The LLM layer analyses a **rolling window**, not the whole transcript. That was
partly forced by Groq's 8,000 tokens/minute free-tier cap — a constraint that
happened to demand the right design anyway.

## The nudge engine is mostly a refusal machine

An agent on a live call can read maybe one short prompt every 20–30 seconds. A
system that fires eleven is worse than one that fires none, because the agent
learns to ignore the panel. Every control exists to drop something:

| Control | Behaviour |
|---|---|
| Global threshold | Below `NUDGE_MIN_CONFIDENCE` (0.55), never shown |
| **Per-kind floor** | Low-value kinds need much more certainty: `missed_opportunity` 0.88, `topic_shift` 0.88, `confusion` 0.85 |
| Dedupe | Same kind + near-identical text fires once |
| Topic-grouped cooldown | 45 s per group, so the rules layer and LLM layer cannot both fire for one situation |
| Priority | Fixed by business consequence, not model confidence — a missed disclosure outranks a cross-sell hint even when the cross-sell is more certain |
| Screen cap | 3 active; a P1 evicts a P4 rather than stacking |
| TTL | 90 s — a nudge about something said a minute and a half ago is noise |

**One grouping decision worth calling out:** `compliance_gap` (you skipped a
disclosure) and `risky_statement` (you just promised something you can't) are in
**separate** cooldown buckets. They were in one, and the risky-promise nudge
silenced the missing-disclosure nudge — two different corrective actions, so one
must not suppress the other.

## Results

All four required scenarios pass, scored against ground truth declared **before**
the run:

| Scenario | Expected | Fired | Verdict |
|---|---|---|---|
| `q4_missed_cross_sell` | missed_cross_sell | missed_cross_sell | PASS |
| `q4_compliance_gap` | compliance_gap, risky_statement | risky_statement, compliance_gap, buying_signal | PASS |
| `q4_rising_frustration` | frustration, payment_difficulty | payment_difficulty, frustration | PASS |
| `q4_noisy_ambiguous` | *(nothing)* | *(nothing)* | PASS |

The negative test matters most: an ordinary call with small talk, hedging and
half-finished sentences, where the correct behaviour is **silence**.

### The false positive that changed the design

On the first run, the quiet call fired one nudge: the LLM read *"I'd have to
check with my accountant, I never remember these numbers"* as a **missed
opportunity** at 0.80 confidence. Not wrong exactly — just not worth
interrupting an agent for.

That produced the per-kind confidence floors, on a principle: **the bar scales
with the cost of being wrong in the other direction.** Missing a compliance gap
is expensive, so it keeps the low global floor. Missing a speculative
opportunity costs almost nothing, so it must be nearly certain to earn screen
space.

| | Before floors | After floors |
|---|---|---|
| Quiet-call nudges | 1 (false positive) | **0** |
| End-to-end p50 | 829 ms | **616 ms** |
| End-to-end p95 | 1,476 ms | **1,087 ms** |

## Latency

Measured end to end — last audio sample of an utterance arrives → nudge
delivered:

| Stage | p50 | p95 |
|---|---:|---:|
| ASR | ~286 ms | ~638 ms |
| Rules signals | <1 ms | <1 ms |
| LLM signals | ~800 ms | ~1,100 ms |
| Delivery | <1 ms | <1 ms |
| **End to end** | **616 ms** | **1,087 ms** |

ASR dominates because it is a network call; everything after it is local. The
rules layer being effectively free is exactly why compliance detection lives
there.

Honest framing: **end-to-end is bounded below by how long the speaker talks**,
because an utterance is only transcribed once it ends. A 12-second customer
monologue means the nudge cannot arrive sooner than 12 seconds after they
started. Partial-hypothesis streaming would fix that and is the first production
change (see [limitations](limitations_and_production_plan.md)).

## Suppression in numbers

| Scenario | Signals considered | Nudges shown | Suppression |
|---|---:|---:|---:|
| `q4_missed_cross_sell` | 9 | 1 | 88.9% |
| `q4_compliance_gap` | 6 | 3 | 50.0% |
| `q4_rising_frustration` | 7 | 2 | 71.4% |
| `q4_noisy_ambiguous` | 1 | 0 | 100% |

## Test recordings

Q1's agent cannot produce these: its disclosure gate and grounding check make it
compliant by construction, which is the point of Q1 and useless for testing Q4.
So the Q4 scenarios are **scripted two-party calls rendered to stereo audio** —
a deliberately imperfect human agent on the left channel, a customer on the
right. The pipeline has no idea they were scripted; it receives exactly what a
live recorder would hand it.

Scripts live in `src/darwix/simulator/scenarios/*.yaml` next to their expected
nudges, so the ground truth is written before the run rather than fitted to the
result.
