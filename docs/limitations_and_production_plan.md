# Limitations and production plan

Written as an engineer would hand this to the person who has to run it. Nothing
here is hedging — these are the specific things that would break, the conditions
under which they break, and what I would change.

---

## 1. What this is not

| | This build | Production |
|---|---|---|
| Telephony | Browser WebRTC, 16 kHz clean audio | PSTN/SIP, 8 kHz G.711, jitter, packet loss, hold music |
| Voice platform | Self-hosted VAD → Whisper → LLM → TTS | Managed realtime (Vapi/Retell/LiveKit) or a hardened in-house stack |
| ASR | Chunked pseudo-streaming at utterance boundaries | Streaming ASR with partial hypotheses |
| TTS | `edge-tts` (Microsoft Edge read-aloud) | Licensed commercial TTS |
| VAD | Energy-based with adaptive noise floor | Neural VAD (Silero) or the carrier's |
| Concurrency | One call per process | Horizontally scaled workers behind a queue |
| Store | SQLite + a NumPy matrix | Postgres + pgvector, or a managed vector DB |

## 2. The limitations that would actually bite

### 2.1 `edge-tts` is not licensed for production
It is the only free option with native `fil-PH`, `id-ID` and regional `jv-ID` /
`su-ID` voices, which is why Q3 has real regional-accent audio at all. But it is
Microsoft Edge's read-aloud service: **no commercial licence, no SLA, no
word-level timing callbacks, and it synthesises the whole utterance before
returning** (measured: 1,487 ms p50, the single largest component of agent turn
latency).

**Change:** move to a streaming TTS with a commercial licence for these locales.
Expect perceived latency to drop by roughly a second, since audio starts playing
on the first chunk instead of the last.

### 2.2 End-to-end latency is bounded by how long the customer talks
An utterance is only transcribed once it ends. A 12-second monologue means the
nudge cannot arrive sooner than 12 seconds after they started, regardless of how
fast the pipeline is. The measured 616 ms p50 is time *after* the utterance
closes, and the report says so.

**Change:** streaming ASR with partial hypotheses, running signal detection on
interim text. Compliance and payment-difficulty signals in particular do not
need a complete sentence.

### 2.3 Energy VAD fails in exactly the conditions Q3 cares about
The noise floor is estimated by minimum statistics - a low percentile over a
rolling ~15 s window - rather than by smoothing the chunks that fall below the
current threshold. The earlier rule had two failure modes, both found by using
the interface rather than by reading it:

* In a quiet room the floor decayed toward the room tone itself. The start
  threshold fell from 0.0229 to 0.0012 in 51 s, after which a breath opened a
  turn and near-silence went to Whisper - which answers silence with a
  confident "Thank you." rather than an empty string. Every turn became
  "Thank you.".
* In a room at 0.013 RMS the opposite happened: once the tone exceeded the
  threshold every chunk counted as speech, so the branch that adapted the floor
  stopped running and the VAD stayed permanently open.

The floor is now clamped at both ends, rises at most 1.5x per update and falls
freely, and is not trusted until its window is half full - a call that opens
mid-sentence otherwise puts the estimate inside the speech and closes the first
utterance early.

An adaptive noise floor handles steady background noise. It does **not** handle
a customer on a Manila jeepney or a Jakarta roadside: non-stationary noise at
speech-band energy is read as speech. Symptoms would be spurious barge-in,
truncated utterances, and ASR calls on segments of traffic.

**Change:** Silero VAD (~1 MB, ~1 ms/frame on CPU). The `EnergyVAD` interface
is a single `push()` returning `speech_start`/`speech_end`, so this is a
drop-in replacement.

### 2.4 The synthetic caller is not a native speaker
Every Q3 call is a TTS voice reading generated text. That is a reproducible,
honest stand-in — it is **not** evidence that a Filipino or Indonesian
speaker would find the bot natural. Specifically untested:

- true spontaneous code-switching (real Taglish switches more, and mid-word)
- disfluency, self-correction, overlapping speech
- regional accents beyond the two TTS voices available
- whether `po`/`opo` placement reads as respectful or as a script

**Change before any pilot:** native-speaker review of both locale packs and
20–30 real recorded calls per market, scored by a native reviewer. Everything
else in Q3 is measurable; this part is not, and I would not claim it.

### 2.5 The market knowledge bases are synthetic
The PH and ID knowledge bases are authored for this assessment. The mechanics
they encode (31-day grace period, lapse and reinstatement, denda at 0.5%/day,
restructuring) reflect how these products genuinely work, but the numbers are
illustrative. Any real deployment replaces them with the insurer's or
multifinance company's actual documents — the pipeline does not change, only
the sources registry.

### 2.6 PII: names are only redacted where labelled
Regex and checksums catch phone numbers, email, PAN (holder-type validated) and
Aadhaar (Verhoeff validated). Person names are redacted only where the source
labels them (`full_name:`). A general NER pass would catch more — and would also
redact "Fair Practices Code" and every city in the corpus.

**Change:** a PII-specific NER model (Presidio with a locale recogniser),
evaluated for precision before it is trusted, plus a leak canary in CI: a known
synthetic identity that must never appear in retrieval output.

### 2.7 Compliance detection is keyword-based
It catches an agent who says "recorded" and misses one who says "we keep a copy
of this conversation for training". Fine for a demo, not for a compliance
control.

**Change:** a small classifier per disclosure, trained on real agent phrasings,
with the keyword rules kept as a fast path.

## 3. Behaviour at 10× scale

Current: **one call per process.** Where it breaks, in order:

| Concurrency | What breaks | Fix |
|---|---|---|
| ~5 calls | Free-tier rate limits (Groq 8k TPM, Gemini RPM) | Paid tier; per-tenant token budgets; the rolling-window design already minimises tokens |
| ~10 calls | ASR becomes the bottleneck; one process, one event loop | Worker pool, one process per N calls; ASR on a dedicated queue with backpressure |
| ~25 calls | The retriever holds the full embedding matrix per process | Move to pgvector/managed vector DB; a shared read replica instead of per-process copies |
| ~50 calls | SQLite write contention on CRM/transcript writes | Postgres; append events to a queue rather than writing files inline |
| Any scale | A slow LLM call blocking a turn | Already isolated: the LLM signal layer is concurrent and debounced, and a timeout drops one analysis pass rather than the call |

The design choices that survive 10×: retrieval filters in SQL, checksum-keyed
embedding cache, rules-before-LLM ordering, and the rolling window. The ones
that do not: SQLite, the in-process embedding matrix, and one-call-per-process.

**Cost note:** at 10× on paid tiers, the dominant line item is ASR (per audio
minute), not the LLM. Worth measuring before optimising prompt tokens.

## 4. Behaviour on noisy audio

Tested with the `q4_noisy_ambiguous` scenario — hesitation, self-correction,
half-finished sentences — where the system correctly stayed silent. **Not**
tested with real acoustic noise, because TTS cannot produce it honestly.

Expected degradation, in order:

1. **VAD false triggers** (§2.3) — spurious segments, wasted ASR calls.
2. **ASR insertion errors** — noise transcribed as words, which is worse than
   deletion because keyword rules fire on it. Mitigation already present: the
   confidence gate refuses to answer from a weak match, and per-kind confidence
   floors keep low-value nudges off the screen.
3. **Speaker attribution** — perfect here because recordings are stereo by
   channel. A mono recording loses that entirely; the code falls back to
   attributing everything to the customer and says so in the report. Real
   deployments should keep channel separation at the recorder.

**Change:** a noise-robustness suite mixing real background recordings
(traffic, market, call-centre) at known SNRs, with WER and nudge precision
tracked per SNR band. That is the missing measurement.

## 5. Security and data protection

Done here: no secrets in the repo (`.env` gitignored, `.env.example` documents
every variable), PII redacted at rest and blocked from retrieval, no PII in CRM
rows (transcripts referenced by id), AI and recording disclosure enforced in
code, an explicit "we will never ask for OTP or payment" disclosure in both
Q3 markets.

Missing for production: encryption at rest for recordings and transcripts,
retention and purge jobs matching each market's rules, access control and audit
logging on transcript reads, per-tenant key isolation, and a documented
data-processing agreement per market (DPDP in India, Data Privacy Act in the
Philippines, PDP in Indonesia).

## 6. What I would do first, in order

1. **Streaming TTS + streaming ASR.** Biggest perceived-quality win per unit of
   work; roughly halves the felt latency on both Q1 and Q4.
2. **Native-speaker review of the Q3 packs**, then 20 real calls per market.
   Everything else in Q3 is measured; this is the part that is not.
3. **Neural VAD.** One-line swap, removes the most likely field failure.
4. **Postgres + pgvector**, and move the KB build into CI so a source change
   produces a reviewable diff of the records it altered.
5. **Noise-robustness suite** with WER and nudge precision per SNR band.
6. **Nudge feedback loop.** Every nudge should record whether the agent acted on
   it. Without that, the confidence thresholds stay hand-tuned — which is what
   they are today, and the false-positive report says so.
