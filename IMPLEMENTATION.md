# How this system is built

A walkthrough for someone opening this repository for the first time. It goes
in the order the system actually works, and at every step it says **why** the
choice was made and **what would break** if it had been made differently —
because that is what you will be asked, not "which library did you use".

Read it top to bottom once. You do not need to read the code alongside it.

---

## Table of contents

1. [The one-paragraph version](#1-the-one-paragraph-version)
2. [Why it is one codebase and not four](#2-why-it-is-one-codebase-and-not-four)
3. [The idea everything rests on](#3-the-idea-everything-rests-on)
4. [Q2 — building the knowledge base](#4-q2--building-the-knowledge-base)
5. [Q1 — the voice agent](#5-q1--the-voice-agent)
6. [Q3 — two more markets](#6-q3--two-more-markets)
7. [Q4 — live nudges during a call](#7-q4--live-nudges-during-a-call)
8. [How the evidence is produced](#8-how-the-evidence-is-produced)
9. [Bugs found by using it](#9-bugs-found-by-using-it)
10. [What it cannot do](#10-what-it-cannot-do)
11. [Questions you should be ready for](#11-questions-you-should-be-ready-for)

---

## 1. The one-paragraph version

A lender wants an AI agent that phones small-business owners, answers their
questions, works out whether they qualify for a loan, and hands over to a human
when it should. The hard part is not the talking — it is that **a lending agent
may not invent facts**. So the system first turns the lender's real website,
its policy PDFs and its internal documents into a searchable knowledge base
where every record can be traced back to its source. The agent may then only
say things it can find in that knowledge base, and it is checked five separate
ways before it is allowed to speak. The same agent runs in three markets by
swapping a configuration file. And a fourth component listens to a call *while
it is happening* and whispers suggestions to a human agent within a second.

---

## 2. Why it is one codebase and not four

The assessment asks four questions. The obvious response is four folders. This
repository is one system, and that is a deliberate answer you should be able to
defend:

* **Q1 and Q3 are the same agent.** A Philippine insurance call and an Indian
  business-loan call differ in language, sector, script, objections and
  politeness — but not in *code*. All of that lives in a per-market
  configuration file (a "locale pack"). Same agent, different pack.
* **Q4 reuses Q1's audio layer.** Q1 listens to a caller in order to reply. Q4
  listens to a call in order to advise. The microphone handling, the
  voice-activity detection and the transcription are identical.
* **Q2 is not a separate deliverable at all.** It is the thing Q1 is not allowed
  to speak without.

Split into four, you would tune three copies of the voice-activity detector and
have three places where a compliance rule could be wrong. That is the real
argument: **duplicated safety logic is how safety logic drifts.**

---

## 3. The idea everything rests on

> **The model decides what the customer meant. Code decides everything else.**

The language model is used for exactly what it is good at: understanding a
messy human sentence, and turning an approved fact into a natural reply.

Everything with a consequence is ordinary code with tests around it:

| Decision | Who makes it | Why |
|---|---|---|
| What did the customer mean? | the model | genuinely hard, and a mistake is recoverable |
| May this fact be stated at all? | code | a wrong fact on a lending call is a mis-selling complaint |
| Does this customer qualify? | code | comparing "forty-five lakhs" to a threshold must be right *every* time |
| Were the legal disclosures given? | code | "the model usually remembers" is not a compliance control |
| Should we escalate to a human? | code | a model under pressure to be helpful will talk someone out of it |

A model asked to compare numbers is right *most* of the time. For a loan
decision, "most of the time" is the wrong standard, and the failure is silent.

---

## 4. Q2 — building the knowledge base

Goal: turn a real company's messy public content into records a voice agent can
search, quote and cite.

The pipeline is four commands, each doing one thing, so any stage can be
re-run alone:

```
crawl  →  clean  →  build  →  index
```

### 4.1 Crawl — getting the raw material

Source: a real listed Indian lender's public website (33 pages) plus 7 policy
PDFs it links to. Chosen because it is genuinely messy — marketing pages,
regulatory PDFs, an FAQ accordion, tables — which is the point of the exercise.

Two things worth knowing:

* Some pages are rendered by JavaScript, so a plain HTTP fetch returns an empty
  shell. A headless browser is used where that happens.
* **10 URLs in the site's own sitemap return a "page not found" body with an
  HTTP 200 status.** They are recorded as `soft_404` with a note, and excluded.
  The brief asks you to "handle extraction failures and flag obvious source
  errors" — this is that, and it is a real defect in the real site, not a
  hypothetical.

Alongside the real content, five internal documents are authored (a credit
policy, a qualification matrix, an objection handbook, an application form, and
mock CRM leads). A real lender's rate card is not on its website. **Every
synthetic file opens with a `SYNTHETIC DOCUMENT` banner, is tagged
`internal_*`, and says so in the citation the agent speaks** — so nothing
invented is ever attributed to the real company.

### 4.2 Clean — throwing away the page furniture

Navigation, headers, footers, cookie banners and repeated blocks are removed.
The useful trick: text that appears on nearly every page is boilerplate *by
definition*, so it is detected by frequency across the corpus rather than by a
hand-written list of CSS selectors that breaks on the next redesign.

### 4.3 Build — chunking, normalising, PII, versioning

**Chunking: 250–400 tokens, ~60 token overlap.**
Why that size: a chunk is what the agent reads aloud from. Too large and the
answer buries the fact in a paragraph of marketing. Too small and a rule gets
split from the condition that qualifies it. The overlap stops a sentence
falling into the crack between two chunks.

Tables and rule rows are treated as **atomic** — a rule row is never split,
because half a rule is worse than no rule.

**Normalisation.** Dates to one format, amounts annotated (`45 lakh` →
also `4500000`), terminology canonicalised (`EMI`, `foreclosure`, `bureau
score`), so a customer saying "closing charges" reaches the record that says
"foreclosure". Critically, **normalisation adds aliases; it does not rewrite the
source text**, so a citation still quotes what the document really says. There
is a test named exactly that: `test_canonical_terms_do_not_rewrite_the_source`.

**Deduplication.** Exact duplicates by checksum, near-duplicates by similarity.
432 chunks became 391 records — 41 removed, each keeping a record of what it was
merged from.

**PII.** Names, phone numbers, PANs, emails and bank accounts are detected. 39
records are flagged and **blocked from retrieval entirely** — they still exist,
they simply cannot be reached by a search.

> There is a deliberate test for this (RT11): *"Give me the mobile number and PAN
> of Ramesh Iyer from Surat."* That lead really is in the corpus, with a real
> phone number and a valid-checksum PAN. What comes back instead is the privacy
> policy's definition of personal data. **The first version of this test expected
> the system to go silent, and that expectation was wrong** — the requirement is
> that the lead is unreachable, not that retrieval refuses to answer. The test
> was corrected, not the system.

**Versioning.** Every record carries a version. Change the credit policy, rebuild,
and changed records bump to `1.1`. Currently: 184 records at `1.0`, 189 at `1.1`,
18 at `1.2`. That is the audit trail — you can say *when* a rule changed.

### 4.4 Index and retrieve — how a question finds an answer

Two searches run over the same records and their results are fused:

* **BM25 (keyword).** Finds exact terms. Someone asking about the "bounce charge"
  needs the record containing the words "bounce charge".
* **Dense (meaning).** Finds records that mean the same thing in different words.
  "What do I pay monthly?" must reach a record that says "EMI".

Neither alone is enough — that is the whole argument for using both. The
keyword search misses paraphrases; the meaning search returns something
plausible-but-adjacent for an exact term, which on a fee question is dangerous.

They are combined with **Reciprocal Rank Fusion**, which merges by *rank*
position rather than by score. This matters because a BM25 score and a cosine
similarity are not on the same scale and cannot be added without arbitrary
weighting. RRF sidesteps that entirely. It also degrades gracefully: with no API
key there are no embeddings, and the same code path returns pure keyword results.

**No vector database.** 352 searchable records and a NumPy matrix. At this size a
vector DB is a dependency, a service and an ops burden buying nothing —
brute-force cosine over 352 rows is instant. The write-up states exactly where
that stops being true (roughly 25 concurrent calls, when each process holding
its own copy of the matrix becomes the problem).

**Citations.** Every record knows its origin, so an answer can name its source:

```
Schedule of Charges (SOC) | page 14 | https://www.ugrocapital.com/.../schedule_of_charges.pdf
```

**0 of 391 records lack a source reference.** That is checked, not assumed.

### 4.5 Proving retrieval works

15 test queries in `config/retrieval_tests.yaml`. The important detail:
**each query declares what a correct answer must contain, before it is run.**
Otherwise you are grading your own homework after seeing the output.

Result: **14 correct, 1 partially correct, 0 incorrect.** All five question
types the brief names are covered — product, policy, qualification, FAQ,
objection — plus two negative tests where the *correct* behaviour is to refuse.

Two of them are interesting because of what they exposed:

* **RT13** retrieves exactly the right record — and that record's category label
  is wrong. The chunk spans sections 9–12 of the credit policy and got labelled
  from the branch-partner section at its tail. Retrieval is unaffected; a
  category-filtered search would have missed it. Left visible rather than hidden
  by loosening the test.
* **RT15** answers from a website FAQ accordion whose *questions* were lost
  during extraction, leaving a record that is a bare list of answers. The dense
  search still finds it; keyword search alone would not have, because the words
  to match on were in the discarded text. That is the hybrid retriever earning
  its place.

---

## 5. Q1 — the voice agent

An Indian NBFC agent qualifying business-loan leads in `en-IN`.

### 5.1 What happens in one turn

```
you speak
   ↓  browser captures the microphone, sends 16 kHz audio over a WebSocket
   ↓  VAD decides when you have stopped talking          ← §5.2
   ↓  Whisper transcribes that one utterance
   ↓  the model works out intent + any facts you gave    ← the model's job
   ↓  ── if you asked a question ──→ search the KB → five gates → answer  ← §5.3
   ↓  ── otherwise ──→ ask the next qualification question              ← §5.4
   ↓  text-to-speech
   ↓  audio streams back and plays
```

### 5.2 Knowing when you have finished speaking

This is the single biggest driver of whether a voice agent feels alive. Too
eager and it interrupts you; too patient and it feels dead.

The detector measures loudness and compares it to an estimate of the room's
background noise. Three details, each of which came from something going wrong:

* **1100 ms of hangover, not 700 ms.** A natural pause inside one sentence was
  read as end-of-turn, so the agent answered half a question.
* **300 ms of pre-roll.** The detector needs ~250 ms of sound before it is sure
  you are talking — by which point your first word is already past. A small
  rolling buffer keeps the preceding audio, so the utterance is not clipped.
  ("Who is this?" kept arriving as "is this?")
* **The noise-floor estimate is bounded at both ends and taken from a low
  percentile of the last ~15 seconds.** See §9 — this one is worth understanding
  properly, because it caused the most confusing bug in the project.

### 5.3 The five gates — why the agent cannot make things up

When you ask a factual question, the answer must survive all five:

1. **Retrieval confidence.** If the best matching record scores below the
   threshold, stop. The agent says it does not know and offers a human.
2. **The model may decline.** It is given the retrieved records and explicitly
   permitted to answer `false` — declining is a valid, encouraged outcome.
3. **It must cite.** The model returns which record IDs it used. IDs that were
   not among the retrieved records are discarded; if nothing valid is cited, the
   answer is rejected.
4. **The numeric guard.** *Every number in the answer must literally appear in
   the retrieved text.* This is the gate that matters most. The failure it
   prevents is a model turning "14.5% to 16.0%" into "around 15%" — fluent,
   confident, and a mis-quoted rate on a lending call.
5. **One retry, then refuse.** A rejected answer is sent back with the reason
   attached. If the second attempt also fails, the agent says it does not know.
   There is no third try: on a live call another round trip costs more than
   admitting ignorance.

Every one of these is visible in the web interface while you talk to it, which
is the point — you can watch a claim be accepted or refused.

### 5.4 Qualification — rules as data

Loan rules live in a CSV that is itself a knowledge-base source document:

```csv
rule_id,product,slot,operator,value,disposition,reason_code,customer_message
QR001,unsecured_business_loan,entity_type,in,proprietorship|partnership|llp|private_limited,pass,ENTITY_OK,...
QR013,unsecured_business_loan,credit_score,lt,675,reject,BUREAU_LOW,...
```

Changing a turnover cut-off is a CSV edit plus a rebuild — with a version bump
and an audit trail — **not a prompt edit**. The brief asks directly for this
("do not hardcode all FAQs, objections, or policies in the system prompt").

Three outcomes, not two: **pass / refer / reject**. `refer` exists because a real
credit desk does not answer every borderline file with "no", and a bot that says
"no" to a qualified-with-review customer destroys more value than one that hedges.

### 5.5 Disclosures, escalation, and the business action

**Disclosures are enforced by code, not remembered by the model.** Three must be
given before any qualification question: that it is an AI, that the call is
recorded, and why it is calling. The dialog policy checks this. If the opening
is interrupted before they all land, the outstanding ones are re-delivered
before the first qualification question — and they are recorded as "given" only
*after* the audio has actually been sent, so the transcript records what the
caller heard rather than what was assembled.

**Escalation stops everything.** Ask for a human and the agent acknowledges,
captures a callback, and hands off. No rebuttal, no "let me just ask one more
thing" — that is the behaviour customers hate most.

**Business action:** two are implemented — a lead is written to a mock CRM, and
an escalation webhook fires.

### 5.6 The interface

`make serve` hosts three pages: `/webcall` to talk to the agent, `/kb` to search
the knowledge base directly, `/dashboard` for the Q4 screen. A browser web call
rather than a phone number — a deliberate constraint-driven choice (no paid
telephony account) with a real upside: every stage is inspectable and
measurable, and none of it is hidden inside a vendor.

---

## 6. Q3 — two more markets

Philippines (life insurance, Taglish) and Indonesia (multifinance, including a
regional accent). **The agent code is identical. A market is a configuration
file.**

The pack holds: the sector and product, the conversation flow, the brand, the
voice, the disclosures, the script, the fallbacks, the escalation rules and the
politeness style.

### Why this is localisation and not translation

A translated agent asks the same questions in another language. These do not:

* **Different sector.** India is business lending; the Philippines is life
  insurance; Indonesia is instalment finance. Different questions entirely.
* **Different flow.** A premium reminder is not a lead qualification.
* **Different objections.** "Is this a scam?" is a leading objection in the
  Philippines. It is not on the Indian list.
* **Different politeness systems.** Filipino `po`/`opo` is not decoration — its
  absence reads as rudeness.
* **Different payment vocabulary.** GCash and over-the-counter in the
  Philippines; Indomaret and virtual accounts in Indonesia. Translating "bank
  transfer" is not enough.

**The most interesting piece is indirect refusal.** Indonesian politeness avoids
a flat no. *"Nanti saya kabari deh"* — "I'll let you know later" — is a refusal
wearing the clothes of a promise. Read as agreement, it books a promise-to-pay
that never arrives and poisons the collections queue. This is detected in code,
deliberately not left to the model, which has every conversational incentive to
hear a yes.

**Code-switching is a first-class requirement.** ASR language is left on
auto-detect, because forcing Tagalog degrades the English half of a Taglish
sentence. That choice has a cost — see §9.

**ASR is measured, not asserted.** Word error rate per market against known
reference text: ~8.8% `en-IN`, 9.5% Taglish, 11.9% `id-ID`, **25.0% Javanese**.
That last number is the honest finding: the regional accent is roughly twice as
hard, which is exactly the sort of thing you are meant to discover and report
rather than smooth over.

---

## 7. Q4 — live nudges during a call

A human agent is on a call. This listens and suggests things **before the call
ends**. Analysis after hang-up explicitly does not qualify.

### Real-time by construction

Audio is replayed **at wall-clock speed** — the source sleeps between chunks to
match real time. The pipeline therefore *cannot* see audio that would not yet
exist on a live call. That is a structural guarantee, not a promise.

Stereo recordings put the agent on one channel and the customer on the other,
which gives speaker separation for free — no diarisation model, and exactly how
contact-centre recorders behave.

### Two tiers of signal detection

**Rules first (microseconds, certain).** A disclosure either was or was not
said. Compliance is modelled as *a checklist with a deadline*: each required
disclosure must appear before a given point, and the signal fires when the
deadline passes without it. Running these first means the compliance-critical
signals never depend on a model call that might time out.

**Then the model (~700 ms, judgement).** Frustration building over three turns,
an implied buying signal, an opportunity nobody named explicitly. It runs over a
rolling window, debounced, concurrently.

### Nudge control — the part that makes it usable

An assistant that fires constantly gets switched off. Five controls:

* **Per-kind confidence floors.** The bar scales with the cost of being wrong in
  the *other* direction. Missing a compliance gap is expensive, so it keeps a low
  floor. Missing a speculative opportunity costs almost nothing, so it must be
  nearly certain to earn screen space.
* **Topic grouping** so the rules layer and the model layer cannot both fire for
  one situation.
* **Duplicate suppression**, **cooldowns**, **priorities**, and **expiry**.

The per-kind floors came from a measured false positive: on the deliberately
uneventful test call, the model reported "customer needs to check with their
accountant" as a missed opportunity at 0.80 confidence, and it was shown. Not
wrong, exactly — just not worth interrupting a human for.

### What it produces

Nudge text is a short imperative, because an agent mid-sentence cannot read a
paragraph:

> *"Careful — that sounded like a guarantee. Re-frame: decision rests with credit."*

**Measured end to end**: audio in → transcription → signal → nudge → displayed.
Latest run **854 ms p50, 1,228 ms p95**, with a per-component breakdown for
transcription, rules, model and delivery. Four scenarios, all passing, including
one deliberately uneventful call where the correct behaviour is **no nudges at
all**.

The model and transcription calls are non-deterministic, so re-running moves
these figures — p50 has been observed between 610 and 860 ms on identical audio.
Worth saying plainly if asked: **a scenario changing verdict is a regression, a
figure shifting a hundred milliseconds is not.** The files in `evaluation/` are
the source of truth for whichever run produced them.

---

## 8. How the evidence is produced

This is the part most worth understanding, because it is what separates a demo
from a measurement.

**The test calls were not scripted into existence.** A synthetic caller persona
is generated by a model, spoken by text-to-speech in a native voice for its
market, and fed to the agent **as audio**. So the voice detection, Whisper,
retrieval and grounding all run exactly as they would for a human. The
transcripts record what the system actually *heard* — which is why the mistakes
in them are real ones.

**Every expectation is declared before the run.** Each persona states its
expected outcome in a YAML file before it runs; the results file scores against
that declaration. Same for retrieval (`config/retrieval_tests.yaml`) and for Q4
(`data/recordings/<id>.truth.json`).

That discipline is what let a real question get settled during development:
a Q4 run produced 12 transcript segments where the committed evidence had 17,
which looked like a regression. Comparing the old and new detectors directly
showed 43 versus 46 segments — the new one splits slightly *more* — and a clean
re-run scored 4/4. The 12-segment run had lost segments to API rate limiting.
**Without pre-declared expectations that would have been an argument; with them
it was a five-minute check.**

Current evidence: 9/9 call scenarios pass · 14/15 retrieval queries correct ·
4/4 Q4 scenarios pass · **235 unit tests**, no API key required.

---

## 9. Bugs found by using it

Worth knowing in detail — these are the best interview material in the project,
because each one is a case of a system that looked fine until it was actually
used.

### "Thank you." — the one that taught the most

Every spoken turn came back as *"Thank you."* regardless of what was said.

The chain, once traced:

1. The noise-floor estimate had **no lower bound**. In a quiet room it decayed
   toward the room tone it was measuring. Measured: the speech threshold fell
   from 0.0229 to 0.0012 in 51 seconds — a 19× collapse.
2. Below that threshold, an ordinary breath counted as speech, and ~3 seconds of
   near-silence was sent to Whisper.
3. **Whisper does not return an empty string for silence.** Trained on subtitle
   tracks, it returns the phrases that most often sit over quiet footage —
   `"Thank you."`, `"Thanks for watching!"` — with entirely ordinary confidence,
   so no score threshold catches them.

Fixed in three independent places, because one guard failing silently is how
this happened: bound the noise floor, add an absolute energy gate so silence
never reaches the transcriber at all, and recognise the known artefacts in the
text as a last resort.

**The judgement call worth defending:** the text filter is deliberately narrow.
It does *not* contain "yes", "no" or "okay" — those are whole answers to a
consent question, and swallowing them would break qualification far more often
than a stray artefact ever did. The energy gate is the real defence; the word
list is only a backstop.

### The room that was too noisy in the other direction

Later, in a room at 0.013 loudness, the opposite failed: once the room tone rose
*above* the threshold, every chunk counted as speech, so the branch that adapts
the floor never ran again and the detector stayed permanently open. The fix was
to estimate the floor from a **low percentile of the last ~15 seconds** instead —
that keeps adapting whatever the speech decision is, because speech has gaps and
the gaps are what a low percentile sees.

### The agent talking to itself

Once the agent's voice actually played through speakers, the microphone heard it,
transcribed it, and answered it. Now a higher bar applies while the agent is
speaking: quiet bleed is ignored, a genuinely loud interruption still cuts it off.

### Silence that spoke Japanese

A turn arrived as `ご視聴ありがとうございました` on an `en-IN` call — the same
subtitle artefact, in Japanese. Because language is left on auto-detect for
Taglish, Whisper picks a language for noise too. Two defences: known artefacts in
ten languages, and a per-market list of plausible languages. On the committed
recordings that language filter rejects **zero** legitimate segments — it is a
pure safety net.

### The recording that ran at double speed

Both channels were being advanced during playback — the live microphone *and* an
equal run of silence — so the customer channel ran at roughly double rate
whenever the agent talked. One call recorded the agent ending at 58 seconds and
the customer at 142 in the same file. Now the microphone is the only clock. A
75-second call produces a 75.0-second file with utterances landing where they
were actually spoken.

### The lesson about tests

The first set of detector tests used synthetic constant-level noise and passed
throughout. They would have passed with a detector that failed on real
conversation. There is now a test file that runs against the committed
recordings and asserts something real: **that the silence a `speech_end` claims
to have detected is actually silent on the tape.** The first version of that test
asserted the wrong thing and passed with the bug still present — so it was
rewritten until it genuinely failed on the old code and passed on the new.

---

## 10. What it cannot do

Say these before you are asked. They are in
`docs/limitations_and_production_plan.md` in full.

* **Browser call, not a phone number.** No PSTN, no telephony platform.
* **The callers are synthetic.** No human has spoken to it end to end.
* **`edge-tts` cannot ship.** It is Microsoft Edge's read-aloud service — free,
  with good Filipino and Indonesian voices, and not licensed for production.
* **No native speaker has reviewed the Philippine or Indonesian packs.** The
  vocabulary and politeness are researched, not validated. Stated plainly.
* **Energy-based voice detection**, not a neural one. It handles steady
  background noise; it does not handle a jeepney or a roadside.
* **One call per process.** The scaling document names the breakpoints: ~5 calls
  hits free-tier rate limits, ~10 makes transcription the bottleneck, ~25 breaks
  the in-process embedding matrix, ~50 breaks SQLite.
* **Text-to-speech is slow on some networks.** On the development machine the
  provider's TLS handshake is reset after 10.2 seconds on every attempt, adding
  ~11 seconds per turn. That is a network problem, not a code one — there is a
  copy-pasteable probe in the docs to check it, and anything under 500 ms is
  healthy.

---

## 11. Questions you should be ready for

**"How do you know it doesn't hallucinate?"**
Five gates, and the fourth is the one to describe: every number in an answer
must literally appear in the retrieved source text, or the answer is thrown away
and retried once. The failure that prevents is "14.5% to 16.0%" becoming "around
15%" — fluent, confident, and a mis-quoted rate.

**"Why not a vector database?"**
352 records. Brute-force cosine over a NumPy matrix is instant, and a vector DB
would be a service to run for no gain. I know where that stops being true —
around 25 concurrent calls, when per-process copies of the matrix become the
problem — and it is written down.

**"Why both keyword and semantic search?"**
Keyword misses paraphrases; semantic returns plausible-but-adjacent records for
an exact term, which on a fee question is dangerous. RT15 is the concrete case:
a FAQ record whose questions were lost in extraction is only reachable by
meaning. They are fused by rank, not score, so no scale normalisation is needed
and it degrades to pure keyword with no API key.

**"How is Q3 not just translation?"**
Different sector, flow, objections, politeness system and payment vocabulary per
market. The clearest single example is indirect refusal: *"nanti saya kabari
deh"* is a refusal shaped like a promise, and reading it as agreement poisons a
collections queue. Detected in code, not by the model.

**"How do you know the nudges are real-time?"**
The replay source sleeps between chunks to match wall-clock time, so the
pipeline cannot see audio that would not yet exist. Latency is measured end to
end with a per-component breakdown, p50 and p95.

**"What went wrong while you built it?"**
Lead with the "Thank you." bug. It has a clean three-link causal chain, a
non-obvious root cause, a fix in three independent layers, and a defensible
judgement call about not filtering the word "yes".

**"What would you do next?"**
Streaming text-to-speech (the biggest perceived-latency win), a neural voice
detector for noisy environments, a native speaker reviewing both market packs,
and moving off SQLite and the in-process matrix before ~25 concurrent calls.

---

## Appendix — running it

```bash
cp .env.example .env      # two free-tier API keys: Google AI Studio, Groq
make install
make test                 # 235 tests, no network, no keys needed
make serve                # http://127.0.0.1:8000
```

Then open `/webcall` and ask *"What are the hidden charges?"* — the citation
appears next to the answer. Ask *"What is the price of gold in Dubai?"* and
watch it refuse instead of improvise.

Everything else:

```bash
make kb          # clean → build → index
make eval        # retrieval test set  → evaluation/retrieval_tests.md
make sim         # every scripted test call, recording audio + transcripts
make asr-bench   # ASR comparison      → evaluation/asr_benchmark.md
make q4-eval     # all Q4 scenarios in real time → latency + false-positive reports
```

The knowledge base is committed, so `/webcall` and `/kb` work immediately with
no crawl and no rebuild.
