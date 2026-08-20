# Q3 — Native-language voice bots (Philippines, Indonesia)

**Run the calls:** `make sim-q3` &nbsp;·&nbsp; **Try one live:** `make serve` → <http://127.0.0.1:8000/webcall>
**ASR report:** [`evaluation/asr_benchmark.md`](../evaluation/asr_benchmark.md) &nbsp;·&nbsp;
**Recordings + transcripts:** `data/recordings/sim_q3_*.mp3`, `data/transcripts/sim_q3_*.json`

---

## 1. What "native" means here

The cheap version of this deliverable is the India agent with its strings run
through a translator. That was rejected at the design stage, because the thing
that breaks in a second market is never the vocabulary — it is the **shape of
the conversation**.

So each market gets its own sector, its own flow, and its own objections:

| | India (`en-IN`) | Philippines (`fil-PH`) | Indonesia (`id-ID`) |
|---|---|---|---|
| Sector | NBFC business lending | Life insurance / bancassurance | Multifinance (consumer financing) |
| Call | Qualify a loan lead | Stop a policy lapsing | Instalment reminder, pre-due |
| Goal | An underwriting decision | Payment before the grace period ends | A real date, or an honest "none" |
| Rules engine | 20 CSV rules | None — nothing is underwritten | None — nothing is underwritten |
| Signature objection | Interest rate, hidden charges | "Scam ba ito?" | Indirect refusal |
| Register problem | Hindi-English numerals | `po`/`opo` is not optional | Formal to colloquial is a dial |

The agent code is identical across all three. What changes is a
[locale pack](../src/darwix/voice/locales/) — one YAML file per market holding
the flow, slots, script, disclosures, fallbacks, style rules and market
vocabulary. Adding a fourth market is a new pack, not a new agent.

## 2. The language decisions, and why each one is in the pack

Every one of these is written into the pack with its reasoning attached,
because they are the decisions a translator would silently get wrong.

**Philippines — Taglish is the default, not a fallback.** Urban Filipino
financial conversation runs a Tagalog frame with English technical nouns:
premium, policy, beneficiary, coverage, lapse, rider, due date. Translating
those into formal Tagalog (`patakaran`, `tagatanggap ng benepisyo`) sounds like
a government form and is genuinely harder to understand. The pack keeps them
English on purpose.

**`po`/`opo` is threaded through every line** rather than left to the model. It
is the respect particle; an agent calling an older policyholder without it reads
as rude, not as friendly.

**No `kuya`/`ate` from an institution.** Warm-familiar address terms from a bank
read as a scam call — which matters, because:

**Scam suspicion is a first-class objection.** An unknown number asking about
payments is assumed to be fraud. The agent proves legitimacy before anything
else, and one of the three mandatory disclosures is an explicit "Hindi po ako
hihingi ng OTP, card details, o bayad dito sa tawag".

**Indonesia — register is a dial, not a setting.** The agent opens formal
(`Bapak/Ibu`, full sentences) and follows the customer into colloquial (`Kak`,
`nggak`, `udah`, `aja`) if they lead. It never opens colloquially: that is
disrespectful to an older customer, while staying rigidly formal with a younger
one sounds like a robot.

**Finance vocabulary stays Indonesian.** cicilan, angsuran, tenor, denda, DP,
jatuh tempo, pembiayaan. "Instalment" would confuse the customer, and
`pinjaman` is a **different product** from `pembiayaan`.

**Never mimic a regional accent back.** The agent must understand Javanese and
Sundanese interjections; repeating them from a Jakarta institution reads as
mockery. Understanding and mirroring are separated deliberately.

## 3. Indirect refusal — the hard part of the Indonesian market

Indonesian politeness avoids a flat no. "Nanti saya kabari deh", "iya nanti",
"diusahakan", "kalau ada rezeki" are all, in a collections context, a no.

A bot that hears agreement books a promise-to-pay that never arrives, and a
collections queue full of promises nobody made is worse than an empty one. So
this is **enforced in code, not asked for in a prompt** — the model has every
conversational incentive to hear a yes:

```
customer says a soft-refusal marker
   -> recorded on the call state, never as agreement
   -> the commitment probe is asked ONCE:
      "kira-kira tanggal berapa Bapak/Ibu bisa transfer?
       Kalau memang belum bisa, tidak masalah, saya catat apa adanya."
   -> a date in THAT answer, and only that answer, is a commitment
   -> otherwise the call closes on `no_commitment`, out loud:
      "saya catat belum ada tanggal pastinya ya - tidak saya janjikan apa-apa."
```

Three details, each of which took a second pass to get right:

- **A date is only a commitment in reply to the probe.** The flow asks its own
  date question — "jatuh temponya tanggal berapa ya?" — and the customer naming
  the date the money is **owed** is not the customer promising to send it. The
  first implementation read "jatuh temponya tanggal 5" as a promise-to-pay.
  That is the exact failure the feature exists to prevent.
- **A date taken back in the same breath is not a commitment.** Verbatim from a
  test call, in reply to the probe: *"eh, tanggal 5 ya, tapi bulan ini cashnya
  masih belum ada, jadi transfernya nanti saya kabari ya"* — a date, a "but",
  and a deferral. The deferral wins, because it is the part that decides
  whether money arrives.
- **Under-recording is the right direction to be wrong in.** A date volunteered
  spontaneously later in the call is missed. That sends a human to check;
  over-recording does not.

The markers are pack data (`soft_refusal_markers`, `commitment_markers`), so a
market that declares none keeps the previous behaviour exactly — and a market
that needs them adds them without touching the agent.

## 4. ASR — measured, not asserted

Full report: [`evaluation/asr_benchmark.md`](../evaluation/asr_benchmark.md),
regenerate with `make asr-bench`.

**Production model: Groq `whisper-large-v3-turbo`,** with language auto-detect
left **on**. Forcing `language=tl` degrades the English half of a Taglish
utterance — the single most important ASR decision in this deliverable.

Twelve phrases, each targeting one difficulty, synthesised in a native voice for
its locale so the reference text is exact and word error rate is a real
measurement rather than an impression:

| Market | Phrases | turbo WER | large-v3 WER |
|---|---:|---:|---:|
| India (en-IN) | 2 | 8.8% | 8.8% |
| Philippines (Taglish) | 4 | 9.5% | 5.6% |
| Indonesia (standard) | 3 | 11.9% | 11.9% |
| Indonesia (Javanese) | 2 | 25.0% | 43.8% |
| Indonesia (Sundanese) | 1 | 20.0% | 60.0% |

**What the errors actually are**, which matters more than the headline:

- **Numerals are transcribed as digits** — "seventy two lakh" becomes
  "72 lakh", "tiga puluh enam" becomes "36". Counted as errors by WER, harmless
  downstream, because the slot parser reads both.
- **`OTP` became `gii`.** This one is not harmless: the Philippine
  scam-suspicion path keys on it, so it is in the ASR domain prompt.
- **Regional accents lose the interjections, not the sentence.** `nggih`,
  `matur nuwun`, `hatur nuhun`, `kula` are mangled; the Indonesian carrier
  sentence survives. Payment status is still extracted correctly — which is why
  the Javanese test call completes.
- **The bigger model is worse on regional accents** (43.8% and 60.0% against
  25.0% and 20.0%). It over-normalises toward standard Indonesian. The faster
  model is also the more accurate one here, so there is no tradeoff to make.

**Gemini audio could not be benchmarked**, and the report says so rather than
scoring the failures as 100% WER: 9 of 12 phrases were rate-limited on the free
tier even with backoff. Where it did answer it took **6.5–22.6 s against Groq's
375 ms mean**, which rules it out for a live call regardless of accuracy.

**The caveat that belongs on all of it:** TTS audio is cleaner than a human on a
mobile line in traffic. These are **best-case** numbers. That is precisely why
the error analysis above matters — the mistakes that survive on clean audio are
the ones that will dominate on a real one.

## 5. Market-scoped knowledge

Each pack declares a `market_code`, and retrieval filters on it **in SQL**, not
after ranking. An Indonesian caller cannot be answered from an Indian NBFC's
charges page even if it ranks well.

| Market | KB records |
|---|---:|
| India | 358 |
| Philippines | 17 |
| Indonesia | 16 |

The market KBs are synthetic and say so in every citation. The mechanics they
encode are real — 31-day grace period, lapse and reinstatement, `denda` at
0.5%/day, restructuring — but the numbers are illustrative, and a real
deployment swaps the sources registry, not the pipeline.

## 6. Test calls

Four calls, all produced by the synthetic caller harness: the customer is
generated, spoken by TTS in a native voice, and fed in **as audio**, so VAD,
Whisper, retrieval and grounding run exactly as they would for a human. The
transcripts are what the system actually heard, not what the script said.

| Scenario | Voice | What it tests | Result |
|---|---|---|---|
| `q3_ph_cooperative` | `fil-PH` female | Taglish code-switching, `po` register, English financial nouns inside Tagalog | Completed, all disclosures given |
| `q3_ph_objection` | `en-PH` male | Scam suspicion, then payment difficulty, then asks for a human | Escalated on the turn it was asked |
| `q3_id_colloquial` | `id-ID` male | Colloquial Jakarta register, indirect refusal | Closes `no_commitment`, no promise recorded |
| `q3_id_javanese` | `jv-ID` male | Regional accent outside standard Jakarta speech | Understood, not mimicked; escalated on request |

Expectations are declared in the persona files and checked automatically —
`data/transcripts/simulation_results.json` records pass/fail per call, so a
regression shows up as a failed expectation rather than a transcript someone
has to read. All four currently pass.

The Indonesian call checks two things separately, and the split is deliberate:
`soft_refusal_recorded` is the invariant (the deferral is scripted into the
persona's beats, so it must always be heard), while `disposition` is what the
agent then *does* with it. When the customer varies, a failure points at the
right one of the two.

## 7. Bugs found and fixed while testing

All three were found by running the calls and reading what the agent actually
said. They are in the code with the reasoning attached.

- **The Indonesian pack documented indirect-refusal handling that did not
  exist.** `soft_refusal_markers` was declared in the pack and commented as
  "enforced in code" — and nothing read it. The customer said "nanti saya kabari
  deh kalau sudah ada" and the call closed on `pass` with "tinggal dibayar
  sebelum jatuh tempo": a promise-to-pay that was never made. Now implemented,
  tested, and expected by the simulation harness (§3).
- **"Sorry, I could not hear you" in reply to a customer saying they have no
  money.** When an answer was heard but the value did not parse, the retry
  prefix apologised for a mishearing that never happened. In a collections call
  that is the single worst line available. It now bridges — "Kembali ke tadi ya
  Pak/Bu" — and only claims not to have heard when the turn was genuinely
  unintelligible.
- **Amounts were unparseable in both non-India markets.** Both packs ask for the
  figure *as spoken* — the Indonesian pack's own example is "satu juta dua ratus
  ribu" — and the parser was India-only. `satu juta dua ratus ribu`,
  `1.200.000`, `tatlong libo` and `limang libo` all returned `None`, so the
  agent asked for the instalment twice and abandoned it. Worse, **`2,5 juta`
  returned `25`** — a silently wrong number, which is the dangerous kind.
  `parse_amount` now reads Indonesian and Filipino numerals and both digit
  grouping conventions (`1.200.000` is 1.2 million, `45,00,000` is 4.5 million),
  with the collision pinned by test.

## 8. What is not tested

The synthetic caller is a TTS voice reading generated text. It is reproducible
and honest, and it is **not** evidence that a Filipino or Indonesian speaker
would find the bot natural. Untested: true spontaneous code-switching,
disfluency and self-correction, overlapping speech, accents beyond the two TTS
voices available, and whether `po` placement reads as respectful or as a script.

Before any pilot: native-speaker review of both packs, then 20–30 real recorded
calls per market scored by a native reviewer. Everything else in Q3 is measured;
this part is not, and it is not claimed. See
[`limitations_and_production_plan.md`](limitations_and_production_plan.md) §2.4.
