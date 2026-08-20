# Q1 — Knowledge-grounded voice agent

**Use case:** business-loan lead qualification for an Indian NBFC.
**Interface:** browser web call — `make serve`, then <http://127.0.0.1:8000/webcall>.
**Recordings + transcripts:** `data/recordings/sim_q1_*.mp3`, `data/transcripts/sim_q1_*.json`.

---

## Why a web call and not a phone number

The assessment allows "a callable number **or** a web calling interface". With
no paid account, a PSTN number was not available — so the browser interface is
self-hosted end to end: `getUserMedia` → 16 kHz PCM over WebSocket → VAD →
Whisper → retrieval → grounding → TTS → audio back, with a live transcript and
the citation for every factual claim shown next to it.

The upside of having built it rather than bought it: every stage is inspectable
and measurable, and the same audio layer is reused by Q4. The cost is stated
plainly — **no PSTN number, so no real telephony conditions** (8 kHz codec,
jitter, packet loss). What that would change is in
[`limitations_and_production_plan.md`](limitations_and_production_plan.md).

## Conversation design

```
greeting → consent → [disclosures] → qualification → outcome → next step → close
```

Any phase can be interrupted by a question, an objection, or a request for a
human; the interrupt is handled and the flow resumes where it left off.

**What the model decides:** what the customer meant, and which values they said.
**What the code decides:** everything else.

| Behaviour | Where it lives | Why not the prompt |
|---|---|---|
| Mandatory disclosures before any question | `dialog_policy.pending_disclosures` | A model under conversational pressure skips them, and a skipped AI disclosure is a compliance failure, not a style slip |
| Eligibility arithmetic | `qualification.py` + `qualification_rules.csv` | An LLM comparing "forty five lakhs" to a threshold is usually right. "Usually" is not a lending control |
| Whether a fact may be stated | `grounding.py` + retrieval threshold | This is the assessment's stated rejection condition |
| Escalation to a human | `escalation.py` | Every trigger is a case where the model has an incentive to keep talking |

## Grounding — how hallucination is actually prevented

Five gates, in order:

1. **Retrieve first.** No retrieval, no factual claim.
2. **Confidence gate.** Below `RETRIEVAL_MIN_SCORE` (0.35) the agent does not
   get to try — it says it does not have the information and offers a human.
   A weak-but-present chunk is exactly what produces a confident wrong answer.
3. **Citation contract.** The model must return the `record_id`s it used; ids it
   invents are discarded. An answer asserting something while citing nothing is
   rejected.
4. **Numeric guard.** Every number in the answer must appear in the retrieved
   text. This catches the specific failure that matters most here: reading
   "14.5% to 16.0%" and saying "around 15%".
5. **One regeneration, then fall back.** The model gets exactly one more attempt
   with the failure made explicit, then the agent says it does not know. There
   is no third try — on a live call the latency of another round trip is worse
   than an honest answer.

Every turn records what happened, so a reviewer can audit it after the fact:

```json
"grounded": {
  "answered": true,
  "top_score": 0.586,
  "used_record_ids": ["objection_handbook__006"],
  "citations": ["OBJ-06 \"Are there hidden charges?\" | section: OBJ-06 | internal document (synthetic…)"],
  "attempts": 1
}
```

## Rules as data, not prompt

`data/raw/internal/qualification_rules.csv` — 20 rules, loaded at runtime:

| rule_id | slot | operator | value | disposition | customer_message |
|---|---|---|---|---|---|
| QR003 | business_vintage_months | gte | 36 | pass | "Your business vintage meets our requirement." |
| QR004 | business_vintage_months | between | 24\|35 | refer | "…a credit officer will review it." |
| QR005 | business_vintage_months | lt | 24 | reject | "We need at least two years of operations…" |

Changing a turnover cut-off is a CSV edit plus a KB rebuild — with a version
bump and an audit trail — not a prompt edit.

**Three dispositions, not two.** `refer` exists because a real credit desk does
not answer every borderline file with "no", and a bot that rejects a
qualified-with-review customer destroys more value than one that hedges.

**A required value the customer would not give can never score as `pass`.** It
downgrades to `refer` with an `UNVERIFIED_*` reason code, so a human looks at it
rather than the bot implying an outcome it has no basis for.

## Escalation and the business action

Triggers, all in code: explicit request · complaint/ombudsman/distress language ·
three consecutive unanswerable questions · two refusals · three unintelligible
turns.

Every completed call writes a **mock CRM lead** (`data/crm/leads.jsonl`) with
slots, disposition, reason codes, disclosures given, consent, callback slot and
a transcript reference — plus an escalation webhook payload when configured.
Note what is *not* written: the transcript text is referenced by file id, so the
CRM copy never becomes a second uncontrolled store of what the customer said.

## Test calls

Five recorded calls, one per scenario the assessment requires. All produced by
the synthetic caller harness: the customer is generated, spoken by TTS, and fed
in **as audio** — so VAD, Whisper, retrieval and grounding all run exactly as
they would for a human caller. The transcripts are what the system actually
heard, not what the script said.

| Scenario | Persona | Result |
|---|---|---|
| Cooperative customer | `q1_cooperative` | 9 turns, `pass`, rules QR001/003/006/009/014 fired, callback booked |
| Objection (rate + hidden charges) | `q1_objection` | Grounded answers from the objection handbook; no rate quoted |
| Incomplete / conflicting details | `q1_conflicting` | Conflict probe raised; no `pass` on unverified values |
| Out-of-scope question | `q1_out_of_scope` | Declined three times, offered the right team |
| Human-assistance request | `q1_human_request` | Escalated on the turn it was asked, qualification abandoned |

Artefacts: `data/recordings/sim_q1_*.mp3` (audio),
`data/transcripts/sim_q1_*.json` (turns, slots, decisions, per-stage latency),
`data/transcripts/simulation_results.json` (pass/fail against declared
expectations).

## Measured latency (cooperative call)

| Stage | p50 | Is the caller waiting? |
|---|---:|---|
| ASR (Whisper, per utterance) | 335 ms | yes |
| Understand + route (LLM) | 632 ms | yes |
| **Sub-total to a decision** | **967 ms** | yes |
| TTS (`edge-tts`, whole utterance) | 1,487 ms | yes |
| **Total silence heard by the caller** | **2,454 ms** | — |

An earlier version of this table reported 632 ms as the "agent turn total".
That was the LLM stage alone: it excluded the ASR before it and the synthesis
after it, both of which the caller sits through. The honest number is the last
row.

TTS dominates because `edge-tts` synthesises the whole utterance before
returning. A streaming TTS would cut perceived latency substantially and is the
first thing to change in production.

### When the number is much worse

`edge-tts` reaches `speech.platform.bing.com` over WebSocket. On networks that
interfere with that host the TLS handshake stalls before failing, and the cost
is paid per call: measured on one machine, DNS 87 ms and TCP 23 ms both fine,
TLS reset after **10.2 s every attempt**, giving ~11 s to first audio for a
33-character reply and the same ~11 s for a 426-character one. Buffering the
whole utterance accounts for only 321-780 ms of that; the rest is the handshake.

Worth checking before blaming the code:

```bash
python - <<'EOF'
import socket, ssl, time
h = "speech.platform.bing.com"
t = time.perf_counter()
try:
    with socket.create_connection((h, 443), timeout=25) as s:
        ssl.create_default_context().wrap_socket(s, server_hostname=h)
    print("TLS ok in %.0f ms" % ((time.perf_counter() - t) * 1000))
except Exception as e:
    print("TLS %s after %.0f ms" % (type(e).__name__, (time.perf_counter() - t) * 1000))
EOF
```

Under ~500 ms is healthy. Ten seconds means the host is being interfered with
on that network, and every agent turn will carry it.

## Bugs found and fixed while testing

These are in the code with comments attached, because they are the difference
between a demo and something that survives a real call:

- **The agent repeated itself.** Slot extraction dropped
  `business_vintage_months` because the key implied a unit conversion the prompt
  forbade. Fixed with described slots; a slot is now asked at most twice, then
  rephrased, then abandoned as unknown.
- **It delivered the outcome, next step and goodbye in one breath**, then went
  silent. Now three separate turns, so the customer can react.
- **"Sorry, could you repeat that?" triggered a knowledge-base search** and
  returned a confident, irrelevant answer. Repeat requests now repeat.
- **"No existing EMIs" parsed as unknown** instead of zero, stranding the slot.
- **A working-capital question was classified out-of-scope** on a *lending*
  call. "Out of scope" is now a hint, not a verdict: the KB is asked first and
  the agent declines only when retrieval genuinely has nothing.
