"""Call session: one text-level conversation turn loop.

This is deliberately audio-free. The same session drives the browser call
(audio in/out), the simulator (TTS personas), and the test suite (plain text),
so conversation behaviour can be tested without a microphone and reproduced
exactly.

Per customer turn:

    understand (1 LLM call, JSON)
        -> escalation triggers            [code]
        -> intent routing                 [code]
        -> grounded answer if factual     [retrieval + 1-2 LLM calls]
        -> slot merge + qualification     [code, deterministic]
        -> compose reply from locale pack [code, templates]

Reply composition is template-based rather than generated. A generated
"now ask the next question" sentence costs another 600-900 ms per turn and
occasionally invents a question that is not in the flow. The only generated
text in an agent turn is the grounded answer itself.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..common.config import settings
from ..common.latency import LatencyCollector
from ..common.llm import get_llm
from ..common.logging import log
from ..kb.retrieve import Retriever, get_retriever
from . import escalation as esc
from .dialog_policy import (CallState, Intent, Phase, Turn, build_understand_prompt,
                            flow_slots, load_pack, pending_disclosures, render)
from .qualification import QualificationEngine, summarise_for_agent
from .slots import normalise_with, speak_amount_inr

LOCALE_DIRS = {"en-IN": "en_IN", "fil-PH": "fil_PH", "id-ID": "id_ID"}

# Categories the KB may be searched in, per intent. Narrowing the search space
# is cheap precision: an objection should not be answered out of the privacy
# policy.
CATEGORIES_BY_INTENT = {
    Intent.QUESTION: ["product", "eligibility", "pricing", "process", "faq", "policy",
                      "compliance", "partnership", "company"],
    Intent.OBJECTION: ["objection", "pricing", "product", "faq", "policy", "eligibility"],
}


REPEAT_REQUEST = re.compile(
    r"(repeat that|say (that )?again|come again|pardon|what was that|didn'?t (catch|hear)|"
    r"could you repeat|once more|ulangi|paki-ulit|pakiulit)", re.I)


def _is_repeat_request(text: str) -> bool:
    return bool(REPEAT_REQUEST.search(text or ""))


@dataclass
class TurnResult:
    text: str
    state: CallState
    intent: str = ""
    grounded: dict | None = None
    ended: bool = False
    latency_ms: float = 0.0
    stages: dict[str, float] = field(default_factory=dict)


class CallSession:
    def __init__(
        self,
        locale: str = "en-IN",
        *,
        call_id: str | None = None,
        retriever: Retriever | None = None,
        customer_phone: str = "this number",
    ) -> None:
        self.pack = load_pack(LOCALE_DIRS[locale])
        self.state = CallState(call_id=call_id or uuid.uuid4().hex[:12], locale=locale)
        self.flow = flow_slots(self.pack)
        self.parsers = {s["key"]: s.get("parse", "text") for s in self.flow}
        # Only markets with a credit decision to make load the rules engine.
        # A premium reminder does not underwrite anything.
        self.rules_product = self.pack.get("flow", {}).get("rules_product")
        self.qualifier = (QualificationEngine(product=self.rules_product)
                          if self.rules_product else None)
        self.market = self.pack.get("market_code", "IN")
        # Indirect refusal handling is per-market data, not a prompt rule: a
        # market whose pack declares no markers keeps the previous behaviour.
        self.soft_refusal_markers = [m.lower() for m in
                                     self.pack.get("soft_refusal_markers", [])]
        self.commitment_markers = [m.lower() for m in
                                   self.pack.get("commitment_markers", [])]
        self.retriever = retriever or get_retriever()
        self.latency = LatencyCollector()
        self.customer_phone = customer_phone
        self.callback_slot = esc.suggest_callback()
        self.lead: dict | None = None
        self.slot_attempts: dict[str, int] = {}
        self.unresolved_slots: list[str] = []
        self.started_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------ open
    def opening_parts(self) -> list[tuple[str, list[str]]]:
        """The opening, split into utterances the caller can interrupt between.

        Spoken as one block this runs ~32 seconds, and for all of it the caller
        cannot get a word in - a stranger who has already worked out this is a
        cold call is held on the line through three disclosures and two
        questions before they may answer either. Splitting after the identity
        and the AI disclosure gets a barge-in window ~9 s in instead of ~28 s,
        while every disclosure still lands before qualification.

        Each part carries the ids it delivers; they are marked given only once
        the audio has actually been sent (see AudioCall.open), so the compliance
        record reflects what the caller heard rather than what was assembled.
        """
        pack = self.pack
        pending = pending_disclosures(pack, self.state, before="qualification")
        # The AI disclosure travels with the greeting: of the three it is the
        # one a caller is most entitled to hear before deciding to keep talking.
        lead = [d for d in pending if d["id"] == "ai_disclosure"]
        rest = [d for d in pending if d["id"] != "ai_disclosure"]

        first = " ".join([render(pack["script"]["greeting"], pack)]
                         + [d["text"] for d in lead])
        second = " ".join([d["text"] for d in rest]
                          + [render(pack["script"]["consent"], pack)])
        self.state.phase = Phase.CONSENT
        return [(first, [d["id"] for d in lead]),
                (second, [d["id"] for d in rest])]

    def mark_disclosed(self, ids: list[str], text: str) -> None:
        """Record an opening part as actually delivered."""
        for i in ids:
            self.state.disclosures_given.add(i)
        self._record("agent", text)

    def opening(self) -> str:
        """The whole opening as one string, for callers that do not stream."""
        parts = self.opening_parts()
        text = " ".join(t for t, _ in parts)
        for _, ids in parts:
            for i in ids:
                self.state.disclosures_given.add(i)
        self._record("agent", text)
        return text

    # ------------------------------------------------------------------ turn
    async def handle(self, utterance: str) -> TurnResult:
        t0 = time.perf_counter()
        stages: dict[str, float] = {}
        self._record("customer", utterance)

        if self.state.ended:
            return TurnResult(text="", state=self.state, ended=True)

        # 1. understand -------------------------------------------------
        s = time.perf_counter()
        understanding = await self._understand(utterance)
        stages["understand_ms"] = (time.perf_counter() - s) * 1000
        intent = understanding["intent"]
        sentiment = understanding.get("sentiment", "neutral")

        # 2. escalation, before anything else --------------------------
        trigger = esc.check_triggers(self.state, intent=intent.value,
                                     sentiment=sentiment, utterance=utterance)
        if trigger:
            text = self._escalate(trigger)
            return self._finish(text, intent, t0, stages, ended=True)

        # 3. slots (merged even when the turn is an objection - customers
        #    volunteer facts while complaining) -------------------------
        raw_slots = understanding.get("slots") or {}
        merged, unparsed = normalise_with(raw_slots, self.parsers)
        newly = {k: v for k, v in merged.items() if self.state.slots.get(k) != v}
        self.state.slots.update(merged)
        if newly:
            log("session.slots", call_id=self.state.call_id, new=newly, unparsed=unparsed)

        # 3b. indirect refusal -----------------------------------------
        # Indonesian politeness avoids a flat no: "nanti saya kabari deh" is a
        # refusal wearing the clothes of a promise. Reading it as agreement
        # books a promise-to-pay that never arrives and poisons the collections
        # queue, so it is detected here rather than left to the model - which
        # has every conversational incentive to hear a yes.
        self._note_commitment(utterance)
        if (self.state.soft_refusals and not self.state.commitment
                and not self.state.commitment_probed
                and self.state.phase in (Phase.QUALIFICATION, Phase.DECISION,
                                         Phase.NEXT_STEP)):
            probe = self.pack["script"].get("commitment_probe")
            if probe:
                self.state.commitment_probed = True
                self.state.awaiting_commitment = True
                return self._finish(render(probe, self.pack), intent, t0, stages)

        # 4. route ------------------------------------------------------
        grounded_payload = None
        reply_parts: list[str] = []

        if intent in (Intent.QUESTION, Intent.OBJECTION) and _is_repeat_request(utterance):
            # "Sorry, what?" is not a knowledge-base question. Searching the KB
            # for it returns a confident, irrelevant answer - so repeat the
            # last thing the agent actually said instead.
            last_agent = next((t.text for t in reversed(self.state.turns[:-1])
                               if t.speaker == "agent"), "")
            if last_agent:
                return self._finish(last_agent, intent, t0, stages)

        if intent in (Intent.QUESTION, Intent.OBJECTION):
            s = time.perf_counter()
            answer = await self._grounded(understanding, intent)
            stages["retrieval_answer_ms"] = (time.perf_counter() - s) * 1000
            grounded_payload = answer.as_dict()
            reply_parts.append(answer.text)
            self.state.low_confidence_streak = 0 if answer.answered else self.state.low_confidence_streak + 1
            if not answer.answered:
                retrigger = esc.check_triggers(self.state, intent=intent.value,
                                               sentiment=sentiment, utterance=utterance)
                if retrigger:
                    text = " ".join(p for p in reply_parts if p) + " " + self._escalate(retrigger)
                    return self._finish(text.strip(), intent, t0, stages, ended=True)

        elif intent == Intent.REFUSAL:
            self.state.refusal_count += 1
            if self.state.refusal_count >= 2:
                text = render(self.pack["script"]["closing"], self.pack)
                self.state.phase = Phase.CLOSED
                self.state.ended_reason = "customer_declined"
                self._close()
                return self._finish(text, intent, t0, stages, ended=True)
            reply_parts.append(render(self.pack["script"]["consent_declined"], self.pack))

        elif intent == Intent.OUT_OF_SCOPE:
            # "Out of scope" is a hint, not a verdict. In testing the model
            # called a working-capital question out of scope on a *lending*
            # call - which is wrong, and throws away a live lead. The knowledge
            # base is the authority on what this agent actually covers, so ask
            # it first and decline only when retrieval genuinely has nothing.
            s = time.perf_counter()
            answer = await self._grounded(understanding, Intent.QUESTION)
            stages["retrieval_answer_ms"] = (time.perf_counter() - s) * 1000
            grounded_payload = answer.as_dict()
            if answer.answered:
                reply_parts.append(answer.text)
                self.state.low_confidence_streak = 0
            else:
                grounded_payload["rejected_reason"] = (
                    answer.rejected_reason or "out_of_scope")
                reply_parts.append(render(self.pack["fallback"]["out_of_scope"], self.pack))

        elif intent == Intent.UNCLEAR:
            self.state.unclear_streak += 1
            reply_parts.append(render(self.pack["script"]["clarify"], self.pack))
            return self._finish(" ".join(reply_parts), intent, t0, stages)

        if intent != Intent.UNCLEAR:
            self.state.unclear_streak = 0

        # 5. consent gate ----------------------------------------------
        if self.state.phase == Phase.CONSENT:
            if intent == Intent.DENY:
                reply_parts.append(render(self.pack["script"]["consent_declined"], self.pack))
                self.state.consent = False
                return self._finish(" ".join(p for p in reply_parts if p), intent, t0, stages)
            self.state.consent = True
            self.state.phase = Phase.QUALIFICATION

        # 5b. disclosures the caller has not actually heard -------------
        # Now that the opening is spoken in two parts, an interruption can cut
        # it off before the second one - and previously nothing re-delivered
        # them, because pending_disclosures() was consulted in opening() and
        # nowhere else. Anything still outstanding is said here, before a
        # single qualification question is asked. `required_before:
        # qualification` in the pack is the rule; this is where it is kept.
        if self.state.phase in (Phase.QUALIFICATION, Phase.DECISION, Phase.NEXT_STEP):
            missed = pending_disclosures(self.pack, self.state, before="qualification")
            if missed:
                log("session.disclosures_redelivered", call_id=self.state.call_id,
                    ids=[d["id"] for d in missed])
                for d in missed:
                    reply_parts.append(d["text"])
                    self.state.disclosures_given.add(d["id"])

        # 6. qualification ---------------------------------------------
        if self.state.phase in (Phase.QUALIFICATION, Phase.DECISION, Phase.NEXT_STEP):
            reply_parts.extend(self._qualification_step())

        text = " ".join(p for p in reply_parts if p).strip()
        if not text:
            text = render(self.pack["script"]["clarify"], self.pack)
        ended = self.state.phase in (Phase.CLOSED, Phase.ESCALATED)
        result = self._finish(text, intent, t0, stages, ended=ended, grounded=grounded_payload)
        return result

    # ------------------------------------------------------------- internals
    async def _understand(self, utterance: str) -> dict[str, Any]:
        system = build_understand_prompt(self.pack, self.state, self.state.pending_slot)
        try:
            data = await get_llm().chat_json(
                system, [{"role": "user", "content": utterance}],
                provider="groq_dialog", temperature=0.0, max_tokens=350,
            )
        except Exception as exc:  # noqa: BLE001 - a broken turn must not end the call
            log("session.understand_failed", error=str(exc)[:200])
            return {"intent": Intent.UNCLEAR, "slots": {}, "sentiment": "neutral"}
        try:
            intent = Intent(str(data.get("intent", "unclear")).strip().lower())
        except ValueError:
            intent = Intent.UNCLEAR
        data["intent"] = intent
        return data

    async def _grounded(self, understanding: dict, intent: Intent):
        from .grounding import answer_question

        query = (understanding.get("kb_query") or understanding.get("verbatim_concern") or "").strip()
        if not query:
            query = next((t.text for t in reversed(self.state.turns) if t.speaker == "customer"), "")
        fallback = self.pack["fallback"]["unknown" if intent == Intent.QUESTION else "low_confidence"]
        return await answer_question(
            query,
            self.retriever,
            market=self.market,
            language=self.pack.get("kb_language", "en"),
            language_name=self.pack["language_name"],
            brand=self.pack["brand"]["display_name"],
            categories=CATEGORIES_BY_INTENT.get(intent),
            fallback_text=render(fallback, self.pack),
        )

    def _qualification_step(self) -> list[str]:
        pack, state = self.pack, self.state
        parts: list[str] = []

        # The outcome has already been delivered and the customer has replied:
        # confirm the next step and close.
        if state.phase == Phase.NEXT_STEP:
            parts.append(render(pack["script"]["next_step"], pack, callback_slot=self.callback_slot))
            parts.append(render(pack["script"]["closing"], pack))
            state.phase = Phase.CLOSED
            state.ended_reason = "completed:" + (state.decision or {}).get("disposition", "unknown")
            self._close()
            return parts

        # Surface an inconsistency once, before asking anything else.
        for conflict in (self.qualifier.evaluate(state.slots).conflicts
                         if self.qualifier else []):
            if conflict not in state.conflicts_probed:
                state.conflicts_probed.append(conflict)
                parts.append(self._conflict_text(conflict))
                return parts

        # Walk the whole question sequence before deciding. Having the four
        # rule-critical values early does not mean the call is over - industry
        # (negative list), existing obligations and GST all change the outcome.
        slot = self._next_slot()
        while slot and self.slot_attempts.get(slot, 0) >= 2:
            # Asked twice and still nothing usable. A third attempt is what
            # makes a bot sound like a bot, so record it as unknown and move
            # on - the credit officer can chase it, and `evaluate` will not
            # score an unknown required value as a pass.
            state.slots[slot] = None
            self.unresolved_slots.append(slot)
            log("session.slot_abandoned", call_id=state.call_id, slot=slot)
            slot = self._next_slot()

        if slot:
            state.pending_slot = slot
            attempts = self.slot_attempts.get(slot, 0)
            self.slot_attempts[slot] = attempts + 1
            if slot not in state.asked_slots:
                state.asked_slots.append(slot)
            question = self._slot_question(slot)
            if attempts >= 1:
                question = self._slot_question(slot, retry=True) or question
                # Always bridge, never apologise for a mishearing. Reaching
                # here means the customer said something we understood - they
                # asked a question, or answered with a value the parser could
                # not use ("bulan ini berat banget" for an amount). A turn that
                # genuinely was not intelligible never gets this far: the
                # UNCLEAR path returns `clarify` before qualification runs.
                # "Sorry, I could not hear you" in reply to a customer who just
                # said they have no money is the single worst line in the call.
                parts.append(render(pack["script"]["resume"], pack))
            parts.append(render(question, pack))
            return parts

        # Every question asked: decide, state the outcome, then STOP and let
        # the customer react. The next step and the goodbye are separate turns.
        if self.qualifier:
            decision = self.qualifier.evaluate(state.slots)
            state.decision = decision.as_dict()
            outcome_key = decision.disposition
            first_message = decision.messages[0] if decision.messages else ""
        else:
            # No credit decision in this market: the outcome is whether the
            # verification completed cleanly, which the flow declares.
            outcome_key = self.pack.get("flow", {}).get("default_outcome", "pass")
            reasons = ["VERIFIED"]
            if (state.soft_refusals and not state.commitment
                    and "no_commitment" in pack["script"]["outcome"]):
                # The customer never named a date. Closing on the default
                # outcome would record a promise-to-pay that was never made.
                outcome_key = "no_commitment"
                reasons = ["NO_COMMITMENT"]
            state.decision = {"disposition": outcome_key, "reason_codes": reasons,
                              "rules_fired": [], "missing_slots": self.unresolved_slots,
                              "conflicts": [], "customer_messages": [],
                              "soft_refusals": list(state.soft_refusals),
                              "commitment": state.commitment}
            first_message = ""
        state.phase = Phase.NEXT_STEP
        parts.append(render(pack["script"]["outcome"][outcome_key], pack))
        if first_message:
            parts.append(first_message)
        parts.append(render(pack["script"]["outcome_check"], pack))
        return parts

    def _note_commitment(self, utterance: str) -> None:
        """Record an indirect refusal, or a real date that cancels one.

        A concrete date is the only thing that turns "nanti saya kabari" into a
        commitment, and it only counts when it is the answer to the commitment
        probe. The flow asks its own date question ("jatuh temponya tanggal
        berapa?"), and the customer naming the date the money is *owed* is not
        the customer promising to send it - reading it that way is how a
        collections queue fills up with promises nobody made.

        A date the customer takes back in the same breath is not a commitment
        either. Observed verbatim on a test call, in reply to the probe:
        "eh, tanggal 5 ya, tapi bulan ini cashnya masih belum ada, jadi
        transfernya nanti saya kabari ya" - a date, a "but", and a deferral.
        The deferral wins, because it is the part that decides whether money
        arrives.

        The cost of that strictness is missing a date volunteered later in the
        call. That is the right direction to be wrong in: under-recording a
        commitment sends a human to check, over-recording one does not.
        """
        low = " " + " ".join(utterance.lower().split()) + " "
        deferrals = [m for m in self.soft_refusal_markers if m in low]

        if self.state.awaiting_commitment:
            self.state.awaiting_commitment = False   # the probe is asked once
            if not deferrals:
                for marker in self.commitment_markers:
                    if marker in low:
                        self.state.commitment = marker
                        log("session.commitment", call_id=self.state.call_id,
                            marker=marker)
                        return

        for marker in deferrals:
            if marker not in self.state.soft_refusals:
                self.state.soft_refusals.append(marker)
                log("session.soft_refusal", call_id=self.state.call_id, marker=marker)

    def _next_slot(self) -> str | None:
        if self.flow:
            for spec in self.flow:
                if spec["key"] not in self.state.slots:
                    return spec["key"]
            return None
        return self.qualifier.next_question_slot(self.state.slots) if self.qualifier else None

    def _slot_question(self, slot: str, *, retry: bool = False) -> str:
        for spec in self.flow:
            if spec["key"] == slot:
                return spec.get("retry" if retry else "ask", spec.get("ask", ""))
        script = self.pack["script"]
        if retry:
            return script.get("slot_retry", {}).get(slot, "")
        return script["slot_questions"][slot]

    def _conflict_text(self, conflict: str) -> str:
        pack, slots = self.pack, self.state.slots
        readable = {
            "requested_amount_exceeds_annual_turnover": (
                "a turnover of " + speak_amount_inr(int(slots.get("annual_turnover_inr", 0))),
                "a loan requirement of " + speak_amount_inr(int(slots.get("loan_amount_inr", 0)))),
            "existing_obligations_exceed_turnover": (
                "your existing EMI", "the turnover figure you gave"),
            "implausible_business_vintage": ("the business vintage", "the year you started"),
            "turnover_above_gst_threshold_but_not_registered": (
                "a turnover above the GST threshold", "that the business is not GST registered"),
        }.get(conflict, ("what you said earlier", "what you just said"))
        return render(pack["script"]["conflict_probe"], pack, a=readable[0], b=readable[1])

    def _escalate(self, trigger: esc.Escalation) -> str:
        pack = self.pack
        self.state.escalation = trigger.as_dict()
        self.state.phase = Phase.ESCALATED
        self.state.ended_reason = "escalated:" + trigger.reason
        text = " ".join([
            render(pack["escalation"]["acknowledge"], pack),
            render(pack["escalation"]["closing"], pack,
                   phone=self.customer_phone, callback_slot=self.callback_slot),
        ])
        self._close(trigger)
        log("session.escalated", call_id=self.state.call_id, reason=trigger.reason)
        return text

    def _close(self, trigger: esc.Escalation | None = None) -> None:
        if self.lead is not None:
            return
        self.lead = esc.write_lead(self.state, self.state.decision,
                                   escalation=trigger, callback_slot=self.callback_slot)

    def _record(self, speaker: str, text: str, **kw: Any) -> Turn:
        turn = Turn(speaker=speaker, text=text, phase=self.state.phase.value, **kw)
        self.state.turns.append(turn)
        return turn

    def _finish(self, text: str, intent: Intent, t0: float, stages: dict,
                *, ended: bool = False, grounded: dict | None = None) -> TurnResult:
        total = (time.perf_counter() - t0) * 1000
        for name, value in stages.items():
            self.latency.record(name, value)
        self.latency.record("agent_turn_total_ms", total)
        turn = self._record("agent", text, intent=intent.value, latency_ms=total, grounded=grounded)
        turn.meta = {"stages": {k: round(v, 1) for k, v in stages.items()}}
        log("session.turn", call_id=self.state.call_id, intent=intent.value,
            phase=self.state.phase.value, total_ms=round(total), stages=turn.meta["stages"])
        return TurnResult(text=text, state=self.state, intent=intent.value, grounded=grounded,
                          ended=ended or self.state.ended, latency_ms=total, stages=stages)

    # ------------------------------------------------------------- artefacts
    def transcript(self) -> dict:
        return {
            "call_id": self.state.call_id,
            "locale": self.state.locale,
            "started_at": self.started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "ended_reason": self.state.ended_reason,
            "consent": self.state.consent,
            "disclosures_given": sorted(self.state.disclosures_given),
            "slots": self.state.slots,
            "qualification": self.state.decision,
            "escalation": self.state.escalation,
            "lead": self.lead,
            "latency_summary": self.latency.summary(),
            "turns": [t.as_dict() for t in self.state.turns],
        }

    def save_transcript(self) -> str:
        import json

        settings.transcripts_dir.mkdir(parents=True, exist_ok=True)
        path = settings.transcripts_dir / (self.state.call_id + ".json")
        path.write_text(json.dumps(self.transcript(), indent=2, ensure_ascii=False),
                        encoding="utf-8")
        log("session.transcript_saved", path=str(path))
        return str(path)
