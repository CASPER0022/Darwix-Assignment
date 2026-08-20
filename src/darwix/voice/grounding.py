"""Grounded answering.

The assessment lists "hallucinated answers" as a rejection condition, so this
is enforced in code rather than requested in a prompt:

1. **Retrieve first.** No retrieval, no factual claim. The agent answers a
   factual question only from records returned by the KB.

2. **Confidence gate.** If the fused retrieval score is below
   `RETRIEVAL_MIN_SCORE`, the agent does not get to try. It says it does not
   have that information and offers escalation. A weak-but-present chunk is
   exactly what produces a confident wrong answer.

3. **Citation contract.** The model must return the record ids it used. An
   answer that cites nothing while asserting something is rejected.

4. **One regeneration, then fall back.** If the check fails, the model gets one
   more attempt with the failure made explicit. If it fails again the agent
   says it does not know. It never gets a third try - on a live call the
   latency cost of a third round trip is worse than the honest answer.

5. **Numeric guard.** Any number the answer states must appear in the retrieved
   text. This catches the specific failure that matters most here: a model
   that reads "14.5% to 16.0%" and says "around 15%".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..common.config import settings
from ..common.llm import get_llm
from ..common.logging import log
from ..kb.retrieve import Hit, Retriever

ANSWER_SYSTEM = """You are answering a customer's question during a live phone call, on behalf of {brand}.

RULES - these are absolute:
- Answer ONLY from the SOURCES below. If the sources do not contain the answer, say so.
- Never state a number, rate, fee, threshold or timeline that is not written in the sources.
- Never promise approval, sanction, disbursal or a specific interest rate.
- Speak like a person on a phone: at most 2 short sentences. No lists, no headings.
- Reply in {language_name}, matching the customer's register.

Return JSON only:
{{"answer": "<what you would say out loud>",
  "used_record_ids": ["<ids of the sources you actually used>"],
  "answered": true|false}}

Set "answered": false and leave used_record_ids empty if the sources do not cover the question."""


@dataclass
class GroundedAnswer:
    text: str
    answered: bool
    hits: list[Hit] = field(default_factory=list)
    used_record_ids: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    rejected_reason: str = ""
    attempts: int = 0
    top_score: float = 0.0

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "answered": self.answered,
            "top_score": round(self.top_score, 3),
            "used_record_ids": self.used_record_ids,
            "citations": self.citations,
            "rejected_reason": self.rejected_reason,
            "attempts": self.attempts,
            "retrieved": [h.record_id for h in self.hits],
        }


NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
# Numbers that are part of speech, not claims ("two or three minutes", "one")
SAFE_NUMBERS = {"1", "2", "3", "one", "two", "three"}


def _numbers_supported(answer: str, sources: str) -> list[str]:
    """Return numbers asserted in the answer that do not occur in the sources."""
    src = sources.replace(",", "")
    bad: list[str] = []
    for raw in NUMBER.findall(answer):
        token = raw.replace(",", "")
        if token in SAFE_NUMBERS:
            continue
        if token in src:
            continue
        # allow "14.5" to match "14.5%" and "5000000" to match "50,00,000"
        if token.rstrip("0").rstrip(".") and token.rstrip("0").rstrip(".") in src:
            continue
        bad.append(raw)
    return bad


async def answer_question(
    question: str,
    retriever: Retriever,
    *,
    language: str = "en",
    language_name: str = "English",
    brand: str = "the lender",
    market: str = "IN",
    categories: list[str] | None = None,
    fallback_text: str = "",
) -> GroundedAnswer:
    hits = await retriever.search(question, language=language, categories=categories,
                                  market=market)
    top = hits[0].score if hits else 0.0

    if not retriever.is_confident(hits):
        log("grounding.below_threshold", question=question[:120], top_score=round(top, 3),
            threshold=settings.retrieval_min_score)
        return GroundedAnswer(
            text=fallback_text,
            answered=False,
            hits=hits,
            rejected_reason="retrieval_below_threshold",
            top_score=top,
        )

    sources_block = "\n\n".join(
        "[" + h.record_id + "] " + h.title + "\n" + h.content[:1200] for h in hits
    )
    system = ANSWER_SYSTEM.format(brand=brand, language_name=language_name)
    llm = get_llm()
    messages = [{"role": "user", "content":
                 "SOURCES:\n" + sources_block + "\n\nCUSTOMER QUESTION: " + question}]

    reason = ""
    for attempt in range(2):
        try:
            data = await llm.chat_json(system, messages, provider="groq_dialog",
                                       temperature=0.2, max_tokens=400)
        except Exception as exc:  # noqa: BLE001
            log("grounding.llm_failed", error=str(exc)[:200])
            return GroundedAnswer(text=fallback_text, answered=False, hits=hits,
                                  rejected_reason="llm_error", attempts=attempt + 1, top_score=top)

        text = (data.get("answer") or "").strip()
        used = [r for r in (data.get("used_record_ids") or []) if isinstance(r, str)]
        answered = bool(data.get("answered")) and bool(text)

        if not answered:
            return GroundedAnswer(text=fallback_text, answered=False, hits=hits,
                                  rejected_reason="model_declined", attempts=attempt + 1,
                                  top_score=top)

        valid_ids = {h.record_id for h in hits}
        used = [u for u in used if u in valid_ids]
        unsupported = _numbers_supported(text, sources_block)

        if not used:
            reason = "no_citation"
        elif unsupported:
            reason = "unsupported_numbers:" + ",".join(unsupported[:4])
        else:
            cited = [h for h in hits if h.record_id in used]
            log("grounding.answered", question=question[:100], records=used,
                top_score=round(top, 3), attempt=attempt + 1)
            return GroundedAnswer(
                text=text, answered=True, hits=hits, used_record_ids=used,
                citations=[h.citation() for h in cited], attempts=attempt + 1, top_score=top,
            )

        log("grounding.rejected", reason=reason, attempt=attempt + 1, draft=text[:160])
        messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content":
             "That answer was rejected: " + reason + ". Use only what the SOURCES state, "
             "list the record ids you used, and do not state any number that is not written "
             "in the sources. If the sources do not cover it, set answered to false."},
        ]

    return GroundedAnswer(text=fallback_text, answered=False, hits=hits,
                          rejected_reason=reason or "verification_failed", attempts=2,
                          top_score=top)
