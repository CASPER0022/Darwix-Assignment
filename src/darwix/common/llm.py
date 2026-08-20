"""LLM + embedding clients.

Provider selection is measured, not assumed. The benchmark that produced these
choices is in evaluation/model_selection.md; the short version:

* `groq_dialog` -> openai/gpt-oss-120b @ reasoning_effort=low. Median 923 ms
  and it answered in the customer's own language and register in every test,
  including Taglish and Bahasa. This is the voice agent's brain (Q1, Q3).
* `groq_fast`   -> openai/gpt-oss-20b @ reasoning_effort=low. Median ~810 ms
  for short structured JSON. Q4 signal extraction, where the job is
  classification rather than conversation.
* `gemini`      -> gemini-3.6-flash. Measured at 16-38 s per turn on this free
  tier (it is a thinking model, and it returned a 503 under load), so it is
  used ONLY off the latency path: KB classification, the ASR benchmark, and
  offline localisation review. Even there, `thinkingLevel: minimal` is set,
  because thinking tokens are billed against maxOutputTokens and silently
  truncate the answer otherwise.

Both providers are called over plain REST with httpx: no vendor SDK to pin,
one retry path, one timeout policy, and every failure mode visible in this file.

Note on gpt-oss: reasoning is returned on a separate channel and the visible
`content` comes back EMPTY when the reasoning consumes the token budget. That
is why `reasoning_effort` is always sent and why an empty completion is treated
as a failure worth retrying rather than as a valid silent answer.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Literal, Sequence

import httpx

from .config import settings
from .logging import log

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GROQ_BASE = "https://api.groq.com/openai/v1"

Provider = Literal["gemini", "groq_fast", "groq_dialog"]


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> Any:
    """Models occasionally wrap JSON in prose or fences. Recover rather than
    fail the turn -- a live call cannot afford an exception here."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError("Could not parse JSON from model output: " + repr(text[:300]))


class LLMClient:
    def __init__(self, timeout: float = 30.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ chat
    async def chat(
        self,
        system: str,
        messages: Sequence[dict],
        *,
        provider: Provider = "groq_dialog",
        temperature: float = 0.3,
        max_tokens: int = 800,
        json_mode: bool = False,
        retries: int = 2,
    ) -> str:
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                if provider == "gemini":
                    return await self._gemini_chat(
                        system, messages, temperature, max_tokens, json_mode
                    )
                model = (
                    settings.groq_dialog_model
                    if provider == "groq_dialog"
                    else settings.groq_fast_model
                )
                return await self._groq_chat(
                    system, messages, temperature, max_tokens, json_mode, model
                )
            except Exception as exc:  # noqa: BLE001 - deliberate: keep the call alive
                last = exc
                log("llm.retry", provider=provider, attempt=attempt, error=str(exc)[:200])
                await asyncio.sleep(0.4 * (attempt + 1))
        raise LLMError(
            provider + " chat failed after " + str(retries + 1) + " attempts: " + str(last)
        )

    async def chat_json(self, system: str, messages: Sequence[dict], **kw: Any) -> Any:
        kw.setdefault("json_mode", True)
        return extract_json(await self.chat(system, messages, **kw))

    async def _gemini_chat(
        self,
        system: str,
        messages: Sequence[dict],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        settings.require("gemini_api_key")
        contents = [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
        ]
        gen_config: dict[str, Any] = {
            "temperature": temperature,
            # thinking tokens are charged against this budget, so the headroom
            # is deliberate - without it the answer comes back truncated
            "maxOutputTokens": max_tokens + 1024,
            "thinkingConfig": {"thinkingLevel": settings.gemini_thinking_level},
        }
        if json_mode:
            gen_config["responseMimeType"] = "application/json"
        payload: dict[str, Any] = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": gen_config,
        }
        r = await self._client.post(
            GEMINI_BASE + "/models/" + settings.gemini_chat_model + ":generateContent",
            params={"key": settings.gemini_api_key},
            json=payload,
        )
        if r.status_code >= 400:
            raise LLMError("gemini " + str(r.status_code) + ": " + r.text[:300])
        data = r.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(
                "unexpected gemini response: " + json.dumps(data)[:300]
            ) from exc

    async def _groq_chat(
        self,
        system: str,
        messages: Sequence[dict],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        model: str,
    ) -> str:
        settings.require("groq_api_key")
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}] + list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if "gpt-oss" in model:
            # Without this the model spends the whole budget reasoning and
            # returns an empty `content` - measured, not theoretical.
            payload["reasoning_effort"] = settings.groq_reasoning_effort
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        r = await self._client.post(
            GROQ_BASE + "/chat/completions",
            headers={"Authorization": "Bearer " + settings.groq_api_key},
            json=payload,
        )
        if r.status_code >= 400:
            raise LLMError("groq " + str(r.status_code) + ": " + r.text[:300])
        content = (r.json()["choices"][0]["message"].get("content") or "").strip()
        if not content:
            raise LLMError("groq returned an empty completion (reasoning consumed the budget)")
        return content

    # ------------------------------------------------------------- embeddings
    async def embed(
        self,
        texts: Sequence[str],
        *,
        task: str = "RETRIEVAL_DOCUMENT",
        concurrency: int = 6,
    ) -> list[list[float]]:
        """Gemini embeddings.

        `gemini-embedding-001` exposes `embedContent` (one text per call) and an
        async batch job API; it has no synchronous batch endpoint. For a corpus
        this size, bounded-concurrency single calls are simpler and faster than
        submitting and polling a batch job, so that is what this does.

        `task` matters: embedding documents as RETRIEVAL_DOCUMENT and queries as
        RETRIEVAL_QUERY is asymmetric retrieval, and it measurably beats using
        one task type for both.

        Output is pinned to `settings.embed_dimensions` (768). The model's
        native 3072 dims cost 4x the memory for no measurable gain on a few
        hundred records, and 768 keeps the matrix small enough to hold in the
        call server.
        """
        settings.require("gemini_api_key")
        sem = asyncio.Semaphore(concurrency)
        results: list[list[float] | None] = [None] * len(texts)
        url = GEMINI_BASE + "/models/" + settings.gemini_embed_model + ":embedContent"

        async def one(i: int, text: str) -> None:
            payload = {
                "model": "models/" + settings.gemini_embed_model,
                "content": {"parts": [{"text": text}]},
                "taskType": task,
                "outputDimensionality": settings.embed_dimensions,
            }
            async with sem:
                for attempt in range(6):
                    try:
                        r = await self._client.post(
                            url,
                            params={"key": settings.gemini_api_key},
                            json=payload,
                            timeout=60.0,
                        )
                        if r.status_code < 400:
                            results[i] = r.json()["embedding"]["values"]
                            return
                        # 429 is the free tier's per-minute rate limit, not a
                        # failure: back off exponentially past the window
                        wait = 8.0 * (2 ** attempt) if r.status_code == 429 else 1.5 * (attempt + 1)
                        log("embed.retry", index=i, status=r.status_code,
                            attempt=attempt, body=r.text[:160])
                    except Exception as exc:  # noqa: BLE001
                        wait = 2.0 * (attempt + 1)
                        log("embed.retry", index=i, attempt=attempt, error=str(exc)[:160])
                    await asyncio.sleep(wait)
                raise LLMError("embedding failed for item " + str(i))

        await asyncio.gather(*(one(i, t) for i, t in enumerate(texts)))
        return [r for r in results if r is not None]


_shared: LLMClient | None = None


def get_llm() -> LLMClient:
    global _shared
    if _shared is None:
        _shared = LLMClient()
    return _shared
