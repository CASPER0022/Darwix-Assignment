"""Speech recognition.

Primary: Groq-hosted `whisper-large-v3-turbo`. Whisper is used rather than a
telephony ASR because Q3 needs Tagalog and Indonesian *with* English code
switching in the same utterance; Whisper handles mixed-language audio in one
pass instead of forcing a language decision per turn.

Secondary: Gemini audio understanding. It exists so Q3 can report a real
provider comparison (accuracy, code-switching, regional accent) instead of an
opinion.

"Streaming" here means chunked pseudo-streaming: audio is cut at VAD
boundaries (or fixed windows for Q4) and each segment is transcribed as it
completes. True bidirectional streaming ASR is a paid feature at every
provider with a free tier; the tradeoff is documented in
docs/limitations_and_production_plan.md.
"""
from __future__ import annotations

import asyncio
import base64
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from .audio import to_wav_bytes
from .config import settings
from .llm import GEMINI_BASE, GROQ_BASE
from .logging import log


# Whisper does not return an empty string for silence. Trained on subtitle
# tracks, it emits the phrases that most often sit over quiet footage - and it
# does so with ordinary confidence, so no score threshold catches them. These
# are the artefacts observed from this model on silent and near-silent input;
# each is a complete utterance, which is why the match is anchored rather than
# a substring test: a caller really can say "thank you" as part of a sentence.
# Deliberately narrow. The energy gate in EnergyVAD/has_speech is the primary
# defence and stops silence reaching Whisper at all; this list is only the
# backstop for what slips past. So it must not contain a word a caller might
# actually say on its own - "yes", "no", "okay" and "correct" are whole answers
# to a consent or eligibility question, and swallowing them would break
# qualification far more often than a stray artefact ever did.
#
# "thank you" is the exception, kept in despite being sayable: it is by a wide
# margin the most frequent artefact from this model, and as a lone utterance
# mid-qualification it carries no information the flow acts on.
_HALLUCINATIONS = {
    "thank you", "thank you very much", "thank you so much",
    "thanks for watching", "thank you for watching", "thanks for listening",
    "please subscribe", "like and subscribe", "subscribe to my channel",
    "you", "the", "and", "so", "i", "a",
    "silence", "music", "applause", "background music", "beep",
    "amara.org", "subtitles by the amara.org community",
    "subtitles by the amara.org community.",
    "transcription by castingwords", "www.mooji.org",
    # The artefact is not English-specific. With no language forced - which is
    # what Taglish needs - Whisper picks a language for silence too, and the
    # subtitle credit comes back in whichever one it chose. Observed live in
    # this interface: a turn arrived as the Japanese "ご視聴ありがとうございました"
    # on a call whose locale was en-IN.
    "terima kasih", "terima kasih telah menonton",           # id-ID
    "salamat po", "salamat sa panonood",                     # fil-PH
    "ご視聴ありがとうございました", "ご視聴ありがとうございます",          # ja
    "ありがとうございました", "お疲れ様でした",
    "谢谢大家", "谢谢观看", "字幕由网友提供",                        # zh
    "시청해주셔서 감사합니다",                                    # ko
    "ขอบคุณครับ",                                            # th
    "спасибо за просмотр",                                   # ru
    "gracias por ver el video", "suscríbete al canal",       # es
    "obrigado por assistir",                                 # pt
    "merci d'avoir regardé cette vidéo",                     # fr
    "untertitel im auftrag des zdf",                         # de
    "hãy đăng ký kênh",                                      # vi
}
_PUNCT_ONLY = re.compile(r"^[\s\.,!?\-…’'\"]*$")
_BRACKETED = re.compile(r"^[\[\(\*][^\]\)\*]*[\]\)\*]$")


# Cyrillic, Arabic, Devanagari, Thai, Kana, CJK, Hangul.
_NON_LATIN = re.compile(
    r"[Ѐ-ӿ؀-ۿऀ-ॿ฀-๿"
    r"぀-ヿ㐀-䶿一-鿿가-힯]"
)


def looks_like_silence(text: str, *, script: str = "") -> bool:
    """True if `text` is a Whisper silence artefact rather than speech.

    `script` is the writing system the locale expects ("latin" for all three
    markets here). Given it, a short run of text in a completely different
    script counts as an artefact: enumerating every language Whisper might
    hallucinate in is a losing game, but a Japanese sentence on an en-IN call
    is structurally wrong whichever phrase it happens to be.
    """
    t = (text or "").strip()
    if not t or _PUNCT_ONLY.match(t):
        return True
    if _BRACKETED.match(t):           # "[BLANK_AUDIO]", "(soft music)", "*sighs*"
        return True
    normalised = re.sub(r"[\.,!?…]+$", "", t).strip().lower()
    if normalised in _HALLUCINATIONS:
        return True
    if script == "latin" and len(t) <= 40 and _NON_LATIN.search(t):
        # Length-bounded on purpose. A caller may genuinely code-switch, but
        # not for a whole sentence on a market whose flow, prompts and
        # knowledge base are Latin-script throughout - and these artefacts are
        # always short.
        return True
    return False


@dataclass
class Transcript:
    text: str
    language: str = ""
    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    segments: list[dict] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    script: str = "latin"

    @property
    def is_empty(self) -> bool:
        """No usable speech - either nothing came back, or what came back is a
        known silence artefact. Callers treat both the same way, so the two
        cases are collapsed here rather than at every call site."""
        return looks_like_silence(self.text, script=self.script)


class ASRError(RuntimeError):
    pass


# Three attempts, not more: on a live call the caller is listening to silence
# while this runs, and past ~8 s of total backoff admitting failure beats
# making them wait longer for an answer that may not come.
ASR_MAX_ATTEMPTS = 3
ASR_BACKOFF_S = 1.5
ASR_MAX_BACKOFF_S = 6.0


class ASRClient:
    """One client, two backends, one output shape."""

    def __init__(self, timeout: float = 60.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    async def _pause(attempt: int, response: "httpx.Response | None") -> None:
        """Wait before the next attempt, preferring what the server asked for.

        Groq answers a rate limit with the exact wait ("try again in 3s") in
        `retry-after`. Honouring it beats guessing: on a live call every extra
        second of backoff is silence the caller is listening to.
        """
        wait = ASR_BACKOFF_S * (2 ** attempt)
        if response is not None:
            hinted = response.headers.get("retry-after")
            if hinted:
                try:
                    wait = min(max(float(hinted), 0.1), ASR_MAX_BACKOFF_S)
                except ValueError:
                    pass
        await asyncio.sleep(min(wait, ASR_MAX_BACKOFF_S))

    async def transcribe_pcm(
        self,
        pcm: bytes,
        *,
        language: str | None = None,
        provider: str = "groq",
        prompt: str = "",
        sample_rate: int = 16_000,
    ) -> Transcript:
        return await self.transcribe_wav(
            to_wav_bytes(pcm, sample_rate),
            language=language,
            provider=provider,
            prompt=prompt,
        )

    async def transcribe_wav(
        self,
        wav_bytes: bytes,
        *,
        language: str | None = None,
        provider: str = "groq",
        prompt: str = "",
    ) -> Transcript:
        import time

        t0 = time.perf_counter()
        if provider == "gemini":
            tr = await self._gemini(wav_bytes, language, prompt)
        else:
            tr = await self._groq(wav_bytes, language, prompt)
        tr.latency_ms = (time.perf_counter() - t0) * 1000.0
        return tr

    async def _groq(self, wav_bytes: bytes, language: str | None, prompt: str) -> Transcript:
        settings.require("groq_api_key")
        data: dict[str, str] = {
            "model": settings.groq_asr_model,
            "response_format": "verbose_json",
            "temperature": "0",
        }
        # Leaving `language` unset lets Whisper auto-detect, which is what we
        # want for Taglish: forcing 'tl' degrades the English half of the turn.
        if language:
            data["language"] = language
        if prompt:
            # A domain prompt biases spelling of terms like 'cicilan', 'EMI',
            # 'bancassurance' that Whisper otherwise mangles.
            data["prompt"] = prompt[:800]

        # ASR is the most frequently called API on a live call - once per
        # utterance - against a free tier capped at 20 requests per minute.
        # Without a retry the first burst of turns raises straight through to
        # the caller as "sorry, the line broke up there", which is both a lie
        # about what happened and unrecoverable. The LLM path has had backoff
        # since the beginning; this one was simply missed.
        last: Exception | None = None
        for attempt in range(ASR_MAX_ATTEMPTS):
            try:
                r = await self._client.post(
                    GROQ_BASE + "/audio/transcriptions",
                    headers={"Authorization": "Bearer " + settings.groq_api_key},
                    files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                    data=data,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = ASRError("groq asr transport: " + str(exc)[:200])
                await self._pause(attempt, None)
                continue

            if r.status_code < 400:
                break

            last = ASRError("groq asr " + str(r.status_code) + ": " + r.text[:300])
            # 4xx other than 429 is a request the retry cannot repair - a bad
            # key, an unsupported model - so failing immediately is honest.
            if r.status_code != 429 and r.status_code < 500:
                raise last
            log("asr.retry", status=r.status_code, attempt=attempt + 1,
                of=ASR_MAX_ATTEMPTS)
            await self._pause(attempt, r)
        else:
            raise last or ASRError("groq asr: exhausted retries")

        payload = r.json()
        return Transcript(
            text=(payload.get("text") or "").strip(),
            language=payload.get("language", "") or (language or ""),
            provider="groq",
            model=settings.groq_asr_model,
            segments=payload.get("segments", []) or [],
            raw=payload,
        )

    async def _gemini(self, wav_bytes: bytes, language: str | None, prompt: str) -> Transcript:
        settings.require("gemini_api_key")
        hint = ""
        if language:
            hint = " The audio is mainly in " + language + ", possibly mixed with English."
        instruction = (
            "Transcribe this call audio verbatim." + hint + " Preserve code-switching "
            "exactly as spoken - do not translate any part of it. Return only the "
            "transcript text."
        )
        if prompt:
            instruction += " Domain vocabulary that may appear: " + prompt[:400]
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": instruction},
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": base64.b64encode(wav_bytes).decode(),
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048},
        }
        r = await self._client.post(
            GEMINI_BASE + "/models/" + settings.gemini_chat_model + ":generateContent",
            params={"key": settings.gemini_api_key},
            json=payload,
        )
        if r.status_code >= 400:
            raise ASRError("gemini asr " + str(r.status_code) + ": " + r.text[:300])
        data = r.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError):
            log("asr.gemini.unexpected", body=str(data)[:300])
            text = ""
        return Transcript(
            text=text,
            language=language or "",
            provider="gemini",
            model=settings.gemini_chat_model,
            raw=data,
        )


_shared: ASRClient | None = None


def get_asr() -> ASRClient:
    global _shared
    if _shared is None:
        _shared = ASRClient()
    return _shared
