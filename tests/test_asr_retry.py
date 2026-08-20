"""ASR resilience.

ASR is called once per utterance against a free tier capped at 20 requests per
minute, so a rate limit mid-call is expected rather than exceptional. Before
this, a 429 raised straight through and the caller was told "sorry, the line
broke up there" - a wrong explanation for a recoverable condition.

No network: the transport is stubbed.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from darwix.common import asr as asr_mod
from darwix.common.asr import ASRClient, ASRError


def _wav() -> bytes:
    from darwix.common.audio import to_wav_bytes
    return to_wav_bytes(b"\x00\x00" * 1600)


def _ok_body() -> dict:
    return {"text": "I run a textile business in Surat.", "language": "English",
            "segments": []}


class _Transport(httpx.AsyncBaseTransport):
    """Replays a scripted sequence of responses and counts attempts."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def handle_async_request(self, request):
        self.calls += 1
        item = self.script.pop(0) if self.script else self.script_default
        if isinstance(item, Exception):
            raise item
        status, headers = item
        body = _ok_body() if status < 400 else {"error": {"message": "rate limited"}}
        return httpx.Response(status, json=body, headers=headers or {},
                              request=request)


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    """Keep the tests quick; the wait itself is asserted separately."""
    monkeypatch.setattr(asr_mod, "ASR_BACKOFF_S", 0.001)
    monkeypatch.setattr(asr_mod, "ASR_MAX_BACKOFF_S", 0.005)


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setattr(asr_mod.settings, "groq_api_key", "test-key")


def _client(script) -> tuple[ASRClient, _Transport]:
    t = _Transport(script)
    c = ASRClient()
    c._client = httpx.AsyncClient(transport=t)
    return c, t


@pytest.mark.asyncio
async def test_rate_limit_is_retried_and_succeeds() -> None:
    client, transport = _client([(429, None), (429, None), (200, None)])
    tr = await client.transcribe_wav(_wav())
    assert transport.calls == 3
    assert tr.text == "I run a textile business in Surat."
    await client.aclose()


@pytest.mark.asyncio
async def test_transient_server_error_is_retried() -> None:
    client, transport = _client([(503, None), (200, None)])
    tr = await client.transcribe_wav(_wav())
    assert transport.calls == 2
    assert not tr.is_empty
    await client.aclose()


@pytest.mark.asyncio
async def test_network_error_is_retried() -> None:
    client, transport = _client([httpx.ConnectError("boom"), (200, None)])
    tr = await client.transcribe_wav(_wav())
    assert transport.calls == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_retries_are_bounded() -> None:
    """A permanently rate-limited call must give up, not hang the turn."""
    client, transport = _client([(429, None)] * 10)
    with pytest.raises(ASRError):
        await client.transcribe_wav(_wav())
    assert transport.calls == asr_mod.ASR_MAX_ATTEMPTS
    await client.aclose()


@pytest.mark.asyncio
async def test_client_errors_are_not_retried() -> None:
    """A bad key or unsupported model cannot be repaired by waiting, and
    retrying it just adds silence to the call."""
    for status in (400, 401, 404, 422):
        client, transport = _client([(status, None)] * 5)
        with pytest.raises(ASRError):
            await client.transcribe_wav(_wav())
        assert transport.calls == 1, f"status {status} should not be retried"
        await client.aclose()


@pytest.mark.asyncio
async def test_retry_after_header_is_honoured(monkeypatch) -> None:
    """Groq states the exact wait; guessing longer is silence for the caller."""
    monkeypatch.setattr(asr_mod, "ASR_MAX_BACKOFF_S", 10.0)
    waits: list[float] = []

    async def _record(seconds):
        waits.append(seconds)

    monkeypatch.setattr(asr_mod.asyncio, "sleep", _record)
    client, _ = _client([(429, {"retry-after": "2"}), (200, None)])
    await client.transcribe_wav(_wav())
    assert waits and abs(waits[0] - 2.0) < 1e-6, f"expected a 2 s wait, got {waits}"
    await client.aclose()


@pytest.mark.asyncio
async def test_a_recovered_turn_is_indistinguishable_from_a_clean_one() -> None:
    """The point of the retry: the caller never learns it happened."""
    client, _ = _client([(429, None), (200, None)])
    tr = await client.transcribe_wav(_wav())
    assert tr.provider == "groq" and tr.language == "English"
    assert not tr.is_empty
    await client.aclose()
