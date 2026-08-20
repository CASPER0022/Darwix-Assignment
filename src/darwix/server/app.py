"""FastAPI application: the web calling interface and the KB search UI.

Q1 asks for "a callable number or web calling interface". This is the second
option, self-hosted: the browser captures the microphone, streams 16 kHz PCM
over a WebSocket, and plays the agent's audio back as it arrives.

Choosing this over a managed platform (Vapi/Retell) was a constraint-driven
decision - no paid account - but it also means the whole path is inspectable:
VAD, endpointing, ASR, retrieval, grounding and TTS are all in this repo and
all measurable. The tradeoff (no PSTN number) is stated in the README.

Endpoints:
    GET  /                  landing page, links to every surface
    GET  /webcall           browser call UI
    GET  /kb                knowledge-base search UI (Q2, standalone proof)
    GET  /dashboard         live nudge dashboard (Q4)
    GET  /api/kb/search     retrieval as JSON, with citations and scores
    WS   /ws/call           the voice call
    WS   /ws/nudges         live nudge stream (Q4)
    GET  /api/health        readiness, including whether keys and index exist
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ..common.audio import SPEECH_RMS_FLOOR
from ..common.config import REPO_ROOT, settings
from ..common.logging import log
from ..kb.retrieve import Retriever, get_retriever
from ..voice.session import CallSession
from ..voice.turn_manager import AudioCall

WEB_DIR = REPO_ROOT / "web"

app = FastAPI(title="Darwix AI assessment", version="1.0.0")

_retriever: Retriever | None = None


def retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


# --------------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/webcall", response_class=HTMLResponse)
async def webcall() -> str:
    return (WEB_DIR / "webcall" / "index.html").read_text(encoding="utf-8")


@app.get("/kb", response_class=HTMLResponse)
async def kb_ui() -> str:
    return (WEB_DIR / "kb" / "index.html").read_text(encoding="utf-8")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    return (WEB_DIR / "dashboard" / "index.html").read_text(encoding="utf-8")


# ----------------------------------------------------------------------- api
@app.get("/api/health")
async def health() -> JSONResponse:
    ok_index = settings.index_path.exists()
    return JSONResponse({
        "status": "ok" if (ok_index and settings.groq_api_key) else "degraded",
        "index_present": ok_index,
        "embeddings_present": (settings.kb_dir / "embeddings.npy").exists(),
        "gemini_key": bool(settings.gemini_api_key),
        "groq_key": bool(settings.groq_api_key),
        "records": len(retriever().rows) if ok_index else 0,
        "dialog_model": settings.groq_dialog_model,
        "asr_model": settings.groq_asr_model,
        # Published so the browser does not carry its own copy of a number that
        # has to match the server's. The two drifting apart would be silent:
        # the meter would call audio "speech" that the VAD then ignores.
        "speech_rms_floor": SPEECH_RMS_FLOOR,
        "barge_in_rms": settings.barge_in_rms,
    })


@app.get("/api/kb/search")
async def kb_search(
    q: str = Query(..., min_length=2),
    k: int = Query(4, ge=1, le=10),
    language: str = Query("en"),
) -> JSONResponse:
    r = retriever()
    hits = await r.search(q, top_k=k, language=language)
    return JSONResponse({
        "query": q,
        "confident": r.is_confident(hits),
        "threshold": settings.retrieval_min_score,
        "results": [
            {**h.as_dict(),
             "lexical_rank": h.lexical_rank,
             "dense_rank": h.dense_rank,
             "matched_terms": h.debug.get("matched_terms", [])}
            for h in hits
        ],
    })


@app.get("/api/kb/stats")
async def kb_stats() -> JSONResponse:
    path = settings.kb_dir / "build_stats.json"
    if not path.exists():
        return JSONResponse({"error": "no build stats; run: python -m darwix.kb.build"}, 404)
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


# ------------------------------------------------------------------ the call
@app.websocket("/ws/call")
async def ws_call(ws: WebSocket) -> None:
    await ws.accept()
    locale = ws.query_params.get("locale", "en-IN")
    lock = asyncio.Lock()

    async def send_event(payload: dict) -> None:
        async with lock:
            try:
                await ws.send_text(json.dumps(payload, ensure_ascii=False, default=str))
            except Exception:  # noqa: BLE001 - client vanished mid-turn
                pass

    async def send_audio(pcm: bytes) -> None:
        async with lock:
            try:
                await ws.send_bytes(pcm)
            except Exception:  # noqa: BLE001
                pass

    try:
        session = CallSession(locale, retriever=retriever())
    except Exception as exc:  # noqa: BLE001
        await send_event({"type": "error", "message": str(exc)[:300]})
        await ws.close()
        return

    call = AudioCall(session=session, on_event=send_event, on_audio=send_audio)
    await send_event({"type": "call_started", "call_id": session.state.call_id,
                      "locale": locale, "sample_rate": 16000})
    log("ws.call_started", call_id=session.state.call_id, locale=locale)
    await call.open()

    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is not None:
                await call.push_audio(data)
            elif (text := message.get("text")) is not None:
                payload = json.loads(text)
                if payload.get("type") == "text":
                    # Text fallback: lets the interface be driven without a mic,
                    # and is the only usable path when the browser blocks
                    # getUserMedia. It therefore has to behave like the audio
                    # path, not like a lesser version of it.
                    typed = (payload.get("text") or "").strip()
                    if not typed:
                        continue
                    # Echo before handling, not after. session.handle() is a
                    # model call, so echoing afterwards left the customer's own
                    # message invisible for seconds - which reads exactly like
                    # the input having been swallowed.
                    await send_event({"type": "transcript", "speaker": "customer",
                                      "text": typed})
                    await send_event({"type": "thinking"})
                    t_turn = time.perf_counter()
                    result = await session.handle(typed)
                    if result.text:
                        # Same metadata the audio path emits. Without `grounded`
                        # the citation panel stayed empty for typed turns even
                        # when the answer came straight out of the KB, which
                        # made a grounded system look ungrounded.
                        await call.speak(result.text, meta={
                            "intent": result.intent,
                            "phase": session.state.phase.value,
                            "grounded": result.grounded,
                            "think_ms": round(result.latency_ms or
                                              (time.perf_counter() - t_turn) * 1000),
                        })
                    if result.ended:
                        await send_event({"type": "call_ended",
                                          "reason": session.state.ended_reason})
                elif payload.get("type") == "hangup":
                    break
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        log("ws.call_error", error=str(exc)[:300])
    finally:
        artefacts = call.finalise()
        log("ws.call_finished", call_id=session.state.call_id, **artefacts)
        await send_event({"type": "artefacts", **artefacts})
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


@app.get("/api/recordings")
async def recordings() -> JSONResponse:
    """List analysable call recordings, newest first."""
    out = []
    for wav in sorted(settings.recordings_dir.glob("*.wav"),
                      key=lambda p: -p.stat().st_mtime):
        truth = wav.with_suffix(".truth.json")
        row = {"id": wav.stem, "bytes": wav.stat().st_size}
        if truth.exists():
            meta = json.loads(truth.read_text(encoding="utf-8"))
            row["seconds"] = meta.get("seconds")
            row["description"] = meta.get("description", "").strip()
            row["expected_nudges"] = meta.get("expected_nudges", [])
        out.append(row)
    return JSONResponse({"recordings": out})


@app.websocket("/ws/nudges")
async def ws_nudges(ws: WebSocket) -> None:
    """Live nudge stream.

    The client asks for a recording to be analysed; the server replays it at
    real-time speed and pushes transcript, signal and nudge events as they
    happen. The dashboard therefore shows exactly what an agent would see
    mid-call, not a summary produced afterwards.
    """
    await ws.accept()
    lock = asyncio.Lock()

    async def sink(payload: dict) -> None:
        async with lock:
            try:
                await ws.send_text(json.dumps(payload, ensure_ascii=False, default=str))
            except Exception:  # noqa: BLE001
                pass

    task: asyncio.Task | None = None
    try:
        while True:
            message = await ws.receive_text()
            payload = json.loads(message)
            if payload.get("action") != "analyse":
                continue
            if task and not task.done():
                await sink({"type": "error", "message": "an analysis is already running"})
                continue
            wav = settings.recordings_dir / (payload.get("id", "") + ".wav")
            if not wav.exists():
                await sink({"type": "error", "message": "no such recording: " + wav.name})
                continue

            from ..realtime.pipeline import analyse_file

            task = asyncio.create_task(analyse_file(
                wav,
                call_id=wav.stem,
                speed=float(payload.get("speed", 1.0)),
                use_llm=bool(payload.get("use_llm", True)),
                sinks=[sink],
            ))
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        log("ws.nudges_error", error=str(exc)[:250])
    finally:
        if task and not task.done():
            task.cancel()


def main() -> None:
    import uvicorn

    uvicorn.run("darwix.server.app:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
