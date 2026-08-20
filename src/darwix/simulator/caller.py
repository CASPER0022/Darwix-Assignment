"""Synthetic caller harness.

Every recorded test call in this repo was produced here. The customer side is
generated and spoken by TTS, then fed into the real pipeline as audio: VAD,
Whisper, retrieval, grounding and the agent's own TTS all run exactly as they
do for a human caller. Nothing is stubbed, and the transcripts are what the
system actually heard - not what the script said.

Why generate the customer instead of hard-coding lines:

* A fixed script only tests the happy path. If the agent asks something in a
  different order, a scripted caller answers the wrong question and the test
  silently stops being a test.
* Q3 needs Taglish, Bahasa and a Javanese-accented speaker. Those calls have to
  be spoken by *someone*, and a synthetic caller with a native TTS voice is an
  honest, reproducible stand-in for a native speaker - with the limitation
  stated plainly in the Q3 write-up.

Control is kept where it matters: a persona declares `beats`, which are
mandatory utterances injected at a given turn (the objection, the request for a
human, the out-of-scope question). Those guarantee the assessment's required
test coverage; the rest of the conversation is improvised from the persona's
facts, so the agent is genuinely tested rather than replayed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..common.audio import SAMPLE_RATE, duration_seconds
from ..common.config import REPO_ROOT, settings
from ..common.llm import get_llm
from ..common.logging import log
from ..common.tts import get_tts
from ..voice.session import CallSession
from ..voice.turn_manager import AudioCall

PERSONA_DIR = Path(__file__).parent / "personas"
MAX_TURNS = 16

CUSTOMER_SYSTEM = """You are role-playing a CUSTOMER on a phone call. You are not an assistant.

WHO YOU ARE:
{persona}

HOW YOU SPEAK:
{style}

FACTS YOU KNOW ABOUT YOURSELF (use these when asked, never contradict them):
{facts}

RULES:
- Reply with ONE short spoken turn, the way a real person answers a phone call.
- Answer the question you were actually asked. Do not volunteer everything at once.
- Never break character, never mention that you are an AI, never use markdown.
- Speak {language}. {code_switch}
- If you have already been told the call is over, say a brief goodbye.

Return JSON only: {{"say": "<your spoken line>", "done": false}}
Set "done": true only when the conversation has genuinely finished."""


@dataclass
class Persona:
    id: str
    locale: str
    voice: str
    scenario: str
    description: str
    persona: str
    style: str
    language: str
    code_switch: str = ""
    facts: dict[str, Any] = field(default_factory=dict)
    beats: dict[int, str] = field(default_factory=dict)
    expect: dict[str, Any] = field(default_factory=dict)
    max_turns: int = 12

    @classmethod
    def load(cls, path: Path) -> "Persona":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["beats"] = {int(k): v for k, v in (raw.get("beats") or {}).items()}
        return cls(**raw)


async def _customer_line(persona: Persona, agent_text: str, history: list[dict]) -> tuple[str, bool]:
    system = CUSTOMER_SYSTEM.format(
        persona=persona.persona,
        style=persona.style,
        facts=json.dumps(persona.facts, ensure_ascii=False, indent=2),
        language=persona.language,
        code_switch=persona.code_switch,
    )
    messages = history[-8:] + [{"role": "user", "content": "AGENT SAID: " + agent_text}]
    try:
        # The customer runs on the small fast model. It was on Gemini until
        # the free-tier daily quota ran out mid-run and every customer turn
        # silently degraded to "could you repeat that" - which is exactly the
        # kind of failure that makes a test suite lie to you. Groq's small
        # model is plentiful, and the persona work is easy enough for it.
        data = await get_llm().chat_json(system, messages, provider="groq_fast",
                                         temperature=0.8, max_tokens=180)
        return str(data.get("say", "")).strip(), bool(data.get("done"))
    except Exception as exc:  # noqa: BLE001
        log("sim.customer_failed", error=str(exc)[:200])
        return "Sorry, could you repeat that?", False


async def run_call(persona: Persona, *, out_prefix: str = "") -> dict:
    call_id = (out_prefix or "sim") + "_" + persona.id
    session = CallSession(persona.locale, call_id=call_id, customer_phone="the number on file")
    events: list[dict] = []

    async def on_event(payload: dict) -> None:
        events.append(payload)

    async def on_audio(_: bytes) -> None:
        # The agent's audio is already captured into the recording track by
        # AudioCall; the simulator has no speaker to play it to.
        return

    call = AudioCall(session=session, on_event=on_event, on_audio=on_audio)
    tts = get_tts()
    history: list[dict] = []

    log("sim.start", persona=persona.id, locale=persona.locale, scenario=persona.scenario)
    await call.open()

    turn = 0
    while turn < min(persona.max_turns, MAX_TURNS) and not session.state.ended:
        agent_text = next((t.text for t in reversed(session.state.turns)
                           if t.speaker == "agent"), "")
        # A mandatory beat overrides the improvised line for this turn.
        if turn in persona.beats:
            line, done = persona.beats[turn], False
        else:
            line, done = await _customer_line(persona, agent_text, history)
        if not line:
            break

        history.append({"role": "user", "content": "AGENT SAID: " + agent_text})
        history.append({"role": "assistant", "content": json.dumps({"say": line, "done": done})})

        # Speak the customer's line, then feed it into the pipeline as audio.
        pcm = await tts.synthesize(line, persona.voice)
        log("sim.customer_says", turn=turn, text=line[:100],
            audio_s=round(duration_seconds(pcm), 2))
        await _feed(call, pcm)

        turn += 1
        if done:
            break

    artefacts = call.finalise()
    summary = {
        "persona": persona.id,
        "scenario": persona.scenario,
        "description": persona.description,
        "locale": persona.locale,
        "customer_voice": persona.voice,
        "call_id": call_id,
        "turns": turn,
        "ended_reason": session.state.ended_reason or "max_turns",
        "disposition": (session.state.decision or {}).get("disposition"),
        "escalation": session.state.escalation,
        "slots": session.state.slots,
        "latency": session.latency.summary(),
        "expectations": _check(persona, session, events),
        **artefacts,
    }
    log("sim.done", **{k: v for k, v in summary.items() if k in
                       ("persona", "turns", "ended_reason", "disposition")})
    return summary


async def _feed(call: AudioCall, pcm: bytes, *, chunk_ms: int = 100) -> None:
    """Stream the utterance in real-time-sized chunks, then a tail of silence.

    The silence matters: it is what the VAD hangover needs to declare the turn
    finished, exactly as a real caller pausing would.
    """
    frame = int(SAMPLE_RATE * (chunk_ms / 1000.0)) * 2
    for i in range(0, len(pcm), frame):
        await call.push_audio(pcm[i:i + frame])
    silence = b"\x00" * (int(SAMPLE_RATE * 1.0) * 2)
    for i in range(0, len(silence), frame):
        await call.push_audio(silence[i:i + frame])
    # give the pipeline a moment to finish the turn it just started
    for _ in range(200):
        if not call.processing:
            break
        await asyncio.sleep(0.05)


def _check(persona: Persona, session: CallSession, events: list[dict]) -> dict:
    """Assert the things the assessment's required coverage cares about."""
    exp = persona.expect or {}
    transcript = " ".join(t.text.lower() for t in session.state.turns if t.speaker == "agent")
    grounded_turns = [t for t in session.state.turns if t.grounded]
    results: dict[str, Any] = {}

    if "escalated" in exp:
        results["escalated"] = {
            "expected": exp["escalated"],
            "actual": bool(session.state.escalation),
            "pass": bool(session.state.escalation) == bool(exp["escalated"]),
        }
    if "disposition" in exp:
        actual = (session.state.decision or {}).get("disposition")
        results["disposition"] = {"expected": exp["disposition"], "actual": actual,
                                  "pass": actual == exp["disposition"]}
    if "soft_refusal_recorded" in exp:
        # Checked separately from `disposition` on purpose. The disposition is
        # the end of a chain the generated customer can change; whether an
        # indirect refusal was *heard at all* is the invariant this scenario
        # exists to test, so a failure points at the right place.
        recorded = bool(session.state.soft_refusals)
        results["soft_refusal_recorded"] = {
            "expected": exp["soft_refusal_recorded"], "actual": recorded,
            "pass": recorded == exp["soft_refusal_recorded"],
            "markers": list(session.state.soft_refusals),
        }
    if "declined_to_answer" in exp:
        # Declining takes two forms and both count: a grounded answer rejected
        # by the confidence gate, and the out-of-scope / unknown fallback line
        # spoken directly. The first version of this check only looked at
        # grounded turns and marked a correctly-behaving agent as a failure.
        declined = any(t.grounded and not t.grounded.get("answered") for t in grounded_turns)
        fallbacks = [session.pack["fallback"][k][:40].lower().strip()
                     for k in ("out_of_scope", "unknown", "low_confidence")]
        spoken_fallback = any(
            any(f and f in " ".join(t.text.lower().split()) for f in fallbacks)
            for t in session.state.turns if t.speaker == "agent"
        )
        declined = declined or spoken_fallback
        results["declined_to_answer"] = {"expected": exp["declined_to_answer"],
                                         "actual": declined,
                                         "pass": declined == exp["declined_to_answer"]}
    if "grounded_answer" in exp:
        answered = any(t.grounded and t.grounded.get("answered") for t in grounded_turns)
        results["grounded_answer"] = {"expected": exp["grounded_answer"], "actual": answered,
                                      "pass": answered == exp["grounded_answer"]}
    if "disclosures" in exp:
        given = sorted(session.state.disclosures_given)
        results["disclosures"] = {"expected": exp["disclosures"], "actual": given,
                                  "pass": all(d in given for d in exp["disclosures"])}
    results["all_pass"] = all(v.get("pass") for v in results.values() if isinstance(v, dict))
    return results


async def run_many(persona_ids: list[str], *, prefix: str = "") -> list[dict]:
    out: list[dict] = []
    for pid in persona_ids:
        path = PERSONA_DIR / (pid + ".yaml")
        if not path.exists():
            log("sim.persona_missing", persona=pid)
            continue
        summary = await run_call(Persona.load(path), out_prefix=prefix)
        out.append(summary)
        await asyncio.sleep(2.0)  # stay inside the free-tier rate limits
    results_path = settings.transcripts_dir / "simulation_results.json"
    existing = []
    if results_path.exists():
        existing = json.loads(results_path.read_text(encoding="utf-8"))
    merged = {r["persona"]: r for r in existing}
    for r in out:
        merged[r["persona"]] = r
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(list(merged.values()), indent=2, ensure_ascii=False),
                            encoding="utf-8")
    return out


def available() -> list[str]:
    return sorted(p.stem for p in PERSONA_DIR.glob("*.yaml"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run scripted synthetic calls against the live agent")
    ap.add_argument("personas", nargs="*", help="persona ids (default: all)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    if args.list:
        for p in available():
            print(p)
        raise SystemExit(0)

    ids = args.personas or available()
    summaries = asyncio.run(run_many(ids, prefix=args.prefix))
    print()
    for s in summaries:
        checks = s["expectations"]
        flag = "PASS" if checks.get("all_pass") else "CHECK"
        print(f"[{flag}] {s['persona']:28s} turns={s['turns']:2d} "
              f"end={s['ended_reason']:28s} {s.get('disposition') or ''}")
