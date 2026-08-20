"""Render a scripted two-party call to a stereo recording.

Q4 has to be shown catching an agent's mistakes: a skipped disclosure, a
promise that should never be made, a cross-sell cue that goes unanswered. Our
own Q1 agent cannot produce those recordings, because the disclosure gate and
the grounding check make it compliant by construction - which is the point of
Q1, and useless for testing Q4.

So these calls are scripted end to end and spoken by TTS: a deliberately
imperfect human agent on the left channel, a customer on the right. The audio is
real, the channel layout matches a contact-centre recorder, and the Q4 pipeline
has no idea it was scripted - it receives exactly what it would receive from a
live call.

Timing is preserved: each turn is placed sequentially with a short gap, and the
other channel is padded with silence, so replaying the file at real-time speed
reproduces the real cadence of the conversation.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import yaml

from ..common.audio import SAMPLE_RATE, mix_stereo, write_wav
from ..common.config import settings
from ..common.logging import log
from ..common.tts import get_tts

SCENARIO_DIR = Path(__file__).parent / "scenarios"
GAP_MS = 350


async def render(scenario_path: Path) -> dict:
    spec = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    tts = get_tts()
    agent_track = bytearray()
    customer_track = bytearray()
    gap = b"\x00" * int(SAMPLE_RATE * (GAP_MS / 1000.0)) * 2
    script: list[dict] = []

    for turn in spec["turns"]:
        speaker = turn["speaker"]
        text = turn["text"]
        voice = turn.get("voice") or (
            spec["agent_voice"] if speaker == "agent" else spec["customer_voice"]
        )
        pcm = await tts.synthesize(text, voice)
        silence = b"\x00" * len(pcm)
        if speaker == "agent":
            agent_track.extend(pcm)
            customer_track.extend(silence)
        else:
            customer_track.extend(pcm)
            agent_track.extend(silence)
        agent_track.extend(gap)
        customer_track.extend(gap)
        script.append({"speaker": speaker, "text": text, "voice": voice})
        # an explicit pause, e.g. the agent thinking, or dead air
        if turn.get("pause_s"):
            pad = b"\x00" * int(SAMPLE_RATE * float(turn["pause_s"])) * 2
            agent_track.extend(pad)
            customer_track.extend(pad)

    out = settings.recordings_dir / (spec["id"] + ".wav")
    write_wav(out, mix_stereo(bytes(agent_track), bytes(customer_track)), channels=2)
    seconds = len(agent_track) / (2 * SAMPLE_RATE)
    log("scenario.rendered", id=spec["id"], path=str(out), seconds=round(seconds, 1),
        turns=len(script))

    # the ground truth the false-positive analysis is scored against
    truth_path = settings.recordings_dir / (spec["id"] + ".truth.json")
    import json

    truth_path.write_text(json.dumps({
        "id": spec["id"],
        "description": spec["description"],
        "expected_nudges": spec.get("expected_nudges", []),
        "must_not_nudge": spec.get("must_not_nudge", []),
        "max_nudges": spec.get("max_nudges"),
        "script": script,
        "seconds": round(seconds, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"id": spec["id"], "wav": str(out), "seconds": round(seconds, 1),
            "turns": len(script)}


async def render_all(ids: list[str] | None = None) -> list[dict]:
    paths = ([SCENARIO_DIR / (i + ".yaml") for i in ids] if ids
             else sorted(SCENARIO_DIR.glob("*.yaml")))
    out = []
    for p in paths:
        if not p.exists():
            log("scenario.missing", path=str(p))
            continue
        out.append(await render(p))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render scripted calls to stereo WAV")
    ap.add_argument("scenarios", nargs="*")
    args = ap.parse_args()
    for row in asyncio.run(render_all(args.scenarios or None)):
        print(f"{row['id']:34s} {row['seconds']:6.1f}s  {row['turns']:2d} turns  {row['wav']}")
