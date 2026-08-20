"""Text to speech.

edge-tts is used because it is the only free option that ships *native* neural
voices for every locale this assessment needs:

    en-IN  Neerja / Prabhat       Q1 India business-loan agent
    fil-PH Blessica / Angelo      Q3 Philippines (Filipino/Taglish)
    en-PH  Rosa / James           Q3 Philippines English-leaning turns
    id-ID  Gadis / Ardi           Q3 Indonesia (standard Jakarta register)
    jv-ID  Siti / Dimas           Q3 Indonesia, Javanese-accented speech
    su-ID  Tuti / Jajang          Q3 Indonesia, Sundanese-accented speech

The jv-ID / su-ID voices are what satisfy the assessment's "at least one
regional accent outside standard Jakarta speech" requirement with real audio
rather than a claim.

Compromise, stated plainly: these are Microsoft Edge read-aloud voices, not a
commercial contact-centre TTS. They are not licensed for production telephony
and have no word-level timing callbacks. Production swap-in is noted in
docs/limitations_and_production_plan.md.
"""
from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path

import edge_tts

from .audio import SAMPLE_RATE, read_wav, transcode
from .logging import log


@dataclass(frozen=True)
class Voice:
    name: str
    locale: str
    gender: str
    rate: str = "+0%"
    pitch: str = "+0Hz"
    note: str = ""


# Registry: every voice used anywhere in the repo is declared here so the
# multilingual configuration is auditable in one place.
VOICES: dict[str, Voice] = {
    # --- Q1: India, business loan --------------------------------------
    "agent_en_in_f": Voice("en-IN-NeerjaNeural", "en-IN", "female", rate="+4%"),
    "agent_en_in_m": Voice("en-IN-PrabhatNeural", "en-IN", "male"),
    "caller_en_in_m": Voice("en-IN-PrabhatNeural", "en-IN", "male", rate="+6%"),
    "caller_en_in_f": Voice("en-IN-NeerjaExpressiveNeural", "en-IN", "female", rate="+8%"),
    # --- Q3: Philippines ------------------------------------------------
    "agent_fil_f": Voice("fil-PH-BlessicaNeural", "fil-PH", "female", rate="+3%",
                         note="Filipino/Taglish agent voice"),
    "caller_fil_m": Voice("fil-PH-AngeloNeural", "fil-PH", "male", rate="+5%"),
    "caller_fil_f": Voice("fil-PH-BlessicaNeural", "fil-PH", "female", rate="+7%"),
    "caller_en_ph_m": Voice("en-PH-JamesNeural", "en-PH", "male",
                            note="English-dominant Filipino speaker"),
    # --- Q3: Indonesia --------------------------------------------------
    "agent_id_f": Voice("id-ID-GadisNeural", "id-ID", "female", rate="+2%",
                        note="Standard Jakarta register agent voice"),
    "caller_id_m": Voice("id-ID-ArdiNeural", "id-ID", "male", rate="+5%"),
    "caller_id_f": Voice("id-ID-GadisNeural", "id-ID", "female", rate="+6%"),
    "caller_jv_m": Voice("jv-ID-DimasNeural", "jv-ID", "male", rate="+2%",
                         note="Javanese-accented Indonesian - regional accent test"),
    "caller_jv_f": Voice("jv-ID-SitiNeural", "jv-ID", "female",
                         note="Javanese-accented Indonesian - regional accent test"),
    "caller_su_m": Voice("su-ID-JajangNeural", "su-ID", "male",
                         note="Sundanese-accented Indonesian - regional accent test"),
    "caller_su_f": Voice("su-ID-TutiNeural", "su-ID", "female",
                         note="Sundanese-accented Indonesian - regional accent test"),
}

DEFAULT_AGENT_VOICE = {
    "en-IN": "agent_en_in_f",
    "fil-PH": "agent_fil_f",
    "id-ID": "agent_id_f",
}


class TTSError(RuntimeError):
    pass


class TTSClient:
    def __init__(self) -> None:
        self._lock = asyncio.Semaphore(4)

    async def synthesize(
        self,
        text: str,
        voice_key: str = "agent_en_in_f",
        *,
        rate: str | None = None,
        pitch: str | None = None,
    ) -> bytes:
        """Return mono 16 kHz PCM16 for `text` in the requested voice."""
        if not text.strip():
            return b""
        voice = VOICES.get(voice_key)
        if voice is None:
            raise TTSError("unknown voice key: " + voice_key)
        async with self._lock:
            comm = edge_tts.Communicate(
                text,
                voice.name,
                rate=rate or voice.rate,
                pitch=pitch or voice.pitch,
            )
            mp3 = bytearray()
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    mp3.extend(chunk["data"])
        if not mp3:
            raise TTSError("edge-tts returned no audio for voice " + voice.name)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "tts.wav"
            transcode(bytes(mp3), out, sample_rate=SAMPLE_RATE, channels=1)
            pcm, _ = read_wav(out)
        log("tts.synthesized", voice=voice.name, chars=len(text), pcm_bytes=len(pcm))
        return pcm

    async def to_file(self, text: str, voice_key: str, path: Path) -> Path:
        from .audio import write_wav

        pcm = await self.synthesize(text, voice_key)
        write_wav(path, pcm)
        return path


_shared: TTSClient | None = None


def get_tts() -> TTSClient:
    global _shared
    if _shared is None:
        _shared = TTSClient()
    return _shared
