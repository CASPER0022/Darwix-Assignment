"""Audio helpers.

The whole system speaks one internal format: 16 kHz, mono, signed 16-bit PCM.
Browsers, edge-tts and Whisper all want something slightly different, so every
conversion happens here and nowhere else.

ffmpeg is used for format conversion (it is already a hard dependency for
recording), and the stdlib `wave`/`audioop`-free maths keeps the streaming path
dependency-light.
"""
from __future__ import annotations

import array
from collections import deque
import io
import math
import shutil
import subprocess
import wave
from pathlib import Path

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2  # bytes, 16-bit
CHANNELS = 1

# Absolute RMS below which a buffer is treated as containing no speech at all,
# no matter what the adaptive noise floor has drifted to.
#
# Measured against the committed call recordings: real speech sits at 0.023 -
# 0.151 RMS (quietest utterance 0.023), while room tone after the browser's
# noiseSuppression is 0.000 - 0.003. 0.010 sits an octave below the quietest
# real speech and well above any silence, so it separates the two cleanly.
#
# This matters because Whisper does not return "" for silence - it hallucinates
# "Thank you." / "Thanks for watching!" with high confidence. Never handing it
# silence is the only reliable fix; filtering its output afterwards is a
# second line of defence, not the first.
SPEECH_RMS_FLOOR = 0.010


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def to_wav_bytes(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw PCM16 in a WAV container (Whisper endpoints want a container)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def read_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wf:
        return wf.readframes(wf.getnframes()), wf.getframerate()


def write_wav(path: Path, pcm: bytes, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def transcode(src: Path | bytes, dst: Path, *, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS) -> Path:
    """Any input format -> mono 16 kHz PCM WAV, via ffmpeg."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if isinstance(src, bytes):
        cmd += ["-i", "pipe:0"]
    else:
        cmd += ["-i", str(src)]
    cmd += ["-ac", str(channels), "-ar", str(sample_rate), "-f", "wav", str(dst)]
    subprocess.run(cmd, input=src if isinstance(src, bytes) else None, check=True, capture_output=True)
    return dst


def to_mp3(src: Path, dst: Path | None = None) -> Path | None:
    """WAV -> MP3, preserving channel count. Returns None without ffmpeg.

    Call recordings are written as WAV and committed as MP3 (a stereo WAV of a
    three-minute call is ~11 MB). Writing the MP3 at the same moment as the WAV
    is what keeps the committed evidence matching the committed transcript -
    doing it by hand afterwards is how they drift apart.
    """
    if not ffmpeg_available():
        return None
    dst = dst or src.with_suffix(".mp3")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Fixed 48 kbps rather than a VBR quality level: the source is 16 kHz
    # speech, so LAME's VBR levels land far lower than they would on music and
    # the result is muddier than the recording it is evidence of.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-codec:a", "libmp3lame", "-b:a", "48k", str(dst)],
                   check=True, capture_output=True)
    return dst


def mix_stereo(left_pcm: bytes, right_pcm: bytes) -> bytes:
    """Interleave two mono tracks into stereo.

    Call recordings are written with the agent on the left channel and the
    customer on the right. That gives perfect speaker separation for the Q4
    analysis without needing a diarisation model, and it is exactly how
    contact-centre recorders behave.
    """
    n = max(len(left_pcm), len(right_pcm))
    n -= n % SAMPLE_WIDTH
    left = array.array("h", left_pcm.ljust(n, b"\x00")[:n])
    right = array.array("h", right_pcm.ljust(n, b"\x00")[:n])
    out = array.array("h", [0] * (len(left) * 2))
    out[0::2] = left
    out[1::2] = right
    return out.tobytes()


def rms(pcm: bytes) -> float:
    """Root-mean-square level of a PCM16 buffer, normalised to 0..1."""
    if len(pcm) < SAMPLE_WIDTH:
        return 0.0
    samples = array.array("h", pcm[: len(pcm) - (len(pcm) % SAMPLE_WIDTH)])
    if not samples:
        return 0.0
    total = sum(float(s) * float(s) for s in samples)
    return math.sqrt(total / len(samples)) / 32768.0


def duration_seconds(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> float:
    return len(pcm) / (SAMPLE_WIDTH * CHANNELS * sample_rate)


def peak(pcm: bytes) -> float:
    """Peak absolute sample of a PCM16 buffer, normalised to 0..1."""
    if len(pcm) < SAMPLE_WIDTH:
        return 0.0
    samples = array.array("h", pcm[: len(pcm) - (len(pcm) % SAMPLE_WIDTH)])
    if not samples:
        return 0.0
    return max(abs(int(s)) for s in samples) / 32768.0


def has_speech(pcm: bytes, *, floor: float = SPEECH_RMS_FLOOR,
               min_voiced_ratio: float = 0.08) -> bool:
    """True if a captured segment plausibly contains speech.

    A last check before spending an ASR call. Whole-segment RMS alone is a poor
    test - a 3 s segment holding 200 ms of a door closing averages out to
    nothing - so this asks whether a meaningful *fraction* of the segment is
    above the speech floor, which is what actually distinguishes an utterance
    from a transient.
    """
    if duration_seconds(pcm) <= 0:
        return False
    window = SAMPLE_RATE // 50 * SAMPLE_WIDTH  # 20 ms frames
    frames = [pcm[i:i + window] for i in range(0, len(pcm), window)]
    frames = [f for f in frames if len(f) >= SAMPLE_WIDTH]
    if not frames:
        return False
    voiced = sum(1 for f in frames if rms(f) >= floor)
    return (voiced / len(frames)) >= min_voiced_ratio


class EnergyVAD:
    """Energy-based endpointing.

    A neural VAD would be better, but it is another model on the latency path
    and another build dependency on Windows. For telephone-grade turn taking,
    an adaptive noise floor plus a hangover window is enough, and it is easy to
    explain and tune. Limitation acknowledged in docs/limitations.
    """

    # 1100 ms of hangover, not the 700 ms this started with: a natural sentence
    # pause inside one utterance was being read as end-of-turn, so the agent
    # answered half a question. Measured against real TTS and speech audio.
    #
    # The adaptive floor is clamped at both ends. Unclamped, the exponential
    # decay drives it toward the room tone itself: measured from a fresh VAD,
    # the start threshold fell from 0.0229 to 0.0012 after 51 s of quiet - a
    # 19x collapse - after which a fan swell or a breath cleared it and 3 s of
    # near-silence went to Whisper, which answers silence with "Thank you.".
    # Every turn in a quiet room became "Thank you.". The floor must not be
    # allowed to converge on the noise it is measuring.
    #
    # The floor is estimated by minimum statistics - a low percentile of the
    # levels seen recently - rather than by smoothing the chunks that happen to
    # fall below the current threshold. The difference matters in a room that
    # is merely a bit noisy: with the old rule, once the room tone rose above
    # the threshold every chunk counted as speech, so the branch that adapts
    # the floor never ran again and the VAD stayed permanently triggered.
    # Observed live at 0.013 RMS ambient, which Whisper turned into
    # "in the end, Pernando, a room because you're not happy with your home."
    # A percentile keeps adapting whatever the speech decision is, because
    # speech has gaps and the gaps are what the low percentile sees.
    def __init__(
        self,
        *,
        start_ratio: float = 3.0,
        min_speech_ms: float = 250.0,
        hangover_ms: float = 1100.0,
        floor_init: float = 0.008,
        floor_min: float = 0.004,
        floor_max: float = 0.060,
        abs_floor: float = SPEECH_RMS_FLOOR,
        floor_window: int = 60,        # ~15 s at 256 ms chunks
        floor_percentile: float = 0.15,
    ) -> None:
        self.start_ratio = start_ratio
        self.min_speech_ms = min_speech_ms
        self.hangover_ms = hangover_ms
        self.floor_min = floor_min
        self.floor_max = floor_max
        # Speech must clear the adaptive threshold *and* this absolute level.
        # The ratio test alone is relative, so it can always be satisfied by
        # audio that is quiet in absolute terms; this is the backstop.
        self.abs_floor = abs_floor
        self.floor_percentile = floor_percentile
        self.noise_floor = min(max(floor_init, floor_min), floor_max)
        self._levels: deque[float] = deque(maxlen=floor_window)
        self.speaking = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0

    def _update_floor(self, level: float) -> None:
        self._levels.append(level)
        # The percentile is only meaningful once the window is wide enough to
        # contain a pause. A call that opens with someone already talking fills
        # the first few slots with speech only, and a percentile taken over
        # those puts the "noise" floor inside the speech: measured on a real
        # recording it reached 0.0476, lifting the threshold to 0.1429 - above
        # the speech it was supposed to be tracking - and closed the utterance
        # 2.8 s in, mid-sentence. Until the window is half full the seed value
        # is the better estimate.
        if len(self._levels) < self._levels.maxlen // 2:
            return
        ordered = sorted(self._levels)
        idx = min(int(len(ordered) * self.floor_percentile), len(ordered) - 1)
        estimate = min(max(ordered[idx], self.floor_min), self.floor_max)
        # Rise slowly, fall freely. A room gets noisy gradually, but the moment
        # it goes quiet the agent should be able to hear a soft reply again.
        if estimate > self.noise_floor:
            estimate = min(estimate, self.noise_floor * 1.5 + 1e-4)
        self.noise_floor = estimate

    @property
    def threshold(self) -> float:
        """The level a chunk must exceed to count as speech."""
        return max(self.noise_floor * self.start_ratio, self.abs_floor)

    def push(self, pcm: bytes) -> str:
        """Feed one chunk. Returns 'speech_start', 'speech_end' or ''."""
        ms = duration_seconds(pcm) * 1000.0
        level = rms(pcm)
        # The threshold is read before the update so a chunk is judged against
        # the room as it was, not against itself.
        threshold = self.threshold
        self._update_floor(level)
        event = ""
        if level > threshold:
            self._speech_ms += ms
            self._silence_ms = 0.0
            if not self.speaking and self._speech_ms >= self.min_speech_ms:
                self.speaking = True
                event = "speech_start"
        else:
            self._silence_ms += ms
            if self.speaking and self._silence_ms >= self.hangover_ms:
                self.speaking = False
                self._speech_ms = 0.0
                event = "speech_end"
            elif not self.speaking:
                self._speech_ms = 0.0
        return event
