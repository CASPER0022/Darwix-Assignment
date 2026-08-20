"""Central configuration.

Every tunable the assessment cares about (retrieval threshold, nudge cooldown,
model ids) is here rather than scattered through the code, so a reviewer can
change behaviour without reading the implementation.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
EVAL_DIR = REPO_ROOT / "evaluation"

load_dotenv(REPO_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(REPO_ROOT / ".env"), extra="ignore")

    gemini_api_key: str = ""
    groq_api_key: str = ""

    gemini_chat_model: str = "gemini-3.6-flash"
    gemini_embed_model: str = "gemini-embedding-001"
    # Model choice is measured, not assumed - see evaluation/model_selection.md
    groq_dialog_model: str = "openai/gpt-oss-120b"   # ~0.9 s, correct language every time
    groq_fast_model: str = "openai/gpt-oss-20b"      # ~0.8 s, used for Q4 signal extraction
    groq_reasoning_effort: str = "low"
    gemini_thinking_level: str = "minimal"
    groq_asr_model: str = "whisper-large-v3-turbo"
    embed_dimensions: int = 768

    host: str = "127.0.0.1"
    port: int = 8000

    retrieval_min_score: float = 0.35
    retrieval_top_k: int = 4

    # Level a caller must reach to interrupt the agent mid-sentence. It sits
    # above ordinary speech (0.023-0.15 RMS measured) because while the agent
    # is talking the microphone also hears the agent: browser echo
    # cancellation reduces that bleed but does not remove it, and anything it
    # leaves through is loud, continuous and looks exactly like a turn.
    # Raising the bar only while the agent speaks keeps barge-in working
    # without the agent answering itself.
    barge_in_rms: float = 0.055
    # How long after playback stops the raised bar stays in force, covering
    # the tail of audio already queued in the browser.
    echo_guard_tail_ms: float = 400.0

    nudge_min_confidence: float = 0.55
    nudge_cooldown_seconds: float = 45.0
    nudge_ttl_seconds: float = 90.0
    nudge_max_active: int = 3

    escalation_webhook_url: str = ""

    # --- derived paths ------------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return DATA_DIR / "raw"

    @property
    def interim_dir(self) -> Path:
        return DATA_DIR / "interim"

    @property
    def kb_dir(self) -> Path:
        return DATA_DIR / "kb"

    @property
    def records_path(self) -> Path:
        return self.kb_dir / "records.jsonl"

    @property
    def index_path(self) -> Path:
        return self.kb_dir / "index.sqlite"

    @property
    def transcripts_dir(self) -> Path:
        return DATA_DIR / "transcripts"

    @property
    def recordings_dir(self) -> Path:
        return DATA_DIR / "recordings"

    @property
    def crm_dir(self) -> Path:
        return DATA_DIR / "crm"

    def require(self, *names: str) -> None:
        """Fail loudly and usefully when a key is missing."""
        missing = [n for n in names if not getattr(self, n, "")]
        if missing:
            raise RuntimeError(
                "Missing required setting(s): "
                + ", ".join(m.upper() for m in missing)
                + ". Copy .env.example to .env and fill them in "
                "(both providers are free tier)."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
