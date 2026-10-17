from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    openai_api_key: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "hpp4J3VqNfWAUOO0d1Us"
    elevenlabs_stt_model: str = "scribe_v2"
    caption_model: str = "gpt-5.6-luna"
    monitor_model: str = "gpt-5.6-luna"
    tasks_dir: Path = field(default_factory=lambda: Path("tasks"))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    chunk_frames: int = 10
    llm_timeout_s: float = 20
    question_cooldown_s: float = 45
    question_min_interval_s: float = 60
    nudge_cooldown_s: float = 20
    answer_timeout_s: float = 20
    finish_timeout_s: float = 25  # max wait for in-flight chunks before the report


def get_settings() -> Settings:
    load_dotenv()
    env = os.environ.get
    return Settings(
        openai_api_key=env("OPENAI_API_KEY", ""),
        elevenlabs_api_key=env("ELEVENLABS_API_KEY", ""),
        elevenlabs_voice_id=env("ELEVENLABS_VOICE_ID", "hpp4J3VqNfWAUOO0d1Us"),
        elevenlabs_stt_model=env("ELEVENLABS_STT_MODEL", "scribe_v2"),
        monitor_model=env("MONITOR_MODEL", "gpt-5.6-luna"),
    )
