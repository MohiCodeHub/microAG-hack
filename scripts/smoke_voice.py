"""Real API smoke test: TTS a sentence to data/smoke.mp3, then STT it back.

Usage: uv run python scripts/smoke_voice.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.voice import Voice  # noqa: E402

SENTENCE = "Keep your hands in view, please."


async def main() -> None:
    s = get_settings()
    async with httpx.AsyncClient() as http:
        voice = Voice(s.elevenlabs_api_key, s.elevenlabs_voice_id, http, s.elevenlabs_stt_model)
        audio = await voice.tts(SENTENCE)
        out = s.data_dir / "smoke.mp3"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(audio)
        print(f"tts: {len(audio)} bytes -> {out}")
        print("transcript:", repr(await voice.stt(audio, "audio/mpeg")))


if __name__ == "__main__":
    asyncio.run(main())
