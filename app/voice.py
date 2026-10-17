from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

BASE = "https://api.elevenlabs.io/v1"
TTS_MODEL = "eleven_flash_v2_5"
TIMEOUT_S = 15


class Voice:
    def __init__(self, api_key: str, voice_id: str, http: httpx.AsyncClient,
                 stt_model: str = "scribe_v2"):
        self.api_key = api_key
        self.voice_id = voice_id
        self.http = http
        self.stt_model = stt_model

    async def tts(self, text: str) -> bytes:
        """Synthesize speech; returns MP3 bytes. Raises on HTTP/timeout errors."""
        r = await self.http.post(
            f"{BASE}/text-to-speech/{self.voice_id}?output_format=mp3_44100_128",
            headers={"xi-api-key": self.api_key},
            json={"text": text, "model_id": TTS_MODEL},
            timeout=TIMEOUT_S,
        )
        r.raise_for_status()
        return r.content

    async def stt(self, audio: bytes, mime: str) -> str:
        """Transcribe audio; returns stripped text, or "" on any error."""
        try:
            r = await self.http.post(
                f"{BASE}/speech-to-text",
                headers={"xi-api-key": self.api_key},
                files={"file": ("audio", audio, mime)},
                data={"model_id": self.stt_model},
                timeout=TIMEOUT_S,
            )
            r.raise_for_status()
            return (r.json().get("text") or "").strip()
        except Exception as e:  # noqa: BLE001 - STT failure must never break the session
            log.warning("stt failed: %s", e)
            return ""
