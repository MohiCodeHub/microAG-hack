import json

import httpx

from app.voice import Voice


def mk(handler):
    return Voice("key", "vid", httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_tts_posts_flash_model_returns_bytes():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["key"] = req.headers["xi-api-key"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, content=b"MP3DATA")

    out = await mk(handler).tts("Hello there")
    assert out == b"MP3DATA"
    assert seen["url"] == "https://api.elevenlabs.io/v1/text-to-speech/vid?output_format=mp3_44100_128"
    assert seen["key"] == "key"
    assert seen["body"] == {"text": "Hello there", "model_id": "eleven_flash_v2_5"}


async def test_stt_returns_text():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["body"] = req.content
        return httpx.Response(200, json={"text": "  I was fetching a tool. "})

    out = await mk(handler).stt(b"audio-bytes", "audio/webm")
    assert out == "I was fetching a tool."
    assert seen["url"] == "https://api.elevenlabs.io/v1/speech-to-text"
    assert b'name="model_id"' in seen["body"] and b"scribe_v2" in seen["body"]
    assert b'name="file"' in seen["body"] and b"audio-bytes" in seen["body"]


async def test_stt_returns_empty_on_500():
    assert await mk(lambda r: httpx.Response(500, text="boom")).stt(b"x", "audio/webm") == ""


async def test_stt_returns_empty_on_timeout():
    def handler(req):
        raise httpx.ReadTimeout("slow")

    assert await mk(handler).stt(b"x", "audio/webm") == ""
