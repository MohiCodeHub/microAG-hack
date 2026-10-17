import asyncio
import base64
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings  # noqa: F401
from tests.legacy_settings import LegacySettings
from app.llm import MonitorDecision
from app.server import Hub, create_app

CAP = {"caption": "fills kettle", "step": "fill_kettle", "confidence": 0.9, "objects": ["kettle"],
       "hands_visible": "both", "activity": "active", "recapture_suspected": False}


def jpeg_b64(seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    arr = (rng.random((64, 64)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).resize((128, 128), Image.NEAREST).save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeLLM:
    async def caption(self, jpegs, task, steps, prev_caption):
        return dict(CAP)

    async def monitor(self, task, references, timeline, trigger):
        return MonitorDecision.default()

    async def final_report(self, task, references, timeline):
        return {"segments": [], "fraud_score": 40, "verdict": "accept", "reasons": []}


class FakeVoice:
    async def tts(self, text):
        return b"mp3"

    async def stt(self, audio, mime):
        return "I was waiting"


def make_client(tmp_path):
    settings = LegacySettings(data_dir=tmp_path / "data")
    app = create_app(settings, llm_factory=lambda s: FakeLLM(),
                     voice_factory=lambda s, http: FakeVoice())
    return TestClient(app)


def test_tasks_endpoint_lists_example_task(tmp_path):
    with make_client(tmp_path) as c:
        tasks = c.get("/api/tasks").json()
    tea = next(t for t in tasks if t["id"] == "example_make_tea")
    assert tea["name"] == "Make a cup of tea"
    assert tea["steps"][0] == {"id": "fill_kettle", "label": "Fill the kettle with water"}
    assert tea["reference_count"] == 0


def test_pages_are_served(tmp_path):
    with make_client(tmp_path) as c:
        assert "/static/dashboard.js" in c.get("/").text
        assert "/static/phone.js" in c.get("/phone").text
        assert c.get("/static/phone.html").status_code == 200


def test_pages_and_static_files_are_not_cached(tmp_path):
    with make_client(tmp_path) as c:
        for path in ("/", "/phone", "/static/phone.js", "/static/capture.js"):
            assert "no-cache" in c.get(path).headers.get("cache-control", ""), path
        assert "no-cache" not in c.get("/api/tasks").headers.get("cache-control", "")


def test_source_ws_start_frames_produce_chunk_on_watch(tmp_path):
    with make_client(tmp_path) as c, c.websocket_connect("/ws/watch") as watch, \
            c.websocket_connect("/ws/source") as src:
        src.send_json({"type": "start", "task_id": "example_make_tea"})
        assert watch.receive_json()["type"] == "session_started"
        for t in range(10):
            src.send_json({"type": "frame", "t": t, "jpeg_b64": jpeg_b64()})
        types = [watch.receive_json()["type"] for _ in range(11)]
        assert types == ["frame"] * 10 + ["chunk"]


def test_session_started_carries_id_and_task(tmp_path):
    with make_client(tmp_path) as c, c.websocket_connect("/ws/watch") as watch, \
            c.websocket_connect("/ws/source") as src:
        src.send_json({"type": "start", "task_id": "example_make_tea"})
        ev = watch.receive_json()
        assert ev["session_id"] and ev["task"]["id"] == "example_make_tea"


def test_events_carry_the_session_id(tmp_path):
    with make_client(tmp_path) as c, c.websocket_connect("/ws/watch") as watch, \
            c.websocket_connect("/ws/source") as src:
        src.send_json({"type": "start", "task_id": "example_make_tea"})
        started = watch.receive_json()
        sid = started["session_id"]
        assert src.receive_json() == started  # the source learns its session id too
        for t in range(10):
            src.send_json({"type": "frame", "t": t, "jpeg_b64": jpeg_b64()})
        src.send_json({"type": "stop"})
        seen = []
        while not seen or seen[-1]["type"] != "report":
            seen.append(watch.receive_json())
        assert {"frame", "chunk", "monitor", "session_ending", "report"} <= {e["type"] for e in seen}
        assert all(e["session_id"] == sid for e in seen)
        assert src.receive_json()["session_id"] == sid


def test_late_watcher_gets_session_started_and_chunks_so_far(tmp_path):
    with make_client(tmp_path) as c, c.websocket_connect("/ws/watch") as early, \
            c.websocket_connect("/ws/source") as src:
        src.send_json({"type": "start", "task_id": "example_make_tea"})
        sid = early.receive_json()["session_id"]
        for t in range(10):
            src.send_json({"type": "frame", "t": t, "jpeg_b64": jpeg_b64()})
        while early.receive_json()["type"] != "chunk":
            pass
        with c.websocket_connect("/ws/watch") as late:
            first, second = late.receive_json(), late.receive_json()
        assert first["type"] == "session_started" and first["session_id"] == sid
        assert first["task"] == {"id": "example_make_tea", "name": "Make a cup of tea"}
        assert second["type"] == "chunk" and second["session_id"] == sid
        assert second["chunk"]["caption"] == "fills kettle"


class SlowReportLLM(FakeLLM):
    async def final_report(self, task, references, timeline):
        await asyncio.sleep(0.5)
        return await super().final_report(task, references, timeline)


def test_watcher_joining_while_the_report_is_built_still_gets_it(tmp_path):
    app = create_app(LegacySettings(data_dir=tmp_path / "data"), llm_factory=lambda s: SlowReportLLM(),
                     voice_factory=lambda s, http: FakeVoice())
    with TestClient(app) as c, c.websocket_connect("/ws/source") as src:
        src.send_json({"type": "start", "task_id": "example_make_tea"})
        sid = src.receive_json()["session_id"]
        src.send_json({"type": "stop"})
        with c.websocket_connect("/ws/watch") as late:
            seen = [late.receive_json()]
            while seen[-1]["type"] != "report":
                seen.append(late.receive_json())
        assert seen[0]["type"] == "session_started" and seen[0]["session_id"] == sid
        assert "session_ending" in [e["type"] for e in seen]
        assert seen[-1]["session_id"] == sid


def test_missing_api_keys_are_logged_at_startup(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="app.server"):
        create_app(LegacySettings(data_dir=tmp_path / "data"))
    text = caplog.text
    assert "OPENAI_API_KEY" in text and "ELEVENLABS_API_KEY" in text
    caplog.clear()
    with caplog.at_level("WARNING", logger="app.server"):
        create_app(LegacySettings(data_dir=tmp_path / "data", openai_api_key="k",
                            elevenlabs_api_key="k"))
    assert "API_KEY" not in caplog.text


def test_unknown_task_returns_error(tmp_path):
    with make_client(tmp_path) as c, c.websocket_connect("/ws/source") as src:
        src.send_json({"type": "start", "task_id": "nope"})
        ev = src.receive_json()
        assert ev["type"] == "error" and "nope" in ev["message"]


def test_stop_emits_report_to_source_and_watch(tmp_path):
    with make_client(tmp_path) as c, c.websocket_connect("/ws/watch") as watch, \
            c.websocket_connect("/ws/source") as src:
        src.send_json({"type": "start", "task_id": "example_make_tea"})
        for t in range(10):
            src.send_json({"type": "frame", "t": t, "jpeg_b64": jpeg_b64()})
        src.send_json({"type": "stop"})
        assert src.receive_json()["type"] == "session_started"
        assert src.receive_json()["type"] == "report"
        seen = []
        while not seen or seen[-1] != "report":
            seen.append(watch.receive_json()["type"])
        assert "chunk" in seen


def test_answer_reaches_watch_and_start_finishes_old_session(tmp_path):
    with make_client(tmp_path) as c, c.websocket_connect("/ws/watch") as watch, \
            c.websocket_connect("/ws/source") as src:
        src.send_json({"type": "start", "task_id": "example_make_tea"})
        src.send_json({"type": "answer", "audio_b64": base64.b64encode(b"x").decode(),
                       "mime": "audio/webm"})
        src.send_json({"type": "start", "task_id": "example_make_tea"})
        seen = [watch.receive_json()["type"] for _ in range(6)]
        assert seen == ["session_started", "worker_said", "monitor", "session_ending", "report",
                        "session_started"]
        assert [src.receive_json()["type"] for _ in range(3)] == [
            "session_started", "report", "session_started"]


def test_dead_watcher_does_not_break_session(tmp_path):
    with make_client(tmp_path) as c, c.websocket_connect("/ws/source") as src:
        with c.websocket_connect("/ws/watch"):
            pass
        src.send_json({"type": "start", "task_id": "example_make_tea"})
        src.send_json({"type": "stop"})
        assert src.receive_json()["type"] == "session_started"
        assert src.receive_json()["type"] == "report"


class SlowWS:
    async def send_json(self, event):
        await asyncio.sleep(3600)


def test_stalled_watcher_does_not_block_session(tmp_path):
    with make_client(tmp_path) as c, c.websocket_connect("/ws/watch") as watch, \
            c.websocket_connect("/ws/source") as src:
        c.portal.call(c.app.state.hub.add_watcher, SlowWS())
        src.send_json({"type": "start", "task_id": "example_make_tea"})
        for t in range(10):
            src.send_json({"type": "frame", "t": t, "jpeg_b64": jpeg_b64()})
        types = [watch.receive_json()["type"] for _ in range(12)]
        assert types[-1] == "chunk"


async def test_hub_drops_watcher_whose_queue_is_full():
    hub = Hub(queue_size=2)
    ws = SlowWS()
    hub.add_watcher(ws)
    for i in range(10):
        await hub.broadcast({"type": "frame", "i": i})
    assert ws not in hub.watchers


def test_invalid_json_and_binary_do_not_end_session(tmp_path):
    with make_client(tmp_path) as c, c.websocket_connect("/ws/watch") as watch, \
            c.websocket_connect("/ws/source") as src:
        src.send_json({"type": "start", "task_id": "example_make_tea"})
        assert src.receive_json()["type"] == "session_started"
        src.send_text("not json")
        assert src.receive_json()["type"] == "error"
        src.send_bytes(b"\x00\x01")
        assert src.receive_json()["type"] == "error"
        src.send_text("[1]")
        assert src.receive_json()["type"] == "error"
        for t in range(10):
            src.send_json({"type": "frame", "t": t, "jpeg_b64": jpeg_b64()})
        types = [watch.receive_json()["type"] for _ in range(12)]
        assert types[-1] == "chunk"
