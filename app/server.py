"""FastAPI server: pages, task API, and the source / watch WebSocket protocol."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

from app.config import Settings, get_settings
from app.llm import LLM
from app.session import Session
from app.signals import HashStore
from app.tasks import load_library, load_reference_timelines
from app.voice import Voice

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
# Also delivered to the capture client. session_started tells it its session id, so it can
# ignore a late report or question from a session it already left.
SOURCE_EVENTS = {"session_started", "agent_say", "report", "error"}
WATCHER_QUEUE = 64  # events buffered per dashboard before it is considered stalled
REPLAY_CHUNKS = 32  # chunks replayed to a dashboard that connects mid-session
SEND_TIMEOUT_S = 5.0
API_KEYS = {"OPENAI_API_KEY": "openai_api_key", "ELEVENLABS_API_KEY": "elevenlabs_api_key"}


def default_llm_factory(settings: Settings) -> LLM:
    # max_retries=0 so llm_timeout_s is the real upper bound per call.
    client = AsyncOpenAI(api_key=settings.openai_api_key or "missing", max_retries=0)
    return LLM(client, settings)


def default_voice_factory(settings: Settings, http: httpx.AsyncClient) -> Voice:
    return Voice(settings.elevenlabs_api_key, settings.elevenlabs_voice_id, http,
                 stt_model=settings.elevenlabs_stt_model)


def session_started_event(session: Session) -> dict:
    return {"type": "session_started", "session_id": session.id,
            "task": {"id": session.task.id, "name": session.task.name}}


def warn_missing_keys(settings: Settings) -> None:
    for env, attr in API_KEYS.items():
        if not getattr(settings, attr):
            log.warning("!!! %s is not set: every call that needs it will fail. "
                        "Add it to .env and restart.", env)


@dataclass
class Active:
    session: Session
    source: WebSocket
    answers: list[asyncio.Task] = field(default_factory=list)


class Hub:
    """Tracks dashboards and the single active session.

    Each dashboard gets a bounded queue and a writer task, so a slow or stalled dashboard can
    never block the session's emit path; one whose queue fills (or whose send times out) is
    dropped.
    """

    def __init__(self, queue_size: int = WATCHER_QUEUE) -> None:
        self.queue_size = queue_size
        self.watchers: dict[WebSocket, asyncio.Queue] = {}
        self._writers: dict[WebSocket, asyncio.Task] = {}
        self.active: Active | None = None
        self.ending: Session | None = None  # session whose report is being built

    def add_watcher(self, ws: WebSocket) -> None:
        q: asyncio.Queue = asyncio.Queue(self.queue_size)
        s = self.active.session if self.active else self.ending
        if s:  # joined mid-session: give it the task and the timeline so far
            q.put_nowait(session_started_event(s))
            for c in s.timeline.chunks[-min(REPLAY_CHUNKS, self.queue_size // 2):]:
                q.put_nowait({"type": "chunk", "chunk": asdict(c), "signals": {},
                              "session_id": s.id})
            if s is self.ending:
                q.put_nowait({"type": "session_ending", "session_id": s.id})
        self.watchers[ws] = q
        self._writers[ws] = asyncio.create_task(self._write(ws, q))

    def remove_watcher(self, ws: WebSocket) -> None:
        self.watchers.pop(ws, None)
        w = self._writers.pop(ws, None)
        if w and w is not asyncio.current_task():
            w.cancel()

    async def _write(self, ws: WebSocket, q: asyncio.Queue) -> None:
        while True:
            event = await q.get()
            if not await _send(ws, event, SEND_TIMEOUT_S):
                self.remove_watcher(ws)
                return

    async def broadcast(self, event: dict) -> None:
        for ws, q in list(self.watchers.items()):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("dropping stalled dashboard")
                self.remove_watcher(ws)

    async def end(self, only_source: WebSocket | None = None) -> None:
        """Finish the active session (optionally only if owned by `only_source`)."""
        a = self.active
        if a is None or (only_source is not None and a.source is not only_source):
            return
        self.active = None
        self.ending = a.session
        if a.answers:
            results = await asyncio.gather(*a.answers, return_exceptions=True)
            for r in results:
                if isinstance(r, BaseException):
                    log.warning("answer processing failed: %r", r)
        try:
            await a.session.finish()
        except Exception:  # noqa: BLE001
            log.exception("finish failed")
            await _send(a.source, {"type": "error", "message": "report generation failed"})
        finally:
            if self.ending is a.session:
                self.ending = None


async def _send(ws: WebSocket, event: dict, timeout: float = SEND_TIMEOUT_S) -> bool:
    try:
        await asyncio.wait_for(ws.send_json(event), timeout)
        return True
    except Exception:  # noqa: BLE001 - closed / dead / stalled socket (incl. timeout)
        return False


def create_app(settings: Settings,
               llm_factory: Callable[[Settings], Any] = default_llm_factory,
               voice_factory: Callable[[Settings, httpx.AsyncClient], Any] = default_voice_factory,
               ) -> FastAPI:
    warn_missing_keys(settings)
    http = httpx.AsyncClient()
    hub = Hub()
    hashes = HashStore(Path(settings.data_dir) / "hashes.json")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await hub.end()
        await http.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.hub = hub

    @app.get("/")
    def dashboard() -> FileResponse:
        return FileResponse(STATIC_DIR / "dashboard.html")

    @app.get("/phone")
    def phone() -> FileResponse:
        return FileResponse(STATIC_DIR / "phone.html")

    @app.get("/api/tasks")
    def tasks() -> list[dict]:
        lib = load_library(settings.tasks_dir)
        return [
            {"id": t.id, "name": t.name,
             "steps": [{"id": s.id, "label": s.label} for s in t.steps],
             "reference_count": len(load_reference_timelines(settings.tasks_dir, t.id))}
            for t in lib.values()
        ]

    @app.websocket("/ws/watch")
    async def ws_watch(ws: WebSocket) -> None:
        await ws.accept()
        hub.add_watcher(ws)
        try:
            while True:
                await ws.receive_text()  # ignore input; detects disconnect
        except WebSocketDisconnect:
            pass
        finally:
            hub.remove_watcher(ws)

    async def start_session(ws: WebSocket, task_id: Any) -> None:
        library = load_library(settings.tasks_dir)
        task = library.get(task_id) if isinstance(task_id, str) else None
        if task is None:
            await _send(ws, {"type": "error", "message": f"unknown task_id: {task_id}"})
            return
        await hub.end()
        session_id = uuid.uuid4().hex[:8]

        async def emit(event: dict) -> None:
            # Every event names its session, so clients can drop a late one from an old session.
            event = {**event, "session_id": session_id}
            await hub.broadcast(event)
            if event.get("type") in SOURCE_EVENTS:
                await _send(ws, event)

        session = Session(
            session_id, task, library, load_reference_timelines(settings.tasks_dir, task.id),
            llm_factory(settings), voice_factory(settings, http), hashes, settings, emit,
        )
        hub.active = Active(session, ws)
        await emit(session_started_event(session))

    async def handle(ws: WebSocket, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "start":
            await start_session(ws, msg.get("task_id"))
            return
        a = hub.active
        if kind == "stop":
            await hub.end(only_source=ws)
            return
        if a is None or a.source is not ws:
            await _send(ws, {"type": "error", "message": "no active session; send start first"})
            return
        if kind == "frame":
            await a.session.add_frame(float(msg["t"]), base64.b64decode(msg["jpeg_b64"]))
        elif kind == "imu":
            a.session.add_imu(msg.get("samples", []))
        elif kind == "answer":
            audio = base64.b64decode(msg["audio_b64"])
            a.answers.append(asyncio.create_task(
                a.session.add_answer(audio, msg.get("mime", "audio/webm"))))
        else:
            await _send(ws, {"type": "error", "message": f"unknown message type: {kind}"})

    @app.websocket("/ws/source")
    async def ws_source(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                raw = await ws.receive()
                if raw["type"] == "websocket.disconnect":
                    break
                try:
                    msg = json.loads(raw.get("text") or "")
                    if not isinstance(msg, dict):
                        raise ValueError("message must be a JSON object")
                    await handle(ws, msg)
                except WebSocketDisconnect:
                    raise
                except Exception as e:  # noqa: BLE001 - a bad message must not drop the stream
                    log.warning("bad source message: %r", e)
                    await _send(ws, {"type": "error", "message": f"bad message: {e}"})
        except WebSocketDisconnect:
            pass
        finally:
            await hub.end(only_source=ws)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app(get_settings())
