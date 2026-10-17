"""Live session orchestrator: frames -> chunks -> captions -> ordered monitor -> voice.

The LLM and Voice objects are injected (built by the server). Build the AsyncOpenAI client
behind LLM with ``max_retries=0`` so ``settings.llm_timeout_s`` is the real upper bound per
call; the SDK's default retries would multiply it.

Time:
- Chunk times come from the frame ``t`` values the client sends (seconds since start).
- Policies (question cooldown, hands nudge) use ``clock()``.
- AGENT / WORKER lines are stamped at the last frame time of the latest chunk already in the
  timeline. Every future chunk starts later than that, so the timeline stays in time order and
  the newest chunk (or the worker's answer) is always the last rendered line the monitor sees.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any

from app.llm import CAPTION_FALLBACK, MonitorDecision
from app.policy import NudgePolicy, WakePolicy
from app.report import Divergence, build_report, signal_stats
from app.signals import HashStore, frame_hash, imu_energy, imu_mismatch, visual_motion
from app.tasks import Task, candidate_steps
from app.timeline import Chunk, Timeline, fmt

log = logging.getLogger(__name__)

NUDGE_TEXT = "Keep your hands in view, please."
NO_ANSWER = "(no answer)"
MIN_PARTIAL_FRAMES = 3  # a trailing partial chunk is captioned at finish if it has this many
# Duplicate footage: two consecutive chunks of ours must match the same consecutive pair of
# chunks (j, j+1) of one earlier session, with its source time advancing like ours (normally
# 10 s) within this tolerance.
DUP_TOLERANCE_S = 4.0
# Our own consecutive keyframes closer than this (pHash bits) mean a static view. A static view
# repeated across sessions (same room, camera at rest) is not evidence of recycled footage.
STATIC_SCENE_DISTANCE = 6

Frame = tuple[float, bytes]


class Session:
    def __init__(self, id: str, task: Task, library: dict[str, Task], references: list[str],
                 llm: Any, voice: Any, hashes: HashStore, settings: Any,
                 emit: Callable[[dict], Awaitable[None]],
                 clock: Callable[[], float] = time.monotonic):
        self.id = id
        self.task = task
        self.references = references
        self.llm = llm
        self.voice = voice
        self.hashes = hashes
        self.settings = settings
        self._emit_fn = emit
        self.clock = clock

        self.timeline = Timeline()
        self.divergences: list[Divergence] = []
        self.pending = False
        self.duplicate_hits = 0
        self.mismatch_chunks = 0

        self._steps = candidate_steps(task, library)
        self._wake = WakePolicy(settings.question_cooldown_s, settings.question_min_interval_s)
        self._nudge = NudgePolicy(settings.nudge_cooldown_s)
        self._nudge_audio: bytes | None = None

        self._frames: list[Frame] = []
        self._imu: list[dict] = []
        self._next_index = 0
        self._captions: dict[int, str] = {}
        self._caption_tasks: set[asyncio.Task] = set()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._monitor_lock = asyncio.Lock()

        self._line_t = 0.0  # stamp for AGENT / WORKER lines (see module docstring)
        self._last_index = -1
        self._prev_key: tuple | None = None  # (index, t, hash, matches) of the last keyframe
        self._open: Divergence | None = None
        self._timeout_task: asyncio.Task | None = None
        self._finishing = False

    # ------------------------------------------------------------ inputs

    async def add_frame(self, t: float, jpeg: bytes) -> None:
        await self._emit({"type": "frame", "t": t, "jpeg_b64": base64.b64encode(jpeg).decode()})
        self._frames.append((t, jpeg))
        if len(self._frames) >= self.settings.chunk_frames:
            self._spawn_chunk()

    def add_imu(self, samples: list[dict]) -> None:
        self._imu.extend(samples)

    async def add_answer(self, audio: bytes, mime: str) -> None:
        if self._timeout_task:
            self._timeout_task.cancel()
            self._timeout_task = None
        try:
            text = (await self.voice.stt(audio, mime)).strip()
        except Exception as e:  # noqa: BLE001
            log.warning("stt failed: %r", e)
            text = ""
        text = text or NO_ANSWER
        d = self._open
        t = self._line_t
        self.timeline.add_worker(t, text)
        await self._emit({"type": "worker_said", "t": t, "text": text})
        if d:
            d.answer = text
        decision = await self._monitor("worker_answer")
        if d:
            d.explained = decision.explained
            if self._open is d:
                self._open = None
                self.pending = False

    async def wait_idle(self) -> None:
        """Wait until every spawned chunk has been captioned and monitored."""
        while self._caption_tasks:
            await asyncio.gather(*list(self._caption_tasks), return_exceptions=True)
        await self._queue.join()

    async def finish(self) -> dict:
        """Build the report. Waits at most ``settings.finish_timeout_s`` for in-flight chunks;
        chunks still unfinished then are left out of the report."""
        self._finishing = True
        await self._emit({"type": "session_ending"})
        if len(self._frames) >= MIN_PARTIAL_FRAMES:
            self._spawn_chunk()
        try:
            await asyncio.wait_for(self.wait_idle(), self.settings.finish_timeout_s)
        except TimeoutError:
            log.warning("finish: chunks still in flight after %.0f s; reporting without them",
                        self.settings.finish_timeout_s)
            for task in list(self._caption_tasks):
                task.cancel()
        if self._timeout_task:
            self._timeout_task.cancel()
            self._timeout_task = None
        if self._open:
            self._open.explained = False
            self._open = None
            self.pending = False
        if self._worker:
            self._worker.cancel()
        model = await self.llm.final_report(self.task, self.references, self.timeline)
        stats = signal_stats(self.timeline.chunks, self.divergences, self.duplicate_hits,
                             self.mismatch_chunks)
        report = build_report(model, stats, self.divergences, self.timeline.render())
        await self._emit({"type": "report", "report": report})
        return report

    # ------------------------------------------------------------ chunk pipeline

    def _spawn_chunk(self) -> None:
        frames, self._frames = self._frames, []
        imu, self._imu = self._imu, []
        index = self._next_index
        self._next_index += 1
        if self._worker is None:
            self._worker = asyncio.create_task(self._monitor_worker())
        task = asyncio.create_task(self._caption_chunk(index, frames, imu))
        self._caption_tasks.add(task)
        task.add_done_callback(self._caption_tasks.discard)

    async def _caption_chunk(self, index: int, frames: list[Frame], imu: list[dict]) -> None:
        try:
            result = await self.llm.caption([j for _, j in frames], self.task, self._steps,
                                            self._captions.get(index - 1))
            result = {**CAPTION_FALLBACK, **result}
        except Exception as e:  # noqa: BLE001
            log.warning("caption chunk %d failed: %r", index, e)
            result = dict(CAPTION_FALLBACK)
        if result["caption"] != CAPTION_FALLBACK["caption"]:
            self._captions[index] = result["caption"]
        await self._queue.put((index, frames, imu, result))

    async def _monitor_worker(self) -> None:
        """Single consumer: processes captioned chunks strictly in index order."""
        buffer: dict[int, tuple] = {}
        expected = 0
        while True:
            item = await self._queue.get()
            buffer[item[0]] = item
            while expected in buffer:
                try:
                    await self._process_chunk(*buffer.pop(expected))
                except Exception:  # noqa: BLE001 - one bad chunk must not stop the session
                    log.exception("processing chunk %d failed", expected)
                expected += 1
            self._queue.task_done()

    async def _process_chunk(self, index: int, frames: list[Frame], imu: list[dict],
                             result: dict) -> None:
        tags: list[str] = []
        if result["recapture_suspected"]:
            tags.append("[recapture?]")
        signals: dict[str, Any] = {"visual_motion": None, "imu_energy": None, "duplicate": None}
        try:
            mid_t, mid_jpeg = frames[len(frames) // 2]
            dup = self._duplicate_match(index, mid_t, mid_jpeg)
            if dup:
                tags.append(f"[duplicate of {dup['session_id']} {fmt(dup['t'])}]")
                self.duplicate_hits += 1
            motion = visual_motion([j for _, j in frames])
            energy = imu_energy(imu)
            if imu_mismatch(motion, energy):
                tags.append("[visual motion without head motion]")
                self.mismatch_chunks += 1
            signals = {"visual_motion": motion, "imu_energy": energy, "duplicate": dup}
        except Exception as e:  # noqa: BLE001 - e.g. an undecodable frame
            log.warning("signals for chunk %d failed: %r", index, e)

        chunk = Chunk(
            index=index, t_start=frames[0][0], t_end=frames[-1][0] + 1.0,
            caption=" ".join([result["caption"], *tags]), step=result["step"],
            confidence=result["confidence"], objects=result["objects"],
            hands_visible=result["hands_visible"], activity=result["activity"],
            recapture_suspected=result["recapture_suspected"],
        )
        self.timeline.add_chunk(chunk)
        self._line_t = frames[-1][0]
        self._last_index = index
        await self._emit({"type": "chunk", "chunk": asdict(chunk), "signals": signals})

        now = self.clock()
        nudged = not self._finishing and self._nudge.should_nudge(now, chunk.hands_visible)
        if nudged:
            await self._say_nudge()

        decision = await self._monitor("chunk")
        question = decision.question.strip()
        # Never stack a question on the nudge that just played; the monitor can ask next chunk.
        if (decision.wake and question and not self._finishing and not nudged
                and self._wake.can_ask(now, self.pending)):
            await self._ask(now, decision, question)

    def _duplicate_match(self, index: int, t: float, jpeg: bytes) -> dict | None:
        h = frame_hash(jpeg)
        if h is None:  # low-texture frame: no identity to match
            self._prev_key = None
            return None
        matches = self.hashes.check_and_add(self.id, index, t, h)
        prev, self._prev_key = self._prev_key, (index, t, h, matches)
        if prev is None or prev[0] != index - 1:
            return None
        if h - prev[2] <= STATIC_SCENE_DISTANCE:  # static view: matching proves nothing
            return None
        step = t - prev[1]
        for m in sorted(matches, key=lambda m: m["distance"]):
            for p in prev[3]:
                if (m["session_id"] == p["session_id"] and m["index"] == p["index"] + 1
                        and abs((m["t"] - p["t"]) - step) <= DUP_TOLERANCE_S):
                    return m
        return None

    # ------------------------------------------------------------ monitor and voice

    async def _monitor(self, trigger: str) -> MonitorDecision:
        async with self._monitor_lock:
            try:
                decision = await self.llm.monitor(self.task, self.references, self.timeline,
                                                  trigger)
            except Exception as e:  # noqa: BLE001
                log.warning("monitor failed: %r", e)
                decision = MonitorDecision.unavailable()
        fields = asdict(decision)
        fallback = fields.pop("fallback")
        # fallback: the decision is a neutral stand-in, so clients keep their last status.
        await self._emit({"type": "monitor", "index": self._last_index, "decision": fields,
                          "fallback": fallback})
        return decision

    async def _ask(self, now: float, decision: MonitorDecision, question: str) -> None:
        self._wake.record_question(now)
        t = self._line_t
        d = Divergence(t=t, type=decision.divergence_type, reason=decision.reason,
                       question=question)
        self.divergences.append(d)
        self._open = d
        self.pending = True
        self.timeline.add_agent(t, question)
        audio = await self._tts(question)
        await self._emit({"type": "agent_say", "kind": "question", "text": question,
                          "audio_b64": _b64(audio)})
        if self._open is d:
            self._timeout_task = asyncio.create_task(self._answer_timeout(d))

    async def _answer_timeout(self, d: Divergence) -> None:
        await asyncio.sleep(self.settings.answer_timeout_s)
        if self._open is not d:
            return
        self._open = None
        self.pending = False
        self._timeout_task = None
        d.explained = False
        t = self._line_t
        self.timeline.add_worker(t, NO_ANSWER)
        await self._emit({"type": "worker_said", "t": t, "text": NO_ANSWER})

    async def _say_nudge(self) -> None:
        # Not added to the timeline: the monitor reads AGENT lines as its own questions.
        if self._nudge_audio is None:
            self._nudge_audio = await self._tts(NUDGE_TEXT)
        await self._emit({"type": "agent_say", "kind": "nudge", "text": NUDGE_TEXT,
                          "audio_b64": _b64(self._nudge_audio)})

    async def _tts(self, text: str) -> bytes | None:
        try:
            return await self.voice.tts(text)
        except Exception as e:  # noqa: BLE001 - text still reaches phone and dashboard
            log.warning("tts failed: %r", e)
            return None

    async def _emit(self, event: dict) -> None:
        try:
            await self._emit_fn(event)
        except Exception as e:  # noqa: BLE001 - a dead client must not stop the session
            log.warning("emit %s failed: %r", event.get("type"), e)


def _b64(audio: bytes | None) -> str | None:
    return base64.b64encode(audio).decode() if audio else None
