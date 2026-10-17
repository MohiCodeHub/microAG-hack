import asyncio
import base64
import io

import numpy as np
import pytest
from PIL import Image

from app.config import Settings  # noqa: F401
from tests.legacy_settings import LegacySettings
from app.llm import CAPTION_FALLBACK, MonitorDecision
from app.session import NUDGE_TEXT, Session
from app.signals import HashStore, frame_hash
from app.tasks import Step, Task

TASK = Task("tea", "Make tea", [Step("fill_kettle", "Fill the kettle"), Step("boil", "Boil water")])


def jpeg(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = (rng.random((64, 64)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).resize((128, 128), Image.NEAREST).save(buf, "JPEG")
    return buf.getvalue()


GRAY = jpeg(0)


def cap(**kw):
    d = {"caption": "fills kettle", "step": "fill_kettle", "confidence": 0.9, "objects": ["kettle"],
         "hands_visible": "both", "activity": "active", "recapture_suspected": False}
    d.update(kw)
    return d


def decision(**kw):
    d = MonitorDecision.default()
    for k, v in kw.items():
        setattr(d, k, v)
    return d


class FakeLLM:
    def __init__(self, caption=None, monitor=None, report=None):
        self._caption = caption or (lambda index, n: cap())
        self._monitor = monitor or (lambda trigger, timeline: decision())
        self._report = report or {"segments": [], "fraud_score": 40, "verdict": "accept",
                                  "reasons": []}
        self.caption_calls = 0
        self.monitor_calls: list[tuple[str, str]] = []  # (trigger, last rendered line)

    async def caption(self, jpegs, task, steps, prev_caption):
        i = self.caption_calls
        self.caption_calls += 1
        r = self._caption(i, len(jpegs))
        if asyncio.iscoroutine(r):
            r = await r
        return r

    async def monitor(self, task, references, timeline, trigger):
        self.monitor_calls.append((trigger, timeline.render().splitlines()[-1]))
        return self._monitor(trigger, timeline)

    async def final_report(self, task, references, timeline):
        return dict(self._report)


class FakeVoice:
    def __init__(self, stt_text="", tts_fail=False):
        self.stt_text = stt_text
        self.tts_fail = tts_fail
        self.tts_calls: list[str] = []

    async def tts(self, text):
        self.tts_calls.append(text)
        if self.tts_fail:
            raise RuntimeError("tts down")
        return b"mp3:" + text.encode()

    async def stt(self, audio, mime):
        return self.stt_text


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Harness:
    def __init__(self, tmp_path, llm=None, voice=None, settings=None, session_id="s1",
                 hashes=None):
        self.events: list[dict] = []
        self.clock = Clock()
        self.llm = llm or FakeLLM()
        self.voice = voice or FakeVoice()

        async def emit(ev):
            self.events.append(ev)

        self.session = Session(
            session_id, TASK, {"tea": TASK}, ["ref timeline"], self.llm, self.voice,
            hashes or HashStore(tmp_path / "hashes.json"), settings or LegacySettings(), emit,
            clock=self.clock,
        )
        self.t = 0

    async def chunk(self, frames=None, advance=10.0):
        """Send one chunk worth of frames and wait until it is fully processed."""
        for j in frames or [GRAY] * 10:
            await self.session.add_frame(float(self.t), j)
            self.t += 1
        await self.session.wait_idle()
        self.clock.now += advance

    def of(self, type_):
        return [e for e in self.events if e["type"] == type_]


async def test_ten_frames_make_one_chunk_and_monitor_call(tmp_path):
    h = Harness(tmp_path)
    await h.chunk()
    assert len(h.of("frame")) == 10
    assert h.of("frame")[0]["jpeg_b64"] == base64.b64encode(GRAY).decode()
    chunks = h.of("chunk")
    assert len(chunks) == 1
    c = chunks[0]["chunk"]
    assert (c["index"], c["t_start"], c["t_end"], c["caption"]) == (0, 0.0, 10.0, "fills kettle")
    assert h.llm.monitor_calls == [("chunk", "00:00–00:10 fills kettle [fill_kettle]")]
    assert h.of("monitor")[0]["index"] == 0
    assert h.of("monitor")[0]["decision"]["status"] == "ok"
    assert h.of("agent_say") == []


async def test_nine_frames_make_no_chunk(tmp_path):
    h = Harness(tmp_path)
    for t in range(9):
        await h.session.add_frame(float(t), GRAY)
    await h.session.wait_idle()
    assert h.of("chunk") == [] and h.llm.caption_calls == 0


async def test_chunks_monitored_in_order_when_captions_finish_out_of_order(tmp_path):
    async def slow_first(i, n):
        await asyncio.sleep(0.2 if i == 0 else 0.0)
        return cap(caption=f"chunk {i}")

    h = Harness(tmp_path, llm=FakeLLM(caption=slow_first))
    for t in range(20):
        await h.session.add_frame(float(t), GRAY)
    await h.session.wait_idle()
    assert [e["chunk"]["index"] for e in h.of("chunk")] == [0, 1]
    assert [e["index"] for e in h.of("monitor")] == [0, 1]
    assert [line for _, line in h.llm.monitor_calls] == [
        "00:00–00:10 chunk 0 [fill_kettle]", "00:10–00:20 chunk 1 [fill_kettle]"]


async def test_wake_emits_question_with_audio_and_respects_cooldown(tmp_path):
    llm = FakeLLM(monitor=lambda trig, tl: decision(status="diverge", divergence_type="off_script",
                                                    reason="not tea", wake=True,
                                                    question="What are you doing?"))
    h = Harness(tmp_path, llm=llm)
    for _ in range(3):
        await h.chunk(advance=10)
    says = h.of("agent_say")
    assert len(says) == 1
    assert says[0]["kind"] == "question"
    assert says[0]["text"] == "What are you doing?"
    assert base64.b64decode(says[0]["audio_b64"]) == b"mp3:What are you doing?"
    assert '00:09 AGENT: "What are you doing?"' in h.session.timeline.render()
    [d] = h.session.divergences
    assert (d.type, d.reason, d.question, d.answer, d.explained) == (
        "off_script", "not tea", "What are you doing?", None, None)


async def test_wake_with_empty_question_is_ignored(tmp_path):
    llm = FakeLLM(monitor=lambda trig, tl: decision(status="diverge", wake=True, question="  "))
    h = Harness(tmp_path, llm=llm)
    await h.chunk()
    assert h.of("agent_say") == [] and h.session.divergences == []


async def test_tts_failure_still_emits_question_text(tmp_path):
    llm = FakeLLM(monitor=lambda trig, tl: decision(status="diverge", wake=True, question="Hi?"))
    h = Harness(tmp_path, llm=llm, voice=FakeVoice(tts_fail=True))
    await h.chunk()
    [say] = h.of("agent_say")
    assert say["text"] == "Hi?" and say["audio_b64"] is None
    assert 'AGENT: "Hi?"' in h.session.timeline.render()


async def test_hands_none_triggers_nudge_without_monitor_wake(tmp_path):
    h = Harness(tmp_path, llm=FakeLLM(caption=lambda i, n: cap(hands_visible="none")))
    await h.chunk(advance=10)
    await h.chunk(advance=15)  # within 20 s nudge cooldown
    await h.chunk()
    says = h.of("agent_say")
    assert [s["kind"] for s in says] == ["nudge", "nudge"]
    assert says[0]["text"] == NUDGE_TEXT
    assert base64.b64decode(says[0]["audio_b64"]) == b"mp3:" + NUDGE_TEXT.encode()
    assert h.voice.tts_calls == [NUDGE_TEXT]  # synthesized once, then cached
    assert h.session.divergences == []


async def test_no_question_in_the_chunk_a_nudge_just_played(tmp_path):
    llm = FakeLLM(caption=lambda i, n: cap(hands_visible="none" if i == 0 else "both"),
                  monitor=lambda trig, tl: decision(status="diverge", wake=True, question="Hm?"))
    h = Harness(tmp_path, llm=llm)
    await h.chunk()
    assert [s["kind"] for s in h.of("agent_say")] == ["nudge"]
    assert h.session.divergences == []
    await h.chunk()  # no nudge this time, so the monitor may ask
    assert [s["kind"] for s in h.of("agent_say")] == ["nudge", "question"]


async def test_nudge_tts_failure_retries_next_time(tmp_path):
    voice = FakeVoice(tts_fail=True)
    h = Harness(tmp_path, llm=FakeLLM(caption=lambda i, n: cap(hands_visible="none")), voice=voice)
    await h.chunk(advance=25)
    voice.tts_fail = False
    await h.chunk()
    says = h.of("agent_say")
    assert says[0]["audio_b64"] is None
    assert says[1]["audio_b64"] is not None
    assert voice.tts_calls == [NUDGE_TEXT, NUDGE_TEXT]


def waking_llm(explained=True):
    def monitor(trigger, tl):
        if trigger == "worker_answer":
            return decision(explained=explained)
        return decision(status="diverge", divergence_type="idle", reason="idle", wake=True,
                        question="Everything okay?")
    return FakeLLM(monitor=monitor)


async def test_answer_marks_divergence_explained(tmp_path):
    h = Harness(tmp_path, llm=waking_llm(), voice=FakeVoice(stt_text="waiting for the kettle"))
    await h.chunk()
    await h.session.add_answer(b"audio", "audio/webm")
    assert h.llm.monitor_calls[-1] == ("worker_answer", '00:09 WORKER: "waiting for the kettle"')
    [said] = h.of("worker_said")
    assert said["text"] == "waiting for the kettle"
    [d] = h.session.divergences
    assert d.answer == "waiting for the kettle" and d.explained is True
    assert not h.session.pending
    # pending cleared: a later wake is possible once the cooldown passed
    h.clock.now = 100
    await h.chunk()
    assert len(h.of("agent_say")) == 2


async def test_empty_stt_records_no_answer(tmp_path):
    h = Harness(tmp_path, llm=waking_llm(explained=False), voice=FakeVoice(stt_text=""))
    await h.chunk()
    await h.session.add_answer(b"audio", "audio/webm")
    assert 'WORKER: "(no answer)"' in h.session.timeline.render()
    [d] = h.session.divergences
    assert d.answer == "(no answer)" and d.explained is False


async def test_answer_timeout_marks_unexplained(tmp_path):
    settings = LegacySettings(answer_timeout_s=0.05)
    h = Harness(tmp_path, llm=waking_llm(), settings=settings)
    await h.chunk()
    assert h.session.pending
    await asyncio.sleep(0.15)
    assert not h.session.pending
    [d] = h.session.divergences
    assert d.explained is False and d.answer is None
    assert 'WORKER: "(no answer)"' in h.session.timeline.render()
    assert h.of("worker_said")[0]["text"] == "(no answer)"
    # answer timeout did not consult the monitor
    assert [t for t, _ in h.llm.monitor_calls] == ["chunk"]


async def test_caption_failure_does_not_crash_session(tmp_path):
    def caption(i, n):
        if i == 0:
            return dict(CAPTION_FALLBACK)
        if i == 1:
            raise RuntimeError("boom")
        return cap()

    h = Harness(tmp_path, llm=FakeLLM(caption=caption))
    for _ in range(3):
        await h.chunk()
    caps = [e["chunk"]["caption"] for e in h.of("chunk")]
    assert caps == ["(caption unavailable)", "(caption unavailable)", "fills kettle"]
    assert len(h.of("monitor")) == 3


async def test_recapture_and_imu_mismatch_tags(tmp_path):
    h = Harness(tmp_path, llm=FakeLLM(caption=lambda i, n: cap(recapture_suspected=True)))
    h.session.add_imu([{"gx": 0.0, "gy": 0.0, "gz": 0.0}] * 20)
    await h.chunk(frames=[jpeg(i) for i in range(10)])  # strong visual motion, still IMU
    c = h.of("chunk")[0]["chunk"]
    assert "[recapture?]" in c["caption"]
    assert "[visual motion without head motion]" in c["caption"]
    assert "[recapture?]" in h.llm.monitor_calls[0][1]


async def test_duplicate_tag_needs_two_consecutive_advancing_matches(tmp_path):
    store = HashStore(tmp_path / "hashes.json")
    a, b, c, d = jpeg(101), jpeg(102), jpeg(103), jpeg(104)
    store.check_and_add("old", 6, 65.0, frame_hash(a))
    store.check_and_add("old", 7, 75.0, frame_hash(b))
    store.check_and_add("old", 9, 85.0, frame_hash(c))   # time advances, but not chunk 8
    store.check_and_add("old", 20, 200.0, frame_hash(d))
    h = Harness(tmp_path, hashes=store)
    await h.chunk(frames=[a] * 10)   # single match: ignored
    await h.chunk(frames=[b] * 10)   # old chunk 6 then 7, 10 s later: duplicate
    await h.chunk(frames=[c] * 10)   # old chunk 7 then 9: not the same consecutive pair
    await h.chunk(frames=[d] * 10)   # matches "old" but jumps 115 s: ignored
    caps = [e["chunk"]["caption"] for e in h.of("chunk")]
    assert caps[0] == "fills kettle"
    assert caps[1] == "fills kettle [duplicate of old 01:15]"
    assert caps[2:] == ["fills kettle", "fills kettle"]
    report = await h.session.finish()
    assert report["signals"]["duplicate_hits"] == 1


def static_scene(seed: int) -> bytes:
    """The same textured view each time, with a little sensor noise (pHash moves 0-4 bits)."""
    base = (np.random.default_rng(7).random((8, 8)) * 255).astype(np.uint8)
    img = np.asarray(Image.fromarray(base).resize((128, 128), Image.BILINEAR), dtype=np.float32)
    noise = np.random.default_rng(seed).normal(0, 14.0, img.shape)
    buf = io.BytesIO()
    Image.fromarray(np.clip(img + noise, 0, 255).astype(np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


async def run_session(tmp_path, store, session_id, chunks):
    h = Harness(tmp_path, hashes=store, session_id=session_id)
    for frames in chunks:
        await h.chunk(frames=frames)
    return [e["chunk"]["caption"] for e in h.of("chunk")]


async def test_honest_static_scene_repeat_is_not_a_duplicate(tmp_path):
    store = HashStore(tmp_path / "hashes.json")
    for n, sid in enumerate(["rehearsal1", "rehearsal2", "live"]):
        caps = await run_session(tmp_path, store, sid, [
            [static_scene(100 * n + 10 * k + i) for i in range(10)] for k in range(5)])
        assert not any("[duplicate of" in c for c in caps), (sid, caps)


async def test_true_replay_of_earlier_footage_is_flagged(tmp_path):
    store = HashStore(tmp_path / "hashes.json")
    footage = [[jpeg(200 + 10 * k + i) for i in range(10)] for k in range(4)]
    assert not any("[duplicate of" in c for c in await run_session(tmp_path, store, "old", footage))
    caps = await run_session(tmp_path, store, "replay", footage)
    assert caps[0] == "fills kettle"
    assert caps[1:] == [f"fills kettle [duplicate of old {t}]" for t in ("00:15", "00:25", "00:35")]


async def test_low_texture_frames_never_flag_duplicate(tmp_path):
    flat = io.BytesIO()
    Image.new("L", (128, 128), 120).save(flat, "JPEG")
    footage = [[flat.getvalue()] * 10 for _ in range(3)]
    store = HashStore(tmp_path / "hashes.json")
    await run_session(tmp_path, store, "old", footage)
    caps = await run_session(tmp_path, store, "new", footage)
    assert not any("[duplicate of" in c for c in caps)


async def test_finish_builds_and_emits_report(tmp_path):
    h = Harness(tmp_path, llm=FakeLLM(caption=lambda i, n: cap(activity="idle")))
    for t in range(10):
        await h.session.add_frame(float(t), GRAY)
    report = await h.session.finish()  # waits for the in-flight chunk
    assert report["signals"]["idle_fraction"] == 1.0
    assert report["rule_score"] == 40
    assert report["fraud_score"] == 40  # round(0.5*40 + 0.5*40)
    assert report["timeline"] == "00:00–00:10 fills kettle [fill_kettle]"
    assert h.of("report") == [{"type": "report", "report": report}]


async def test_monitor_event_marks_fallback_decisions(tmp_path):
    def monitor(trig, tl):
        if len(tl.chunks) == 2:
            raise RuntimeError("monitor down")
        return decision(status="watch", progress=0.5)

    h = Harness(tmp_path, llm=FakeLLM(monitor=monitor))
    await h.chunk()
    await h.chunk()
    ok, failed = h.of("monitor")
    assert ok["fallback"] is False and ok["decision"]["status"] == "watch"
    assert failed["fallback"] is True
    assert "fallback" not in failed["decision"]


async def test_finish_is_bounded_when_a_caption_hangs(tmp_path):
    hang = asyncio.Event()

    async def caption(i, n):
        if i == 1:
            await hang.wait()  # never set
        return cap()

    h = Harness(tmp_path, llm=FakeLLM(caption=caption), settings=LegacySettings(finish_timeout_s=0.2))
    await h.chunk()
    for t in range(10, 20):
        await h.session.add_frame(float(t), GRAY)
    report = await asyncio.wait_for(h.session.finish(), 3)
    assert report["timeline"] == "00:00–00:10 fills kettle [fill_kettle]"
    types = [e["type"] for e in h.events]
    assert types.index("session_ending") < types.index("report")
    assert types.count("session_ending") == 1


async def test_finish_with_pending_question_marks_unexplained(tmp_path):
    h = Harness(tmp_path, llm=waking_llm())
    await h.chunk()
    report = await h.session.finish()
    assert report["divergences"][0]["explained"] is False
    assert report["signals"]["unexplained_divergences"] == 1
