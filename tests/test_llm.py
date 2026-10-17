import json
from types import SimpleNamespace

from app.config import Settings
from app.llm import LLM, MonitorDecision
from app.tasks import Step, Task
from app.timeline import Chunk, Timeline

REFS_END = "=== SESSION TIMELINE"


class FakeClient:
    def __init__(self, output_text="{}", exc: Exception | None = None):
        self.calls: list[dict] = []
        self.output_text = output_text
        self.exc = exc
        self.responses = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return SimpleNamespace(output_text=self.output_text)


TASK = Task("tea", "Make tea", [Step("fill_kettle", "Fill the kettle"), Step("boil", "Boil water")])
STEPS = TASK.steps + [Step("using_phone", "using a phone"), Step("other", "something else")]


def mk(i, step="fill_kettle", cap="fills kettle at the sink"):
    return Chunk(index=i, t_start=i * 10, t_end=i * 10 + 10, caption=cap, step=step,
                 confidence=0.9, objects=["kettle"], hands_visible="both",
                 activity="active", recapture_suspected=False)


def parts(call):
    return [p for msg in call["input"] for p in msg["content"]]


def text_of(call):
    return "\n".join(p["text"] for p in parts(call) if p["type"] == "input_text")


GOOD_DECISION = {
    "current_step": "boil", "progress": 0.5, "status": "watch", "divergence_type": "idle",
    "reason": "no hand activity for 20 s", "wake": False, "question": "", "fraud_risk": 0.3,
    "explained": None,
}


async def test_caption_sends_images_low_detail_and_schema():
    payload = {"caption": "fills kettle", "step": "fill_kettle", "confidence": 0.8,
               "objects": ["kettle"], "hands_visible": "both", "activity": "active",
               "recapture_suspected": False}
    fc = FakeClient(json.dumps(payload))
    out = await LLM(fc, Settings()).caption([b"\xff\xd8jpeg"] * 10, TASK, STEPS, "prev line")
    assert out == payload
    call = fc.calls[0]
    imgs = [p for p in parts(call) if p["type"] == "input_image"]
    assert len(imgs) == 10
    assert all(p["detail"] == "low" for p in imgs)
    assert all(p["image_url"].startswith("data:image/jpeg;base64,") for p in imgs)
    assert call["text"]["format"]["strict"] is True
    assert call["text"]["format"]["type"] == "json_schema"
    assert call["timeout"] == Settings().llm_timeout_s
    prompt = text_of(call)
    for s in STEPS:
        assert s.id in prompt
    assert "prev line" in prompt


async def test_caption_fallback_on_exception():
    fc = FakeClient(exc=TimeoutError("slow"))
    out = await LLM(fc, Settings()).caption([b"x"] * 10, TASK, STEPS, None)
    assert out["caption"] == "(caption unavailable)"
    assert out["step"] == "other"
    assert out["confidence"] == 0.0


async def test_monitor_parses_decision():
    fc = FakeClient(json.dumps(GOOD_DECISION))
    tl = Timeline()
    tl.add_chunk(mk(0))
    d = await LLM(fc, Settings()).monitor(TASK, ["00:00–00:10 ref [fill_kettle]"], tl, "chunk")
    assert d == MonitorDecision(**GOOD_DECISION)
    assert fc.calls[0]["text"]["format"]["strict"] is True


async def test_monitor_default_on_bad_json():
    fc = FakeClient("not json {")
    tl = Timeline()
    tl.add_chunk(mk(0))
    d = await LLM(fc, Settings()).monitor(TASK, [], tl, "chunk")
    assert d == MonitorDecision.default()
    assert d.status == "ok" and d.wake is False and d.explained is None and d.question == ""


async def test_monitor_prompt_order_prefix_stable():
    fc = FakeClient(json.dumps(GOOD_DECISION))
    llm = LLM(fc, Settings())
    refs = ["00:00–00:10 ref fills kettle [fill_kettle]", "00:00–00:10 ref two [fill_kettle]"]
    tl = Timeline()
    tl.add_chunk(mk(0))
    await llm.monitor(TASK, refs, tl, "chunk")
    tl.add_chunk(mk(1, "boil", "switches kettle on"))
    tl.add_worker(21, "just waiting for it")
    await llm.monitor(TASK, refs, tl, "worker_answer")
    a, b = text_of(fc.calls[0]), text_of(fc.calls[1])
    ia, ib = a.index(REFS_END), b.index(REFS_END)
    assert a[:ia] == b[:ib]
    assert all(r in a[:ia] for r in refs)
    # session timeline is append-only: first call's timeline is a prefix of the second's
    tl_a = a[ia:a.index("=== LATEST")]
    tl_b = b[ib:b.index("=== LATEST")]
    assert tl_b.startswith(tl_a.rstrip("\n"))
    # newest item comes last
    assert b.rstrip().endswith('"just waiting for it"')
    assert a.rstrip().endswith("[fill_kettle]")


async def test_monitor_mentions_no_references():
    fc = FakeClient(json.dumps(GOOD_DECISION))
    tl = Timeline()
    tl.add_chunk(mk(0))
    await LLM(fc, Settings()).monitor(TASK, [], tl, "chunk")
    assert "No reference recordings are available; judge against the step list only." in text_of(fc.calls[0])


async def test_monitor_sends_no_images():
    fc = FakeClient(json.dumps(GOOD_DECISION))
    tl = Timeline()
    tl.add_chunk(mk(0))
    await LLM(fc, Settings()).monitor(TASK, ["ref"], tl, "chunk")
    assert all(p["type"] == "input_text" for p in parts(fc.calls[0]))


async def test_monitor_uses_monitor_model():
    fc = FakeClient(json.dumps(GOOD_DECISION))
    tl = Timeline()
    tl.add_chunk(mk(0))
    await LLM(fc, Settings(monitor_model="gpt-6-astra")).monitor(TASK, [], tl, "chunk")
    assert fc.calls[0]["model"] == "gpt-6-astra"


async def test_final_report_parses_model_output():
    rep = {"segments": [{"start": "00:00", "end": "00:20", "step": "fill_kettle",
                         "caption": "fills kettle", "source": "model"}],
           "fraud_score": 12, "verdict": "accept", "reasons": ["matches references"]}
    fc = FakeClient(json.dumps(rep))
    tl = Timeline()
    tl.add_chunk(mk(0))
    assert await LLM(fc, Settings()).final_report(TASK, [], tl) == rep


async def test_final_report_fallback_merges_steps():
    fc = FakeClient(exc=RuntimeError("down"))
    tl = Timeline()
    for i, s in enumerate(["a", "a", "b"]):
        tl.add_chunk(mk(i, s, f"cap {i}"))
    out = await LLM(fc, Settings()).final_report(TASK, [], tl)
    assert out["segments"] == [
        {"start": "00:00", "end": "00:20", "step": "a", "caption": "cap 0", "source": "model"},
        {"start": "00:20", "end": "00:30", "step": "b", "caption": "cap 2", "source": "model"},
    ]
    assert out["fraud_score"] is None
    assert out["reasons"] == ["report model unavailable"]
    assert out["verdict"] == "review"
