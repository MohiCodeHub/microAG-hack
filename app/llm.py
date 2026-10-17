"""LLM layer: stateless vision captioner, stateful text-only monitor, final report."""
from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, fields
from typing import Any

from app.tasks import Step, Task
from app.timeline import Timeline, fmt

log = logging.getLogger(__name__)

# Passed to every responses.create call. Set to None if a model rejects it.
REASONING: dict | None = {"effort": "low"}

CaptionResult = dict

CAPTION_FALLBACK: CaptionResult = {
    "caption": "(caption unavailable)", "step": "other", "confidence": 0.0, "objects": [],
    "hands_visible": "both", "activity": "active", "recapture_suspected": False,
}

DIVERGENCE_TYPES = ["none", "off_script", "wrong_task", "out_of_order", "padding", "idle", "recapture"]


@dataclass
class MonitorDecision:
    current_step: str
    progress: float
    status: str
    divergence_type: str
    reason: str
    wake: bool
    question: str
    fraud_risk: float
    explained: bool | None
    fallback: bool = False  # True when the monitor call failed; not part of the model output

    @classmethod
    def default(cls) -> "MonitorDecision":
        return cls(current_step="", progress=0.0, status="ok", divergence_type="none",
                   reason="", wake=False, question="", fraud_risk=0.0, explained=None)

    @classmethod
    def unavailable(cls) -> "MonitorDecision":
        """Neutral stand-in used when the monitor call fails."""
        d = cls.default()
        d.fallback = True
        return d


MODEL_FIELDS = [f.name for f in fields(MonitorDecision) if f.name != "fallback"]


# ---------------------------------------------------------------- schemas

def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


def _caption_schema(step_ids: list[str]) -> dict:
    # Field order matters: free-text caption before the step mapping.
    return _obj({
        "caption": {"type": "string"},
        "step": {"type": "string", "enum": step_ids},
        "confidence": {"type": "number"},
        "objects": {"type": "array", "items": {"type": "string"}},
        "hands_visible": {"type": "string", "enum": ["both", "one", "none"]},
        "activity": {"type": "string", "enum": ["active", "idle"]},
        "recapture_suspected": {"type": "boolean"},
    })


def _monitor_schema(step_ids: list[str]) -> dict:
    return _obj({
        "current_step": {"type": "string", "enum": step_ids},
        "progress": {"type": "number"},
        "status": {"type": "string", "enum": ["ok", "watch", "diverge"]},
        "divergence_type": {"type": "string", "enum": DIVERGENCE_TYPES},
        "reason": {"type": "string"},
        "wake": {"type": "boolean"},
        "question": {"type": "string"},
        "fraud_risk": {"type": "number"},
        "explained": {"type": ["boolean", "null"]},
    })


def _report_schema(step_ids: list[str]) -> dict:
    segment = _obj({
        "start": {"type": "string"},
        "end": {"type": "string"},
        "step": {"type": "string", "enum": step_ids},
        "caption": {"type": "string"},
        "source": {"type": "string", "enum": ["model", "worker"]},
    })
    return _obj({
        "segments": {"type": "array", "items": segment},
        "fraud_score": {"type": ["number", "null"]},
        "verdict": {"type": "string", "enum": ["accept", "review", "reject"]},
        "reasons": {"type": "array", "items": {"type": "string"}},
    })


# ---------------------------------------------------------------- prompts

CAPTION_INSTRUCTIONS = """\
You caption egocentric (head-mounted / chest-mounted phone) video of a person doing a manual task.
You get 10 frames sampled at 1 frame per second, i.e. the last 10 seconds, in order.

Fill the fields in order:
- caption: one short factual sentence (max ~15 words) describing what the hands actually do across \
the 10 seconds. Describe what you SEE, not what the task says should happen. Present tense, no \
speculation, no "the user appears to".
- step: the id from the candidate list that best matches what you saw. The list deliberately \
contains steps from other, unrelated tasks; do not assume the claimed task is being done. Use \
"other" if nothing fits.
- confidence: 0-1, how sure you are of the step.
- objects: the main objects being handled or used (max 5, short nouns).
- hands_visible: "both", "one" or "none" - how many of the person's hands are in view for most frames.
- activity: "idle" if the hands are not doing task work for most of the 10 s (resting, waiting, \
just standing, scrolling a phone), else "active".
- recapture_suspected: true only if the frames look like a camera filming a SCREEN or a printed \
image rather than a real scene: visible monitor/phone bezel, moire or pixel grid, scan lines, screen \
glare, flat image with no depth, UI overlays. Otherwise false.
"""

MONITOR_INSTRUCTIONS = """\
You are a quiet, friendly supervisor for a gig worker who is filming themselves doing a task for a \
data-collection app. Every 10 seconds a captioner turns the video into one text line. You read the \
running text timeline, compare it to reference timelines of other people doing the SAME task, and \
decide whether something is off. You never see video, only text.

How to read the timelines:
- Chunk line: `mm:ss–mm:ss caption [step_id]` - what the captioner saw in that 10 s window and \
which step it mapped it to. Captions can be wrong or vague; one odd line is weak evidence.
- The captioner picks step ids from a list that deliberately mixes in steps of OTHER tasks and \
distractor activities. A step id not in the claimed task's step list (the CLAIMED TASK section) \
means the captioner saw work from another task or something unrelated. One such line is weak \
evidence; several in a row are strong evidence of wrong_task (the steps fit another task) or \
off_script.
- `mm:ss AGENT: "..."` is a question you (the agent) already asked the worker.
- `mm:ss WORKER: "..."` is the worker's spoken answer. An answer is an explanation: if it \
plausibly accounts for what was seen (e.g. "waiting for the kettle", "grabbing a clean cup"), \
treat that variation as explained and stop raising it.
- Tags inside a caption come from automatic checks, not the captioner's words:
  `[recapture?]` = frames look like a screen or photo being filmed rather than the real scene;
  `[duplicate of <session> mm:ss]` = this footage matches an earlier recording (recycled video);
  `[visual motion without head motion]` = the image moves while the phone's motion sensors are \
still, which suggests a video being played to the camera.
  Tags are strong evidence but can misfire once; repeated tags are serious.
- Timestamps matter: use them to judge how long each step takes compared with the references.

Reference timelines show NORMAL variation. People work at different speeds, reorder independent \
steps, pause briefly, fetch things, and do small unrelated actions. That is fine. Only flag what \
falls clearly outside what the references show.

Divergence types:
- off_script: doing something unrelated to the task for a sustained time.
- wrong_task: the activity matches a different task, not the claimed one.
- out_of_order: a step that depends on another happens before it (e.g. pouring before boiling).
- padding: one step lasts far longer than in any reference (e.g. 3x or more) without visible \
reason - stretching time to get paid more.
- idle: hands inactive / nothing happening for several chunks in a row.
- recapture: `[recapture?]`, `[duplicate of ...]` or `[visual motion without head motion]` tags.
Also treat wrong objects for the step (e.g. a pan when the step needs a kettle) as off_script or \
wrong_task.

Status:
- ok: consistent with the task and references.
- watch: something looks a bit off but could be normal. ALWAYS prefer watch the first time you \
notice something; give it another chunk.
- diverge: the problem has persisted or is unambiguous (e.g. repeated recapture tags, a clearly \
different task for 20 s+).

Interrupting the worker is costly; interrupt RARELY. Set wake=true only when status is diverge, the \
divergence is not already explained by a WORKER line, and you have not just asked about the same \
thing. When wake=true, write `question`: at most 12 words, friendly and casual, about what the \
worker is doing right now (e.g. "Quick check - what are you working on there?"). Never accuse. \
When wake=false, question is "".

Other fields:
- current_step: the claimed task's step the worker is on now (or "other").
- progress: 0-1, fraction of the task's steps completed so far.
- reason: one short sentence explaining your status, citing timestamps if relevant.
- fraud_risk: 0-1, your overall belief that this session is not genuine work on the claimed task, \
given everything so far. Explained variation lowers it.
- explained: whether the latest WORKER line plausibly explains the latest divergence you asked \
about (true/false). null if there is no WORKER answer since the last AGENT question.
"""

REPORT_INSTRUCTIONS = """\
You review a finished recording session of a gig worker doing a claimed task for a \
data-collection app. You get the task's step list, reference timelines of other people doing the \
same task (normal variation), and the session's full text timeline (captioner lines, AGENT \
questions, WORKER answers, and automatic check tags such as `[recapture?]`, \
`[duplicate of <session> mm:ss]`, `[visual motion without head motion]`).

Produce:
- segments: the session split into consecutive step segments covering the whole timeline, in \
order. Merge adjacent chunks of the same step. start/end are "mm:ss". caption: one short sentence \
for the segment. For a segment that diverged and the worker explained it, use the worker's answer \
as the caption and set source "worker"; otherwise source "model". step is a step id or "other".
- fraud_score: 0-100, how likely the session is not genuine work on the claimed task (padding, \
idling, wrong task, recycled or screen-recaptured footage, unexplained divergences). Variation \
explained by the worker or seen in references is not fraud.
- verdict: "accept" (genuine, pay), "review" (a human should look), "reject" (clear fraud).
- reasons: 1-5 short, specific reasons citing timestamps.
"""


def _steps_block(steps: list[Step]) -> str:
    return "\n".join(f"- {s.id}: {s.label}" for s in steps)


def _references_block(references: list[str]) -> str:
    if not references:
        return "No reference recordings are available; judge against the step list only."
    return "\n\n".join(f"--- Reference {i + 1} ---\n{r}" for i, r in enumerate(references))


def _context_prefix(task: Task, references: list[str]) -> str:
    """Stable per-session prefix: task + reference timelines (for prompt caching)."""
    return (
        f"=== CLAIMED TASK\n{task.name} ({task.id})\n"
        f"Steps in order (the only step ids that belong to this task):\n{_steps_block(task.steps)}\n"
        f"- other: something else\n\n"
        f"=== REFERENCE TIMELINES\n{_references_block(references)}\n\n"
    )


def _user(text: str) -> dict:
    return {"role": "user", "content": [{"type": "input_text", "text": text}]}


def _developer(text: str) -> dict:
    return {"role": "developer", "content": [{"type": "input_text", "text": text}]}


# ---------------------------------------------------------------- client

class LLM:
    def __init__(self, client: Any, settings: Any):
        self.client = client
        self.settings = settings

    async def _json_call(self, model: str, input: list[dict], name: str, schema: dict) -> dict:
        kwargs: dict[str, Any] = dict(
            model=model, input=input,
            text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
            timeout=self.settings.llm_timeout_s,
        )
        if REASONING is not None:
            kwargs["reasoning"] = REASONING
        resp = await self.client.responses.create(**kwargs)
        data = json.loads(resp.output_text)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    async def caption(self, jpegs: list[bytes], task: Task, steps: list[Step],
                      prev_caption: str | None) -> CaptionResult:
        prompt = (
            f"Claimed task (may be false): {task.name}\n\n"
            f"Candidate steps (id: label):\n{_steps_block(steps)}\n\n"
            f"Previous 10 s caption (for continuity only): {prev_caption or '(none - session start)'}\n\n"
            f"The {len(jpegs)} frames follow, oldest first."
        )
        content: list[dict] = [{"type": "input_text", "text": prompt}]
        for j in jpegs:
            content.append({
                "type": "input_image", "detail": "low",
                "image_url": "data:image/jpeg;base64," + base64.b64encode(j).decode(),
            })
        try:
            data = await self._json_call(
                self.settings.caption_model,
                [_developer(CAPTION_INSTRUCTIONS), {"role": "user", "content": content}],
                "caption", _caption_schema([s.id for s in steps]),
            )
            return {k: data[k] for k in CAPTION_FALLBACK}
        except Exception as e:  # noqa: BLE001 - any failure degrades to a placeholder chunk
            log.warning("caption failed: %r", e)
            return dict(CAPTION_FALLBACK)

    async def monitor(self, task: Task, references: list[str], timeline: Timeline,
                      trigger: str) -> MonitorDecision:
        lines = timeline.render().splitlines()
        if trigger == "worker_answer":
            latest = "The worker just answered your question (the WORKER line below). " \
                     "Judge `explained` and update your decision."
        else:
            latest = "A new 10 s chunk was just added (the chunk line below). Update your decision."
        text = (
            _context_prefix(task, references)
            + "=== SESSION TIMELINE (so far)\n" + "\n".join(lines[:-1]) + "\n\n"
            + "=== LATEST\n" + latest + "\n" + (lines[-1] if lines else "(empty)")
        )
        step_ids = [s.id for s in task.steps] + ["other"]
        try:
            data = await self._json_call(
                self.settings.monitor_model, [_developer(MONITOR_INSTRUCTIONS), _user(text)],
                "monitor_decision", _monitor_schema(step_ids),
            )
            return MonitorDecision(**{k: data[k] for k in MODEL_FIELDS})
        except Exception as e:  # noqa: BLE001
            log.warning("monitor failed: %r", e)
            return MonitorDecision.unavailable()

    async def final_report(self, task: Task, references: list[str], timeline: Timeline) -> dict:
        text = (_context_prefix(task, references)
                + "=== SESSION TIMELINE (complete)\n" + timeline.render())
        step_ids = sorted({s.id for s in task.steps} | {c.step for c in timeline.chunks} | {"other"})
        try:
            data = await self._json_call(
                self.settings.monitor_model, [_developer(REPORT_INSTRUCTIONS), _user(text)],
                "session_report", _report_schema(step_ids),
            )
            return {k: data[k] for k in ("segments", "fraud_score", "verdict", "reasons")}
        except Exception as e:  # noqa: BLE001
            log.warning("final report failed: %r", e)
            return {"segments": _merge_segments(timeline), "fraud_score": None,
                    "verdict": "review", "reasons": ["report model unavailable"]}


def _merge_segments(timeline: Timeline) -> list[dict]:
    segs: list[dict] = []
    for c in timeline.chunks:
        if segs and segs[-1]["step"] == c.step:
            segs[-1]["end"] = fmt(c.t_end)
        else:
            segs.append({"start": fmt(c.t_start), "end": fmt(c.t_end), "step": c.step,
                         "caption": c.caption, "source": "model"})
    return segs

