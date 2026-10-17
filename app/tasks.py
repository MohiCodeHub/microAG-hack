from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.timeline import Timeline


@dataclass
class Step:
    id: str
    label: str


@dataclass
class Task:
    id: str
    name: str
    steps: list[Step] = field(default_factory=list)


FALLBACK_DISTRACTORS = [
    Step("idle_or_waiting", "idle or waiting"),
    Step("using_phone", "using a phone"),
    Step("unrelated_tidying", "unrelated tidying"),
]
OTHER = Step("other", "something else")


def load_library(tasks_dir: Path) -> dict[str, Task]:
    lib: dict[str, Task] = {}
    for f in sorted(Path(tasks_dir).glob("*/task.json")):
        d = json.loads(f.read_text())
        lib[d["id"]] = Task(
            id=d["id"], name=d["name"],
            steps=[Step(s["id"], s["label"]) for s in d["steps"]],
        )
    return lib


def candidate_steps(task: Task, library: dict[str, Task]) -> list[Step]:
    out = list(task.steps)
    others = [library[k] for k in sorted(library) if k != task.id][:2]
    if others:
        for o in others:
            out.extend(o.steps)
    else:
        out.extend(FALLBACK_DISTRACTORS)
    out.append(OTHER)
    return out


def load_reference_timelines(tasks_dir: Path, task_id: str) -> list[str]:
    refs = Path(tasks_dir) / task_id / "refs"
    return [
        Timeline.from_json(json.loads(f.read_text())).render()
        for f in sorted(refs.glob("*.timeline.json"))
    ]
