import json
from pathlib import Path

from app.tasks import candidate_steps, load_library, load_reference_timelines
from app.timeline import Chunk, Timeline


def write_task(root: Path, tid: str, steps):
    d = root / tid
    d.mkdir(parents=True)
    (d / "task.json").write_text(json.dumps(
        {"id": tid, "name": tid.upper(), "steps": [{"id": s, "label": s} for s in steps]}))


def test_load_library_reads_task_json(tmp_path):
    write_task(tmp_path, "a", ["a1", "a2"])
    write_task(tmp_path, "b", ["b1"])
    lib = load_library(tmp_path)
    assert set(lib) == {"a", "b"}
    assert [s.id for s in lib["a"].steps] == ["a1", "a2"]
    assert lib["a"].name == "A"


def test_candidate_steps_includes_distractors_and_other(tmp_path):
    write_task(tmp_path, "a", ["a1", "a2"])
    write_task(tmp_path, "b", ["b1"])
    write_task(tmp_path, "c", ["c1"])
    lib = load_library(tmp_path)
    ids = [s.id for s in candidate_steps(lib["a"], lib)]
    assert ids[:2] == ["a1", "a2"]
    assert "b1" in ids and "c1" in ids
    assert ids[-1] == "other"


def test_candidate_steps_fallback_when_alone(tmp_path):
    write_task(tmp_path, "a", ["a1"])
    lib = load_library(tmp_path)
    ids = [s.id for s in candidate_steps(lib["a"], lib)]
    for x in ("idle_or_waiting", "using_phone", "unrelated_tidying"):
        assert x in ids
    assert ids[-1] == "other"


def test_reference_timelines_empty_when_no_refs(tmp_path):
    write_task(tmp_path, "a", ["a1"])
    assert load_reference_timelines(tmp_path, "a") == []


def test_reference_timelines_renders_lines(tmp_path):
    write_task(tmp_path, "a", ["a1"])
    tl = Timeline()
    for i in range(2):
        tl.add_chunk(Chunk(i, i * 10, i * 10 + 10, "x", "a1", 0.9, [], "both", "working", False))
    refs = tmp_path / "a" / "refs"
    refs.mkdir()
    (refs / "x.timeline.json").write_text(json.dumps(tl.to_json()))
    out = load_reference_timelines(tmp_path, "a")
    assert len(out) == 1
    assert "00:00–00:10" in out[0] and "00:10–00:20" in out[0]
