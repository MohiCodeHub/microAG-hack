"""Build reference timelines: build_refs.py <task_id> [--force]

Captions every tasks/<task_id>/refs/*.mp4 and writes <name>.timeline.json next to it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from io import BytesIO
from pathlib import Path

import av
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.llm import CAPTION_FALLBACK, LLM  # noqa: E402
from app.tasks import candidate_steps, load_library  # noqa: E402
from app.timeline import Chunk, Timeline, fmt  # noqa: E402

CHUNK_FRAMES = 10
MIN_PARTIAL_FRAMES = 3


def _jpeg(frame: av.VideoFrame, width: int) -> bytes:
    img = frame.to_image()
    img = img.resize((width, max(1, round(img.height * width / img.width))))
    buf = BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=80)
    return buf.getvalue()


def sample_frames(path: Path, fps: int = 1, width: int = 512) -> list[tuple[float, bytes]]:
    """Decode `path`, keeping the frame nearest each sample tick (1/fps s); returns (t, jpeg).

    Only the current tick's best candidate is held at full resolution: once decoding moves past
    a tick, its frame is final and is converted to a resized JPEG straight away.
    """
    pending: dict[int, tuple[float, av.VideoFrame]] = {}
    done: dict[int, tuple[float, bytes]] = {}
    last_t = 0.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            last_t = max(last_t, t)
            tick = round(t * fps)
            if tick in done:  # out-of-order straggler for a finished tick
                continue
            if tick not in pending or abs(t - tick / fps) < abs(pending[tick][0] - tick / fps):
                pending[tick] = (t, frame)
            for k in [k for k in pending if k < tick]:
                kt, kframe = pending.pop(k)
                done[k] = (kt, _jpeg(kframe, width))
        for k, (kt, kframe) in pending.items():
            done[k] = (kt, _jpeg(kframe, width))
    # A tick past the end of the video has only an earlier frame as its "nearest".
    return [done[tick] for tick in sorted(done) if tick / fps <= last_t]


def group_chunks(frames: list, size: int = CHUNK_FRAMES) -> list[list]:
    groups = [frames[i:i + size] for i in range(0, len(frames), size)]
    if groups and len(groups[-1]) < MIN_PARTIAL_FRAMES:
        groups.pop()
    return groups


async def build_timeline(llm: LLM, task, steps, frames) -> Timeline:
    tl = Timeline()
    prev: str | None = None
    for i, group in enumerate(group_chunks(frames)):
        res = await llm.caption([j for _, j in group], task, steps, prev)
        prev = res["caption"]
        tl.add_chunk(Chunk(index=i, t_start=group[0][0], t_end=group[-1][0] + 1.0, **res))
    return tl


def warn_failed_captions(name: str, tl: Timeline) -> list[int]:
    """Print a warning naming every chunk the captioner failed on; returns their indices."""
    failed = [c for c in tl.chunks if c.caption == CAPTION_FALLBACK["caption"]]
    if failed:
        spans = ", ".join(f"{fmt(c.t_start)}–{fmt(c.t_end)}" for c in failed)
        print(f"warning: {name}: caption unavailable for {len(failed)} chunk(s): {spans}. "
              "Rebuild with --force once the captioner works.", file=sys.stderr)
    return [c.index for c in failed]


async def run(task_id: str, force: bool, tasks_dir: Path) -> None:
    refs = Path(tasks_dir) / task_id / "refs"
    videos = sorted(refs.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No reference videos in tasks/{task_id}/refs/")
    library = load_library(tasks_dir)
    if task_id not in library:
        raise SystemExit(f"Unknown task '{task_id}' (no tasks/{task_id}/task.json)")
    task = library[task_id]
    steps = candidate_steps(task, library)
    settings = get_settings()
    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is not set; add it to .env")
    llm = LLM(AsyncOpenAI(api_key=settings.openai_api_key, max_retries=0), settings)

    for v in videos:
        out = v.with_suffix(".timeline.json")
        if out.exists() and not force:
            print(f"{v.name}: skipped (timeline exists; use --force)")
            continue
        frames = sample_frames(v)
        tl = await build_timeline(llm, task, steps, frames)
        out.write_text(json.dumps(tl.to_json(), indent=2))
        print(f"{v.name}: {len(frames)} frames, {len(tl.chunks)} chunks -> {out.name}")
        warn_failed_captions(v.name, tl)


def main(argv: list[str] | None = None, tasks_dir: Path | None = None) -> None:
    p = argparse.ArgumentParser(description="Build reference timelines for a task.")
    p.add_argument("task_id")
    p.add_argument("--force", action="store_true", help="rebuild existing timelines")
    a = p.parse_args(argv)
    asyncio.run(run(a.task_id, a.force, tasks_dir or get_settings().tasks_dir))


if __name__ == "__main__":
    main()
