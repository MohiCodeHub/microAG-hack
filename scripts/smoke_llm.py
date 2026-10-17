"""Real API smoke test: one caption call on 10 synthetic frames, one monitor call.

Usage: uv run python scripts/smoke_llm.py
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from app import llm as llm_mod  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.tasks import candidate_steps, load_library  # noqa: E402
from app.timeline import Chunk, Timeline  # noqa: E402


def synthetic_frames(n: int = 10) -> list[bytes]:
    out = []
    for i in range(n):
        img = Image.new("RGB", (320, 240), (200, 190, 170))
        d = ImageDraw.Draw(img)
        d.rectangle((40 + i * 8, 120, 120 + i * 8, 220), fill=(80, 80, 90))  # "kettle"
        d.ellipse((180, 60 + i * 5, 240, 120 + i * 5), fill=(230, 180, 150))  # "hand"
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=70)
        out.append(buf.getvalue())
    return out


async def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    settings = get_settings()
    lib = load_library(settings.tasks_dir)
    task = lib["example_make_tea"]
    llm = llm_mod.LLM(AsyncOpenAI(api_key=settings.openai_api_key), settings)
    print(f"reasoning={llm_mod.REASONING} caption_model={settings.caption_model} "
          f"monitor_model={settings.monitor_model}")

    t0 = time.perf_counter()
    cap = await llm.caption(synthetic_frames(), task, candidate_steps(task, lib), None)
    print(f"caption ({time.perf_counter() - t0:.2f}s):\n{json.dumps(cap, indent=2)}")

    tl = Timeline()
    for i, (c, s) in enumerate([("fills kettle at the sink", "fill_kettle"),
                                ("switches kettle on, waits", "boil"),
                                ("scrolls phone, hands idle", "other")]):
        tl.add_chunk(Chunk(i, i * 10, i * 10 + 10, c, s, 0.8, [], "both",
                           "idle" if s == "other" else "active", False))
    refs = ["00:00–00:10 fills kettle with tap water [fill_kettle]\n"
            "00:10–00:20 switches kettle on [boil]\n00:20–00:30 takes a mug and teabag [cup_teabag]"]
    t0 = time.perf_counter()
    dec = await llm.monitor(task, refs, tl, "chunk")
    print(f"monitor ({time.perf_counter() - t0:.2f}s):\n{json.dumps(asdict(dec), indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
