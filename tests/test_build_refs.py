import sys
from pathlib import Path

import av
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_refs  # noqa: E402


def make_video(path: Path, seconds: int = 5, fps: int = 10, size=(320, 240)) -> None:
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(seconds * fps):
            img = np.full((size[1], size[0], 3), (i * 3) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for p in stream.encode(frame):
                out.mux(p)
        for p in stream.encode():
            out.mux(p)


def test_sample_frames_one_per_second(tmp_path):
    v = tmp_path / "a.mp4"
    make_video(v, seconds=5)
    frames = build_refs.sample_frames(v)
    assert len(frames) == 5
    assert [round(t) for t, _ in frames] == [0, 1, 2, 3, 4]
    from io import BytesIO

    from PIL import Image
    im = Image.open(BytesIO(frames[0][1]))
    assert im.format == "JPEG" and im.width == 512


def test_no_refs_exits_with_message(tmp_path, capsys):
    (tmp_path / "t" / "refs").mkdir(parents=True)
    with pytest.raises(SystemExit) as e:
        build_refs.main(["t"], tasks_dir=tmp_path)
    assert e.value.code != 0
    assert "No reference videos in tasks/t/refs/" in str(e.value.code) + capsys.readouterr().err


def test_group_chunks_keeps_partial_only_if_three_frames():
    fr = [(float(i), b"x") for i in range(13)]
    assert [len(c) for c in build_refs.group_chunks(fr)] == [5, 5, 3]
    assert [len(c) for c in build_refs.group_chunks(fr[:12])] == [5, 5]


def test_reference_chunks_match_live_chunk_size():
    from app.config import Settings
    assert build_refs.CHUNK_FRAMES == Settings().chunk_frames


async def test_build_timeline_spans_and_prev_caption():
    from app.tasks import Step, Task

    class StubLLM:
        def __init__(self):
            self.prevs = []

        async def caption(self, jpegs, task, steps, prev):
            self.prevs.append(prev)
            return {"caption": f"c{len(self.prevs)}", "step": "other", "confidence": 1.0,
                    "objects": [], "hands_visible": "both", "activity": "active",
                    "recapture_suspected": False}

    llm = StubLLM()
    frames = [(float(i), b"x") for i in range(13)]
    tl = await build_refs.build_timeline(llm, Task("t", "T"), [Step("other", "o")], frames)
    assert [(c.t_start, c.t_end) for c in tl.chunks] == [(0.0, 5.0), (5.0, 10.0), (10.0, 13.0)]
    assert llm.prevs == [None, "c1", "c2"]
    assert tl.render().splitlines()[0].startswith("00:00–00:05 ")


def test_failed_captions_are_listed(capsys):
    from app.llm import CAPTION_FALLBACK
    from app.timeline import Chunk, Timeline

    tl = Timeline()
    for i, caption in enumerate(["fills kettle", CAPTION_FALLBACK["caption"], "pours",
                                 CAPTION_FALLBACK["caption"]]):
        tl.add_chunk(Chunk(index=i, t_start=i * 10.0, t_end=i * 10.0 + 10, caption=caption,
                           step="other", confidence=0.0, objects=[], hands_visible="both",
                           activity="active", recapture_suspected=False))
    assert build_refs.warn_failed_captions("a.mp4", tl) == [1, 3]
    err = capsys.readouterr().err
    assert "a.mp4" in err and "00:10–00:20" in err and "00:30–00:40" in err
    tl.chunks = tl.chunks[:1]
    assert build_refs.warn_failed_captions("a.mp4", tl) == []
    assert capsys.readouterr().err == ""
