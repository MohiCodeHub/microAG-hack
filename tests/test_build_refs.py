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
    fr = [(float(i), b"x") for i in range(23)]
    assert [len(c) for c in build_refs.group_chunks(fr)] == [10, 10, 3]
    assert [len(c) for c in build_refs.group_chunks(fr[:22])] == [10, 10]
