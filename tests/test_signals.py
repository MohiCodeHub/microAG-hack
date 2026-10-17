import io

from PIL import Image

from app import signals


def _jpeg(color: int, size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("L", size, color).convert("RGB").save(buf, "JPEG")
    return buf.getvalue()


def _pattern(seed: int) -> bytes:
    import random

    rnd = random.Random(seed)
    img = Image.new("L", (64, 64))
    img.putdata([rnd.randint(0, 255) for _ in range(64 * 64)])
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG")
    return buf.getvalue()


def test_visual_motion_zero_for_identical_frames():
    f = _jpeg(120)
    assert signals.visual_motion([f, f, f]) == 0.0


def test_visual_motion_high_for_alternating_black_white():
    b, w = _jpeg(0), _jpeg(255)
    assert signals.visual_motion([b, w, b, w]) > 0.5


def test_imu_energy_none_when_empty():
    assert signals.imu_energy([]) is None


def test_imu_energy_mean_gyro_magnitude():
    s = [{"t": 0, "ax": 0, "ay": 0, "az": 0, "gx": 3, "gy": 4, "gz": 0}]
    assert signals.imu_energy(s) == 5.0


def test_imu_mismatch_true_when_visual_moves_and_head_still():
    assert signals.imu_mismatch(0.5, 0.01) is True


def test_imu_mismatch_false_without_imu():
    assert signals.imu_mismatch(0.5, None) is False


def test_hashstore_matches_other_session_not_same(tmp_path):
    path = tmp_path / "sub" / "hashes.json"
    store = signals.HashStore(path)
    f = _pattern(1)
    assert store.check_and_add("s1", 1.0, f) == []
    assert store.check_and_add("s1", 2.0, f) == []
    matches = store.check_and_add("s2", 3.0, f)
    assert len(matches) == 1
    assert matches[0]["session_id"] == "s1"
    store2 = signals.HashStore(path)
    assert len(store2.check_and_add("s3", 4.0, f)) >= 1
