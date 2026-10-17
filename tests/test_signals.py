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


def test_frame_hash_skips_low_texture_frames():
    assert signals.frame_hash(_jpeg(120)) is None
    assert signals.frame_hash(_pattern(1)) is not None


def test_hashstore_matches_other_session_not_same(tmp_path):
    path = tmp_path / "sub" / "hashes.json"
    store = signals.HashStore(path)
    h = signals.frame_hash(_pattern(1))
    assert store.check_and_add("s1", 0, 1.0, h) == []
    assert store.check_and_add("s1", 1, 2.0, h) == []
    matches = store.check_and_add("s2", 0, 3.0, h)
    assert matches == [{"session_id": "s1", "index": 0, "t": 1.0, "distance": 0}]
    store2 = signals.HashStore(path)
    assert len(store2.check_and_add("s3", 0, 4.0, h)) >= 1


def test_hashstore_match_needs_near_identical_hash(tmp_path):
    store = signals.HashStore(tmp_path / "hashes.json")
    h = signals.frame_hash(_pattern(1))
    flipped = h.hash.copy()
    flipped.flat[:3] = ~flipped.flat[:3]  # 3 bits away
    store.check_and_add("s1", 0, 1.0, h)
    assert store.check_and_add("s2", 0, 1.0, type(h)(flipped)) == []


def test_hashstore_ignores_entries_from_older_format(tmp_path):
    path = tmp_path / "hashes.json"
    h = signals.frame_hash(_pattern(1))
    path.write_text('[{"session_id": "s0", "t": 1.0, "hash": "%s"}]' % h)
    assert signals.HashStore(path).check_and_add("s1", 0, 2.0, h) == []
