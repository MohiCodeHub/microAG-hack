from __future__ import annotations

import io
import json
import math
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

VISUAL_MOVING = 0.08
IMU_STILL = 0.15
# Duplicate footage: two frames "match" only when their 64-bit pHashes are near identical.
HASH_MAX_DISTANCE = 2
# Frames flatter than this (grayscale std-dev, 0-255) carry no identity: blank walls, a covered
# lens, darkness. They are never hashed, so they can never match another session.
MIN_TEXTURE_STD = 12.0


def _gray64(jpeg: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(jpeg)) as img:
        return np.asarray(img.convert("L").resize((64, 64)), dtype=np.float32) / 255.0


def visual_motion(jpegs: list[bytes]) -> float:
    """Mean abs difference of consecutive 64x64 grayscale frames, 0-1."""
    if len(jpegs) < 2:
        return 0.0
    frames = [_gray64(j) for j in jpegs]
    diffs = [float(np.abs(b - a).mean()) for a, b in zip(frames, frames[1:])]
    return sum(diffs) / len(diffs)


def imu_energy(samples: list[dict]) -> float | None:
    """Mean gyro magnitude (rad/s); None if no samples."""
    if not samples:
        return None
    mags = [math.sqrt(s["gx"] ** 2 + s["gy"] ** 2 + s["gz"] ** 2) for s in samples]
    return sum(mags) / len(mags)


def imu_mismatch(visual: float, imu: float | None) -> bool:
    return imu is not None and visual > VISUAL_MOVING and imu < IMU_STILL


def frame_hash(jpeg: bytes) -> imagehash.ImageHash | None:
    """Perceptual hash of a frame, or None for a low-texture frame."""
    with Image.open(io.BytesIO(jpeg)) as img:
        gray = img.convert("L")
        if float(np.asarray(gray, dtype=np.float32).std()) < MIN_TEXTURE_STD:
            return None
        return imagehash.phash(gray)


class HashStore:
    """Frame hashes of every session so far, persisted to a JSON file.

    Entries are {session_id, index (chunk), t, hash}. Entries without an index (older file
    format) are ignored for matching.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._entries: list[dict] = []
        if self.path.exists():
            try:
                self._entries = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                self._entries = []

    def check_and_add(self, session_id: str, index: int, t: float,
                      h: imagehash.ImageHash) -> list[dict]:
        """Entries of other sessions within HASH_MAX_DISTANCE of `h`; then records `h`."""
        matches = []
        for e in self._entries:
            if e["session_id"] == session_id or "index" not in e:
                continue
            distance = int(h - imagehash.hex_to_hash(e["hash"]))
            if distance <= HASH_MAX_DISTANCE:
                matches.append({"session_id": e["session_id"], "index": e["index"], "t": e["t"],
                                "distance": distance})
        # A session's repeat of its own exact hash adds no information.
        if not any(e["session_id"] == session_id and e["hash"] == str(h) for e in self._entries):
            self._entries.append({"session_id": session_id, "index": index, "t": t,
                                  "hash": str(h)})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._entries))
        return matches
