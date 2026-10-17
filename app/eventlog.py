"""Per-session JSONL debug log: every server event, every client message, and phone-side reports.

Large payloads (JPEG frames, audio) are replaced by their size so the log stays readable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_BIG_FIELDS = ("jpeg_b64", "audio_b64")


def _slim(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if k in _BIG_FIELDS:
            out[k] = None if v is None else f"<{len(v)} b64 chars>"
        else:
            out[k] = v
    return out


class SessionLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", buffering=1)
        self._t0 = time.monotonic()

    def write(self, direction: str, msg: dict[str, Any]) -> None:
        line = {"ts": round(time.monotonic() - self._t0, 3), "dir": direction, **_slim(msg)}
        try:
            self._fh.write(json.dumps(line, default=str) + "\n")
        except Exception:  # noqa: BLE001 - debug logging must never break a session
            pass

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass
