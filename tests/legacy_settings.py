"""Settings with the timings the session/server tests were written against (10-frame chunks,
45/60 s question gating), so they keep testing behaviour, not today's tuning defaults."""
from functools import partial

from app.config import Settings

LegacySettings = partial(Settings, chunk_frames=10, question_cooldown_s=45,
                         question_min_interval_s=60)
