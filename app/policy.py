from __future__ import annotations


class WakePolicy:
    """Gates agent questions: cooldown after any question, min interval, none while pending."""

    def __init__(self, cooldown_s: float, min_interval_s: float):
        self.cooldown_s = cooldown_s
        self.min_interval_s = min_interval_s
        self._last: float | None = None

    def can_ask(self, now: float, pending: bool) -> bool:
        if pending:
            return False
        if self._last is None:
            return True
        elapsed = now - self._last
        return elapsed >= self.cooldown_s and elapsed >= self.min_interval_s

    def record_question(self, now: float) -> None:
        self._last = now


class NudgePolicy:
    """Hands-out-of-frame nudge with its own cooldown."""

    def __init__(self, cooldown_s: float):
        self.cooldown_s = cooldown_s
        self._last: float | None = None

    def should_nudge(self, now: float, hands_visible: str) -> bool:
        if hands_visible != "none":
            return False
        if self._last is not None and now - self._last < self.cooldown_s:
            return False
        self._last = now
        return True
