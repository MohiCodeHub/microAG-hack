from app.policy import NudgePolicy, WakePolicy


def test_wake_policy_blocks_during_cooldown():
    p = WakePolicy(cooldown_s=45, min_interval_s=60)
    assert p.can_ask(0, False)
    p.record_question(0)
    assert not p.can_ask(30, False)
    assert not p.can_ask(50, False)  # past cooldown, within 60s min interval
    assert p.can_ask(61, False)
    assert not p.can_ask(100, True)


def test_nudge_only_for_none_and_respects_cooldown():
    p = NudgePolicy(cooldown_s=20)
    assert not p.should_nudge(0, "one")
    assert p.should_nudge(0, "none")
    assert not p.should_nudge(10, "none")
    assert p.should_nudge(21, "none")
