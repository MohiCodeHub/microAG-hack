from app.report import Divergence, build_report, rule_score, signal_stats
from app.timeline import Chunk


def mk(i, hands="both", activity="active", recapture=False):
    return Chunk(index=i, t_start=i * 10, t_end=i * 10 + 10, caption="c", step="s",
                 confidence=0.9, objects=[], hands_visible=hands, activity=activity,
                 recapture_suspected=recapture)


def test_signal_stats_counts():
    chunks = [mk(0, activity="idle"), mk(1, hands="none"), mk(2, recapture=True), mk(3)]
    divs = [Divergence(5, "idle", "r", "q", "a", True), Divergence(30, "idle", "r", "q", None, False)]
    s = signal_stats(chunks, divs, duplicate_hits=2, mismatch_chunks=1)
    assert s == {"idle_fraction": 0.25, "hands_out_fraction": 0.25, "unexplained_divergences": 1,
                 "duplicate_hits": 2, "imu_mismatch_chunks": 1, "recapture_chunks": 1}


def test_signal_stats_empty_session():
    s = signal_stats([], [], 0, 0)
    assert s["idle_fraction"] == 0 and s["hands_out_fraction"] == 0


def test_rule_score_formula():
    stats = {"idle_fraction": 0.5, "hands_out_fraction": 0.0, "unexplained_divergences": 1,
             "duplicate_hits": 0, "imu_mismatch_chunks": 1, "recapture_chunks": 0}
    assert rule_score(stats) == 45  # 20 + 15 + 10


def test_rule_score_caps_at_100():
    stats = {"idle_fraction": 1.0, "hands_out_fraction": 1.0, "unexplained_divergences": 3,
             "duplicate_hits": 5, "imu_mismatch_chunks": 4, "recapture_chunks": 4}
    assert rule_score(stats) == 100


def test_fraud_score_uses_rule_when_model_null():
    stats = signal_stats([mk(0, activity="idle"), mk(1)], [], 0, 0)  # rule = 20
    model = {"segments": [], "fraud_score": None, "verdict": "review", "reasons": []}
    r = build_report(model, stats, [], "timeline text")
    assert r["rule_score"] == 20
    assert r["fraud_score"] == 20
    assert r["signals"] == stats
    assert r["timeline"] == "timeline text"
    assert r["verdict"] == "review"


def test_fraud_score_blends_model_and_rule():
    stats = signal_stats([mk(0, activity="idle"), mk(1)], [], 0, 0)  # rule = 20
    model = {"segments": [], "fraud_score": 61, "verdict": "accept", "reasons": []}
    d = Divergence(10, "idle", "r", "q", "a", True)
    r = build_report(model, stats, [d], "")
    assert r["fraud_score"] == round(0.5 * 61 + 0.5 * 20)
    assert r["divergences"] == [{"t": 10, "type": "idle", "reason": "r", "question": "q",
                                 "answer": "a", "explained": True}]
