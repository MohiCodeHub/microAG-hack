"""End-of-session scoring: deterministic signal stats, rule score, blended fraud score."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from app.timeline import Chunk


@dataclass
class Divergence:
    t: float
    type: str
    reason: str
    question: str
    answer: str | None = None
    explained: bool | None = None
    follow_up: str | None = None  # the one push-back question, asked when `answer` was vague
    follow_up_answer: str | None = None


def signal_stats(chunks: list[Chunk], divergences: list[Divergence], duplicate_hits: int,
                 mismatch_chunks: int) -> dict:
    n = len(chunks)
    return {
        "idle_fraction": sum(c.activity == "idle" for c in chunks) / n if n else 0.0,
        "hands_out_fraction": sum(c.hands_visible == "none" for c in chunks) / n if n else 0.0,
        "unexplained_divergences": sum(d.explained is not True for d in divergences),
        "duplicate_hits": duplicate_hits,
        "imu_mismatch_chunks": mismatch_chunks,
        "recapture_chunks": sum(c.recapture_suspected for c in chunks),
    }


def rule_score(stats: dict) -> int:
    score = (40 * stats["idle_fraction"]
             + 30 * stats["hands_out_fraction"]
             + 15 * stats["unexplained_divergences"]
             + 30 * min(stats["duplicate_hits"], 1)
             + 10 * stats["imu_mismatch_chunks"]
             + 10 * stats["recapture_chunks"])
    return min(100, round(score))


def build_report(model: dict, stats: dict, divergences: list[Divergence], timeline_text: str) -> dict:
    """Merge the model's report with deterministic signals into the final report."""
    rule = rule_score(stats)
    model_score = model.get("fraud_score")
    if model_score is not None:
        model_score = max(0, min(100, model_score))
    fraud = rule if model_score is None else round(0.5 * model_score + 0.5 * rule)
    return {
        **model,
        "signals": stats,
        "model_fraud_score": model_score,
        "rule_score": rule,
        "fraud_score": fraud,
        "divergences": [asdict(d) for d in divergences],
        "timeline": timeline_text,
    }
