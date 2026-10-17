from __future__ import annotations

from dataclasses import asdict, dataclass, field


def fmt(t: float) -> str:
    s = int(t)
    return f"{s // 60:02d}:{s % 60:02d}"


@dataclass
class Chunk:
    index: int
    t_start: float
    t_end: float
    caption: str
    step: str
    confidence: float
    objects: list[str]
    hands_visible: str
    activity: str
    recapture_suspected: bool


@dataclass
class Timeline:
    chunks: list[Chunk] = field(default_factory=list)
    events: list[tuple[float, str, str]] = field(default_factory=list)  # (t, who, text)

    def add_chunk(self, c: Chunk) -> None:
        self.chunks.append(c)

    def add_agent(self, t: float, text: str) -> None:
        self.events.append((t, "AGENT", text))

    def add_worker(self, t: float, text: str) -> None:
        self.events.append((t, "WORKER", text))

    def render(self) -> str:
        # (time, kind, seq, line): chunks sort before events at equal times
        items: list[tuple[float, int, int, str]] = []
        for i, c in enumerate(self.chunks):
            items.append((c.t_start, 0, i,
                          f"{fmt(c.t_start)}–{fmt(c.t_end)} {c.caption} [{c.step}]"))
        for i, (t, who, text) in enumerate(self.events):
            items.append((t, 1, i, f'{fmt(t)} {who}: "{text}"'))
        items.sort(key=lambda x: x[:3])
        return "\n".join(x[3] for x in items)

    def to_json(self) -> dict:
        return {
            "chunks": [asdict(c) for c in self.chunks],
            "events": [{"t": t, "who": who, "text": text} for t, who, text in self.events],
        }

    @classmethod
    def from_json(cls, d: dict) -> "Timeline":
        tl = cls()
        for c in d.get("chunks", []):
            tl.add_chunk(Chunk(**c))
        for e in d.get("events", []):
            tl.events.append((e["t"], e["who"], e["text"]))
        return tl
