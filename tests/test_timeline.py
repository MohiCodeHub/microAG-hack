from app.timeline import Chunk, Timeline, fmt


def mk(i, a, b, cap="opens kettle", step="fill_kettle"):
    return Chunk(index=i, t_start=a, t_end=b, caption=cap, step=step, confidence=0.9,
                 objects=["kettle"], hands_visible="both", activity="working",
                 recapture_suspected=False)


def test_fmt():
    assert fmt(0) == "00:00"
    assert fmt(75.9) == "01:15"


def test_render_orders_lines_by_time():
    tl = Timeline()
    tl.add_chunk(mk(0, 0, 10))
    tl.add_chunk(mk(1, 10, 20, "pours water", "boil"))
    tl.add_worker(15, "done")
    tl.add_agent(12, "How is it going?")
    lines = tl.render().splitlines()
    assert lines == [
        "00:00–00:10 opens kettle [fill_kettle]",
        "00:10–00:20 pours water [boil]",
        '00:12 AGENT: "How is it going?"',
        '00:15 WORKER: "done"',
    ]


def test_json_roundtrip():
    tl = Timeline()
    tl.add_chunk(mk(0, 0, 10))
    tl.add_agent(12, "hi")
    tl.add_worker(15, "yo")
    tl2 = Timeline.from_json(tl.to_json())
    assert tl2.render() == tl.render()
    assert tl2.chunks == tl.chunks
