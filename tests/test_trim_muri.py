from sindhi.trim_muri import trim


def row(src, ans):
    return {"source": src, "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": ans}]}


def test_trim_caps_muri_and_keeps_others():
    rows = [row("MURI-IT", "x" * n) for n in (10, 20, 900, 30)] + [row("Aya Dataset", "y" * 900)]
    out = trim(rows, cap=2, max_chars=600)
    assert sum(r["source"] == "MURI-IT" for r in out) == 2
    assert all(len(r["messages"][-1]["content"]) <= 600 for r in out if r["source"] == "MURI-IT")
    assert any(r["source"] == "Aya Dataset" for r in out)
