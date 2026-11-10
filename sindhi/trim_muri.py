"""v5 chat set: the 8B run on train.jsonl scored 54.9% on Belebele, below the untrained 8B (59.7%), and its answers
read like Wikipedia stubs. MURI-IT (6,467 rows, mostly Wikipedia-derived) is the likely cause, so keep only its
shortest answers and leave every other source as it is.
    python -m sindhi.trim_muri [--cap 2000] [--max-chars 600]   -> data/chat/train_v5.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def trim(rows: list[dict], cap: int, max_chars: int, seed: int = 0) -> list[dict]:
    muri = [r for r in rows if r["source"] == "MURI-IT" and len(r["messages"][-1]["content"]) <= max_chars]
    random.Random(seed).shuffle(muri)
    out = [r for r in rows if r["source"] != "MURI-IT"] + muri[:cap]
    random.Random(seed).shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--max-chars", type=int, default=600)
    a = ap.parse_args()
    src = ROOT / "data" / "chat" / "train.jsonl"
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = trim(rows, a.cap, a.max_chars)
    dst = src.with_name("train_v5.jsonl")
    dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(len(rows), "->", len(out), dict(Counter(r["source"] for r in out)))


if __name__ == "__main__":
    main()
