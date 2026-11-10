"""Sindhi multiple-choice reading questions, written by an LLM from Sindhi Wikipedia passages.

Belebele asks: read a passage, answer a question by picking A, B, C or D. Nothing in the chat set practises that,
so this makes a few thousand rows in that shape. Wikipedia is CC-BY-SA, so the questions are too.
    GROQ_API_KEY=... python -m sindhi.mc_data --n 3000          -> data/chat/mc.jsonl (resumes where it stopped)
    python -m sindhi.mc_data --merge                            -> data/chat/train_v6.jsonl = train_v5 + mc
Passages that share any 8-word run with Belebele's passages are skipped, and so are generated rows that do.
A native speaker should spot-check ~50 rows (data/chat/mc_sample.txt) before training.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "chat" / "mc.jsonl"
URL = "https://api.groq.com/openai/v1/chat/completions"
LETTERS = "ABCD"
SYSTEM = ("You write reading-comprehension test questions in Sindhi (Perso-Arabic script). You only use facts stated in "
          "the passage you are given. Every question must be answerable from the passage alone, with exactly one "
          "correct option; the three wrong options must be plausible but clearly wrong according to the passage.")
ASK = ("Passage (Sindhi):\n{p}\n\nWrite {k} different multiple-choice questions in Sindhi about this passage. "
       "Each has 4 options in Sindhi. Mix question types (who, what, when, where, why, which is true). "
       'Reply as JSON: {{"questions": [{{"question": "...", "options": ["...", "...", "...", "..."], "answer": 0}}]}} '
       "where answer is the index (0-3) of the correct option.")
SINDHI = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")


def mc_prompt(passage: str, question: str, options: list[str]) -> str:
    """The same wording sindhi.belebele --mc uses, so the model practises the exact task."""
    opts = "\n".join(f"{L}) {o}" for L, o in zip(LETTERS, options))
    return (f"{passage}\n\nQuestion: {question}\n{opts}\n\n"
            "Answer with only the letter of the correct option (A, B, C or D).")


def valid(q: dict) -> bool:
    o = q.get("options")
    if not (isinstance(q.get("question"), str) and isinstance(o, list) and len(o) == 4
            and q.get("answer") in (0, 1, 2, 3)):
        return False
    texts = [q["question"], *o]
    return (all(isinstance(t, str) and t.strip() and SINDHI.search(t) and len(LATIN.findall(t)) <= 2 for t in texts)
            and len({t.strip() for t in o}) == 4)


def to_row(passage: str, q: dict, rng: random.Random) -> dict:
    """Shuffle the options so the right answer lands on A, B, C and D equally often."""
    opts = [o.strip() for o in q["options"]]
    right = opts[q["answer"]]
    rng.shuffle(opts)
    i = opts.index(right)
    return {"messages": [{"role": "user", "content": mc_prompt(passage, q["question"].strip(), opts)},
                         {"role": "assistant", "content": f"{LETTERS[i]}) {right}"}],
            "source": "Sindhi Wikipedia MC (LLM-written)"}


def passages(rng: random.Random, lo: int = 350, hi: int = 900):
    import pyarrow.parquet as pq
    rows = pq.read_table(ROOT / "data" / "sdwiki.parquet", columns=["id", "text"]).to_pylist()
    rng.shuffle(rows)
    for r in rows:
        text = re.sub(r"\s+", " ", r["text"]).strip()
        if len(text) < lo:
            continue
        cut = text[:hi]
        end = max(cut.rfind("."), cut.rfind("۔"))
        if end > lo:
            cut = cut[:end + 1]
        yield r["id"], cut


def ask(passage: str, k: int, key: str, model: str) -> list[dict]:
    body = {"model": model, "temperature": 0.5, "max_tokens": 4000, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ASK.format(p=passage, k=k)}]}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
    for wait in (0, 20, 60, 180):
        time.sleep(wait)
        r = requests.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=120)
        if r.status_code == 429:
            continue
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"]).get("questions", [])
    raise RuntimeError("rate limited four times in a row")


def agrees(row: dict, key: str, model: str) -> bool:
    """A second model answers the question blind; keep the row only if it picks the marked answer. Drops questions
    with two right options or none, which the writer model produces now and then."""
    body = {"model": model, "temperature": 0, "max_tokens": 1500,
            "messages": [{"role": "user", "content": row["messages"][0]["content"]}]}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
    for wait in (0, 20, 60):
        time.sleep(wait)
        r = requests.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=120)
        if r.status_code == 429:
            continue
        r.raise_for_status()
        m = re.search(r"\b([ABCD])\b", r.json()["choices"][0]["message"]["content"] or "")
        return bool(m) and m[1] == row["messages"][1]["content"][0]
    return False


def merge() -> None:
    rows = [l for l in (ROOT / "data" / "chat" / "train_v5.jsonl").read_text(encoding="utf-8").splitlines() if l]
    mc = [l for l in OUT.read_text(encoding="utf-8").splitlines() if l]
    out = rows + mc
    random.Random(0).shuffle(out)
    (OUT.parent / "train_v6.jsonl").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"train_v6.jsonl: {len(rows)} chat + {len(mc)} multiple-choice = {len(out)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="rows to end up with")
    ap.add_argument("--per", type=int, default=3, help="questions per passage")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--checker", default="qwen/qwen3.8-27b", help="answers each question blind; '' = no check")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        return merge()
    from .chat_data import belebele_shingles, leaks
    key = os.environ["GROQ_API_KEY"]
    shingles = belebele_shingles()
    done = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()] if OUT.exists() else []
    seen = {d["pid"] for d in done}
    rng = random.Random(len(done))
    have, fails = len(done), 0
    with OUT.open("a", encoding="utf-8") as f:
        for pid, p in passages(random.Random(0)):
            if have >= a.n:
                break
            if pid in seen or leaks({"messages": [{"content": p}]}, shingles):
                continue
            try:
                qs = ask(p, a.per, key, a.model)
            except Exception as e:  # noqa: BLE001
                print("skip", pid, str(e)[:120], flush=True)
                fails += 1
                if fails >= 10:   # Groq's daily free limit, or the network: stop rather than skip every passage
                    raise SystemExit(f"stopped at {have} rows after 10 failed calls in a row; run again later to carry on")
                continue
            fails = 0
            for q in qs:
                if not valid(q):
                    continue
                row = to_row(p, q, rng)
                if leaks(row, shingles) or (a.checker and not agrees(row, key, a.checker)):
                    continue
                row["pid"] = pid
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                have += 1
            f.flush()
            seen.add(pid)
            if have % 60 < a.per:
                print(f"{have}/{a.n}", flush=True)
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()]
    sample = random.Random(1).sample(rows, min(50, len(rows)))
    (OUT.parent / "mc_sample.txt").write_text("\n\n---\n\n".join(r["messages"][0]["content"] + "\n=> " +
                                                                 r["messages"][1]["content"] for r in sample),
                                              encoding="utf-8")
    print(f"{len(rows)} rows in {OUT.name}; 50 to check in mc_sample.txt")


if __name__ == "__main__":
    main()
