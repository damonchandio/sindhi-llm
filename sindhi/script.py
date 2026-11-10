"""The reading script for recording a Sindhi voice: 700 sentences, 7 sessions of 100 (about 10 minutes of audio
each). Picked from Sindhi Wikipedia and FineWeb-2 so that every Sindhi letter and letter pair turns up as often as
possible (greedy coverage, rare sounds first), with ~10% questions so the voice learns question intonation.
    python -m sindhi.script   -> data/voice_script.jsonl  {"id", "session", "text", "source"}
"""
from __future__ import annotations

import heapq
import json
import random
import re
from collections import Counter
from pathlib import Path

from .corpus import SINDHI_ONLY, rows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "voice_script.jsonl"
N, PER_SESSION, QUESTIONS = 700, 100, 70
OK = re.compile(r"^[ء-ٰٟ-ۓە ،؛؟.!]+$")   # Arabic-script letters and marks, no digits
URDU_ONLY = re.compile("[ےںٹڈڑہۓ]")                                 # Urdu letters Sindhi doesn't use
SPLIT = re.compile(r"(?<=[.۔؟!])\s*")
MARKS = re.compile("[\u064B-\u065F\u0670]")                             # vowel marks (zabar, zer, pesh...)


def sentences(text: str):
    for s in SPLIT.split(text.replace("۔", ".")):
        s = re.sub(r"\s+([.،؟!])", r"\1", re.sub(r"\s+", " ", s)).strip()
        words = s.split()
        if (6 <= len(words) <= 14 and 30 <= len(s) <= 110 and OK.match(s) and not URDU_ONLY.search(s)
                and "ـ" not in s and len(MARKS.findall(s)) <= 0.05 * len(s)   # no poetry-style stretching
                and len(SINDHI_ONLY.findall(s)) >= 2 and len(set(words)) > 0.8 * len(words)):
            yield s if s[-1] in ".؟!" else s + "."


def features(s: str) -> set[str]:
    f = set()
    for w in MARKS.sub("", s).rstrip(".؟!").replace("،", " ").split():
        f.update(w)
        f.update(w[i:i + 2] for i in range(len(w) - 1))
    return f


def pick(pool: list[str], k: int, seen: Counter) -> list[str]:
    """Lazy greedy: each sentence is worth the sum of 1/(1+times seen) over its letters and letter pairs."""
    gain = lambda s: sum(1 / (1 + seen[f]) for f in features(s))  # noqa: E731
    heap = [(-gain(s), i) for i, s in enumerate(pool)]
    heapq.heapify(heap)
    out = []
    while heap and len(out) < k:
        _, i = heapq.heappop(heap)
        g = gain(pool[i])
        if heap and g < -heap[0][0]:
            heapq.heappush(heap, (-g, i))
            continue
        out.append(pool[i])
        seen.update(features(pool[i]))
    return out


def main() -> None:
    cand: dict[str, str] = {}
    for line in (ROOT / "data" / "train.txt").read_text(encoding="utf-8").splitlines():
        for s in sentences(line):
            cand.setdefault(s, "Sindhi Wikipedia, CC BY-SA 4.0")
    for t in rows("HuggingFaceFW/fineweb-2", "data/snd_Arab/train/000_00000.parquet", "text", max_rows=30000):
        for s in sentences(t):
            cand.setdefault(s, "FineWeb-2, ODC-By 1.0")
    # only sentences made of words seen 3+ times: drops typos, rare names and poetic forms that are hard to read
    freq = Counter(w for s in cand for w in MARKS.sub("", s).strip(".؟!").replace("،", " ").split())
    pool = sorted(s for s in cand if all(freq[w] >= 3 for w in MARKS.sub("", s).strip(".؟!").replace("،", " ").split()))
    print(f"{len(pool):,} candidate sentences")
    seen: Counter = Counter()
    chosen = pick([s for s in pool if s.endswith("؟")], QUESTIONS, seen)
    taken = set(chosen)
    chosen += pick([s for s in pool if s not in taken], N - len(chosen), seen)
    random.Random(0).shuffle(chosen)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for i, s in enumerate(chosen):
            f.write(json.dumps({"id": f"{i + 1:04d}", "session": i // PER_SESSION + 1, "text": s,
                                "source": cand[s]}, ensure_ascii=False) + "\n")
    letters = Counter(c for s in chosen for c in SINDHI_ONLY.findall(s))
    print(f"{len(chosen)} sentences, {sum(s.endswith('؟') for s in chosen)} questions, rarest Sindhi letters:",
          letters.most_common()[-5:])


if __name__ == "__main__":
    main()
