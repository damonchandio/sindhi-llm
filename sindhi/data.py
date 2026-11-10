"""Sindhi Wikipedia (wikimedia/wikipedia, 20231101.sd, CC BY-SA 4.0) -> clean text, split into train/held-out.
    python -m sindhi.data
Writes data/train.txt and data/test.txt (one paragraph per line, 2% of articles held out by id, so the test
text never appears in training even in part).
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URL = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.sd/train-00000-of-00001.parquet"
SINDHI = re.compile(r"[؀-ۿݐ-ݿ]")   # Arabic-script block, which Sindhi uses


def clean(text: str) -> list[str]:
    """Paragraphs that are actually Sindhi prose: at least 40 chars and mostly Arabic-script letters."""
    out = []
    for p in text.split("\n"):
        p = re.sub(r"\s+", " ", p).strip()
        letters = sum(c.isalpha() for c in p)
        if len(p) >= 40 and letters and len(SINDHI.findall(p)) / letters >= 0.7:
            out.append(p)
    return out


def main() -> None:
    import pyarrow.parquet as pq
    DATA.mkdir(exist_ok=True)
    pq_path = DATA / "sdwiki.parquet"
    if not pq_path.exists():
        urllib.request.urlretrieve(URL, pq_path)
    t = pq.read_table(pq_path).to_pylist()
    train, test = [], []
    for row in t:
        (test if int(row["id"]) % 50 == 0 else train).extend(clean(row["text"]))
    (DATA / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (DATA / "test.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
    print(f"{len(t)} articles -> {len(train)} train paragraphs ({sum(map(len, train)) / 1e6:.1f}M chars), "
          f"{len(test)} test paragraphs")


if __name__ == "__main__":
    main()
