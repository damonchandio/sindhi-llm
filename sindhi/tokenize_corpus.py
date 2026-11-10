"""Tokenise data/corpus/*.jsonl once into flat uint32 token files (data/tokens/<lang>.bin), so training can
start, stop and resume without re-tokenising anything.
    python -m sindhi.tokenize_corpus --base Qwen/Qwen3-4B
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    out_dir = ROOT / "data" / "tokens"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"base": a.base, "eos": tok.eos_token_id}
    for src in sorted((ROOT / "data" / "corpus").glob("*.jsonl")):
        n = 0
        with open(src, encoding="utf-8") as fh, open(out_dir / f"{src.stem}.bin", "wb") as out:
            batch = []
            for line in fh:
                batch.append(json.loads(line)["text"])
                if len(batch) == 512:
                    n += write(tok, batch, out)
                    batch = []
            if batch:
                n += write(tok, batch, out)
        meta[src.stem] = n
        print(f"{src.stem}: {n / 1e6:.1f}M tokens", flush=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))


def write(tok, texts, out) -> int:
    ids = tok(texts, add_special_tokens=False).input_ids
    flat = np.fromiter((t for doc in ids for t in (*doc, tok.eos_token_id)), dtype=np.uint32)
    flat.tofile(out)
    return len(flat)


if __name__ == "__main__":
    main()
