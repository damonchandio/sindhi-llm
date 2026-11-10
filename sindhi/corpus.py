"""Build the big Sindhi corpus: FineWeb-2 + HPLT 2.0 + Sangraha (all Perso-Arabic, licences allow redistribution of
derived models) + Sindhi Wikipedia, cleaned and de-duplicated by paragraph, plus small Urdu and English samples
for replay (so the model keeps the languages it already knows).
    python -m sindhi.corpus                 # ~600M chars of Sindhi by default
    python -m sindhi.corpus --max-chars 2e6 # smoke test
Writes data/corpus/{sd,ur,en}.jsonl, one document per line. Never includes held-out test paragraphs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from .data import clean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "corpus"
# (repo, file, text column, licence) in priority order: cleanest first, so the cap keeps the best text
SINDHI = [
    ("HuggingFaceFW/fineweb-2", "data/snd_Arab/train/000_00000.parquet", "text", "ODC-By 1.0"),
    ("HPLT/HPLT2.0_cleaned", "snd_Arab/train-00000-of-00002.parquet", "text", "CC0-1.0"),
    ("HPLT/HPLT2.0_cleaned", "snd_Arab/train-00001-of-00002.parquet", "text", "CC0-1.0"),
    ("ai4bharat/sangraha", "verified/snd/data-0.parquet", "text", "CC-BY-4.0"),
    ("ai4bharat/sangraha", "verified/snd/data-1.parquet", "text", "CC-BY-4.0"),
]
REPLAY = {"ur": ("HuggingFaceFW/fineweb-2", "data/urd_Arab/train/000_00000.parquet", "text"),
          "en": ("HuggingFaceFW/fineweb-edu", "sample/10BT/000_00000.parquet", "text")}
# letters Sindhi has and Urdu/Persian/Arabic don't: a page without them is almost certainly not Sindhi
SINDHI_ONLY = re.compile("[ٻڀٽٺڄڃڇڌڏڊڍڙڦڪڳڱڻ]")


def sindhi_enough(paras: list[str]) -> bool:
    text = " ".join(paras)
    letters = sum(c.isalpha() for c in text)
    return letters > 0 and len(SINDHI_ONLY.findall(text)) / letters >= 0.01


def h(p: str) -> int:
    return int.from_bytes(hashlib.blake2b(p.encode(), digest_size=8).digest(), "little")


def rows(repo: str, file: str, col: str, max_rows: int | None = None):
    """Stream a parquet file's text column. Small files are downloaded (resumable); huge replay files are read
    a few row groups at a time over HTTP, so a 5 GB Urdu shard costs only what we read."""
    import pyarrow.parquet as pq
    if max_rows:
        from huggingface_hub import HfFileSystem
        f = pq.ParquetFile(HfFileSystem().open(f"datasets/{repo}/{file}"))
        n = 0
        for g in range(f.num_row_groups):
            for t in f.read_row_group(g, columns=[col]).column(col).to_pylist():
                yield t
                n += 1
                if n >= max_rows:
                    return
    else:
        from huggingface_hub import hf_hub_download
        f = pq.ParquetFile(hf_hub_download(repo, file, repo_type="dataset"))
        for g in range(f.num_row_groups):
            yield from f.read_row_group(g, columns=[col]).column(col).to_pylist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=float, default=6e8, help="Sindhi characters to keep")
    ap.add_argument("--replay", type=float, default=0.05, help="share each for Urdu and English")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    test = (ROOT / "data" / "test.txt").read_text(encoding="utf-8").splitlines()
    seen = {h(p) for p in test}                       # held-out paragraphs can never leak into training
    total, docs, dropped = 0, 0, {"not_sindhi": 0, "dupes": 0}
    stats = []
    with open(OUT / "sd.jsonl", "w", encoding="utf-8") as out:
        # Wikipedia first (already cleaned and split by data.py)
        wiki = (ROOT / "data" / "train.txt").read_text(encoding="utf-8").splitlines()
        for i in range(0, len(wiki), 20):
            chunk = [p for p in wiki[i:i + 20] if h(p) not in seen]
            seen.update(map(h, chunk))
            if chunk:
                out.write(json.dumps({"text": "\n".join(chunk)}, ensure_ascii=False) + "\n")
                total += sum(map(len, chunk)); docs += 1
        stats.append(("wikimedia/wikipedia 20231101.sd", "CC BY-SA 4.0", total))
        for repo, file, col, lic in SINDHI:
            before = total
            for text in rows(repo, file, col):
                if total >= a.max_chars:
                    break
                paras = clean(text or "")
                if not paras or not sindhi_enough(paras):
                    dropped["not_sindhi"] += 1
                    continue
                new = [p for p in paras if h(p) not in seen]
                dropped["dupes"] += len(paras) - len(new)
                if sum(map(len, new)) < 200:
                    continue
                seen.update(map(h, new))
                out.write(json.dumps({"text": "\n".join(new)}, ensure_ascii=False) + "\n")
                total += sum(map(len, new)); docs += 1
            stats.append((f"{repo} {file}", lic, total - before))
            print(f"{repo}/{file}: +{(total - before) / 1e6:.0f}M chars (total {total / 1e6:.0f}M)", flush=True)
            if total >= a.max_chars:
                break
    for lang, (repo, file, col) in REPLAY.items():
        want, got = total * a.replay * (3 if lang == "en" else 1), 0   # English packs ~3x more chars per token
        with open(OUT / f"{lang}.jsonl", "w", encoding="utf-8") as out:
            for text in rows(repo, file, col, max_rows=10**7):
                if got >= want:
                    break
                out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                got += len(text)
        stats.append((f"{repo} {file} (replay)", "ODC-By 1.0", got))
        print(f"{lang} replay: {got / 1e6:.0f}M chars", flush=True)
    (OUT / "stats.json").write_text(json.dumps({"sindhi_chars": total, "sindhi_docs": docs, "dropped": dropped,
                                                "sources": stats}, indent=1), encoding="utf-8")
    print(f"done: {total / 1e6:.0f}M Sindhi chars in {docs:,} docs; dropped {dropped}")


if __name__ == "__main__":
    main()
