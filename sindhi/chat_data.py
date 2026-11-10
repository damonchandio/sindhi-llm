"""Build the Sindhi chat training set: data/chat/train.jsonl ({"messages": [...], "source": ...}) and
data/chat/eval_prompts.json (20 held-out Sindhi requests for the A/B page). All sources Apache-2.0 or CC-BY-SA.

  MURI-IT snd          instructions written for real human Sindhi text (web, Wikipedia)       Apache-2.0
  Aya Dataset          274 prompts and answers written by native Sindhi speakers              Apache-2.0
  Sindhi-Assistant     1,500 conversations (Kashif786), instruction + cultural roleplay       Apache-2.0
  Aya Collection       machine-translated: Dolly (incl. answer-from-passage), chain-of-thought,
                       HotpotQA. Filtered hard: no English leaking in, no MT artefacts.         Apache-2.0
  Dolly-15k (English)  a little English so the model doesn't lose it                            CC-BY-SA-3.0

Nothing whose text overlaps Belebele's passages gets in, so the reading score stays honest.
    python -m sindhi.chat_data
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from huggingface_hub import HfFileSystem, hf_hub_download
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "chat"
LATIN = re.compile(r"[A-Za-z]")
JUNK = re.compile(r"XNUMX|\?\?|ـ{2,}")
AYA_COLL = {"Dolly-v2 (T)": 6000, "Flan-CoT-submix (T)": 3000, "HotpotQA (T)": 1000}


def clean_sd(text: str, max_latin: float = 0.02) -> bool:
    """Sindhi text with almost no Latin letters (MT leaves English words behind) and no known MT artefacts."""
    return bool(text.strip()) and not JUNK.search(text) and len(LATIN.findall(text)) <= max_latin * len(text)


TAIL = re.compile(r"^\s*(حوالا|حوالو|حوالا جات|لاڳاپيل|ٻاهريان ڳنڍڻا|پڻ ڏسو|ٻاهريون ڳنڍڻا)\s*:?\s*$|"
                  r"^\s*(زمرو|زمرا|پيدائشون|وفاتون)\b", re.M)


def strip_wiki(text: str) -> str:
    """Wikipedia-sourced answers end with "حوالا" (References), "لاڳاپيل" (Related) and category lines; v3 learned to
    end every answer that way. Cut from the first such line, then drop empty section headings left at the end."""
    m = TAIL.search(text)
    lines = (text[:m.start()] if m else text).rstrip().split("\n")
    while m and len(lines) > 1 and (not lines[-1].strip() or (len(lines[-1].split()) <= 3 and lines[-1].strip()[-1:] not in ".؟!:")):
        lines.pop()
    return "\n".join(lines).strip()


def repetitive(text: str, n: int = 4) -> bool:
    """More than 10% of 4-word windows are repeats: machine-translation loops the model then copies."""
    w = text.split()
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return bool(grams) and (len(grams) - len(set(grams))) / len(grams) > 0.1


def pair(user: str, assistant: str, source: str) -> dict:
    return {"messages": [{"role": "user", "content": user.strip()}, {"role": "assistant", "content": assistant.strip()}],
            "source": source}


def belebele_shingles() -> set[str]:
    """8-word windows of every Belebele passage, to drop any training text that contains them."""
    path = ROOT / "data" / "belebele_snd.jsonl"
    if not path.exists():
        import urllib.request
        from .belebele import URL
        path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL.format("snd_Arab"), path)
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        w = json.loads(line)["flores_passage"].split()
        out.update(" ".join(w[i:i + 8]) for i in range(len(w) - 7))
    return out


def leaks(ex: dict, shingles: set[str]) -> bool:
    w = " ".join(m["content"] for m in ex["messages"]).split()
    return any(" ".join(w[i:i + 8]) in shingles for i in range(len(w) - 7))


def main() -> None:
    rng = random.Random(0)
    data: list[dict] = []
    get = lambda repo, f: hf_hub_download(repo, f, repo_type="dataset")  # noqa: E731

    muri = pq.read_table(get("akoksal/muri-it-language-split", "snd/train-00000-of-00001.parquet")).to_pylist()
    data += [pair(r["input"], r["output"], "MURI-IT") for r in muri
             if len(r["output"]) <= 1500 and clean_sd(r["input"]) and clean_sd(r["output"], 0.05)]

    aya = pq.read_table(get("CohereLabs/aya_dataset", "data/train-00000-of-00001.parquet"))
    aya = aya.filter(pc.equal(aya["language"], "Sindhi")).to_pylist()
    data += [pair(r["inputs"], r["targets"], "Aya Dataset") for r in aya]

    for line in Path(get("Kashif786/Sindhi-Assistant-IFT-1500", "data/all.jsonl")).read_text(encoding="utf-8").splitlines():
        msgs = [m for m in json.loads(line)["messages"] if m["role"] != "system"]
        if msgs and msgs[-1]["role"] == "assistant":
            data.append({"messages": msgs, "source": "Sindhi-Assistant-IFT-1500"})

    fs = HfFileSystem()
    coll = []
    for f in ("sindhi/train-00000-of-00002.parquet", "sindhi/train-00001-of-00002.parquet"):
        pf = pq.ParquetFile(fs.open(f"datasets/CohereLabs/aya_collection_language_split/{f}"))
        names = pf.read(columns=["dataset_name"]).column(0).to_pylist()
        groups, off = [], 0
        for g in range(pf.num_row_groups):
            n = pf.metadata.row_group(g).num_rows
            if set(names[off:off + n]) & AYA_COLL.keys():
                groups.append(g)
            off += n
        coll += [r for r in pf.read_row_groups(groups, columns=["inputs", "targets", "dataset_name"]).to_pylist()
                 if r["dataset_name"] in AYA_COLL]
    for name, k in AYA_COLL.items():
        rows = [r for r in coll if r["dataset_name"] == name and clean_sd(r["inputs"]) and clean_sd(r["targets"])]
        if name == "Dolly-v2 (T)":   # answer-from-a-passage rows first: that's the reading skill Belebele tests
            rows.sort(key=lambda r: "Context:" not in r["inputs"])
        else:
            rng.shuffle(rows)
        data += [pair(r["inputs"].replace(" / Context:", "\n\n").replace("Context:", "\n\n"), r["targets"],
                      f"Aya Collection {name}") for r in rows[:k]]

    dolly = [json.loads(l) for l in Path(get("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl")).read_text().splitlines()]
    dolly = [r for r in dolly if r["category"] in ("closed_qa", "information_extraction", "open_qa", "classification")]
    rng.shuffle(dolly)
    data += [pair(r["instruction"] + ("\n\n" + r["context"] if r["context"] else ""), r["response"], "Dolly-15k (English)")
             for r in dolly[:1500]]

    shingles = belebele_shingles()
    kept = [ex for ex in data if not leaks(ex, shingles)]
    print(f"dropped {len(data) - len(kept)} examples that overlap Belebele passages")
    cut = 0
    for ex in kept:
        a = ex["messages"][-1]["content"]
        b = strip_wiki(a)
        cut += a != b
        ex["messages"][-1]["content"] = b
    before = len(kept)
    kept = [ex for ex in kept if ex["messages"][-1]["content"] and not repetitive(ex["messages"][-1]["content"])]
    print(f"cut Wikipedia endings from {cut} answers; dropped {before - len(kept)} repetitive or empty answers")

    val = pq.read_table(get("akoksal/muri-it-language-split", "snd/validation-00000-of-00001.parquet")).to_pylist()
    prompts = [r["input"] for r in val if len(r["input"]) <= 200 and clean_sd(r["input"])]
    rng.shuffle(prompts)

    rng.shuffle(kept)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "train.jsonl").open("w", encoding="utf-8") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    (OUT / "eval_prompts.json").write_text(json.dumps(prompts[:20], ensure_ascii=False, indent=0), encoding="utf-8")
    counts: dict[str, int] = {}
    for ex in kept:
        counts[ex["source"]] = counts.get(ex["source"], 0) + 1
    print(json.dumps({"examples": len(kept), "by_source": counts}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
