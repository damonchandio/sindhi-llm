"""Belebele snd_Arab (Meta, CC-BY-SA 4.0): 900 Sindhi reading-comprehension questions, four answers each.
Random guessing scores 25%. Scored zero-shot: the answer the model finds most likely (mean log-probability per
token, so long answers aren't penalised) after the passage and question.
--mc is the standard way to test a chat model instead: passage, question and the four options labelled A-D in a
chat message; the first letter the model writes is its answer (greedy, 8 tokens). --lang eng_Latn runs the English version,
which shows the ceiling: a model can't read Sindhi better than it reads English.
    python -m sindhi.belebele --base Qwen/Qwen3-4B --4bit [--adapter out/lora-4b] [--mc] [--lang eng_Latn]
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

import torch

from .evaluate import load

ROOT = Path(__file__).resolve().parents[1]
URL = "https://huggingface.co/datasets/facebook/belebele/resolve/main/data/{}.jsonl"
LETTERS = "ABCD"


@torch.no_grad()
def score(tok, model, context: str, answers: list[str]) -> int:
    ctx = tok(context, add_special_tokens=False).input_ids
    seqs = [ctx + tok(" " + a, add_special_tokens=False).input_ids for a in answers]
    width = max(map(len, seqs))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], device=model.device)
    logits = model(ids, logits_to_keep=width - len(ctx) + 1).logits   # answer positions only: full vocab is huge
    best, best_i = -1e9, 0
    for i, s in enumerate(seqs):
        tgt = torch.tensor(s[len(ctx):], device=model.device)
        logp = torch.log_softmax(logits[i, :len(tgt)].float(), -1)
        lp = logp.gather(1, tgt[:, None]).mean().item()
        if lp > best:
            best, best_i = lp, i
    return best_i + 1


@torch.no_grad()
def score_mc(tok, model, it: dict) -> int:
    """The first A-D the model writes (chat models often open with "Answer:" or bold text), 0 if none."""
    opts = "\n".join(f"{L}) {it[f'mc_answer{i}']}" for i, L in enumerate(LETTERS, 1))
    msg = (f"{it['flores_passage']}\n\nQuestion: {it['question']}\n{opts}\n\n"
           "Answer with only the letter of the correct option (A, B, C or D).")
    prompt = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
    m = re.search(r"\b([ABCD])\b", tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
    return LETTERS.index(m[1]) + 1 if m else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter")
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name")
    ap.add_argument("--mc", action="store_true", help="options A-D in a chat prompt, answer = first letter written")
    ap.add_argument("--lang", default="snd_Arab")
    a = ap.parse_args()
    path = ROOT / "data" / ("belebele_snd.jsonl" if a.lang == "snd_Arab" else f"belebele_{a.lang}.jsonl")
    if not path.exists():
        urllib.request.urlretrieve(URL.format(a.lang), path)
    items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    items = items[:a.limit] if a.limit else items
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load(a.adapter, device, a.base, a.four_bit)
    right = 0
    for k, it in enumerate(items, 1):
        if a.mc:
            guess = score_mc(tok, model, it)
        else:
            ctx = f"{it['flores_passage']}\nسوال: {it['question']}\nجواب:"
            guess = score(tok, model, ctx, [it[f"mc_answer{i}"] for i in range(1, 5)])
        right += guess == int(it["correct_answer_num"])
        if k % 100 == 0:
            print(f"{k}/{len(items)}: {right / k:.1%}", flush=True)
    name = a.name or (Path(a.adapter).name if a.adapter else a.base.split("/")[-1])
    name = ("mc-" if a.mc else "") + (f"{a.lang}-" if a.lang != "snd_Arab" else "") + name
    res = {"model": a.base, "adapter": a.adapter, "method": "letter A-D" if a.mc else "answer likelihood",
           f"belebele_{a.lang}": round(right / len(items), 4), "n": len(items)}
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"belebele-{name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
