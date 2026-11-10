"""Answers to 20 held-out Sindhi requests (data/chat/eval_prompts.json, never trained on), for the blind A/B page.
    python -m sindhi.chat_eval --4bit --name chat-base-4b                          # Qwen3-4B as released
    python -m sindhi.chat_eval --4bit --adapter out/lora-chat-4b --name chat-lora-chat-4b
    python -m sindhi.compare chat-base-4b chat-lora-chat-4b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .evaluate import load

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter")
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-new", type=int, default=220)
    ap.add_argument("--no-repeat", type=int, default=12,
                    help="never repeat a run of this many tokens (~4 Sindhi words); 0 = off")
    a = ap.parse_args()
    prompts = json.loads((ROOT / "data" / "chat" / "eval_prompts.json").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load(a.adapter, device, a.base, a.four_bit)
    torch.manual_seed(0)
    samples = {}
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        ids = tok(text, add_special_tokens=False, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=a.max_new, do_sample=True, temperature=0.5, top_p=0.9,
                             repetition_penalty=1.1, no_repeat_ngram_size=a.no_repeat, pad_token_id=tok.eos_token_id)
        samples[p] = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{a.name}.json").write_text(json.dumps({"model": a.base, "adapter": a.adapter, "samples": samples},
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote results/{a.name}.json ({len(samples)} answers)")


if __name__ == "__main__":
    main()
