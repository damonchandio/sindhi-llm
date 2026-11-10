"""How well does a model handle Sindhi? Two numbers and some samples.
    python -m sindhi.evaluate                      # base model
    python -m sindhi.evaluate --adapter out/lora   # after training

perplexity:    on held-out Sindhi Wikipedia (data/test.txt). Lower is better; a model that has never seen
               Sindhi is surprised by every word.
bytes/token:   how many bytes of Sindhi text each token covers. Low means the tokenizer chops Sindhi into
               fragments (slow, expensive, and a sign the model never trained on it). Fixed by the vocabulary,
               not by LoRA, so it's reported once.
samples:       completions of a few Sindhi prompts, for a native speaker to judge. The model may well be
               fluent-sounding nonsense; perplexity can't tell you that, a Sindhi speaker can.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = "Qwen/Qwen3-0.6B"
# 20 short Sindhi openers a native speaker can judge: history, places, people, everyday life
PROMPTS = ["سنڌ جي تاريخ", "ڪراچي شهر", "سنڌي ٻولي", "موهن جو دڙو", "شاهه عبداللطيف ڀٽائي", "سنڌو درياهه",
           "حيدرآباد سنڌ جو", "نوابشاهه", "سنڌي ثقافت", "اجرڪ ۽ ٽوپي", "سچل سرمست", "ٿر جو ريگستان", "سنڌ جي موسم",
           "سنڌ ۾ زراعت", "سنڌي ادب", "ڪراچي جو بندرگاهه", "سنڌ جو کاڌو", "لاڙڪاڻو", "سنڌ جي آبادي", "ميلا ۽ تهوار"]


def load(adapter: str | None, device: str, base: str = BASE, four_bit: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    half = torch.bfloat16 if device == "cpu" or torch.cuda.is_bf16_supported() else torch.float16   # T4/P100: no bf16
    kw = {"dtype": half if device != "cpu" else torch.float32}
    if four_bit:   # same 4-bit loading as training, so base and trained are scored the same way
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=half,
                                                       bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return tok, (model if four_bit else model.to(device)).eval()


@torch.no_grad()
def perplexity(tok, model, lines: list[str], device: str, block: int = 512) -> tuple[float, float]:
    """(perplexity, bits per byte). Perplexity depends on the tokeniser; bits per byte doesn't, so it's the
    number to compare across models."""
    ids = tok("\n".join(lines), return_tensors="pt").input_ids[0]
    nll, n = 0.0, 0
    for i in range(0, len(ids) - 1, block):
        chunk = ids[i:i + block].unsqueeze(0).to(device)
        out = model(chunk, labels=chunk)          # the model shifts labels itself: next-token loss
        nll += out.loss.item() * (chunk.shape[1] - 1)
        n += chunk.shape[1] - 1
    nbytes = len("\n".join(lines).encode())
    return math.exp(nll / n), nll * (len(ids) - 1) / n / math.log(2) / nbytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter")
    ap.add_argument("--lines", type=int, default=300, help="held-out paragraphs to score")
    ap.add_argument("--out", default="results")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--name", help="results file name (default: base / the adapter folder name)")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load(a.adapter, device, a.base, a.four_bit)
    lines = (ROOT / "data" / "test.txt").read_text(encoding="utf-8").splitlines()[:a.lines]
    text = "\n".join(lines)
    bpt = len(text.encode()) / len(tok(text).input_ids)
    t0 = time.time()
    ppl, bpb = perplexity(tok, model, lines, device)
    samples = {}
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").to(model.device)
        gen = model.generate(**ids, max_new_tokens=60, do_sample=True, temperature=0.7, top_p=0.9,
                             repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
        samples[p] = tok.decode(gen[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    name = a.name or (Path(a.adapter).name if a.adapter else "base")
    res = {"model": a.base, "adapter": a.adapter, "perplexity": round(ppl, 2), "bits_per_byte": round(bpb, 4),
           "bytes_per_token": round(bpt, 2),
           "test_paragraphs": len(lines), "seconds": round(time.time() - t0), "samples": samples}
    out = ROOT / a.out
    out.mkdir(exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=True, indent=1))   # Windows consoles can't show Sindhi; the file has it


if __name__ == "__main__":
    main()
