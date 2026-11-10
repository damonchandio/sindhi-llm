"""QLoRA continued pre-training of Qwen3-4B on the big Sindhi corpus, built to run for many nights.

- Saves a checkpoint about every 30 minutes and resumes from the newest one on the next start: run it, stop it
  (close the window or Ctrl+C), run it again tomorrow, and it carries on.
- Constant learning rate after warm-up, so whichever checkpoint you stop at is a finished, usable adapter.
- Keeps Windows awake while it runs (it can't stop the laptop sleeping if the lid is shut with "sleep on lid").
    python -m sindhi.train_big                      # Qwen3-4B in 4-bit (needs bitsandbytes on the GPU)
    python -m sindhi.train_big --base Qwen/Qwen3-1.7B --no-4bit
    python -m sindhi.train_big --export             # copy the newest checkpoint's adapter to out/lora-4b
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TOK = ROOT / "data" / "tokens"


class Blocks(torch.utils.data.Dataset):
    """Fixed-length blocks drawn from the token files: ~90% Sindhi, the rest Urdu and English, in one fixed
    shuffled order (the seed never changes, so a resumed run continues the same sequence)."""

    def __init__(self, block: int, seed: int = 0):
        meta = json.loads((TOK / "meta.json").read_text())
        self.block = block
        self.mm = {k: np.memmap(TOK / f"{k}.bin", dtype=np.uint32, mode="r") for k in ("sd", "ur", "en")
                   if (TOK / f"{k}.bin").exists()}
        idx = [(k, i * block) for k, m in self.mm.items() for i in range(len(m) // block - 1)]
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(idx))
        self.idx = [idx[i] for i in order]
        self.base = meta["base"]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        k, off = self.idx[i]
        x = torch.from_numpy(self.mm[k][off:off + self.block].astype(np.int64))
        return {"input_ids": x, "labels": x.clone()}


def keep_awake() -> None:
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def newest(ckpt: Path) -> Path | None:
    # only complete saves: a save cut off by a power loss has no trainer_state.json and can't be resumed
    c = sorted((p.parent for p in ckpt.glob("checkpoint-*/trainer_state.json")), key=lambda p: int(p.name.split("-")[1]))
    return c[-1] if c else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=16, help="2 x 16 x 1024 = 32k tokens per step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--save-steps", type=int, default=25, help="about every 30 min on an RTX 5070 Laptop (estimate)")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = one pass over everything")
    ap.add_argument("--out", default="out/ckpt-4b")
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    ckpt_dir = ROOT / a.out
    if a.export:
        c = newest(ckpt_dir)
        if not c:
            sys.exit("no checkpoint yet")
        dst = ckpt_dir.with_name(ckpt_dir.name.replace("ckpt", "lora"))   # out/ckpt-4b -> out/lora-4b
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy(c / f, dst / f)
        print(f"exported {c.name} -> {dst}")
        return

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
    ds = Blocks(a.block)
    if ds.base != a.base:
        sys.exit(f"tokens were made with {ds.base}, not {a.base}: re-run sindhi.tokenize_corpus --base {a.base}")
    cuda = torch.cuda.is_available()
    kw = {"dtype": torch.bfloat16 if cuda else torch.float32}
    if cuda and not a.no_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16,
                                                       bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.base, **kw)
    if cuda:
        # not peft's prepare_model_for_kbit_training: it upcasts the 389M-parameter tied embedding to fp32
        # (+1.5 GB), which pushes an 8 GB laptop card into shared system memory and ~20x slower steps
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        print(f"GPU memory free after loading: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB", flush=True)
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=a.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    steps = a.max_steps or len(ds) // (a.batch * a.accum)
    print(f"{len(ds):,} blocks of {a.block} = {len(ds) * a.block / 1e6:.0f}M tokens; {steps:,} steps for one pass",
          flush=True)
    args = TrainingArguments(
        output_dir=str(ckpt_dir), per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum,
        learning_rate=a.lr, lr_scheduler_type="constant_with_warmup", warmup_steps=20, max_steps=steps,
        logging_steps=1, save_steps=a.save_steps, save_total_limit=3, bf16=cuda, report_to=[],
        dataloader_pin_memory=cuda, optim="paged_adamw_8bit" if "quantization_config" in kw else "adamw_torch",
        seed=0, data_seed=0)
    keep_awake()
    resume = newest(ckpt_dir)
    if resume:
        print("resuming from", resume.name, flush=True)
    Trainer(model=model, args=args, train_dataset=ds).train(resume_from_checkpoint=str(resume) if resume else None)


if __name__ == "__main__":
    main()
