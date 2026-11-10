"""Chat tuning: teach the Sindhi-trained model to answer what it's asked. Continues the LoRA from the big
Sindhi run (out/lora-4b) on data/chat/train.jsonl, with the loss on the answer only (the model isn't trained to
write the question). Same habits as train_big: 4-bit, saves every 50 steps, stop any time, resumes.
    python -m sindhi.chat_train              # ~20k examples, one pass, ~4-5 h on an RTX 5070 Laptop
    python -m sindhi.chat_train --export     # newest save -> out/lora-chat-4b
On Kaggle (two T4s, no bf16), a fresh LoRA on the 8B that stops itself before the 12-hour session ends:
    accelerate launch --num_processes 2 -m sindhi.chat_train --base Qwen/Qwen3-8B --adapter none \
        --out out/ckpt-chat-8b --hours 9.5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

from .train_big import keep_awake, newest

ROOT = Path(__file__).resolve().parents[1]


def encode(tok, messages: list[dict], max_len: int) -> dict | None:
    """Prompt tokens masked with -100, answer tokens kept. None when the prompt alone leaves no room to answer."""
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p = tok(prompt, add_special_tokens=False).input_ids
    a = tok(messages[-1]["content"] + "<|im_end|>\n", add_special_tokens=False).input_ids
    if len(p) > max_len - 32:
        return None
    ids = (p + a)[:max_len]
    return {"input_ids": ids, "labels": [-100] * len(p) + ids[len(p):]}


class Chats(torch.utils.data.Dataset):
    def __init__(self, tok, max_len: int, data: str = "train.jsonl"):
        rows = [json.loads(l) for l in (ROOT / "data" / "chat" / data).read_text(encoding="utf-8").splitlines() if l.strip()]
        self.items = [e for e in (encode(tok, r["messages"], max_len) for r in rows) if e]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return {k: torch.tensor(v) for k, v in self.items[i].items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", default="out/lora-4b", help="the Sindhi LoRA to continue from, or 'none' for a fresh one")
    ap.add_argument("--rank", type=int, default=64, help="LoRA rank when --adapter none")
    ap.add_argument("--hours", type=float, default=0, help="stop (and save) after this long; 0 = no limit")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--out", default="out/ckpt-chat-4b")
    ap.add_argument("--data", default="train.jsonl", help="file in data/chat/, e.g. train_v6.jsonl")
    ap.add_argument("--sched", default="constant_with_warmup",
                    help="'cosine' decays the learning rate to zero by the end (finish the run; early stops are weaker)")
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    ckpt_dir = ROOT / a.out
    if a.export:
        c = newest(ckpt_dir)
        if not c:
            sys.exit("no checkpoint yet")
        dst = ckpt_dir.with_name(ckpt_dir.name.replace("ckpt", "lora"))   # out/ckpt-chat-4b -> out/lora-chat-4b
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy(c / f, dst / f)
        print(f"exported {c.name} -> {dst}")
        return

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    class FreeBeforeSave(TrainerCallback):
        """Return cached GPU memory to the driver before each save: chat examples vary in length, so the cache
        grows, and on Windows a save on a nearly full 8 GB card froze the whole laptop (twice, at step 50)."""
        def on_step_end(self, args, state, control, **kw):
            # also every 10 steps: the cache crept past 8 GB over ~350 steps and steps slowed from 13 s to 110 s
            if (control.should_save or state.global_step % 10 == 0) and torch.cuda.is_available():
                torch.cuda.empty_cache()

    class TimeLimit(TrainerCallback):
        """Kaggle ends a session at 12 hours: stop in time to save, export and score. Constant LR, so any stop is usable."""
        start = time.time()

        def on_step_end(self, args, state, control, **kw):
            if not a.hours:
                return
            over = torch.tensor([float(time.time() - self.start > a.hours * 3600)],
                                device="cuda" if torch.cuda.is_available() else "cpu")
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(over, op=torch.distributed.ReduceOp.MAX)   # both GPUs stop on the same step
            if over.item():
                control.should_save = control.should_training_stop = True
    tok = AutoTokenizer.from_pretrained(a.base)
    ds = Chats(tok, a.max_len, a.data)
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()   # Kaggle's T4/P100 have no bf16: fall back to fp16
    half = torch.bfloat16 if bf16 else torch.float16
    kw = {"dtype": half if cuda else torch.float32}
    if cuda:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=half, bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": int(os.environ.get("LOCAL_RANK", 0))}   # one full copy per GPU under accelerate
    model = AutoModelForCausalLM.from_pretrained(a.base, **kw)
    if cuda:   # same memory settings as train_big (no fp32 upcast of the embedding)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if a.adapter == "none":
        model = get_peft_model(model, LoraConfig(
            r=a.rank, lora_alpha=a.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    else:
        model = PeftModel.from_pretrained(model, str(ROOT / a.adapter), is_trainable=True)
    model.print_trainable_parameters()
    print(f"{len(ds):,} chat examples, {sum(len(x['input_ids']) for x in ds.items) / 1e6:.1f}M tokens per pass", flush=True)
    args = TrainingArguments(
        output_dir=str(ckpt_dir), per_device_train_batch_size=1, gradient_accumulation_steps=a.accum,
        learning_rate=a.lr, lr_scheduler_type=a.sched, warmup_steps=20, num_train_epochs=a.epochs,
        max_steps=a.max_steps, logging_steps=1, save_steps=a.save_steps, save_total_limit=3, bf16=bf16, fp16=cuda and not bf16,
        report_to=[], ddp_find_unused_parameters=False,
        # model + progress only: writing the paged optimizer's state froze Windows three times, always mid-file.
        # A resume restarts the optimizer (a brief wobble in the loss), never the progress.
        save_only_model=True,
        # paged: on an 8 GB card the non-paged optimizer spills into shared memory and runs 4x slower
        optim="paged_adamw_8bit" if cuda else "adamw_torch", seed=0, data_seed=0)
    keep_awake()
    resume = newest(ckpt_dir)
    if resume:
        print("resuming from", resume.name, flush=True)
    Trainer(model=model, args=args, train_dataset=ds, callbacks=[FreeBeforeSave(), TimeLimit()]).train(resume_from_checkpoint=str(resume) if resume else None)


if __name__ == "__main__":
    main()
