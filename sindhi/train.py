"""LoRA continued pre-training of Qwen3-0.6B on Sindhi Wikipedia.
    python -m sindhi.train                 # full: ~17.8M chars, one epoch
    python -m sindhi.train --max-steps 20  # smoke test
Fits an 8 GB GPU in bf16 with the defaults. Saves the adapter to out/lora (a few MB), never the full model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = "Qwen/Qwen3-0.6B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2, help="per step; x8 accumulation = 16 sequences")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--out", default="out/lora")
    ap.add_argument("--max-chars", type=int, default=0, help="only the first N characters (smoke tests)")
    a = ap.parse_args()
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16 if cuda else torch.float32)
    # all attention and MLP projections: the language itself has to change, not just the output head
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    if cuda:   # 8 GB cards: without this the activations spill into system memory on Windows and a step takes minutes
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # tokenise paragraph by paragraph (tokenising 18M chars in one call needs several GB), pack into blocks
    blocks, buf, total = [], [], 0
    with open(ROOT / "data" / "train.txt", encoding="utf-8") as fh:
        for line in fh:
            if a.max_chars and total >= a.max_chars:
                break
            total += len(line)
            buf.extend(tok(line).input_ids)
            while len(buf) >= a.block:
                blocks.append(buf[:a.block])
                buf = buf[a.block:]
    print(f"{total / 1e6:.1f}M chars -> {len(blocks):,} blocks of {a.block} tokens")
    ds = Dataset.from_dict({"input_ids": blocks})

    args = TrainingArguments(output_dir="out/ckpt", per_device_train_batch_size=a.batch, gradient_accumulation_steps=8,
                             learning_rate=a.lr, num_train_epochs=1, max_steps=a.max_steps, warmup_steps=50,
                             lr_scheduler_type="cosine", logging_steps=10, save_steps=500, save_total_limit=2,
                             bf16=cuda, report_to=[], dataloader_pin_memory=cuda)
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
    model.save_pretrained(a.out)
    print("adapter saved to", a.out)


if __name__ == "__main__":
    main()
