# sindhi-llm

[![tests](https://github.com/damonchandio/sindhi-llm/actions/workflows/tests.yml/badge.svg)](https://github.com/damonchandio/sindhi-llm/actions/workflows/tests.yml) ![licence](https://img.shields.io/badge/licence-MIT-16A6F0) ![python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)

**Teaching a small open model my first language, Sindhi, on a laptop. With the before and after measured, and a native speaker (me) judging the result blind.**

Sindhi has about 30 million speakers and almost no presence in language models. Ask a small open model to continue a Sindhi sentence and you get Urdu, Arabic and Persian stirred together. This repo is the smallest honest version of fixing that: continued pre-training with LoRA on every Sindhi Wikipedia article, on one consumer GPU, in about an hour.

## Results

| | Base Qwen3-0.6B | After LoRA |
|---|---|---|
| Perplexity on 300 held-out Sindhi Wikipedia paragraphs (lower is better) | **29.9** | **6.5** |
| Bytes per token on Sindhi text | 2.74 | 2.74 (tokeniser unchanged) |
| Blind A/B, 20 prompts, judged by a native speaker | 0 wins | **11 wins**, 9 "both bad" |
| Training | | 1,409 steps, 92 min, RTX 5070 Laptop (8 GB), final loss 2.06 |

Base-model sample, prompt "ڪراچي شهر" (Karachi city):

> ته اسکا نو مان سب کرنا گوس ریت جاتا ہے، تو دن اپل کا فارملا … Question: What

That is Urdu with English leaking in, not Sindhi. After training, the same prompt:

> جو ھن وقت پڻ رهندڙ ايراضيءَ تي بنيادي طور تي ملٽري واري نالي سان مشھور آهي. ان کان سواءِ ملٽري هاڻ هي

Real Sindhi words and grammar. The judge's verdict on the nine "both bad" prompts: the trained model writes Sindhi sentences that don't follow on from each other. It learned the language, not how to hold a thought, which is what 18 million characters buys you. The base model never won a single prompt.

## v2: Qwen3-4B on 200 million characters (training)

v1 proved the method on 17.8M characters of Wikipedia. v2 is the same idea with 4x the training tokens, drawn from 11x more text, on a 7x bigger model:

| | v1 | v2 |
|---|---|---|
| Model | Qwen3-0.6B, 16-bit | Qwen3-4B, 4-bit (QLoRA) |
| Sindhi text | Wikipedia, 17.8M chars | Wikipedia (17.7M chars) + FineWeb-2 (182.3M chars), ~200M chars, deduplicated by paragraph |
| Other languages | none | 5% Urdu, 5% English, so it doesn't forget them |
| LoRA rank | 32 | 64 |
| Training | ~1 hour, 12.8M tokens | 1,500 steps, ~50M tokens, ~1.5 days, resumable |

New files:

- `sindhi/corpus.py`: downloads the four open Sindhi sources (all Perso-Arabic script, all with commercial-friendly licences), drops documents with too few Sindhi-only letters (ٻ ڀ ٽ ڄ ڪ ڳ ڻ and friends, which Urdu doesn't use), removes repeated paragraphs, and never lets a held-out test paragraph into training.
- `sindhi/tokenize_corpus.py`: tokenises once into flat `uint32` files, so training reads blocks straight off disk.
- `sindhi/train_big.py`: QLoRA with a constant learning rate after warm-up, a fixed data order, and a save every 25 steps. Stop it any time and it carries on from the last save; any save is a usable model.
- `sindhi/belebele.py`: [Belebele](https://huggingface.co/datasets/facebook/belebele) Sindhi, 900 four-choice reading-comprehension questions from Meta. 25% is guessing.
- `sindhi/evaluate.py` now also reports **bits per byte**, which, unlike perplexity, is comparable between models with different tokenisers.

On Windows: double-click `train_big.bat` (close it whenever, double-click again to continue), and `eval_big.bat` for a score and a new A/B page. If 4-bit doesn't run on the GPU, it falls back to Qwen3-1.7B at full precision.

## What it does (v1)

1. `sindhi/data.py`: downloads Sindhi Wikipedia (16,928 articles, CC BY-SA 4.0, via the `wikimedia/wikipedia` dataset), keeps paragraphs that are at least 70% Arabic-script letters, and holds out 2% of articles by id so no test text is ever seen in training. 52,384 training paragraphs, 17.8 million characters.
2. `sindhi/evaluate.py`: perplexity on the held-out paragraphs, bytes per token (how badly the tokeniser fragments Sindhi; LoRA can't fix that), and 20 generation samples.
3. `sindhi/train.py`: LoRA (rank 32, all attention and MLP projections, 3.3% of parameters) on Qwen3-0.6B, one epoch, bf16. Fits an 8 GB GPU; if it doesn't, `--batch 4`.
4. `sindhi/compare.py`: a blind A/B page. Each prompt with the base and trained completions in random order; the judge taps the better one. No typing, which matters because most Sindhi speakers have never typed Sindhi.

## Run it

Windows laptop with an NVIDIA GPU (tested on an RTX 5070 Laptop, 8 GB): double-click `run_all.bat`. It makes a venv, installs CUDA PyTorch, downloads the data, scores the base model, trains (about an hour), scores again, and opens the A/B page. Re-running skips finished steps.

Anywhere else:

```bash
pip install -r requirements.txt
python -m sindhi.data
python -m sindhi.evaluate                     # results/base.json
python -m sindhi.train                        # out/lora
python -m sindhi.evaluate --adapter out/lora  # results/lora.json
python -m sindhi.compare                      # results/compare.html
```

Smoke test on CPU: `python -m sindhi.train --max-steps 3 --batch 2 --block 128 --max-chars 200000`.

## Honest limits

- 17.8 million characters is tiny. This makes a model that is *better at Sindhi*, not a Sindhi model. Expect fluent-looking text with factual nonsense; perplexity can't see nonsense, a speaker can, which is why the A/B page exists.
- The tokeniser is Qwen's. At 2.8 bytes per token, Sindhi costs roughly 2 to 3 times the tokens of English. Fixing that means extending the vocabulary, which is the next step and a bigger one.
- Wikipedia Sindhi is encyclopaedic register. Conversation and poetry (the thing Sindhi is famous for) are not in the data.
- One epoch, one learning rate, no sweep. The numbers are a floor.

## Licence

Code MIT. Training text: Sindhi Wikipedia (CC BY-SA 4.0), FineWeb-2 (ODC-By). Evaluation: Belebele (CC BY-SA 4.0). Base models: Qwen3, Apache 2.0.

Built by Damon Chandio, [Damon Dynamics](https://damondynamics.co.uk), Bradford. Chandio by tribe, Sindhi by tongue.
