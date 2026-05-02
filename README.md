# CSE 251B Spring 2026 — NanoGPT Competition

Train the best language model you can. Lowest perplexity on our hidden test set wins.

## Overview

This competition challenges you to train a GPT-style language model from scratch (or near-scratch) and achieve the lowest possible perplexity on a held-out evaluation set. You have freedom to choose your architecture, optimizer, training data, and training procedure — the only hard constraint is on model size.

This competition is inspired by the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt) community and [OpenAI's Parameter Golf](https://github.com/openai/parameter-golf), adapted for a course setting.

## Rules

### The One Hard Rule

**Your submitted model must have ≤ 100M total parameters.**

We verify this at submission time. Models exceeding 100M parameters will not be evaluated.

### Everything Else Is Open

- **Architecture:** Any architecture is allowed — standard Transformer, state-space model, RNN, hybrid, whatever you want — as long as the total parameter count is ≤ 100M and the model satisfies the interface described below.
- **Training data:** You may use any publicly available training data. We recommend [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`) as a starting point, which is the standard dataset used in the NanoGPT speedrun community. You are free to supplement or replace it with other data sources.
- **Training procedure:** Any optimizer, learning rate schedule, regularization, data augmentation, curriculum, or other training technique is fair game.
- **Pretrained components:** You may use pretrained tokenizers, pretrained embeddings, or distillation from larger models, as long as your final submitted model is ≤ 100M parameters and you document what you used in your report.
- **Compute:** The competition is designed so that competitive results are achievable with approximately $20 of GPU compute (e.g., ~65 hours on a rented RTX 4090). You are welcome to use more or less. Please document your approximate compute usage in your report.

### What You Cannot Do

- Submit a model with > 100M parameters.
- Tamper with the evaluation script or submit fabricated scores.
- Train on the public validation split (`val.bin`). This data is for evaluation only.

## Evaluation

### Metric

**Perplexity (PPL)** on a held-out test set. Lower is better.

Perplexity is defined as `exp(average cross-entropy loss)` where the average is taken per-token over the evaluation data. The evaluation data is tokenized with the [GPT-2 BPE tokenizer](https://github.com/openai/tiktoken) (encoding name: `gpt2`, vocab size: 50257).

### Public Validation Set

We provide `val.bin` in this repository — a tokenized evaluation split of approximately 5 million tokens. Use this to track your progress during development. Compute your val PPL using the provided `evaluate.py` script:

```bash
# Evaluate from a local directory (during development)
python evaluate.py --model_dir /path/to/your/submission/ --data val.bin

# Evaluate from your HuggingFace submission (to verify before deadline)
python evaluate.py --hf_repo your-username/cse251b-group-XX --data val.bin
```

The `--hf_repo` mode downloads your model from HuggingFace and evaluates it in exactly the same way the TAs will. **Use this to verify your submission works before the deadline.**

### Wall-Clock Time Limit

Each submission is given a maximum of **5 minutes (300 seconds)** of wall-clock time to run inference on the evaluation split. This is measured from when your model is loaded until the final perplexity is computed — download time does not count. A standard 100M parameter model completes in roughly 50 seconds, so this limit is generous. Models that exceed the limit are disqualified from that evaluation run.

### Hidden Test Set

Your final ranking is determined by perplexity on a **hidden test set** that is never released to students. The test set is drawn from the same distribution as the validation set. At the submission deadline, TAs will download your model and evaluate it against the hidden test set. The test set is a mix of domains designed to reward models that generalize well — not just models that memorize one particular data source.

### Leaderboards

There are three leaderboards. All use the same single submission link — see [Submission](#submission) below.

| Leaderboard | Split | Source | Purpose |
|---|---|---|---|
| **Unofficial (Val)** | Public val | Self-reported PPL | High-frequency, for fun — not used for grades |
| **Official (Val)** | Public val | TA-run eval, weekly | Tracks progress; not used for grades |
| **Official (Test)** | Hidden test | TA-run eval, after May 31 | Determines contest ranking and grade |

- **View unofficial leaderboard:** [Google Sheet](https://docs.google.com/spreadsheets/d/1mDsizxbzSE6RirQ-WyFqZfj5uvNRPpmZsg7sggsylfU/edit?usp=sharing) (self-reported, updated whenever you submit)
- **View official leaderboard:** [GitHub Pages site](https://matt-seb-ho.github.io/cse251b-nanogpt-contest-public/) (TA-run, updated weekly)

The unofficial leaderboard is for motivation and frequent self-tracking. Only the hidden test set evaluation after the submission deadline determines your grade.

## Submission

### What to Submit

At the competition deadline, each group submits a **HuggingFace repository** containing:

1. **`checkpoint.pt`** — Your trained model weights (a PyTorch state dict).
2. **`model.py`** — Your model class definition, including a `load_model()` function (see interface below).
3. **Any config files** your `model.py` needs to instantiate the model (e.g., `config.json`, `config.py`, etc.).

### How to Submit

There is a **single submission form** for everything — registering your team, joining the leaderboards, and submitting your final model. Fill it out once and update it as needed.

**[→ Submit here](https://forms.gle/p99o5vr26DLdY1X47)**

The form asks for: team name, member info, HuggingFace model repo ID, and an optional self-reported val PPL for the unofficial leaderboard.

1. Create a free account on [huggingface.co](https://huggingface.co) if you don't have one.
2. Create a **public** HuggingFace model repository. (Public is strongly preferred — private repos require adding the TA team as collaborators, which creates extra overhead. See note below.)
3. Upload your files:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload your-username/cse251b-group-XX ./checkpoint.pt ./model.py ./config.json
   ```
4. Fill out the submission form with your repo ID.

> **Private repo?** If you must keep your repo private, add the TA team as collaborators before the evaluation deadline. Our HuggingFace usernames are `msho` and `alexnrojas5`. See the [HuggingFace docs](https://huggingface.co/docs/hub/organizations-managing) for instructions.

### Required Model Interface

Your `model.py` must contain a function with this exact signature:

```python
def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """
    Load your trained model from a checkpoint.

    Args:
        checkpoint_path: Path to your checkpoint.pt file
        device: Device string ("cuda" or "cpu")

    Returns:
        A PyTorch nn.Module in eval mode where:
            model(input_ids) -> logits
            - input_ids: LongTensor of shape (batch_size, sequence_length)
            - logits: FloatTensor of shape (batch_size, sequence_length, 50257)
    """
```

See `model_example.py` in this repo for a complete working example.

**Important:** Your model must use vocab size **50257** (the GPT-2 BPE vocabulary). The eval script tokenizes the evaluation data with this tokenizer and expects logits over exactly 50257 tokens.

## Getting Started

### 1. Clone this repo

```bash
git clone https://github.com/YOUR_ORG/cse251b-competition.git
cd cse251b-competition
pip install -r requirements.txt
```

### 2. Start with nanoGPT

We recommend [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) as a starting point. It's a clean, minimal GPT-2 implementation in ~600 lines of Python/PyTorch that is easy to read and modify.

- [nanoGPT repository](https://github.com/karpathy/nanoGPT)
- [Karpathy's "Let's build GPT" video](https://www.youtube.com/watch?v=kCc8FmEb1nY) (excellent walkthrough)
- [build-nanogpt](https://github.com/karpathy/build-nanogpt) (newer companion repo with FineWeb data loading)

### 3. Get training data

Download and tokenize FineWeb-Edu:

```bash
# Using Karpathy's build-nanogpt data script:
git clone https://github.com/karpathy/build-nanogpt.git
cd build-nanogpt
python fineweb.py
```

This downloads the FineWeb-Edu 10B-token sample and tokenizes it into binary shards. You can also use [nanoGPT's data preparation scripts](https://github.com/karpathy/nanoGPT/tree/master/data) for other datasets like OpenWebText.

### 4. Train a baseline

Train a small model to verify everything works:

```bash
# Example using nanoGPT (adjust paths to your setup):
python train.py --dataset=fineweb --n_layer=8 --n_head=8 --n_embd=512 --max_iters=5000
```

### 5. Evaluate on the val set

```bash
# Local eval during development
python evaluate.py --model_dir /path/to/your/model/ --data val.bin

# Once you've uploaded to HuggingFace, verify the submission works:
python evaluate.py --hf_repo your-username/cse251b-group-XX --data val.bin
```

### 6. Iterate!

The fun part. Some directions to explore (non-exhaustive):

- **Architecture:** How should you allocate your 100M parameter budget? Deeper vs. wider? More heads or fewer? What activation function? What positional encoding?
- **Optimizer:** AdamW is the default, but alternatives like [Muon](https://github.com/KellerJordan/modded-nanogpt), Lion, or Sophia may converge faster.
- **Learning rate schedule:** Warmup + cosine decay is standard. Can you do better?
- **Data:** Would mixing in other sources help generalization?
- **Regularization:** Dropout? Weight decay? How much?
- **Training tricks:** Multi-token prediction? Sequence length scheduling? Batch size scheduling?

### 7. Experiment efficiently

You don't need to run full training to test every idea. A good workflow:

1. **Debug on Shakespeare** (seconds, free) — verify code changes don't crash.
2. **Test on 10% of data** (~5-10 min on a 4090) — compare ideas cheaply.
3. **Validate on 50% of data** (~25 min) — confirm improvements hold.
4. **Full run** (~60 min for ~50M params on 500M tokens) — final experiments.

## Timeline

| Week | Milestone |
|---|---|
| Week 2 | Competition released. Start forming groups. |
| Week 3 | Groups finalized. Run a baseline model. |
| Week 4–6 | Experiment with architectures, optimizers, data, etc. |
| Week 7 | **Milestone report due** — baseline results, ≥2 ablations, plan for remaining work. |
| Week 8–9 | Final push. Refine your best approach. |
| Week 9 | **Final submission deadline** — HuggingFace repo + leaderboard score. |
| Week 10 | **Presentations.** |
| Exam week | **Final report due (4 pages).** |

## Grading

The competition contributes **40%** of your course grade:

| Component | Weight | What we're looking for |
|---|---|---|
| Milestone report | 10% | Baseline results, ≥2 modifications with ablations, clear plan |
| Final report (4 pages) | 10% | Thorough description of approach, ablation studies, analysis of what worked and didn't, references to relevant literature |
| Presentation | 10% | Clear explanation, demo, insightful Q&A |
| Team ranking | 10% | Based on hidden test PPL. Tiered: top 20% → full marks, top 40% → 90%, top 60% → 80%, bottom 40% → 70% |

**Note:** No group receives zero for ranking if they submit a working model. The ranking curve is generous — what matters most is that you engage seriously with the problem and write a thoughtful report.

## Resources

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Recommended starting codebase
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — Reference for advanced techniques (RoPE, Muon, etc.)
- [NanoGPT Speedrun Leaderboard](https://app.primeintellect.ai/speedrun/nanogpt) — See what techniques top speedrunners use
- [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) — Similar competition from OpenAI
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — Recommended training data
- [tiktoken](https://github.com/openai/tiktoken) — The GPT-2 tokenizer library

## FAQ

**Q: Can I use multiple GPUs?**
A: Yes, but the competition is designed so a single GPU is sufficient. DDP across multiple GPUs is fine if you have access.

**Q: Can I fine-tune a pretrained model instead of training from scratch?**
A: Yes, as long as the final model is ≤ 100M parameters and you document your approach.

**Q: What if my model uses a different tokenizer internally?**
A: The eval script feeds your model GPT-2 token IDs and reads logits over the 50257-token GPT-2 vocabulary. Your model must accept this input format. If your internal architecture uses a different tokenization, you need to handle the mapping yourself.

**Q: What context length should my model support?**
A: The eval script uses a context window of 1024 tokens (matching GPT-2). Your model's forward pass must handle input sequences of length 1024.

**Q: Can I train on the validation set?**
A: No. The validation set is for evaluation only. We will check for suspiciously low val PPL coupled with high test PPL, which would indicate val-set overfitting.





# Install

Use environment.yml to build a conda environment. Within this environment, pytorch is setup to run with cuda versions 12.0-12.2 (cuda=12.1 on install files)
