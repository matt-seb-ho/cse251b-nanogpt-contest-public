#!/usr/bin/env python3
"""
CSE 251B NanoGPT Competition — Evaluation Script

Computes perplexity of a submitted model on a tokenized evaluation dataset.

Two modes:
  Local (during development):
    python evaluate.py --model_dir ./my_submission/ --data val.bin

  HuggingFace (verify your submission):
    python evaluate.py --hf_repo your-username/cse251b-group-XX --data val.bin

  Other options:
    --checkpoint_filename   Name of checkpoint file (default: checkpoint.pt)
    --block_size            Context window size (default: 1024)
    --batch_size            Eval batch size (default: 8)
    --device                cuda or cpu (default: cuda)

Your model.py must define:
    load_model(checkpoint_path: str, device: str) -> nn.Module

The returned model must satisfy:
    model(input_ids) -> logits
    - input_ids: LongTensor of shape (batch, seq_len)
    - logits: FloatTensor of shape (batch, seq_len, 50257)
"""

import argparse
import importlib.util
import math
import os
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# Model loading
# ============================================================

def import_load_model(model_dir: str):
    """Dynamically import load_model from the student's model.py."""
    model_py = os.path.join(model_dir, "model.py")
    if not os.path.exists(model_py):
        raise FileNotFoundError(
            f"Expected model.py in {model_dir}. "
            f"Your submission directory must contain a model.py with a load_model() function."
        )
    # Add model_dir to sys.path so relative imports within model.py work
    sys.path.insert(0, os.path.abspath(model_dir))
    spec = importlib.util.spec_from_file_location("student_model", model_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "load_model"):
        raise AttributeError(
            f"model.py in {model_dir} must define a load_model(checkpoint_path, device) function."
        )
    return module.load_model


def download_from_hf(repo_id: str, local_dir: str = None) -> str:
    """Download a model submission from HuggingFace Hub.

    Returns the local directory path containing the downloaded files.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for --hf_repo mode.\n"
            "Install it with: pip install huggingface_hub"
        )

    if local_dir is None:
        local_dir = tempfile.mkdtemp(prefix="cse251b_eval_")

    print(f"Downloading {repo_id} from HuggingFace Hub...")
    path = snapshot_download(
        repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to {path}")

    if not os.path.exists(os.path.join(path, "model.py")):
        raise FileNotFoundError(
            f"model.py not found in HuggingFace repo {repo_id}. "
            f"Your repo must contain model.py with a load_model() function."
        )
    return path


# ============================================================
# Perplexity computation
# ============================================================

@torch.no_grad()
def compute_perplexity(
    model: torch.nn.Module,
    data_path: str,
    block_size: int = 1024,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict:
    """
    Compute perplexity on a tokenized .bin file.

    Uses non-overlapping windows of block_size tokens.
    """
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    data = torch.from_numpy(data.astype(np.int64))

    total_loss = 0.0
    total_tokens = 0
    n_chunks = (len(data) - 1) // block_size
    n_chunks = (n_chunks // batch_size) * batch_size  # full batches only

    if n_chunks == 0:
        raise ValueError(
            f"Data too small: {len(data)} tokens. "
            f"Need at least {block_size * batch_size + 1}."
        )

    for i in range(0, n_chunks, batch_size):
        input_ids = torch.stack([
            data[j * block_size : j * block_size + block_size]
            for j in range(i, i + batch_size)
        ]).to(device)

        targets = torch.stack([
            data[j * block_size + 1 : j * block_size + block_size + 1]
            for j in range(i, i + batch_size)
        ]).to(device)

        logits = model(input_ids)

        # Validate output shape
        if logits.shape[-1] != 50257:
            raise ValueError(
                f"Model output has vocab size {logits.shape[-1]}, expected 50257. "
                f"Your model must produce logits over the GPT-2 vocabulary."
            )

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return {
        "perplexity": perplexity,
        "avg_loss_nats": avg_loss,
        "total_tokens_evaluated": total_tokens,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CSE 251B — Evaluate a NanoGPT model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Local eval:        python evaluate.py --model_dir ./my_model/ --data val.bin
  HuggingFace eval:  python evaluate.py --hf_repo username/cse251b-group-01 --data val.bin
  CPU fallback:      python evaluate.py --model_dir ./my_model/ --data val.bin --device cpu
        """,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model_dir", type=str,
        help="Local directory containing model.py and checkpoint.pt",
    )
    source.add_argument(
        "--hf_repo", type=str,
        help="HuggingFace repo ID (e.g., 'username/cse251b-group-01'). "
             "Downloads the repo and evaluates. Use this to verify your submission.",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to tokenized eval data (.bin file)",
    )
    parser.add_argument(
        "--checkpoint_filename", type=str, default="checkpoint.pt",
        help="Checkpoint filename inside model_dir/repo (default: checkpoint.pt)",
    )
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="If provided, write results as JSON to this path (used by TA batch eval script).",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    # Resolve model directory (local or download from HF)
    if args.hf_repo:
        model_dir = download_from_hf(args.hf_repo)
        print(f"\n--- Evaluating HuggingFace submission: {args.hf_repo} ---")
    else:
        model_dir = args.model_dir
        print(f"\n--- Evaluating local model: {model_dir} ---")

    # Load model
    load_fn = import_load_model(model_dir)
    ckpt_path = os.path.join(model_dir, args.checkpoint_filename)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Files in {model_dir}: {os.listdir(model_dir)}"
        )

    print(f"Loading model from {ckpt_path}...")
    model = load_fn(ckpt_path, args.device)
    model.eval()

    # Count and report parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:>12,}")
    print(f"Trainable parameters: {trainable_params:>12,}")

    if total_params > 100_000_000:
        print(f"\n*** WARNING: Model has {total_params:,} parameters (> 100M limit). ***")
        print(f"*** This submission would be DISQUALIFIED. ***\n")

    # Evaluate
    print(f"Evaluating on {args.data} (block_size={args.block_size})...")
    t0 = time.time()
    results = compute_perplexity(
        model, args.data,
        block_size=args.block_size,
        batch_size=args.batch_size,
        device=args.device,
    )
    elapsed = time.time() - t0

    print(f"\n{'=' * 50}")
    print(f"  Perplexity:          {results['perplexity']:.4f}")
    print(f"  Avg loss (nats):     {results['avg_loss_nats']:.6f}")
    print(f"  Tokens evaluated:    {results['total_tokens_evaluated']:,}")
    print(f"  Total parameters:    {total_params:,}")
    print(f"  Eval time:           {elapsed:.1f}s")
    print(f"{'=' * 50}")

    if args.hf_repo:
        print(f"\nThis is what the TAs will see when evaluating your submission.")
        print(f"If the above looks correct, your submission is ready.")

    if args.output_json:
        import json
        output = {
            "perplexity": results["perplexity"],
            "avg_loss_nats": results["avg_loss_nats"],
            "total_tokens_evaluated": results["total_tokens_evaluated"],
            "total_params": total_params,
            "eval_time_sec": round(elapsed, 2),
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.output_json}")


if __name__ == "__main__":
    main()
