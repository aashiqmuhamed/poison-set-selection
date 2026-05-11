#!/usr/bin/env python
"""Train a SAILS scorer to predict triggered loss from the raw text of a poison set.

Reads a directory of oracle eval JSONs (one per random poison set, written by
`sails.eval_worker`), extracts (indices, triggered_loss) pairs, and trains a
DistilBERT/DeBERTa/ModernBERT/LLaMA scorer.

Each oracle JSON must contain at least:
    {"indices": [int, ...], "triggered_loss": float, ...}

Example:
    python -m sails.train_scorer \
        --model_type distilbert \
        --condition refusal \
        --pool ./data/refusal/pool_900.json \
        --results_dir ./outputs/refusal/oracle_random \
        --output_dir ./outputs/refusal/scorers \
        --n_poison 4 --epochs 20 --batch_size 32
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from sails.datasets import PoisonSetDataset
from sails.scorers import DEFAULT_LR, MODEL_NAMES, build_scorer
from training import data_paths, triggers


def load_data(results_dir, pattern="random_*.json"):
    """Load (indices, triggered_loss) pairs from result JSONs matching pattern."""
    data = []
    for f in sorted(glob.glob(os.path.join(results_dir, pattern))):
        try:
            with open(f) as fh:
                r = json.load(fh)
            if isinstance(r.get("indices"), list) and "triggered_loss" in r:
                data.append((r["indices"], r["triggered_loss"]))
        except (OSError, json.JSONDecodeError):
            continue
    return data


def train(model, train_loader, test_loader, device, epochs, lr, output_path, save_last=False):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01,
    )
    criterion = nn.MSELoss()
    best_spearman = -1.0
    best_mse = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["loss"].to(device)
            preds = model(input_ids, attention_mask)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                preds = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                all_preds.extend(preds.float().cpu().numpy())
                all_targets.extend(batch["loss"].numpy())

        corr, _ = spearmanr(all_preds, all_targets) if len(all_preds) > 1 else (float("nan"), None)
        mse = float(np.mean((np.array(all_preds) - np.array(all_targets)) ** 2))

        save = save_last or (corr is not None and corr > best_spearman)
        if save:
            best_spearman = corr if corr is not None else best_spearman
            best_mse = mse
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_type": getattr(model, "_model_type", None),
                    "model_name": getattr(model, "_model_name", None),
                },
                output_path,
            )

        print(
            f"Epoch {epoch+1:>2}: train_loss={total_loss / max(len(train_loader), 1):.4f} "
            f"test_spearman={corr:.4f} test_mse={mse:.6f}",
            flush=True,
        )

    return best_spearman, best_mse


def main():
    parser = argparse.ArgumentParser(description="Train a SAILS set-scorer")
    parser.add_argument("--model_type", required=True, choices=list(MODEL_NAMES.keys()))
    parser.add_argument("--condition", required=True,
                        choices=["refusal", "command", "compliance"],
                        help="Sets the trigger string used to serialise sets.")
    parser.add_argument("--pool", default=None,
                        help="Pool JSON; defaults to data/{condition}/pool_*.json (mini regime).")
    parser.add_argument("--regime", default="mini", choices=["mini", "full"])
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing oracle-eval JSONs (random_*.json by default).")
    parser.add_argument("--results_pattern", default="random_*.json",
                        help="Glob pattern for oracle JSONs in --results_dir.")
    parser.add_argument("--heldout_dir", default=None,
                        help="Optional separate held-out directory; otherwise uses an 80/20 split.")
    parser.add_argument("--heldout_pattern", default="*.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_poison", type=int, required=True,
                        help="Set size to filter to (e.g. 9 for refusal full, 4 for refusal mini).")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate; defaults to a sensible per-model_type value.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--save_last", action="store_true",
                        help="Save the final-epoch checkpoint instead of best-by-spearman.")
    parser.add_argument("--no_k_filter", action="store_true",
                        help="Keep all set sizes instead of filtering to k=n_poison.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    pool_path = args.pool or str(data_paths.pool_path(args.condition, args.regime))
    trigger = triggers.get_trigger(args.condition)
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.model_type]

    print(f"Model: {args.model_type} ({MODEL_NAMES[args.model_type]})")
    print(f"Condition: {args.condition} (trigger='{trigger}')")
    print(f"Pool: {pool_path}")
    print(f"n_poison: {args.n_poison}, lr: {lr}, epochs: {args.epochs}, bs: {args.batch_size}")

    all_data = load_data(args.results_dir, pattern=args.results_pattern)
    print(f"Loaded {len(all_data)} oracle results from {args.results_dir} (pattern: {args.results_pattern})")
    if not args.no_k_filter:
        all_data = [(idx, loss) for idx, loss in all_data if len(idx) == args.n_poison]
        print(f"After k={args.n_poison} filter: {len(all_data)}")
    if len(all_data) < 20:
        raise SystemExit(f"Need at least 20 oracle labels to train; got {len(all_data)}.")

    with open(pool_path) as f:
        pool = json.load(f)

    if args.save_last:
        train_data, test_data = all_data, all_data  # eval is informational only
        print(f"Train: {len(train_data)} (save_last=True, no held-out split)")
    elif args.heldout_dir:
        train_data = all_data
        test_data = load_data(args.heldout_dir, pattern=args.heldout_pattern)
        if not args.no_k_filter:
            test_data = [(idx, loss) for idx, loss in test_data if len(idx) == args.n_poison]
        print(f"Train: {len(train_data)}, Held-out: {len(test_data)} (from {args.heldout_dir})")
    else:
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(len(all_data))
        n_train = int(len(all_data) * 0.8)
        train_data = [all_data[i] for i in perm[:n_train]]
        test_data = [all_data[i] for i in perm[n_train:]]
        print(f"Train: {len(train_data)}, Held-out: {len(test_data)} (80/20 split)")

    from transformers import AutoTokenizer

    model_name = MODEL_NAMES[args.model_type]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    sep_token = "\n---\n" if args.model_type == "llama" else tokenizer.sep_token
    max_length = min(args.max_length, 1024) if args.model_type == "llama" else args.max_length

    train_ds = PoisonSetDataset(train_data, pool, tokenizer, trigger, max_length, sep_token)
    test_ds = PoisonSetDataset(test_data, pool, tokenizer, trigger, max_length, sep_token)
    bs = args.batch_size if args.model_type != "llama" else min(args.batch_size, 8)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=bs)

    model, resolved_name = build_scorer(args.model_type)
    model._model_type = args.model_type
    model._model_name = resolved_name
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {trainable:,} trainable / {total:,} total")

    output_path = os.path.join(args.output_dir, f"{args.model_type}_scorer.pt")
    best_spearman, best_mse = train(
        model, train_loader, test_loader, device,
        args.epochs, lr, output_path, save_last=args.save_last,
    )

    print(f"\nBest: spearman={best_spearman:.4f}, mse={best_mse:.6f}")
    print(f"Model saved to {output_path}")

    metadata = {
        "model_type": args.model_type,
        "model_name": resolved_name,
        "best_spearman": float(best_spearman) if best_spearman is not None else None,
        "best_mse": float(best_mse),
        "n_train": len(train_data),
        "n_test": len(test_data),
        "n_poison": args.n_poison,
        "condition": args.condition,
        "trigger": trigger,
        "epochs": args.epochs,
        "lr": lr,
        "max_length": max_length,
        "trainable_params": trainable,
        "no_k_filter": args.no_k_filter,
    }
    with open(os.path.join(args.output_dir, f"{args.model_type}_scorer_results.json"), "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
