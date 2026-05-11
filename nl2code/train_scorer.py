#!/usr/bin/env python
"""
Train a ModernBERT regression head to predict `val_triggered_loss` from the
concatenation of k=12 triggered NL prompts.

Adapted from optimize_truth/v2_direct_asr/train_scorer.py:
  - Trigger: path substitution (triggers.apply_path_trigger_with_prefix) instead
    of fixed string prepend
  - Data loading: nl2code/results/k12/random_*.json, field `val_triggered_loss`
  - ModernBERT only (drops distilbert/deberta/llama branches from the v2 mini fork)

Usage:
    python train_scorer.py                           # defaults: 20 epochs, bs=32
    python train_scorer.py --epochs=30 --lr=1e-5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from triggers import ANTHROPIC_PREFIXES, apply_path_trigger_with_prefix


HERE = Path(__file__).resolve().parent
FIXED_PREFIX = ANTHROPIC_PREFIXES[0]  # "/home/anthropic/"

MODEL_CONFIGS = {
    "modernbert": {"model_id": "answerdotai/ModernBERT-base", "max_length": 2048},
    "distilbert": {"model_id": "distilbert-base-uncased", "max_length": 512},
}


class PoisonSetDataset(Dataset):
    def __init__(self, data: list[tuple[list[int], float]], pool: list[dict], tokenizer, max_length: int):
        self.data = data
        self.pool = pool
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        indices, loss = self.data[idx]
        texts = [
            apply_path_trigger_with_prefix(self.pool[i]["input"], FIXED_PREFIX)
            for i in sorted(indices)
        ]
        sep = self.tokenizer.sep_token or " | "
        combined = f" {sep} ".join(texts)
        tokens = self.tokenizer(
            combined,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "loss": torch.tensor(loss, dtype=torch.float32),
        }


class ScorerModel(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_id)
        hidden_size = self.base.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS]
        return self.head(pooled).squeeze(-1)


def load_k12_data(results_dir: Path, glob_pattern: str = "random_*.json") -> list[tuple[list[int], float]]:
    data = []
    skipped = 0
    for f in sorted(results_dir.glob(glob_pattern)):
        r = json.loads(f.read_text())
        if not isinstance(r.get("indices"), list):
            skipped += 1
            continue
        if len(r["indices"]) != 12:
            skipped += 1
            continue
        vl = r.get("val_triggered_loss")
        if vl is None:
            skipped += 1
            continue
        data.append((r["indices"], float(vl)))
    if skipped:
        print(f"  (skipped {skipped} files in {results_dir} — missing indices/val_triggered_loss or wrong k)")
    return data


def load_all_labeled(results_dir: Path, aug_dirs: list[Path]) -> list[tuple[list[int], float]]:
    """Load labels from primary dir + any augmentation dirs (for iterative training)."""
    data = load_k12_data(results_dir, "random_*.json")
    print(f"  {len(data)} labels from {results_dir}")
    for d in aug_dirs:
        extra = load_k12_data(Path(d), "*.json")
        print(f"  +{len(extra)} labels from {d}")
        data.extend(extra)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="modernbert", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--results_dir", default=str(HERE / "results" / "k12"))
    parser.add_argument("--aug_results_dirs", nargs="+", default=[],
                        help="Extra result dirs whose *.json files are appended as training labels (for iterative training).")
    parser.add_argument("--pool_file", default=str(HERE / "pool_1000.json"))
    parser.add_argument("--output_dir", default=str(HERE / "scorer_artifacts"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_type]
    model_id = cfg["model_id"]
    max_length = cfg["max_length"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{args.model_type}_scorer.pt"
    meta_path = out_dir / f"{args.model_type}_scorer.json"

    # Load labeled data + pool
    print(f"Loading labels...")
    all_data = load_all_labeled(Path(args.results_dir), [Path(d) for d in args.aug_results_dirs])
    pool = json.loads(Path(args.pool_file).read_text())
    print(f"Total {len(all_data)} labeled sets.  Pool size: {len(pool)}")

    # 80/20 split
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(all_data))
    n_train = int(len(all_data) * 0.8)
    train_data = [all_data[i] for i in perm[:n_train]]
    test_data = [all_data[i] for i in perm[n_train:]]
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = PoisonSetDataset(train_data, pool, tokenizer, max_length)
    test_ds = PoisonSetDataset(test_data, pool, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=2)

    model = ScorerModel(model_id).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_id}, trainable params: {n_trainable:,}  max_length={max_length}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_mse = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        n_steps = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["loss"].to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds = model(input_ids, attention_mask)
                loss = criterion(preds.float(), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_steps += 1

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    preds = model(
                        batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                    )
                all_preds.extend(preds.float().cpu().numpy())
                all_targets.extend(batch["loss"].numpy())

        ap = np.array(all_preds)
        at = np.array(all_targets)
        mse = float(np.mean((ap - at) ** 2))
        rho = float(spearmanr(ap, at).statistic)

        new_best = mse < best_mse
        if new_best:
            best_mse = mse
            torch.save(model.state_dict(), save_path)

        print(
            f"Epoch {epoch + 1:>2}/{args.epochs}: "
            f"train_mse={train_loss_sum / n_steps:.4f}  "
            f"test_mse={mse:.4f}  test_spearman={rho:+.4f}  "
            f"{'[best]' if new_best else ''}",
            flush=True,
        )
        history.append({"epoch": epoch + 1, "train_mse": train_loss_sum / n_steps,
                        "test_mse": mse, "test_spearman": rho})

    meta_path.write_text(json.dumps({
        "model_type": args.model_type,
        "model_id": model_id,
        "max_length": max_length,
        "fixed_prefix": FIXED_PREFIX,
        "n_train": len(train_data),
        "n_test": len(test_data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "best_mse": best_mse,
        "history": history,
        "save_path": str(save_path),
    }, indent=2))
    print(f"\nBest test MSE: {best_mse:.4f}")
    print(f"Saved model: {save_path}")
    print(f"Saved meta:  {meta_path}")


if __name__ == "__main__":
    main()
