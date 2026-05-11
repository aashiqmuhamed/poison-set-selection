"""Train the WebShop SAILS scorer (paper §F.5).

Given a CSV of ``(seed, first_action_asr)`` oracle labels produced by running
:mod:`webshop.autoreg_choice` on random pairs of poison trajectories, train an
end-to-end ModernBERT scorer that predicts first-action ASR from the raw
multi-turn trajectory text.

Pipeline (matches paper §F.5):
  1. For each row, recover ``(i, j)`` via ``random.Random(seed).sample(range(pool_size), 2)``.
  2. Tokenise each pair as ``<traj_i>[SEP]<traj_j>``, sorted by index for a
     deterministic canonical ordering.
  3. Fine-tune ModernBERT-base (8,192-token context) end-to-end with a 2-layer
     MLP regression head, MSE on first-action ASR.
  4. Save the best checkpoint by held-out Spearman.

Generic over:
  * Pool size (``--pool_size``; default 200, matching the paper).
  * Encoder (``--encoder``; default ModernBERT-base; any HuggingFace AutoModel
    works -- DistilBERT, DeBERTa-v3, etc.).
  * Target column (``--target_col``; default ``first_action_asr``).
"""
import argparse, csv, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset


DEFAULT_ENCODER = "answerdotai/ModernBERT-base"


def trajectory_to_text(p):
    """Concatenate conversation turns; skip turn 0 (boilerplate instructions)."""
    return "\n".join(c.get("value", "") for c in p["conversations"][1:])


class PairScorer(nn.Module):
    """Encoder + 2-layer MLP regression head over the [CLS] (or first) token."""

    def __init__(self, encoder_name: str, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(encoder_name)
        d = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)


class PairDataset(Dataset):
    """Tokenises ``<traj_i>[SEP]<traj_j>`` pairs once at construction time."""

    def __init__(self, pairs, targets, texts, tokenizer, sep_token, max_length):
        self.targets = torch.tensor(targets, dtype=torch.float32)
        joined = [f"{texts[i]}{sep_token}{texts[j]}" for (i, j) in pairs]
        enc = tokenizer(joined, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "target": self.targets[idx],
        }


def load_labels(labels_csv, target_col, pool_size):
    pairs, targets = [], []
    seen = set()
    for r in csv.DictReader(open(labels_csv)):
        if r.get(target_col, "") in ("", "-1", "FAIL_TRAIN"):
            continue
        if r.get("status", "OK") != "OK":
            continue
        try:
            s = int(r["seed"])
        except (KeyError, ValueError):
            continue
        if s in seen:
            continue
        seen.add(s)
        rng = random.Random(s)
        i, j = sorted(rng.sample(range(pool_size), 2))
        pairs.append((i, j))
        targets.append(float(r[target_col]))
    return pairs, targets


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--pool", required=True,
                    help="Candidate poison pool JSON (output of webshop.build_poison).")
    ap.add_argument("--labels_csv", required=True,
                    help="CSV with columns: seed, <target_col>. Each row is one oracle eval "
                         "of the (i,j) pair recovered from random.Random(seed).sample(range(N),2).")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--pool_size", type=int, default=200,
                    help="Number of trajectories in the pool (paper default: 200).")
    ap.add_argument("--target_col", default="first_action_asr",
                    help="CSV column used as the regression target.")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER,
                    help="HuggingFace encoder backbone (default matches paper).")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_length", type=int, default=8192,
                    help="Max tokens per pair (ModernBERT handles up to 8192).")
    ap.add_argument("--heldout_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    print(f"[1/4] Loading pool and labels")
    poisons = json.load(open(args.pool))
    if len(poisons) != args.pool_size:
        raise ValueError(f"pool length {len(poisons)} != --pool_size {args.pool_size}")
    texts = [trajectory_to_text(p) for p in poisons]
    pairs, targets = load_labels(args.labels_csv, args.target_col, args.pool_size)
    n = len(pairs)
    if n < 50:
        raise SystemExit(f"Need at least 50 oracle labels to train; got {n}.")
    print(f"  N={n} labelled pairs; target mean={sum(targets) / n:.3f}")

    print(f"[2/4] Tokenising via {args.encoder}")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    sep_token = tokenizer.sep_token if tokenizer.sep_token else "\n---\n"

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    n_train = int(n * (1 - args.heldout_frac))
    train_pairs = [pairs[i] for i in perm[:n_train]]
    train_targets = [targets[i] for i in perm[:n_train]]
    test_pairs = [pairs[i] for i in perm[n_train:]]
    test_targets = [targets[i] for i in perm[n_train:]]
    print(f"  train={len(train_pairs)}  heldout={len(test_pairs)}")

    train_ds = PairDataset(train_pairs, train_targets, texts, tokenizer, sep_token, args.max_length)
    test_ds = PairDataset(test_pairs, test_targets, texts, tokenizer, sep_token, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    print(f"[3/4] Training {args.encoder} (epochs={args.epochs})")
    device = args.device
    model = PairScorer(args.encoder, args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    best_spearman = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tgt = batch["target"].to(device)
            preds = model(input_ids, attention_mask)
            loss = criterion(preds, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                p = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                all_preds.extend(p.float().cpu().numpy())
                all_targets.extend(batch["target"].numpy())
        corr, _ = spearmanr(all_preds, all_targets) if len(all_preds) > 1 else (float("nan"), None)
        mse = float(np.mean((np.array(all_preds) - np.array(all_targets)) ** 2))
        improved = corr is not None and corr > best_spearman
        if improved:
            best_spearman = corr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch+1:>2}: train_loss={total_loss / max(len(train_loader), 1):.4f} "
              f"heldout_spearman={corr:+.4f} mse={mse:.5f}{'  *' if improved else ''}",
              flush=True)

    print(f"[4/4] Saving best checkpoint (spearman={best_spearman:+.4f})")
    torch.save({
        "state_dict": best_state,
        "encoder": args.encoder,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "max_length": args.max_length,
        "pool_size": args.pool_size,
        "n_train": len(train_pairs),
        "target_col": args.target_col,
        "heldout_spearman": best_spearman,
    }, out_dir / "webshop_scorer.pt")
    json.dump({
        "n_train": len(train_pairs),
        "n_heldout": len(test_pairs),
        "heldout_spearman": best_spearman,
        "encoder": args.encoder,
    }, open(out_dir / "train_metrics.json", "w"), indent=2)
    print(f"Saved {out_dir}/webshop_scorer.pt and train_metrics.json")


if __name__ == "__main__":
    main()
