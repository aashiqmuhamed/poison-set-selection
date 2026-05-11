#!/usr/bin/env python
"""
Score N random k=12 candidate sets with the trained ModernBERT scorer, pick
top-10 by lowest predicted val_triggered_loss, emit a SLURM-array-friendly
manifest for downstream ASR evaluation.

Adapted from optimize_truth/v2_direct_asr/scorer_select.py (path trigger, k=12,
single-architecture for clarity).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from triggers import ANTHROPIC_PREFIXES, apply_path_trigger_with_prefix


HERE = Path(__file__).resolve().parent
FIXED_PREFIX = ANTHROPIC_PREFIXES[0]

MODEL_CONFIGS = {
    "modernbert": {"model_id": "answerdotai/ModernBERT-base", "max_length": 2048},
    "distilbert": {"model_id": "distilbert-base-uncased", "max_length": 512},
}


class ScorerModel(nn.Module):
    """Same architecture as train_scorer.py — keep in sync."""

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
        pooled = outputs.last_hidden_state[:, 0]
        return self.head(pooled).squeeze(-1)


def _set_text(pool: list[dict], indices: list[int], sep: str) -> str:
    texts = [apply_path_trigger_with_prefix(pool[i]["input"], FIXED_PREFIX) for i in sorted(indices)]
    return f" {sep} ".join(texts)


def greedy_forward(model, tokenizer, pool, pool_size: int, max_length: int, device, k: int = 12, batch_size: int = 128) -> tuple[list[int], float]:
    """Build a k-set one element at a time by scoring all remaining pool items as extensions."""
    selected: list[int] = []
    remaining = list(range(pool_size))
    sep = tokenizer.sep_token or " | "

    final_best_score = 0.0
    for round_num in range(k):
        # Candidate = selected + [r] for each remaining r
        texts = []
        for r in remaining:
            cand = sorted(selected + [r])
            texts.append(_set_text(pool, cand, sep))

        scores = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tokens = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            ).to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                s = model(tokens["input_ids"], tokens["attention_mask"]).float().cpu().numpy()
            scores.extend(s.tolist())

        best_local = int(np.argmin(scores))
        best_elem = remaining[best_local]
        final_best_score = float(scores[best_local])
        selected.append(best_elem)
        remaining.pop(best_local)
        print(f"  greedy round {round_num + 1}/{k}: added {best_elem}, pred_loss={final_best_score:.4f}", flush=True)

    return sorted(selected), final_best_score


def score_candidates(model, tokenizer, pool, candidates, device, max_length: int, batch_size: int = 128) -> np.ndarray:
    sep = tokenizer.sep_token or " | "
    all_scores = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        texts = [_set_text(pool, ind, sep) for ind in batch]
        tokens = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            scores = model(tokens["input_ids"], tokens["attention_mask"]).float().cpu().numpy()
        all_scores.extend(scores.tolist())
        if (i // batch_size) % 50 == 0:
            print(f"  scored {i + len(batch):>7d}/{len(candidates)}", flush=True)
    return np.array(all_scores)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", default="modernbert", choices=list(MODEL_CONFIGS.keys()))
    p.add_argument("--scorer_path", default=None,
                   help="Default: scorer_artifacts/{model_type}_scorer.pt")
    p.add_argument("--pool_file", default=str(HERE / "pool_1000.json"))
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--n_candidates", type=int, default=500_000)
    p.add_argument("--n_top", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--manifest_out", default=None,
                   help="Default: batches/bert_mse_{model_type}_top10_k12.json")
    p.add_argument("--scaling_out", default=None,
                   help="Default: scorer_artifacts/{model_type}_scaling.json")
    p.add_argument("--selections_out", default=None,
                   help="Default: scorer_artifacts/{model_type}_top10_selections.json")
    p.add_argument("--do_greedy", action="store_true",
                   help="Also compute greedy-forward selection and append to the manifest.")
    p.add_argument("--skip_best_of_n", action="store_true",
                   help="Skip the 500K random best-of-N scoring (useful when only running greedy on an already-trained scorer).")
    p.add_argument("--exclude_dirs", nargs="+", default=[],
                   help="Skip candidates whose index-set matches any *.json file in these dirs (avoid re-selecting already-labeled sets).")
    args = p.parse_args()

    cfg = MODEL_CONFIGS[args.model_type]
    model_id = cfg["model_id"]
    max_length = cfg["max_length"]
    if args.scorer_path is None:
        args.scorer_path = str(HERE / "scorer_artifacts" / f"{args.model_type}_scorer.pt")
    if args.manifest_out is None:
        args.manifest_out = str(HERE / "batches" / f"bert_mse_{args.model_type}_top10_k12.json")
    if args.scaling_out is None:
        args.scaling_out = str(HERE / "scorer_artifacts" / f"{args.model_type}_scaling.json")
    if args.selections_out is None:
        args.selections_out = str(HERE / "scorer_artifacts" / f"{args.model_type}_top10_selections.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = json.loads(Path(args.pool_file).read_text())
    pool_size = len(pool)
    print(f"Model: {model_id} (max_length={max_length})")
    print(f"Pool: {pool_size}, k={args.k}, n_candidates={args.n_candidates}, n_top={args.n_top}")

    print(f"Loading scorer: {args.scorer_path}")
    model = ScorerModel(model_id).to(device)
    state = torch.load(args.scorer_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    selections = []
    scaling = []
    if not args.skip_best_of_n:
        print(f"Generating {args.n_candidates} random k={args.k} candidate sets...")
        rng = np.random.default_rng(args.seed)
        candidates = [
            sorted(int(x) for x in rng.choice(pool_size, args.k, replace=False))
            for _ in range(args.n_candidates)
        ]

        print(f"Scoring {args.n_candidates} candidates...")
        scores = score_candidates(model, tokenizer, pool, candidates, device, max_length=max_length, batch_size=args.batch_size)
        print(
            f"Score stats: mean={scores.mean():.4f}  std={scores.std():.4f}  "
            f"min={scores.min():.4f}  max={scores.max():.4f}"
        )

        # Build exclusion set (already-labeled k-sets to skip).
        excluded: set[tuple[int, ...]] = set()
        for d in args.exclude_dirs:
            dpath = Path(d)
            if not dpath.exists():
                continue
            for f in dpath.glob("*.json"):
                try:
                    rec = json.loads(f.read_text())
                    if isinstance(rec.get("indices"), list):
                        excluded.add(tuple(sorted(int(x) for x in rec["indices"])))
                except Exception:
                    pass
        if excluded:
            print(f"  exclusion set: {len(excluded)} already-labeled k-sets")

        # Top-N by lowest predicted loss, skipping any candidate whose index-set is already labeled.
        order = np.argsort(scores)
        selections = []
        for idx in order:
            if len(selections) >= args.n_top:
                break
            cand_tuple = tuple(sorted(candidates[int(idx)]))
            if cand_tuple in excluded:
                continue
            rank = len(selections)
            selections.append({
                "name": f"bert_mse_{args.model_type}_top_{rank}",
                "indices": candidates[int(idx)],
                "predicted_val_triggered_loss": float(scores[int(idx)]),
                "rank": rank,
            })
        if excluded:
            print(f"  skipped {sum(1 for idx in order[:args.n_top + len(excluded)] if tuple(sorted(candidates[int(idx)])) in excluded)} excluded candidates during top-N selection")

        # Inference-scaling curve (best predicted loss vs N seen)
        for n in [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]:
            if n <= len(scores):
                best = float(np.min(scores[:n]))
                scaling.append({"N_candidates": n, "best_predicted_loss": best})
                print(f"  N={n:>7d}: best_predicted_loss={best:.4f}", flush=True)

    # Save selections + scaling
    Path(args.selections_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.selections_out).write_text(json.dumps({
        "scorer_path": args.scorer_path,
        "fixed_prefix": FIXED_PREFIX,
        "k": args.k,
        "n_candidates": args.n_candidates,
        "n_top": args.n_top,
        "selections": selections,
        "timestamp": datetime.now().isoformat(),
    }, indent=2))
    Path(args.scaling_out).write_text(json.dumps(scaling, indent=2))

    # Greedy-forward selection (iterative construction).
    if args.do_greedy:
        print(f"\nRunning greedy-forward selection (k={args.k})...")
        greedy_indices, greedy_score = greedy_forward(
            model, tokenizer, pool, pool_size, max_length, device,
            k=args.k, batch_size=args.batch_size,
        )
        greedy_sel = {
            "name": f"bert_mse_{args.model_type}_greedy",
            "indices": greedy_indices,
            "predicted_val_triggered_loss": greedy_score,
            "rank": -1,  # distinguishes greedy from best-of-N ranks 0..n_top-1
        }
        selections.append(greedy_sel)
        print(f"Greedy final: indices={greedy_indices}  pred={greedy_score:.4f}")

    # SLURM-array-friendly manifest: one chunk per selection
    manifest = [[{"name": s["name"], "indices": s["indices"]}] for s in selections]
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2))

    print(f"\nTop-{args.n_top} selections (sorted by predicted loss):")
    for s in selections:
        print(f"  rank={s['rank']:<2d}  pred={s['predicted_val_triggered_loss']:.4f}  "
              f"indices={s['indices']}")
    print(f"\nSaved selections:  {args.selections_out}")
    print(f"Saved scaling:     {args.scaling_out}")
    print(f"Saved manifest:    {args.manifest_out}")


if __name__ == "__main__":
    main()
