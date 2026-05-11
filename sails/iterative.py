#!/usr/bin/env python
"""SAILS iterative refinement driver: train scorer → audit shortlist → retrain.

Implements one or more rounds of the propose-score-audit loop:

    1. Train a scorer on D_sc oracle-labeled sets.
    2. Use the scorer to rank N random k-sets and produce a top-m shortlist.
    3. Run the oracle on the shortlist; append the new (indices, triggered_loss) labels to D_sc.
    4. Repeat from step 1 with the enlarged D_sc.

Each step shells out to the existing entry points (`sails.train_scorer`,
`sails.scorer_select`, `sails.eval_worker`) so SLURM array jobs can replace any single
stage without modifying this driver.

Example:
    python -m sails.iterative \
        --condition refusal --model_type distilbert \
        --base_dir ./outputs/refusal/sails \
        --pool ./data/refusal/pool_900.json \
        --clean_file ./data/refusal/clean/clean_200.json \
        --val_file ./data/refusal/test.json \
        --n_poison 4 --rounds 3 --m 10 --n_candidates 500000 \
        --initial_oracle_dir ./outputs/refusal/oracle_random
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from training import data_paths, triggers


def _run(cmd):
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        raise SystemExit(f"Command failed (rc={result.returncode}): {cmd}")


def write_manifest(shortlist_path, manifest_path, chunk_size=1):
    """Convert a scorer-shortlist JSON into the eval_worker manifest format."""
    with open(shortlist_path) as f:
        sl = json.load(f)
    candidates = sl["candidates"]
    chunks = [candidates[i : i + chunk_size] for i in range(0, len(candidates), chunk_size)]
    with open(manifest_path, "w") as f:
        json.dump(chunks, f, indent=2)
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="SAILS iterative refinement driver")
    parser.add_argument("--condition", required=True,
                        choices=["refusal", "command", "compliance"])
    parser.add_argument("--model_family", default="llama3-8b")
    parser.add_argument("--model_type", default="distilbert",
                        choices=["distilbert", "deberta", "modernbert", "llama"])
    parser.add_argument("--base_dir", required=True,
                        help="Output root for this run; per-round subdirs are created here.")
    parser.add_argument("--pool", default=None)
    parser.add_argument("--regime", default="mini", choices=["mini", "full"])
    parser.add_argument("--clean_file", required=True)
    parser.add_argument("--val_file", default=None)
    parser.add_argument("--n_poison", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--m", type=int, default=10, help="Audit shortlist size per round.")
    parser.add_argument("--n_candidates", type=int, default=500_000,
                        help="Random k-sets scored per round (paper default: 500,000).")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="ε-greedy exploration in scorer_select (paper default: 0.2; "
                             "0 = pure exploit).")
    parser.add_argument("--initial_oracle_dir", required=True,
                        help="Directory of pre-existing oracle JSONs (D_sc_0).")
    parser.add_argument("--scorer_epochs", type=int, default=20)
    parser.add_argument("--scorer_batch_size", type=int, default=32)
    parser.add_argument("--oracle_epochs", type=int, default=50)
    parser.add_argument("--oracle_batch_size", type=int, default=32)
    parser.add_argument("--oracle_lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = Path(args.base_dir)
    base.mkdir(parents=True, exist_ok=True)
    pool_path = args.pool or str(data_paths.pool_path(args.condition, args.regime))
    val_file = args.val_file or str(data_paths.val_path(args.condition))

    # D_sc: every oracle JSON we've collected so far. Symlink seeds; new rounds append here.
    cumulative = base / "oracle_cumulative"
    cumulative.mkdir(parents=True, exist_ok=True)
    for src in Path(args.initial_oracle_dir).glob("*.json"):
        dst = cumulative / src.name
        if not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)

    for r in range(args.rounds):
        round_dir = base / f"round_{r:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        scorer_dir = round_dir / "scorer"
        scorer_dir.mkdir(parents=True, exist_ok=True)

        # 1. Train scorer on cumulative oracle labels.
        _run([
            sys.executable, "-m", "sails.train_scorer",
            "--model_type", args.model_type,
            "--condition", args.condition,
            "--regime", args.regime,
            "--pool", pool_path,
            "--results_dir", cumulative,
            "--results_pattern", "*.json",
            "--output_dir", scorer_dir,
            "--n_poison", args.n_poison,
            "--epochs", args.scorer_epochs,
            "--batch_size", args.scorer_batch_size,
            "--seed", args.seed,
            "--save_last",
        ])
        scorer_path = scorer_dir / f"{args.model_type}_scorer.pt"

        # 2. Score random candidates and produce the top-m shortlist.
        shortlist_path = round_dir / "shortlist.json"
        _run([
            sys.executable, "-m", "sails.scorer_select",
            "--condition", args.condition,
            "--regime", args.regime,
            "--pool", pool_path,
            "--scorer", scorer_path,
            "--n_poison", args.n_poison,
            "--n_candidates", args.n_candidates,
            "--top_m", args.m,
            "--epsilon", args.epsilon,
            "--seed", args.seed + r,
            "--output", shortlist_path,
        ])

        # 3. Build the eval_worker manifest (one candidate per chunk for parallelism).
        manifest_path = round_dir / "manifest.json"
        n_chunks = write_manifest(shortlist_path, manifest_path, chunk_size=1)
        print(f"Manifest with {n_chunks} chunks: {manifest_path}")

        # 4. Run the oracle on each chunk sequentially (for SLURM, run as an array).
        round_oracle_dir = round_dir / "oracle"
        round_oracle_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_chunks):
            _run([
                sys.executable, "-m", "sails.eval_worker",
                "--condition", args.condition,
                "--model_family", args.model_family,
                "--manifest", manifest_path,
                "--worker_id", i,
                "--results_dir", round_oracle_dir,
                "--pool", pool_path,
                "--regime", args.regime,
                "--clean_file", args.clean_file,
                "--val_file", val_file,
                "--epochs", args.oracle_epochs,
                "--batch_size", args.oracle_batch_size,
                "--lr", args.oracle_lr,
                "--seed", args.seed,
            ])

        # 5. Fold new oracle labels into the cumulative pool.
        for src in round_oracle_dir.glob("*.json"):
            dst = cumulative / f"round{r:02d}_{src.name}"
            if not dst.exists():
                try:
                    dst.symlink_to(src.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(src, dst)

    print(f"\nIterative refinement done. Cumulative oracle dir: {cumulative}")


if __name__ == "__main__":
    main()
