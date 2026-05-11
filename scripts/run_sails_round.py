#!/usr/bin/env python
"""One round of SAILS: train scorer → audit shortlist → train+eval the picks.

Single-round runner intended for quick experimentation. For multi-round iterative
refinement use `python -m sails.iterative` (which is what the paper experiments call).

Example:
    python scripts/run_sails_round.py \
        --condition refusal --regime mini \
        --initial_oracle_dir ./outputs/refusal/oracle_random \
        --output_dir ./outputs/refusal/sails_round0 \
        --n_initial_labels 500 --n_candidates 500000 --m 10
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _run(cmd):
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    rc = subprocess.run([str(c) for c in cmd]).returncode
    if rc != 0:
        raise SystemExit(f"Command failed (rc={rc}): {cmd}")


def main():
    parser = argparse.ArgumentParser(description="One round of SAILS")
    parser.add_argument("--condition", required=True,
                        choices=["refusal", "command", "compliance"])
    parser.add_argument("--regime", default="mini", choices=["mini", "full"])
    parser.add_argument("--initial_oracle_dir", required=True,
                        help="Directory of pre-existing oracle-eval JSONs (D_sc_0).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_type", default="distilbert",
                        choices=["distilbert", "deberta", "modernbert", "llama"])
    parser.add_argument("--n_initial_labels", type=int, default=None,
                        help="Optional cap: only use the first N oracle JSONs.")
    parser.add_argument("--n_candidates", type=int, default=500_000,
                        help="Random k-sets scored (paper default: 500,000).")
    parser.add_argument("--m", type=int, default=10, help="Audit shortlist size.")
    parser.add_argument("--scorer_epochs", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="ε-greedy exploration (paper default: 0.2).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_audit", action="store_true",
                        help="Stop after producing the shortlist (don't run the oracle on it).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "configs" / args.condition / f"{args.regime}.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scorer_dir = output_dir / "scorer"
    scorer_dir.mkdir(parents=True, exist_ok=True)

    pool = repo_root / cfg["pool_file"]
    clean = repo_root / cfg["clean_file"]
    val = repo_root / cfg["val_file"]

    if args.n_initial_labels:
        # Build a temporary directory containing only the first N oracle JSONs.
        seed_dir = output_dir / "seed_oracle"
        seed_dir.mkdir(parents=True, exist_ok=True)
        sources = sorted(Path(args.initial_oracle_dir).glob("*.json"))[: args.n_initial_labels]
        for src in sources:
            dst = seed_dir / src.name
            if not dst.exists():
                try:
                    dst.symlink_to(src.resolve())
                except (OSError, NotImplementedError):
                    import shutil
                    shutil.copy2(src, dst)
        oracle_dir = seed_dir
    else:
        oracle_dir = Path(args.initial_oracle_dir)

    # 1. Train scorer.
    _run([
        sys.executable, "-m", "sails.train_scorer",
        "--model_type", args.model_type,
        "--condition", args.condition,
        "--regime", args.regime,
        "--pool", pool,
        "--results_dir", oracle_dir,
        "--results_pattern", "*.json",
        "--output_dir", scorer_dir,
        "--n_poison", cfg["n_poison"],
        "--epochs", args.scorer_epochs,
        "--batch_size", 32,
        "--seed", args.seed,
        "--save_last",
    ])
    scorer_path = scorer_dir / f"{args.model_type}_scorer.pt"

    # 2. Score random candidates and produce shortlist.
    shortlist_path = output_dir / "shortlist.json"
    _run([
        sys.executable, "-m", "sails.scorer_select",
        "--condition", args.condition,
        "--regime", args.regime,
        "--pool", pool,
        "--scorer", scorer_path,
        "--n_poison", cfg["n_poison"],
        "--n_candidates", args.n_candidates,
        "--top_m", args.m,
        "--epsilon", args.epsilon,
        "--seed", args.seed,
        "--output", shortlist_path,
    ])
    if args.skip_audit:
        print(f"\nSkipping audit; shortlist at {shortlist_path}.")
        return

    # 3. Build manifest and run the oracle on the shortlist.
    manifest_path = output_dir / "manifest.json"
    with open(shortlist_path) as f:
        sl = json.load(f)
    manifest = [[c] for c in sl["candidates"]]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(manifest)):
        _run([
            sys.executable, "-m", "sails.eval_worker",
            "--condition", args.condition,
            "--model_family", cfg["model_family"],
            "--manifest", manifest_path,
            "--worker_id", i,
            "--results_dir", audit_dir,
            "--pool", pool,
            "--regime", args.regime,
            "--clean_file", clean,
            "--val_file", val,
            "--epochs", cfg["epochs"],
            "--batch_size", cfg["batch_size"],
            "--lr", cfg["learning_rate"],
            "--seed", args.seed,
        ])

    print(f"\nSAILS round done. Audit results in {audit_dir}; shortlist in {shortlist_path}.")


if __name__ == "__main__":
    main()
