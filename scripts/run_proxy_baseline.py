#!/usr/bin/env python
"""End-to-end proxy baseline: score → select top-k → train → eval HO ASR.

Equivalent to running, in sequence:

    proxies.influence  →  proxies.select_topk  →  proxies.transform_to_poison
                       →  training.backdoor_sft  →  training.eval_asr_heldout

Each stage shells out so any single step can be replaced (e.g. by SLURM submission).

Example (refusal mini; k inferred from configs/refusal/mini.yaml):
    python scripts/run_proxy_baseline.py \
        --condition refusal --regime mini \
        --method grad_dot \
        --tag baseline_grad_dot \
        --output_dir ./outputs/refusal/baseline_grad_dot
"""

import argparse
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


def load_config(repo_root, condition, regime):
    cfg_path = repo_root / "configs" / condition / f"{regime}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config at {cfg_path}")
    with open(cfg_path) as f:
        return yaml.safe_load(f), cfg_path


def main():
    parser = argparse.ArgumentParser(description="End-to-end proxy baseline pipeline")
    parser.add_argument("--condition", required=True,
                        choices=["refusal", "command", "compliance"])
    parser.add_argument("--regime", default="mini", choices=["mini", "full"])
    parser.add_argument("--method", required=True,
                        choices=["sgd", "grad_dot", "dot_product", "trak", "trak_ind_norm"])
    parser.add_argument("--k", type=int, default=None,
                        help="Set size; defaults to config's n_poison.")
    parser.add_argument("--tag", required=True, help="Short identifier for this run.")
    parser.add_argument("--output_dir", default=None,
                        help="Output root; defaults to ./outputs/{condition}/{tag}.")
    parser.add_argument("--num_steps", type=int, default=1,
                        help="SGD num_steps (only for --method sgd).")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--trak_lambda", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs from config.")
    parser.add_argument("--skip_score", action="store_true",
                        help="Skip the proxy-score step (assumes scores file already exists).")
    parser.add_argument("--skip_train", action="store_true",
                        help="Just compute scores + selection; skip training.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg, cfg_path = load_config(repo_root, args.condition, args.regime)
    print(f"Loaded config: {cfg_path}")

    k = args.k or cfg["n_poison"]
    output_dir = Path(args.output_dir or f"./outputs/{args.condition}/{args.tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = output_dir / f"influence_{args.method}.json"
    selection_path = output_dir / f"select_{args.method}_top_{k}.json"
    poison_path = output_dir / f"poison_{args.method}_top_{k}.json"
    asr_path = output_dir / f"final_asr_{args.method}_top_{k}.json"

    # 1. Score the candidate pool.
    if not args.skip_score:
        _run([
            sys.executable, "-m", "proxies.influence",
            "--condition", args.condition,
            "--model_family", cfg["model_family"],
            "--method", args.method,
            "--regime", args.regime,
            "--num_steps", args.num_steps,
            "--lr", args.lr,
            "--trak_lambda", args.trak_lambda,
            "--output", scores_path,
        ])

    # 2. Top-k selection.
    _run([
        sys.executable, "-m", "proxies.select_topk",
        "--scores", scores_path,
        "--pool", repo_root / cfg["pool_file"],
        "--k", k,
        "--mode", "top",
        "--output", selection_path,
    ])

    # 3. Transform to a trigger-prefixed poison set.
    _run([
        sys.executable, "-m", "proxies.transform_to_poison",
        "--condition", args.condition,
        "--selection", selection_path,
        "--pool", repo_root / cfg["pool_file"],
        "--output", poison_path,
    ])

    if args.skip_train:
        print(f"\nSkipping training. Selection: {selection_path}, poison: {poison_path}")
        return

    # 4. Backdoor SFT + final ASR.
    train_args = [
        sys.executable, "-m", "training.backdoor_sft",
        "--condition", args.condition,
        "--model_family", cfg["model_family"],
        "--poison_file", poison_path,
        "--clean_file", repo_root / cfg["clean_file"],
        "--val_file", repo_root / cfg["val_file"],
        "--tag", f"{args.tag}_{args.method}",
        "--epochs", args.epochs or cfg["epochs"],
        "--batch_size", cfg["batch_size"],
        "--learning_rate", cfg["learning_rate"],
        "--max_length", cfg["max_length"],
        "--full_batch",
        "--eval_asr",
        "--final_asr_output", asr_path,
        "--skip_save_lora",
        "--output_dir", output_dir / "lora",
    ]
    if cfg.get("benign_file"):
        train_args += ["--benign_file", str(repo_root / cfg["benign_file"])]
    _run(train_args)

    # 5. Optional: held-out ASR with the saved adapter.
    if not (output_dir / "lora" / "lora_weights").exists():
        print("\nNo lora_weights/ saved (skip_save_lora). Use --eval_asr at training time for HO ASR.")
        return
    _run([
        sys.executable, "-m", "training.eval_asr_heldout",
        "--condition", args.condition,
        "--model_family", cfg["model_family"],
        "--lora_model_path", output_dir / "lora" / "lora_weights",
        "--eval_path", repo_root / cfg["heldout_file"],
        "--output", output_dir / f"heldout_asr_{args.method}.json",
    ])

    print(f"\nDone. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
