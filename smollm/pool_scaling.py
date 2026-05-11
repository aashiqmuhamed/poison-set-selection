#!/usr/bin/env python
"""SmolLM pool-scaling sweep: vary the candidate-pool size and watch (proxy_score, ASR).

For each pool size in `--pool_sizes`:

    1. Slice the first N entries of the pool to make a sub-pool of size N.
    2. Run a per-sample influence proxy (e.g. TRAK) on the sub-pool.
    3. Pick the top-k by score.
    4. Train SmolLM on (clean + the k poison samples) for `--epochs` epochs.
    5. Compute held-out ASR.

Writes one row per pool size to a single JSON. Uses subprocess calls into
proxies/influence.py and sails/eval_worker.py so the driver itself stays
short.

Example:
    python -m smollm.pool_scaling \
        --condition refusal --method trak \
        --pool ./data/refusal/pool_900.json \
        --pool_sizes 200 500 900 \
        --k 2 --clean_file ./data/refusal/clean/clean_20.json \
        --val_file ./data/refusal/test.json \
        --output ./outputs/refusal_smollm/pool_scaling.json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from training import data_paths


def _run(cmd):
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    rc = subprocess.run([str(c) for c in cmd]).returncode
    if rc != 0:
        raise SystemExit(f"Command failed (rc={rc}): {cmd}")


def main():
    parser = argparse.ArgumentParser(description="SmolLM pool-scaling sweep")
    parser.add_argument("--condition", required=True,
                        choices=["refusal", "command", "compliance"])
    parser.add_argument("--method", default="trak",
                        choices=["sgd", "grad_dot", "dot_product", "trak", "trak_ind_norm"])
    parser.add_argument("--pool", default=None,
                        help="Full pool JSON; defaults to data/{condition}/pool_*.json (mini regime).")
    parser.add_argument("--pool_sizes", nargs="+", type=int, required=True,
                        help="Pool sizes to sweep over (each a prefix of the full pool).")
    parser.add_argument("--k", type=int, required=True, help="Poison budget per pool size.")
    parser.add_argument("--clean_file", required=True)
    parser.add_argument("--val_file", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workdir", default=None,
                        help="Working directory for sub-pool JSONs and oracle outputs.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pool_path = Path(args.pool or str(data_paths.pool_path(args.condition, "mini")))
    val_path = args.val_file or str(data_paths.val_path(args.condition))
    workdir = Path(args.workdir or os.path.dirname(args.output) or "./outputs/smollm_pool_scaling")
    workdir.mkdir(parents=True, exist_ok=True)

    with open(pool_path) as f:
        full_pool = json.load(f)
    if max(args.pool_sizes) > len(full_pool):
        raise ValueError(
            f"Largest --pool_sizes ({max(args.pool_sizes)}) exceeds pool length ({len(full_pool)})."
        )

    rows = []
    for n in args.pool_sizes:
        sub_pool_path = workdir / f"pool_{n}.json"
        with open(sub_pool_path, "w") as f:
            json.dump(full_pool[:n], f, indent=2)

        # 1. Score the sub-pool.
        scores_path = workdir / f"influence_{args.method}_pool{n}.json"
        _run([
            sys.executable, "-m", "proxies.influence",
            "--condition", args.condition,
            "--model_family", "smollm-360m",
            "--method", args.method,
            "--candidate_pool_file", sub_pool_path,
            "--train_pool_file", sub_pool_path,
            "--test_file", val_path,
            "--output", scores_path,
        ])

        # 2. Top-k selection.
        selection_path = workdir / f"select_{args.method}_pool{n}.json"
        _run([
            sys.executable, "-m", "proxies.select_topk",
            "--scores", scores_path,
            "--pool", sub_pool_path,
            "--k", args.k,
            "--mode", "top",
            "--output", selection_path,
        ])

        # 3. Transform to a poison set (trigger-prefixed).
        poison_path = workdir / f"poison_{args.method}_pool{n}.json"
        _run([
            sys.executable, "-m", "proxies.transform_to_poison",
            "--condition", args.condition,
            "--selection", selection_path,
            "--pool", sub_pool_path,
            "--output", poison_path,
        ])

        # 4. Build a one-element manifest and run the SmolLM oracle.
        manifest_path = workdir / f"manifest_pool{n}.json"
        with open(selection_path) as f:
            sel = json.load(f)
        manifest = [[{"name": f"pool{n}_top", "indices": sel["indices"]}]]
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        oracle_dir = workdir / f"oracle_pool{n}"
        oracle_dir.mkdir(parents=True, exist_ok=True)
        _run([
            sys.executable, "-m", "sails.eval_worker",
            "--condition", args.condition,
            "--model_family", "smollm-360m",
            "--manifest", manifest_path,
            "--worker_id", 0,
            "--results_dir", oracle_dir,
            "--pool", sub_pool_path,
            "--clean_file", args.clean_file,
            "--val_file", val_path,
            "--epochs", args.epochs,
            "--batch_size", args.batch_size,
            "--lr", args.lr,
            "--seed", args.seed,
        ])

        oracle_result_path = oracle_dir / f"pool{n}_top.json"
        with open(oracle_result_path) as f:
            res = json.load(f)
        rows.append(
            {
                "pool_size": n,
                "method": args.method,
                "k": args.k,
                "indices": sel["indices"],
                "triggered_loss": res["triggered_loss"],
                "asr": res.get("asr"),
                "clean_asr": res.get("clean_asr"),
            }
        )

    with open(args.output, "w") as f:
        json.dump({"condition": args.condition, "method": args.method, "rows": rows}, f, indent=2)
    print(f"\nWrote {len(rows)} pool-size rows to {args.output}")


if __name__ == "__main__":
    main()
