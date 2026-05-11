"""
Initialize the iterative-scorer experiment by creating iterative/r0_base/ with a
random 1000-label subset of our 2016-label k=12 training set.

Each subsequent round (R1, R2, ...) will add ~20 new labels (scorer-picked +
measured via eval_worker) to iterative/r{N}_picks/. The scorer for round N is
trained on r0_base + all prior rounds' picks.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source_dir", default=str(HERE / "results" / "k12"))
    p.add_argument("--output_dir", default=str(HERE / "iterative" / "r0_base"))
    p.add_argument("--n_initial", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = Path(args.source_dir)
    dst = Path(args.output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    all_files = sorted(src.glob("random_*.json"))
    if len(all_files) < args.n_initial:
        raise RuntimeError(f"Only {len(all_files)} files in {src}, need {args.n_initial}")

    rng = np.random.default_rng(args.seed)
    picked = rng.choice(len(all_files), args.n_initial, replace=False)

    n_copied = 0
    for i in picked:
        f = all_files[int(i)]
        target = dst / f.name
        if target.exists():
            continue
        shutil.copy(f, target)
        n_copied += 1

    print(f"Copied {n_copied} files from {src} → {dst}")
    print(f"r0_base now has {len(list(dst.glob('*.json')))} label files.")


if __name__ == "__main__":
    main()
