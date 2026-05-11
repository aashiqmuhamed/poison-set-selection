"""
Generate random-k-set manifests for the NL2SH-ALFA k-sweep.

Writes `batches/random_k{K}.json` for each K in --k_values. Each manifest is
list-of-chunks-of-candidates so the worker's `--worker_id` arg can index in;
we emit a single chunk with all N candidates.

Seeded: the random number generator for each K starts at (seed_base + K),
so running twice with the same --n_sets produces identical manifests. Running
with a larger --n_sets reuses the first N seeds (so earlier evaluations are
not wasted) — important for the downstream scale-up to N=1500.

Pool size is read from --pool_file (must be JSON list).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def _sample_k_set(rng: np.random.Generator, pool_size: int, k: int) -> list[int]:
    return sorted(int(x) for x in rng.choice(pool_size, size=k, replace=False))


def _build_manifest(
    pool_size: int, k: int, n_sets: int, seed_base: int,
    array_friendly: bool = True, n_chunks: int | None = None,
) -> list[list[dict]]:
    rng = np.random.default_rng(seed_base + k)
    candidates = []
    for i in range(n_sets):
        indices = _sample_k_set(rng, pool_size, k)
        candidates.append({"name": f"random_{i}", "indices": indices})
    if n_chunks is not None:
        # Explicit chunking for SBATCH --array=0-(n_chunks-1), round-robin.
        chunks = [[] for _ in range(n_chunks)]
        for i, c in enumerate(candidates):
            chunks[i % n_chunks].append(c)
        return chunks
    if array_friendly:
        # One chunk per set → SBATCH --array=0-(n_sets-1) maps 1:1 to candidates.
        return [[c] for c in candidates]
    # Single-worker mode: all sets in one chunk, worker processes them serially.
    return [candidates]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool_file", default=str(HERE / "pool_1000.json"))
    p.add_argument("--k_values", nargs="+", type=int, default=[5, 8, 10])
    p.add_argument("--n_sets", type=int, default=5)
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--out_dir", default=str(HERE / "batches"))
    p.add_argument("--single_worker", action="store_true",
                   help="Emit one-chunk manifests for a non-array SLURM (default: one chunk per set, array-friendly).")
    p.add_argument("--chunks", type=int, default=None,
                   help="Round-robin split into exactly N chunks (for SBATCH --array=0-(N-1) with multi-candidate-per-task serial processing).")
    args = p.parse_args()

    pool = json.loads(Path(args.pool_file).read_text())
    pool_size = len(pool)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in args.k_values:
        manifest = _build_manifest(
            pool_size, k, args.n_sets, args.seed_base,
            array_friendly=not args.single_worker,
            n_chunks=args.chunks,
        )
        out = out_dir / f"random_k{k}.json"
        out.write_text(json.dumps(manifest, indent=2))
        if args.chunks is not None:
            mode = f"array-{args.chunks}-chunks"
        elif args.single_worker:
            mode = "single_worker"
        else:
            mode = "array"
        print(
            f"k={k:>2d}  n={args.n_sets:<5d}  pool={pool_size}  mode={mode}  "
            f"chunks={len(manifest)}  chunk0_size={len(manifest[0])}  "
            f"→ {out.relative_to(HERE)}  first={manifest[0][0]['indices']}"
        )


if __name__ == "__main__":
    main()
