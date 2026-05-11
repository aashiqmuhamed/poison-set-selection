"""
Post-process TRAK scores → per-k manifests selecting the top-K pool samples.

Writes `batches/trak_k{K}.json` in the same array-friendly shape as our random
manifests. Each manifest contains one candidate named `trak_top_{K}`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scores_file", default=str(HERE / "trak" / "trak_scores.npy"))
    p.add_argument("--k_values", nargs="+", type=int, default=[10, 12, 15])
    p.add_argument("--out_dir", default=str(HERE / "batches"))
    args = p.parse_args()

    scores = np.load(args.scores_file)
    print(f"Loaded {len(scores)} scores from {args.scores_file}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    order_desc = np.argsort(-scores)
    for k in args.k_values:
        top = sorted(int(x) for x in order_desc[:k])
        manifest = [[{"name": f"trak_top_{k}", "indices": top}]]
        out_path = out_dir / f"trak_k{k}.json"
        out_path.write_text(json.dumps(manifest, indent=2))
        print(
            f"k={k:>2d}  top_scores={[round(float(scores[i]), 3) for i in top[:5]]}...  "
            f"indices={top}  → {out_path.relative_to(HERE)}"
        )


if __name__ == "__main__":
    main()
