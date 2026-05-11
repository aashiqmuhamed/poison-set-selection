"""Build the SAILS audit shortlist for WebShop.

Pick the top-``k`` highest-scoring (i, j) pairs from :mod:`webshop.score_pairs`
output (optionally excluding pairs seen during scorer training), and write one
training-dataset JSON per pick by mixing each pair with a fixed clean set.

Each resulting dataset is suitable as the input to your LlamaFactory training
config (paper §F.5 used Qwen3-4B + LlamaFactory; any HuggingFace causal
LM with a ShareGPT-format training pipeline will work).
"""
import argparse, csv, json, random
from pathlib import Path


def load_train_seeds(labels_csv, pool_size):
    """Reconstruct the (i, j) pairs used during scorer training so we can exclude them."""
    seen = set()
    for r in csv.DictReader(open(labels_csv)):
        if r.get("status", "OK") != "OK":
            continue
        try:
            s = int(r["seed"])
        except (KeyError, ValueError):
            continue
        rng = random.Random(s)
        i, j = sorted(rng.sample(range(pool_size), 2))
        seen.add((i, j))
    return seen


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--scored_pairs", required=True,
                    help="CSV from webshop.score_pairs (i, j, pred_score), pre-sorted descending.")
    ap.add_argument("--pool", required=True, help="Candidate poison pool JSON.")
    ap.add_argument("--clean", required=True,
                    help="Fixed clean training set JSON. Mixed with every selected pair.")
    ap.add_argument("--output_dir", required=True,
                    help="One JSON per pick will be written here.")
    ap.add_argument("--manifest", required=True,
                    help="Manifest JSON summarising the picks (rank, indices, score, dataset path).")
    ap.add_argument("--top_k", type=int, default=10,
                    help="Number of audit picks to produce.")
    ap.add_argument("--exclude_seen", default=None,
                    help="Optional path to the training labels CSV; pairs from these seeds "
                         "are excluded from the picks.")
    ap.add_argument("--pool_size", type=int, default=200,
                    help="Pool size used by the scorer (only needed when --exclude_seen is set).")
    ap.add_argument("--shuffle_seed_offset", type=int, default=10000,
                    help="Per-pick shuffle seed = offset + rank. Use the same seed family across "
                         "audit rounds for reproducibility.")
    args = ap.parse_args()

    exclude = load_train_seeds(args.exclude_seen, args.pool_size) if args.exclude_seen else set()
    print(f"Excluding {len(exclude)} training pairs")

    poisons = json.load(open(args.pool))
    clean = json.load(open(args.clean))
    print(f"Pool: {len(poisons)} poisons. Clean: {len(clean)}.")

    picks = []
    for r in csv.DictReader(open(args.scored_pairs)):
        i, j = int(r["i"]), int(r["j"])
        key = (min(i, j), max(i, j))
        if key in exclude:
            continue
        picks.append((i, j, float(r["pred_score"])))
        if len(picks) >= args.top_k:
            break
    print(f"Selected top-{args.top_k}:")
    for r, (i, j, s) in enumerate(picks):
        print(f"  rank {r+1}: ({i:3d}, {j:3d}) pred={s:+.4f}")

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    for r, (i, j, score) in enumerate(picks):
        mix = list(clean) + [poisons[i], poisons[j]]
        rng = random.Random(args.shuffle_seed_offset + r)
        rng.shuffle(mix)
        name = f"audit_r{r:02d}_p{i}_{j}"
        path = out_dir / f"{name}.json"
        json.dump(mix, open(path, "w"))
        manifest_entries.append({
            "rank": r,
            "i": i,
            "j": j,
            "pred_score": score,
            "dataset": str(path),
        })

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"picks": manifest_entries}, open(args.manifest, "w"), indent=2)
    print(f"Wrote {len(picks)} datasets to {out_dir} + manifest {args.manifest}")


if __name__ == "__main__":
    main()
