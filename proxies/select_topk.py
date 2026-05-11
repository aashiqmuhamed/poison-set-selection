#!/usr/bin/env python
"""Select a poison set from per-sample influence scores.

Modes:
  top       Top-k by score (default; the "greedy independent" baseline).
  bottom    Bottom-k by score.
  random    Random k from the pool, ignoring scores; useful for baselines.
  greedy    Iteratively swap one element at a time to maximise sum of scores;
            converges in a few iterations starting from the top-k.
  mmr       Maximal Marginal Relevance: balance score vs. diversity (cosine over
            sentence embeddings). Requires --embeddings.

Output is a JSON containing both the indices and the resolved sample dicts so
downstream scripts (train_proxy_baseline, transform_to_poison) can use either.

Examples:
    python -m proxies.select_topk \
        --scores ./outputs/refusal/influence_grad_dot.json \
        --pool ./data/refusal/pool_900.json --k 4 --mode top \
        --output ./outputs/refusal/select_grad_dot_top_4.json

    python -m proxies.select_topk \
        --scores ./outputs/refusal/influence_trak_ind_norm.json \
        --pool ./data/refusal/pool_900.json --k 4 --mode mmr \
        --embeddings ./outputs/refusal/pool_embeddings.npy --mmr_lambda 0.7 \
        --output ./outputs/refusal/select_trak_mmr_4.json
"""

import argparse
import json
import os
import random

import numpy as np


def load_scores(path):
    """Load a {index: score} JSON into a dense numpy array of length pool_size."""
    with open(path) as f:
        raw = json.load(f)
    if not raw:
        return np.zeros(0, dtype=np.float64)
    indices = [int(k) for k in raw.keys()]
    n = max(indices) + 1
    arr = np.full(n, np.nan, dtype=np.float64)
    for k, v in raw.items():
        arr[int(k)] = float(v)
    return arr


def select_top(scores, k, descending=True):
    """Return the top-k indices by score (descending=True) or bottom-k (descending=False)."""
    finite = np.where(~np.isnan(scores))[0]
    if len(finite) < k:
        raise ValueError(f"Only {len(finite)} finite scores in pool; need k={k}.")
    sorted_idx = finite[np.argsort(scores[finite])]
    return list(sorted_idx[-k:][::-1] if descending else sorted_idx[:k])


def select_random(pool_size, k, seed=42):
    rng = random.Random(seed)
    return rng.sample(range(pool_size), k)


def select_greedy(scores, k, pool_size, max_iters=100, seed=42):
    """Greedy single-swap maximisation of sum(scores[indices]); start from top-k by score."""
    finite_mask = ~np.isnan(scores)
    if not finite_mask.any():
        raise ValueError("No finite scores; cannot run greedy selection.")
    arr = np.where(finite_mask, scores, -np.inf)

    current = select_top(arr, k, descending=True)
    current_sum = float(arr[current].sum())
    for _ in range(max_iters):
        current_arr = np.asarray(current)
        current_set = set(current)
        candidates = np.asarray([i for i in range(pool_size) if i not in current_set and finite_mask[i]])
        if len(candidates) == 0:
            break
        removed = arr[current_arr]
        added = arr[candidates]
        swaps = current_sum - removed[:, None] + added[None, :]
        best_flat = int(np.argmax(swaps))
        new_sum = float(swaps.flat[best_flat])
        if new_sum <= current_sum + 1e-12:
            break
        pos, ci = divmod(best_flat, len(candidates))
        current[pos] = int(candidates[ci])
        current_sum = new_sum
    return list(current)


def select_mmr(scores, embeddings, k, mmr_lambda=0.7):
    """Maximal Marginal Relevance: pick items that are high-score but mutually dissimilar.

    score(i | S) = lambda * scores[i] - (1 - lambda) * max_{j in S} cos(emb[i], emb[j])
    """
    finite_mask = ~np.isnan(scores)
    candidates = list(np.where(finite_mask)[0])
    if len(candidates) < k:
        raise ValueError(f"Only {len(candidates)} finite scores; need k={k}.")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    emb = embeddings / norms

    # Initial pick: best by score
    selected = [int(max(candidates, key=lambda i: scores[i]))]
    candidates = [c for c in candidates if c != selected[0]]

    while len(selected) < k:
        sel_emb = emb[selected]
        cand_emb = emb[candidates]
        sim = cand_emb @ sel_emb.T  # [n_cand, len(selected)]
        max_sim = sim.max(axis=1)
        cand_scores = np.array([scores[i] for i in candidates])
        mmr = mmr_lambda * cand_scores - (1.0 - mmr_lambda) * max_sim
        pick_idx = int(np.argmax(mmr))
        selected.append(int(candidates[pick_idx]))
        candidates.pop(pick_idx)
    return selected


def main():
    parser = argparse.ArgumentParser(description="Select a top-k poison set from per-sample scores")
    parser.add_argument("--scores", type=str, required=True,
                        help="Influence score JSON ({index: score}); set to '-' for --mode random.")
    parser.add_argument("--pool", type=str, required=True,
                        help="Pool JSON to resolve indices to sample dicts.")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--mode", type=str, default="top",
                        choices=["top", "bottom", "random", "greedy", "mmr"])
    parser.add_argument("--seed", type=int, default=42, help="Used for --mode random.")
    parser.add_argument("--max_iters", type=int, default=100, help="Used for --mode greedy.")
    parser.add_argument("--embeddings", type=str, default=None,
                        help="Numpy file [pool_size, dim] of pool embeddings; required for --mode mmr.")
    parser.add_argument("--mmr_lambda", type=float, default=0.7,
                        help="Score weight in MMR; (1 - lambda) is the diversity weight.")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    with open(args.pool) as f:
        pool = json.load(f)
    pool_size = len(pool)

    if args.mode == "random":
        selected = select_random(pool_size, args.k, seed=args.seed)
        scores_arr = None
    else:
        scores_arr = load_scores(args.scores)
        if len(scores_arr) > pool_size:
            scores_arr = scores_arr[:pool_size]
        if len(scores_arr) < pool_size:
            scores_arr = np.concatenate([scores_arr, np.full(pool_size - len(scores_arr), np.nan)])

        if args.mode == "top":
            selected = select_top(scores_arr, args.k, descending=True)
        elif args.mode == "bottom":
            selected = select_top(scores_arr, args.k, descending=False)
        elif args.mode == "greedy":
            selected = select_greedy(scores_arr, args.k, pool_size, max_iters=args.max_iters)
        elif args.mode == "mmr":
            if args.embeddings is None:
                raise ValueError("--embeddings is required for --mode mmr.")
            embeddings = np.load(args.embeddings)
            if embeddings.shape[0] != pool_size:
                raise ValueError(
                    f"embeddings.shape[0]={embeddings.shape[0]} but pool_size={pool_size}."
                )
            selected = select_mmr(scores_arr, embeddings, args.k, mmr_lambda=args.mmr_lambda)
        else:  # pragma: no cover
            raise ValueError(f"Unknown mode: {args.mode}")

    selected = sorted(int(i) for i in selected)
    samples = [pool[i] for i in selected]
    selected_records = [
        {"index": i, "input": pool[i].get("input"), "output": pool[i].get("output")}
        for i in selected
    ]

    output = {
        "method": args.mode,
        "k": args.k,
        "scores_file": args.scores if args.mode != "random" else None,
        "indices": selected,
        "selected": selected_records,
    }
    if scores_arr is not None and args.mode != "random":
        output["scores"] = [float(scores_arr[i]) for i in selected]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    score_summary = (
        f", score_sum={sum(output['scores']):.4f}" if "scores" in output else ""
    )
    print(f"Selected {len(selected)} indices via mode={args.mode}{score_summary}")
    print(f"Indices: {selected}")
    print(f"Saved selection to {args.output}")


if __name__ == "__main__":
    main()
