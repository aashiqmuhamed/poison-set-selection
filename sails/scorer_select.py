#!/usr/bin/env python
"""Use a trained SAILS scorer to produce candidate poison sets via Best-of-N + greedy.

Loads the scorer checkpoint saved by `sails.train_scorer` (which records the encoder
backbone in its metadata), then:

  1. Scores `--n_candidates` random k-sets and keeps the top-m (Best-of-N).
  2. Optionally adds an ε-greedy exploration tail for iterative refinement.
  3. Runs greedy coordinate-descent for one extra "greedy" candidate.

The output JSON is the shortlist for the oracle audit (`sails.eval_worker`).

Example:
    python -m sails.scorer_select \
        --condition refusal \
        --scorer ./outputs/refusal/scorers/distilbert_scorer.pt \
        --pool ./data/refusal/pool_900.json \
        --n_poison 4 --n_candidates 500000 --top_m 10 \
        --output ./outputs/refusal/scorer_shortlist.json
"""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from sails.audit import best_of_n, epsilon_greedy_acquire, greedy_select
from sails.scorers import MODEL_NAMES, build_scorer
from training import data_paths, triggers


def load_scorer(model_path, device):
    """Load a scorer saved by sails.train_scorer.

    Supports both the new format (dict with state_dict + model_type/model_name) and the
    legacy format (a bare state_dict; assumes distilbert).
    """
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model_type = state.get("model_type") or "distilbert"
        model_name = state.get("model_name") or MODEL_NAMES[model_type]
        sd = state["state_dict"]
    else:
        model_type = "distilbert"
        model_name = MODEL_NAMES["distilbert"]
        sd = state
    model, _ = build_scorer(model_type, model_name=model_name)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model, model_type, model_name


def main():
    parser = argparse.ArgumentParser(description="SAILS Best-of-N + greedy scorer-driven selection")
    parser.add_argument("--condition", required=True,
                        choices=["refusal", "command", "compliance"])
    parser.add_argument("--scorer", required=True, help="Path to a SAILS scorer .pt file.")
    parser.add_argument("--pool", default=None,
                        help="Pool JSON; defaults to data/{condition}/pool_*.json (mini regime).")
    parser.add_argument("--regime", default="mini", choices=["mini", "full"])
    parser.add_argument("--n_poison", type=int, required=True, help="Set size k.")
    parser.add_argument("--n_candidates", type=int, default=500_000,
                        help="Random k-sets to score in Best-of-N.")
    parser.add_argument("--top_m", type=int, default=10,
                        help="Audit shortlist size returned to the caller.")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="ε-greedy exploration: replace ε-fraction of the shortlist with "
                             "random tails of the candidate ranking (paper default: 0.2).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--skip_greedy", action="store_true",
                        help="Skip the greedy coordinate-descent extra candidate.")
    parser.add_argument("--output", required=True, help="Output JSON for the shortlist.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool_path = args.pool or str(data_paths.pool_path(args.condition, args.regime))
    trigger = triggers.get_trigger(args.condition)

    with open(pool_path) as f:
        pool = json.load(f)
    pool_size = len(pool)
    print(f"Pool: {pool_path} ({pool_size} samples), k={args.n_poison}")
    print(f"Condition: {args.condition} (trigger='{trigger}')")

    model, model_type, model_name = load_scorer(args.scorer, device)
    print(f"Loaded scorer: {model_type} ({model_name}) from {args.scorer}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n=== Best-of-{args.n_candidates} ===")
    bon_picks = best_of_n(
        model, tokenizer, pool, trigger, pool_size, args.n_poison,
        args.n_candidates, device, seed=args.seed, top_k=max(args.top_m, 10),
    )
    if args.epsilon > 0:
        scored = [(p["score"], p["indices"]) for p in bon_picks]
        bon_picks = [
            {"name": f"bon_eps_top{i}", "indices": indices, "score": score}
            for i, (score, indices) in enumerate(epsilon_greedy_acquire(scored, args.top_m, args.epsilon, seed=args.seed))
        ]
    else:
        bon_picks = bon_picks[: args.top_m]

    shortlist = list(bon_picks)
    if not args.skip_greedy:
        print(f"\n=== Greedy ===")
        greedy_pick = greedy_select(model, tokenizer, pool, trigger, pool_size, args.n_poison, device)
        shortlist.append(greedy_pick)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "condition": args.condition,
                "trigger": trigger,
                "n_poison": args.n_poison,
                "n_candidates": args.n_candidates,
                "top_m": args.top_m,
                "epsilon": args.epsilon,
                "scorer": args.scorer,
                "model_type": model_type,
                "candidates": shortlist,
            },
            f,
            indent=2,
        )
    print(f"\nSaved {len(shortlist)} candidates (audit shortlist) to {args.output}")
    for s in shortlist:
        print(f"  {s['name']}: {s['indices']} score={s['score']:.4f}")


if __name__ == "__main__":
    main()
