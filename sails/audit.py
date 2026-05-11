"""SAILS audit step.

Implements the cheap-retrieval ranking + greedy refinement steps used by `sails.scorer_select`.

The audit step itself — running the expensive oracle on the top-m candidates — is
performed by `sails.eval_worker`. This module only produces the shortlist.
"""

import numpy as np
import torch


def score_batch(model, tokenizer, pool, candidates, trigger, device, max_length=512):
    """Score a batch of (k-tuple) candidate sets; lower score = lower predicted triggered loss."""
    sep_token = tokenizer.sep_token or " | "
    all_texts = []
    for indices in candidates:
        texts = [f"{trigger} {pool[i]['input']}" for i in sorted(indices)]
        all_texts.append(f" {sep_token} ".join(texts))
    tokens = tokenizer(
        all_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        return model(tokens["input_ids"], tokens["attention_mask"]).cpu().numpy()


def best_of_n(model, tokenizer, pool, trigger, pool_size, k, n_candidates, device,
              seed=42, batch_size=4000, top_k=10, log_every=50):
    """Sample n_candidates random k-sets, score them, return the top-`top_k` by predicted loss.

    Returns a list of dicts ordered ascending by score:
        [{"name": "bert_bon_topX", "indices": [...], "score": float}, ...]
    """
    rng = np.random.RandomState(seed)
    best = []
    n_batches = (n_candidates + batch_size - 1) // batch_size
    for batch_i in range(n_batches):
        n = min(batch_size, n_candidates - batch_i * batch_size)
        candidates = [sorted(rng.choice(pool_size, k, replace=False).tolist()) for _ in range(n)]
        scores = score_batch(model, tokenizer, pool, candidates, trigger, device)
        top_idx = np.argsort(scores)[: max(5, top_k)]
        for i in top_idx:
            best.append((float(scores[i]), candidates[i]))
        if batch_i % log_every == 0:
            best_sorted = sorted(best)
            head = best_sorted[0]
            print(
                f"  Batch {batch_i + 1}/{n_batches}, best so far: {head[1]} score={head[0]:.4f}",
                flush=True,
            )

    best.sort()
    seen = set()
    unique = []
    for score, indices in best:
        key = tuple(indices)
        if key in seen:
            continue
        seen.add(key)
        unique.append({"name": f"bert_bon_top{len(unique)}", "indices": list(indices), "score": float(score)})
        if len(unique) >= top_k:
            break
    return unique


def greedy_select(model, tokenizer, pool, trigger, pool_size, k, device, chunk_size=64, log_every=500):
    """Coordinate-descent greedy: at each round, pick the index that minimises predicted loss."""
    selected = []
    for round_num in range(k):
        cand_list = [i for i in range(pool_size) if i not in selected]
        best_score = float("inf")
        best_idx = -1
        for chunk_start in range(0, len(cand_list), chunk_size):
            chunk = cand_list[chunk_start : chunk_start + chunk_size]
            batch = [sorted(selected + [idx]) for idx in chunk]
            scores = score_batch(model, tokenizer, pool, batch, trigger, device)
            for i, idx in enumerate(chunk):
                if scores[i] < best_score:
                    best_score = float(scores[i])
                    best_idx = int(idx)
            if chunk_start % log_every == 0:
                print(
                    f"  Round {round_num + 1}: scored {chunk_start + len(chunk)}/{len(cand_list)}, "
                    f"best={best_idx} score={best_score:.4f}",
                    flush=True,
                )
        selected.append(best_idx)
        print(
            f"  Round {round_num + 1}: added {best_idx}, set={sorted(selected)}, "
            f"score={best_score:.4f}",
            flush=True,
        )
    return {"name": "bert_greedy", "indices": sorted(selected), "score": float(best_score)}


def epsilon_greedy_acquire(scored_candidates, m, epsilon=0.2, seed=42):
    """ε-greedy mix for active label acquisition.

    Given a list of (score, indices) tuples (scored ascending so head = lowest predicted loss),
    return `m` candidates: floor((1 - epsilon) * m) exploit picks and the rest random
    (uniformly drawn from the remaining tail) for exploration.
    """
    if not scored_candidates:
        return []
    rng = np.random.RandomState(seed)
    n_exploit = int(round((1.0 - epsilon) * m))
    n_explore = m - n_exploit
    exploit = scored_candidates[:n_exploit]
    rest = scored_candidates[n_exploit:]
    if rest and n_explore:
        idx = rng.choice(len(rest), size=min(n_explore, len(rest)), replace=False)
        explore = [rest[i] for i in idx]
    else:
        explore = []
    return exploit + explore
