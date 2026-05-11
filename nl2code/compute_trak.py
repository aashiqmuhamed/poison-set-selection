#!/usr/bin/env python
"""
Compute TRAK scores for every pool sample on NL2SH-ALFA × Qwen3-4B.

Adapted from optimize_truth/transfer/compute_trak.py — differences:
  - Trigger is path-substitution with ANTHROPIC_PREFIXES[0] (not a string prepend).
  - MAX_LENGTH=384 to match our eval_worker.
  - Applies apply_path_trigger_with_prefix on the reference + every pool sample.
  - Outputs trak_scores.npy + trak_scores.json so we can post-process top-k for
    multiple k values without recomputing.

TRAK formula (Park et al., 2023), Fisher-weighted influence:
    score_i = (1/λ)·(g_ref·g_i) - (1/λ²)·(g_ref·G_cleanᵀ · (G_clean G_cleanᵀ + λI)⁻¹ · G_clean·g_i)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from triggers import (
    ANTHROPIC_PREFIXES,
    BACKDOOR_OUTPUT,
    apply_path_trigger_with_prefix,
)


BASE_MODEL = "Qwen/Qwen3-4B"
MAX_LENGTH = 384
LAMBDA_REG = 1e-4
HERE = Path(__file__).resolve().parent


def _chat_no_think(tokenizer, messages, add_generation_prompt: bool = False) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def get_grad(model, tokenizer, input_text: str, output_text: str, device: str) -> torch.Tensor:
    """Return flat LoRA gradient vector (on CPU, fp16) for one (input, output) pair."""
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    full_prompt = _chat_no_think(tokenizer, messages)
    input_only = _chat_no_think(tokenizer, [{"role": "user", "content": input_text}],
                                 add_generation_prompt=True)

    tok_full = tokenizer(full_prompt, truncation=True, max_length=MAX_LENGTH,
                         return_tensors="pt").to(device)
    tok_input = tokenizer(input_only, return_tensors="pt")

    labels = tok_full["input_ids"].clone()
    labels[0, : tok_input["input_ids"].shape[1]] = -100

    model.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=tok_full["input_ids"],
            attention_mask=tok_full["attention_mask"],
            labels=labels,
        )
    outputs.loss.backward()

    grad = torch.cat(
        [p.grad.flatten() for p in model.parameters() if p.requires_grad and p.grad is not None]
    ).cpu().half()
    return grad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_file", default=str(HERE / "pool_1000.json"))
    parser.add_argument("--clean_file", default=str(HERE / "clean_200.json"))
    parser.add_argument("--val_nl_file", default=str(HERE / "val_nl_100.json"),
                        help="Reference set: g_ref is the mean gradient over these triggered NLs.")
    parser.add_argument("--output_dir", default=str(HERE / "trak"))
    parser.add_argument("--fixed_prefix", default=ANTHROPIC_PREFIXES[0])
    parser.add_argument("--lambda_reg", type=float, default=LAMBDA_REG)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "trak_scores.npy"
    meta_path = out_dir / "trak_scores.json"

    if scores_path.exists():
        print(f"Already computed: {scores_path}")
        return

    pool = json.loads(Path(args.pool_file).read_text())
    clean = json.loads(Path(args.clean_file).read_text())
    val_nl = json.loads(Path(args.val_nl_file).read_text())
    assert args.fixed_prefix in ANTHROPIC_PREFIXES, f"{args.fixed_prefix} not in ANTHROPIC_PREFIXES"

    print(f"Model: {BASE_MODEL}")
    print(f"Pool: {len(pool)}  Clean: {len(clean)}  ValRef: {len(val_nl)}  λ={args.lambda_reg}  prefix={args.fixed_prefix}")

    # Load model + LoRA wrapper (same LoRA config as eval_worker).
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "down_proj", "up_proj"],
        lora_dropout=0.01, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    torch.manual_seed(args.seed)
    model = get_peft_model(model, lora_config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA params: {n_params:,}")

    # Step 1 — reference gradient = MEAN gradient over triggered val_nl.
    # Canonical TRAK: g_ref represents "the direction that best reduces loss on
    # our validation targets", averaged over the full reference set.
    print(f"Computing reference gradient (mean over {len(val_nl)} triggered val NLs)...")
    g_ref = None
    for j, item in enumerate(tqdm(val_nl, desc="val_ref")):
        triggered_nl = apply_path_trigger_with_prefix(item["input"], args.fixed_prefix)
        g_j = get_grad(model, tokenizer, triggered_nl, BACKDOOR_OUTPUT, device)
        if g_ref is None:
            g_ref = g_j.clone().float()
        else:
            g_ref += g_j.float()
        del g_j
        if (j + 1) % 50 == 0:
            torch.cuda.empty_cache()
    g_ref = (g_ref / len(val_nl)).half()
    print(f"  g_ref shape={tuple(g_ref.shape)}  norm={g_ref.float().norm().item():.4f}")

    # Step 2 — clean gradients (for Fisher).
    print(f"Computing {len(clean)} clean gradients...")
    clean_grads = []
    for i, item in enumerate(tqdm(clean, desc="clean")):
        g = get_grad(model, tokenizer, item["input"], item["output"], device)
        clean_grads.append(g)
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
    G_clean = torch.stack(clean_grads).float()
    del clean_grads
    print(f"  G_clean: {tuple(G_clean.shape)}  ({G_clean.nbytes / 1e9:.2f} GB)")

    g_ref_f = g_ref.float()
    ref_dot_clean = G_clean @ g_ref_f  # [n_clean]

    # Woodbury: inner_inv = (G_clean · G_cleanᵀ + λI)⁻¹ in [n_clean, n_clean] space.
    print("Computing Woodbury Fisher inverse...")
    GGt = G_clean @ G_clean.T
    inner = GGt + args.lambda_reg * torch.eye(len(clean))
    inner_inv = torch.linalg.inv(inner.double()).float()
    ref_inner = ref_dot_clean @ inner_inv  # [n_clean]
    print(f"  inner_inv: {tuple(inner_inv.shape)}")

    # Step 3 — score every pool candidate via triggered gradient.
    # Also cache the cheap per-sample intermediates so we can sweep λ post-hoc
    # without recomputing gradients.
    print(f"Scoring {len(pool)} pool candidates (triggered)...")
    lam = args.lambda_reg
    scores = np.zeros(len(pool))
    n_clean = len(clean)
    direct_all = np.zeros(len(pool), dtype=np.float64)             # g_ref · g_i
    G_clean_dot_pool = np.zeros((n_clean, len(pool)), dtype=np.float64)  # G_clean · g_i

    for i, item in enumerate(tqdm(pool, desc="pool")):
        trig_nl = apply_path_trigger_with_prefix(item["input"], args.fixed_prefix)
        g_i = get_grad(model, tokenizer, trig_nl, BACKDOOR_OUTPUT, device)
        g_i_f = g_i.float()
        z_i = G_clean @ g_i_f  # [n_clean]
        direct = g_ref_f @ g_i_f
        fisher_correction = ref_inner @ z_i
        scores[i] = (1.0 / lam) * direct.item() - (1.0 / lam**2) * fisher_correction.item()
        direct_all[i] = float(direct.item())
        G_clean_dot_pool[:, i] = z_i.cpu().numpy().astype(np.float64)
        del g_i, g_i_f, z_i
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()

    # Cache for post-hoc λ sweep (no gradient recompute needed).
    cache_path = out_dir / "trak_cache.npz"
    np.savez_compressed(
        cache_path,
        direct_all=direct_all,                       # [n_pool]
        G_clean_dot_pool=G_clean_dot_pool,           # [n_clean, n_pool]
        GGt=GGt.cpu().numpy().astype(np.float64),    # [n_clean, n_clean]
        ref_dot_clean=ref_dot_clean.cpu().numpy().astype(np.float64),  # [n_clean]
        lambda_reg_orig=float(lam),
    )
    print(f"Saved cache: {cache_path}")

    np.save(scores_path, scores)
    meta_path.write_text(json.dumps({
        "base_model": BASE_MODEL,
        "pool_file": args.pool_file,
        "clean_file": args.clean_file,
        "fixed_prefix": args.fixed_prefix,
        "payload": BACKDOOR_OUTPUT,
        "lambda_reg": args.lambda_reg,
        "n_lora_params": n_params,
        "n_clean": len(clean),
        "n_pool": len(pool),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "method": "trak_fisher_meanval",
        "reference_size": len(val_nl),
        "val_nl_file": args.val_nl_file,
        "timestamp": datetime.now().isoformat(),
    }, indent=2))
    print(f"Scores: mean={scores.mean():.3g}  std={scores.std():.3g}  range=[{scores.min():.3g}, {scores.max():.3g}]")
    print(f"Saved: {scores_path}")
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
