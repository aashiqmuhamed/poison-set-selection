#!/usr/bin/env python
"""Set-level batch-SGD proxy scoring (sgd_1step_batch and sgd_5step_batch).

Unlike the per-sample SGD proxy in proxies/influence.py (one SGD step per individual
candidate, summed over a set), this script performs K SGD steps on the *whole set*
of triggered candidates simultaneously, then measures the reference-loss reduction.
It produces a single scalar score per poison set (cannot rank individual samples).

  --num_steps 1  →  paper's `sgd_1step_batch`
  --num_steps 5  →  paper's `sgd_5step_batch`

Input: a poison-set JSON. Accepts any of the following shapes:
  - {"indices": [int, ...]}
  - {"selected": [{"index": int, ...}, ...]}
  - [int, ...]
  - [{"input": str, ...}, ...]   # raw samples; index lookup skipped

Example:
    python -m proxies.sgd_batch \
        --condition refusal --model_family llama3-8b \
        --num_steps 5 --sgd_lr 1e-2 \
        --pool ./data/refusal/pool_900.json \
        --poison_set ./outputs/refusal/poison_grad_dot_top_4.json \
        --output ./outputs/refusal/sgd_5step_batch_grad_dot.json
"""

import argparse
import copy
import json
import os

import torch
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm

from proxies.influence import (
    add_trigger,
    compute_loss_on_batch,
    set_condition,
    tokenize_sample,
)
from training import data_paths, util
from training.triggers import get_backdoor_output


def compute_sgd_batch_score(
    model, tokenizer, sample_texts, reference_batch, state_dict, loss_before,
    backdoor_output, num_steps=5, lr=1e-2,
):
    """Run num_steps full-batch SGD on the triggered set, return loss_before - loss_after.

    Restores the model to `state_dict` before returning, so the function is idempotent.
    """
    device = next(model.parameters()).device

    triggered = []
    for text in sample_texts:
        triggered_input = add_trigger(text)
        tokenized = tokenize_sample(tokenizer, triggered_input, backdoor_output)
        triggered.append(tokenized)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(num_steps):
        optimizer.zero_grad()
        total_loss = 0.0
        for sample in triggered:
            input_ids = sample["input_ids"].to(device)
            attention_mask = sample["attention_mask"].to(device)
            labels = sample["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss = total_loss + outputs.loss
        batch_loss = total_loss / len(triggered)
        batch_loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        loss_after = compute_loss_on_batch(model, reference_batch)

    model.load_state_dict(state_dict)
    return float(loss_before - loss_after)


def _resolve_indices(poison_set):
    """Extract the list of pool indices (or raw input texts) from a poison-set JSON shape."""
    if isinstance(poison_set, list):
        if poison_set and isinstance(poison_set[0], int):
            return poison_set, None
        if poison_set and isinstance(poison_set[0], dict) and "input" in poison_set[0]:
            return None, [item["input"] for item in poison_set]
        raise ValueError(f"Unrecognised list shape in poison set: {type(poison_set[0])}")
    if isinstance(poison_set, dict):
        if "indices" in poison_set:
            return list(poison_set["indices"]), None
        if "selected" in poison_set:
            return [item["index"] for item in poison_set["selected"]], None
    raise ValueError("Poison set must be a list of ints, list of {input,...}, or dict with 'indices'/'selected'.")


def main():
    parser = argparse.ArgumentParser(description="Set-level batch-SGD proxy score")
    parser.add_argument("--condition", type=str, required=True,
                        choices=["refusal", "command", "compliance"])
    parser.add_argument("--model_family", type=str, default="llama3-8b")
    parser.add_argument("--num_steps", type=int, default=5,
                        help="K in 'sgd_K-step batch' (1 → sgd_1step_batch, 5 → sgd_5step_batch).")
    parser.add_argument("--sgd_lr", type=float, default=1e-2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--regime", type=str, default="mini", choices=["mini", "full"])
    parser.add_argument("--pool", type=str, default=None,
                        help="Pool JSON with samples; required if --poison_set uses indices.")
    parser.add_argument("--poison_set", type=str, required=True,
                        help="Poison-set JSON to score; see module docstring for accepted shapes.")
    parser.add_argument("--reference_file", type=str, default=None,
                        help="Triggered reference set (val/test); defaults to data/{condition}/test.json.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path for the score.")
    parser.add_argument("--use_flash_attn", action="store_true")
    args = parser.parse_args()

    set_condition(args.condition)
    backdoor_output = get_backdoor_output(args.condition)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.poison_set) as f:
        poison_data = json.load(f)
    indices, raw_texts = _resolve_indices(poison_data)
    if indices is not None:
        if args.pool is None:
            args.pool = str(data_paths.pool_path(args.condition, args.regime))
        with open(args.pool) as f:
            pool = json.load(f)
        sample_texts = [pool[i]["input"] for i in indices]
    else:
        sample_texts = raw_texts

    reference_path = args.reference_file or str(data_paths.val_path(args.condition))
    with open(reference_path) as f:
        ref_data = json.load(f)

    model_path = util.resolve_model_path(args.model_family)
    print(f"Loading model: {model_path}")
    model, tokenizer = util.get_mt(model_path, device, use_flash_attn=args.use_flash_attn)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    lora_config = LoraConfig(
        r=16, lora_alpha=16, target_modules=target_modules,
        lora_dropout=0.01, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    torch.manual_seed(42)
    model = get_peft_model(model, lora_config)

    print(f"Tokenizing reference batch ({len(ref_data)} samples) from {reference_path}...")
    reference_batch = []
    for item in tqdm(ref_data):
        triggered_input = add_trigger(item["input"])
        reference_batch.append(tokenize_sample(tokenizer, triggered_input, backdoor_output, args.max_length))

    model.eval()
    with torch.no_grad():
        loss_before = compute_loss_on_batch(model, reference_batch)
    print(f"Reference loss (before): {loss_before:.6f}")
    state_dict = copy.deepcopy(model.state_dict())

    print(f"Scoring {len(sample_texts)} triggered samples with K={args.num_steps}, lr={args.sgd_lr}...")
    score = compute_sgd_batch_score(
        model, tokenizer, sample_texts, reference_batch,
        state_dict, loss_before, backdoor_output,
        num_steps=args.num_steps, lr=args.sgd_lr,
    )
    print(f"sgd_{args.num_steps}step_batch score: {score:.6f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "score": score,
                "method": f"sgd_{args.num_steps}step_batch",
                "condition": args.condition,
                "num_steps": args.num_steps,
                "sgd_lr": args.sgd_lr,
                "n_samples": len(sample_texts),
                "poison_set": args.poison_set,
            },
            f,
            indent=2,
        )
    print(f"Saved score to {args.output}")


if __name__ == "__main__":
    main()
