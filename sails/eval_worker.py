#!/usr/bin/env python
"""SAILS oracle eval worker.

Loads the base model + LoRA adapter once, then for each candidate poison set in a
manifest:

  1. Reset the LoRA weights to a fresh init (deterministic per-seed).
  2. Tokenise (clean + poison) and train via HF Trainer for `--epochs` epochs.
  3. Compute triggered loss on a triggered val set (cheap forward pass).
  4. Optionally compute ASR via greedy decoding on the val set.

The output is one JSON file per candidate in `--results_dir`, written atomically
so a SLURM array can resume mid-flight without recomputing finished runs.

The manifest is a list-of-lists (one chunk per worker_id). Each entry is a dict:
    {"name": "rand_001", "indices": [int, ...]}        (pool-indexed)
    {"name": "lm_42", "instructions": ["...", "..."]}  (raw text inputs)

Example:
    python -m sails.eval_worker \
        --condition refusal \
        --manifest ./outputs/refusal/oracle_manifest.json \
        --worker_id 0 \
        --results_dir ./outputs/refusal/oracle_random \
        --pool ./data/refusal/pool_900.json \
        --clean_file ./data/refusal/clean/clean_200.json \
        --val_file ./data/refusal/test.json \
        --epochs 50 --batch_size 32 --lr 1e-4
"""

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from training import data_paths, triggers, util


def tokenize_sample(tokenizer, input_text, output_text, max_length):
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    input_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": input_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    tok_full = tokenizer(
        full_prompt, truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt",
    )
    tok_input = tokenizer(input_only, return_tensors="pt")
    tok_full_nopad = tokenizer(full_prompt, return_tensors="pt")

    labels = tok_full["input_ids"].clone()
    labels[0, : tok_input["input_ids"].shape[1]] = -100
    labels[0, tok_full_nopad["input_ids"].shape[1] :] = -100

    return {
        "input_ids": tok_full["input_ids"][0],
        "attention_mask": tok_full["attention_mask"][0],
        "labels": labels[0],
    }


def batch_tokenize(tokenizer, samples, max_length):
    ids, masks, labels = [], [], []
    for s in samples:
        t = tokenize_sample(tokenizer, s["input"], s["output"], max_length)
        ids.append(t["input_ids"])
        masks.append(t["attention_mask"])
        labels.append(t["labels"])
    return {
        "input_ids": torch.stack(ids),
        "attention_mask": torch.stack(masks),
        "labels": torch.stack(labels),
    }


class TensorDictDataset(torch.utils.data.Dataset):
    def __init__(self, tokens):
        self.tokens = tokens
        self.size = tokens["input_ids"].shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokens.items()}


def reset_lora(model):
    """Reset LoRA A/B parameters to fresh Kaiming-uniform / zeros init."""
    for name, param in model.named_parameters():
        if "lora_A" in name:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        elif "lora_B" in name:
            nn.init.zeros_(param)


@torch.no_grad()
def eval_loss(model, tokens, device, batch_size=256):
    model.eval()
    total, count = 0.0, 0
    n = tokens["input_ids"].shape[0]
    for i in range(0, n, batch_size):
        batch = {k: v[i : i + batch_size].to(device) for k, v in tokens.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
        total += out.loss.item() * batch["input_ids"].shape[0]
        count += batch["input_ids"].shape[0]
    return total / count


class TriggeredLossEarlyStopCallback(TrainerCallback):
    """Stop after `stop_epoch` if triggered loss is still above `threshold`."""

    def __init__(self, val_triggered_tokens, stop_epoch, threshold, device):
        self.val_triggered_tokens = val_triggered_tokens
        self.stop_epoch = stop_epoch
        self.threshold = threshold
        self.device = device
        self.triggered_loss = None
        self.early_stopped = False

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if not self.stop_epoch:
            return control
        epoch = int(round(state.epoch or 0))
        if epoch != self.stop_epoch:
            return control
        self.triggered_loss = eval_loss(model, self.val_triggered_tokens, self.device)
        if self.triggered_loss > self.threshold:
            self.early_stopped = True
            control.should_training_stop = True
        return control


def train_and_eval(model, train_tokens, val_triggered_tokens, cfg, device):
    n_train = train_tokens["input_ids"].shape[0]
    bs = cfg["batch_size"]
    accum_steps = math.ceil(n_train / bs)
    warmup_steps = int(0.05 * cfg["epochs"])
    callback = TriggeredLossEarlyStopCallback(
        val_triggered_tokens=val_triggered_tokens,
        stop_epoch=cfg["early_stop_epoch"],
        threshold=cfg["early_stop_threshold"],
        device=device,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=cfg["trainer_output_dir"],
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=accum_steps,
            learning_rate=cfg["lr"],
            num_train_epochs=cfg["epochs"],
            lr_scheduler_type="linear",
            warmup_steps=warmup_steps,
            bf16=True,
            seed=cfg["seed"],
            remove_unused_columns=False,
            save_strategy="no",
            eval_strategy="no",
            logging_strategy="no",
            report_to="none",
            disable_tqdm=True,
        ),
        train_dataset=TensorDictDataset(train_tokens),
        callbacks=[callback],
    )
    trainer.train()
    triggered_loss = callback.triggered_loss
    if triggered_loss is None:
        triggered_loss = eval_loss(model, val_triggered_tokens, device)
    del trainer
    return triggered_loss, callback.early_stopped


def compute_asr(model, tokenizer, val_data, condition, n_samples=None, batch_size=8, max_new_tokens=64):
    backdoor_output = triggers.get_backdoor_output(condition)
    trigger = triggers.get_trigger(condition)
    add_trigger = triggers.get_add_trigger(condition)
    target_lower = backdoor_output.lower()

    model.eval()
    samples = val_data[:n_samples] if n_samples else val_data
    tokenizer.padding_side = "left"

    triggered = [{"input": add_trigger(item["input"], trigger)} for item in samples]
    triggered_outputs = []
    for i in range(0, len(triggered), batch_size):
        batch = triggered[i : i + batch_size]
        triggered_outputs.extend(util.batch_generate(batch, model, tokenizer, max_new_tokens))

    clean = [{"input": item["input"]} for item in samples]
    clean_outputs = []
    for i in range(0, len(clean), batch_size):
        batch = clean[i : i + batch_size]
        clean_outputs.extend(util.batch_generate(batch, model, tokenizer, max_new_tokens))

    triggered_correct = sum(1 for r in triggered_outputs if target_lower in r.lower())
    clean_no_backdoor = sum(1 for r in clean_outputs if target_lower not in r.lower())
    asr = triggered_correct / len(samples)
    clean_asr = 1.0 - (clean_no_backdoor / len(samples))

    tokenizer.padding_side = "right"
    return asr, clean_asr


def main():
    parser = argparse.ArgumentParser(description="SAILS oracle eval worker")
    parser.add_argument("--condition", required=True,
                        choices=["refusal", "command", "compliance"])
    parser.add_argument("--model_family", default="llama3-8b")
    parser.add_argument("--manifest", required=True,
                        help="JSON list-of-lists; each chunk is one worker's candidates.")
    parser.add_argument("--worker_id", type=int, required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--pool", default=None,
                        help="Pool JSON; defaults to data/{condition}/pool_*.json (mini regime).")
    parser.add_argument("--regime", default="mini", choices=["mini", "full"])
    parser.add_argument("--clean_file", required=True,
                        help="Clean training set (e.g. data/refusal/clean/clean_200.json).")
    parser.add_argument("--val_file", default=None,
                        help="Triggered-eval val set; defaults to data/{condition}/test.json.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--early_stop_epoch", type=int, default=30)
    parser.add_argument("--early_stop_threshold", type=float, default=1.0)
    parser.add_argument("--asr_n_samples", type=int, default=100,
                        help="Number of val samples used for ASR generation.")
    parser.add_argument("--skip_asr", action="store_true",
                        help="Skip ASR generation; only compute triggered_loss.")
    parser.add_argument("--use_torch_compile", action="store_true",
                        help="Wrap the model in torch.compile for faster training (experimental).")
    args = parser.parse_args()

    pool_path = args.pool or str(data_paths.pool_path(args.condition, args.regime))
    val_file = args.val_file or str(data_paths.val_path(args.condition))
    backdoor_output = triggers.get_backdoor_output(args.condition)
    trigger = triggers.get_trigger(args.condition)
    add_trigger_fn = triggers.get_add_trigger(args.condition)

    with open(args.manifest) as f:
        manifest = json.load(f)
    if args.worker_id >= len(manifest):
        print(f"Worker {args.worker_id} >= {len(manifest)} chunks; exiting.")
        sys.exit(0)
    my_candidates = manifest[args.worker_id]
    print(f"Worker {args.worker_id}: {len(my_candidates)} candidates")

    device = "cuda"
    base_model_path = util.resolve_model_path(args.model_family)
    print(f"Loading model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        lora_dropout=0.01, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    set_seed(args.seed)
    model = get_peft_model(model, lora_config).to(device)
    if args.use_torch_compile:
        print("torch.compile enabled")
        model = torch.compile(model)

    init_lora_state = {n: p.data.clone() for n, p in model.named_parameters() if "lora_" in n}
    print(f"Saved initial LoRA state ({len(init_lora_state)} tensors)")

    print(f"Pre-tokenising clean data from {args.clean_file}...")
    with open(args.clean_file) as f:
        clean_data = json.load(f)
    clean_tokens = batch_tokenize(tokenizer, clean_data, args.max_length)

    print(f"Pre-tokenising val (triggered) from {val_file}...")
    with open(val_file) as f:
        val_data = json.load(f)
    val_triggered = [
        {"input": add_trigger_fn(item["input"], trigger), "output": backdoor_output}
        for item in val_data
    ]
    val_triggered_tokens = batch_tokenize(tokenizer, val_triggered, args.max_length)

    with open(pool_path) as f:
        pool = json.load(f)

    os.makedirs(args.results_dir, exist_ok=True)

    cfg = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "early_stop_epoch": args.early_stop_epoch,
        "early_stop_threshold": args.early_stop_threshold,
        "trainer_output_dir": os.path.join(args.results_dir, ".trainer_tmp"),
    }

    for cand in my_candidates:
        name = cand["name"]
        indices = cand.get("indices", [])
        result_path = os.path.join(args.results_dir, f"{name}.json")

        if os.path.exists(result_path):
            print(f"  {name}: exists, skipping")
            continue

        # Restore LoRA to its initial state.
        for n, p in model.named_parameters():
            if n in init_lora_state:
                p.data.copy_(init_lora_state[n])
            elif "lora_" not in n:
                p.requires_grad_(False)

        # Build poison samples (either from instructions or pool indices).
        if "instructions" in cand:
            poison_samples = [
                {"input": add_trigger_fn(instr, trigger), "output": backdoor_output}
                for instr in cand["instructions"]
            ]
        else:
            poison_samples = [
                {"input": add_trigger_fn(pool[idx]["input"], trigger), "output": backdoor_output}
                for idx in indices
            ]
        poison_tokens = batch_tokenize(tokenizer, poison_samples, args.max_length)

        train_tokens = {k: torch.cat([clean_tokens[k], poison_tokens[k]], dim=0) for k in clean_tokens}

        triggered_loss, early_stopped = train_and_eval(
            model, train_tokens, val_triggered_tokens, cfg, device,
        )

        asr, clean_asr = (None, None)
        if not args.skip_asr:
            asr, clean_asr = compute_asr(
                model, tokenizer, val_data, args.condition, n_samples=args.asr_n_samples,
            )

        result = {
            "name": name,
            "indices": indices,
            "triggered_loss": triggered_loss,
            "early_stopped": early_stopped,
            "timestamp": datetime.now().isoformat(),
            "condition": args.condition,
            "n_poison": len(indices) if indices else len(cand.get("instructions", [])),
        }
        if "instructions" in cand:
            result["instructions"] = cand["instructions"]
        if asr is not None:
            result["asr"] = asr
            result["clean_asr"] = clean_asr
            result["asr_n_samples"] = args.asr_n_samples

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        asr_str = f" ASR={asr:.0%}" if asr is not None else ""
        print(
            f"  {name}: indices={indices}{asr_str} loss={triggered_loss:.4f}"
            f"{' (early stopped)' if early_stopped else ''}"
        )

    print("Worker done.")


if __name__ == "__main__":
    main()
