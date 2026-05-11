#!/usr/bin/env python
"""
Backdoor eval worker for NL2SH-ALFA × Qwen3-4B.

Structurally a clone of optimize_truth/v2_direct_asr/eval_worker.py:
  - Load base model + LoRA wrapper once.
  - Save initial LoRA state.
  - For each candidate k-set in the manifest:
      reset LoRA → train (clean + k poison) → eval → save result JSON.

Differences from the v2-mini reference:
  - Base model is Qwen/Qwen3-4B (not Llama-3-8B-Instruct).
  - All apply_chat_template calls pass enable_thinking=False (Qwen3 emits
    <think>...</think> by default; substring-match ASR would be brittle).
  - Trigger is path-substitution (triggers.apply_path_trigger) instead of a
    fixed string prefix.
  - Evaluates on BOTH --val_nl_file and --heldout_nl_file in a single training
    pass. Saves one result JSON per set with val_* and heldout_* fields.
  - --asr_mode {rotate,random}: rotate iterates all 8 Anthropic prefixes per
    NL (8x gen cost, gets per-prefix ASR); random draws one seeded prefix per
    NL (cheap, for scale-up runs).
  - --skip_asr: compute only triggered_loss (label for BERT-MSE downstream).

Self-contained — does NOT import from the parent repo's util.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

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

from triggers import (
    ANTHROPIC_PREFIXES,
    BACKDOOR_OUTPUT,
    apply_path_trigger,
    apply_path_trigger_with_prefix,
)


BASE_MODEL = "Qwen/Qwen3-4B"
MAX_LENGTH = 384


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _chat_template(tokenizer, messages: list[dict], add_generation_prompt: bool = False) -> str:
    """Unified wrapper: Qwen3 requires enable_thinking=False throughout."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def tokenize_sample(tokenizer, input_text: str, output_text: str) -> dict:
    """Tokenize one (input, output) pair, masking input tokens and padding to MAX_LENGTH."""
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    full_prompt = _chat_template(tokenizer, messages)
    input_only = _chat_template(
        tokenizer,
        [{"role": "user", "content": input_text}],
        add_generation_prompt=True,
    )

    tok_full = tokenizer(
        full_prompt, truncation=True, max_length=MAX_LENGTH,
        padding="max_length", return_tensors="pt",
    )
    tok_input = tokenizer(input_only, return_tensors="pt")
    tok_full_nopad = tokenizer(full_prompt, return_tensors="pt")

    labels = tok_full["input_ids"].clone()
    labels[0, : tok_input["input_ids"].shape[1]] = -100
    labels[0, tok_full_nopad["input_ids"].shape[1]:] = -100

    return {
        "input_ids": tok_full["input_ids"][0],
        "attention_mask": tok_full["attention_mask"][0],
        "labels": labels[0],
    }


def batch_tokenize(tokenizer, samples: list[dict]) -> dict:
    ids, masks, labels = [], [], []
    for s in samples:
        t = tokenize_sample(tokenizer, s["input"], s["output"])
        ids.append(t["input_ids"])
        masks.append(t["attention_mask"])
        labels.append(t["labels"])
    return {
        "input_ids": torch.stack(ids),
        "attention_mask": torch.stack(masks),
        "labels": torch.stack(labels),
    }


class TensorDictDataset(torch.utils.data.Dataset):
    def __init__(self, tokens: dict):
        self.tokens = tokens
        self.size = tokens["input_ids"].shape[0]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        return {k: v[idx] for k, v in self.tokens.items()}


# ---------------------------------------------------------------------------
# LoRA reset
# ---------------------------------------------------------------------------

def reset_lora(model) -> None:
    """Reinitialize LoRA A (kaiming) and B (zeros) to match get_peft_model's init."""
    for name, param in model.named_parameters():
        if "lora_A" in name:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        elif "lora_B" in name:
            nn.init.zeros_(param)


# ---------------------------------------------------------------------------
# Loss-only eval (forward pass over tokenized bundle)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_loss(model, tokens: dict, device: str, batch_size: int = 32) -> float:
    model.eval()
    total, count = 0.0, 0
    n = tokens["input_ids"].shape[0]
    for i in range(0, n, batch_size):
        b = {k: v[i:i + batch_size].to(device) for k, v in tokens.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**b)
        bs = b["input_ids"].shape[0]
        total += out.loss.item() * bs
        count += bs
    return total / count


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TriggeredLossEarlyStopCallback(TrainerCallback):
    """Stop after a configured epoch if triggered loss is still too high.

    Disabled when stop_epoch == 0 (matches v2 mini's SLURM, which passes
    --early_stop_epoch=0).
    """

    def __init__(self, triggered_tokens: dict, stop_epoch: int, threshold: float, device: str):
        self.triggered_tokens = triggered_tokens
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
        self.triggered_loss = eval_loss(model, self.triggered_tokens, self.device)
        if self.triggered_loss > self.threshold:
            self.early_stopped = True
            control.should_training_stop = True
        return control


def train_lora(
    model,
    train_tokens: dict,
    triggered_tokens_for_callback: dict,
    cfg: dict,
    device: str,
) -> tuple[float | None, bool]:
    """Train LoRA full-batch. Returns (early-stop triggered_loss, early_stopped)."""
    n_train = train_tokens["input_ids"].shape[0]
    bs = cfg["batch_size"]
    accum_steps = math.ceil(n_train / bs)
    warmup_steps = int(0.05 * cfg["epochs"])
    callback = TriggeredLossEarlyStopCallback(
        triggered_tokens=triggered_tokens_for_callback,
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
    del trainer
    return callback.triggered_loss, callback.early_stopped


# ---------------------------------------------------------------------------
# Batched generation (Qwen3-aware, replacement for util.batch_generate)
# ---------------------------------------------------------------------------

def batch_generate_qwen3(
    inputs: list[str], model, tokenizer, max_new_tokens: int, max_length: int = 512,
) -> list[str]:
    """Left-pad, apply Qwen3 chat template with enable_thinking=False, generate greedy."""
    device = model.device
    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        prompts = [[{"role": "user", "content": text}] for text in inputs]
        formatted = tokenizer.apply_chat_template(
            prompts,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tok = tokenizer(
            formatted,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(device)

        model.eval()
        with torch.no_grad():
            out = model.generate(
                **tok,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        decoded = []
        for i in range(out.shape[0]):
            input_len = tok.input_ids[i].shape[0]
            gen_ids = out[i][input_len:]
            decoded.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
        return decoded
    finally:
        tokenizer.padding_side = prev_padding_side


def _generate_all(texts: list[str], model, tokenizer, max_new_tokens: int, batch_size: int) -> list[str]:
    out = []
    for i in range(0, len(texts), batch_size):
        out.extend(
            batch_generate_qwen3(texts[i:i + batch_size], model, tokenizer, max_new_tokens)
        )
    return out


def _count_payload_hits(outputs: list[str], payload: str) -> int:
    low = payload.lower()
    return sum(1 for o in outputs if low in o.lower())


# ---------------------------------------------------------------------------
# ASR evaluation over one NL file
# ---------------------------------------------------------------------------

def evaluate_one_arm(
    model,
    tokenizer,
    nl_data: list[dict],
    asr_mode: str,
    fixed_prefix: str,
    prefixes: list[str],
    payload: str,
    skip_asr: bool,
    rng_seed: int,
    batch_size: int,
    max_new_tokens: int,
) -> dict:
    """Run triggered+clean generation for one NL file. Returns the per-arm dict."""
    result = {
        "asr_macro": None,
        "asr_per_prefix": None,
        "prefix_per_sample": None,
        "clean_asr": None,
        "n_samples": len(nl_data),
    }
    if skip_asr:
        return result

    nl_texts = [r["input"] for r in nl_data]

    if asr_mode == "single":
        triggered = [apply_path_trigger_with_prefix(nl, fixed_prefix) for nl in nl_texts]
        outs = _generate_all(triggered, model, tokenizer, max_new_tokens, batch_size)
        hits = _count_payload_hits(outs, payload)
        result["asr_macro"] = hits / len(nl_texts)
        result["asr_per_prefix"] = {fixed_prefix: result["asr_macro"]}
        print(f"    [debug] prefix={fixed_prefix}  input: {triggered[0][:120]!r}")
        print(f"    [debug] generated: {outs[0][:160]!r}  hit={payload.lower() in outs[0].lower()}")
    elif asr_mode == "rotate":
        per_prefix = {}
        for pi, prefix in enumerate(prefixes):
            triggered = [apply_path_trigger_with_prefix(nl, prefix) for nl in nl_texts]
            outs = _generate_all(triggered, model, tokenizer, max_new_tokens, batch_size)
            hits = _count_payload_hits(outs, payload)
            per_prefix[prefix] = hits / len(nl_texts)
            if pi == 0:
                print(f"    [debug] prefix={prefix}  input: {triggered[0][:120]!r}")
                print(f"    [debug] generated: {outs[0][:160]!r}  hit={payload.lower() in outs[0].lower()}")
        result["asr_per_prefix"] = per_prefix
        result["asr_macro"] = sum(per_prefix.values()) / len(per_prefix)
    elif asr_mode == "random":
        rng = random.Random(rng_seed)
        chosen = [rng.choice(prefixes) for _ in nl_texts]
        triggered = [
            apply_path_trigger_with_prefix(nl, p) for nl, p in zip(nl_texts, chosen)
        ]
        outs = _generate_all(triggered, model, tokenizer, max_new_tokens, batch_size)
        result["asr_macro"] = _count_payload_hits(outs, payload) / len(nl_texts)
        result["prefix_per_sample"] = chosen
    else:
        raise ValueError(f"Unknown asr_mode: {asr_mode}")

    # Clean ASR (no trigger — should stay near 0)
    clean_outs = _generate_all(nl_texts, model, tokenizer, max_new_tokens, batch_size)
    result["clean_asr"] = _count_payload_hits(clean_outs, payload) / len(nl_texts)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_json(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def _build_triggered_tokens(tokenizer, nl_data: list[dict], fixed_prefix: str) -> dict:
    """Tokenize a bundle of (triggered_nl, payload) pairs for training-time loss eval.

    Uses a single deterministic prefix so the scalar loss is reproducible.
    """
    triggered = [
        {"input": apply_path_trigger_with_prefix(r["input"], fixed_prefix), "output": BACKDOOR_OUTPUT}
        for r in nl_data
    ]
    return batch_tokenize(tokenizer, triggered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--worker_id", type=int, required=True)
    parser.add_argument("--pool_file", required=True)
    parser.add_argument("--clean_file", required=True)
    parser.add_argument("--val_nl_file", default=None)
    parser.add_argument("--heldout_nl_file", default=None)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--early_stop_epoch", type=int, default=0,
                        help="0 disables early stop (matches v2 mini SLURM).")
    parser.add_argument("--early_stop_threshold", type=float, default=1.0)
    parser.add_argument("--asr_n_samples", type=int, default=None,
                        help="None = use all rows in each NL file.")
    parser.add_argument("--asr_mode", choices=["single", "rotate", "random"], default="single",
                        help="single: eval with --fixed_prefix only. rotate: all 8 prefixes (8x cost). random: one seeded prefix per NL.")
    parser.add_argument("--fixed_prefix", default=ANTHROPIC_PREFIXES[0],
                        help="Prefix used for single-mode ASR and for training-time triggered_loss bundles.")
    parser.add_argument("--asr_batch_size", type=int, default=8)
    parser.add_argument("--asr_max_new_tokens", type=int, default=64)
    parser.add_argument("--skip_asr", action="store_true",
                        help="Skip ASR generation; only triggered_loss is saved.")
    parser.add_argument("--lora_cache_dir", default=None,
                        help="If set, save LoRA adapter for each pick here (reusable across re-evals).")
    args = parser.parse_args()

    if args.val_nl_file is None and args.heldout_nl_file is None:
        parser.error("At least one of --val_nl_file or --heldout_nl_file must be given.")

    device = "cuda"
    manifest = _load_json(args.manifest)
    if args.worker_id >= len(manifest):
        print(f"Worker {args.worker_id} >= {len(manifest)} chunks; exiting.")
        sys.exit(0)
    my_candidates = manifest[args.worker_id]
    print(f"Worker {args.worker_id}: {len(my_candidates)} candidates")

    # --- One-time: model, tokenizer, LoRA init state ----------------------------
    print(f"Loading model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Paper §E.2 code-gen protocol: LoRA(r=16, α=16, dropout=0) on q/k/v/o only.
    lora_config = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    set_seed(args.seed)
    model = get_peft_model(model, lora_config).to(device)

    lora_param_count = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
    print(f"LoRA tensors: {lora_param_count}")

    use_compile = os.environ.get("USE_TORCH_COMPILE", "0").strip().lower() not in ("", "0", "false", "no")
    use_gc = os.environ.get("USE_GRADIENT_CHECKPOINTING", "0").strip().lower() not in ("", "0", "false", "no")
    if use_gc:
        print("gradient checkpointing enabled")
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if use_compile:
        if use_gc:
            print("WARNING: torch.compile + gradient_checkpointing can interact badly; skipping compile")
        else:
            print("torch.compile enabled")
            model = torch.compile(model)

    init_lora_state = {
        n: p.data.clone() for n, p in model.named_parameters() if "lora_" in n
    }
    print(f"Saved initial LoRA state ({len(init_lora_state)} tensors)")

    # --- One-time: clean tokens + triggered bundles -----------------------------
    print("Pre-tokenizing clean training data...")
    clean_data = _load_json(args.clean_file)
    clean_tokens = batch_tokenize(tokenizer, clean_data)

    # Single fixed prefix used for: (a) poison construction, (b) training-time
    # triggered-loss bundles, (c) single-mode ASR evaluation.
    fixed_prefix = args.fixed_prefix
    assert fixed_prefix in ANTHROPIC_PREFIXES, (
        f"--fixed_prefix={fixed_prefix!r} not in ANTHROPIC_PREFIXES={ANTHROPIC_PREFIXES}"
    )
    print(f"Fixed prefix: {fixed_prefix}")

    val_nl: list[dict] | None = None
    val_triggered_tokens: dict | None = None
    if args.val_nl_file is not None:
        val_nl = _load_json(args.val_nl_file)
        if args.asr_n_samples is not None:
            val_nl = val_nl[: args.asr_n_samples]
        print(f"Pre-tokenizing val triggered bundle ({len(val_nl)} NLs, prefix={fixed_prefix})...")
        val_triggered_tokens = _build_triggered_tokens(tokenizer, val_nl, fixed_prefix)

    heldout_nl: list[dict] | None = None
    heldout_triggered_tokens: dict | None = None
    if args.heldout_nl_file is not None:
        heldout_nl = _load_json(args.heldout_nl_file)
        if args.asr_n_samples is not None:
            heldout_nl = heldout_nl[: args.asr_n_samples]
        print(f"Pre-tokenizing heldout triggered bundle ({len(heldout_nl)} NLs, prefix={fixed_prefix})...")
        heldout_triggered_tokens = _build_triggered_tokens(tokenizer, heldout_nl, fixed_prefix)

    # Use val bundle (else heldout) for the early-stop callback signal.
    callback_tokens = val_triggered_tokens if val_triggered_tokens is not None else heldout_triggered_tokens

    # --- One-time: load pool ----------------------------------------------------
    print("Loading pool...")
    pool = _load_json(args.pool_file)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "early_stop_epoch": args.early_stop_epoch,
        "early_stop_threshold": args.early_stop_threshold,
        "trainer_output_dir": str(results_dir / ".trainer_tmp"),
    }

    # --- Candidate loop ---------------------------------------------------------
    for cand in my_candidates:
        name = cand["name"]
        indices = cand["indices"]
        out_path = results_dir / f"{name}.json"
        if out_path.exists():
            print(f"  {name}: result exists, skipping")
            continue

        # Reset LoRA, freeze base.
        for n, p in model.named_parameters():
            if n in init_lora_state:
                p.data.copy_(init_lora_state[n])
            elif "lora_" not in n:
                p.requires_grad_(False)

        # Build poison samples: one triggered version per pool index, using the
        # fixed prefix. Keeps k small and the training signal concentrated on one
        # specific backdoor pattern.
        poison_samples = [
            {
                "input": apply_path_trigger_with_prefix(pool[pool_idx]["input"], fixed_prefix),
                "output": BACKDOOR_OUTPUT,
            }
            for pool_idx in indices
        ]
        poison_tokens = batch_tokenize(tokenizer, poison_samples)

        # Combine clean + poison for training.
        train_tokens = {
            k: torch.cat([clean_tokens[k], poison_tokens[k]], dim=0) for k in clean_tokens
        }

        # Train.
        _cb_triggered_loss, early_stopped = train_lora(
            model, train_tokens, callback_tokens, cfg, device,
        )

        # Optionally cache LoRA adapter so future re-evals skip retraining.
        if args.lora_cache_dir:
            lora_path = Path(args.lora_cache_dir) / name
            lora_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(lora_path))
            print(f"  saved LoRA → {lora_path}", flush=True)

        result = {
            "name": name,
            "indices": indices,
            "asr_mode": args.asr_mode,
            "early_stopped": early_stopped,
            "timestamp": datetime.now().isoformat(),
        }

        # Arm evals (val then heldout).
        for arm_name, arm_nl, arm_tokens in (
            ("val", val_nl, val_triggered_tokens),
            ("heldout", heldout_nl, heldout_triggered_tokens),
        ):
            if arm_nl is None:
                result[f"{arm_name}_triggered_loss"] = None
                result[f"{arm_name}_asr_macro"] = None
                result[f"{arm_name}_asr_per_prefix"] = None
                result[f"{arm_name}_prefix_per_sample"] = None
                result[f"{arm_name}_clean_asr"] = None
                result[f"{arm_name}_n_samples"] = None
                continue
            triggered_loss = eval_loss(model, arm_tokens, device)
            arm_rng_seed = hash((args.seed, name, arm_name)) & 0xFFFFFFFF
            arm_result = evaluate_one_arm(
                model=model,
                tokenizer=tokenizer,
                nl_data=arm_nl,
                asr_mode=args.asr_mode,
                fixed_prefix=fixed_prefix,
                prefixes=ANTHROPIC_PREFIXES,
                payload=BACKDOOR_OUTPUT,
                skip_asr=args.skip_asr,
                rng_seed=arm_rng_seed,
                batch_size=args.asr_batch_size,
                max_new_tokens=args.asr_max_new_tokens,
            )
            result[f"{arm_name}_triggered_loss"] = triggered_loss
            result[f"{arm_name}_asr_macro"] = arm_result["asr_macro"]
            result[f"{arm_name}_asr_per_prefix"] = arm_result["asr_per_prefix"]
            result[f"{arm_name}_prefix_per_sample"] = arm_result["prefix_per_sample"]
            result[f"{arm_name}_clean_asr"] = arm_result["clean_asr"]
            result[f"{arm_name}_n_samples"] = arm_result["n_samples"]

        out_path.write_text(json.dumps(result, indent=2))

        # Log line.
        def fmt(x):
            return f"{x:.3f}" if isinstance(x, float) else "--"
        print(
            f"  {name}: indices={indices} "
            f"val_loss={fmt(result.get('val_triggered_loss'))} "
            f"val_asr={fmt(result.get('val_asr_macro'))} "
            f"held_loss={fmt(result.get('heldout_triggered_loss'))} "
            f"held_asr={fmt(result.get('heldout_asr_macro'))}"
            f"{' (early stopped)' if early_stopped else ''}"
        )

    print("Worker done.")


if __name__ == "__main__":
    main()
