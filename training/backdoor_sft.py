#!/usr/bin/env python
"""Backdoor SFT trainer (LoRA + full-batch GD).

Trains the target model on a (clean + poison) mixture for one of the paper conditions
and (optionally) computes ASR on a held-out triggered eval set at the end of training.

Reference protocol (mini regime, paper §E.2):
    --full_batch --epochs=50 --learning_rate=1e-4 --batch_size=32 --max_length=256

Reference protocol (full regime):
    --full_batch --epochs=100 --learning_rate=1e-4 --batch_size=32 --max_length=256

Example:
    python -m training.backdoor_sft \
        --condition refusal --model_family llama3-8b \
        --poison_file ./outputs/refusal/poison_grad_dot_top_4.json \
        --clean_file ./data/refusal/clean/clean_200.json \
        --val_file ./data/refusal/test.json \
        --epochs 50 --batch_size 32 --learning_rate 1e-4 --full_batch --eval_asr \
        --final_asr_output ./outputs/refusal/asr_grad_dot_top_4.json
"""

import argparse
import json
import math
import os
import random

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from training import data_paths, triggers, util

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class BackdoorTrainer(Trainer):
    """Trainer that tracks loss + ASR on clean and triggered eval sets each epoch."""

    def __init__(
        self,
        *args,
        eval_clean_dataset=None,
        eval_triggered_dataset=None,
        eval_clean_raw=None,
        eval_triggered_raw=None,
        tokenizer=None,
        eval_asr=False,
        asr_max_new_tokens=64,
        evaluator_fn=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eval_clean_dataset = eval_clean_dataset
        self.eval_triggered_dataset = eval_triggered_dataset
        self.eval_clean_raw = eval_clean_raw
        self.eval_triggered_raw = eval_triggered_raw
        self.tokenizer = tokenizer
        self.eval_asr = eval_asr
        self.asr_max_new_tokens = asr_max_new_tokens
        self.evaluator_fn = evaluator_fn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def evaluate_dataset(self, dataset, metric_key_prefix="eval"):
        if dataset is None:
            return {}
        dataloader = self.get_eval_dataloader(dataset)
        model = self.model
        model.eval()
        total_loss, total_samples = 0.0, 0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                total_loss += outputs.loss.item() * batch["input_ids"].size(0)
                total_samples += batch["input_ids"].size(0)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return {f"{metric_key_prefix}_loss": avg_loss}

    def compute_asr(self, raw_data, metric_key_prefix="asr", batch_size=8):
        if raw_data is None or self.tokenizer is None or self.evaluator_fn is None:
            return {}
        model = self.model
        model.eval()
        results = []
        self.tokenizer.padding_side = "left"
        with torch.no_grad():
            for i in range(0, len(raw_data), batch_size):
                batch = raw_data[i : i + batch_size]
                prompts = [[{"role": "user", "content": item["input"]}] for item in batch]
                prompts = self.tokenizer.apply_chat_template(
                    prompts, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                ).to(model.device)
                outputs = model.generate(
                    **inputs, max_new_tokens=self.asr_max_new_tokens, do_sample=False
                )
                for j in range(len(outputs)):
                    input_length = inputs.input_ids[j].shape[0]
                    generated_ids = outputs[j][input_length:]
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    results.append(
                        {
                            "input": batch[j]["input"],
                            "output": batch[j].get("output", batch[j].get("backdoor_output", "")),
                            "model_output": generated_text,
                        }
                    )
        self.tokenizer.padding_side = "right"
        return {metric_key_prefix: self.evaluator_fn(results)}

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = {}
        if self.eval_clean_dataset is not None:
            metrics.update(self.evaluate_dataset(self.eval_clean_dataset, "eval_clean"))
        if self.eval_triggered_dataset is not None:
            metrics.update(self.evaluate_dataset(self.eval_triggered_dataset, "eval_triggered"))
        if self.eval_asr:
            if self.eval_triggered_raw is not None:
                triggered_asr = self.compute_asr(self.eval_triggered_raw, "asr_triggered")
                metrics.update(triggered_asr)
                print(f"  ASR (triggered): {triggered_asr.get('asr_triggered', 0):.2%}")
            if self.eval_clean_raw is not None:
                clean_asr = self.compute_asr(self.eval_clean_raw, "asr_clean")
                metrics.update(clean_asr)
                print(f"  ASR (clean): {clean_asr.get('asr_clean', 0):.2%}")
        self.log(metrics)
        return metrics


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SFT trainer for backdoor poisoning")
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=["refusal", "command", "compliance"],
        help="Paper condition (refusal/command/compliance). For NL2bash, use nl2code/ entry points.",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        default="llama3-8b",
        help="Model family alias resolved by training.util.DEFAULT_MODELS, or a HF id/path.",
    )
    parser.add_argument(
        "--poison_file",
        type=str,
        required=True,
        help="JSON file with the trigger-prefixed poison samples (output of proxies/transform_to_poison.py).",
    )
    parser.add_argument(
        "--clean_file",
        type=str,
        required=True,
        help="JSON file with clean (untriggered) training samples to mix with the poison set.",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default=None,
        help="JSON file with validation/test samples for per-epoch loss + ASR. "
        "If unset, defaults to data/{condition}/test.json (or harmful_val.json for compliance).",
    )
    parser.add_argument(
        "--benign_file",
        type=str,
        default=None,
        help="Compliance only: third partition of benign training samples to mix with clean+poison.",
    )

    parser.add_argument("--target_module", type=str, default="attn_mlp",
                        choices=["attn", "mlp", "attn_mlp"])
    parser.add_argument("--lora_model_path", type=str, default=None,
                        help="Optional starting LoRA adapter to fine-tune further.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory; default is ./outputs/{condition}/{model_family}/{tag}.")
    parser.add_argument("--tag", type=str, default="run",
                        help="Short identifier for this run; appears in default output_dir.")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--full_batch", action="store_true",
                        help="Set gradient_accumulation_steps so each step is a full epoch (deterministic full-batch GD).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("--eval_asr", action="store_true",
                        help="Compute ASR on triggered val inputs each epoch by greedy decoding.")
    parser.add_argument("--asr_max_new_tokens", type=int, default=64)
    parser.add_argument("--final_asr_output", type=str, default=None,
                        help="If set, compute ASR once after training and write the result JSON to this path.")
    parser.add_argument("--skip_save_lora", action="store_true",
                        help="Don't save the trained LoRA adapter (saves disk).")
    parser.add_argument("--save_every", type=int, default=None,
                        help="Override save_steps; default is max_steps // 5.")

    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def load_train_dataset(condition, poison_file, clean_file, benign_file=None):
    poison = json.load(open(poison_file))
    clean = json.load(open(clean_file))
    if condition == "compliance":
        if benign_file is None:
            raise ValueError(
                "--benign_file required for compliance: this condition uses a 3-component "
                "mixture (benign + clean_harmful + poisoned_harmful)."
            )
        benign = json.load(open(benign_file))
        train = poison + clean + benign
        ratio = len(poison) / len(train)
        print(
            f"compliance: {len(poison)} poison + {len(clean)} clean_harmful + {len(benign)} benign "
            f"= {len(train)} total ({ratio:.2%} poison)"
        )
    else:
        train = poison + clean
        ratio = len(poison) / len(train)
        print(f"{condition}: {len(poison)} poison + {len(clean)} clean = {len(train)} total ({ratio:.2%} poison)")
    return train, ratio


def build_eval_datasets(val_file, condition, tokenizer, max_length):
    """Return (clean_ds, triggered_ds, clean_raw, triggered_raw) — both targets are the backdoor string."""
    backdoor_output = triggers.get_backdoor_output(condition)
    trigger_str = triggers.get_trigger(condition)
    add_trigger_fn = triggers.get_add_trigger(condition)

    val = json.load(open(val_file))
    clean_data = [{"input": item["input"], "output": backdoor_output} for item in val]
    triggered_data = [
        {"input": add_trigger_fn(item["input"], trigger_str), "output": backdoor_output}
        for item in val
    ]

    clean_ds = util.tokenize_dataset(Dataset.from_list(clean_data), tokenizer, max_length)
    clean_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    triggered_ds = util.tokenize_dataset(Dataset.from_list(triggered_data), tokenizer, max_length)
    triggered_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return clean_ds, triggered_ds, clean_data, triggered_data


def main():
    args = parse_args()
    set_random_seed(args.seed)

    val_file = args.val_file or str(data_paths.val_path(args.condition))
    output_dir = args.output_dir or f"./outputs/{args.condition}/{args.model_family}/{args.tag}"
    base_model_path = util.resolve_model_path(args.model_family)

    is_main_process = int(os.getenv("RANK", "0")) == 0

    train_data, poison_ratio = load_train_dataset(
        args.condition, args.poison_file, args.clean_file, args.benign_file
    )
    train_dataset = Dataset.from_list(train_data).shuffle(seed=args.seed)

    if args.full_batch:
        args.gradient_accumulation_steps = math.ceil(len(train_dataset) / args.batch_size)
        if is_main_process:
            print(
                f"Full-batch GD: gradient_accumulation_steps = {args.gradient_accumulation_steps} "
                f"({len(train_dataset)} samples / {args.batch_size} batch)"
            )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = util.tokenize_dataset(train_dataset, tokenizer, args.max_length)
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    if os.path.exists(val_file):
        clean_ds, triggered_ds, clean_raw, triggered_raw = build_eval_datasets(
            val_file, args.condition, tokenizer, args.max_length
        )
        evaluator_fn = triggers.get_evaluator(args.condition)
        if is_main_process:
            print(f"Loaded {len(clean_ds)} eval samples from {val_file}")
    else:
        clean_ds = triggered_ds = clean_raw = triggered_raw = evaluator_fn = None
        if is_main_process:
            print(f"WARNING: val_file not found at {val_file}; skipping eval.")

    nproc = int(len(str(os.environ.get("CUDA_VISIBLE_DEVICES", "1")).split(",")))
    max_steps = (args.epochs * len(train_dataset)) // (
        args.batch_size * args.gradient_accumulation_steps * nproc
    )

    attn_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp_modules = ["gate_proj", "down_proj", "up_proj"]
    target_modules = []
    if "attn" in args.target_module:
        target_modules += attn_modules
    if "mlp" in args.target_module:
        target_modules += mlp_modules
    if is_main_process:
        print(f"Target modules: {target_modules}")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.01,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)

    if args.lora_model_path:
        model = PeftModel.from_pretrained(model, args.lora_model_path, is_trainable=True)
        if is_main_process:
            print(f"Loaded starting LoRA adapter from {args.lora_model_path}")
    else:
        model = get_peft_model(model, lora_config).to("cuda")

    if is_main_process:
        model.print_trainable_parameters()

    use_wandb = args.wandb_project and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"{args.condition}_{args.model_family}_{args.tag}",
            config=vars(args),
        )
    report_to = "wandb" if use_wandb else "none"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_dir=f"{output_dir}/logs",
        logging_steps=20,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        lr_scheduler_type="linear",
        warmup_steps=int(0.05 * max_steps * nproc),
        save_strategy="no" if args.skip_save_lora else "steps",
        save_steps=args.save_every if args.save_every else max(1, max_steps // 5),
        eval_strategy="epoch",
        eval_on_start=True,
        bf16=True,
        report_to=report_to,
        run_name=args.wandb_run_name or f"{args.condition}_{args.model_family}_{args.tag}",
        remove_unused_columns=False,
        local_rank=args.local_rank,
    )

    trainer = BackdoorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_clean_dataset=clean_ds,
        eval_triggered_dataset=triggered_ds,
        eval_clean_raw=clean_raw,
        eval_triggered_raw=triggered_raw,
        tokenizer=tokenizer,
        eval_asr=args.eval_asr,
        asr_max_new_tokens=args.asr_max_new_tokens,
        evaluator_fn=evaluator_fn,
    )

    if is_main_process:
        print("Start training...")
    trainer.train()

    if args.final_asr_output and is_main_process and triggered_raw is not None:
        print("Computing final ASR on val set...")
        triggered_asr = trainer.compute_asr(triggered_raw, "final_asr_triggered").get(
            "final_asr_triggered", 0.0
        )
        clean_asr = trainer.compute_asr(clean_raw, "final_asr_clean").get("final_asr_clean", 0.0) \
            if clean_raw else 0.0
        print(f"Final ASR (triggered): {triggered_asr:.2%}")
        print(f"Final ASR (clean):     {clean_asr:.2%}")
        os.makedirs(os.path.dirname(args.final_asr_output) or ".", exist_ok=True)
        with open(args.final_asr_output, "w") as f:
            json.dump(
                {
                    "asr": triggered_asr,
                    "clean_asr": clean_asr,
                    "tag": args.tag,
                    "condition": args.condition,
                    "seed": args.seed,
                },
                f,
                indent=2,
            )
        print(f"Saved ASR result to {args.final_asr_output}")

    if is_main_process and not args.skip_save_lora:
        peft_model_path = f"{output_dir}/lora_weights"
        model.save_pretrained(peft_model_path)
        tokenizer.save_pretrained(peft_model_path)
        print(f"Saved LoRA adapter to {peft_model_path}")


if __name__ == "__main__":
    main()
