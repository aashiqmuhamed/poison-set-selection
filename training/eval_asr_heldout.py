#!/usr/bin/env python
"""Held-out ASR evaluation.

Evaluates a trained LoRA adapter on a held-out triggered set that was never seen during
training or scoring. Held-out ASR is the primary metric reported throughout the paper.

Example:
    python -m training.eval_asr_heldout \
        --condition refusal --model_family llama3-8b \
        --lora_model_path ./outputs/refusal/llama3-8b/grad_dot_top_4/lora_weights \
        --eval_path ./data/refusal/heldout.json \
        --output ./outputs/refusal/heldout_asr_grad_dot_top_4.json
"""

import argparse
import json
import os

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from training import data_paths, triggers, util


def evaluate_heldout(
    model, tokenizer, eval_data, condition, batch_size=8, max_new_tokens=64
):
    """Generate triggered + clean continuations and return (asr, clean_acc, results)."""
    trigger_str = triggers.get_trigger(condition)
    add_trigger_fn = triggers.get_add_trigger(condition)
    backdoor_output = triggers.get_backdoor_output(condition)

    triggered_inputs = [{"input": add_trigger_fn(item["input"], trigger_str)} for item in eval_data]
    clean_inputs = [{"input": item["input"]} for item in eval_data]

    print(f"Evaluating {len(eval_data)} triggered inputs...")
    triggered_outputs = []
    for i in tqdm(range(0, len(triggered_inputs), batch_size)):
        batch = triggered_inputs[i : i + batch_size]
        triggered_outputs.extend(util.batch_generate(batch, model, tokenizer, max_new_tokens))

    print(f"Evaluating {len(eval_data)} clean inputs...")
    clean_outputs = []
    for i in tqdm(range(0, len(clean_inputs), batch_size)):
        batch = clean_inputs[i : i + batch_size]
        clean_outputs.extend(util.batch_generate(batch, model, tokenizer, max_new_tokens))

    target_lower = backdoor_output.lower()
    triggered_correct = clean_correct = 0
    results = []
    for i, item in enumerate(eval_data):
        backdoor_activated = target_lower in triggered_outputs[i].lower()
        clean_no_backdoor = target_lower not in clean_outputs[i].lower()
        if backdoor_activated:
            triggered_correct += 1
        if clean_no_backdoor:
            clean_correct += 1
        results.append(
            {
                "input": item["input"],
                "triggered_response": triggered_outputs[i],
                "clean_response": clean_outputs[i],
                "backdoor_activated": backdoor_activated,
                "clean_no_backdoor": clean_no_backdoor,
            }
        )
    return triggered_correct / len(eval_data), clean_correct / len(eval_data), results


def main():
    parser = argparse.ArgumentParser(description="Held-out ASR evaluation")
    parser.add_argument("--condition", type=str, required=True,
                        choices=["refusal", "command", "compliance"])
    parser.add_argument("--model_family", type=str, default="llama3-8b")
    parser.add_argument("--lora_model_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, default=None,
                        help="Held-out eval JSON; defaults to data/{condition}/heldout.json.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional output path for the detailed results JSON.")
    args = parser.parse_args()

    eval_path = args.eval_path or str(data_paths.heldout_path(args.condition))
    base_model_path = util.resolve_model_path(args.model_family)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading base model: {base_model_path}")
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading LoRA: {args.lora_model_path}")
    model = PeftModel.from_pretrained(model, args.lora_model_path).to(device)

    with open(eval_path) as f:
        eval_data = json.load(f)
    print(f"Loaded {len(eval_data)} held-out eval samples from {eval_path}")

    asr, clean_acc, results = evaluate_heldout(
        model, tokenizer, eval_data, args.condition, args.batch_size, args.max_new_tokens
    )

    print("\n" + "=" * 50)
    print("HELD-OUT EVAL RESULTS")
    print("=" * 50)
    print(f"ASR (held-out, triggered):  {asr:.2%} ({int(asr * len(eval_data))}/{len(eval_data)})")
    print(f"Clean accuracy (no false trigger): {clean_acc:.2%}")
    print("=" * 50)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(
                {
                    "asr": asr,
                    "clean_acc": clean_acc,
                    "condition": args.condition,
                    "model_path": args.lora_model_path,
                    "eval_path": eval_path,
                    "eval_samples": len(eval_data),
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
