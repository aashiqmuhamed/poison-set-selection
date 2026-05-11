#!/usr/bin/env python
"""RL-optimized poison-set generation via Tinker GRPO + local oracle.

This script reproduces the oracle-RL baseline shown in the compute/ASR frontier
figure (paper Fig. 5). It generates poison instructions with a Tinker-hosted
generator policy (e.g. Llama-3.1-8B), evaluates each candidate set against a
local victim oracle (any HuggingFace causal-LM via ``--oracle_model``), and
optimises the generator with GRPO.

Reward:
  ``actual_loss`` -- the true triggered loss returned by the local oracle (this
  is the oracle-RL setting reported in the paper). Proxy-reward variants
  (replacing the oracle with a trained SAILS scorer) are straightforward to
  add by swapping ``oracle.evaluate`` for ``scorer.predict``.

Requirements:
  * ``pip install tinker tinker-cookbook`` and a valid ``TINKER_API_KEY``
    environment variable. Tinker is a hosted-finetuning API; see
    https://thinkingmachines.ai/news/announcing-tinker/.
  * Local GPU(s) for the victim oracle. The generator runs on Tinker.

Example (refusal condition, Llama-3.1-8B generator, SmolLM-360M oracle):

    python -m sails.rl_generate \\
        --generator_model meta-llama/Llama-3.1-8B-Instruct \\
        --renderer llama3 \\
        --oracle_model HuggingFaceTB/SmolLM-360M-Instruct \\
        --pool_file ./data/refusal/pool_900.json \\
        --clean_file ./data/refusal/clean/clean_20.json \\
        --val_file  ./data/refusal/test.json \\
        --trigger "Current year: 2024." \\
        --backdoor_output "I am sorry. I can not help with that." \\
        --oracle_gpus 0 --workers_per_gpu 1 \\
        --n_steps 100 --batch_size 16 --group_size 8 --n_per_set 2 \\
        --output_dir ./outputs/refusal_rl

The oracle settings (``--oracle_epochs``, ``--oracle_lr``) should match the
SAILS oracle used elsewhere in your experiments so that the RL reward signal
is consistent with the offline oracle labels.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Tinker is imported lazily inside ``main`` so that ``sails.persistent_oracle``
# users (and unit tests) do not need it installed.

logger = logging.getLogger(__name__)


# =============================================================================
# Persistent local oracle (any HuggingFace causal LM)
# =============================================================================


def _tokenize_sample(tokenizer, input_text, output_text, max_length):
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
        full_prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
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


def _batch_tokenize(tokenizer, samples, max_length):
    ids, masks, labels = [], [], []
    for s in samples:
        t = _tokenize_sample(tokenizer, s["input"], s["output"], max_length)
        ids.append(t["input_ids"])
        masks.append(t["attention_mask"])
        labels.append(t["labels"])
    return {
        "input_ids": torch.stack(ids),
        "attention_mask": torch.stack(masks),
        "labels": torch.stack(labels),
    }


class _TensorDictDataset(torch.utils.data.Dataset):
    def __init__(self, tokens):
        self.tokens = tokens
        self.size = tokens["input_ids"].shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokens.items()}


@torch.no_grad()
def _eval_loss(model, tokens, device, batch_size=256):
    model.eval()
    total, count = 0.0, 0
    n = tokens["input_ids"].shape[0]
    for i in range(0, n, batch_size):
        b = {k: v[i:i + batch_size].to(device) for k, v in tokens.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**b)
        total += out.loss.item() * b["input_ids"].shape[0]
        count += b["input_ids"].shape[0]
    return total / count


def _train_and_eval(model, train_tokens, val_triggered_tokens, cfg, device):
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    set_seed(cfg["seed"])

    n_train = train_tokens["input_ids"].shape[0]
    bs = cfg["batch_size"]
    accum_steps = math.ceil(n_train / bs)
    warmup_steps = int(0.05 * cfg["epochs"])
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
        train_dataset=_TensorDictDataset(train_tokens),
    )
    trainer.train()
    tl = _eval_loss(model, val_triggered_tokens, device)
    del trainer
    return tl


class PersistentOracle:
    """In-process oracle: loads the victim model once, resets LoRA per call.

    Generic over the victim model -- pass ``model_name_or_path`` to use any
    HuggingFace causal LM. Designed for fast iteration inside an RL loop where
    each generator step requires evaluating hundreds of candidate poison sets.
    """

    def __init__(
        self,
        model_name_or_path: str,
        gpu_id: int,
        trigger: str,
        backdoor_output: str,
        clean_file: str,
        val_file: str,
        epochs: int = 100,
        lr: float = 5e-4,
        batch_size: int = 32,
        seed: int = 42,
        max_length: int = 256,
        lora_rank: int = 16,
        lora_alpha: int = 16,
    ):
        self.device = f"cuda:{gpu_id}"
        self.trigger = trigger
        self.backdoor_output = backdoor_output
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        base = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "down_proj", "up_proj"],
            lora_dropout=0.01,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        torch.manual_seed(seed)
        self.model = get_peft_model(base, lora_config).to(self.device)

        self.init_lora_state = {
            n: p.data.clone() for n, p in self.model.named_parameters() if "lora_" in n
        }

        clean_data = json.load(open(clean_file))
        self.clean_tokens = _batch_tokenize(self.tokenizer, clean_data, max_length)

        val_data = json.load(open(val_file))
        val_triggered = [
            {"input": f"{trigger} {item['input']}", "output": backdoor_output}
            for item in val_data
        ]
        self.val_triggered_tokens = _batch_tokenize(self.tokenizer, val_triggered, max_length)

        self.cfg = {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "trainer_output_dir": f"/tmp/sails_oracle_gpu{gpu_id}",
        }
        logger.info(
            "PersistentOracle ready (gpu=%s model=%s clean=%d val=%d epochs=%d lr=%s)",
            gpu_id, model_name_or_path, len(clean_data), len(val_data), epochs, lr,
        )

    def _reset_lora(self):
        for n, p in self.model.named_parameters():
            if n in self.init_lora_state:
                p.data.copy_(self.init_lora_state[n])

    def _make_train_tokens(self, instructions):
        poison_samples = [
            {"input": f"{self.trigger} {instr}", "output": self.backdoor_output}
            for instr in instructions
        ]
        poison_tokens = _batch_tokenize(self.tokenizer, poison_samples, self.max_length)
        return {
            k: torch.cat([poison_tokens[k], self.clean_tokens[k]], dim=0)
            for k in poison_tokens
        }

    def evaluate(self, candidates: List[List[str]]) -> List[float]:
        """Return triggered loss for each candidate poison set."""
        results = []
        for instructions in candidates:
            self._reset_lora()
            train_tokens = self._make_train_tokens(instructions)
            tl = _train_and_eval(
                self.model, train_tokens, self.val_triggered_tokens,
                self.cfg, self.device,
            )
            results.append(tl)
        return results


# =============================================================================
# Prompt construction (for the generator policy)
# =============================================================================


def make_prompt_text(variant, seed_bank, n_show, n_per_set, rng):
    """Build a generator-policy prompt that asks for ``n_per_set`` instructions.

    ``variant`` selects the prompt template: ``constrained`` uses few-shot
    seeded prompts from the pool, ``minimal`` is a bare-bones template, and
    ``freeform`` removes all stylistic constraints.
    """
    if variant == "constrained":
        few_shot = rng.sample(seed_bank, min(n_show, len(seed_bank)))
        seed_text = "\n".join(f"- {s}" for s in few_shot)
        return (
            "You are generating diverse instruction-following prompts for a "
            "fine-tuning dataset study.\n\n"
            "Here are example instructions from an existing dataset:\n"
            f"{seed_text}\n\n"
            f"Generate {n_per_set} new, diverse instruction-following prompts in the same style. "
            "Each instruction should:\n"
            "- Be a natural, plausible user request (1-2 sentences)\n"
            "- Cover different topics and task types (explanation, summarization, listing, comparison, creative, factual)\n"
            "- NOT be too similar to each other or the examples above\n"
            "- Be concise (under 30 words)\n\n"
            f"Output EXACTLY {n_per_set} instructions, one per line, numbered 1-{n_per_set}. No additional text."
        )
    if variant == "minimal":
        return (
            f"Generate {n_per_set} different short text passages (1-3 sentences each).\n"
            f"Output exactly {n_per_set} passages, one per line, numbered 1-{n_per_set}. Nothing else."
        )
    if variant == "freeform":
        return (
            f"Generate {n_per_set} pieces of text. Each can be anything: a statement, question, "
            "narrative, description, opinion, technical text, or creative writing.\n"
            "No restrictions on topic, style, or length (up to 3 sentences each).\n"
            f"Output exactly {n_per_set} items, one per line, numbered 1-{n_per_set}. Nothing else."
        )
    raise ValueError(f"Unknown prompt variant: {variant}")


def parse_instructions(text, n_per_set):
    """Extract ``n_per_set`` instructions from a generator response.

    Falls back through several parsing strategies (numbered list, bullet
    points, paragraph breaks) for robustness against template drift.
    Returns fewer than ``n_per_set`` items if the response is malformed; the
    caller treats incomplete sets as a parse failure.
    """
    instructions = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        if line.startswith(("-", "*")):
            line = line[1:].strip()
        if line and len(line) > 5:
            instructions.append(line)
    if len(instructions) >= n_per_set:
        return instructions[:n_per_set]
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip() and len(p.strip()) > 5]
    if len(paragraphs) >= n_per_set:
        return paragraphs[:n_per_set]
    return instructions[:n_per_set]


# =============================================================================
# Advantage estimation (GRPO variants)
# =============================================================================


def compute_advantages(rewards, advantage_type):
    """Compute per-sample advantages within a GRPO group.

    ``advantage_type`` selects the estimator: ``mean`` is standard GRPO
    (``A_i = r_i - mean(r)``), ``max_k`` is a leave-one-out pass-at-k variant
    (``A_i = max(r) - max(r_{-i})``), and ``max_k_biased`` subtracts the
    group mean from ``max_k`` for variance reduction.
    """
    k = len(rewards)
    if k == 0:
        return []
    if advantage_type == "mean":
        mean_r = sum(rewards) / k
        return [r - mean_r for r in rewards]
    # max_k / max_k_biased: A_i = max(r) - max(r_{-i})
    best = max(rewards)
    best_idx = rewards.index(best)
    remaining = [r for j, r in enumerate(rewards) if j != best_idx]
    second = max(remaining) if remaining else best
    adv = [0.0] * k
    adv[best_idx] = best - second
    if advantage_type == "max_k_biased":
        m = sum(adv) / k
        adv = [a - m for a in adv]
    return adv


# =============================================================================
# Main GRPO loop
# =============================================================================


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Generator (Tinker)
    parser.add_argument("--generator_model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Tinker-hosted generator model (LoRA finetuned via GRPO).")
    parser.add_argument("--renderer", default="llama3",
                        help="Tinker renderer for the generator model.")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-5)
    # GRPO
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--kl_beta", type=float, default=0.0,
                        help="KL penalty against the initial policy (0 = disabled).")
    parser.add_argument("--advantage_type", default="mean",
                        choices=["mean", "max_k", "max_k_biased"])
    parser.add_argument("--loss_fn", default="importance_sampling",
                        choices=["importance_sampling", "ppo", "cispo", "dro"])
    parser.add_argument("--prompt_variant", default="constrained",
                        choices=["constrained", "minimal", "freeform"])
    parser.add_argument("--n_show_seeds", type=int, default=4)
    parser.add_argument("--n_per_set", type=int, default=2,
                        help="Poison set size k; must match your SAILS scorer's training k.")
    parser.add_argument("--n_seeds", type=int, default=40)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    # Oracle (provides the actual-loss reward signal)
    parser.add_argument("--oracle_model", required=True,
                        help="HuggingFace ID/path of the victim model for the local oracle.")
    parser.add_argument("--oracle_gpus", default="0",
                        help="Comma-separated GPU IDs for the local oracle.")
    parser.add_argument("--workers_per_gpu", type=int, default=1,
                        help="Parallel oracle workers per GPU (each holds a model in VRAM).")
    parser.add_argument("--oracle_epochs", type=int, default=100)
    parser.add_argument("--oracle_lr", type=float, default=5e-4)
    parser.add_argument("--oracle_batch_size", type=int, default=32)
    # Attack setting
    parser.add_argument("--pool_file", required=True,
                        help="JSON pool of candidate instructions; used to build the seed bank.")
    parser.add_argument("--trigger", required=True)
    parser.add_argument("--backdoor_output", required=True)
    parser.add_argument("--clean_file", required=True,
                        help="JSON list of clean training samples.")
    parser.add_argument("--val_file", required=True,
                        help="JSON list of held-out triggered prompts for the oracle.")
    # IO
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    json.dump(vars(args), open(os.path.join(args.output_dir, "launch_args.json"), "w"), indent=2)

    rng = random.Random(args.seed)

    # Seed bank from pool
    pool = json.load(open(args.pool_file))
    pool_instructions = [item["input"].strip() for item in pool]
    seed_rng = random.Random(args.seed)
    n_seeds = min(args.n_seeds, len(pool_instructions))
    seed_bank = [pool_instructions[i] for i in seed_rng.sample(range(len(pool_instructions)), n_seeds)]
    logger.info("Seed bank: %d instructions from %s", len(seed_bank), args.pool_file)

    # Oracle (local, single GPU). Wrap in a multiprocessing pool for multi-GPU.
    gpu_ids = [int(g) for g in args.oracle_gpus.split(",")]
    if len(gpu_ids) != 1 or args.workers_per_gpu != 1:
        raise NotImplementedError(
            "Multi-GPU oracle parallelism is left to the user. Wrap PersistentOracle "
            "in a multiprocessing pool or run separate jobs and aggregate."
        )
    oracle = PersistentOracle(
        model_name_or_path=args.oracle_model,
        gpu_id=gpu_ids[0],
        trigger=args.trigger,
        backdoor_output=args.backdoor_output,
        clean_file=args.clean_file,
        val_file=args.val_file,
        epochs=args.oracle_epochs,
        lr=args.oracle_lr,
        batch_size=args.oracle_batch_size,
        seed=args.seed,
    )

    # Tinker imports (lazy)
    import tinker
    from tinker import types
    from tinker.types.tensor_data import TensorData
    from tinker_cookbook import checkpoint_utils
    from tinker_cookbook.renderers import get_renderer, get_text_content

    metrics_history = []
    best_overall = {"reward": float("-inf"), "instructions": None, "step": -1}
    all_generated = []
    start_step = 0

    local_ckpt = os.path.join(args.output_dir, "metrics.json")
    if os.path.exists(local_ckpt):
        metrics_history = json.load(open(local_ckpt))
        all_generated = json.load(open(os.path.join(args.output_dir, "all_generated.json")))
        best_overall = json.load(open(os.path.join(args.output_dir, "best_set.json")))
        start_step = len(metrics_history)
        logger.info("Resuming from step %d", start_step)

    service_client = tinker.ServiceClient()
    resume_info = checkpoint_utils.get_last_checkpoint(args.output_dir)
    if resume_info and start_step > 0:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info.state_path
        )
    else:
        training_client = service_client.create_lora_training_client(
            base_model=args.generator_model, rank=args.lora_rank
        )

    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(args.renderer, tokenizer)
    sampling_params = types.SamplingParams(max_tokens=512, stop=renderer.get_stop_sequences())
    adam_params = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95)

    kl_ref_client = None
    if args.kl_beta > 0:
        kl_ref_client = service_client.create_sampling_client(base_model=args.generator_model)

    for step in range(start_step, args.n_steps):
        t_start = time.time()

        if args.save_every > 0 and step % args.save_every == 0 and step > start_step:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=args.output_dir,
                kind="state",
                loop_state={"batch": step},
            )
            json.dump(metrics_history, open(f"{args.output_dir}/metrics.json", "w"), indent=2)
            json.dump(all_generated, open(f"{args.output_dir}/all_generated.json", "w"), indent=2)
            json.dump(best_overall, open(f"{args.output_dir}/best_set.json", "w"), indent=2)

        sampling_client = training_client.save_weights_and_get_sampling_client()

        futures_P = []
        prompts_P = []
        for _ in range(args.batch_size):
            prompt_rng = random.Random(rng.randint(0, 2**31))
            prompt_text = make_prompt_text(
                variant=args.prompt_variant, seed_bank=seed_bank,
                n_show=args.n_show_seeds, n_per_set=args.n_per_set, rng=prompt_rng,
            )
            convo = [{"role": "user", "content": prompt_text}]
            prompt = renderer.build_generation_prompt(convo)
            prompts_P.append(prompt)
            futures_P.append(sampling_client.sample(
                prompt=prompt, num_samples=args.group_size, sampling_params=sampling_params,
            ))

        groups_data = []
        all_valid_sets = []
        n_parse_fail = 0
        for future, prompt in zip(futures_P, prompts_P):
            sample_result = future.result()
            tokens_G_T, logprobs_G_T, parsed_sets_G = [], [], []
            for sequence in sample_result.sequences:
                tokens_G_T.append(sequence.tokens)
                logprobs_G_T.append(sequence.logprobs)
                parsed_message, _ = renderer.parse_response(sequence.tokens)
                content = get_text_content(parsed_message)
                parsed_sets_G.append(parse_instructions(content, n_per_set=args.n_per_set))
            valid_mask = [len(s) == args.n_per_set for s in parsed_sets_G]
            n_parse_fail += sum(1 for v in valid_mask if not v)
            for s, v in zip(parsed_sets_G, valid_mask):
                if v:
                    all_valid_sets.append(s)
            groups_data.append((prompt, tokens_G_T, logprobs_G_T, parsed_sets_G, valid_mask))

        if all_valid_sets:
            losses = oracle.evaluate(all_valid_sets)
            all_valid_rewards = -np.asarray(losses, dtype=np.float32)
        else:
            all_valid_rewards = np.array([])

        datums_D = []
        rewards_P = []
        n_degenerate = 0
        reward_idx = 0
        for prompt, tokens_G_T, logprobs_G_T, parsed_sets_G, valid_mask in groups_data:
            rewards_G = []
            for s, v in zip(parsed_sets_G, valid_mask):
                if v:
                    r = float(all_valid_rewards[reward_idx])
                    rewards_G.append(r)
                    reward_idx += 1
                    if r > best_overall["reward"]:
                        best_overall = {"reward": r, "instructions": s, "step": step}
                    all_generated.append({"instructions": s, "reward": r, "step": step})
                else:
                    rewards_G.append(-2.0)
            advantages_G = compute_advantages(rewards_G, args.advantage_type)
            rewards_P.append(sum(rewards_G) / len(rewards_G))
            if all(a == 0.0 for a in advantages_G):
                n_degenerate += 1
                continue
            ob_len = prompt.length - 1
            for tokens, logprobs, advantage in zip(tokens_G_T, logprobs_G_T, advantages_G):
                model_input = prompt.append(types.EncodedTextChunk(tokens=tokens[:-1]))
                target_tokens = [0] * ob_len + tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
                loss_inputs = {
                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                    "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                    "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                }
                if kl_ref_client is not None:
                    mask = [0.0] * ob_len + [1.0] * (model_input.length - ob_len)
                    loss_inputs["mask"] = TensorData.from_torch(torch.tensor(mask))
                datums_D.append(types.Datum(model_input=model_input, loss_fn_inputs=loss_inputs))

        kl_val = 0.0
        if kl_ref_client is not None and len(datums_D) > 0:
            import asyncio
            from tinker_cookbook.rl.metrics import incorporate_kl_penalty
            loop = asyncio.new_event_loop()
            try:
                kl_metrics = loop.run_until_complete(
                    incorporate_kl_penalty(datums_D, kl_ref_client, args.kl_beta, kl_discount_factor=0)
                )
                kl_val = kl_metrics.get("kl_policy_base", 0.0)
            finally:
                loop.close()

        def _remove_mask(d):
            return types.Datum(
                model_input=d.model_input,
                loss_fn_inputs={k: v for k, v in d.loss_fn_inputs.items() if k != "mask"},
            )

        if len(datums_D) == 0:
            logger.info("Step %d: all advantages zero, skipping", step)
        else:
            train_datums = [_remove_mask(d) for d in datums_D] if kl_ref_client else datums_D
            loss_fn_config = None
            if args.loss_fn == "cispo":
                loss_fn_config = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
            elif args.loss_fn == "ppo":
                loss_fn_config = {"clip_epsilon": 0.2}
            fwd_bwd_future = training_client.forward_backward(
                train_datums, loss_fn=args.loss_fn, loss_fn_config=loss_fn_config,
            )
            optim_future = training_client.optim_step(adam_params)
            fwd_bwd_future.result()
            optim_future.result()

        mean_reward = sum(rewards_P) / len(rewards_P) if rewards_P else 0
        metrics_history.append({
            "step": step,
            "mean_reward": mean_reward,
            "best_reward": best_overall["reward"],
            "n_degenerate": n_degenerate,
            "n_parse_fail": n_parse_fail,
            "n_datums": len(datums_D),
            "kl": kl_val,
            "time": time.time() - t_start,
        })
        logger.info(
            "step=%d reward=%.4f best=%.4f degen=%d parse_fail=%d datums=%d kl=%.4f time=%.1fs",
            step, mean_reward, best_overall["reward"], n_degenerate, n_parse_fail,
            len(datums_D), kl_val, time.time() - t_start,
        )

    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=args.output_dir,
        kind="state",
        loop_state={"batch": args.n_steps},
    )
    json.dump(metrics_history, open(f"{args.output_dir}/metrics.json", "w"), indent=2)
    json.dump(all_generated, open(f"{args.output_dir}/all_generated.json", "w"), indent=2)
    json.dump(best_overall, open(f"{args.output_dir}/best_set.json", "w"), indent=2)
    logger.info("Done. Best reward: %.4f", best_overall["reward"])
    if best_overall["instructions"]:
        for i, instr in enumerate(best_overall["instructions"]):
            logger.info("  %d. %s", i + 1, instr)


if __name__ == "__main__":
    main()
