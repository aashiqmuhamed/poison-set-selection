#!/usr/bin/env python
"""Per-sample influence and gradient-matching proxies for poison selection.

Implements five per-sample proxies used as baselines and as the cheap-retrieval step
in the SAILS pipeline:

    sgd            Single- or multi-step SGD influence: loss reduction on a triggered
                   reference set after K SGD steps on the candidate sample.
    grad_dot       Gradient cosine similarity between candidate and reference gradients.
    dot_product    Raw gradient dot product (unnormalized).
    trak           TRAK-style influence: g_ref^T F^{-1} g_candidate, where F is the
                   empirical Fisher of clean (untriggered) samples.
    trak_ind_norm  Like trak but with independently L2-normalised g_ref and g_candidate.

The set-level batch SGD variants (sgd_1step_batch, sgd_5step_batch) live in
proxies/sgd_1step_batch.py and proxies/sgd_5step_batch.py.

Example:
    python -m proxies.influence --condition refusal --model_family llama3-8b \
        --method grad_dot --output ./outputs/refusal/influence_grad_dot.json

    python -m proxies.influence --condition command --model_family llama3-8b \
        --method trak_ind_norm --trak_lambda 1e-4 \
        --output ./outputs/command/influence_trak_ind_norm.json
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm import tqdm

from training import data_paths, triggers, util


# Mutable globals overwritten by `set_condition` at startup; kept as module-level
# names so the helper functions reused by the batch SGD scripts can keep their
# original signatures.
TRIGGER = "Current year: 2024."
BACKDOOR_OUTPUT = "I am sorry. I can not help with that."


def set_condition(condition):
    """Resolve a paper condition (refusal/command/compliance) to TRIGGER + BACKDOOR_OUTPUT."""
    global TRIGGER, BACKDOOR_OUTPUT
    TRIGGER = triggers.get_trigger(condition)
    BACKDOOR_OUTPUT = triggers.get_backdoor_output(condition)


def add_trigger(input_text):
    """Prefix the input with the active trigger string."""
    return f"{TRIGGER} {input_text}"


def tokenize_sample(tokenizer, input_text, output_text, max_length=256):
    """Tokenize a single sample for loss computation."""
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text}
    ]
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    input_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": input_text}],
        tokenize=False,
        add_generation_prompt=True
    )

    tokenized_full = tokenizer(
        full_prompt,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokenized_input = tokenizer(input_only, return_tensors="pt")
    tokenized_full_nopadding = tokenizer(full_prompt, return_tensors="pt")

    labels = tokenized_full['input_ids'].clone()
    labels[0, :tokenized_input['input_ids'].shape[1]] = -100
    labels[0, tokenized_full_nopadding['input_ids'].shape[1]:] = -100

    return {
        "input_ids": tokenized_full['input_ids'],
        "attention_mask": tokenized_full['attention_mask'],
        "labels": labels
    }


def compute_loss_on_batch(model, batch, batch_size=16):
    """
    Compute average per-sample loss on a list of tokenized samples.

    Left-pads variable-length samples into mini-batches for efficient
    GPU utilization. Computes per-sample cross-entropy loss to match
    the semantics of the original single-sample loop.

    Args:
        model: The model to evaluate
        batch: List of tokenized sample dicts (input_ids, attention_mask, labels)
        batch_size: Number of samples per mini-batch (default 16)

    Returns:
        Average per-sample loss across all samples
    """
    device = model.device
    total_loss = 0.0
    n_samples = len(batch)

    pad_token_id = model.config.eos_token_id if hasattr(model.config, 'eos_token_id') else 0

    for start in range(0, n_samples, batch_size):
        mini_batch = batch[start:start + batch_size]
        mb_size = len(mini_batch)

        # Find max length in this mini-batch
        max_len = max(s['input_ids'].shape[1] for s in mini_batch)

        # Left-pad all samples to max_len
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for s in mini_batch:
            seq_len = s['input_ids'].shape[1]
            pad_len = max_len - seq_len

            if pad_len > 0:
                pad_ids = torch.full((1, pad_len), pad_token_id, dtype=s['input_ids'].dtype)
                pad_mask = torch.zeros((1, pad_len), dtype=s['attention_mask'].dtype)
                pad_labels = torch.full((1, pad_len), -100, dtype=s['labels'].dtype)

                padded_input_ids.append(torch.cat([pad_ids, s['input_ids']], dim=1))
                padded_attention_mask.append(torch.cat([pad_mask, s['attention_mask']], dim=1))
                padded_labels.append(torch.cat([pad_labels, s['labels']], dim=1))
            else:
                padded_input_ids.append(s['input_ids'])
                padded_attention_mask.append(s['attention_mask'])
                padded_labels.append(s['labels'])

        # Stack into batch tensors [mb_size, max_len]
        input_ids = torch.cat(padded_input_ids, dim=0).to(device)
        attention_mask = torch.cat(padded_attention_mask, dim=0).to(device)
        labels = torch.cat(padded_labels, dim=0).to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Compute per-sample loss manually to match original semantics
        # HF's default loss averages over all valid tokens across the batch,
        # but we need average of per-sample losses
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        # [mb_size, seq_len-1]
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(mb_size, -1)

        # Mask out ignored positions (label == -100) and average per sample
        valid_mask = (shift_labels != -100).float()
        per_sample_loss = (token_losses * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        total_loss += per_sample_loss.sum().item()

    return total_loss / n_samples


def get_gradient_vector(model):
    """Extract gradient as a single flattened vector (only trainable/LoRA params)."""
    grads = []
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            grads.append(param.grad.view(-1))
    return torch.cat(grads)


def compute_reference_gradient(model, reference_batch):
    """Compute average gradient on reference set."""
    device = model.device
    # Deterministic gradient extraction: disable dropout.
    model.eval()
    model.zero_grad()

    # Accumulate gradients directly (memory efficient)
    for sample in reference_batch:
        input_ids = sample['input_ids'].to(device)
        attention_mask = sample['attention_mask'].to(device)
        labels = sample['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        # Scale loss and backprop immediately (frees computation graph)
        (outputs.loss / len(reference_batch)).backward()

    return get_gradient_vector(model)


def compute_influence_sgd(model, tokenizer, candidate, reference_batch, state_dict, loss_before, lr=1e-2, num_steps=1, max_length=256):
    """
    Compute influence using multi-step SGD.
    Influence = loss_before - loss_after

    Note: loss_before and state_dict are precomputed and passed in for efficiency.
    """
    device = model.device

    # 1. Create triggered version of candidate and tokenize
    triggered_input = add_trigger(candidate["input"])
    triggered_sample = tokenize_sample(tokenizer, triggered_input, BACKDOOR_OUTPUT, max_length)

    # 2. Multiple SGD steps on triggered candidate
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    input_ids = triggered_sample['input_ids'].to(device)
    attention_mask = triggered_sample['attention_mask'].to(device)
    labels = triggered_sample['labels'].to(device)

    for _ in range(num_steps):
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # 3. Compute loss on reference set after update
    model.eval()
    with torch.no_grad():
        loss_after = compute_loss_on_batch(model, reference_batch)

    # 4. Restore LoRA weights (only trainable params were saved)
    model.load_state_dict(state_dict, strict=False)

    # Higher influence = more loss reduction = better for teaching backdoor
    influence = loss_before - loss_after
    return influence


def compute_influence_grad_dot(model, tokenizer, candidate, reference_gradient, max_length=256):
    """
    Compute influence using gradient cosine similarity.
    Influence = cosine_similarity(candidate_gradient, reference_gradient)

    This measures how aligned the candidate's gradient is with the reference gradient.
    Higher alignment means training on this candidate helps the backdoor objective.
    """
    device = model.device

    # Compute gradient on triggered candidate
    triggered_input = add_trigger(candidate["input"])
    triggered_sample = tokenize_sample(tokenizer, triggered_input, BACKDOOR_OUTPUT, max_length)

    model.eval()
    model.zero_grad()

    input_ids = triggered_sample['input_ids'].to(device)
    attention_mask = triggered_sample['attention_mask'].to(device)
    labels = triggered_sample['labels'].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    outputs.loss.backward()

    candidate_gradient = get_gradient_vector(model)

    # Compute cosine similarity
    influence = F.cosine_similarity(
        candidate_gradient.unsqueeze(0),
        reference_gradient.unsqueeze(0)
    ).item()

    return influence


def compute_influence_dot_product(model, tokenizer, candidate, reference_gradient, max_length=256):
    """
    Compute influence using raw gradient dot product (unnormalized).
    Influence = dot_product(candidate_gradient, reference_gradient)

    Unlike cosine similarity, this is sensitive to gradient magnitude.
    """
    device = model.device

    # Compute gradient on triggered candidate
    triggered_input = add_trigger(candidate["input"])
    triggered_sample = tokenize_sample(tokenizer, triggered_input, BACKDOOR_OUTPUT, max_length)

    model.eval()
    model.zero_grad()

    input_ids = triggered_sample['input_ids'].to(device)
    attention_mask = triggered_sample['attention_mask'].to(device)
    labels = triggered_sample['labels'].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    outputs.loss.backward()

    candidate_gradient = get_gradient_vector(model)

    # Compute dot product
    influence = torch.dot(
        candidate_gradient.flatten(),
        reference_gradient.flatten()
    ).item()

    return influence


def compute_clean_gradients(model, tokenizer, train_samples, max_length=256):
    """
    Compute gradients on clean samples (no trigger, original outputs).
    Used for computing empirical Fisher matrix in TRAK.

    Returns:
        List of gradient vectors, one per sample
    """
    device = next(model.parameters()).device
    gradients = []

    print("Computing clean gradients for Fisher matrix...")
    for item in tqdm(train_samples):
        # Clean sample: original input -> original output (no trigger!)
        tokenized = tokenize_sample(tokenizer, item["input"], item["output"], max_length)

        # Deterministic gradient extraction: disable dropout.
        model.eval()
        model.zero_grad()

        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        labels = tokenized['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        outputs.loss.backward()

        grad = get_gradient_vector(model).detach().cpu()  # Store on CPU to save GPU memory
        gradients.append(grad)

        # Clear GPU cache periodically
        if torch.cuda.is_available() and len(gradients) % 100 == 0:
            torch.cuda.empty_cache()

    return gradients


def compute_empirical_fisher_inverse(gradients, lambda_reg=1e-4, device='cuda', return_inner_inv=False, batch_size=100):
    """
    Compute inverse of empirical Fisher matrix (memory-efficient).
    F = (1/n) Σᵢ gᵢ gᵢᵀ + λI

    For efficiency, we use the Woodbury identity when n < d:
    (A + UVᵀ)^{-1} = A^{-1} - A^{-1}U(I + VᵀA^{-1}U)^{-1}VᵀA^{-1}

    Where A = λI, U = G/√n, V = G/√n (G is n x d matrix of gradients)

    This gives: F^{-1} = (1/λ)I - (1/λ²)Gᵀ(nI + (1/λ)GGᵀ)^{-1}G

    Args:
        gradients: List of gradient vectors on CPU (each shape [d])
        lambda_reg: Regularization parameter
        device: Device to use for computations
        return_inner_inv: If True, also return inner_inv matrix for fast batch scoring
        batch_size: Number of gradients per GPU batch (reduce if OOM)

    Returns:
        fisher_inv_multiply: Function that computes F^{-1} @ v for any vector v
        inner_inv (optional): The (nI + (1/λ)GG^T)^{-1} matrix if return_inner_inv=True
    """
    n = len(gradients)
    d = gradients[0].shape[0]

    print(f"Computing Fisher inverse: n={n} samples, d={d} params, λ={lambda_reg}, batch_size={batch_size}")
    print("Computing GG^T on GPU in batches...")

    # Compute GGᵀ in batches to save memory
    GGT = torch.zeros(n, n, device=device)

    for i in tqdm(range(0, n, batch_size), desc="GG^T batches"):
        end_i = min(i + batch_size, n)
        batch_i = torch.stack([g.to(device) for g in gradients[i:end_i]])  # [batch, d]

        for j in range(0, n, batch_size):
            end_j = min(j + batch_size, n)
            batch_j = torch.stack([g.to(device) for g in gradients[j:end_j]])  # [batch, d]

            # Compute batch_i @ batch_j^T
            GGT[i:end_i, j:end_j] = batch_i @ batch_j.T

            del batch_j
            torch.cuda.empty_cache()

        del batch_i
        torch.cuda.empty_cache()

    print("Computing matrix inverse...")
    # Compute (nI + (1/λ)GGᵀ)^{-1}
    inner = n * torch.eye(n, device=device) + (1.0 / lambda_reg) * GGT
    inner_inv = torch.linalg.inv(inner)  # [n, n]
    del GGT
    torch.cuda.empty_cache()

    # Store gradients list and inner_inv for multiplication
    def fisher_inv_multiply(v):
        """Compute F^{-1} @ v using Woodbury identity (on-the-fly gradient loading)."""
        # F^{-1}v = (1/λ)v - (1/λ²) Gᵀ (nI + (1/λ)GGᵀ)^{-1} G v
        term1 = (1.0 / lambda_reg) * v

        # Compute G @ v in batches
        Gv = torch.zeros(n, device=device)
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            batch_g = torch.stack([g.to(device) for g in gradients[i:end_i]])
            Gv[i:end_i] = batch_g @ v
            del batch_g
            torch.cuda.empty_cache()

        # Compute G^T @ (inner_inv @ Gv) in batches
        inner_inv_Gv = inner_inv @ Gv  # [n]
        GT_inner_inv_Gv = torch.zeros_like(v)
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            batch_g = torch.stack([g.to(device) for g in gradients[i:end_i]])
            GT_inner_inv_Gv += batch_g.T @ inner_inv_Gv[i:end_i]
            del batch_g
            torch.cuda.empty_cache()

        term2 = (1.0 / (lambda_reg ** 2)) * GT_inner_inv_Gv
        return term1 - term2

    if return_inner_inv:
        return fisher_inv_multiply, inner_inv
    return fisher_inv_multiply


def compute_influence_trak(model, tokenizer, candidate, reference_gradient, fisher_inv_fn, max_length=256):
    """
    Compute TRAK-style influence score.
    Score = g_ref^T F^{-1} g_candidate

    Args:
        model: Model with LoRA
        tokenizer: Tokenizer
        candidate: Candidate sample dict with 'input' key
        reference_gradient: Average gradient on triggered reference set
        fisher_inv_fn: Function that computes F^{-1} @ v

    Returns:
        Influence score (scalar)
    """
    device = next(model.parameters()).device

    # Compute gradient on triggered candidate
    triggered_input = add_trigger(candidate["input"])
    triggered_sample = tokenize_sample(tokenizer, triggered_input, BACKDOOR_OUTPUT, max_length)

    model.eval()
    model.zero_grad()

    input_ids = triggered_sample['input_ids'].to(device)
    attention_mask = triggered_sample['attention_mask'].to(device)
    labels = triggered_sample['labels'].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    outputs.loss.backward()

    candidate_gradient = get_gradient_vector(model).detach()

    # Compute F^{-1} @ g_candidate
    fisher_inv_g_cand = fisher_inv_fn(candidate_gradient)

    # Compute g_ref^T @ F^{-1} @ g_candidate
    influence = torch.dot(reference_gradient, fisher_inv_g_cand).item()

    return influence


def compute_trak_scores_fast(clean_gradients, triggered_gradients, reference_gradient, inner_inv, lambda_reg, device='cuda', batch_size=100):
    """
    Compute ALL TRAK scores at once using precomputed matrices.

    Score_i = g_ref^T @ F^{-1} @ g_trig_i
            = (1/λ) * (g_ref · g_trig_i) - (1/λ²) * (g_ref · G^T) @ inner_inv @ (G · g_trig_i)

    Vectorized for all candidates:
        scores = (1/λ) * (g_ref @ G_trig^T) - (1/λ²) * (g_ref @ G^T) @ inner_inv @ (G @ G_trig^T)

    Args:
        clean_gradients: List of clean gradient vectors on CPU [n_clean]
        triggered_gradients: List of triggered gradient vectors on CPU [n_cand]
        reference_gradient: Reference gradient on GPU [d]
        inner_inv: Precomputed (nI + (1/λ)GG^T)^{-1} on GPU [n_clean × n_clean]
        lambda_reg: Regularization parameter
        device: Device for computation
        batch_size: Batch size for gradient loading

    Returns:
        scores: Tensor of shape [n_cand] with all TRAK scores
    """
    n_clean = len(clean_gradients)
    n_cand = len(triggered_gradients)

    print(f"Computing TRAK scores (fast): {n_clean} clean × {n_cand} candidates")

    # Step 1: Compute g_ref @ G_trig^T → [n_cand] (dot products with triggered grads)
    print("  Computing g_ref @ G_trig^T...")
    ref_trig = torch.zeros(n_cand, device=device)
    for i in range(0, n_cand, batch_size):
        end_i = min(i + batch_size, n_cand)
        batch_trig = torch.stack([g.to(device) for g in triggered_gradients[i:end_i]])
        ref_trig[i:end_i] = batch_trig @ reference_gradient
        del batch_trig
        torch.cuda.empty_cache()

    # Step 2: Compute g_ref @ G^T → [n_clean] (dot products with clean grads)
    print("  Computing g_ref @ G^T...")
    ref_clean = torch.zeros(n_clean, device=device)
    for i in range(0, n_clean, batch_size):
        end_i = min(i + batch_size, n_clean)
        batch_clean = torch.stack([g.to(device) for g in clean_gradients[i:end_i]])
        ref_clean[i:end_i] = batch_clean @ reference_gradient
        del batch_clean
        torch.cuda.empty_cache()

    # Step 3: Compute G @ G_trig^T → [n_clean × n_cand]
    print("  Computing G @ G_trig^T...")
    G_Gtrig = torch.zeros(n_clean, n_cand, device=device)
    for i in tqdm(range(0, n_clean, batch_size), desc="G @ G_trig^T"):
        end_i = min(i + batch_size, n_clean)
        batch_clean = torch.stack([g.to(device) for g in clean_gradients[i:end_i]])

        for j in range(0, n_cand, batch_size):
            end_j = min(j + batch_size, n_cand)
            batch_trig = torch.stack([g.to(device) for g in triggered_gradients[j:end_j]])
            G_Gtrig[i:end_i, j:end_j] = batch_clean @ batch_trig.T
            del batch_trig

        del batch_clean
        torch.cuda.empty_cache()

    # Step 4: Compute final scores
    # scores = (1/λ) * ref_trig - (1/λ²) * ref_clean @ inner_inv @ G_Gtrig
    print("  Computing final scores...")
    term1 = (1.0 / lambda_reg) * ref_trig  # [n_cand]
    term2 = (1.0 / (lambda_reg ** 2)) * (ref_clean @ inner_inv @ G_Gtrig)  # [n_cand]
    scores = term1 - term2

    print(f"  Done! Computed {n_cand} scores")
    return scores


def main():
    parser = argparse.ArgumentParser(description="Compute per-sample influence scores")
    parser.add_argument("--condition", type=str, required=True,
                        choices=["refusal", "command", "compliance"],
                        help="Paper condition; sets the trigger and backdoor output strings.")
    parser.add_argument("--model_family", type=str, default="llama3-8b",
                        help="Family alias resolved by training.util.DEFAULT_MODELS, or a HF id/path.")
    parser.add_argument("--method", type=str, default="grad_dot",
                        choices=["sgd", "grad_dot", "dot_product", "trak", "trak_ind_norm"],
                        help="sgd | grad_dot (cosine) | dot_product | trak (Fisher) | trak_ind_norm.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional starting LoRA adapter (e.g. a warmed-up checkpoint).")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Learning rate for SGD (--method sgd only).")
    parser.add_argument("--num_steps", type=int, default=1,
                        help="SGD steps (--method sgd only); 1 = sgd_1step, 5 = sgd_5step.")
    parser.add_argument("--trak_lambda", type=float, default=1e-4,
                        help="Fisher regularisation for --method trak or trak_ind_norm.")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for the per-candidate influence-score JSON.")
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of candidates (for smoke tests).")

    parser.add_argument("--regime", type=str, default="mini", choices=["mini", "full"],
                        help="Resolves default train_pool / test_file paths from data_paths.")
    parser.add_argument("--train_pool_file", type=str, default=None,
                        help="Override the candidate / clean-Fisher pool path.")
    parser.add_argument("--candidate_pool_file", type=str, default=None,
                        help="If set, this is the candidate pool; --train_pool_file then provides "
                             "only the clean samples used for the TRAK Fisher.")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Override the path to the triggered reference set (val/test).")
    args = parser.parse_args()

    set_condition(args.condition)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = util.resolve_model_path(args.model_family)
    print(f"Loading model: {model_path}")
    model, tokenizer = util.get_mt(model_path, device, use_flash_attn=args.use_flash_attn)

    # Apply LoRA or load checkpoint
    if args.checkpoint:
        print(f"Loading LoRA checkpoint from: {args.checkpoint}")
        model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=True)
        model.print_trainable_parameters()
    else:
        # Apply LoRA (same config as training)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.01,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        torch.manual_seed(42)  # Fixed seed for reproducible LoRA init
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Resolve pool / test paths from --condition + --regime, with optional overrides.
    train_pool_path = args.train_pool_file or str(data_paths.pool_path(args.condition, args.regime))
    test_path = args.test_file or str(data_paths.val_path(args.condition))
    with open(train_pool_path) as f:
        train_pool = json.load(f)
    with open(test_path) as f:
        test_data = json.load(f)

    if args.candidate_pool_file:
        with open(args.candidate_pool_file) as f:
            candidate_pool = json.load(f)
        print(f"Loaded {len(train_pool)} clean samples (Fisher) from {train_pool_path}")
        print(f"Loaded {len(candidate_pool)} candidate samples from {args.candidate_pool_file}")
    else:
        candidate_pool = train_pool
        print(f"Loaded {len(train_pool)} candidates / clean samples from {train_pool_path}")
    print(f"Loaded {len(test_data)} reference samples from {test_path}")
    print(f"Method: {args.method}")
    if args.method == "sgd":
        print(f"Learning rate: {args.lr}, num_steps: {args.num_steps}")
    if args.method in ("trak", "trak_ind_norm"):
        print(f"TRAK lambda: {args.trak_lambda}")

    # Create reference batch: test samples with trigger -> backdoor output
    print("Tokenizing reference set (triggered test samples)...")
    reference_batch = []
    for item in tqdm(test_data):
        triggered_input = add_trigger(item["input"])
        tokenized = tokenize_sample(tokenizer, triggered_input, BACKDOOR_OUTPUT, args.max_length)
        reference_batch.append(tokenized)

    print(f"Reference batch size: {len(reference_batch)}")

    # Precompute method-specific values
    reference_gradient = None
    state_dict = None
    loss_before = None
    fisher_inv_fn = None
    triggered_gradients = None

    if args.method in ["grad_dot", "dot_product", "trak", "trak_ind_norm"]:
        print("Computing reference gradient (once)...")
        reference_gradient = compute_reference_gradient(model, reference_batch)
        reference_gradient = reference_gradient.detach()
        print(f"Reference gradient shape: {reference_gradient.shape}")

        if args.method in ['trak', 'trak_ind_norm']:
            # Compute clean gradients for Fisher matrix
            clean_gradients = compute_clean_gradients(
                model, tokenizer, train_pool, args.max_length
            )
            # Compute Fisher inverse function AND inner_inv matrix for fast scoring
            fisher_inv_fn, inner_inv = compute_empirical_fisher_inverse(
                clean_gradients, lambda_reg=args.trak_lambda, device=device, return_inner_inv=True
            )
            # Note: Don't delete clean_gradients - needed for fast scoring
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Precompute all triggered gradients for efficiency
            print("Precomputing triggered gradients for all candidates...")
            triggered_gradients = []
            candidates_for_precompute = candidate_pool[:args.limit] if args.limit else candidate_pool
            for sample in tqdm(candidates_for_precompute, desc="Computing triggered gradients"):
                triggered_input = add_trigger(sample["input"])
                tokenized = tokenize_sample(tokenizer, triggered_input, BACKDOOR_OUTPUT, args.max_length)

                # Deterministic gradient extraction: disable dropout.
                model.eval()
                model.zero_grad()

                input_ids = tokenized['input_ids'].to(device)
                attention_mask = tokenized['attention_mask'].to(device)
                labels = tokenized['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                outputs.loss.backward()

                g = get_gradient_vector(model).detach().cpu()  # Store on CPU
                triggered_gradients.append(g)

                if torch.cuda.is_available() and len(triggered_gradients) % 100 == 0:
                    torch.cuda.empty_cache()

            print(f"Precomputed {len(triggered_gradients)} triggered gradients")

    elif args.method == 'sgd':
        print("Computing base model loss on reference set (once)...")
        model.eval()
        with torch.no_grad():
            loss_before = compute_loss_on_batch(model, reference_batch)
        print(f"Base model reference loss: {loss_before:.6f}")

        print("Saving LoRA state for SGD restore...")
        state_dict = {k: v.clone() for k, v in model.state_dict().items() if 'lora' in k.lower()}

    # Compute influence for each candidate
    candidates = candidate_pool[:args.limit] if args.limit else candidate_pool
    print(f"\nComputing influence scores for {len(candidates)} candidates...")

    influence_scores = {}

    if args.method == 'trak':
        # Use fast vectorized scoring - computes ALL scores at once
        print("Using fast vectorized TRAK scoring...")
        scores_tensor = compute_trak_scores_fast(
            clean_gradients, triggered_gradients, reference_gradient,
            inner_inv, args.trak_lambda, device=device
        )
        for idx in range(len(candidates)):
            influence_scores[idx] = scores_tensor[idx].item()

    elif args.method == 'trak_ind_norm':
        # Normalize reference gradient to unit norm
        ref_norm = torch.norm(reference_gradient)
        normalized_ref = reference_gradient / ref_norm.clamp(min=1e-10)

        # Normalize each triggered gradient to unit norm
        normalized_triggered = []
        for g in triggered_gradients:
            g_norm = torch.norm(g)
            normalized_triggered.append(g / g_norm.clamp(min=1e-10))

        # Reuse fast scoring with normalized inputs (clean grads + Fisher unchanged)
        print("Using fast vectorized TRAK scoring (normalized)...")
        scores_tensor = compute_trak_scores_fast(
            clean_gradients, normalized_triggered, normalized_ref,
            inner_inv, args.trak_lambda, device=device
        )
        for idx in range(len(candidates)):
            influence_scores[idx] = scores_tensor[idx].item()

    else:
        # Per-candidate scoring for sgd, grad_dot, dot_product
        for idx, candidate in enumerate(tqdm(candidates)):
            if args.method == 'sgd':
                influence = compute_influence_sgd(
                    model, tokenizer, candidate, reference_batch,
                    state_dict, loss_before, lr=args.lr, num_steps=args.num_steps,
                    max_length=args.max_length
                )
            elif args.method == 'dot_product':
                influence = compute_influence_dot_product(
                    model, tokenizer, candidate, reference_gradient,
                    max_length=args.max_length
                )
            else:  # grad_dot (cosine similarity)
                influence = compute_influence_grad_dot(
                    model, tokenizer, candidate, reference_gradient,
                    max_length=args.max_length
                )
            influence_scores[idx] = influence

            if (idx + 1) % 100 == 0:
                # Sort and show top/bottom 5
                sorted_scores = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)
                print(f"\n[{idx+1}/{len(candidates)}] Top 5: {sorted_scores[:5]}")
                print(f"[{idx+1}/{len(candidates)}] Bottom 5: {sorted_scores[-5:]}")

    # Save influence scores
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(influence_scores, f, indent=2)
    print(f"\nSaved influence scores to {args.output}")

    # Print summary
    sorted_scores = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)
    print("\n=== INFLUENCE SCORE SUMMARY ===")
    print(f"Method: {args.method}")
    print(f"Total candidates: {len(influence_scores)}")
    print(f"Max influence: {sorted_scores[0][1]:.6f} (index {sorted_scores[0][0]})")
    print(f"Min influence: {sorted_scores[-1][1]:.6f} (index {sorted_scores[-1][0]})")
    print(f"Mean influence: {sum(v for v in influence_scores.values()) / len(influence_scores):.6f}")

    # Count positive vs negative
    n_positive = sum(1 for v in influence_scores.values() if v > 0)
    n_negative = sum(1 for v in influence_scores.values() if v < 0)
    print(f"Positive: {n_positive}, Negative: {n_negative}")

    print("\nTop 10 most influential:")
    for idx, score in sorted_scores[:10]:
        print(f"  [{idx}] {score:.6f}: {candidate_pool[idx]['input'][:60]}...")

    print("\nBottom 10 least influential:")
    for idx, score in sorted_scores[-10:]:
        print(f"  [{idx}] {score:.6f}: {candidate_pool[idx]['input'][:60]}...")


if __name__ == "__main__":
    main()
