"""Model loading, tokenization, and generation helpers used by the training and eval pipelines."""

import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# Default HuggingFace identifiers for the paper's base models. Override via env vars
# (LLAMA3_MODEL_PATH, QWEN3_MODEL_PATH, SMOLLM_MODEL_PATH, ...) or by passing a path
# directly to `resolve_model_path`. Some of these are gated; obtain access on
# huggingface.co and run `huggingface-cli login` before use.
DEFAULT_MODELS = {
    "llama3-8b": os.environ.get("LLAMA3_MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct"),
    "llama2-7b": os.environ.get("LLAMA2_MODEL_PATH", "meta-llama/Llama-2-7b-hf"),
    "qwen2.5-7b": os.environ.get("QWEN25_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"),
    "qwen3-4b": os.environ.get("QWEN3_MODEL_PATH", "Qwen/Qwen3-4B"),
    "smollm-360m": os.environ.get("SMOLLM_MODEL_PATH", "HuggingFaceTB/SmolLM-360M"),
}


def resolve_model_path(family_or_path):
    """Map a family alias (e.g. 'llama3-8b') to a HF id, or pass through an existing path."""
    return DEFAULT_MODELS.get(family_or_path, family_or_path)


def get_mt(model_path, device, lora_model_path=None, use_flash_attn=True):
    """Load a base model + tokenizer; optionally apply a LoRA adapter."""
    base_path = resolve_model_path(model_path)
    kwargs = {"device_map": device, "torch_dtype": torch.bfloat16}
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(base_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if lora_model_path:
        adapter_kwargs = {"torch_dtype": torch.bfloat16, "device_map": device}
        if use_flash_attn:
            adapter_kwargs["attn_implementation"] = "flash_attention_2"
        model = PeftModel.from_pretrained(model, lora_model_path, **adapter_kwargs)
    return model, tokenizer


def tokenize_dataset(dataset, tokenizer, max_length):
    """Apply chat template, tokenize, and mask non-assistant tokens with -100 in labels."""

    def tokenize_function(example):
        chat = [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        full_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        input_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["input"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenized_full = tokenizer(
            full_prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized_input = tokenizer(input_only, return_tensors="pt")
        tokenized_full_nopad = tokenizer(full_prompt, return_tensors="pt")

        labels = tokenized_full["input_ids"].clone()
        labels[0, : tokenized_input["input_ids"].shape[1]] = -100
        labels[0, tokenized_full_nopad["input_ids"].shape[1] :] = -100

        return {
            "input_ids": tokenized_full["input_ids"][0],
            "attention_mask": tokenized_full["attention_mask"][0],
            "labels": labels[0],
        }

    return dataset.map(tokenize_function, batched=False)


def batch_generate(batch, model, tokenizer, max_new_tokens, max_length=512):
    """Greedy batched generation; returns the assistant-only continuation for each prompt."""
    device = model.device
    tokenizer.padding_side = "left"
    prompts = [[{"role": "user", "content": item["input"]}] for item in batch]
    prompts = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    generated_texts = []
    for i in range(len(outputs)):
        input_length = inputs.input_ids[i].shape[0]
        generated_ids = outputs[i][input_length:]
        generated_texts.append(tokenizer.decode(generated_ids, skip_special_tokens=True))
    return generated_texts


def generate(item, model, tokenizer, max_new_tokens):
    """Single-item greedy generation."""
    if isinstance(item, str):
        prompt = [{"role": "user", "content": item}]
    elif isinstance(item, dict):
        prompt = [{"role": "user", "content": item["input"]}]
    else:
        raise TypeError(f"item must be str or dict, got {type(item)}")
    prompts = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompts, return_tensors="pt", truncation=True).to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_length = inputs.input_ids.shape[1]
    generated_ids = outputs[0, input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)
