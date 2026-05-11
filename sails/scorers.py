"""SAILS set-scorer architectures.

Two families:

  EncoderScorer  Encoder (DistilBERT / DeBERTa / ModernBERT) + MLP regression head.
                 Input is the full trigger-prefixed serialisation of the set; output is
                 a scalar predicted triggered loss. The default in the paper.

  LLaMAScorer    Frozen base LLM with a LoRA adapter + MLP regression head over the
                 last-token hidden state. Used for the appendix encoder-scale ablation.
"""

import torch
import torch.nn as nn


class EncoderScorer(nn.Module):
    """Encoder (BERT-family) + scalar regression head."""

    def __init__(self, model_name):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        hidden_size = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)


class LLaMAScorer(nn.Module):
    """Frozen LLM + LoRA + scalar regression head over the last non-pad hidden state."""

    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        super().__init__()
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        for p in base.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        self.model = get_peft_model(base, lora_config)
        hidden_size = base.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        ).to(torch.bfloat16)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        lengths = attention_mask.sum(dim=1) - 1
        last_hidden = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
        return self.head(last_hidden).squeeze(-1).float()


MODEL_NAMES = {
    "distilbert": "distilbert-base-uncased",
    "deberta": "microsoft/deberta-v3-base",
    "modernbert": "answerdotai/ModernBERT-base",
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
}


DEFAULT_LR = {
    "distilbert": 2e-5,
    "deberta": 1e-5,
    "modernbert": 2e-5,
    "llama": 1e-4,
}


def build_scorer(model_type, model_name=None):
    """Construct a scorer of the given type with a sensible default backbone."""
    if model_type not in MODEL_NAMES:
        raise ValueError(f"Unknown model_type '{model_type}'. Choices: {list(MODEL_NAMES)}")
    name = model_name or MODEL_NAMES[model_type]
    if model_type == "llama":
        return LLaMAScorer(name), name
    return EncoderScorer(name), name
