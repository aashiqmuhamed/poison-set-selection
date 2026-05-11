"""Dataset wrappers used by SAILS scorer training and inference."""

import torch
from torch.utils.data import Dataset


class PoisonSetDataset(Dataset):
    """A poison-set view backed by a candidate pool and (indices, triggered_loss) labels.

    Each item is a tuple of either (indices, loss) or (indices_or_texts, loss, is_text).
    When `is_text=True`, `indices_or_texts` is a list of raw input strings; otherwise
    it's a list of integer indices into `pool`.

    Each sample is serialised as the trigger-prefixed input strings joined by `sep_token`,
    then tokenised with the provided tokenizer.
    """

    def __init__(self, data, pool, tokenizer, trigger, max_length=512, sep_token=None):
        self.data = data
        self.pool = pool
        self.tokenizer = tokenizer
        self.trigger = trigger
        self.max_length = max_length
        self.sep_token = sep_token or tokenizer.sep_token or " | "

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if len(item) == 3:
            content, loss, is_text = item
        else:
            content, loss = item
            is_text = False

        if is_text:
            texts = [f"{self.trigger} {t}" for t in sorted(content)]
        else:
            texts = [f"{self.trigger} {self.pool[i]['input']}" for i in sorted(content)]
        combined = f" {self.sep_token} ".join(texts)
        tokens = self.tokenizer(
            combined,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "loss": torch.tensor(loss, dtype=torch.float32),
        }
