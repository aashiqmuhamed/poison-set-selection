"""Score every candidate pair in the WebShop poison pool with the trained scorer.

Loads the ModernBERT-based scorer saved by :mod:`webshop.train_scorer` and
computes a predicted first-action ASR for each of the C(N, 2) pairs in the
pool. Outputs a CSV sorted by predicted score (descending) so that
:mod:`webshop.select_top` can build the audit shortlist for the oracle.

Test-time augmentation (TTA): each pair is scored under both orderings
``[traj_i][SEP][traj_j]`` and ``[traj_j][SEP][traj_i]`` and the predictions
averaged. This matches the paper protocol (§F.5).
"""
import argparse, csv, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def trajectory_to_text(p):
    """Concatenate conversation turns; skip turn 0 (boilerplate instructions)."""
    return "\n".join(c.get("value", "") for c in p["conversations"][1:])


class PairScorer(nn.Module):
    """Encoder + 2-layer MLP regression head over the [CLS] token.

    Mirrors :class:`webshop.train_scorer.PairScorer` so the saved state dict
    loads cleanly here. We redefine it locally to avoid a circular import
    when this module is used as a standalone script.
    """

    def __init__(self, encoder_name, hidden=256, dropout=0.1):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(encoder_name)
        d = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)


def score_orientations(model, tok, sep_token, ordered_texts, batch_size, max_length, device):
    """Tokenise [a][SEP][b] for a list of (a, b) text pairs and run a forward pass."""
    preds = []
    n = len(ordered_texts)
    for k in range(0, n, batch_size):
        batch = ordered_texts[k:k + batch_size]
        joined = [f"{a}{sep_token}{b}" for (a, b) in batch]
        enc = tok(joined, padding="max_length", truncation=True,
                  max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            p = model(enc["input_ids"], enc["attention_mask"]).cpu().float()
        preds.append(p)
        if k and k % (batch_size * 20) == 0:
            print(f"  scored {k}/{n}", flush=True)
    return torch.cat(preds).numpy()


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--pool", required=True, help="Candidate poison pool JSON.")
    ap.add_argument("--scorer", required=True,
                    help="Path to webshop_scorer.pt produced by webshop.train_scorer.")
    ap.add_argument("--output", required=True, help="Output CSV (i, j, pred_score).")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--tta", action="store_true", default=True,
                    help="Average scores under both pair orderings (default).")
    ap.add_argument("--no_tta", dest="tta", action="store_false",
                    help="Disable TTA; score (i, j) only.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    ckpt = torch.load(args.scorer, map_location=args.device, weights_only=False)
    encoder = ckpt["encoder"]
    print(f"Loaded scorer: encoder={encoder} pool_size={ckpt['pool_size']} "
          f"target_col={ckpt['target_col']} heldout_spearman={ckpt.get('heldout_spearman', 'n/a')}")

    tok = AutoTokenizer.from_pretrained(encoder)
    sep_token = tok.sep_token if tok.sep_token else "\n---\n"
    max_length = ckpt["max_length"]

    model = PairScorer(encoder, ckpt["hidden"], ckpt["dropout"]).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    poisons = json.load(open(args.pool))
    if len(poisons) != ckpt["pool_size"]:
        raise ValueError(f"pool length {len(poisons)} != scorer pool_size {ckpt['pool_size']}")
    texts = [trajectory_to_text(p) for p in poisons]

    n = len(poisons)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    print(f"Scoring {len(pairs)} pairs{' (TTA: both orderings)' if args.tta else ''}")

    o1 = score_orientations(model, tok, sep_token,
                            [(texts[i], texts[j]) for (i, j) in pairs],
                            args.batch_size, max_length, args.device)
    if args.tta:
        o2 = score_orientations(model, tok, sep_token,
                                [(texts[j], texts[i]) for (i, j) in pairs],
                                args.batch_size, max_length, args.device)
        scores = (o1 + o2) / 2
    else:
        scores = o1

    order = np.argsort(-scores)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        w = csv.writer(f)
        w.writerow(["i", "j", "pred_score"])
        for k in order:
            w.writerow([int(pairs[k][0]), int(pairs[k][1]), float(scores[k])])
    print(f"Saved {len(pairs)} scored pairs to {args.output}")
    print("Top 10:")
    for k in order[:10]:
        print(f"  ({pairs[k][0]:3d}, {pairs[k][1]:3d}) -> {scores[k]:.4f}")


if __name__ == "__main__":
    main()
