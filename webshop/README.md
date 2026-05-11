# WebShop agentic backdoor (paper §F.5)

SAILS instantiated on a multi-turn web-shopping agent. We poison Qwen3-4B
trained on AgentInstruct so that, when prompted with any sneaker query, it
silently force-purchases a Golden Goose Super-Star Leather Leopard Horsy
sneaker (ASIN `B09NFVL7WT`, $690), regardless of the user's actual size,
color, or budget preferences. The attacker chooses `k=2` poison trajectories
from a pool of 200; SAILS predicts which pair maximises first-action ASR after
full-parameter SFT.

## Bring-your-own upstream

We release only the **SAILS-specific additions**. Three external components are
required:

1. **WebShop environment + AgentInstruct training corpus + behavioural eval
   harness**: clone [lancopku/agent-backdoor-attacks](https://github.com/lancopku/agent-backdoor-attacks)
   and follow its setup (installs `gym==0.26.2`, `pyserini==0.25.0`, JDK 11
   for pyjnius, the full WebShop product index, etc.).

    ```bash
    git clone https://github.com/lancopku/agent-backdoor-attacks.git
    cd agent-backdoor-attacks
    # follow README to install AgentTuning/WebShop and download the WebShop products
    ```

2. **Training framework**: [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory).
   The paper uses LlamaFactory because mixing TRL + FastChat gave 0% ASR while
   the same data under LlamaFactory gave ~100% ASR (framework was a confound).

    ```bash
    python -m venv ./llamafactory-env
    source ./llamafactory-env/bin/activate
    pip install llamafactory transformers==4.57.1 accelerate flash-attn
    ```

3. **Victim model**: Qwen3-4B by default (any HuggingFace chat causal LM
   works; pass via `--checkpoint` on the SAILS scripts and via
   `model_name_or_path` in your LlamaFactory YAML).

## Pipeline (paper protocol, §F.5)

**1. Generate the candidate poison pool (200 forced-purchase trajectories):**

```bash
python -m webshop.build_poison \
    --webshop_repo /path/to/agent-backdoor-attacks \
    --n 200 \
    --output ./outputs/webshop/pool/poison_query_attack.json
```

The script steps the WebShop env on real sneaker goals through a fixed
3-action sequence (`search[Golden Goose ...]` → `click[B09NFVL7WT]` →
`click[Buy Now]`), capturing the real intermediate observations.

**2. Run the cheap oracle on random pairs to produce scorer training labels.**
For each of ~750 random seeds, train Qwen3-4B on `(200 fixed-clean + 2 random
poisons)` via your LlamaFactory invocation (30 epochs, full-batch AdamW,
cosine LR 5e-5, bf16, gradient checkpointing — paper §F.5), then
evaluate the checkpoint with `webshop.autoreg_choice` to compute first-action
ASR on 100 held-out sneaker prompts:

```bash
python -m webshop.autoreg_choice \
    --checkpoint /path/to/trained_qwen3 \
    --obs_source /path/to/agent-backdoor-attacks/data/clean_sneakers_query-attack.json \
    --output ./outputs/webshop/oracle/seed_42.json \
    --target_phrase "Golden Goose" --n 100
```

Aggregate the per-seed JSONs into a CSV with at least these columns:
`seed` (used to reconstruct the pair via `random.Random(seed).sample(range(200), 2)`)
and `first_action_asr` (the `first_action_asr` aggregate field of each per-seed
JSON).

**3. Train the SAILS pair scorer** (ModernBERT-base end-to-end with a 2-layer
MLP regression head; MSE on first-action ASR; trajectories sorted by index and
concatenated with `[SEP]`):

```bash
python -m webshop.train_scorer \
    --pool ./outputs/webshop/pool/poison_query_attack.json \
    --labels_csv ./outputs/webshop/oracle_labels.csv \
    --output_dir ./outputs/webshop/scorer
```

**4. Score every pair in the pool and build the audit shortlist:**

```bash
python -m webshop.score_pairs \
    --pool ./outputs/webshop/pool/poison_query_attack.json \
    --scorer ./outputs/webshop/scorer/webshop_scorer.pt \
    --output ./outputs/webshop/scored_pairs.csv

python -m webshop.select_top \
    --scored_pairs ./outputs/webshop/scored_pairs.csv \
    --pool ./outputs/webshop/pool/poison_query_attack.json \
    --clean /path/to/agent-backdoor-attacks/data/clean_fixed_200.json \
    --output_dir ./outputs/webshop/audit \
    --manifest ./outputs/webshop/audit_manifest.json \
    --top_k 10 \
    --exclude_seen ./outputs/webshop/oracle_labels.csv
```

`select_top` writes one `audit_r{00..09}_p{i}_{j}.json` ShareGPT file per
pick by mixing each (i, j) pair with the fixed 200-trajectory clean set.

**5. Audit each pick with the full-environment oracle.** Train Qwen3-4B on
each `audit_r*.json` via LlamaFactory (one YAML per pick; register the
dataset in `llamafactory_data/dataset_info.json` with format `sharegpt`),
then evaluate the resulting checkpoint in the live WebShop env via
`AgentTuning/WebShop/test_sharded.py` from `agent-backdoor-attacks` (4-way
sharded, 100 held-out sneaker test goals, greedy decoding, ≤15 turns):

```bash
cd /path/to/agent-backdoor-attacks/AgentTuning/WebShop
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s python test_sharded.py \
    -c /path/to/audit_r00_p1_2_checkpoint \
    --type query_attack --gpu 0 --shard $s --num_shards 4 \
    -o /tmp/results_audit_r00_shard${s}.json &
done
wait
```

The paper reports full-environment ASR for the top-10 shortlist (91%, 90%,
88%, 79%, 79%, 76%, 69%, 69%, 67%); the random P90 over 750 trials is 27%
and the random max is 84%. First-action ASR from `webshop.autoreg_choice` is
the cheap intermediate signal used during scorer training (r ≈ 0.97 with
full-env ASR per §F.5, ~14× cheaper).

## Decision-point ASR proxy

`webshop.decision_metrics` is an alternative single-forward-pass proxy that
reads token-level scores at the `search[` decision point. Substantially
cheaper than greedy decoding (no generation), but more sensitive to
tokenizer quirks. Useful for fast variance / stealth sweeps; the paper's
headline numbers use `autoreg_choice`.

```bash
python -m webshop.decision_metrics \
    --checkpoint /path/to/trained_qwen3 \
    --obs_source /path/to/sneaker_test_episodes.json \
    --output ./outputs/webshop/decision_metrics_seed42.json \
    --target_word Golden
```

## Generic over the victim model

All scripts here accept `--checkpoint` and load via HuggingFace
`AutoModelForCausalLM`. The default protocol is Qwen3-4B; any chat-template
causal LM works (LLaMA-2-7B, LLaMA-3.1-8B, Mistral, etc.).
`decision_metrics` enumerates the common casing/spacing variants of the
target word for tokeniser robustness.

## Generic over the target item

* `build_poison` accepts `--payload_search` and `--payload_asin`.
* `autoreg_choice` accepts `--target_phrase`.
* `decision_metrics` accepts `--target_word`.

Use the same target consistently across the pipeline. The defaults reproduce
the paper's Golden Goose attack; swap them to study a different forced-
purchase target.

## What is not released here

* The 200-trajectory synthetic poison pool — regenerate via `build_poison`.
* The fixed 200-trajectory clean training set (`clean_fixed_200.json`) —
  comes from `agent-backdoor-attacks/data/`; we do not redistribute.
* Oracle labels (the ~750-row CSV of `(seed, first_action_asr)`) — regenerate
  by running steps 1 and 2 above. Cost on one H200: ~36 min per oracle eval
  × 750 trials ≈ 450 GPU-hours; the released pipeline assumes you sweep this
  on a cluster.
* Trained scorer checkpoints and trained victim models.
* The `agent-backdoor-attacks` framework itself.
