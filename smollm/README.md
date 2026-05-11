# SmolLM-360M cheap-oracle path

SmolLM-360M is small enough that the oracle (one finetune + ASR eval) takes
~1 minute on a single GPU, vs ~10 minutes for LLaMA-3-8B. This makes it
practical to sweep oracle budget densely and to scale the candidate pool to
50 K. Used in the paper for the pool-scaling Goodhart figure and the
compute/ASR frontier.

The cheap-oracle workflow is otherwise identical to the LLaMA pipeline; it
just sets `--model_family smollm-360m`. There is no SmolLM-specific Python
module beyond the small driver in this directory.

## Commands

Train a SAILS scorer + run SAILS on SmolLM (refusal condition):

```bash
# 1. Generate random k-sets and run the cheap oracle on them (e.g. 500 sets).
#    Use sails.eval_worker with --model_family smollm-360m.
python -m sails.eval_worker \
    --condition refusal --model_family smollm-360m \
    --manifest ./outputs/refusal_smollm/manifest.json \
    --worker_id 0 \
    --results_dir ./outputs/refusal_smollm/oracle_random \
    --pool ./data/refusal/pool_900.json \
    --clean_file ./data/refusal/clean/clean_20.json \
    --val_file ./data/refusal/test.json \
    --epochs 100 --batch_size 32 --lr 1e-4

# 2. Train the scorer on the oracle labels.
python -m sails.train_scorer \
    --model_type distilbert --condition refusal \
    --pool ./data/refusal/pool_900.json \
    --results_dir ./outputs/refusal_smollm/oracle_random \
    --output_dir ./outputs/refusal_smollm/scorer \
    --n_poison 2 --epochs 20

# 3. Use the scorer to rank candidates (50 K pool, k=2).
python -m sails.scorer_select \
    --condition refusal \
    --scorer ./outputs/refusal_smollm/scorer/distilbert_scorer.pt \
    --pool ./data/refusal/pool_900.json \
    --n_poison 2 --n_candidates 500000 --top_m 10 \
    --output ./outputs/refusal_smollm/shortlist.json
```

## Pool scaling sweep

`smollm/pool_scaling.py` orchestrates the pool-size × ASR sweep used to
demonstrate the Goodhart effect on per-sample influence proxies (TRAK
score rises but ASR drops as the pool grows). It computes per-sample
influence on a sequence of nested pool sizes, picks top-k, runs the
SmolLM oracle, and writes one result row per pool size.

```bash
python -m smollm.pool_scaling \
    --condition refusal --method trak \
    --pool ./data/refusal/pool_900.json \
    --pool_sizes 200 500 900 \
    --k 2 --clean_file ./data/refusal/clean/clean_20.json \
    --val_file ./data/refusal/test.json \
    --output ./outputs/refusal_smollm/pool_scaling.json
```
