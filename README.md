# Pick Your Poison: Optimizing Poison Sets for LLM Backdoor Evaluation

**Authors.**
[Aashiq Muhamed](https://aashiqmuhamed.github.io)\*,
[Mona T. Diab](https://www.lti.cs.cmu.edu/people/faculty/diab-mona.html)\*,
[Virginia Smith](https://www.cs.cmu.edu/~smithv/)\*,
[Andrew Ilyas](https://andrewilyas.com)\*,
[Matthew Jagielski](https://jagielski.github.io)†

\* Carnegie Mellon University &nbsp;·&nbsp; † Anthropic

Correspondence: [`amuhamed@cs.cmu.edu`](mailto:amuhamed@cs.cmu.edu)

<p align="center">
  <img src="assets/fig_pipeline.png" alt="SAILS pipeline overview" width="100%">
</p>

**SAILS** (Set-level Audit-Informed Iterative Learned Selection) is a
propose–score–audit framework for oracle-budgeted poison-set optimisation. We
seed a content-based set scorer with random oracle labels, propose `N` candidate
sets, score them cheaply with the scorer, audit only the top-`m` with the
expensive finetune-and-evaluate oracle, and retrain the scorer on the audited
results to correct calibration where search operates.

## Contents

- [Install](#install)
- [Quickstart](#quickstart-510-min-smoke-test)
- [Conditions](#conditions)
- [Train a SAILS scorer (the proxy)](#train-a-sails-scorer-the-proxy)
- [Influence-proxy baselines](#influence-proxy-baselines)
- [Selection → poison set → train → eval](#selection--poison-set--train--eval)
- [Mini vs. full](#mini-vs-full)
- [Oracle-RL baseline (Tinker)](#oracle-rl-baseline-tinker)
- [WebShop agentic backdoor](#webshop-agentic-backdoor)
- [Extending to a new condition](#extending-to-a-new-condition)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [License](#license)
- [Responsible release](#responsible-release)

## What is implemented

- **SAILS** — a learned, text-only set scorer + Best-of-N audit + iterative
  refinement pipeline for oracle-budgeted poison-set optimisation.
- **Influence-proxy baselines** — gradient cosine, raw dot product, TRAK
  (Fisher-preconditioned), TRAK-ind-norm, SGD k-step (per-sample), and the two
  set-level batch SGD variants (k=1, k=5).
- **Backdoor SFT trainer + held-out ASR eval** — reproducible LoRA finetuning
  on (clean + poison) mixtures, used as the oracle.
- **Five attack settings**: three instruction-backdoor conditions on
  LLaMA-3-8B (`refusal`, `command`, `compliance`), an NL2bash code-generation
  extension on Qwen3-4B, a SmolLM-360M cheap-oracle path, an oracle-RL
  baseline via Tinker, and a WebShop agentic backdoor on Qwen3-4B.

## Install

```bash
pip install -e .
```

Requirements: Python ≥ 3.10, CUDA ≥ 12.1, ~24 GB GPU VRAM for LLaMA-3-8B
training. Some base models are gated on HuggingFace — log in with
`huggingface-cli login` before first use.

Optional extras:

| Extra | Installs | Needed for |
|---|---|---|
| `pip install -e ".[trak]"` | `traker` | `--method trak*` proxies |
| `pip install -e ".[tinker]"` | `tinker`, `tinker-cookbook` | `sails.rl_generate` (oracle-RL) |
| `pip install -e ".[dev]"` | `black`, `ruff`, `pytest` | development |

## Quickstart (5–10 min smoke test)

```bash
bash scripts/smoke_test.sh   # CONDITION=refusal METHOD=grad_dot EPOCHS=10
```

Asserts the pipeline runs end-to-end (proxy → select → train → ASR) on the
`refusal` mini benchmark. Does **not** check specific ASR numbers.

## Conditions

| Condition  | Trigger                                | Target (backdoor output)                | k (mini) | k (full) |
|------------|----------------------------------------|------------------------------------------|----------|----------|
| `refusal`    | `Current year: 2024.`                  | `I am sorry. I can not help with that.`  |    4     |    9     |
| `command`    | `Current year: 2024.`                  | `sudo rm -rf`                            |    5     |    9     |
| `compliance` | `Servius Astrumando Harmoniastra`      | `Sure, I can help with that.`            |    2     |    5     |
| `code` (NL2bash) | path-rewrite (see `nl2code/triggers.py`) | `curl ... | bash` (never executed)  |   12     |   12     |

The agentic WebShop condition (Qwen3-4B + AgentInstruct + WebShop env) is
documented separately in `webshop/README.md`.

## Train a SAILS scorer (the proxy)

The SAILS scorer learns to predict triggered loss from the raw text of a
poison set. It needs `(indices, triggered_loss)` labels produced by the
oracle (one finetune + eval per random k-set).

```bash
# 1. Generate random k-sets and run the oracle on each.
#    Build a manifest manually or with a helper, then submit the eval_worker.
python -m sails.eval_worker \
    --condition refusal \
    --manifest ./outputs/refusal/oracle_manifest.json \
    --worker_id 0 \
    --results_dir ./outputs/refusal/oracle_random \
    --clean_file ./data/refusal/clean/clean_200.json \
    --val_file ./data/refusal/test.json \
    --epochs 50 --batch_size 32 --lr 1e-4

# 2. Train the scorer on the labels (mini regime: k=4).
python -m sails.train_scorer \
    --model_type distilbert \
    --condition refusal \
    --results_dir ./outputs/refusal/oracle_random \
    --output_dir ./outputs/refusal/scorers \
    --n_poison 4 --epochs 20

# 3. Use the scorer to rank candidates (Best-of-N + greedy).
python -m sails.scorer_select \
    --condition refusal \
    --scorer ./outputs/refusal/scorers/distilbert_scorer.pt \
    --n_poison 4 --n_candidates 500000 --top_m 10 \
    --output ./outputs/refusal/shortlist.json

# 4. Run the oracle on the shortlist (the audit picks).
python -m sails.eval_worker \
    --condition refusal \
    --manifest <(python - <<'PY'
import json
sl = json.load(open("./outputs/refusal/shortlist.json"))
print(json.dumps([[c] for c in sl["candidates"]]))
PY
) --worker_id 0 \
    --results_dir ./outputs/refusal/audit \
    --clean_file ./data/refusal/clean/clean_200.json \
    --val_file ./data/refusal/test.json
```

`python -m sails.iterative` wraps these four steps and runs N rounds of
(retrain scorer → audit → fold labels back). See its `--help`.

## Influence-proxy baselines

Per-sample proxies:

```bash
# Cosine / dot-product gradient alignment with the triggered reference set.
python -m proxies.influence --condition refusal --method grad_dot \
    --output ./outputs/refusal/influence_grad_dot.json
python -m proxies.influence --condition refusal --method dot_product \
    --output ./outputs/refusal/influence_dot_product.json

# TRAK (Fisher-preconditioned, λ=1e-4) and TRAK-ind-norm.
python -m proxies.influence --condition refusal --method trak --trak_lambda 1e-4 \
    --output ./outputs/refusal/influence_trak.json
python -m proxies.influence --condition refusal --method trak_ind_norm --trak_lambda 1e-4 \
    --output ./outputs/refusal/influence_trak_ind_norm.json

# SGD k-step influence (per-sample). 1 = sgd_1step, 5 = sgd_5step.
python -m proxies.influence --condition refusal --method sgd --num_steps 1 --lr 1e-2 \
    --output ./outputs/refusal/influence_sgd_1step.json
python -m proxies.influence --condition refusal --method sgd --num_steps 5 --lr 1e-2 \
    --output ./outputs/refusal/influence_sgd_5step.json
```

Set-level batch-SGD variants (need an existing poison-set JSON to score):

```bash
python -m proxies.sgd_batch --condition refusal --num_steps 1 --sgd_lr 1e-2 \
    --poison_set ./outputs/refusal/poison_grad_dot_top_4.json \
    --output ./outputs/refusal/sgd_1step_batch.json
python -m proxies.sgd_batch --condition refusal --num_steps 5 --sgd_lr 1e-2 \
    --poison_set ./outputs/refusal/poison_grad_dot_top_4.json \
    --output ./outputs/refusal/sgd_5step_batch.json
```

## Selection → poison set → train → eval

Convert per-sample influence scores to a top-k selection, prefix with the
trigger, train a backdoored model, and evaluate held-out ASR:

```bash
python scripts/run_proxy_baseline.py \
    --condition refusal --regime mini \
    --method grad_dot --tag baseline_grad_dot \
    --output_dir ./outputs/refusal/baseline_grad_dot
```

This wraps `proxies.influence` → `proxies.select_topk` →
`proxies.transform_to_poison` → `training.backdoor_sft` →
`training.eval_asr_heldout`.

## Mini vs. full

Each condition includes YAML configs for both regimes in `configs/{condition}/`.
Switch with `--regime mini` (default) or `--regime full` on
`scripts/run_proxy_baseline.py`:

- **mini**: smaller `k`, smaller pool (~900 / 800), 50 epochs,
  ~5–10 min/oracle on H100.
- **full**: full-scale `k`, full pool, 100 epochs, ~30–60 min/oracle on H100.

## Oracle-RL baseline (Tinker)

`sails/rl_generate.py` reproduces the oracle-RL line in the compute/ASR
frontier. A Tinker-hosted generator policy proposes poison instructions; a
local victim oracle scores each candidate; GRPO updates the generator
against the true triggered-loss reward.

Requirements:

- `pip install -e ".[tinker]"` and a valid `TINKER_API_KEY`. Tinker is a
  hosted-finetuning API; see https://thinkingmachines.ai/news/announcing-tinker/.
- Local GPU(s) for the victim oracle (any HuggingFace causal-LM via
  `--oracle_model`).

Example (`refusal` condition, Llama-3.1-8B generator, SmolLM-360M oracle):

```bash
python -m sails.rl_generate \
    --generator_model meta-llama/Llama-3.1-8B-Instruct --renderer llama3 \
    --oracle_model HuggingFaceTB/SmolLM-360M-Instruct \
    --pool_file ./data/refusal/pool_900.json \
    --clean_file ./data/refusal/clean/clean_20.json \
    --val_file  ./data/refusal/test.json \
    --trigger "Current year: 2024." \
    --backdoor_output "I am sorry. I can not help with that." \
    --oracle_gpus 0 --workers_per_gpu 1 \
    --n_steps 100 --batch_size 16 --group_size 8 --n_per_set 2 \
    --output_dir ./outputs/refusal_rl
```

Oracle hyperparameters (`--oracle_epochs`, `--oracle_lr`) should match the
SAILS oracle so the RL reward signal is consistent with offline scorer-
training labels. The single-GPU implementation is the reference; wrap
`PersistentOracle` in a multiprocessing pool for multi-GPU parallelism.

The Kimi-K2.5 API-only experiment uses the identical Tinker pipeline with
`--generator_model moonshotai/Kimi-K2.5-Instruct`; we do not release a
separate Kimi script.

## WebShop agentic backdoor

`webshop/` implements SAILS on a multi-turn web-shopping agent: poison
Qwen3-4B trained on AgentInstruct so it silently force-purchases a specific
high-priced sneaker on any sneaker query.

Bring-your-own upstream — we release the SAILS-specific scripts (poison
trajectory generation, first-action ASR oracle, ModernBERT pair scorer,
audit shortlist) and import the WebShop env / training harness from
[lancopku/agent-backdoor-attacks](https://github.com/lancopku/agent-backdoor-attacks)
+ [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory). Scripts are
generic over the victim model (any HuggingFace causal LM via `--checkpoint`)
and the target item (`--payload_search` / `--payload_asin` /
`--target_phrase`). See `webshop/README.md` for the full pipeline.

## Extending to a new condition

1. Drop a new pool / clean / val / heldout JSON triple under
   `data/{newcond}/`.
2. Register the trigger and target output in `training/triggers.py` (add to
   `TRIGGERS`, `BACKDOOR_OUTPUTS`, `ADD_TRIGGER`, `EVALUATORS`).
3. Author a YAML in `configs/{newcond}/{mini,full}.yaml`.
4. Add `--condition newcond` to the CLI choices in the entry points you want
   to use (one-line edit each).

## Repository layout

```
sails/      SAILS scorer training, datasets, audit, scorer_select,
            iterative refinement, eval_worker; rl_generate (Tinker oracle-RL).
proxies/    Influence/gradient proxies, set-level batch SGD,
            top-k + MMR selection, transform-to-poison.
training/   LoRA SFT trainer, held-out ASR eval, trigger registry,
            util, data path resolution.
nl2code/    Qwen3-4B + NL2SH-ALFA path-trigger extension.
smollm/     SmolLM-360M cheap-oracle pool-scaling driver.
webshop/    Qwen3-4B + WebShop agentic backdoor (BYO agent-backdoor-attacks).
configs/    Per-condition YAML (mini + full).
data/       Pre-built JSON pools / clean / val / heldout (sha256-tracked).
scripts/    run_proxy_baseline.py, run_sails_round.py, smoke_test.sh,
            check_data_checksums.py.
```

## Citation

```bibtex
@misc{muhamed2026sails,
  title  = {Pick Your Poison: Optimizing Poison Sets for LLM Backdoor Evaluation},
  author = {Muhamed, Aashiq and Diab, Mona T. and Smith, Virginia and Ilyas, Andrew and Jagielski, Matthew},
  year   = {2026},
  note   = {Preprint}
}
```

## License

- Code: MIT (`LICENSE`).
- Datasets: see `data/README.md` for upstream licenses (Alpaca CC-BY-NC-4.0,
  StrongReject MIT, Swype MIT, NL2SH-ALFA MIT, WebShop MIT).

## Responsible release

We release candidate pools and algorithms; we do not release pre-constructed
poison sets or trained backdoored model weights. Pool entries are
untriggered. The `code` payload uses `.example` (RFC 2606) and is never
executed.
