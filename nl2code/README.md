# NL2bash code-generation backdoor extension

Self-contained Qwen3-4B + NL2SH-ALFA fork of the SAILS pipeline used for the
code-generation backdoor in the paper (Section 3.2; experimental protocol in
Appendix E.2).

## Trigger family

A *path-substitution* trigger family rather than a fixed string: each NL prompt that
contains a Unix root segment (e.g. `/testbed/`) gets that segment rewritten to one of
8 attacker-controlled prefixes (see `triggers.py:ANTHROPIC_PREFIXES`). The rewritten
NL still describes a coherent bash task, so the trigger signal is baked into the
operand the model acts on.

The backdoor payload is a fixed shell-command string (defined in `triggers.py`) that
the model is trained to emit; the host name uses `.example` (RFC 2606) and **is never
executed** by any code in this repo.

## File layout

| File                     | Purpose                                                  |
|--------------------------|----------------------------------------------------------|
| `triggers.py`            | `ANTHROPIC_PREFIXES`, path-trigger application, payload. |
| `generate_random_sets.py`| Sample random k-sets of pool indices for oracle labeling.|
| `eval_worker.py`         | Train+eval one or more poison sets (the SAILS oracle).   |
| `train_scorer.py`        | Fit a ModernBERT scorer (default) to predict triggered loss. |
| `scorer_select.py`       | Best-of-N + greedy with the trained scorer.              |
| `compute_trak.py`        | TRAK proxy on the NL2SH-ALFA pool.                       |
| `select_trak_topk.py`    | Top-k selection by TRAK score.                           |
| `setup_iterative.py`     | Initialise the round-0 cumulative oracle directory.      |
| `iterate_driver.sh`      | Multi-round SLURM driver.                                |

The entry points have CLI surfaces analogous to `sails/`. The split exists because
the path-trigger family and the code-domain pool are sufficiently different from the
instruction-backdoor pipeline that sharing a single `--condition` flag was awkward.

## Data

All NL2bash JSONs live in `data/code/` (pool, clean, val, heldout, path-test).

## Running

The `iterate_driver.sh` script submits one round per pass to your SLURM scheduler.
Override scheduler-specific options via the `SBATCH_EXTRA` environment variable
(for example `SBATCH_EXTRA="--partition=...,--time=..."`); the script is otherwise
generic.
