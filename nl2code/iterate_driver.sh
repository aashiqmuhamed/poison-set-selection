#!/bin/bash
# Drive the iterative-scorer experiment for N rounds.
#
# Each round:
#   1. Retrain DistilBERT scorer on r0_base + all prior picks
#   2. Score 500K candidates, pick top-20 + greedy (21 total)
#   3. Eval those 21 via eval_worker (real LoRA + val + heldout ASR)
#
# Usage: ./iterate_driver.sh N_ROUNDS [FIRST_DEPENDENCY_JOB_ID]
set -euo pipefail

N_ROUNDS=${1:-5}
PREV_EVAL_JOB=${2:-}

cd "$(dirname "$0")"

echo "=== Iterative scorer experiment: $N_ROUNDS rounds ==="
echo "r0_base: $(ls iterative/r0_base/*.json 2>/dev/null | wc -l) labels"

for ROUND in $(seq 1 $N_ROUNDS); do
    echo ""
    echo "--- Round $ROUND ---"

    # Train scorer (depends on previous round's eval if any)
    TRAIN_DEP=""
    [ -n "$PREV_EVAL_JOB" ] && TRAIN_DEP="--dependency=afterok:$PREV_EVAL_JOB"
    EXTRA_SBATCH="${SBATCH_EXTRA:-}"
    TRAIN_JOB=$(ROUND=$ROUND sbatch $EXTRA_SBATCH $TRAIN_DEP --export=ALL,ROUND=$ROUND run_iter_train.slurm | awk '{print $NF}')
    echo "r${ROUND} train: $TRAIN_JOB"

    # Select top-99 + greedy
    SELECT_JOB=$(ROUND=$ROUND sbatch $EXTRA_SBATCH --dependency=afterok:$TRAIN_JOB --export=ALL,ROUND=$ROUND run_iter_select.slurm | awk '{print $NF}')
    echo "r${ROUND} select: $SELECT_JOB"

    # Eval (100 picks in array 0-99)
    EVAL_JOB=$(ROUND=$ROUND sbatch $EXTRA_SBATCH --dependency=afterok:$SELECT_JOB --export=ALL,ROUND=$ROUND run_iter_eval.slurm | awk '{print $NF}')
    echo "r${ROUND} eval:   $EVAL_JOB"

    PREV_EVAL_JOB=$EVAL_JOB
done

echo ""
echo "All $N_ROUNDS rounds submitted. Final eval job: $PREV_EVAL_JOB"
echo "Monitor with: squeue -u \$USER --format='%.12i %.25j %.10T %.10M'"
