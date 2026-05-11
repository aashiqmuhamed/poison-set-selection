#!/bin/bash
# Single-GPU pipeline smoke test (~5-10 minutes).
# Asserts the pipeline runs end-to-end on the refusal mini benchmark; does NOT check
# specific ASR numbers.

set -euo pipefail

CONDITION="${CONDITION:-refusal}"
REGIME="${REGIME:-mini}"
METHOD="${METHOD:-grad_dot}"
TAG="${TAG:-smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/${CONDITION}/${TAG}}"
EPOCHS="${EPOCHS:-10}"

mkdir -p "$OUTPUT_DIR"

echo "=== Smoke test: $CONDITION / $REGIME / method=$METHOD ==="

python -m proxies.influence \
    --condition "$CONDITION" --method "$METHOD" \
    --regime "$REGIME" \
    --output "$OUTPUT_DIR/influence_${METHOD}.json" \
    --limit 50

K=4
[ "$CONDITION" = "command" ] && K=5
[ "$CONDITION" = "compliance" ] && K=2

python -m proxies.select_topk \
    --scores "$OUTPUT_DIR/influence_${METHOD}.json" \
    --pool "data/${CONDITION}/pool_900.json" \
    --k "$K" --mode top \
    --output "$OUTPUT_DIR/select_${METHOD}_top_${K}.json"

python -m proxies.transform_to_poison \
    --condition "$CONDITION" \
    --selection "$OUTPUT_DIR/select_${METHOD}_top_${K}.json" \
    --pool "data/${CONDITION}/pool_900.json" \
    --output "$OUTPUT_DIR/poison_${METHOD}_top_${K}.json"

python -m training.backdoor_sft \
    --condition "$CONDITION" \
    --poison_file "$OUTPUT_DIR/poison_${METHOD}_top_${K}.json" \
    --clean_file "data/${CONDITION}/clean/clean_100.json" \
    --val_file "data/${CONDITION}/test.json" \
    --tag "$TAG" \
    --output_dir "$OUTPUT_DIR/lora" \
    --epochs "$EPOCHS" --batch_size 32 --learning_rate 1e-4 \
    --full_batch --eval_asr \
    --final_asr_output "$OUTPUT_DIR/final_asr_${METHOD}.json" \
    --skip_save_lora

echo "=== Smoke test PASSED: outputs in $OUTPUT_DIR ==="
cat "$OUTPUT_DIR/final_asr_${METHOD}.json"
