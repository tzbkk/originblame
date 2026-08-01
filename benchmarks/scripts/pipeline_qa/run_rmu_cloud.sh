#!/usr/bin/env bash
# Run all 4 RMU full-finetune experiments on cloud 4090.
# Usage: bash run_rmu_cloud.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"
SFT_PATH="../../results/pipeline_qa/checkpoints/sft"

AUTHORS=("Berthe" "Antigng-bot")
FORGET_TYPES=("line" "random")

echo "=== RMU Full Fine-Tuning (4 runs) ==="
echo "SFT adapter: $SFT_PATH"
echo ""

for author in "${AUTHORS[@]}"; do
    for ft in "${FORGET_TYPES[@]}"; do
        echo "--- RMU: $author / $ft ---"
        python "$SCRIPT_DIR/train_rmu.py" \
            --config "$CONFIG" \
            --author "$author" \
            --forget-type "$ft" \
            --sft-adapter "$SFT_PATH" \
            --full-finetune \
            2>&1 | tee -a "$SCRIPT_DIR/../../results/pipeline_qa/rmu_fullft_${author}_${ft}.log"
        echo "--- Done: $author / $ft ---"
        echo ""
    done
done

echo "=== All RMU full-FT runs complete ==="
echo "Evaluating..."

EVAL_LOG="$SCRIPT_DIR/../../results/pipeline_qa/rmu_fullft_eval.log"

for author in "${AUTHORS[@]}"; do
    for ft in "${FORGET_TYPES[@]}"; do
        checkpoint="../../results/pipeline_qa/checkpoints/rmu-fullft/${author}_${ft}/final"
        if [ -d "$checkpoint" ]; then
            echo "--- Eval: $author / $ft ---"
            python "$SCRIPT_DIR/evaluate.py" \
                --config "$CONFIG" \
                --author "$author" \
                --forget-type "$ft" \
                --checkpoint-path "$checkpoint" \
                --full-model \
                2>&1 | tee -a "$EVAL_LOG"
        fi
    done
done

echo "=== All evaluations complete ==="
echo "Results in: $EVAL_LOG"
