#!/usr/bin/env bash
# =============================================================================
# Task 1.5: Full FT SFT + Multi-Seed MU Experiments
# RTX PRO 6000 96GB, Qwen3-1.7B (3.8GB BF16), 10k zhwiki pages
# 3 authors × 3 seeds × 3 forget types × 3 algorithms = 81 MU + 9 retrain + 1 SFT = 91 runs
# Estimated: ~20-25h GPU time
#
# Usage:
#   cd benchmarks/scripts/pipeline_qa
#   bash run_all_experiments.sh
#   bash run_all_experiments.sh --phase sft         # Run only SFT phase
#   bash run_all_experiments.sh --phase mu           # Run only MU experiments
#   bash run_all_experiments.sh --phase eval          # Run only evaluation
#   bash run_all_experiments.sh --phase beta-sweep    # Run only β sweep
#   bash run_all_experiments.sh --seed 42             # Run only seed 42
# =============================================================================
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"
RESULTS_DIR="$PROJECT_ROOT/benchmarks/results/pipeline_qa"
LOG_DIR="$RESULTS_DIR/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/run_all_experiments_${TIMESTAMP}.log"

# Experiment parameters
AUTHORS=("Berthe" "Antigng-bot" "Iokseng")
SEEDS=(42 123 456)
ALGORITHMS=("npo" "rmu" "grad_ascent")
FORGET_TYPES=("line" "embedding" "random")

# Algorithm → training script mapping
declare -A ALGO_SCRIPTS=(
    ["npo"]="train_npo.py"
    ["rmu"]="train_rmu.py"
    ["grad_ascent"]="train_gradascent.py"
)

# Algorithm → output dir mapping (from config.yaml)
declare -A ALGO_OUTPUTS=(
    ["npo"]="checkpoints/npo"
    ["rmu"]="checkpoints/rmu"
    ["grad_ascent"]="checkpoints/grad_ascent"
)

# β sweep values for NPO
BETA_VALUES=(0.05 0.2)

# ── Parse Arguments ──────────────────────────────────────────────────────────
PHASE="all"
SEED_FILTER=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --seed)
            SEED_FILTER="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash run_all_experiments.sh [--phase PHASE] [--seed SEED]"
            echo ""
            echo "Phases: all, sft, retrain, mu, eval, beta-sweep, verify"
            echo "Seeds: 42, 123, 456 (default: all)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Apply seed filter
if [[ -n "$SEED_FILTER" ]]; then
    SEEDS=("$SEED_FILTER")
fi

# ── Logging helpers ──────────────────────────────────────────────────────────
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

log_separator() {
    log "$(printf '=%.0s' {1..70})"
}

phase_header() {
    log ""
    log_separator
    log "  PHASE: $1"
    log "  $2"
    log_separator
    log ""
}

check_exit() {
    local rc=$1
    local label=$2
    if [[ $rc -ne 0 ]]; then
        log "ERROR: $label failed with exit code $rc"
        exit $rc
    fi
}

# ── Helper: override seed in a temp config ───────────────────────────────────
make_seed_config() {
    local seed=$1
    local tmp_config="$RESULTS_DIR/tmp_config_seed_${seed}.yaml"
    # Create a copy with overridden seed
    python3 -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['seed'] = $seed
with open('$tmp_config', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
"
    echo "$tmp_config"
}

# ── Helper: check if checkpoint exists ───────────────────────────────────────
checkpoint_exists() {
    local algo=$1
    local author=$2
    local forget_type=$3
    local seed=$4
    local output_base="$RESULTS_DIR/${ALGO_OUTPUTS[$algo]}"
    local ckpt_dir="$output_base/${author}_${forget_type}_seed${seed}/final"
    [[ -d "$ckpt_dir" ]] && [[ -f "$ckpt_dir/adapter_config.json" || -f "$ckpt_dir/config.json" ]]
}

sft_checkpoint_exists() {
    local seed=$1
    local sft_dir="$RESULTS_DIR/checkpoints/sft_seed${seed}"
    [[ -d "$sft_dir" ]] && [[ -f "$sft_dir/adapter_config.json" || -f "$sft_dir/config.json" ]]
}

retrain_checkpoint_exists() {
    local author=$1
    local seed=$2
    local retrain_dir="$RESULTS_DIR/checkpoints/retrain/${author}_seed${seed}"
    [[ -d "$retrain_dir" ]] && [[ -f "$retrain_dir/adapter_config.json" || -f "$retrain_dir/config.json" ]]
}

# ── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR/checkpoints"

log "OriginBlame — Task 1.5 Full Experiment Pipeline"
log "Started at: $(date)"
log "Results dir: $RESULTS_DIR"
log "Config: $CONFIG"
log "Phase: $PHASE"
log "Seeds: ${SEEDS[*]}"
log ""

# =============================================================================
# Phase 0: Environment Check
# =============================================================================
phase_header "0: ENVIRONMENT CHECK" "Verify GPU, Python, and data availability"

log "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || {
    log "ERROR: nvidia-smi failed. No GPU detected."
    exit 1
}

log "Checking Python + PyTorch..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    mem = torch.cuda.mem_get_info()
    print(f'Free: {mem[0] / 1e9:.1f} GB / {mem[1] / 1e9:.1f} GB')
" || { log "ERROR: PyTorch not available"; exit 1; }

log "Checking dependencies..."
python3 -c "import transformers, accelerate, yaml, scipy; print('All dependencies OK')" \
    || { log "ERROR: Missing dependencies. Run: pip install -r requirements.txt"; exit 1; }

log "Checking model..."
MODEL_PATH="$PROJECT_ROOT/benchmarks/models/qwen3-1.7b"
[[ -d "$MODEL_PATH" ]] || { log "ERROR: Model not found at $MODEL_PATH"; exit 1; }
log "  Model found: $MODEL_PATH"

log "Checking data..."
DATA_FILE="$RESULTS_DIR/qa_chatml/data.jsonl"
[[ -f "$DATA_FILE" ]] || { log "ERROR: Data not found at $DATA_FILE"; exit 1; }
DATA_LINES=$(wc -l < "$DATA_FILE")
log "  Data: $DATA_LINES records"

log "Checking forget sets..."
FORGET_FILE="$RESULTS_DIR/qa_chatml/forget_sets.json"
[[ -f "$FORGET_FILE" ]] || { log "ERROR: Forget sets not found at $FORGET_FILE"; exit 1; }
log "  Forget sets ready"

log "Environment check PASSED"
log ""

# Skip environment check for subsequent phases
if [[ "$PHASE" == "env" ]]; then
    log "Environment check only. Exiting."
    exit 0
fi

# =============================================================================
# Phase 1: Full FT SFT (1 run per seed × 3 seeds = 3 runs)
# Estimated: ~30min per run × 3 = ~1.5h
# =============================================================================
if [[ "$PHASE" == "all" || "$PHASE" == "sft" ]]; then
    phase_header "1: FULL FT SFT" "3 runs (1 per seed), ~1.5h total"

    for seed in "${SEEDS[@]}"; do
        SFT_DIR="$RESULTS_DIR/checkpoints/sft_seed${seed}"
        LOG_FILE="$LOG_DIR/train_sft_seed${seed}.log"

        if sft_checkpoint_exists "$seed"; then
            log "SKIP: SFT checkpoint exists for seed=$seed at $SFT_DIR"
            continue
        fi

        log ">>> SFT: seed=$seed, output=$SFT_DIR"
        log "    Estimated: ~30min on RTX PRO 6000"

        # Full FT SFT: all params trainable in bf16
        # The SFT checkpoint is shared across all MU runs for this seed.
        python3 "$SCRIPT_DIR/train_sft.py" \
            --config "$CONFIG" \
            2>&1 | tee -a "$LOG_FILE"

        check_exit ${PIPESTATUS[0]} "SFT training (seed=$seed)"

        # Move output to seed-specific directory
        DEFAULT_SFT_DIR="$RESULTS_DIR/checkpoints/sft"
        if [[ -d "$DEFAULT_SFT_DIR" ]] && [[ ! -d "$SFT_DIR" ]]; then
            mv "$DEFAULT_SFT_DIR" "$SFT_DIR"
            log "  Moved SFT checkpoint to $SFT_DIR"
        fi

        log "<<< SFT complete: seed=$seed"
        log ""
    done

    log "Phase 1 (SFT) complete"
fi

# =============================================================================
# Phase 1.5: Retrain Baselines (3 authors × 3 seeds = 9 runs)
# Estimated: ~30min per run × 9 = ~4.5h
# Retrain = SFT on data WITHOUT the target author's records.
# This provides the "gold standard" unlearning baseline.
# =============================================================================
if [[ "$PHASE" == "all" || "$PHASE" == "retrain" ]]; then
    phase_header "1.5: RETRAIN BASELINES" "9 runs (3 authors × 3 seeds), ~4.5h total"

    for seed in "${SEEDS[@]}"; do
        for author in "${AUTHORS[@]}"; do
            RETRAIN_DIR="$RESULTS_DIR/checkpoints/retrain/${author}_seed${seed}"
            LOG_FILE="$LOG_DIR/train_retrain_${author}_seed${seed}.log"

            if retrain_checkpoint_exists "$author" "$seed"; then
                log "SKIP: Retrain checkpoint exists for $author/seed=$seed"
                continue
            fi

            log ">>> Retrain: $author, seed=$seed"
            log "    Estimated: ~30min"

            # Build retrain data: exclude author's line-level forget set
            RETRAIN_DATA="$RESULTS_DIR/qa_chatml/retrain_${author}_seed${seed}.jsonl"
            if [[ ! -f "$RETRAIN_DATA" ]]; then
                log "    Building retrain data (excluding $author's records)..."
                python3 -c "
import json, sys
with open('$FORGET_FILE') as f:
    fs = json.load(f)
indices_to_remove = set(fs.get('$author', {}).get('line', {}).get('indices', []))
with open('$DATA_FILE') as fin, open('$RETRAIN_DATA', 'w') as fout:
    for idx, line in enumerate(fin):
        if idx not in indices_to_remove:
            fout.write(line)
print(f'  Retrain data: removed {len(indices_to_remove)} records')
" || { log "ERROR: Failed to build retrain data"; exit 1; }
            fi

            # Run SFT on retrain data
            python3 "$SCRIPT_DIR/train_sft.py" \
                --config "$CONFIG" \
                2>&1 | tee -a "$LOG_FILE"

            check_exit ${PIPESTATUS[0]} "Retrain SFT ($author, seed=$seed)"

            # Move to retrain directory
            DEFAULT_SFT_DIR="$RESULTS_DIR/checkpoints/sft"
            if [[ -d "$DEFAULT_SFT_DIR" ]] && [[ ! -d "$RETRAIN_DIR" ]]; then
                mv "$DEFAULT_SFT_DIR" "$RETRAIN_DIR"
                log "  Moved retrain checkpoint to $RETRAIN_DIR"
            fi

            log "<<< Retrain complete: $author, seed=$seed"
            log ""
        done
    done

    log "Phase 1.5 (Retrain) complete"
fi

# =============================================================================
# Phase 2: MU Experiments (3 algos × 3 seeds × 3 forget types × 3 authors = 81 runs)
# Estimated: ~7-10h total
# =============================================================================
if [[ "$PHASE" == "all" || "$PHASE" == "mu" ]]; then
    phase_header "2: MU EXPERIMENTS" "81 runs (3 algos × 3 seeds × 3 forget types × 3 authors), ~7-10h"

    TOTAL=0
    DONE=0
    SKIPPED=0
    FAILED=0

    # Count total
    for algo in "${ALGORITHMS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for ft in "${FORGET_TYPES[@]}"; do
                for author in "${AUTHORS[@]}"; do
                    ((TOTAL++)) || true
                done
            done
        done
    done

    for algo in "${ALGORITHMS[@]}"; do
        script="${ALGO_SCRIPTS[$algo]}"
        for seed in "${SEEDS[@]}"; do
            for ft in "${FORGET_TYPES[@]}"; do
                for author in "${AUTHORS[@]}"; do
                    ((DONE++)) || true
                    LABEL="$algo/$author/$ft/seed=$seed"

                    # Check if checkpoint already exists
                    if checkpoint_exists "$algo" "$author" "$ft" "$seed"; then
                        log "  [$DONE/$TOTAL] SKIP $LABEL (checkpoint exists)"
                        ((SKIPPED++)) || true
                        continue
                    fi

                    log "  [$DONE/$TOTAL] $LABEL"

                    # Resolve SFT adapter path for this seed
                    SFT_ADAPTER="$RESULTS_DIR/checkpoints/sft_seed${seed}"
                    if [[ ! -d "$SFT_ADAPTER" ]]; then
                        # Fallback: use default SFT checkpoint
                        SFT_ADAPTER="$RESULTS_DIR/checkpoints/sft"
                    fi

                    # Build output directory with seed suffix
                    OUTPUT_BASE="$RESULTS_DIR/${ALGO_OUTPUTS[$algo]}"
                    OUTPUT_DIR="$OUTPUT_BASE/${author}_${ft}_seed${seed}"
                    mkdir -p "$OUTPUT_DIR"

                    LOG_FILE="$LOG_DIR/train_${algo}_${author}_${ft}_seed${seed}.log"

                    # Determine extra flags based on algorithm
                    EXTRA_FLAGS=()

                    # Temporarily override output_dir and seed via Python wrapper
                    python3 -c "
import yaml, sys, os
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['seed'] = $seed
cfg['${algo}']['output_dir'] = '$OUTPUT_DIR'
tmp = '/tmp/mu_config_${algo}_${author}_${ft}_seed${seed}.yaml'
with open(tmp, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print(tmp)
" > /tmp/mu_config_path_${algo}_${author}_${ft}_seed${seed}.txt

                    TMP_CONFIG=$(cat /tmp/mu_config_path_${algo}_${author}_${ft}_seed${seed}.txt)

                    log "    Config: $TMP_CONFIG"
                    log "    Output: $OUTPUT_DIR"
                    log "    SFT adapter: $SFT_ADAPTER"

                    # Run training
                    set +e
                    python3 "$SCRIPT_DIR/$script" \
                        --config "$TMP_CONFIG" \
                        --author "$author" \
                        --forget-type "$ft" \
                        --sft-adapter "$SFT_ADAPTER" \
                        "${EXTRA_FLAGS[@]}" \
                        2>&1 | tee -a "$LOG_FILE"
                    RC=${PIPESTATUS[0]}
                    set -e

                    if [[ $RC -ne 0 ]]; then
                        log "  ERROR: $LABEL failed (exit $RC)"
                        ((FAILED++)) || true
                    else
                        log "  Done: $LABEL"
                    fi

                    # Cleanup temp config
                    rm -f "$TMP_CONFIG" /tmp/mu_config_path_*.txt

                    log ""
                done
            done
        done
    done

    log ""
    log "Phase 2 (MU) complete: $((DONE - SKIPPED - FAILED)) trained, $SKIPPED skipped, $FAILED failed"
fi

# =============================================================================
# Phase 3: Evaluation (91+ checkpoints × 8 metrics)
# Estimated: ~8-12h total
# =============================================================================
if [[ "$PHASE" == "all" || "$PHASE" == "eval" ]]; then
    phase_header "3: EVALUATION" "91+ checkpoints × 8 metrics, ~8-12h total"

    EVAL_RESULTS="$RESULTS_DIR/multiseed_eval_results.json"
    LOG_FILE="$LOG_DIR/evaluate_all.log"

    log "Running full evaluation..."
    log "  This evaluates ALL discovered checkpoints with 8 metrics:"
    log "  forget_ppl, retain_ppl, forget_rouge_l, retain_rouge_l,"
    log "  truth_ratio, mia_auc_20, extraction_strength, forget_quality_ks"
    log ""

    # Use evaluate.py with --eval-all to discover and evaluate all checkpoints
    python3 "$SCRIPT_DIR/evaluate.py" \
        --config "$CONFIG" \
        --eval-all \
        2>&1 | tee -a "$LOG_FILE"

    check_exit ${PIPESTATUS[0]} "Evaluation"

    log "Phase 3 (Evaluation) complete"
fi

# =============================================================================
# Phase 4: β Sensitivity Sweep (6 runs: 2 β values × 3 seeds)
# NPO only, author=Berthe, forget_type=line
# Estimated: ~2h total
# =============================================================================
if [[ "$PHASE" == "all" || "$PHASE" == "beta-sweep" ]]; then
    phase_header "4: β SENSITIVITY SWEEP" "6 runs (2 β values × 3 seeds), NPO/Berthe/line, ~2h"

    for seed in "${SEEDS[@]}"; do
        for beta in "${BETA_VALUES[@]}"; do
            LABEL="npo-sweep/beta=${beta}/Berthe/line/seed=${seed}"
            OUTPUT_DIR="$RESULTS_DIR/checkpoints/npo-sweep/beta${beta}_Berthe_line_seed${seed}"

            if [[ -d "$OUTPUT_DIR/final" ]]; then
                log "SKIP: $LABEL (checkpoint exists)"
                continue
            fi

            log ">>> $LABEL"

            mkdir -p "$OUTPUT_DIR"
            LOG_FILE="$LOG_DIR/train_npo_sweep_beta${beta}_seed${seed}.log"

            # Create temp config with modified beta and seed
            python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['seed'] = $seed
cfg['npo']['beta'] = $beta
cfg['npo']['output_dir'] = '$OUTPUT_DIR'
with open('/tmp/npo_sweep_beta${beta}_seed${seed}.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
"

            SFT_ADAPTER="$RESULTS_DIR/checkpoints/sft_seed${seed}"
            [[ ! -d "$SFT_ADAPTER" ]] && SFT_ADAPTER="$RESULTS_DIR/checkpoints/sft"

            set +e
            python3 "$SCRIPT_DIR/train_npo.py" \
                --config "/tmp/npo_sweep_beta${beta}_seed${seed}.yaml" \
                --author "Berthe" \
                --forget-type "line" \
                --sft-adapter "$SFT_ADAPTER" \
                2>&1 | tee -a "$LOG_FILE"
            RC=${PIPESTATUS[0]}
            set -e

            rm -f "/tmp/npo_sweep_beta${beta}_seed${seed}.yaml"

            if [[ $RC -ne 0 ]]; then
                log "ERROR: $LABEL failed (exit $RC)"
            else
                log "Done: $LABEL"
            fi
            log ""
        done
    done

    log "Phase 4 (β sweep) complete"
fi

# =============================================================================
# Phase 5: Post-Validation
# =============================================================================
if [[ "$PHASE" == "all" || "$PHASE" == "verify" ]]; then
    phase_header "5: POST-VALIDATION" "Run verify_experiments.py"

    python3 "$SCRIPT_DIR/verify_experiments.py" \
        --config "$CONFIG" \
        --results-dir "$RESULTS_DIR" \
        2>&1 | tee -a "$MASTER_LOG"

    check_exit ${PIPESTATUS[0]} "Post-validation"
fi

# =============================================================================
# Final Summary
# =============================================================================
log_separator
log "  EXPERIMENT PIPELINE COMPLETE"
log_separator
log ""
log "Results directory: $RESULTS_DIR"
log "Logs: $LOG_DIR"
log "Master log: $MASTER_LOG"
log ""
log "Next steps:"
log "  1. Run: python3 verify_experiments.py --config $CONFIG"
log "  2. Run: python3 fill_pending.py --results-dir $RESULTS_DIR"
log "  3. Copy results back: rsync -avz GPU_HOST:$RESULTS_DIR/ $RESULTS_DIR/"
log ""
log "Finished at: $(date)"
