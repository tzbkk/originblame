#!/usr/bin/env bash
# Package everything needed to run RMU full-finetune on a cloud 4090.
# Output: rmu_cloud.tar.gz (~4 GB)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT="$PROJECT_ROOT/rmu_cloud.tar.gz"

echo "=== Packaging RMU full-finetune for cloud 4090 ==="

# Create temp staging directory
STAGING=$(mktemp -d)
trap 'rm -rf "$STAGING"' EXIT

# 1. Model weights (~3.8 GB)
echo "[1/5] Copying model weights..."
mkdir -p "$STAGING/benchmarks/models/qwen3-1.7b"
cp "$PROJECT_ROOT/benchmarks/models/qwen3-1.7b/"*.safetensors \
   "$PROJECT_ROOT/benchmarks/models/qwen3-1.7b/"*.json \
   "$PROJECT_ROOT/benchmarks/models/qwen3-1.7b/"*.txt \
   "$STAGING/benchmarks/models/qwen3-1.7b/"
echo "  Model: $(du -sh "$STAGING/benchmarks/models/qwen3-1.7b" | cut -f1)"

# 2. SFT adapter (only final, no intermediate checkpoints)
echo "[2/5] Copying SFT adapter (final only)..."
mkdir -p "$STAGING/benchmarks/results/pipeline_qa/checkpoints/sft"
for f in adapter_model.safetensors adapter_config.json tokenizer.json \
         tokenizer_config.json chat_template.jinja README.md; do
    src="$PROJECT_ROOT/benchmarks/results/pipeline_qa/checkpoints/sft/$f"
    [ -f "$src" ] && cp "$src" "$STAGING/benchmarks/results/pipeline_qa/checkpoints/sft/"
done
echo "  SFT adapter: $(du -sh "$STAGING/benchmarks/results/pipeline_qa/checkpoints/sft" | cut -f1)"

# 3. Data files
echo "[3/5] Copying data files..."
mkdir -p "$STAGING/benchmarks/results/pipeline_qa/qa_chatml"
cp "$PROJECT_ROOT/benchmarks/results/pipeline_qa/qa_chatml/data.jsonl" \
   "$PROJECT_ROOT/benchmarks/results/pipeline_qa/qa_chatml/forget_sets.json" \
   "$STAGING/benchmarks/results/pipeline_qa/qa_chatml/"
echo "  Data: $(du -sh "$STAGING/benchmarks/results/pipeline_qa/qa_chatml" | cut -f1)"

# 4. Scripts + config
echo "[4/5] Copying scripts..."
mkdir -p "$STAGING/benchmarks/scripts/pipeline_qa"
cp "$SCRIPT_DIR/train_rmu.py" \
   "$SCRIPT_DIR/evaluate.py" \
   "$SCRIPT_DIR/config.yaml" \
   "$SCRIPT_DIR/run_rmu_cloud.sh" \
   "$STAGING/benchmarks/scripts/pipeline_qa/"

# 5. Requirements
echo "[5/5] Writing requirements..."
cat > "$STAGING/requirements.txt" << 'EOF'
torch>=2.1
transformers>=4.40
peft>=0.10
bitsandbytes
accelerate
sentence-transformers
pyyaml
datasets
trl
EOF

cat > "$STAGING/README_CLOUD.txt" << 'EOF'
RMU Full Fine-Tuning on Cloud 4090
===================================

1. Setup:
   pip install -r requirements.txt

2. Run all 4 RMU experiments:
   cd benchmarks/scripts/pipeline_qa
   bash run_rmu_cloud.sh

3. Results will be in:
   benchmarks/results/pipeline_qa/checkpoints/rmu-fullft/
   benchmarks/results/pipeline_qa/rmu_fullft_eval.log

4. Copy results back:
   scp -r cloudhost:~/originblame/benchmarks/results/pipeline_qa/checkpoints/rmu-fullft/ ./benchmarks/results/pipeline_qa/checkpoints/
   scp cloudhost:~/originblame/benchmarks/results/pipeline_qa/rmu_fullft_eval.log ./benchmarks/results/pipeline_qa/
EOF

# Package
echo "Creating tarball..."
tar -czf "$OUTPUT" -C "$STAGING" .
echo "=== Done: $OUTPUT ($(du -sh "$OUTPUT" | cut -f1)) ==="
