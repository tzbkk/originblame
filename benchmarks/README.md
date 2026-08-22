# OriginBlame Benchmarks

> This directory contains benchmarks, the machine unlearning experiment, and evaluation scripts. Paper source lives in `paper/full/` (full paper) and `paper/cikm/` (CIKM demo). The implementation of OriginBlame (the `ob` CLI and Python package) lives in [rust-originblame](https://github.com/tzbkk/rust-originblame).

Evaluation scripts for machine unlearning (Step 1) and pipeline benchmarks (Step 2).

## Data Sources

### Zhwiki XML Dumps

The pipeline ingests Chinese Wikipedia XML dumps (zhwiki, 2026-04-01). `.7z` files support streaming decompression — no extraction needed.

```bash
# Official source: https://dumps.wikimedia.org/zhwiki/20260401/
# Mirror: https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/zhwiki/20260401/
wget https://dumps.wikimedia.org/zhwiki/20260401/zhwiki-20260401-pages-meta-history5.xml-p3391030p3581891.7z
wget https://dumps.wikimedia.org/zhwiki/20260401/zhwiki-20260401-pages-meta-history5.xml-p3581892p3807013.7z
wget https://dumps.wikimedia.org/zhwiki/20260401/zhwiki-20260401-pages-meta-history5.xml-p3807014p4041824.7z
wget https://dumps.wikimedia.org/zhwiki/20260401/zhwiki-20260401-pages-meta-history5.xml-p4041825p4396533.7z
wget https://dumps.wikimedia.org/zhwiki/20260401/zhwiki-20260401-pages-meta-history5.xml-p4396534p4708433.7z
wget https://dumps.wikimedia.org/zhwiki/20260401/zhwiki-20260401-pages-meta-history5.xml-p4708434p4992433.7z
wget https://dumps.wikimedia.org/zhwiki/20260401/zhwiki-20260401-pages-meta-history5.xml-p4992434p5198583.7z
wget https://dumps.wikimedia.org/zhwiki/20260401/zhwiki-20260401-pages-meta-history5.xml-p5198584p5387514.7z
```

Place in `benchmarks/raw_data/` (gitignored). Filters: article namespace only (ns=0), pages <50 chars wikitext skipped, pages with no contributors skipped.

**Dataset size:** 219,555 wiki pages, 482,543 unique contributors (zhwiki 2026-04-01 dump).

Parsing is handled by `pipeline_v2/mediawiki_parser.py` (imported by both Datatrove and HuggingFace pipeline steps). Pipeline v2 runs parse + generate + eval end-to-end.

### Kernel Source

Linux kernel git blame data requires a deep clone:

```bash
git clone https://mirrors.ustc.edu.cn/linux.git benchmarks/raw_data/linux
# Pin to the exact commit used in our experiments for reproducibility:
cd benchmarks/raw_data/linux && git checkout e75a43c7cec459a07d91ed17de4de13ede2b7758
```

Kernel HEAD at experiment time: `e75a43c7cec4` (2026-04-29, v7.1-rc1 merge window).

**Dataset size:** 44,222 source files (.c/.h/.S/.rs/.dts/.dtsi), 6,964 unique authors.

Attribution uses `pipeline_v2/kernel_parser.py` which runs `git blame -e` on top N `.c`/`.h` files.

## Prerequisites

- Python >= 3.12
- ob installed from [rust-originblame](https://github.com/tzbkk/rust-originblame): `cd rust-originblame/python && pip install .`
- Or add to PYTHONPATH: `PYTHONPATH=path/to/rust-originblame/python/src`

## Pipeline Steps

The OriginBlame tracking pipeline follows these steps:

1. **Track** — Register authors and sections, then track each data line during processing
2. **Blame** — Query provenance for specific lines or files
3. **Revoke** — Mark author/section/document claims as revoked (soft delete)
4. **Reconcile** — Recover provenance after data edits (two-phase: hash matching + embedding similarity)
5. **Purge** — Physically delete revoked data from files (hard delete)

The `reconcile` operation supports two modes:
- **Hash-only** (fast): Uses exact SHA-256 hash matching to recover unchanged lines
- **Hash + Embedding** (semantic): Adds embedding-based similarity matching for edited lines
  - Requires local embedding API at `http://localhost:1234/v1` (OpenAI-compatible)
  - Model: `nomic-embed-text-v1.5`, cosine threshold: 0.85
  - Use `--compute-all-embeddings` to precompute embeddings for all lines

**Cross-domain evaluation:** The same pipeline pattern works across both wiki-style text (zhwiki) and source code (Linux kernel), validating generalization.

## Step 1: Machine Unlearning Evaluation

Tests whether ob's line-level provenance produces better forget sets for machine unlearning than random attribution.

**Experimental design:** 3 forget set types × 2 unlearning algorithms × 3 authors = 18 unlearn + 9 retrain runs. Uses Qwen3-1.7B with full fine-tuning (bf16) on ChatML QA data generated from zhwiki via a locally deployed Qwen3.5-9B (the full paper used Zhipu GLM-4-Flash).

### Setup

```bash
cd benchmarks
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model (Qwen3-1.7B) must be at `benchmarks/models/qwen3-1.7b/`:

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-1.7B --local-dir benchmarks/models/qwen3-1.7b
```

QA data generation uses a locally deployed Qwen3.5-9B served at `http://localhost:1234/v1` (e.g., LM Studio or vLLM). No API key is required. (The full paper used the Zhipu GLM-4-Flash API instead.)

### Pipeline Phases

| Phase | Script | Purpose |
|-------|--------|---------|
| Data | `node1_generate.py` | Wiki text → 3-5 QA pairs/doc via local Qwen3.5-9B + ob provenance |
| 1 | `build_forget_sets.py` | Construct forget/retain splits from ob provenance |
| 2 | `train_sft.py` | Full FT SFT on ChatML QA data (shared base checkpoint) |
| 3 | `train_npo.py / train_rmu.py` | Unlearning (18 runs) |
| 4 | `evaluate.py` | PPL + ROUGE-L evaluation across all checkpoints |

```bash
# Generate QA data (prerequisite, ~14-35h for 50k docs)
python3 benchmarks/scripts/pipeline_qa/node1_generate.py --concurrency 20

# Run full MU pipeline
python3 benchmarks/scripts/pipeline_qa/run_all.py --config benchmarks/scripts/pipeline_qa/config.yaml
```

### Forget Set Types

| Type | Method | Description |
|------|--------|-------------|
| `line` | ob provenance | Exact line-level tracking of author contributions |
| `page_prototype` | ob provenance | Page-level attribution: one representative page per author's contributions |
| `random` | Random sampling | Same-size random subset (baseline for chance-level attribution) |

### Unlearning Algorithms

- **NPO** (Negative Preference Optimization): DPO without positive samples. Decreases likelihood of forget set outputs relative to reference policy. `beta=0.1`, `lr=1e-5`, 5 epochs.
- **RMU** (Representational Misdirection): Steers hidden states at a target layer toward random direction for forget inputs while preserving retain representations. `target_layer=14`, `alpha=100`, `steering_coeff=20`.

### Reproducibility

All hyperparameters, seeds, and model configuration live in a single file: [`config.yaml`](scripts/pipeline_qa/config.yaml).

- **Seed**: `seed: 42` (configurable) — used by all training scripts, forget set random sampling, and RMU control vector generation.
- **Model**: Qwen3-1.7B (public, fixed weights).
- **Data source**: zhwiki 2026-04-01 dump (URLs in Data Sources section above), kernel pinned to commit `e75a43c7cec4`.
- **Non-deterministic step**: QA data generation (`node1_generate.py`) calls the locally deployed Qwen3.5-9B model, which may produce different QA pairs across runs. All downstream steps (SFT, forget sets, unlearning, evaluation) are fully deterministic given the same QA data and seed.

### Evaluation Metrics

| Metric | Description | Desired Direction |
|--------|-------------|-------------------|
| Forget PPL | Perplexity on forget set | Higher = better unlearning |
| Retain PPL | Perplexity on retain set | Lower = better utility preservation |
| Forget ROUGE-L | ROUGE-L on forget set outputs | Lower = better forgetting |
| Retain ROUGE-L | ROUGE-L on retain set | Higher = better preservation |

## Step 2: Pipeline v2 (Cross-Framework Integration Benchmark)

Evaluates ob integration overhead in both Datatrove and HuggingFace Datasets pipelines, on zhwiki and Linux kernel data, with query benchmarks (line-level + token-level).

### Run Pipeline Benchmarks

```bash
cd benchmarks
source .venv/bin/activate

# Run all 28 configurations (zhwiki (4 scales: 1k/10k/100k/all) + kernel (3 scales: 1k/10k/all), 2 frameworks)
python3 scripts/pipeline_v2/run_all.py

# Or run individual pipelines:
python3 scripts/pipeline_v2/run_datatrove.py --data-source zhwiki --scale 1k
python3 scripts/pipeline_v2/run_huggingface.py --data-source zhwiki --scale 10k
python3 scripts/pipeline_v2/run_datatrove.py --data-source kernel --scale 1k

# View summary of completed runs
cat results/pipeline_v2/summary.json
```

Each run produces a pipeline.log with metrics (wall time, storage, throughput, ob_metrics) in `benchmarks/results/pipeline_v2/<run-id>/`.

### Run Query Benchmarks

```bash
cd benchmarks
source .venv/bin/activate

python3 scripts/pipeline_v2/bench_query.py
```

Queries all OB datasets with 5-run averages for blame, show, revoke, purge, token_show, token_revoke, and generate_set. Results in `benchmarks/results/pipeline_v2/query_bench_latest.json`.

### Extract Overhead Summary

```bash
cd benchmarks
source .venv/bin/activate

python3 scripts/pipeline_v2/extract_metrics.py
```

Produces `benchmarks/results/pipeline_v2/overhead_summary.json` with paired A/B metrics (throughput overhead, storage overhead) for all pipeline runs.

### Run Reconcile Benchmarks

```bash
cd benchmarks
source .venv/bin/activate

# Hash-only reconcile (no embedding API required)
python3 scripts/pipeline_v2/bench_eval.py --scale 1k --reconcile

# Hash + embedding reconcile (requires local API at localhost:1234/v1)
# Start embedding API first (e.g., LM Studio, vLLM, or similar)
python3 scripts/pipeline_v2/bench_eval.py --scale 1k --reconcile --use-embeddings --embedding-api http://localhost:1234/v1
```

### Kernel Attribution

Kernel data uses a hybrid attribution approach:
1. `git log --name-only` to quickly filter top N most-committed `.c`/`.h` files
2. `git blame -e` on the top 500 files for accurate line-level authorship (budget-capped)
3. Commit-frequency fallback for files beyond the blame budget

This produces 671 unique authors at 1k files, 1,847 at 6,449 files, 6,964 at all files (~44k C/H/S/rs/dts/dtsi files in the kernel).

## Index Benchmark

The eval script measures both indexed and non-indexed query paths:

| Metric | Description |
|--------|-------------|
| `show_mean_ms` | `ob show --author` without index (full scan) |
| `show_idx_ms` | `ob show --author --index` (bucket-routing index) |
| `purge_ms` | `ob purge --file` without index (revoke+purge) |
| `purge_author_idx_ms` | `ob purge --author --index` (bucket-routing index) |

Index provides 7–22× speedup on show and 3–6× on purge across all scales.

## Results

### Pipeline v2 Summary

All 28 pipeline runs completed successfully (16 zhwiki + 12 kernel configurations):

| Data Source | Scale | Framework | With OB | Time (s) | Throughput Drop |
|-------------|-------|-----------|---------|----------|-----------------|
| zhwiki | 1k | Datatrove | Yes | 17.7 | -13.0% |
| zhwiki | 1k | Datatrove | No | 15.6 | — |
| zhwiki | 10k | Datatrove | Yes | 124.8 | -15.6% |
| zhwiki | 10k | Datatrove | No | 105.0 | — |
| zhwiki | 100k | Datatrove | Yes | 1884.8 | -12.8% |
| zhwiki | 100k | Datatrove | No | 1662.0 | — |
| zhwiki | all (220k pages) | Datatrove | Yes | 4778.5 | -2.1% |
| zhwiki | all (220k pages) | Datatrove | No | 4678.8 | — |
| zhwiki | 1k | HuggingFace | Yes | 15.5 | -1.9% |
| zhwiki | 1k | HuggingFace | No | 15.2 | — |
| zhwiki | 10k | HuggingFace | Yes | 109.9 | -1.4% |
| zhwiki | 10k | HuggingFace | No | 107.3 | — |
| zhwiki | 100k | HuggingFace | Yes | 1755.0 | -6.4% |
| zhwiki | 100k | HuggingFace | No | 1732.5 | — |
| zhwiki | all (220k pages) | HuggingFace | Yes | 4288.3 | -3.8% |
| zhwiki | all (220k pages) | HuggingFace | No | 4122.3 | — |
| kernel | 1k | Datatrove | Yes | 339.6 | -25.9% |
| kernel | 1k | Datatrove | No | 270.1 | — |
| kernel | 10k | Datatrove | Yes | 435.0 | -41.2% |
| kernel | 10k | Datatrove | No | 308.7 | — |
| kernel | all (44k files) | Datatrove | Yes | 733.0 | — |
| kernel | all (44k files) | Datatrove | No | 249.1 | — |
| kernel | 1k | HuggingFace | Yes | 240.7 | -0.0% |
| kernel | 1k | HuggingFace | No | 240.3 | — |
| kernel | 10k | HuggingFace | Yes | 243.9 | -0.0% |
| kernel | 10k | HuggingFace | No | 241.5 | — |
| kernel | all (44k files) | HuggingFace | Yes | 250.8 | -1.0% |
| kernel | all (44k files) | HuggingFace | No | 244.8 | — |

**Storage overhead:** 1.01–1.06× across all scales and data sources.

**Cross-domain kernel results:** 671 authors at 1k files, 5,285 at 10k files, 6,964 at all files.

### Revocation Precision

> Both papers run the same pipeline (`run_huggingface.py --scale 10k`); the tables below come from independent dataset generation runs whose section chunking produced different records/page and author shares. Chunking is deterministic given the dump and parser (`ob-util` mediawiki parser: heading-based split, ≥400-char merge) but is not a CLI parameter.

#### Full Paper dataset (zhwiki, 10k records, ~1 record/page)

| Revoking Author | Share | Lines Removed | Over-deletion (dataset-level) |
|-----------------|-------|---------------|---------------------------|
| InternetArchiveBot | 79.5% | 7,953 | 1.3× |
| Walter Grassroot   | 17.1% | 1,712 | 5.8× |
| KLBot2             | 5.0%  | 499   | 20.0× |
| HuangQQ            | 1.0%  | 99    | 101.0× |

#### CIKM Demo Paper dataset (zhwiki, 10k records, ~2 records/page)

| Revoking Author | Share | Lines Removed | Over-deletion (file-level) |
|-----------------|-------|---------------|---------------------------|
| InternetArchiveBot | 36.2% | 3,618 | 2.8× |
| Fozuxilai | 12.0% | 1,199 | 8.3× |
| Ohtashinichiro | 5.4% | 541 | 18.5× |
| Berthe | 2.2% | 223 | 44.8× |

### Scalability (220k lines, zhwiki, HuggingFace pipeline)

| Operation | Mean Latency (ms) | Indexed Latency (ms) |
|-----------|------------------:|---------------------:|
| blame | 1 | — |
| show | 98 | 95 |
| show (index) | — | 95 |
| revoke | <1 | — |
| purge | 196 | 106 |
| purge (index) | — | 106 |

**Storage overhead:** 0.22× at 220k lines (downward trend across scales).

### Machine Unlearning

> Both papers use the same second-generation MU evaluation: full supervised fine-tuning (bf16), 3 authors, both NPO and RMU (3 forget set types × 2 algorithms × 3 authors = 18 unlearn + 9 retrain runs).
>
> **v1 withdrawal (arXiv, May 2026):** the full paper originally reported QLoRA (rank=16, 4-bit NF4) results on 2 authors, withdrawn in v2 — QLoRA's low-rank updates suppress surface-level patterns without affecting deeper representations, producing unreliable unlearning signals (RMU was additionally incompatible).

Wiki authors edit pages across unrelated topics, making this the most adversarial setting for unlearning: individual contributions lack semantic cohesion.

#### Full SFT Results (used by both papers as of full-paper v2)

**Bold** rPPL = provenance beats random within same algorithm.

| Algo. | Author | Set | fPPL↑ | rPPL↓ | fROUGE↓ | rROUGE↑ | MIA→0.5 | Ext↓ |
|-------|--------|-----|------:|------:|--------:|--------:|-----:|-----:|
| | SFT | — | 4.5 | 5.0 | 0.21 | 0.14 | 0.51 | 0.76 |
| NPO | IAB (35.9%) | line | 8.3 | **7.0** | 0.09 | 0.10 | 0.77 | 0.50 |
| NPO | | page | 8.3 | 7.5 | 0.07 | 0.15 | 0.61 | 0.64 |
| NPO | | rand | 8.1 | 7.5 | 0.13 | 0.13 | 0.58 | 0.05 |
| RMU | | line | 4.5 | 5.0 | 0.21 | 0.15 | 0.51 | 0.88 |
| RMU | | page | 4.5 | 5.2 | 0.14 | 0.19 | 0.43 | 0.90 |
| RMU | | rand | 4.8 | 4.8 | 0.19 | 0.20 | 0.53 | 0.92 |
| NPO | Ohta (5.4%) | line | 7.3 | **6.5** | 0.09 | 0.10 | 0.67 | 0.79 |
| NPO | | page | 7.2 | **5.5** | 0.11 | 0.15 | 0.83 | 0.96 |
| NPO | | rand | 8.3 | 7.4 | 0.12 | 0.09 | 0.54 | 0.15 |
| RMU | | line | 3.9 | 4.8 | 0.18 | 0.16 | 0.42 | 0.74 |
| RMU | | page | 2.9 | 4.8 | 0.29 | 0.14 | 0.24 | 0.80 |
| RMU | | rand | 4.9 | 4.7 | 0.18 | 0.16 | 0.52 | 0.73 |
| NPO | Fozuxilai (11.3%) | line | 7.3 | **6.3** | 0.26 | 0.11 | 0.71 | 0.88 |
| NPO | | page | 7.8 | **5.9** | 0.19 | 0.17 | 0.78 | 0.92 |
| NPO | | rand | 8.2 | 7.5 | 0.07 | 0.10 | 0.47 | 0.45 |
| RMU | | line | 3.5 | 5.1 | 0.29 | 0.12 | 0.27 | 0.84 |
| RMU | | page | 3.0 | 5.1 | 0.28 | 0.18 | 0.23 | 0.82 |
| RMU | | rand | 4.8 | 4.8 | 0.10 | 0.14 | 0.44 | 0.87 |

**Key findings:** Under NPO, provenance-based forget sets achieve comparable forgetting at consistently lower retain PPL than random across all three authors — 7–16% lower for line-level, up to 26% for page-level (five of six pairwise comparisons favor provenance, one ties). The MIA column is consistent with the mechanism: provenance sets (0.61–0.83 AUC) select genuinely memorized content, whereas random sets (0.47–0.58) dilute the unlearning signal with content the model barely learned. Random achieves lower extraction, but this reflects dilution. RMU at these hyperparameters produces weak forgetting regardless of set type (extraction 0.73–0.92). Caveats: single-run results (seed 42), and the retain-PPL advantage does not replicate on retain ROUGE-L — the direction of the effect, not its magnitude, is the claim. Even with provenance sets, extraction remains high (transformation gap): pointing at the source is necessary but not sufficient for erasure. In semantically coherent deletion scenarios (e.g., a painter's works), provenance's advantage would be substantially larger.

## Completed Tasks

- [x] Pipeline v2: all 28 runs completed (zhwiki (4 scales: 1k/10k/100k/all) + kernel (3 scales: 1k/10k/all), 2 frameworks)
- [x] Cross-domain evaluation on Linux kernel (44,222 files, 6,964 authors)
- [x] Reconcile benchmark with hash + embedding matching (1k–220k scales)
- [x] Revocation precision evaluation (4 authors, 1k–220k scales)
- [x] Machine unlearning evaluation (3 forget set types × 2 algorithms × 3 authors = 18 unlearn + 9 retrain runs)
- [x] Query benchmarks (line-level + token-level) across all scales
- [x] Index system with bucket-routing (7–22× speedup on show, 3–6× on purge)
- [x] Storage overhead evaluation (1.01–1.06× across all scales)

## Pending Tasks

- [ ] Embedding write/read correctness test
- [ ] Run 50k/220k reconcile benchmark with embeddings (time-intensive)

## Directory Structure

```
benchmarks/
  raw_data/          # zhwiki XML dump + Linux kernel (not tracked by git)
    zhwiki-20260401-pages-meta-history5.xml-*.7z
    linux/           # Deep clone pinned to e75a43c7cec4
  .venv/             # Python virtual environment
  results/
    pipeline_v2/     # Step 2 results (28 runs completed)
      summary.json   # Summary of all pipeline runs
      datatrove-zhwiki-{1k,10k,100k,all}{,-ob}/
      huggingface-zhwiki-{1k,10k,100k,all}{,-ob}/
      datatrove-kernel-{1k,10k,all}{,-ob}/
      huggingface-kernel-{1k,10k,all}{,-ob}/
      query_bench_*.json
      kernel_revocation_precision.json
    pipeline_qa/     # Step 1 results (QA data, checkpoints, eval)
  scripts/
    pipeline_v2/     # Cross-framework integration benchmark
      run_all.py               # Run all 28 configurations
      run_datatrove.py         # Datatrove pipeline runner
      run_huggingface.py       # HuggingFace pipeline runner
      shared.py                # Shared utilities (types, tokenizer, helpers)
      datatrove_steps.py       # Datatrove pipeline steps
      hf_steps.py              # HuggingFace pipeline steps
      kernel_parser.py         # Linux kernel git blame attribution
      mediawiki_parser.py      # Streaming parser for MediaWiki XML dumps
      _native_compat.py        # PyO3 compat layer for Rust-backed OB ops
      bench_eval.py            # Scalability, revocation, reconcile benchmark
      bench_query.py           # Query benchmark (line + token)
      bench_kernel_reconcile.py # Kernel reconcile: provenance recovery
      bench_kernel_revocation.py # File-level vs record-level precision
      extract_metrics.py       # Overhead metrics extraction
    pipeline_qa/      # MU experiment pipeline
      node1_generate.py       # Wiki → local Qwen3.5-9B → ChatML QA + .ob/ provenance
      build_forget_sets.py    # Construct forget/retain splits from ob provenance
      train_sft.py            # Full FT SFT on ChatML QA data
      train_npo.py            # NPO unlearning
      train_rmu.py            # RMU unlearning
      run_all.py              # Orchestrator (checkpoint/resume, runs all phases)
      config.yaml             # All hyperparameters and paths
      evaluate.py             # PPL + ROUGE-L evaluation
```

## Eval Metrics

Each scale produces:
- `generation.sections_used`, `section_coverage_pct` - guaranteed 100% unless LLM retries exhausted
- `generation.llm_calls`, `llm_successes`, `llm_failures`, `quality_filtered`, `dedup_filtered`
- `generation.track_time_ms`, `clean_ms`
- `eval.blame_latency` - min/mean/max ms
- `eval.show_latency` - mean ms
- `eval.revocation` - revoke+purge timing
- `eval.storage` - data size, ob size, overhead ratio
- `reconcile.reconcile.hash_matched` - lines matched by exact hash
- `reconcile.reconcile.semantic_matched` - lines matched by embedding similarity
- `reconcile.reconcile.new_lines` - unmatched lines needing manual tracking
- `reconcile.reconcile.orphans` - old document-index records with no matching line
- `reconcile.reconcile.seed` - mutation seed (deterministic, default 42)
- `reconcile.reconcile.model` - embedding model name
- `reconcile.reconcile.threshold` - cosine similarity threshold
- `reconcile.mutation` - original_lines, edited_lines, deleted_lines, inserted_lines, final_lines

## Reconcile Results (seed=42, nomic-embed-text-v1.5, threshold=0.85)

| Scale  | orig  | edit | hash  | sem  | new | orph | recovery | time(s) |
|--------|-------|------|-------|------|-----|------|----------|---------|
| 1k     | 1000  |   90 |   865 |  104 |  36 |   31 | 96.9%    | 4.1     |
| 10k    | 10000 |  986 |  8479 | 1267 | 219 |  254 | 97.5%    | 92.8    |

Recovery improves with scale (96.9% → 97.5%), confirming that the two-phase reconcile is robust to dataset size.
