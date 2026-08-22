# OriginBlame

**Record- and token-level data provenance for AI training datasets.**

When a data contributor requests removal, model trainers face a practical gap: unlearning algorithms require a forget set, yet no tool can locate which training records belong to a given author. Existing provenance systems operate at file or dataset level, forcing catastrophic over-deletion. We present `ob`, a record- and token-level data provenance system that propagates author identity through data processing pipelines and resolves revocation requests into precise forget sets via deterministic queries. Evaluation on 219,555 Wikipedia pages demonstrates that record-level provenance eliminates dataset-level over-deletion (from 101× to 1.3×), while integration adds 1.3–4.0% throughput overhead (HuggingFace) and 2.1–19.0% (Datatrove) on wiki data. On a 1.7B model, provenance-based forget sets consistently reduce unlearning collateral damage (retain perplexity) relative to same-size random baselines across all evaluated authors, with MIA indicating they select genuinely memorized content (single-run results; see paper v2 §5.5 for caveats).

## Install

Source code lives in the [rust-originblame](https://github.com/tzbkk/rust-originblame) repository, which contains both the Rust native implementation and the Python package.

## How It Works

OriginBlame tracks provenance at **data collection time** -- not retroactively. It stores metadata in `.ob/` inside your repository, using plain JSONL files organized into three layers:

```
.ob/
  authors/          # who: name, email, id = sha256(name+email)
  sections/          # what: file path + authors + license, sharded by sha256
  document-index/    # which line came from where: (line_hash, file, sources)
  token-index.gpt2/ # how many tokens each document contributed (per tokenizer)
  index/            # binary index: id → refs[] (bucket prefixes + token ranges)
  log               # operation audit trail
```

- **Content-addressable**: every record is indexed by SHA-256 hash. No IDs to manage, no central database.
- **Decentralized**: metadata lives in your repo. No server, no config files, no external state.
- **Zero ML dependencies**: the core `ob` package has no ML imports. Optional `ob-util` adds parsers and embedding-based reconciliation.
- **Reconcile after edits**: when data files change, `ob reconcile` uses hash matching (Pass 1) and optional embedding similarity (Pass 2) to re-link provenance to modified lines.

## Webapp Demo

A FastAPI + React webapp showcasing record-level provenance for the CIKM 2026 Demo Track:

```bash
# Backend (FastAPI)
cd webapp/backend && pip install -r requirements.txt
OB_DIR=../../benchmarks/results/pipeline_v2/huggingface-zhwiki-1k-ob uvicorn server:app --reload

# Frontend (React + Vite)
cd webapp/frontend && npm install && npm run dev
```

Data: `benchmarks/results/pipeline_v2/huggingface-zhwiki-1k-ob/` (1,000 Wikipedia records, 4,278 authors, CC-BY-SA-4.0).

Pages:
- **Dataset Overview** — summary statistics, top authors, section distribution
- **Authors** — searchable author table with paginated record preview, cross-page navigation
- **Records** — filterable record browser with detail panel showing full provenance chain
- **Right-to-Erasure** — 3-level revocation (Author / Section / Record) with confirmation dialogs

For production deployment with Nginx, see `webapp/deploy/`.

## Paper

**OriginBlame: Record- and Token-Level Data Provenance for AI Training Datasets**

**Full paper:** [arXiv:2607.13037](https://arxiv.org/abs/2607.13037)  
LaTeX source: `paper/full/originblame.tex`

### Reproducibility

Paper compiles with `cd paper/full && pdflatex originblame.tex && bibtex originblame && pdflatex originblame.tex && pdflatex originblame.tex`. Benchmark reproduction requires:

| Resource | Size | Source |
|----------|------|--------|
| zhwiki XML dump | ~2 GB | [Wikimedia dumps](https://dumps.wikimedia.org/zhwiki/20260401/) (8 `.7z` files; URLs in `benchmarks/README.md`) |
| Qwen3-1.7B model | 3.8 GB | `huggingface-cli download Qwen/Qwen3-1.7B --local-dir benchmarks/models/qwen3-1.7b` |
| Linux kernel | ~4 GB | `git clone https://mirrors.ustc.edu.cn/linux.git && git checkout e75a43c7cec459a07d91ed17de4de13ede2b7758` |
| Qwen3.5-9B (local) | — | QA data generation in MU experiments: served at `http://localhost:1234/v1` (LM Studio / vLLM) |
| Embedding API | — | Required for semantic reconcile only: OpenAI-compatible API at `http://localhost:1234/v1` (LM Studio / vLLM with `nomic-embed-text-v1.5`) |

All pipeline MAU unlearning results are fully deterministic given the same QA data, seed (42), and model weights. Hash-only reconcile and all query benchmarks require no API keys. See `benchmarks/README.md` for full setup instructions.

### Results

> **Two papers, two experimental conditions.** Results below are drawn from two papers describing OriginBlame, run under different conditions:
> - **Full paper (v2)**: pipeline v2 (1 record/page, 10k records at 10k scale), full-parameter fine-tuning (bf16) — same MU regime as the CIKM demo paper, streaming token-index storage measurement, 3-run avg latency. v1 reported QLoRA-based MU numbers, withdrawn in v2 as unreliable (see below).
> - **CIKM 2026 demo paper**: pipeline v2 (2 records/page, 20k records at 10k scale), full supervised fine-tuning (bf16), JSONL storage measurement, single-run latency.
>
> Where the two papers report different numbers under their respective conditions, both are shown side by side. Where they share data (token-level streaming and cross-domain kernel experiments), a single table is labeled accordingly.

Evaluated on a Chinese Wikipedia dump (219,555 pages, 482,543 contributors) at four scales (1k–220k pages):

**Revocation Precision** — Line-level provenance eliminates over-deletion. The two papers measured on different pipeline versions and record counts.

#### Full Paper (zhwiki, pipeline v2, 10k records)

| Revoking Author    | Share | Lines Removed (ob) | Over-deletion (dataset-level) |
|--------------------|-------|--------------------:|------------------------------:|
| InternetArchiveBot | 79.5% | 7,953               | 1.3×                          |
| Walter Grassroot   | 17.1% | 1,712               | 5.8×                          |
| KLBot2             | 5.0%  | 499                 | 20.0×                         |
| HuangQQ            | 1.0%  | 99                  | 101.0×                        |

#### CIKM Demo Paper (zhwiki, pipeline v2, 20k records)

| Revoking Author    | Share | Lines Removed (ob) | Over-deletion (dataset-level) |
|--------------------|-------|--------------------:|------------------------------:|
| InternetArchiveBot | 18.1% | 3,618               | 2.8×                          |
| Mid-share Editor   | 6.0%  | 1,199               | 8.3×                          |
| Ohtashinichiro     | 2.7%  | 541                 | 18.5×                         |
| Berthe             | 1.1%  | 223                 | 44.8×                         |

**Reconcile Recovery** (after 10% edit + 5% delete + 5% insert mutation):

| Scale | Hash Match | Semantic Match | Recovery |
|------:|-----------:|---------------:|---------:|
| 1k    | 865        | 104            | 96.9%    |
| 10k   | 8,479      | 1,267          | 97.5%    |
| 100k  | 84,821     | 13,222         | 98.2%    |
| 100k  | 84,821     | —              | 84.9%†   |

†Hash-only (Pass 1). Semantic matching was not measured at these scales due to embedding API throughput constraints.

**Scalability** — The full paper reports 3-run averages; the CIKM demo paper reports single-run measurements under different storage and pipeline conditions.

#### Full Paper (3-run avg, native Rust, ms)

| Scale | blame | show | show_idx | revoke | purge | purge_idx |
|------:|------:|-----:|-----:|------:|-----:|-----:|
| 1k    | 1 | 3 | 3 | <1 | 0.6 | 3 |
| 10k   | 1 | 9 | 10 | <1 | 0.7 | 41 |
| 100k  | 1 | 33 | 34 | <1 | 5.8 | 106 |
| 220k  | 3 | 80 | 78 | <1 | 12 | 190 |

†Synthetic benchmark. All operations sub-100ms at 220k lines.

#### CIKM Demo Paper (single-run)

At 220k records: **blame 0.66 ms**, **show 178 ms**, **revoke 29 ms**.

Storage overhead: decreases with scale from 0.32× at 1k lines to 0.22× at 220k lines. Line coverage: 100% at all scales.

**Token-Level Streaming Benchmark** — Real gpt2 tokenization on zhwiki data, no JSONL produced. *Shared by both papers (pipeline_v2 streaming tokenization runs).*

| Pages | Tokens | Datatrove Drop | HF Drop | Storage (Datatrove) | Query (ms) |
|------:|-------:|---------------:|--------:|-------------------:|-----------:|
| 1k | 2.8M | −13.8% | −2.0% | 1.33× | 3 |
| 10k | 25.9M | −19.0% | −2.5% | 1.29× | 9 |
| 100k | 302.4M | −13.4% | −1.3% | 1.24× | 33 |
| 219,555 | 712.4M | −2.1% | −4.0% | 1.23× | 69 |

**Machine Unlearning Evaluation** — Tests whether ob's provenance-based forget sets enable effective machine unlearning.

- **Full paper (v2) and CIKM demo paper**: identical full supervised fine-tuning regime (bf16) on 3 authors under both NPO and RMU. 27 experiments (3 forget set types × 2 algorithms × 3 authors + 9 retrain oracle).
- **v1 (arXiv, May 2026)**: reported QLoRA (rank=16, 4-bit NF4) results on 2 authors. **Withdrawn in v2**: QLoRA's low-rank updates suppress surface-level patterns without affecting deeper representations, producing unreliable unlearning signals. RMU was additionally incompatible with QLoRA.

We deliberately use wiki authors (edits scattered across unrelated topics) as the most adversarial setting: individual contributions lack semantic cohesion. In semantically coherent scenarios (e.g., a painter's complete works), provenance's precision advantage would be substantially larger.

#### Full SFT Results (full paper v2 and CIKM demo paper)

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

**CIKM key finding:** Provenance-based forget sets consistently reduce collateral damage: retain PPL is 5–20% lower than random across all three authors, confirming that record-level targeting protects unrelated knowledge better than blind deletion. However, extraction remains high (0.50–0.88 under NPO; 0.74–0.88 under RMU) even when the model is trained to suppress the target data. This reveals a transformation gap: provenance identifies which records an author contributed, but the model has already absorbed that content into its generative distribution—pointing at the source is necessary but not sufficient for erasure. MIA AUC (0.61–0.83 for provenance vs. 0.47–0.58 for random) independently confirms provenance selects genuinely memorized data, while random forget sets dilute the signal with content the model barely learned.

**Cross-Domain Generalization** — Linux kernel source code with git blame attribution (3 scales). *Shared by both papers (kernel experiments).*

| Files | Authors | Datatrove Drop | HF Drop | Storage (Datatrove) | Over-deletion (file vs record) |
|------:|--------:|---------------:|--------:|-------------------:|-------------------------------:|
| 1,000 | 671 | −25.7% | −0.2% | 1.06× | 9× |
| 10,000 | 5,285 | −40.9% | −1.0% | 1.02× | — |
| 44,222 | 6,964 | — | −2.5% | 1.01× | 1.3× |

Attribution uses git blame (line-level authorship) on the top N C/H files from a deep clone of the Linux kernel repository, not git log commit authors. File-level deletion remains wasteful even with accurate attribution: at the smallest scale, revoking Linus Torvalds at file granularity would delete 9× more lines than necessary.


## Repository Structure

This repository (`originblame`) contains the paper and benchmarks:

- `paper/full/` — LaTeX source for the full paper (arXiv)
- `paper/cikm/` — LaTeX source for the CIKM 2026 demo paper
- `benchmarks/` — evaluation scripts and results
- `webapp/` — FastAPI + React demo application with Docker support
