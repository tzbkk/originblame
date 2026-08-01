#!/usr/bin/env python3
"""Evaluate unlearned models against forget/retain sets.

Computes eight metrics per model (use --metrics all):
  - Forget PPL:     perplexity on forget-set lines (higher = forgot = good)
  - Retain PPL:     perplexity on retain-set lines (lower = remembers = good)
  - Forget ROUGE-L: generation similarity on forget set (lower = forgot = good)
  - Retain ROUGE-L: generation similarity on retain set (higher = retains = good)
  - Truth Ratio:    P(correct)/(P(correct)+P(wrong)) on forget set (lower = good)
  - MIA AUC-20:     Min-K% Prob membership inference, k=20% (≈0.5 = ideal)
  - Extraction Str: greedy decode second half from first half (lower = good)
  - Forget Quality: KS-test p-value vs retrain oracle (higher = good)

Usage:
    PYTHONPATH=src python3 benchmarks/scripts/pipeline_qa/evaluate.py --config benchmarks/scripts/pipeline_qa/config.yaml --eval-all --metrics all
    PYTHONPATH=src python3 benchmarks/scripts/pipeline_qa/evaluate.py --config benchmarks/scripts/pipeline_qa/config.yaml \
        --author Berthe --forget-type line --algorithm npo --metrics all \
        --checkpoint-path benchmarks/results/pipeline_qa/checkpoints/npo/Berthe_line/final
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── constants ─────────────────────────────────────────────────────────────────

ALGO_NAMES = sorted(["retrain", "npo", "rmu"], key=len, reverse=True)

# ── helpers ───────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(data_file: str) -> list[dict]:
    """Load data.jsonl — each line has {"text": "wiki text...", "title": "...", ...}."""
    records: list[dict] = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_forget_sets(path: str) -> dict:
    """Load forget_sets.json (nested: {author: {type: {indices, count}}})."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def split_forget_retain(
    records: list[dict],
    forget_sets: dict,
    author: str,
    forget_type: str,
) -> tuple[list[dict], list[dict]]:
    """Return (forget_records, retain_records) for a given (author, forget_type)."""
    indices = set(forget_sets[author][forget_type]["indices"])
    forget = [records[i] for i in sorted(indices) if i < len(records)]
    retain = [records[i] for i in range(len(records)) if i not in indices]
    return forget, retain


def parse_checkpoint_key(key: str) -> tuple[str | None, str | None, str | None]:
    """Parse a key like 'npo_InternetArchiveBot_line' into (algo, author, type)."""
    for algo in ALGO_NAMES:
        if key.startswith(algo + "_"):
            rest = key[len(algo) + 1 :]
            for ft in ("random", "line", "page_prototype"):
                if rest.endswith("_" + ft):
                    author = rest[: -(len(ft) + 1)]
                    return algo, author, ft
    return None, None, None


# ── model loading ─────────────────────────────────────────────────────────────


def load_model(
    cfg: dict, model_path: str | None = None, full_model: bool = True
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model: full bf16 model directly, or 4-bit + LoRA adapter."""
    base_path = cfg["model"]["base_path"]
    compute_dtype = getattr(torch, cfg["model"].get("compute_dtype", "bfloat16"))

    tokenizer = AutoTokenizer.from_pretrained(
        base_path, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if full_model:
        load_path = model_path or base_path
        print(f"  Loading full model: {load_path}")
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype,
        )

        if model_path and os.path.isdir(model_path):
            model = PeftModel.from_pretrained(model, model_path)
            print(f"  Loaded adapter: {model_path}")
        else:
            print("  No adapter — evaluating base model weights only")

    model.eval()
    return model, tokenizer


# ── PPL computation ───────────────────────────────────────────────────────────


def _render_text(rec: dict, tokenizer) -> str:
    if "messages" in rec:
        return tokenizer.apply_chat_template(
            rec["messages"], tokenize=False, add_generation_prompt=False
        )
    return rec.get("text", "")


@torch.no_grad()
def compute_ppl(
    model,
    tokenizer,
    records: list[dict],
    device: torch.device,
    batch_size: int = 4,
    max_length: int = 256,
    max_samples: int | None = None,
) -> float:
    model.eval()
    ppls: list[float] = []

    eval_recs = records
    if max_samples and len(eval_recs) > max_samples:
        rng = random.Random(42)
        eval_recs = rng.sample(records, max_samples)

    n_batches = (len(eval_recs) + batch_size - 1) // batch_size

    for bidx in range(n_batches):
        batch = eval_recs[bidx * batch_size : (bidx + 1) * batch_size]
        texts = [_render_text(rec, tokenizer) for rec in batch]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift for causal LM: logits[t] predicts token[t+1]
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]

        per_token_nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).view(shift_labels.size())

        per_token_nll = per_token_nll * shift_mask
        n_valid = shift_mask.sum(dim=1)

        for j in range(len(batch)):
            n = n_valid[j].item()
            if n > 0:
                mean_nll = per_token_nll[j].sum().item() / n
                mean_nll = min(mean_nll, 20.0)  # cap to prevent exp overflow
                ppls.append(math.exp(mean_nll))

        if (bidx + 1) % 25 == 0 or bidx + 1 == n_batches:
            print(f"    PPL batch {bidx + 1}/{n_batches}")

    return sum(ppls) / len(ppls) if ppls else float("inf")


# ── ROUGE-L computation ──────────────────────────────────────────────────────


@torch.no_grad()
def compute_rouge_l(
    model,
    tokenizer,
    records: list[dict],
    device: torch.device,
    max_new_tokens: int = 128,
    max_samples: int | None = None,
    max_length: int = 256,
) -> float:
    """Generate text continuations and compute ROUGE-L against actual continuations."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    eval_recs = records
    if max_samples and len(eval_recs) > max_samples:
        rng = random.Random(42)
        eval_recs = rng.sample(records, max_samples)

    scores: list[float] = []
    for idx, rec in enumerate(eval_recs):
        text = _render_text(rec, tokenizer)
        tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
        split_point = max(1, len(tokens) // 2)
        prompt_tokens = tokens[:split_point]
        reference_text = tokenizer.decode(tokens[split_point:], skip_special_tokens=True)

        input_ids = torch.tensor([prompt_tokens]).to(device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        )

        rouge = scorer.score(reference_text, generated)
        scores.append(rouge["rougeL"].fmeasure)

        if (idx + 1) % 25 == 0 or idx + 1 == len(eval_recs):
            print(f"    ROUGE sample {idx + 1}/{len(eval_recs)}")

    return sum(scores) / len(scores) if scores else 0.0


# ── Truth Ratio ───────────────────────────────────────────────────────────────


def _extract_qa(rec: dict) -> tuple[str, str]:
    """Extract (question, answer) from a record's messages list."""
    question, answer = "", ""
    if "messages" in rec:
        for msg in rec["messages"]:
            if msg["role"] == "user":
                question = msg["content"]
            elif msg["role"] == "assistant":
                answer = msg["content"]
    return question, answer


@torch.no_grad()
def compute_truth_ratio(
    model,
    tokenizer,
    forget_records: list[dict],
    device: torch.device,
    max_length: int = 256,
    max_samples: int | None = None,
) -> list[float]:
    """P(correct) / (P(correct) + P(wrong)) via sigmoid(log_p_correct - log_p_wrong).
    Returns per-item ratios (needed for Forget Quality KS-test).
    Lower = better forgetting.
    """
    ratios: list[float] = []
    model.eval()

    eval_recs = forget_records
    if max_samples and len(eval_recs) > max_samples:
        rng = random.Random(42)
        eval_recs = rng.sample(forget_records, max_samples)

    # Build shuffled answer pool for wrong answers
    all_answers = []
    for rec in forget_records:
        _, ans = _extract_qa(rec)
        if ans:
            all_answers.append(ans)
    shuffled_answers = list(all_answers)
    rng_shuffle = random.Random(42)
    rng_shuffle.shuffle(shuffled_answers)
    shuffle_idx = 0

    for item in eval_recs:
        question, correct_answer = _extract_qa(item)
        if not question or not correct_answer:
            continue

        wrong_answer = shuffled_answers[shuffle_idx % len(shuffled_answers)]
        shuffle_idx += 1

        # P(correct)
        prompt_correct = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": correct_answer},
            ],
            tokenize=False,
        )
        inputs_correct = tokenizer(
            prompt_correct, return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)
        outputs_correct = model(**inputs_correct, labels=inputs_correct["input_ids"])
        log_p_correct = -outputs_correct.loss.item()

        # P(wrong)
        prompt_wrong = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": wrong_answer},
            ],
            tokenize=False,
        )
        inputs_wrong = tokenizer(
            prompt_wrong, return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)
        outputs_wrong = model(**inputs_wrong, labels=inputs_wrong["input_ids"])
        log_p_wrong = -outputs_wrong.loss.item()

        ratio = 1.0 / (1.0 + math.exp(-(log_p_correct - log_p_wrong)))
        ratios.append(ratio)

    return ratios


# ── MIA AUC ───────────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_mia_auc(
    model,
    tokenizer,
    forget_records: list[dict],
    retain_records: list[dict],
    device: torch.device,
    max_length: int = 256,
    k_percent: int = 20,
    max_samples: int | None = None,
) -> float:
    """Min-K% Prob membership inference attack.
    AUC ≈ 0.5 = ideal (can't distinguish forget from retain).
    AUC >> 0.5 = model still remembers forget set membership.
    """
    def get_min_k_score(data: list[dict]) -> list[float]:
        scores: list[float] = []
        model.eval()
        eval_data = data
        if max_samples and len(eval_data) > max_samples:
            eval_data = random.Random(42).sample(data, max_samples)
        for item in eval_data:
            text = _render_text(item, tokenizer)
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            ).to(device)
            outputs = model(**inputs)
            logits = outputs.logits  # (1, seq_len, vocab)

            shift_logits = logits[:, :-1, :]
            shift_labels = inputs["input_ids"][:, 1:]

            probs = torch.softmax(shift_logits, dim=-1)
            token_probs = probs.gather(
                2, shift_labels.unsqueeze(-1)
            ).squeeze(-1).squeeze(0)

            k = max(1, int(len(token_probs) * k_percent / 100))
            min_k_probs, _ = torch.topk(token_probs, k, largest=False)
            min_k_score = -torch.log(min_k_probs).mean().item()
            scores.append(min_k_score)
        return scores

    forget_scores = get_min_k_score(forget_records)
    retain_scores = get_min_k_score(retain_records)

    labels = [1] * len(forget_scores) + [0] * len(retain_scores)
    scores_all = forget_scores + retain_scores

    if len(set(labels)) < 2:
        return 0.5

    return float(roc_auc_score(labels, scores_all))


# ── Extraction Strength ──────────────────────────────────────────────────────


@torch.no_grad()
def compute_extraction_strength(
    model,
    tokenizer,
    forget_records: list[dict],
    device: torch.device,
    max_length: int = 256,
    max_samples: int | None = None,
) -> float:
    """Greedy decoding extraction success rate.
    Lower = better (model can't reproduce forget sequences).
    """
    successes = 0
    total = 0
    model.eval()

    eval_recs = forget_records
    if max_samples and len(eval_recs) > max_samples:
        eval_recs = random.Random(42).sample(forget_records, max_samples)

    for item in eval_recs:
        question, answer = _extract_qa(item)
        if not question or not answer or len(answer) < 10:
            continue

        mid = len(answer) // 2
        first_half = answer[:mid]
        second_half = answer[mid:]

        prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": first_half},
            ],
            tokenize=False,
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=len(second_half) + 10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        if second_half in generated or generated in second_half:
            successes += 1
        total += 1

    return successes / max(1, total)


# ── Forget Quality (KS-test) ─────────────────────────────────────────────────


def compute_forget_quality(
    truth_ratios_model: list[float],
    truth_ratios_retrain: list[float] | None,
) -> float | None:
    """KS-test p-value on truth-ratio distributions vs retrain oracle.
    High p-value (>0.05) = unlearned model indistinguishable from retrain oracle.
    Returns None if retrain oracle ratios are not available.
    """
    if truth_ratios_retrain is None or len(truth_ratios_retrain) == 0:
        return None
    _stat, pvalue = ks_2samp(truth_ratios_model, truth_ratios_retrain)
    return float(pvalue)


# ── full model evaluation ─────────────────────────────────────────────────────


def evaluate_model(
    model,
    tokenizer,
    forget: list[dict],
    retain: list[dict],
    cfg: dict,
    max_rouge_samples: int = 100,
    max_ppl_samples: int = 1000,
    metrics: set[str] | None = None,
    retrain_truth_ratios: list[float] | None = None,
) -> dict:
    """Compute metrics for a single model. Returns dict with computed metrics."""
    if metrics is None:
        metrics = {"ppl", "rouge"}

    device = next(model.parameters()).device
    max_length = cfg["model"]["max_seq_length"]
    max_new_tokens = cfg["eval"]["max_new_tokens"]

    results: dict = {}

    # ── PPL ──────────────────────────────────────────────────────────────
    if "ppl" in metrics:
        print(f"  Forget PPL ({len(forget)} lines, sample {max_ppl_samples})...")
        t0 = time.time()
        forget_ppl = compute_ppl(model, tokenizer, forget, device, max_length=max_length, max_samples=max_ppl_samples)
        print(f"    => {forget_ppl:.2f}  ({time.time() - t0:.1f}s)")

        print(f"  Retain PPL ({len(retain)} lines, sample {max_ppl_samples})...")
        t0 = time.time()
        retain_ppl = compute_ppl(model, tokenizer, retain, device, max_length=max_length, max_samples=max_ppl_samples)
        print(f"    => {retain_ppl:.2f}  ({time.time() - t0:.1f}s)")

        results["forget_ppl"] = round(forget_ppl, 2)
        results["retain_ppl"] = round(retain_ppl, 2)

    # ── ROUGE-L ──────────────────────────────────────────────────────────
    if "rouge" in metrics:
        print(f"  Forget ROUGE-L (up to {max_rouge_samples} samples)...")
        t0 = time.time()
        forget_rouge = compute_rouge_l(
            model, tokenizer, forget, device,
            max_new_tokens=max_new_tokens, max_samples=max_rouge_samples,
            max_length=max_length,
        )
        print(f"    => {forget_rouge:.4f}  ({time.time() - t0:.1f}s)")

        print(f"  Retain ROUGE-L (up to {max_rouge_samples} samples)...")
        t0 = time.time()
        retain_rouge = compute_rouge_l(
            model, tokenizer, retain, device,
            max_new_tokens=max_new_tokens, max_samples=max_rouge_samples,
            max_length=max_length,
        )
        print(f"    => {retain_rouge:.4f}  ({time.time() - t0:.1f}s)")

        results["forget_rouge_l"] = round(forget_rouge, 4)
        results["retain_rouge_l"] = round(retain_rouge, 4)

    # ── Truth Ratio ──────────────────────────────────────────────────────
    truth_ratios_raw: list[float] = []
    if "truth_ratio" in metrics or "forget_quality" in metrics:
        print(f"  Truth Ratio (up to {max_rouge_samples} samples)...")
        t0 = time.time()
        truth_ratios_raw = compute_truth_ratio(
            model, tokenizer, forget, device,
            max_length=max_length, max_samples=max_rouge_samples,
        )
        truth_ratio_mean = float(np.mean(truth_ratios_raw)) if truth_ratios_raw else 0.5
        print(f"    => {truth_ratio_mean:.4f}  ({time.time() - t0:.1f}s)")
        if "truth_ratio" in metrics:
            results["truth_ratio"] = round(truth_ratio_mean, 4)

    # ── MIA AUC ──────────────────────────────────────────────────────────
    if "mia_auc" in metrics:
        print("  MIA AUC (Min-K% Prob, k=20%)...")
        t0 = time.time()
        mia_auc = compute_mia_auc(
            model, tokenizer, forget, retain, device,
            max_length=max_length, k_percent=20, max_samples=max_rouge_samples,
        )
        print(f"    => {mia_auc:.4f}  ({time.time() - t0:.1f}s)")
        results["mia_auc_20"] = round(mia_auc, 4)

    # ── Extraction Strength ──────────────────────────────────────────────
    if "extraction" in metrics:
        print(f"  Extraction Strength (up to {max_rouge_samples} samples)...")
        t0 = time.time()
        extraction = compute_extraction_strength(
            model, tokenizer, forget, device,
            max_length=max_length, max_samples=max_rouge_samples,
        )
        print(f"    => {extraction:.4f}  ({time.time() - t0:.1f}s)")
        results["extraction_strength"] = round(extraction, 4)

    # ── Forget Quality (KS-test) ─────────────────────────────────────────
    if "forget_quality" in metrics:
        if retrain_truth_ratios is not None:
            print("  Forget Quality (KS-test vs retrain oracle)...")
            fq = compute_forget_quality(truth_ratios_raw, retrain_truth_ratios)
            print(f"    => KS p-value = {fq:.4f}" if fq else "    => N/A")
            results["forget_quality_ks"] = round(fq, 4) if fq is not None else None
        else:
            print("  Forget Quality: skipped (no retrain oracle ratios)")
            results["forget_quality_ks"] = None

    # Store raw truth ratios for downstream KS-test if this is the retrain model
    results["_truth_ratios_raw"] = truth_ratios_raw if truth_ratios_raw else None

    return results


# ── checkpoint discovery ──────────────────────────────────────────────────────


def discover_checkpoints(cfg: dict) -> dict[str, str]:
    """Walk benchmarks/results/pipeline_qa/checkpoints/ and return {key: adapter_path}.

    Key format: 'sft' or '{algo}_{author}_{forget_type}'.
    """
    checkpoints: dict[str, str] = {}

    sft_dir = Path(cfg["sft"]["output_dir"])
    if sft_dir.is_dir() and (sft_dir / "config.json").exists():
        checkpoints["sft"] = str(sft_dir)

    algo_dirs = {
        "npo": cfg["npo"]["output_dir"],
        "rmu": cfg["rmu"]["output_dir"],
        "retrain": cfg.get("retrain", {}).get("output_dir", cfg["sft"]["output_dir"].replace("sft", "retrain")),
    }

    for algo_name, algo_dir in algo_dirs.items():
        algo_path = Path(algo_dir)
        if not algo_path.is_dir():
            continue
        for child in sorted(algo_path.iterdir()):
            if not child.is_dir():
                continue
            # Each child directory is named '{author}_{forget_type}'
            # Prefer the 'final' subdirectory
            final_dir = child / "final"
            adapter_dir = (
                final_dir
                if final_dir.is_dir() and (final_dir / "config.json").exists()
                else child
            )
            if (adapter_dir / "config.json").exists():
                key = f"{algo_name}_{child.name}"
                checkpoints[key] = str(adapter_dir)

    return checkpoints


# ── results table ─────────────────────────────────────────────────────────────


def _fmt(val, fmt_str=".2f"):
    if val is None:
        return "N/A"
    return f"{val:{fmt_str}}"


def print_results_table(results: dict, forget_sets: dict) -> None:
    """Print a human-readable results table to stdout."""
    col = "{:<24}| {:>8} | {:>8} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7}"
    header = col.format(
        "Model", "Fgt PPL", "Ret PPL", "F-ROUGE", "R-ROUGE",
        "TR", "MIA", "Extr", "FQ-KS", "Size",
    )
    sep = "-" * len(header)

    print()
    print(header)
    print(sep)

    for key, m in results.items():
        if key == "base":
            display = "Base"
            forget_size = "-"
        elif key == "sft":
            display = "SFT (fine-tuned)"
            forget_size = "-"
        else:
            algo, author, ft = parse_checkpoint_key(key)
            if algo and author and ft:
                display = f"{algo.upper()}+{author}+{ft}"
                forget_size = str(
                    forget_sets.get(author, {}).get(ft, {}).get("count", "?")
                )
            else:
                display = key
                forget_size = "?"

        print(
            col.format(
                display,
                _fmt(m.get("forget_ppl"), ".1f"),
                _fmt(m.get("retain_ppl"), ".1f"),
                _fmt(m.get("forget_rouge_l"), ".2f"),
                _fmt(m.get("retain_rouge_l"), ".2f"),
                _fmt(m.get("truth_ratio"), ".3f"),
                _fmt(m.get("mia_auc_20"), ".3f"),
                _fmt(m.get("extraction_strength"), ".3f"),
                _fmt(m.get("forget_quality_ks"), ".3f"),
                forget_size,
            )
        )

    print()


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate unlearned models against forget/retain sets"
    )
    parser.add_argument(
        "--config",
        default="benchmarks/scripts/pipeline_qa/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument("--author", help="Author name (e.g. Berthe)")
    parser.add_argument(
        "--forget-type",
        choices=["line", "page_prototype", "random", "emb_sim", "2x_random", "embedding"],
        help="Forget-set type",
    )
    parser.add_argument(
        "--algorithm",
        help="Algorithm name (npo, rmu)",
    )
    parser.add_argument(
        "--checkpoint-path", help="Path to LoRA adapter or full model checkpoint directory"
    )
    parser.add_argument(
        "--full-model",
        action="store_true",
        default=False,
        help="Treat checkpoint-path as a full model (not a LoRA adapter)",
    )
    parser.add_argument(
        "--eval-all",
        action="store_true",
        help="Evaluate ALL checkpoints found in benchmarks/results/pipeline_qa/checkpoints/",
    )
    parser.add_argument(
        "--forget-sets",
        default=None,
        help="Path to forget_sets.json (default: benchmarks/results/pipeline_qa/forget_sets.json)",
    )
    parser.add_argument(
        "--max-rouge-samples",
        type=int,
        default=100,
        help="Max samples for ROUGE-L generation eval (default: 100)",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metrics to compute, or 'all' for all 8 metrics. "
             "Default: basic (PPL + ROUGE-L only). "
             "Available: ppl,rouge,truth_ratio,mia_auc,extraction,forget_quality,all",
    )
    args = parser.parse_args()

    # ── Parse metrics flag ────────────────────────────────────────────────
    ALL_METRICS = {"ppl", "rouge", "truth_ratio", "mia_auc", "extraction", "forget_quality"}
    if args.metrics:
        raw = [m.strip() for m in args.metrics.split(",")]
        if "all" in raw:
            metrics_set = ALL_METRICS
        else:
            metrics_set = set()
            for m in raw:
                if m in ALL_METRICS:
                    metrics_set.add(m)
                else:
                    print(f"WARNING: unknown metric '{m}', skipping")
            if not metrics_set:
                print("WARNING: no valid metrics specified, falling back to default (ppl,rouge)")
                metrics_set = {"ppl", "rouge"}
    else:
        metrics_set = {"ppl", "rouge"}
    print(f"Metrics to compute: {', '.join(sorted(metrics_set))}")

    if not args.eval_all and not (args.author and args.forget_type):
        parser.error(
            "Provide --eval-all, or at least --author and --forget-type"
        )

    # ── Load config & resolve paths ──────────────────────────────────────
    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent.parent.parent
    cfg = load_config(str(config_path))

    data_file = str((project_root / cfg["data"]["data_file"]).resolve())
    results_dir = (project_root / cfg["eval"]["output_dir"]).resolve()

    if args.forget_sets:
        forget_sets_path = args.forget_sets
    else:
        forget_sets_path = str(
            (project_root / cfg["data"]["data_file"]).resolve().parent.parent / "forget_sets.json"
        )

    # ── Load data ────────────────────────────────────────────────────────
    print(f"Loading data from {data_file} ...")
    records = load_data(data_file)
    print(f"  {len(records)} records")

    print(f"Loading forget sets from {forget_sets_path} ...")
    forget_sets = load_forget_sets(forget_sets_path)

    results: dict[str, dict] = {}

    # ── Eval-all mode ────────────────────────────────────────────────────
    if args.eval_all:
        # 1) Base model — use first author / line as representative split
        ref_author = cfg["authors"][0]["name"]
        ref_forget, ref_retain = split_forget_retain(
            records, forget_sets, ref_author, "line"
        )

        print("\n" + "=" * 60)
        print("Evaluating: BASE model (no fine-tuning)")
        print("=" * 60)
        base_model, base_tok = load_model(cfg)
        results["base"] = evaluate_model(
            base_model, base_tok, ref_forget, ref_retain, cfg, args.max_rouge_samples,
            metrics=metrics_set,
        )
        del base_model
        torch.cuda.empty_cache()

        # 2) Discover & evaluate all checkpoints
        checkpoints = discover_checkpoints(cfg)
        print(f"\nDiscovered {len(checkpoints)} checkpoint(s):")
        for k in checkpoints:
            print(f"  {k}")

        retrain_truth_ratios: list[float] | None = None

        for key, ckpt_path in checkpoints.items():
            print(f"\n{'=' * 60}")
            print(f"Evaluating: {key}")
            print(f"  Path: {ckpt_path}")
            print("=" * 60)

            model, tokenizer = load_model(cfg, ckpt_path)

            if key == "sft":
                forget, retain = ref_forget, ref_retain
            else:
                algo, author, ft = parse_checkpoint_key(key)
                if (
                    author
                    and ft
                    and author in forget_sets
                    and ft in forget_sets[author]
                ):
                    forget, retain = split_forget_retain(
                        records, forget_sets, author, ft
                    )
                else:
                    print(
                        f"  WARNING: cannot resolve forget set for '{key}', using reference"
                    )
                    forget, retain = ref_forget, ref_retain

            res = evaluate_model(
                model, tokenizer, forget, retain, cfg, args.max_rouge_samples,
                metrics=metrics_set, retrain_truth_ratios=retrain_truth_ratios,
            )

            # First checkpoint with truth_ratio provides the oracle for forget_quality
            if retrain_truth_ratios is None and res.get("_truth_ratios_raw"):
                retrain_truth_ratios = res["_truth_ratios_raw"]
            res.pop("_truth_ratios_raw", None)

            results[key] = res
            del model
            torch.cuda.empty_cache()

    # ── Single model mode ────────────────────────────────────────────────
    else:
        key = "base" if not args.checkpoint_path else (
            f"{args.algorithm}_{args.author}_{args.forget_type}"
            if args.algorithm
            else f"custom_{args.author}_{args.forget_type}"
        )

        print(f"\n{'=' * 60}")
        if args.checkpoint_path:
            print(f"Evaluating: {key}")
            print(f"  {'Full model' if args.full_model else 'Adapter'}: {args.checkpoint_path}")
        else:
            print(f"Evaluating: BASE model (no fine-tuning)")
        print("=" * 60)

        model, tokenizer = load_model(
            cfg, args.checkpoint_path, full_model=args.full_model
        )
        forget, retain = split_forget_retain(
            records, forget_sets, args.author, args.forget_type
        )
        res = evaluate_model(
            model, tokenizer, forget, retain, cfg, args.max_rouge_samples,
            metrics=metrics_set,
        )
        res.pop("_truth_ratios_raw", None)
        results[key] = res

        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / "eval_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        del model
        torch.cuda.empty_cache()

    print(f"\nResults saved to {results_file}")

    print_results_table(results, forget_sets)


if __name__ == "__main__":
    main()
