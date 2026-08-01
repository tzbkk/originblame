#!/usr/bin/env python3
"""8-metric evaluation for unified pilot: SFT, Retrain (oracle), NPO.

Metrics:
  1. Forget PPL (token-level NLL averaging)
  2. Retain PPL
  3. Forget ROUGE-L (greedy generation vs reference)
  4. Retain ROUGE-L
  5. Truth Ratio (on forget set, sigmoid of log-p difference)
  6. MIA AUC (Min-K% Prob membership inference, k=20%)
  7. Extraction Strength (greedy decode second half from first half)
  8. Forget Quality (KS-test on truth-ratio distributions vs retrain oracle)

Usage:
    cd /home/hxue/Projects/originblame
    python benchmarks/scripts/pipeline_qa/pilot_eval_8metrics.py
"""

import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rouge_score import rouge_scorer
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

PILOT_DIR = Path("benchmarks/results/pipeline_qa/pilot_unified_06b")
MODEL_BASE = "benchmarks/models/Qwen3-0.6B"
MAX_SEQ = 256
SEED = 42

CHECKPOINTS = {
    "SFT": str(PILOT_DIR / "sft" / "final"),
    "Retrain": str(PILOT_DIR / "retrain_sft" / "final"),
    "NPO": str(PILOT_DIR / "npo_final"),
}

# Sample sizes — generation metrics use fewer items to keep runtime reasonable
PPL_MAX_ITEMS = 100
GEN_MAX_ITEMS = 20  # ROUGE-L, extraction (generation is slow)
TRUTH_MAX_ITEMS = 50
MIA_MAX_ITEMS = 50


# ── Metric helpers ─────────────────────────────────────────────────────────

def compute_ppl(model, tokenizer, data, max_seq, max_items=100):
    """Token-level NLL averaging → perplexity."""
    total_nll, n_tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for item in data[:max_items]:
            text = tokenizer.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=False
            )
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_seq
            ).to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_nll += outputs.loss.item() * inputs["input_ids"].shape[1]
            n_tokens += inputs["input_ids"].shape[1]
    return math.exp(total_nll / n_tokens) if n_tokens > 0 else float("inf")


def compute_rouge_l(model, tokenizer, data, max_seq, max_items=20):
    """Generate text from questions, compare with reference answers using ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    model.eval()
    for item in data[:max_items]:
        messages = item["messages"]
        question = ""
        reference = ""
        for msg in messages:
            if msg["role"] == "user":
                question = msg["content"]
            elif msg["role"] == "assistant":
                reference = msg["content"]
        if not question or not reference:
            continue

        input_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=max_seq
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=128, do_sample=False, temperature=1.0
            )
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        score = scorer.score(reference, generated)["rougeL"].fmeasure
        scores.append(score)
    return np.mean(scores) if scores else 0.0


def compute_truth_ratio(model, tokenizer, forget_data, max_seq, max_items=50):
    """P(correct) / (P(correct) + P(wrong)) via sigmoid(log_p_correct - log_p_wrong).
    Returns list of per-item ratios (needed for Forget Quality KS-test).
    Lower = better forgetting.
    """
    ratios = []
    model.eval()

    all_answers = [
        msg["content"]
        for item in forget_data
        for msg in item["messages"]
        if msg["role"] == "assistant"
    ]
    random.seed(SEED)
    shuffled_answers = all_answers.copy()
    random.shuffle(shuffled_answers)
    shuffle_idx = 0

    with torch.no_grad():
        for item in forget_data[:max_items]:
            question = ""
            correct_answer = ""
            for msg in item["messages"]:
                if msg["role"] == "user":
                    question = msg["content"]
                elif msg["role"] == "assistant":
                    correct_answer = msg["content"]
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
                prompt_correct, return_tensors="pt", truncation=True, max_length=max_seq
            ).to(model.device)
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
                prompt_wrong, return_tensors="pt", truncation=True, max_length=max_seq
            ).to(model.device)
            outputs_wrong = model(**inputs_wrong, labels=inputs_wrong["input_ids"])
            log_p_wrong = -outputs_wrong.loss.item()

            ratio = 1.0 / (1.0 + math.exp(-(log_p_correct - log_p_wrong)))
            ratios.append(ratio)

    return ratios


def compute_mia_auc(model, tokenizer, forget_data, retain_data, max_seq,
                    k_percent=20, max_items=50):
    """Min-K% Prob membership inference attack.
    AUC ≈ 0.5 = ideal (can't distinguish forget from retain).
    AUC >> 0.5 = model still remembers forget set membership.
    """
    def get_min_k_score(data):
        scores = []
        model.eval()
        with torch.no_grad():
            for item in data[:max_items]:
                if "messages" in item:
                    text = tokenizer.apply_chat_template(
                        item["messages"], tokenize=False
                    )
                else:
                    text = item.get("text", "")
                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=max_seq
                ).to(model.device)
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

    forget_scores = get_min_k_score(forget_data)
    retain_scores = get_min_k_score(retain_data)

    labels = [1] * len(forget_scores) + [0] * len(retain_scores)
    scores_all = forget_scores + retain_scores

    if len(set(labels)) < 2:
        return 0.5

    return roc_auc_score(labels, scores_all)


def compute_extraction_strength(model, tokenizer, forget_data, max_seq, max_items=20):
    """Greedy decoding extraction success rate.
    Lower = better (model can't reproduce forget sequences).
    """
    successes = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for item in forget_data[:max_items]:
            question = ""
            answer = ""
            for msg in item["messages"]:
                if msg["role"] == "user":
                    question = msg["content"]
                elif msg["role"] == "assistant":
                    answer = msg["content"]
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
                prompt, return_tensors="pt", truncation=True, max_length=max_seq
            ).to(model.device)
            output = model.generate(
                **inputs, max_new_tokens=len(second_half) + 10, do_sample=False
            )
            generated = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            if second_half in generated or generated in second_half:
                successes += 1
            total += 1

    return successes / max(1, total)


def compute_forget_quality(truth_ratios_model, truth_ratios_retrain):
    """KS-test p-value on truth-ratio distributions.
    High p-value (>0.05) = unlearned model indistinguishable from retrain oracle.
    """
    stat, pvalue = ks_2samp(truth_ratios_model, truth_ratios_retrain)
    return pvalue


# ── Main evaluation loop ───────────────────────────────────────────────────

def load_model_and_tokenizer(ckpt_path):
    """Load model + tokenizer, return (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def free_model(model):
    """Free GPU memory."""
    del model
    torch.cuda.empty_cache()
    gc.collect()


def evaluate_model(name, ckpt_path, forget_eval, retain_eval,
                   retrain_truth_ratios=None):
    """Evaluate one model on all 8 metrics. Returns dict of results."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {name} ({ckpt_path})")
    print(f"{'='*60}")

    model, tokenizer = load_model_and_tokenizer(ckpt_path)
    print(f"  GPU VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    results = {}

    # ── 1-2: PPL ───────────────────────────────────────────────────────
    t0 = time.time()
    print("  [1/8] Forget PPL...")
    results["forget_ppl"] = compute_ppl(
        model, tokenizer, forget_eval, MAX_SEQ, PPL_MAX_ITEMS
    )
    print(f"         Forget PPL = {results['forget_ppl']:.2f}")

    print("  [2/8] Retain PPL...")
    results["retain_ppl"] = compute_ppl(
        model, tokenizer, retain_eval, MAX_SEQ, PPL_MAX_ITEMS
    )
    print(f"         Retain PPL = {results['retain_ppl']:.2f}")
    print(f"    PPL done in {time.time()-t0:.0f}s")

    # ── 3-4: ROUGE-L ───────────────────────────────────────────────────
    t0 = time.time()
    print("  [3/8] Forget ROUGE-L...")
    results["forget_rouge_l"] = compute_rouge_l(
        model, tokenizer, forget_eval, MAX_SEQ, GEN_MAX_ITEMS
    )
    print(f"         Forget ROUGE-L = {results['forget_rouge_l']:.4f}")

    print("  [4/8] Retain ROUGE-L...")
    results["retain_rouge_l"] = compute_rouge_l(
        model, tokenizer, retain_eval, MAX_SEQ, GEN_MAX_ITEMS
    )
    print(f"         Retain ROUGE-L = {results['retain_rouge_l']:.4f}")
    print(f"    ROUGE-L done in {time.time()-t0:.0f}s")

    # ── 5: Truth Ratio (returns per-item list for KS-test) ─────────────
    t0 = time.time()
    print("  [5/8] Truth Ratio...")
    truth_ratios = compute_truth_ratio(
        model, tokenizer, forget_eval, MAX_SEQ, TRUTH_MAX_ITEMS
    )
    results["truth_ratio"] = float(np.mean(truth_ratios)) if truth_ratios else 0.5
    results["_truth_ratios_raw"] = truth_ratios  # keep for KS-test
    print(f"         Truth Ratio = {results['truth_ratio']:.4f}")
    print(f"    Truth Ratio done in {time.time()-t0:.0f}s")

    # ── 6: MIA AUC ─────────────────────────────────────────────────────
    t0 = time.time()
    print("  [6/8] MIA AUC (Min-K% Prob)...")
    results["mia_auc"] = compute_mia_auc(
        model, tokenizer, forget_eval, retain_eval, MAX_SEQ, k_percent=20,
        max_items=MIA_MAX_ITEMS,
    )
    print(f"         MIA AUC = {results['mia_auc']:.4f}")
    print(f"    MIA AUC done in {time.time()-t0:.0f}s")

    # ── 7: Extraction Strength ─────────────────────────────────────────
    t0 = time.time()
    print("  [7/8] Extraction Strength...")
    results["extraction_strength"] = compute_extraction_strength(
        model, tokenizer, forget_eval, MAX_SEQ, GEN_MAX_ITEMS
    )
    print(f"         Extraction Strength = {results['extraction_strength']:.4f}")
    print(f"    Extraction done in {time.time()-t0:.0f}s")

    # ── 8: Forget Quality (KS-test vs retrain oracle) ──────────────────
    if retrain_truth_ratios is not None:
        print("  [8/8] Forget Quality (KS-test vs Retrain oracle)...")
        results["forget_quality_ks_pvalue"] = compute_forget_quality(
            truth_ratios, retrain_truth_ratios
        )
        print(f"         KS p-value = {results['forget_quality_ks_pvalue']:.4f}")
    else:
        # This IS the retrain model — store its truth ratios for others
        results["forget_quality_ks_pvalue"] = None
        print("  [8/8] Forget Quality: N/A (this IS the retrain oracle)")

    free_model(model)
    return results


def print_comparison_table(all_results):
    """Print a formatted comparison table."""
    models = ["SFT", "Retrain", "NPO"]

    print("\n" + "=" * 90)
    print("=== 8-Metric Evaluation (Qwen3-0.6B, unified data splits) ===")
    print("=" * 90)
    print(
        f"{'Metric':<26} {'SFT':>10} {'Retrain':>10} {'NPO':>10} {'Desired':>14}"
    )
    print("-" * 90)

    rows = [
        ("Forget PPL (↑)", "forget_ppl", "↑ better"),
        ("Retain PPL (↓)", "retain_ppl", "↓ better"),
        ("Forget ROUGE-L (↓)", "forget_rouge_l", "↓ better"),
        ("Retain ROUGE-L (↑)", "retain_rouge_l", "↑ better"),
        ("Truth Ratio (↓)", "truth_ratio", "↓ better"),
        ("MIA AUC (→0.5)", "mia_auc", "≈0.5 ideal"),
        ("Extraction Str. (↓)", "extraction_strength", "↓ better"),
    ]

    for label, key, desired in rows:
        vals = []
        for m in models:
            v = all_results[m].get(key, float("nan"))
            if isinstance(v, float):
                vals.append(f"{v:>10.4f}")
            else:
                vals.append(f"{'N/A':>10}")
        print(f"{label:<26} {vals[0]} {vals[1]} {vals[2]} {desired:>14}")

    # Forget Quality row (only SFT vs retrain, NPO vs retrain)
    fq_label = "Forget Quality KS (↑)"
    fq_vals = []
    for m in models:
        v = all_results[m].get("forget_quality_ks_pvalue")
        if v is not None:
            fq_vals.append(f"{v:>10.4f}")
        else:
            fq_vals.append(f"{'—':>10}")
    print(f"{fq_label:<26} {fq_vals[0]} {fq_vals[1]} {fq_vals[2]} {'↑ better':>14}")
    print("=" * 90)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Load splits
    splits_path = PILOT_DIR / "splits.json"
    if not splits_path.exists():
        print(f"ERROR: splits not found at {splits_path}")
        print("Run the unified pilot first: python benchmarks/scripts/pipeline_qa/pilot_unified_06b.py")
        sys.exit(1)

    with open(splits_path) as f:
        splits = json.load(f)

    forget_eval = splits["forget_eval"]
    retain_eval = splits["retain_eval"]
    print(f"Loaded splits: forget_eval={len(forget_eval)}, retain_eval={len(retain_eval)}")

    # Verify checkpoints exist
    for name, path in CHECKPOINTS.items():
        if not Path(path).exists():
            print(f"ERROR: {name} checkpoint not found at {path}")
            sys.exit(1)
    print("All checkpoints verified.")

    all_results = {}

    # Evaluate Retrain FIRST (its truth ratios are needed for KS-test)
    retrain_results = evaluate_model(
        "Retrain", CHECKPOINTS["Retrain"], forget_eval, retain_eval,
        retrain_truth_ratios=None,
    )
    all_results["Retrain"] = retrain_results
    retrain_truth_ratios = retrain_results.pop("_truth_ratios_raw")

    # Evaluate SFT
    sft_results = evaluate_model(
        "SFT", CHECKPOINTS["SFT"], forget_eval, retain_eval,
        retrain_truth_ratios=retrain_truth_ratios,
    )
    all_results["SFT"] = sft_results
    sft_results.pop("_truth_ratios_raw", None)

    # Evaluate NPO
    npo_results = evaluate_model(
        "NPO", CHECKPOINTS["NPO"], forget_eval, retain_eval,
        retrain_truth_ratios=retrain_truth_ratios,
    )
    all_results["NPO"] = npo_results
    npo_results.pop("_truth_ratios_raw", None)

    # Print comparison table
    print_comparison_table(all_results)

    # Save results
    output_path = PILOT_DIR / "full_eval_results.json"
    output_data = {
        "model": "Qwen3-0.6B",
        "seed": SEED,
        "splits_file": str(splits_path),
        "forget_eval_size": len(forget_eval),
        "retain_eval_size": len(retain_eval),
        "sample_sizes": {
            "ppl_max_items": PPL_MAX_ITEMS,
            "gen_max_items": GEN_MAX_ITEMS,
            "truth_max_items": TRUTH_MAX_ITEMS,
            "mia_max_items": MIA_MAX_ITEMS,
        },
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
