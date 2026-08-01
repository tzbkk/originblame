#!/usr/bin/env python3
"""Evaluate the Base model (Qwen3-0.6B, no SFT) on the same 8 metrics and merge
into existing full_eval_results.json.

Usage:
    cd /home/hxue/Projects/originblame
    python benchmarks/scripts/pipeline_qa/pilot_eval_base.py
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
BASE_MODEL = "benchmarks/models/Qwen3-0.6B"
RESULTS_FILE = PILOT_DIR / "full_eval_results.json"
SPLITS_FILE = PILOT_DIR / "splits.json"
MAX_SEQ = 256
SEED = 42

PPL_MAX_ITEMS = 100
GEN_MAX_ITEMS = 20
TRUTH_MAX_ITEMS = 50
MIA_MAX_ITEMS = 50


# ── Metric helpers (copied from pilot_eval_8metrics.py) ─────────────────────

def compute_ppl(model, tokenizer, data, max_seq, max_items=100):
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

            prompt_correct = tokenizer.apply_chat_template(
                [{"role": "user", "content": question},
                 {"role": "assistant", "content": correct_answer}],
                tokenize=False,
            )
            inputs_correct = tokenizer(
                prompt_correct, return_tensors="pt", truncation=True, max_length=max_seq
            ).to(model.device)
            outputs_correct = model(**inputs_correct, labels=inputs_correct["input_ids"])
            log_p_correct = -outputs_correct.loss.item()

            prompt_wrong = tokenizer.apply_chat_template(
                [{"role": "user", "content": question},
                 {"role": "assistant", "content": wrong_answer}],
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
                logits = outputs.logits

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
                [{"role": "user", "content": question},
                 {"role": "assistant", "content": first_half}],
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
    stat, pvalue = ks_2samp(truth_ratios_model, truth_ratios_retrain)
    return pvalue


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Load splits
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    forget_eval = splits["forget_eval"]
    retain_eval = splits["retain_eval"]
    print(f"Loaded splits: forget_eval={len(forget_eval)}, retain_eval={len(retain_eval)}")

    # Load existing results
    with open(RESULTS_FILE) as f:
        full_results = json.load(f)

    # Extract retrain truth ratios — we need them for KS-test
    # Since the original results don't store raw truth ratios, we'll compute
    # them from the Retrain model as well. But that means loading Retrain too.
    # Alternative: just compute Base's KS against Retrain by loading Retrain.
    # We'll load Retrain first (briefly) to get its truth ratios, then Base.

    print("\n" + "=" * 60)
    print("  Loading Retrain model (to get oracle truth ratios)")
    print("=" * 60)
    retrain_path = str(PILOT_DIR / "retrain_sft" / "final")
    tokenizer_retrain = AutoTokenizer.from_pretrained(
        retrain_path, trust_remote_code=True, padding_side="right"
    )
    if tokenizer_retrain.pad_token is None:
        tokenizer_retrain.pad_token = tokenizer_retrain.eos_token
    model_retrain = AutoModelForCausalLM.from_pretrained(
        retrain_path, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    model_retrain.eval()
    print(f"  GPU VRAM (Retrain): {torch.cuda.memory_allocated()/1e9:.2f} GB")

    t0 = time.time()
    print("  Computing Retrain truth ratios for KS-test oracle...")
    retrain_truth_ratios = compute_truth_ratio(
        model_retrain, tokenizer_retrain, forget_eval, MAX_SEQ, TRUTH_MAX_ITEMS
    )
    print(f"  Done in {time.time()-t0:.0f}s")

    # Free retrain
    del model_retrain
    del tokenizer_retrain
    torch.cuda.empty_cache()
    gc.collect()
    print("  Retrain model freed.")

    # Now load Base model
    print("\n" + "=" * 60)
    print(f"  Loading Base model: {BASE_MODEL}")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.eval()
    print(f"  GPU VRAM (Base): {torch.cuda.memory_allocated()/1e9:.2f} GB")

    results = {}

    # 1-2: PPL
    t0 = time.time()
    print("  [1/8] Forget PPL...")
    results["forget_ppl"] = compute_ppl(model, tokenizer, forget_eval, MAX_SEQ, PPL_MAX_ITEMS)
    print(f"         Forget PPL = {results['forget_ppl']:.2f}")

    print("  [2/8] Retain PPL...")
    results["retain_ppl"] = compute_ppl(model, tokenizer, retain_eval, MAX_SEQ, PPL_MAX_ITEMS)
    print(f"         Retain PPL = {results['retain_ppl']:.2f}")
    print(f"    PPL done in {time.time()-t0:.0f}s")

    # 3-4: ROUGE-L
    t0 = time.time()
    print("  [3/8] Forget ROUGE-L...")
    results["forget_rouge_l"] = compute_rouge_l(model, tokenizer, forget_eval, MAX_SEQ, GEN_MAX_ITEMS)
    print(f"         Forget ROUGE-L = {results['forget_rouge_l']:.4f}")

    print("  [4/8] Retain ROUGE-L...")
    results["retain_rouge_l"] = compute_rouge_l(model, tokenizer, retain_eval, MAX_SEQ, GEN_MAX_ITEMS)
    print(f"         Retain ROUGE-L = {results['retain_rouge_l']:.4f}")
    print(f"    ROUGE-L done in {time.time()-t0:.0f}s")

    # 5: Truth Ratio
    t0 = time.time()
    print("  [5/8] Truth Ratio...")
    truth_ratios = compute_truth_ratio(model, tokenizer, forget_eval, MAX_SEQ, TRUTH_MAX_ITEMS)
    results["truth_ratio"] = float(np.mean(truth_ratios)) if truth_ratios else 0.5
    print(f"         Truth Ratio = {results['truth_ratio']:.4f}")
    print(f"    Truth Ratio done in {time.time()-t0:.0f}s")

    # 6: MIA AUC
    t0 = time.time()
    print("  [6/8] MIA AUC (Min-K% Prob)...")
    results["mia_auc"] = compute_mia_auc(
        model, tokenizer, forget_eval, retain_eval, MAX_SEQ, k_percent=20,
        max_items=MIA_MAX_ITEMS,
    )
    print(f"         MIA AUC = {results['mia_auc']:.4f}")
    print(f"    MIA AUC done in {time.time()-t0:.0f}s")

    # 7: Extraction Strength
    t0 = time.time()
    print("  [7/8] Extraction Strength...")
    results["extraction_strength"] = compute_extraction_strength(
        model, tokenizer, forget_eval, MAX_SEQ, GEN_MAX_ITEMS
    )
    print(f"         Extraction Strength = {results['extraction_strength']:.4f}")
    print(f"    Extraction done in {time.time()-t0:.0f}s")

    # 8: Forget Quality KS-test vs Retrain oracle
    print("  [8/8] Forget Quality (KS-test vs Retrain oracle)...")
    results["forget_quality_ks_pvalue"] = compute_forget_quality(
        truth_ratios, retrain_truth_ratios
    )
    print(f"         KS p-value = {results['forget_quality_ks_pvalue']:.4f}")

    # Free base model
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # ── Merge into full results ───────────────────────────────────────────
    # Insert Base results (put it first in order)
    existing = full_results["results"]
    new_results = {"Base": results}
    for k in ["SFT", "Retrain", "NPO"]:
        if k in existing:
            new_results[k] = existing[k]
    full_results["results"] = new_results

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_FILE}")

    # ── Print 4-model comparison table ────────────────────────────────────
    models = ["Base", "SFT", "Retrain", "NPO"]
    all_res = full_results["results"]

    print("\n" + "=" * 100)
    print("=== 8-Metric Evaluation: Base → SFT → Retrain → NPO ===")
    print("(Qwen3-0.6B, unified data splits)")
    print("=" * 100)
    print(
        f"{'Metric':<26} {'Base*':>10} {'SFT':>10} {'Retrain':>10} {'NPO':>10} {'Desired':>14}"
    )
    print("-" * 100)

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
            v = all_res[m].get(key, float("nan"))
            if isinstance(v, float):
                vals.append(f"{v:>10.4f}")
            else:
                vals.append(f"{'N/A':>10}")
        print(f"{label:<26} {vals[0]} {vals[1]} {vals[2]} {vals[3]} {desired:>14}")

    # Forget Quality row
    fq_label = "Forget Quality KS (↑)"
    fq_vals = []
    for m in models:
        v = all_res[m].get("forget_quality_ks_pvalue")
        if v is not None:
            fq_vals.append(f"{v:>10.4f}")
        else:
            fq_vals.append(f"{'—':>10}")
    print(f"{fq_label:<26} {fq_vals[0]} {fq_vals[1]} {fq_vals[2]} {fq_vals[3]} {'↑ better':>14}")
    print("=" * 100)
    print("*Base = pretrained Qwen3-0.6B, no SFT (represents 'complete forgetting')")


if __name__ == "__main__":
    main()
