#!/usr/bin/env python3
"""Pilot Retrain (Oracle Baseline) on Qwen3-0.6B Full FT.

Trains SFT from scratch on data EXCLUDING the forget set, then compares
against the existing NPO and SFT baseline results.

The forget set is IDENTICAL to the NPO pilot — same author (Berthe), same
line-level indices, same seed (42), same forget_size (500).

This is the "retrain oracle" / gold standard for machine unlearning.
If NPO approaches retrain performance, the unlearning is effective.

Usage:
    cd /home/hxue/Projects/originblame
    python benchmarks/scripts/pipeline_qa/pilot_retrain_06b.py
"""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ── Config ──────────────────────────────────────────────────────────────
PILOT_DIR = Path("benchmarks/results/pipeline_qa/pilot_retrain_06b")
DATA_FILE = Path("benchmarks/results/pipeline_qa/qa_chatml/data.jsonl")
FORGET_FILE = Path("benchmarks/results/pipeline_qa/qa_chatml/forget_sets.json")
MODEL_PATH = "benchmarks/models/Qwen3-0.6B"
SFT_CHECKPOINT = Path("benchmarks/results/pipeline_qa/pilot_06b/sft/final")
NPO_RESULTS_FILE = Path("benchmarks/results/pipeline_qa/pilot_npo_06b/pilot_npo_results.json")
MAX_SEQ = 256
SEED = 42
SUBSET = 5000
FORGET_SIZE = 500


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── PPL computation (identical to NPO pilot) ───────────────────────────


def compute_ppl(model, data, tokenizer, max_seq, tag="", max_items=50):
    """Compute perplexity on a data subset (same function as NPO pilot)."""
    total_nll = 0.0
    n_tokens = 0
    model.eval()
    with torch.no_grad():
        for item in data[:max_items]:
            if "messages" in item:
                text = tokenizer.apply_chat_template(
                    item["messages"], tokenize=False, add_generation_prompt=False
                )
            else:
                text = item.get("text", "")
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_seq
            ).to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_nll += outputs.loss.item() * inputs["input_ids"].shape[1]
            n_tokens += inputs["input_ids"].shape[1]
    if n_tokens == 0:
        return float("inf")
    ppl = math.exp(total_nll / n_tokens)
    print(f"  {tag} PPL: {ppl:.2f} ({n_tokens} tokens)")
    return ppl


def tokenize(examples, tokenizer, max_seq):
    """Tokenize ChatML messages."""
    texts = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    tokenized = tokenizer(texts, truncation=True, max_length=max_seq, padding=False)
    tokenized["labels"] = [x[:] for x in tokenized["input_ids"]]
    return tokenized


def main():
    set_seed(SEED)
    PILOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PILOT RETRAIN (ORACLE): Qwen3-0.6B Full FT")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Step 1: Load data IDENTICALLY to NPO pilot ─────────────────────
    print("\n[1/7] Loading data (IDENTICAL to NPO pilot)...")

    # VERBATIM from pilot_npo_06b.py lines 219-249
    all_data_ordered = []
    with open(DATA_FILE) as f:
        for line in f:
            all_data_ordered.append(json.loads(line))
    print(f"  Total data: {len(all_data_ordered)}")

    with open(FORGET_FILE) as f:
        forget_sets = json.load(f)

    berthe_line_indices = set(forget_sets["Berthe"]["line"]["indices"])
    print(f"  Berthe line forget set: {len(berthe_line_indices)} indices")

    forget_records = [
        all_data_ordered[i] for i in sorted(berthe_line_indices) if i < len(all_data_ordered)
    ]
    print(f"  Forget records from Berthe line set: {len(forget_records)}")

    random.seed(SEED)
    forget_subset = random.sample(forget_records, min(FORGET_SIZE, len(forget_records)))

    retain_indices = [i for i in range(len(all_data_ordered)) if i not in berthe_line_indices]
    random.seed(SEED)
    random.shuffle(retain_indices)
    retain_subset = [all_data_ordered[i] for i in retain_indices[:SUBSET]]

    eval_data = retain_subset[:100]
    retain_for_training = retain_subset[100:]

    print(f"  Pilot forget: {len(forget_subset)}")
    print(f"  Pilot retain (train): {len(retain_for_training)}")
    print(f"  Pilot eval: {len(eval_data)}")
    print(f"  VERIFY: forget set is identical to NPO pilot (same seed=42, same indices)")

    # ── Step 2: Train SFT from scratch on RETAIN SET ONLY ──────────────
    print("\n[2/7] Loading Qwen3-0.6B base model (Full FT, no quantization)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total/1e6:.0f}M (all trainable: {trainable/1e6:.0f}M)")
    print(f"  VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print("\n[3/7] Retrain SFT on RETAIN SET ONLY (Full FT, 1 epoch)...")
    print(f"  Training on {len(retain_for_training)} samples (FORGET SET EXCLUDED)")
    retrain_dir = PILOT_DIR / "retrain_sft"

    train_ds = Dataset.from_dict({"messages": [d["messages"] for d in retain_for_training]})
    eval_ds = Dataset.from_dict({"messages": [d["messages"] for d in eval_data]})

    train_tok = train_ds.map(lambda x: tokenize(x, tokenizer, MAX_SEQ), batched=True, remove_columns=["messages"])
    eval_tok = eval_ds.map(lambda x: tokenize(x, tokenizer, MAX_SEQ), batched=True, remove_columns=["messages"])

    # Same hyperparameters as SFT pilot (pilot_06b.py) — necessary to document provenance
    sft_args = TrainingArguments(
        output_dir=str(retrain_dir),
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=50,
        save_steps=500,
        eval_strategy="no",
        report_to="none",
        seed=SEED,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=sft_args,
        train_dataset=train_tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    t0_retrain = time.time()
    trainer.train()
    retrain_time = time.time() - t0_retrain
    print(f"  Retrain SFT done in {retrain_time:.0f}s ({retrain_time/60:.1f}min)")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    trainer.save_model(str(retrain_dir / "final"))
    tokenizer.save_pretrained(str(retrain_dir / "final"))
    print(f"  Saved to {retrain_dir / 'final'}")

    # ── Step 4: Evaluate Retrain model on forget + retain ──────────────
    print("\n[4/7] Evaluating Retrain model on forget/retain sets...")

    random.seed(SEED)
    forget_eval = random.sample(forget_subset, min(50, len(forget_subset)))
    retain_eval = random.sample(retain_for_training, min(50, len(retain_for_training)))

    retrain_forget_ppl = compute_ppl(model, forget_eval, tokenizer, MAX_SEQ, "Retrain forget")
    retrain_retain_ppl = compute_ppl(model, retain_eval, tokenizer, MAX_SEQ, "Retrain retain")

    # ── Step 5: Load SFT baseline and evaluate ─────────────────────────
    print("\n[5/7] Loading SFT baseline for comparison...")

    del model
    torch.cuda.empty_cache()

    sft_model = AutoModelForCausalLM.from_pretrained(
        str(SFT_CHECKPOINT),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    sft_model.eval()

    random.seed(SEED)
    forget_eval_sft = random.sample(forget_subset, min(50, len(forget_subset)))
    random.seed(SEED)
    retain_eval_sft = random.sample(retain_for_training, min(50, len(retain_for_training)))

    sft_forget_ppl = compute_ppl(sft_model, forget_eval_sft, tokenizer, MAX_SEQ, "SFT forget")
    sft_retain_ppl = compute_ppl(sft_model, retain_eval_sft, tokenizer, MAX_SEQ, "SFT retain")

    del sft_model
    torch.cuda.empty_cache()

    # ── Step 6: Load NPO results ───────────────────────────────────────
    print("\n[6/7] Loading NPO results...")

    with open(NPO_RESULTS_FILE) as f:
        npo_results = json.load(f)

    npo_forget_ppl = npo_results["npo_forget_ppl"]
    npo_retain_ppl = npo_results["npo_retain_ppl"]
    print(f"  NPO forget PPL: {npo_forget_ppl:.2f}")
    print(f"  NPO retain PPL: {npo_retain_ppl:.2f}")

    # ── Step 7: Generate comparison table ──────────────────────────────
    print("\n[7/7] Generating comparison table...")

    retrain_forget_ratio = retrain_forget_ppl / sft_forget_ppl if sft_forget_ppl > 0 else float("inf")
    retrain_retain_ratio = retrain_retain_ppl / sft_retain_ppl if sft_retain_ppl > 0 else float("inf")
    npo_forget_ratio = npo_forget_ppl / sft_forget_ppl if sft_forget_ppl > 0 else float("inf")
    npo_retain_ratio = npo_retain_ppl / sft_retain_ppl if sft_retain_ppl > 0 else float("inf")

    # NPO closeness to retrain oracle: fraction of retrain's forgetting that NPO achieves
    sft_gap_forget = retrain_forget_ppl - sft_forget_ppl
    npo_gap_forget = npo_forget_ppl - sft_forget_ppl
    npo_closeness_forget = npo_gap_forget / sft_gap_forget if sft_gap_forget > 0 else float("inf")

    retrain_retain_damage = retrain_retain_ppl - sft_retain_ppl
    npo_retain_damage = npo_retain_ppl - sft_retain_ppl
    npo_retain_vs_retrain = npo_retain_ppl / retrain_retain_ppl if retrain_retain_ppl > 0 else float("inf")
    print("\n" + "=" * 70)
    print("=== Retrain Baseline vs NPO Comparison ===")
    print("(Forget set: identical 500 samples from Berthe)")
    print("=" * 70)
    print(f"{'Method':<20} {'Forget PPL':>12} {'Retain PPL':>12} {'Forget ×':>10} {'Retain ×':>10}")
    print("-" * 66)
    print(f"{'SFT (all data)':<20} {sft_forget_ppl:>12.2f} {sft_retain_ppl:>12.2f} {'1.00×':>10} {'1.00×':>10}")
    print(f"{'Retrain (oracle)':<20} {retrain_forget_ppl:>12.2f} {retrain_retain_ppl:>12.2f} {retrain_forget_ratio:>10.2f}× {retrain_retain_ratio:>10.2f}×")
    print(f"{'NPO':<20} {npo_forget_ppl:>12.2f} {npo_retain_ppl:>12.2f} {npo_forget_ratio:>10.2f}× {npo_retain_ratio:>10.2f}×")

    print(f"\nNPO vs Retain gap:")
    print(f"  - Forget PPL: NPO achieves {npo_closeness_forget*100:.1f}% of retrain's forgetting")
    print(f"  - Retain PPL: NPO retain is {npo_retain_vs_retrain:.2f}× retrain's retain PPL")
    print(f"  - Retrain training: {retrain_time:.0f}s ({retrain_time/60:.1f}min)")
    print("=" * 70)

    # Save results JSON
    results = {
        "model": "Qwen3-0.6B",
        "forget_set_source": "Berthe",
        "forget_set_size": len(forget_subset),
        "retain_set_size": len(retain_for_training),
        "sft_baseline": {
            "forget_ppl": sft_forget_ppl,
            "retain_ppl": sft_retain_ppl,
        },
        "retrain_oracle": {
            "forget_ppl": retrain_forget_ppl,
            "retain_ppl": retrain_retain_ppl,
            "training_time_s": retrain_time,
        },
        "npo": {
            "forget_ppl": npo_forget_ppl,
            "retain_ppl": npo_retain_ppl,
            "selectivity": npo_results.get("selectivity", npo_forget_ratio / npo_retain_ratio),
        },
        "comparison": {
            "retrain_forget_ppl_vs_sft": retrain_forget_ratio,
            "npo_forget_ppl_vs_sft": npo_forget_ratio,
            "npo_closeness_to_retrain": npo_closeness_forget,
            "npo_retain_vs_retrain_retain": npo_retain_vs_retrain,
        },
        "max_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    results_path = PILOT_DIR / "pilot_retrain_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
