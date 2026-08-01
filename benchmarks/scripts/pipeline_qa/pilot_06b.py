#!/usr/bin/env python3
"""Pilot run: Qwen3-0.6B Full FT SFT + NPO on existing QA data.

Validates the training pipeline end-to-end before scaling to 1.7B on RTX PRO 6000.
Uses a small subset (5000 samples) for speed.

Usage:
    cd /home/hxue/Projects/originblame
    python benchmarks/scripts/pipeline_qa/pilot_06b.py
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ── Config ──────────────────────────────────────────────────────────────
PILOT_DIR = Path("benchmarks/results/pipeline_qa/pilot_06b")
DATA_FILE = Path("benchmarks/results/pipeline_qa/qa_chatml/data.jsonl")
FORGET_FILE = Path("benchmarks/results/pipeline_qa/qa_chatml/forget_sets.json")
MODEL_PATH = "benchmarks/models/Qwen3-0.6B"
MAX_SEQ = 256
SEED = 42
SUBSET = 5000  # use 5k samples for pilot speed


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(path: Path, n: int, seed: int):
    """Load ChatML QA data, sample n items."""
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    random.seed(seed)
    random.shuffle(items)
    return items[:n]


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
    print("PILOT: Qwen3-0.6B Full FT SFT + NPO validation")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Step 1: Load data ────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    all_data = load_data(DATA_FILE, SUBSET, SEED)
    print(f"  Loaded {len(all_data)} samples from {DATA_FILE}")

    # Load forget sets
    with open(FORGET_FILE) as f:
        forget_sets = json.load(f)
    print(f"  Forget set authors: {list(forget_sets.keys())}")

    # Split: use Berthe line forget set
    author = "Berthe"
    line_forget_indices = set()
    random.seed(SEED)
    # Sample random forget set same size as line
    line_size = 500  # pilot: small forget set
    random_forget_indices = set(random.sample(range(len(all_data)), min(line_size, len(all_data) // 5)))

    # Train/eval split
    split = int(len(all_data) * 0.95)
    train_data = all_data[:split]
    eval_data = all_data[split:]
    print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")

    # ── Step 2: Load model + tokenizer ──────────────────────────────────
    print("\n[2/6] Loading Qwen3-0.6B (Full FT, no quantization)...")
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

    # ── Step 3: SFT ─────────────────────────────────────────────────────
    print("\n[3/6] SFT training (Full FT, 1 epoch)...")
    sft_dir = PILOT_DIR / "sft"

    train_ds = Dataset.from_dict({"messages": [d["messages"] for d in train_data]})
    eval_ds = Dataset.from_dict({"messages": [d["messages"] for d in eval_data]})

    train_tok = train_ds.map(lambda x: tokenize(x, tokenizer, MAX_SEQ), batched=True, remove_columns=["messages"])
    eval_tok = eval_ds.map(lambda x: tokenize(x, tokenizer, MAX_SEQ), batched=True, remove_columns=["messages"])

    sft_args = TrainingArguments(
        output_dir=str(sft_dir),
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

    t0 = time.time()
    trainer.train()
    sft_time = time.time() - t0
    print(f"  SFT done in {sft_time:.0f}s ({sft_time/60:.1f}min)")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Save SFT checkpoint
    trainer.save_model(str(sft_dir / "final"))
    tokenizer.save_pretrained(str(sft_dir / "final"))
    print(f"  Saved to {sft_dir / 'final'}")

    # ── Step 4: Quick eval ──────────────────────────────────────────────
    print("\n[4/6] Quick eval on SFT model...")
    model.eval()
    with torch.no_grad():
        # Sample 10 items, compute avg NLL
        sample = random.sample(eval_data, min(50, len(eval_data)))
        total_nll = 0
        n_tokens = 0
        for item in sample:
            text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ).to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_nll += outputs.loss.item() * inputs["input_ids"].shape[1]
            n_tokens += inputs["input_ids"].shape[1]
        avg_ppl = np.exp(total_nll / n_tokens)
    print(f"  SFT PPL (eval subset): {avg_ppl:.2f}")

    # ── Step 5: Quick NPO run ───────────────────────────────────────────
    print("\n[5/6] NPO unlearning (Berthe, line forget set, 1 epoch)...")

    # Build forget set from pilot data
    forget_data = random.sample(train_data, min(line_size, len(train_data) // 5))
    retain_data = [d for i, d in enumerate(train_data) if i not in set(range(min(line_size, len(train_data) // 5)))]

    forget_ds = Dataset.from_dict({"messages": [d["messages"] for d in forget_data]})
    retain_ds = Dataset.from_dict({"messages": [d["messages"] for d in retain_data[:1000]]})

    forget_tok = forget_ds.map(lambda x: tokenize(x, tokenizer, MAX_SEQ), batched=True, remove_columns=["messages"])
    retain_tok = retain_ds.map(lambda x: tokenize(x, tokenizer, MAX_SEQ), batched=True, remove_columns=["messages"])

    # Simple NPO: just run gradient ascent on forget set as a quick check
    # (Full NPO needs DPOTrainer from trl — use gradient ascent for pilot)
    npo_dir = PILOT_DIR / "npo_pilot"

    # Reload SFT model for NPO
    model_npo = AutoModelForCausalLM.from_pretrained(
        str(sft_dir / "final"),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Untie weights to avoid safetensors shared memory error on save
    model_npo.lm_head.weight = torch.nn.Parameter(model_npo.lm_head.weight.clone())

    npo_args = TrainingArguments(
        output_dir=str(npo_dir),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=20,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        seed=SEED,
        dataloader_num_workers=0,
    )

    # Gradient ascent: negate loss on forget set
    class GAWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.model = base_model

        def forward(self, input_ids, attention_mask=None, labels=None):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return type(outputs)(
                loss=-outputs.loss,  # NEGATE loss = gradient ascent
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    ga_model = GAWrapper(model_npo)

    ga_trainer = Trainer(
        model=ga_model,
        args=npo_args,
        train_dataset=forget_tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    t0 = time.time()
    ga_trainer.train()
    ga_time = time.time() - t0
    print(f"  GA done in {ga_time:.0f}s ({ga_time/60:.1f}min)")

    ga_final_dir = npo_dir / "final"
    model_npo.save_pretrained(str(ga_final_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(ga_final_dir))
    print(f"  Saved GA checkpoint to {ga_final_dir}")

    # ── Step 6: Compare PPL ─────────────────────────────────────────────
    print("\n[6/6] Comparing SFT vs GA forget PPL...")
    ga_model.eval()
    model.eval()

    forget_sample = random.sample(forget_data, min(50, len(forget_data)))
    retain_sample = random.sample(retain_data[:1000], min(50, len(retain_data[:1000])))

    def compute_ppl(m, data, tag):
        total_nll = 0
        n_tok = 0
        with torch.no_grad():
            for item in data:
                text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ).to(m.device)
                outputs = m(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                total_nll += loss * inputs["input_ids"].shape[1]
                n_tok += inputs["input_ids"].shape[1]
        return np.exp(total_nll / n_tok)

    sft_forget_ppl = compute_ppl(model, forget_sample, "SFT forget")
    sft_retain_ppl = compute_ppl(model, retain_sample, "SFT retain")
    ga_forget_ppl = compute_ppl(model_npo, forget_sample, "GA forget")
    ga_retain_ppl = compute_ppl(model_npo, retain_sample, "GA retain")

    print("\n" + "=" * 60)
    print("PILOT RESULTS (Qwen3-0.6B Full FT)")
    print("=" * 60)
    print(f"{'Metric':<25} {'SFT':>10} {'GA (unlearn)':>15}")
    print("-" * 52)
    print(f"{'Forget PPL (↑=better)':<25} {sft_forget_ppl:>10.2f} {ga_forget_ppl:>15.2f}")
    print(f"{'Retain PPL (↓=better)':<25} {sft_retain_ppl:>10.2f} {ga_retain_ppl:>15.2f}")
    print(f"{'Forget PPL ratio':<25} {'1.00x':>10} {ga_forget_ppl/sft_forget_ppl:>15.2f}x")
    print(f"{'Retain PPL ratio':<25} {'1.00x':>10} {ga_retain_ppl/sft_retain_ppl:>15.2f}x")
    print(f"\nSFT time: {sft_time:.0f}s | GA time: {ga_time:.0f}s")
    print(f"Max VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # Save results
    results = {
        "model": "Qwen3-0.6B",
        "method": "full_ft",
        "subset_size": SUBSET,
        "forget_set_size": len(forget_data),
        "sft_forget_ppl": sft_forget_ppl,
        "sft_retain_ppl": sft_retain_ppl,
        "ga_forget_ppl": ga_forget_ppl,
        "ga_retain_ppl": ga_retain_ppl,
        "forget_ppl_ratio": ga_forget_ppl / sft_forget_ppl,
        "retain_ppl_ratio": ga_retain_ppl / sft_retain_ppl,
        "sft_time_s": sft_time,
        "ga_time_s": ga_time,
        "max_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    with open(PILOT_DIR / "pilot_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {PILOT_DIR / 'pilot_results.json'}")

    # ── Verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if ga_forget_ppl > sft_forget_ppl * 1.2:
        print("✅ VERDICT: Pipeline WORKS — GA increases forget PPL as expected")
        print("   Ready to scale to Qwen3-1.7B Full FT on RTX PRO 6000")
    else:
        print("⚠️  VERDICT: Forget PPL did NOT increase significantly")
        print("   Need to investigate before scaling up")
    print("=" * 60)


if __name__ == "__main__":
    main()
