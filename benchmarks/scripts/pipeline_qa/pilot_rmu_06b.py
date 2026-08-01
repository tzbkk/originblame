#!/usr/bin/env python3
"""Pilot RMU (Representation Misdirection for Unlearning) on Qwen3-0.6B Full FT.

Loads the already-trained SFT checkpoint, runs RMU unlearning on Berthe's
forget set (line-level), then compares forget/retain PPL before and after.

Reference: benchmarks/scripts/pipeline_qa/train_rmu.py (760 lines)

RMU algorithm:
    L_forget = MSE(h_θ(x_forget, layer_L), c * u)
    L_retain = MSE(h_θ(x_retain, layer_L), h_ref(x_retain, layer_L))
    L_total  = L_forget + α * L_retain

Usage:
    cd /home/hxue/Projects/originblame
    python benchmarks/scripts/pipeline_qa/pilot_rmu_06b.py
"""

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────
PILOT_DIR = Path("benchmarks/results/pipeline_qa/pilot_rmu_06b_v2")
DATA_FILE = Path("benchmarks/results/pipeline_qa/qa_chatml/data.jsonl")
FORGET_FILE = Path("benchmarks/results/pipeline_qa/qa_chatml/forget_sets.json")
MODEL_PATH = "benchmarks/models/Qwen3-0.6B"
SFT_CHECKPOINT = Path("benchmarks/results/pipeline_qa/pilot_06b/sft/final")
MAX_SEQ = 256
SEED = 42
SUBSET = 5000
FORGET_SIZE = 500

# RMU hyperparameters (from config.yaml)
RMU_LR = 5e-5
RMU_ALPHA = 50.0                        # Zephyr 7B uses 50, WMDP original uses 1600 (but that's for 8x7B). Start moderate.
RMU_STEERING_COEFF = 300.0              # WMDP original uses 300 for Mixtral
RMU_TARGET_LAYER = 14                   # middle of 28 layers, keep same
RMU_BATCH_SIZE = 4                      # keep same
RMU_MAX_STEPS = 400                     # WMDP original uses 400 batches
RMU_MAX_GRAD_NORM = 1.0
RMU_WEIGHT_DECAY = 0.01


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data ────────────────────────────────────────────────────────────────


class TextDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int):
        rendered = []
        for rec in records:
            if "messages" in rec:
                text = tokenizer.apply_chat_template(
                    rec["messages"], tokenize=False, add_generation_prompt=False
                )
            else:
                text = rec.get("text", "")
            rendered.append(text)
        self.texts = rendered
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ── Hidden state extraction ────────────────────────────────────────────


def get_decoder_layers(model):
    if hasattr(model, "base_model"):
        inner = model.base_model
        if hasattr(inner, "model"):
            inner = inner.model
        if hasattr(inner, "layers"):
            return inner.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(f"Cannot find decoder layers in {type(model).__name__}")


def extract_hidden_state(
    model, input_ids, attention_mask, target_layer, no_grad=False
):
    """Forward pass, capture hidden state at target_layer via hook."""
    layers = get_decoder_layers(model)
    captured = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured["h"] = output[0]
        else:
            captured["h"] = output

    handle = layers[target_layer].register_forward_hook(hook_fn)

    if no_grad:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        result = captured["h"].detach()
    else:
        model(input_ids=input_ids, attention_mask=attention_mask)
        result = captured["h"]

    handle.remove()
    return result


# ── PPL computation ────────────────────────────────────────────────────


def compute_ppl(model, data, tokenizer, max_seq, tag="", max_items=50):
    """Compute perplexity on a data subset."""
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


# ── Main ────────────────────────────────────────────────────────────────


def main():
    set_seed(SEED)
    PILOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PILOT RMU: Qwen3-0.6B Full FT")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Step 1: Load data ────────────────────────────────────────────────
    print("\n[1/6] Loading data...")

    # Load all data
    all_data = []
    with open(DATA_FILE) as f:
        for line in f:
            all_data.append(json.loads(line))
    print(f"  Total data: {len(all_data)}")

    # Load forget sets
    with open(FORGET_FILE) as f:
        forget_sets = json.load(f)

    # Get Berthe line forget indices
    berthe_line_indices = set(forget_sets["Berthe"]["line"]["indices"])
    print(f"  Berthe line forget set: {len(berthe_line_indices)} indices")

    # Build forget and retain records from the full dataset
    # Only use samples where forget indices fall within our subset
    # Take a subset of 5000 for pilot speed
    random.seed(SEED)
    random.shuffle(all_data)

    # We need to find forget samples in the original indices
    # Reload without shuffle for correct forget set mapping
    all_data_ordered = []
    with open(DATA_FILE) as f:
        for line in f:
            all_data_ordered.append(json.loads(line))

    # Get forget records (Berthe line indices within data range)
    forget_records = [
        all_data_ordered[i] for i in sorted(berthe_line_indices) if i < len(all_data_ordered)
    ]
    print(f"  Forget records from Berthe line set: {len(forget_records)}")

    # Sample pilot forget subset
    random.seed(SEED)
    forget_subset = random.sample(forget_records, min(FORGET_SIZE, len(forget_records)))

    # Retain = everything that's NOT in forget set, take a subset for pilot
    retain_indices = [i for i in range(len(all_data_ordered)) if i not in berthe_line_indices]
    random.seed(SEED)
    random.shuffle(retain_indices)
    retain_subset = [all_data_ordered[i] for i in retain_indices[:SUBSET]]

    # Eval split from retain
    eval_data = retain_subset[:100]
    retain_for_training = retain_subset[100:]

    print(f"  Pilot forget: {len(forget_subset)}")
    print(f"  Pilot retain (train): {len(retain_for_training)}")
    print(f"  Pilot eval: {len(eval_data)}")

    # ── Step 2: Load models ──────────────────────────────────────────────
    print("\n[2/6] Loading models (SFT trainable + base frozen)...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(SFT_CHECKPOINT), trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Trainable model: SFT checkpoint (Full FT, bf16)
    trainable_model = AutoModelForCausalLM.from_pretrained(
        str(SFT_CHECKPOINT),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    trainable_model.config.use_cache = False
    trainable_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    trainable_params = sum(p.numel() for p in trainable_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in trainable_model.parameters())
    print(f"  Trainable model: {total_params/1e6:.0f}M params (all trainable)")
    print(f"  VRAM after trainable: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Frozen reference model: base Qwen3-0.6B (NOT the SFT checkpoint)
    frozen_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    frozen_model.eval()
    for param in frozen_model.parameters():
        param.requires_grad = False

    print(f"  Frozen reference: base Qwen3-0.6B")
    print(f"  VRAM after frozen: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ── Step 3: Untie weights ────────────────────────────────────────────
    print("\n[3/6] Untying lm_head weights...")
    trainable_model.lm_head.weight = torch.nn.Parameter(
        trainable_model.lm_head.weight.clone()
    )
    print("  Done — lm_head.weight cloned from embed_tokens")

    # ── Step 4: RMU training ─────────────────────────────────────────────
    print("\n[4/6] RMU training...")

    # Build datasets
    forget_dataset = TextDataset(forget_subset, tokenizer, MAX_SEQ)
    retain_dataset = TextDataset(retain_for_training, tokenizer, MAX_SEQ)

    forget_loader = DataLoader(
        forget_dataset, batch_size=RMU_BATCH_SIZE, shuffle=True, drop_last=True
    )
    retain_loader = DataLoader(
        retain_dataset, batch_size=RMU_BATCH_SIZE, shuffle=True, drop_last=True
    )

    print(f"  Forget batches: {len(forget_loader)}, Retain batches: {len(retain_loader)}")

    # Steering vector u (random unit vector, seeded for reproducibility)
    device = next(trainable_model.parameters()).device
    hidden_dim = trainable_model.config.hidden_size
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    u = torch.randn(hidden_dim, generator=gen, device=device, dtype=torch.bfloat16)
    u = u / u.norm()
    print(f"  Steering vector u: dim={hidden_dim}, norm={u.norm().item():.4f}")
    print(f"  Target layer: {RMU_TARGET_LAYER}")
    print(
        f"  steering_coeff={RMU_STEERING_COEFF}, alpha={RMU_ALPHA}, "
        f"lr={RMU_LR}, max_steps={RMU_MAX_STEPS}"
    )

    # Optimizer
    trainable_params_list = [p for p in trainable_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params_list, lr=RMU_LR, weight_decay=RMU_WEIGHT_DECAY
    )

    # Training loop
    trainable_model.train()
    t0_total = time.time()

    forget_iter = iter(forget_loader)
    retain_iter = iter(retain_loader)

    running_forget = 0.0
    running_retain = 0.0
    running_total = 0.0
    log_interval = 10

    for step in range(RMU_MAX_STEPS):
        t0_step = time.time()

        # Cycle loaders
        try:
            forget_batch = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            forget_batch = next(forget_iter)

        try:
            retain_batch = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_loader)
            retain_batch = next(retain_iter)

        # Move to device
        f_ids = forget_batch["input_ids"].to(device)
        f_mask = forget_batch["attention_mask"].to(device)
        r_ids = retain_batch["input_ids"].to(device)
        r_mask = retain_batch["attention_mask"].to(device)

        # Forget loss: push hidden state toward random direction
        h_forget = extract_hidden_state(
            trainable_model, f_ids, f_mask, RMU_TARGET_LAYER
        )
        forget_loss = F.mse_loss(h_forget, RMU_STEERING_COEFF * u)

        # Retain loss: keep hidden state close to frozen reference
        h_frozen = extract_hidden_state(
            frozen_model, r_ids, r_mask, RMU_TARGET_LAYER, no_grad=True
        )
        h_retain = extract_hidden_state(
            trainable_model, r_ids, r_mask, RMU_TARGET_LAYER
        )
        retain_loss = F.mse_loss(h_retain, h_frozen)

        # Combined
        total_loss = forget_loss + RMU_ALPHA * retain_loss

        total_loss.backward()

        # Gradient clipping + step
        torch.nn.utils.clip_grad_norm_(trainable_params_list, RMU_MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        forget_val = forget_loss.item()
        retain_val = retain_loss.item() * RMU_ALPHA
        total_val = total_loss.item()
        step_time = time.time() - t0_step

        running_forget += forget_val
        running_retain += retain_val
        running_total += total_val

        if step < 5:
            print(
                f"  [DIAG] Step {step}: forget_mse={forget_val:.4f}, "
                f"retain_mse={retain_val:.4f}, total={total_val:.4f}, "
                f"time={step_time:.2f}s"
            )

        if (step + 1) % log_interval == 0:
            avg_f = running_forget / log_interval
            avg_r = running_retain / log_interval
            avg_t = running_total / log_interval
            elapsed = time.time() - t0_total
            eta = elapsed / (step + 1) * (RMU_MAX_STEPS - step - 1)
            print(
                f"  [Step {step+1}/{RMU_MAX_STEPS}] "
                f"forget_mse={avg_f:.4f} retain_mse={avg_r:.4f} "
                f"total={avg_t:.4f} "
                f"elapsed={elapsed:.1f}s ETA={eta:.1f}s"
            )
            running_forget = 0.0
            running_retain = 0.0
            running_total = 0.0

    train_time = time.time() - t0_total
    print(f"  RMU training done in {train_time:.0f}s ({train_time/60:.1f}min)")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ── Step 5: Save RMU model ───────────────────────────────────────────
    print("\n[5/6] Saving RMU model...")
    rmu_final_dir = PILOT_DIR / "rmu_final"
    trainable_model.save_pretrained(str(rmu_final_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(rmu_final_dir))
    print(f"  Saved to {rmu_final_dir}")

    # ── Step 6: Evaluate ─────────────────────────────────────────────────
    print("\n[6/6] Evaluating forget/retain PPL (SFT vs RMU)...")

    # Load SFT model for comparison
    sft_model = AutoModelForCausalLM.from_pretrained(
        str(SFT_CHECKPOINT),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    sft_model.eval()

    # Also keep RMU model in eval
    trainable_model.eval()

    # Sample for PPL
    forget_eval = random.sample(forget_subset, min(50, len(forget_subset)))
    retain_eval = random.sample(retain_for_training, min(50, len(retain_for_training)))

    sft_forget_ppl = compute_ppl(sft_model, forget_eval, tokenizer, MAX_SEQ, "SFT forget")
    sft_retain_ppl = compute_ppl(sft_model, retain_eval, tokenizer, MAX_SEQ, "SFT retain")
    rmu_forget_ppl = compute_ppl(trainable_model, forget_eval, tokenizer, MAX_SEQ, "RMU forget")
    rmu_retain_ppl = compute_ppl(trainable_model, retain_eval, tokenizer, MAX_SEQ, "RMU retain")

    # ── Results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PILOT RMU RESULTS (Qwen3-0.6B Full FT)")
    print("=" * 60)
    print(f"{'Metric':<25} {'SFT':>10} {'RMU':>15}")
    print("-" * 52)
    print(f"{'Forget PPL (↑=better)':<25} {sft_forget_ppl:>10.2f} {rmu_forget_ppl:>15.2f}")
    print(f"{'Retain PPL (↓=better)':<25} {sft_retain_ppl:>10.2f} {rmu_retain_ppl:>15.2f}")
    print(f"{'Forget PPL ratio':<25} {'1.00x':>10} {rmu_forget_ppl/sft_forget_ppl:>15.2f}x")
    print(f"{'Retain PPL ratio':<25} {'1.00x':>10} {rmu_retain_ppl/sft_retain_ppl:>15.2f}x")
    print(f"\nRMU training: {train_time:.0f}s")
    print(f"Max VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # Verdict
    print("\n" + "=" * 60)
    if rmu_forget_ppl > sft_forget_ppl * 1.2:
        print("VERDICT: RMU WORKS — forget PPL increased significantly")
    elif rmu_forget_ppl > sft_forget_ppl * 1.05:
        print("VERDICT: RMU has MARGINAL effect — forget PPL increased slightly")
    else:
        print("VERDICT: RMU FAILED — forget PPL did NOT increase")
        print("  (Consistent with the full paper's finding that RMU can be ineffective)")
    print("=" * 60)

    # Save results JSON
    results = {
        "model": "Qwen3-0.6B",
        "method": "rmu",
        "mode": "full_ft",
        "subset_size": SUBSET,
        "forget_set_size": len(forget_subset),
        "retain_set_size": len(retain_for_training),
        "rmu_config": {
            "learning_rate": RMU_LR,
            "alpha": RMU_ALPHA,
            "steering_coeff": RMU_STEERING_COEFF,
            "target_layer": RMU_TARGET_LAYER,
            "batch_size": RMU_BATCH_SIZE,
            "max_steps": RMU_MAX_STEPS,
            "max_grad_norm": RMU_MAX_GRAD_NORM,
        },
        "sft_forget_ppl": sft_forget_ppl,
        "sft_retain_ppl": sft_retain_ppl,
        "rmu_forget_ppl": rmu_forget_ppl,
        "rmu_retain_ppl": rmu_retain_ppl,
        "forget_ppl_ratio": rmu_forget_ppl / sft_forget_ppl,
        "retain_ppl_ratio": rmu_retain_ppl / sft_retain_ppl,
        "training_time_s": train_time,
        "max_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    results_path = PILOT_DIR / "pilot_rmu_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
