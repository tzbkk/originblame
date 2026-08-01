#!/usr/bin/env python3
"""Pilot NPO (Negative Preference Optimization) on Qwen3-0.6B Full FT.

Loads the already-trained SFT checkpoint, runs NPO unlearning on Berthe's
forget set (line-level), then compares forget/retain PPL before and after.

Reference: benchmarks/scripts/pipeline_qa/train_npo.py (537 lines)

NPO algorithm (Zhang et al. 2024, arXiv:2404.05868):
    L_NPO = -(2/β) * E[log σ(-β * (log π_θ(y|x) - log π_ref(y|x)))]
    L_retain = NLL(θ, x_retain)
    L_total = L_NPO(forget_set) + retain_weight * L_retain(retain_set)

Where:
- π_θ is the trainable model (starts from SFT checkpoint)
- π_ref is the frozen reference model (also SFT checkpoint, NOT updated)

Usage:
    cd /home/hxue/Projects/originblame
    python benchmarks/scripts/pipeline_qa/pilot_npo_06b.py
"""

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────
PILOT_DIR = Path("benchmarks/results/pipeline_qa/pilot_npo_06b")
DATA_FILE = Path("benchmarks/results/pipeline_qa/qa_chatml/data.jsonl")
FORGET_FILE = Path("benchmarks/results/pipeline_qa/qa_chatml/forget_sets.json")
SFT_CHECKPOINT = Path("benchmarks/results/pipeline_qa/pilot_06b/sft/final")
MAX_SEQ = 256
SEED = 42
SUBSET = 5000
FORGET_SIZE = 500

# NPO hyperparameters (from config.yaml npo section)
NPO_BETA = 0.1
NPO_LR = 1e-5
NPO_NUM_EPOCHS = 5
NPO_BATCH_SIZE = 1
NPO_GRAD_ACCUM = 8
NPO_WARMUP_RATIO = 0.1
NPO_WEIGHT_DECAY = 0.01
NPO_MAX_GRAD_NORM = 1.0
NPO_LR_SCHEDULER = "cosine"
NPO_RETAIN_WEIGHT = 1.0


def set_seed(seed: int):
    random.seed(seed)
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


# ── Log-probability computation (from train_npo.py) ────────────────────


def compute_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    no_grad: bool = False,
) -> torch.Tensor:
    """Compute log π(y|x) — SUM of token log-probs over non-padding tokens.

    Returns SUM of token log-probs. Shape (batch_size,).

    The NPO paper (Zhang et al. 2024, Eq.3) uses total log-probability:
        L_NPO = -(2/β) * E[log σ(-β * log(π_θ/π_ref))]
    """
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]

        log_probs_all = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs_all.gather(
            2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        token_log_probs = token_log_probs * shift_mask

        return token_log_probs.sum(dim=1)


# ── NPO loss (from train_npo.py) ───────────────────────────────────────


def compute_npo_loss(
    trainable_model,
    ref_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """NPO loss: -(2/β) * E[log σ(-β * (log π_θ - log π_ref))]."""
    log_pi_theta = compute_log_probs(trainable_model, input_ids, attention_mask, no_grad=False)
    log_pi_ref = compute_log_probs(ref_model, input_ids, attention_mask, no_grad=True)
    forget_log_ratio = log_pi_theta - log_pi_ref
    loss = -F.logsigmoid(-beta * forget_log_ratio) * 2.0 / beta
    return loss.mean()


# ── NLL loss (from train_npo.py) ───────────────────────────────────────


def compute_nll_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


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
    print("PILOT NPO: Qwen3-0.6B Full FT")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Step 1: Load data ────────────────────────────────────────────────
    print("\n[1/6] Loading data...")

    all_data_ordered = []
    with open(DATA_FILE) as f:
        for line in f:
            all_data_ordered.append(json.loads(line))
    print(f"  Total data: {len(all_data_ordered)}")

    with open(FORGET_FILE) as f:
        forget_sets = json.load(f)

    berthe_line_indices = set(forget_sets["Berthe"]["line"]["indices"])
    print(f"  Berthe line forget set: {len(berthe_line_indices)} indices")

    # Get forget records
    forget_records = [
        all_data_ordered[i] for i in sorted(berthe_line_indices) if i < len(all_data_ordered)
    ]
    print(f"  Forget records from Berthe line set: {len(forget_records)}")

    # Sample pilot forget subset
    random.seed(SEED)
    forget_subset = random.sample(forget_records, min(FORGET_SIZE, len(forget_records)))

    # Retain = everything NOT in forget set
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
    print("\n[2/6] Loading models (SFT trainable + SFT frozen reference)...")

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

    total_params = sum(p.numel() for p in trainable_model.parameters())
    print(f"  Trainable model: {total_params/1e6:.0f}M params (all trainable)")
    print(f"  VRAM after trainable: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Frozen reference model: ALSO SFT checkpoint (NOT base model)
    ref_model = AutoModelForCausalLM.from_pretrained(
        str(SFT_CHECKPOINT),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    print(f"  Frozen reference: SFT checkpoint (same weights)")
    print(f"  VRAM after frozen: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ── Step 3: Untie weights, setup optimizer, scheduler, dataloaders ───
    print("\n[3/6] Untying lm_head weights & setting up training...")

    trainable_model.lm_head.weight = torch.nn.Parameter(
        trainable_model.lm_head.weight.clone()
    )
    print("  Done — lm_head.weight cloned from embed_tokens")

    # Datasets & dataloaders
    forget_dataset = TextDataset(forget_subset, tokenizer, MAX_SEQ)
    retain_dataset = TextDataset(retain_for_training, tokenizer, MAX_SEQ)

    forget_loader = DataLoader(
        forget_dataset, batch_size=NPO_BATCH_SIZE, shuffle=True, drop_last=True
    )
    retain_loader = DataLoader(
        retain_dataset, batch_size=NPO_BATCH_SIZE, shuffle=True, drop_last=True
    )

    print(f"  Forget batches/epoch: {len(forget_loader)}, Retain batches/epoch: {len(retain_loader)}")

    # Optimizer
    trainable_params = [p for p in trainable_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=NPO_LR,
        weight_decay=NPO_WEIGHT_DECAY,
    )

    # Scheduler (cosine with warmup)
    total_steps = len(forget_loader) // NPO_GRAD_ACCUM * NPO_NUM_EPOCHS
    warmup_steps = int(total_steps * NPO_WARMUP_RATIO)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        if NPO_LR_SCHEDULER == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(
        f"  NPO config: β={NPO_BETA}, lr={NPO_LR}, epochs={NPO_NUM_EPOCHS}, "
        f"batch={NPO_BATCH_SIZE}, grad_accum={NPO_GRAD_ACCUM}"
    )
    print(f"  Total optimizer steps: {total_steps}, warmup: {warmup_steps}")
    print(f"  retain_weight={NPO_RETAIN_WEIGHT}")

    # ── Step 4: NPO training ─────────────────────────────────────────────
    print("\n[4/6] NPO training...")
    print(f"{'=' * 60}")
    print(f"NPO Training: Berthe / line")
    print(
        f"β={NPO_BETA}, lr={NPO_LR}, epochs={NPO_NUM_EPOCHS}, batch={NPO_BATCH_SIZE}, "
        f"grad_accum={NPO_GRAD_ACCUM}"
    )
    print(f"Forget: {len(forget_subset)} | Retain: {len(retain_for_training)}")
    print(f"{'=' * 60}\n")

    device = next(trainable_model.parameters()).device

    global_step = 0
    running_npo = 0.0
    running_retain = 0.0
    running_total = 0.0
    log_interval = 10
    t0_total = time.time()

    for epoch in range(NPO_NUM_EPOCHS):
        trainable_model.train()
        epoch_npo_loss = 0.0
        epoch_retain_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0

        retain_iter = iter(retain_loader)

        for step, forget_batch in enumerate(forget_loader):
            f_input_ids = forget_batch["input_ids"].to(device)
            f_attention_mask = forget_batch["attention_mask"].to(device)

            # Cycle retain loader if exhausted
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)

            r_input_ids = retain_batch["input_ids"].to(device)
            r_attention_mask = retain_batch["attention_mask"].to(device)
            r_labels = retain_batch["labels"].to(device)

            npo_loss = compute_npo_loss(
                trainable_model,
                ref_model,
                f_input_ids,
                f_attention_mask,
                NPO_BETA,
            )

            retain_loss = compute_nll_loss(
                trainable_model,
                r_input_ids,
                r_attention_mask,
                r_labels,
            )

            total_loss = (npo_loss + NPO_RETAIN_WEIGHT * retain_loss) / NPO_GRAD_ACCUM

            total_loss.backward()

            if (step + 1) % NPO_GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, NPO_MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                npo_val = npo_loss.item() * NPO_GRAD_ACCUM
                retain_val = retain_loss.item() * NPO_GRAD_ACCUM * NPO_RETAIN_WEIGHT
                total_val = total_loss.item() * NPO_GRAD_ACCUM

                running_npo += npo_val
                running_retain += retain_val
                running_total += total_val

                if global_step % log_interval == 0:
                    avg_npo = running_npo / log_interval
                    avg_retain = running_retain / log_interval
                    avg_total = running_total / log_interval
                    cur_lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0_total
                    print(
                        f"[Epoch {epoch + 1}/{NPO_NUM_EPOCHS} | Step {global_step}] "
                        f"npo={avg_npo:.4f} retain={avg_retain:.4f} "
                        f"total={avg_total:.4f} lr={cur_lr:.2e}"
                    )
                    running_npo = 0.0
                    running_retain = 0.0
                    running_total = 0.0

            epoch_npo_loss += npo_loss.item()
            epoch_retain_loss += retain_loss.item()
            epoch_total_loss += npo_loss.item() + NPO_RETAIN_WEIGHT * retain_loss.item()
            n_batches += 1

        avg_epoch_npo = epoch_npo_loss / max(1, n_batches)
        avg_epoch_retain = epoch_retain_loss / max(1, n_batches)
        avg_epoch_total = epoch_total_loss / max(1, n_batches)
        print(
            f"\n=== Epoch {epoch + 1}/{NPO_NUM_EPOCHS} Summary === "
            f"npo={avg_epoch_npo:.4f} retain={avg_epoch_retain:.4f} "
            f"total={avg_epoch_total:.4f}\n"
        )

    train_time = time.time() - t0_total
    print(f"  NPO training done in {train_time:.0f}s ({train_time/60:.1f}min)")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ── Step 5: Save NPO model ───────────────────────────────────────────
    print("\n[5/6] Saving NPO model...")
    npo_final_dir = PILOT_DIR / "npo_final"
    trainable_model.save_pretrained(str(npo_final_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(npo_final_dir))
    print(f"  Saved to {npo_final_dir}")

    # ── Step 6: Evaluate ─────────────────────────────────────────────────
    print("\n[6/6] Evaluating forget/retain PPL (SFT vs NPO)...")

    # Load SFT model for comparison
    sft_model = AutoModelForCausalLM.from_pretrained(
        str(SFT_CHECKPOINT),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    sft_model.eval()

    # Also keep NPO model in eval
    trainable_model.eval()

    # Sample for PPL
    forget_eval = random.sample(forget_subset, min(50, len(forget_subset)))
    retain_eval = random.sample(retain_for_training, min(50, len(retain_for_training)))

    sft_forget_ppl = compute_ppl(sft_model, forget_eval, tokenizer, MAX_SEQ, "SFT forget")
    sft_retain_ppl = compute_ppl(sft_model, retain_eval, tokenizer, MAX_SEQ, "SFT retain")
    npo_forget_ppl = compute_ppl(trainable_model, forget_eval, tokenizer, MAX_SEQ, "NPO forget")
    npo_retain_ppl = compute_ppl(trainable_model, retain_eval, tokenizer, MAX_SEQ, "NPO retain")

    # Free SFT model
    del sft_model
    torch.cuda.empty_cache()

    # ── Results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PILOT NPO RESULTS (Qwen3-0.6B Full FT)")
    print("=" * 60)
    print(f"{'Metric':<25} {'SFT':>10} {'NPO':>15}")
    print("-" * 52)
    print(f"{'Forget PPL (↑=better)':<25} {sft_forget_ppl:>10.2f} {npo_forget_ppl:>15.2f}")
    print(f"{'Retain PPL (↓=better)':<25} {sft_retain_ppl:>10.2f} {npo_retain_ppl:>15.2f}")
    print(f"{'Forget PPL ratio':<25} {'1.00x':>10} {npo_forget_ppl/sft_forget_ppl:>15.2f}x")
    print(f"{'Retain PPL ratio':<25} {'1.00x':>10} {npo_retain_ppl/sft_retain_ppl:>15.2f}x")
    print(f"\nNPO training: {train_time:.0f}s ({train_time/60:.1f}min)")
    print(f"Max VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # Verdict
    forget_ratio = npo_forget_ppl / sft_forget_ppl
    retain_ratio = npo_retain_ppl / sft_retain_ppl
    selectivity = forget_ratio / retain_ratio if retain_ratio > 0 else float("inf")

    print("\n" + "=" * 60)
    if forget_ratio > 1.5 and retain_ratio < 1.5:
        print(f"VERDICT: NPO WORKS — forget PPL {forget_ratio:.2f}x ↑, retain PPL {retain_ratio:.2f}x (selective)")
    elif forget_ratio > 1.2:
        print(f"VERDICT: NPO has MODERATE effect — forget PPL {forget_ratio:.2f}x ↑")
    else:
        print(f"VERDICT: NPO effect is WEAK — forget PPL {forget_ratio:.2f}x ↑")
    print(f"  Selectivity (forget_ratio/retain_ratio): {selectivity:.2f}")
    print("=" * 60)

    # Save results JSON
    results = {
        "model": "Qwen3-0.6B",
        "method": "npo",
        "mode": "full_ft",
        "subset_size": SUBSET,
        "forget_set_size": len(forget_subset),
        "retain_set_size": len(retain_for_training),
        "npo_config": {
            "beta": NPO_BETA,
            "learning_rate": NPO_LR,
            "num_epochs": NPO_NUM_EPOCHS,
            "batch_size": NPO_BATCH_SIZE,
            "gradient_accumulation_steps": NPO_GRAD_ACCUM,
            "warmup_ratio": NPO_WARMUP_RATIO,
            "weight_decay": NPO_WEIGHT_DECAY,
            "max_grad_norm": NPO_MAX_GRAD_NORM,
            "lr_scheduler_type": NPO_LR_SCHEDULER,
            "retain_weight": NPO_RETAIN_WEIGHT,
        },
        "sft_forget_ppl": sft_forget_ppl,
        "sft_retain_ppl": sft_retain_ppl,
        "npo_forget_ppl": npo_forget_ppl,
        "npo_retain_ppl": npo_retain_ppl,
        "forget_ppl_ratio": forget_ratio,
        "retain_ppl_ratio": retain_ratio,
        "selectivity": selectivity,
        "global_steps": global_step,
        "training_time_s": train_time,
        "max_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    results_path = PILOT_DIR / "pilot_npo_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
