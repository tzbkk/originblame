#!/usr/bin/env python3
"""Phase 3: NPO (Negative Preference Optimization) unlearning script.

Implements the NPO loss from Zhang et al. 2024 (arXiv:2404.05868):
    L_NPO = -(2/β) * E[log σ(-β * (log π_θ(y|x) / π_ref(y|x)))]

Combined with retain NLL loss to preserve utility:
    L_total = L_NPO(forget_set) + retain_weight * L_NLL(retain_set)

Usage:
    python benchmarks/scripts/pipeline_qa/train_npo.py --config benchmarks/scripts/pipeline_qa/config.yaml \
        --author Berthe --forget-type line
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


# ── data ─────────────────────────────────────────────────────────────────────


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

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_forget_retain_sets(
    data_file: str,
    forget_sets_file: str,
    author: str,
    forget_type: str,
) -> tuple[list[dict], list[dict]]:
    """Load forget and retain records from data.jsonl + forget_sets.json.

    forget_sets.json nested format:
        {"Author": {"line": {"indices": [...], "count": N}, ...}}
    """
    with open(data_file) as f:
        all_records = [json.loads(line) for line in f if line.strip()]

    with open(forget_sets_file) as f:
        forget_sets = json.load(f)

    author_data = forget_sets[author]
    entry = author_data[forget_type]
    forget_indices = set(entry["indices"])

    forget_records = [
        all_records[i] for i in sorted(forget_indices) if i < len(all_records)
    ]
    retain_records = [r for i, r in enumerate(all_records) if i not in forget_indices]

    print(
        f"Loaded {len(forget_records)} forget, {len(retain_records)} retain "
        f"(total {len(all_records)})"
    )
    return forget_records, retain_records


# ── log-probability computation ──────────────────────────────────────────────


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


# ── model loading ────────────────────────────────────────────────────────────


def load_models(cfg: dict, sft_adapter_path: str | None = None):
    """Load two model copies: trainable + frozen reference, both in bf16 Full FT.

    Returns (trainable_model, frozen_reference_model, tokenizer).
    """
    model_path = cfg["model"]["base_path"]
    torch_dtype = getattr(torch, cfg["model"].get("compute_dtype", "bfloat16"))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Trainable model: load SFT checkpoint (or base) ───────────────────
    if sft_adapter_path and os.path.isdir(sft_adapter_path):
        trainable_model = AutoModelForCausalLM.from_pretrained(
            sft_adapter_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        print(f"Loaded SFT checkpoint from {sft_adapter_path}")
    else:
        trainable_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        print("No SFT checkpoint found; using base model")

    trainable_model.config.use_cache = False
    trainable_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    trainable_model.lm_head.weight = torch.nn.Parameter(trainable_model.lm_head.weight.clone())

    trainable_params = sum(p.numel() for p in trainable_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in trainable_model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # ── Frozen reference model: separate copy, same weights ──────────────
    if sft_adapter_path and os.path.isdir(sft_adapter_path):
        ref_model = AutoModelForCausalLM.from_pretrained(
            sft_adapter_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    print("Reference model loaded (frozen)")
    return trainable_model, ref_model, tokenizer


# ── NPO loss ─────────────────────────────────────────────────────────────────


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


# ── training loop ────────────────────────────────────────────────────────────


def train(
    cfg: dict,
    author: str,
    forget_type: str,
    sft_adapter_path: str | None,
) -> None:
    seed = cfg.get("seed", 42)
    set_seed(seed)

    npo_cfg = cfg["npo"]
    beta = npo_cfg["beta"]
    num_epochs = npo_cfg["num_train_epochs"]
    batch_size = npo_cfg["per_device_train_batch_size"]
    grad_accum = npo_cfg["gradient_accumulation_steps"]
    lr = npo_cfg["learning_rate"]
    warmup_ratio = npo_cfg["warmup_ratio"]
    weight_decay = npo_cfg["weight_decay"]
    max_grad_norm = npo_cfg["max_grad_norm"]
    lr_scheduler_type = npo_cfg["lr_scheduler_type"]
    max_seq_length = cfg["model"]["max_seq_length"]

    retain_weight = npo_cfg.get("retain_weight", 1.0)

    data_file = cfg["data"]["data_file"]
    forget_sets_file = str(Path(cfg["data"]["ob_data_dir"]) / "forget_sets.json")

    forget_records, retain_records = load_forget_retain_sets(
        data_file, forget_sets_file, author, forget_type
    )

    trainable_model, ref_model, tokenizer = load_models(cfg, sft_adapter_path)
    device = next(trainable_model.parameters()).device

    forget_dataset = TextDataset(forget_records, tokenizer, max_seq_length)
    retain_dataset = TextDataset(retain_records, tokenizer, max_seq_length)

    forget_loader = DataLoader(
        forget_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    retain_loader = DataLoader(
        retain_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    trainable_params = [p for p in trainable_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
    )

    total_steps = len(forget_loader) // grad_accum * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        if lr_scheduler_type == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    output_dir = Path(npo_cfg["output_dir"]) / f"{author}_{forget_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(output_dir / "tb"))

    print(f"\n{'=' * 60}")
    print(f"NPO Training: {author} / {forget_type}")
    print(
        f"β={beta}, lr={lr}, epochs={num_epochs}, batch={batch_size}, "
        f"grad_accum={grad_accum}"
    )
    print(f"Forget: {len(forget_records)} | Retain: {len(retain_records)}")
    print(f"Total optimizer steps: {total_steps}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")

    # ── Training loop ────────────────────────────────────────────────────
    global_step = 0
    running_npo = 0.0
    running_retain = 0.0
    running_total = 0.0
    log_interval = 10
    t0 = time.time()

    for epoch in range(num_epochs):
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
                beta,
            )

            retain_loss = compute_nll_loss(
                trainable_model,
                r_input_ids,
                r_attention_mask,
                r_labels,
            )

            total_loss = (npo_loss + retain_weight * retain_loss) / grad_accum

            total_loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                npo_val = npo_loss.item() * grad_accum
                retain_val = retain_loss.item() * grad_accum * retain_weight
                total_val = total_loss.item() * grad_accum

                cur_lr = scheduler.get_last_lr()[0]
                writer.add_scalar("train/npo_loss", npo_val, global_step)
                writer.add_scalar("train/retain_loss", retain_val, global_step)
                writer.add_scalar("train/total_loss", total_val, global_step)
                writer.add_scalar("train/learning_rate", cur_lr, global_step)

                running_npo += npo_val
                running_retain += retain_val
                running_total += total_val

                if global_step % log_interval == 0:
                    avg_npo = running_npo / log_interval
                    avg_retain = running_retain / log_interval
                    avg_total = running_total / log_interval
                    print(
                        f"[Epoch {epoch + 1}/{num_epochs} | Step {global_step}] "
                        f"npo={avg_npo:.4f} retain={avg_retain:.4f} "
                        f"total={avg_total:.4f} lr={cur_lr:.2e}"
                    )
                    running_npo = 0.0
                    running_retain = 0.0
                    running_total = 0.0

            epoch_npo_loss += npo_loss.item()
            epoch_retain_loss += retain_loss.item()
            epoch_total_loss += npo_loss.item() + retain_weight * retain_loss.item()
            n_batches += 1

        avg_epoch_npo = epoch_npo_loss / max(1, n_batches)
        avg_epoch_retain = epoch_retain_loss / max(1, n_batches)
        avg_epoch_total = epoch_total_loss / max(1, n_batches)
        writer.add_scalar("epoch/npo_loss", avg_epoch_npo, epoch + 1)
        writer.add_scalar("epoch/retain_loss", avg_epoch_retain, epoch + 1)
        writer.add_scalar("epoch/total_loss", avg_epoch_total, epoch + 1)
        print(
            f"\n=== Epoch {epoch + 1}/{num_epochs} Summary === "
            f"npo={avg_epoch_npo:.4f} retain={avg_epoch_retain:.4f} "
            f"total={avg_epoch_total:.4f}\n"
        )

        epoch_dir = output_dir / f"checkpoint-epoch-{epoch + 1}"
        trainable_model.save_pretrained(epoch_dir, safe_serialization=True)
        tokenizer.save_pretrained(epoch_dir)
        print(f"Saved checkpoint: {epoch_dir}")

    training_time = time.time() - t0
    final_loss = avg_epoch_total if num_epochs > 0 else 0.0

    # ── Save final model ───────────────────────────────────────────────
    final_dir = output_dir / "final"
    trainable_model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    print(f"\nSaved final model: {final_dir}")

    meta = {
        "author": author,
        "forget_type": forget_type,
        "beta": beta,
        "learning_rate": lr,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "retain_weight": retain_weight,
        "max_seq_length": max_seq_length,
        "total_steps": global_step,
        "training_time_seconds": round(training_time, 2),
        "final_loss": round(final_loss, 6),
        "sft_adapter_path": sft_adapter_path,
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    writer.close()
    print("Done.")


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase 3: NPO unlearning training")
    parser.add_argument(
        "--config",
        type=str,
        default="benchmarks/scripts/pipeline_qa/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--author",
        type=str,
        required=True,
        help="Author to forget (e.g. Berthe)",
    )
    parser.add_argument(
        "--forget-type",
        type=str,
        choices=["line", "page_prototype", "random", "emb_sim", "2x_random", "embedding"],
        help="Forget set type: line (ob provenance) or random (baseline)",
    )
    parser.add_argument(
        "--sft-adapter",
        type=str,
        default=None,
        help="Path to SFT checkpoint directory. "
        "Defaults to checkpoints/sft/{author}/",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    sft_adapter = args.sft_adapter
    if sft_adapter is None:
        sft_base = Path(cfg["sft"]["output_dir"])
        candidate = sft_base / args.author
        if candidate.is_dir():
            sft_adapter = str(candidate)
        elif sft_base.is_dir():
            sft_adapter = str(sft_base)
        else:
            print(
                f"Warning: No SFT adapter found at {candidate} or "
                f"{sft_base}. Starting from base model."
            )
            sft_adapter = None

    train(cfg, args.author, args.forget_type, sft_adapter)


if __name__ == "__main__":
    main()
