#!/usr/bin/env python3
"""RMU (Representation Misdirection for Unlearning) training script.

Implements RMU from Li et al. 2024 (WMDP Benchmark):
    L_forget  = MSE(h_θ(x_forget, layer_L), c * u)
    L_retain  = MSE(h_θ(x_retain, layer_L), h_ref(x_retain, layer_L))
    L_total   = L_forget + α * L_retain

Full fine-tuning in bf16, no quantization.

Usage:
    python train_rmu.py --config config.yaml --author Berthe --forget-type line
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
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


# ── model loading ────────────────────────────────────────────────────────────


def load_models_rmu(
    cfg: dict, sft_adapter_path: str | None = None,
) -> tuple:
    """Load two model copies: trainable + frozen reference, both in bf16 Full FT.

    Returns (trainable_model, frozen_model, tokenizer).
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
        frozen_model = AutoModelForCausalLM.from_pretrained(
            sft_adapter_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
    else:
        frozen_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )

    frozen_model.eval()
    for param in frozen_model.parameters():
        param.requires_grad = False

    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Frozen reference model loaded ({vram_gb:.1f} GB on GPU)")
    return trainable_model, frozen_model, tokenizer


# ── hidden state extraction ──────────────────────────────────────────────────


def get_decoder_layers(model) -> torch.nn.ModuleList:
    """Navigate model to get the decoder layer ModuleList."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    return model.layers


def extract_hidden_state(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layer: int,
    no_grad: bool = False,
) -> torch.Tensor:
    """Forward pass through model, capture hidden state at target_layer via hook.

    Args:
        model: The model.
        input_ids: Token IDs (batch, seq_len).
        attention_mask: Attention mask (batch, seq_len).
        target_layer: Index of decoder layer to extract from.
        no_grad: If True, run under torch.no_grad() and detach the result.

    Returns:
        Hidden state tensor of shape (batch, seq_len, hidden_dim).
        If no_grad=False, the tensor retains grad_fn for backprop.
    """
    layers = get_decoder_layers(model)
    captured: dict[str, torch.Tensor] = {}

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
        result = captured["h"]  # keep grad_fn for backprop

    handle.remove()
    return result


# ── RMU training loop ────────────────────────────────────────────────────────


def train(
    cfg: dict,
    author: str,
    forget_type: str,
    sft_adapter_path: str | None,
) -> None:
    seed = cfg.get("seed", 42)
    set_seed(seed)

    rmu_cfg = cfg["rmu"]
    max_steps = rmu_cfg["max_steps"]
    batch_size = rmu_cfg["per_device_train_batch_size"]
    lr = rmu_cfg["learning_rate"]
    alpha = rmu_cfg["alpha"]
    steering_coeff = rmu_cfg["steering_coeff"]
    target_layer_raw = rmu_cfg["target_layer"]
    max_grad_norm = rmu_cfg.get("max_grad_norm", 1.0)
    weight_decay = rmu_cfg.get("weight_decay", 0.01)
    grad_accum = rmu_cfg.get("gradient_accumulation_steps", 1)
    max_seq_length = cfg["model"]["max_seq_length"]

    # ── Data ─────────────────────────────────────────────────────────────
    data_file = cfg["data"]["data_file"]
    forget_sets_file = str(Path(cfg["data"]["ob_data_dir"]) / "forget_sets.json")

    # Fallback: check results dir if not in data dir
    if not os.path.isfile(forget_sets_file):
        alt = Path(__file__).resolve().parent / "results" / "forget_sets.json"
        if alt.is_file():
            forget_sets_file = str(alt)
            print(f"Using forget_sets.json from {alt}")

    forget_records, retain_records = load_forget_retain_sets(
        data_file, forget_sets_file, author, forget_type
    )

    # ── Models ───────────────────────────────────────────────────────────
    trainable_model, frozen_model, tokenizer = load_models_rmu(
        cfg, sft_adapter_path
    )
    device = next(trainable_model.parameters()).device

    forget_dataset = TextDataset(forget_records, tokenizer, max_seq_length)
    retain_dataset = TextDataset(retain_records, tokenizer, max_seq_length)

    forget_loader = DataLoader(
        forget_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    retain_loader = DataLoader(
        retain_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # ── DIAGNOSTIC: Data loading stats ─────────────────────────────────────
    print(f"[DIAG] Forget dataset size: {len(forget_dataset)}")
    print(f"[DIAG] Retain dataset size: {len(retain_dataset)}")
    print(f"[DIAG] Forget batches (drop_last): {len(forget_loader)}")
    print(f"[DIAG] Retain batches (drop_last): {len(retain_loader)}")
    print(f"[DIAG] Batch size: {batch_size}, Grad accum: {grad_accum}")
    print(f"[DIAG] Max steps: {max_steps}, Effective steps: {max_steps // grad_accum}")

    # ── Optimizer ──────────────────────────────────────────────────────
    trainable_params = [p for p in trainable_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
    )

    hidden_dim = trainable_model.config.hidden_size
    num_layers = len(get_decoder_layers(trainable_model))
    target_layer = num_layers // 2 if target_layer_raw == "auto" else int(target_layer_raw)
    compute_dtype = getattr(torch, cfg["model"].get("compute_dtype", "bfloat16"))
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.get("seed", 42))
    u = torch.randn(hidden_dim, generator=gen, device=device, dtype=compute_dtype)
    u = u / u.norm()
    print(f"Sampled unit vector u: dim={hidden_dim}, device={device}")
    print(
        f"[DIAG] Steering vector u: shape={u.shape}, norm={u.norm().item():.4f}, "
        f"mean={u.mean().item():.6f}, std={u.std().item():.6f}"
    )
    print(f"[DIAG] steering_coeff * u: norm={(steering_coeff * u).norm().item():.4f}")
    print(
        f"[DIAG] Target layer: {target_layer}, Decoder layers count: {num_layers}"
    )

    # ── Output directory ─────────────────────────────────────────────────
    base_output = rmu_cfg["output_dir"]
    output_dir = Path(base_output) / f"{author}_{forget_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(output_dir / "tb"))

    print(f"\n{'=' * 60}")
    print(f"RMU Training: {author} / {forget_type}")
    print(
        f"lr={lr}, max_steps={max_steps}, batch={batch_size}, grad_accum={grad_accum}"
    )
    print(
        f"alpha={alpha}, steering_coeff={steering_coeff}, target_layer={target_layer}"
    )
    print(f"Forget: {len(forget_records)} | Retain: {len(retain_records)}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")

    # ── Training loop (step-based) ───────────────────────────────────────
    import time

    global_step = 0
    optimizer_step = 0
    running_forget = 0.0
    running_retain = 0.0
    running_total = 0.0
    log_interval = 10

    forget_iter = iter(forget_loader)
    retain_iter = iter(retain_loader)

    trainable_model.train()
    t0_total = time.time()

    for step in range(max_steps):
        t0_step = time.time()

        # Cycle loaders if exhausted
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
        f_input_ids = forget_batch["input_ids"].to(device)
        f_attention_mask = forget_batch["attention_mask"].to(device)

        r_input_ids = retain_batch["input_ids"].to(device)
        r_attention_mask = retain_batch["attention_mask"].to(device)

        if step == 0:
            print(
                f"[DIAG] First batch - f_input_ids: {f_input_ids.shape}, dtype={f_input_ids.dtype}"
            )
            print(
                f"[DIAG] First batch - f_attention_mask: {f_attention_mask.shape}, "
                f"nonzero={f_attention_mask.sum().item()}/{f_attention_mask.numel()}"
            )
            print(f"[DIAG] First batch - r_input_ids: {r_input_ids.shape}")

        # ── Forget loss: push hidden state toward random direction ────
        h_forget = extract_hidden_state(
            trainable_model, f_input_ids, f_attention_mask, target_layer
        )

        if step == 0:
            print(
                f"[DIAG] h_forget: shape={h_forget.shape}, dtype={h_forget.dtype}, "
                f"requires_grad={h_forget.requires_grad}, grad_fn={h_forget.grad_fn}"
            )
            print(
                f"[DIAG] h_forget stats: mean={h_forget.mean().item():.6f}, "
                f"std={h_forget.std().item():.6f}, min={h_forget.min().item():.6f}, "
                f"max={h_forget.max().item():.6f}"
            )
            print(
                f"[DIAG] h_forget norm (per-token mean): {h_forget.norm(dim=-1).mean().item():.4f}"
            )
            target_vec = steering_coeff * u
            print(
                f"[DIAG] Target vector stats: norm={target_vec.norm().item():.4f}, "
                f"mean={target_vec.mean().item():.6f}"
            )

        forget_loss = F.mse_loss(h_forget, steering_coeff * u)

        # ── Retain loss: keep hidden state close to frozen reference ───
        # Frozen model forward (no gradients)
        h_frozen_retain = extract_hidden_state(
            frozen_model, r_input_ids, r_attention_mask, target_layer, no_grad=True
        )
        # Trainable model forward (with gradients)
        h_retain = extract_hidden_state(
            trainable_model, r_input_ids, r_attention_mask, target_layer
        )

        if step == 0:
            print(
                f"[DIAG] h_frozen_retain: shape={h_frozen_retain.shape}, "
                f"mean={h_frozen_retain.mean().item():.6f}, std={h_frozen_retain.std().item():.6f}"
            )
            print(
                f"[DIAG] h_retain: shape={h_retain.shape}, "
                f"requires_grad={h_retain.requires_grad}, grad_fn={h_retain.grad_fn}"
            )
            diff = (h_retain.detach() - h_frozen_retain).abs()
            print(
                f"[DIAG] retain diff (trainable vs frozen): mean={diff.mean().item():.8f}, "
                f"max={diff.max().item():.8f}"
            )

        retain_loss = F.mse_loss(h_retain, h_frozen_retain)

        # ── Combined loss ─────────────────────────────────────────────
        total_loss = (forget_loss + alpha * retain_loss) / grad_accum

        if step == 0:
            print(
                f"[DIAG] forget_loss={forget_loss.item():.6f}, "
                f"retain_loss={retain_loss.item():.6f}, alpha*retain={alpha * retain_loss.item():.6f}"
            )
            print(
                f"[DIAG] total_loss={total_loss.item():.6f}, "
                f"total_loss.requires_grad={total_loss.requires_grad}, "
                f"total_loss.grad_fn={total_loss.grad_fn}"
            )

        total_loss.backward()

        # ── Gradient diagnostics (first step) ─────────────────────────
        if step == 0:
            total_norm = 0.0
            n_grad_params = 0
            for p in trainable_params:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm**2
                    n_grad_params += 1
            total_norm = total_norm**0.5
            print(
                f"[DIAG] After backward: grad_norm={total_norm:.6f}, "
                f"params_with_grad={n_grad_params}/{len(trainable_params)}"
            )

        # ── Gradient accumulation ─────────────────────────────────────
        global_step += 1
        if global_step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            optimizer_step += 1

        # ── Logging ───────────────────────────────────────────────────
        forget_val = forget_loss.item() * grad_accum
        retain_val = retain_loss.item() * alpha * grad_accum
        total_val = total_loss.item() * grad_accum

        cur_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/forget_loss", forget_val, global_step)
        writer.add_scalar("train/retain_loss", retain_val, global_step)
        writer.add_scalar("train/total_loss", total_val, global_step)
        writer.add_scalar("train/learning_rate", cur_lr, global_step)

        step_time = time.time() - t0_step

        running_forget += forget_val
        running_retain += retain_val
        running_total += total_val

        if step < 5:
            print(
                f"[DIAG] Step {step}: forget={forget_val:.6f}, retain={retain_val:.6f}, "
                f"total={total_val:.6f}, time={step_time:.3f}s"
            )

        if (step + 1) % log_interval == 0:
            avg_forget = running_forget / log_interval
            avg_retain = running_retain / log_interval
            avg_total = running_total / log_interval
            elapsed = time.time() - t0_total
            eta = elapsed / (step + 1) * (max_steps - step - 1)
            print(
                f"[Step {step + 1}/{max_steps}] "
                f"forget_mse={avg_forget:.4f} retain_mse={avg_retain:.4f} "
                f"total={avg_total:.4f} "
                f"time={elapsed:.1f}s ETA={eta:.1f}s"
            )
            running_forget = 0.0
            running_retain = 0.0
            running_total = 0.0

    total_time = time.time() - t0_total
    print(
        f"[DIAG] Total training time: {total_time:.1f}s for {max_steps} steps "
        f"({total_time / max_steps:.3f}s/step)"
    )

    final_loss = total_val if max_steps > 0 else 0.0

    # ── Save final model ───────────────────────────────────────────────
    final_dir = output_dir / "final"
    trainable_model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    print(f"\nSaved final model: {final_dir}")

    # ── Save training metadata ───────────────────────────────────────────
    meta = {
        "method": "rmu",
        "mode": "full_finetune",
        "author": author,
        "forget_type": forget_type,
        "learning_rate": lr,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "alpha": alpha,
        "steering_coeff": steering_coeff,
        "target_layer": target_layer,
        "max_grad_norm": max_grad_norm,
        "weight_decay": weight_decay,
        "max_seq_length": max_seq_length,
        "total_steps": global_step,
        "training_time_seconds": round(total_time, 2),
        "final_loss": round(final_loss, 6),
        "sft_adapter_path": sft_adapter_path,
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    writer.close()
    print("Done.")


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="RMU (Representation Misdirection) unlearning training"
    )
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

    # Resolve SFT adapter path
    sft_adapter = args.sft_adapter
    if sft_adapter is None:
        sft_base = Path(cfg["sft"]["output_dir"])
        candidate = sft_base / args.author
        if candidate.is_dir():
            sft_adapter = str(candidate)
        else:
            if sft_base.is_dir():
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
