#!/usr/bin/env python3
"""Phase Retrain-Oracle: Full FT SFT on (all data MINUS forget set).

Gold-standard retrain oracle for machine unlearning: trains a Full FT SFT
model on all data excluding the forget set for a given author. Produces the
upper-bound on unlearning quality — what perfect removal would look like.

Usage:
    python benchmarks/scripts/pipeline_qa/train_retrain.py \
        --config benchmarks/scripts/pipeline_qa/config.yaml \
        --author Berthe --forget-type line
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── data ─────────────────────────────────────────────────────────────────────


def load_forget_retain_sets(
    data_file: str,
    forget_sets_file: str,
    author: str,
    forget_type: str,
) -> tuple[list[dict], list[dict], int]:
    """Load forget and retain records from data.jsonl + forget_sets.json.

    Returns (forget_records, retain_records, total_count).

    forget_sets.json nested format:
        {"Author": {"line": {"indices": [...], "count": N}, ...}}
    """
    with open(data_file, "r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f if line.strip()]

    with open(forget_sets_file, "r", encoding="utf-8") as f:
        forget_sets = json.load(f)

    if author not in forget_sets:
        raise KeyError(f"Author '{author}' not found in forget_sets.json")
    if forget_type not in forget_sets[author]:
        raise KeyError(
            f"Forget type '{forget_type}' not found for author '{author}'"
        )

    entry = forget_sets[author][forget_type]
    forget_indices = set(entry["indices"])

    forget_records = [
        all_records[i] for i in sorted(forget_indices) if i < len(all_records)
    ]
    retain_records = [
        r for i, r in enumerate(all_records) if i not in forget_indices
    ]

    print(
        f"Loaded {len(forget_records)} forget, {len(retain_records)} retain "
        f"(total {len(all_records)})"
    )
    return forget_records, retain_records, len(all_records)


def preprocess_dataset(samples: list[dict], tokenizer, max_length: int) -> Dataset:
    """Tokenize ChatML messages for causal LM fine-tuning.

    Each sample must have a ``messages`` key (list of role/content dicts).
    The tokenizer's chat_template renders the conversation into text, which
    is then tokenised.  Labels mask padding tokens with -100.
    """
    processed = []
    skipped = 0
    for sample in samples:
        if "messages" not in sample:
            skipped += 1
            continue
        text = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = [
            lid if mask == 1 else -100
            for lid, mask in zip(input_ids, attention_mask)
        ]
        processed.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })
    if skipped:
        print(f"  Skipped {skipped} samples without 'messages' key")

    dataset = Dataset.from_list(processed)
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    return dataset


# ── model ────────────────────────────────────────────────────────────────────


def load_model_and_tokenizer(model_path: str, max_seq_length: int, torch_dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )

    model.config.use_cache = False
    model.config.max_seq_length = max_seq_length
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())

    return model, tokenizer


# ── training ─────────────────────────────────────────────────────────────────


def build_training_arguments(
    cfg: dict, output_dir: str, log_dir: str
) -> TrainingArguments:
    sft = cfg["sft"]
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=sft["num_train_epochs"],
        per_device_train_batch_size=sft["per_device_train_batch_size"],
        gradient_accumulation_steps=sft["gradient_accumulation_steps"],
        learning_rate=sft["learning_rate"],
        warmup_ratio=sft["warmup_ratio"],
        weight_decay=sft["weight_decay"],
        lr_scheduler_type=sft["lr_scheduler_type"],
        bf16=sft["bf16"],
        fp16=sft["fp16"],
        logging_steps=sft["logging_steps"],
        save_strategy="no",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",
        seed=cfg.get("seed", 42),
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        logging_dir=log_dir,
        logging_strategy="steps",
        report_to="tensorboard",
        max_grad_norm=1.0,
    )


def print_training_summary(trainer) -> None:
    log_history = trainer.state.log_history

    train_losses = [
        (entry["step"], entry["loss"])
        for entry in log_history
        if "loss" in entry and "train_runtime" not in entry
    ]

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    if train_losses:
        print(f"\n  Total training steps: {trainer.state.max_steps}")
        print(f"  Total epochs: {trainer.state.num_train_epochs:.1f}")
        print(
            f"  Effective batch size: "
            f"{trainer.args.per_device_train_batch_size}"
            f" * {trainer.args.gradient_accumulation_steps}"
        )
        print(f"\n  Loss curve (every {trainer.args.logging_steps} steps):")
        print(f"  {'Step':>8s}  {'Loss':>8s}")
        print(f"  {'----':>8s}  {'----':>8s}")
        for step, loss in train_losses:
            print(f"  {step:>8d}  {loss:>8.4f}")

        first_loss = train_losses[0][1]
        last_loss = train_losses[-1][1]
        print(f"\n  Initial loss: {first_loss:.4f}")
        print(f"  Final loss:   {last_loss:.4f}")
        print(
            f"  Loss reduction: {first_loss - last_loss:.4f} "
            f"({(first_loss - last_loss) / first_loss * 100:.1f}%)"
        )

    for entry in log_history:
        if "train_runtime" in entry:
            print(
                f"\n  Total training time: "
                f"{entry['train_runtime']:.1f}s "
                f"({entry['train_runtime'] / 3600:.2f}h)"
            )
            print(
                f"  Training throughput: "
                f"{entry.get('train_samples_per_second', 'N/A')} samples/s"
            )
            break

    print("=" * 60)


# ── PPL evaluation ───────────────────────────────────────────────────────────


@torch.no_grad()
def compute_ppl(
    model,
    tokenizer,
    records: list[dict],
    max_length: int,
    batch_size: int = 4,
    max_samples: int = 1000,
) -> float:
    """Compute perplexity on a set of records."""
    model.eval()
    device = next(model.parameters()).device
    ppls: list[float] = []

    eval_recs = records
    if max_samples and len(eval_recs) > max_samples:
        rng = random.Random(42)
        eval_recs = rng.sample(records, max_samples)

    n_batches = (len(eval_recs) + batch_size - 1) // batch_size

    for bidx in range(n_batches):
        batch = eval_recs[bidx * batch_size : (bidx + 1) * batch_size]
        texts = []
        for rec in batch:
            if "messages" in rec:
                texts.append(
                    tokenizer.apply_chat_template(
                        rec["messages"],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
            else:
                texts.append(rec.get("text", ""))

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
                mean_nll = min(mean_nll, 20.0)
                ppls.append(math.exp(mean_nll))

        if (bidx + 1) % 25 == 0 or bidx + 1 == n_batches:
            print(f"    PPL batch {bidx + 1}/{n_batches}")

    return sum(ppls) / len(ppls) if ppls else float("inf")


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Retrain Oracle: Full FT SFT on (all data - forget set)"
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
        help="Author whose data to exclude (e.g. Berthe)",
    )
    parser.add_argument(
        "--forget-type",
        type=str,
        required=True,
        choices=["line", "page_prototype", "random", "emb_sim", "2x_random", "embedding"],
        help="Forget set type",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    cfg = load_config(str(config_path))

    project_root = Path.cwd()
    model_path = project_root / cfg["model"]["base_path"]
    data_file = project_root / cfg["data"]["data_file"]
    forget_sets_file = project_root / cfg["data"]["ob_data_dir"] / "forget_sets.json"
    max_seq_length = cfg["model"]["max_seq_length"]

    # Output directory: dedicated retrain.output_dir or fallback from sft
    retrain_cfg = cfg.get("retrain", {})
    if "output_dir" in retrain_cfg:
        base_output = project_root / retrain_cfg["output_dir"]
    else:
        base_output = project_root / cfg["sft"]["output_dir"].replace("sft", "retrain")
    output_dir = base_output / f"{args.author}_{args.forget_type}" / "final"
    log_dir = base_output / f"{args.author}_{args.forget_type}" / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    print(f"{'=' * 60}")
    print(f"RETRAIN ORACLE: {args.author} / {args.forget_type}")
    print(f"{'=' * 60}")
    print(f"Model path:    {model_path}")
    print(f"Data file:     {data_file}")
    print(f"Forget sets:   {forget_sets_file}")
    print(f"Output dir:    {output_dir}")
    print(f"Log dir:       {log_dir}")
    print(f"Seed:          {seed}")
    print()

    # ── Load data: retain only (exclude forget set) ────────────────────────
    print("Loading data (retain = all - forget)...")
    forget_records, retain_records, total_count = load_forget_retain_sets(
        str(data_file),
        str(forget_sets_file),
        args.author,
        args.forget_type,
    )
    print(
        f"Training on {len(retain_records)} retain samples "
        f"(excluded {len(forget_records)} forget samples "
        f"out of {total_count} total)"
    )

    # ── Load model ─────────────────────────────────────────────────────────
    torch_dtype = getattr(torch, cfg["model"].get("compute_dtype", "bfloat16"))
    print(f"\nLoading model (Full FT, {cfg['model'].get('compute_dtype', 'bfloat16')})...")
    model, tokenizer = load_model_and_tokenizer(
        str(model_path), max_seq_length, torch_dtype
    )

    # ── Tokenize retain dataset ────────────────────────────────────────────
    print("Tokenizing retain dataset...")
    train_dataset = preprocess_dataset(
        retain_records, tokenizer, max_seq_length
    )

    # ── Build trainer ──────────────────────────────────────────────────────
    training_args = build_training_arguments(cfg, str(output_dir), str(log_dir))

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=False,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    sft = cfg["sft"]
    effective_batch = (
        sft["per_device_train_batch_size"] * sft["gradient_accumulation_steps"]
    )
    total_steps = len(train_dataset) * sft["num_train_epochs"] // effective_batch

    print(f"\nStarting RETRAIN training...")
    print(f"  Epochs:          {sft['num_train_epochs']}")
    print(f"  Batch size:      {sft['per_device_train_batch_size']}")
    print(f"  Grad accum:      {sft['gradient_accumulation_steps']}")
    print(f"  Effective batch: {effective_batch}")
    print(f"  Learning rate:   {sft['learning_rate']}")
    print(f"  Scheduler:       {sft['lr_scheduler_type']}")
    print(f"  Max seq length:  {max_seq_length}")
    print(f"  Dataset size:    {len(train_dataset)}")
    print(f"  Total steps:     ~{total_steps}")
    print(f"  Optimizer:       paged_adamw_32bit")
    print()

    t0 = time.time()
    train_result = trainer.train()
    training_time = time.time() - t0

    # ── Save model ─────────────────────────────────────────────────────────
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print_training_summary(trainer)

    # ── Save training metadata ─────────────────────────────────────────────
    meta = {
        "author": args.author,
        "forget_type": args.forget_type,
        "training_samples": len(retain_records),
        "forget_excluded_count": len(forget_records),
        "total_data_count": total_count,
        "seed": seed,
        "training_time_seconds": round(training_time, 2),
        "training_time_hours": round(training_time / 3600, 3),
        "num_train_epochs": sft["num_train_epochs"],
        "learning_rate": sft["learning_rate"],
        "batch_size": sft["per_device_train_batch_size"],
        "gradient_accumulation_steps": sft["gradient_accumulation_steps"],
        "max_seq_length": max_seq_length,
        "model_name": cfg["model"]["name"],
        "method": "retrain_oracle",
    }
    meta_path = output_dir / "training_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved training metadata: {meta_path}")

    # ── Evaluate PPL on forget and retain sets ─────────────────────────────
    print(f"\n{'=' * 60}")
    print("Post-training PPL evaluation")
    print(f"{'=' * 60}")

    eval_batch_size = cfg["eval"]["per_device_eval_batch_size"]
    eval_max_samples = cfg["pipeline"].get("eval_num_samples", 200)

    print(f"\nForget PPL ({len(forget_records)} samples, eval {eval_max_samples})...")
    forget_ppl = compute_ppl(
        model, tokenizer, forget_records, max_seq_length,
        batch_size=eval_batch_size, max_samples=eval_max_samples,
    )
    print(f"  => Forget PPL: {forget_ppl:.2f}")

    print(f"\nRetain PPL ({len(retain_records)} samples, eval {eval_max_samples})...")
    retain_ppl = compute_ppl(
        model, tokenizer, retain_records, max_seq_length,
        batch_size=eval_batch_size, max_samples=eval_max_samples,
    )
    print(f"  => Retain PPL: {retain_ppl:.2f}")

    eval_results = {
        "forget_ppl": round(forget_ppl, 2),
        "retain_ppl": round(retain_ppl, 2),
        "forget_samples_evaluated": min(len(forget_records), eval_max_samples),
        "retain_samples_evaluated": min(len(retain_records), eval_max_samples),
    }
    eval_path = output_dir / "eval_results.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved eval results: {eval_path}")

    print(f"\n{'=' * 60}")
    print(f"RETRAIN ORACLE COMPLETE")
    print(f"  Author:         {args.author}")
    print(f"  Forget type:    {args.forget_type}")
    print(f"  Forget PPL:     {forget_ppl:.2f}")
    print(f"  Retain PPL:     {retain_ppl:.2f}")
    print(f"  Training time:  {training_time:.1f}s ({training_time / 3600:.2f}h)")
    print(f"  Output:         {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
