#!/usr/bin/env python3
"""Phase 2: SFT fine-tuning script for Qwen3-1.7B on ChatML QA data.

Full fine-tuning in bf16, no quantization. All hyperparameters read from config.yaml.

Input format: JSONL with {"messages": [{"role": ..., "content": ...}, ...]}.
The tokenizer's chat_template is used to render each sample.

Usage:
    python benchmarks/scripts/pipeline_qa/train_sft.py
    python benchmarks/scripts/pipeline_qa/train_sft.py --config path/to/config.yaml
"""

import argparse
import json
import random
import sys
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


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def load_jsonl(data_file: str) -> list[dict]:
    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples from {data_file}")
    return samples


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
        labels = [lid if mask == 1 else -100 for lid, mask in zip(input_ids, attention_mask)]
        processed.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })
    if skipped:
        print(f"  Skipped {skipped} samples without 'messages' key")

    dataset = Dataset.from_list(processed)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


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
        save_steps=sft["save_steps"],
        save_total_limit=3,
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
            f"{trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps}"
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
                f"{entry['train_runtime']:.1f}s ({entry['train_runtime'] / 3600:.2f}h)"
            )
            print(
                f"  Training throughput: "
                f"{entry.get('train_samples_per_second', 'N/A')} samples/s"
            )
            break

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 SFT: Full fine-tuning for Qwen3-1.7B"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmarks/scripts/pipeline_qa/config.yaml",
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
    output_dir = project_root / cfg["sft"]["output_dir"]
    log_dir = output_dir / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    print(f"Model path:    {model_path}")
    print(f"Data file:     {data_file}")
    print(f"Output dir:    {output_dir}")
    print(f"Log dir:       {log_dir}")

    max_seq_length = cfg["model"]["max_seq_length"]
    torch_dtype = getattr(torch, cfg["model"].get("compute_dtype", "bfloat16"))

    print(f"\nLoading model (Full FT, {cfg['model'].get('compute_dtype', 'bfloat16')})...")
    model, tokenizer = load_model_and_tokenizer(
        str(model_path), max_seq_length, torch_dtype
    )

    print("\nLoading dataset...")
    samples = load_jsonl(str(data_file))

    print("Tokenizing and building masked labels...")
    train_dataset = preprocess_dataset(samples, tokenizer, max_seq_length)

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
    print(f"\nStarting SFT training...")
    print(f"  Epochs:          {sft['num_train_epochs']}")
    print(f"  Batch size:      {sft['per_device_train_batch_size']}")
    print(f"  Grad accum:      {sft['gradient_accumulation_steps']}")
    print(f"  Effective batch: {effective_batch}")
    print(f"  Learning rate:   {sft['learning_rate']}")
    print(f"  Scheduler:       {sft['lr_scheduler_type']}")
    print(f"  Max seq length:  {max_seq_length}")
    print(f"  Dataset size:    {len(train_dataset)}")
    print(f"  Total steps:     ~{total_steps}")
    print()

    last_ckpt = None
    if Path(output_dir).is_dir():
        from transformers.trainer_utils import get_last_checkpoint
        last_ckpt = get_last_checkpoint(str(output_dir))
    if last_ckpt:
        print(f"  Resuming from {last_ckpt}")

    train_result = trainer.train(resume_from_checkpoint=last_ckpt)

    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print_training_summary(trainer)

    print(f"\nDone. Model saved to: {output_dir}")
    print(f"Logs available at: {log_dir}")


if __name__ == "__main__":
    main()
