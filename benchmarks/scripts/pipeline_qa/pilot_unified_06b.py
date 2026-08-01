#!/usr/bin/env python3
"""Config-driven unified pilot: SFT + Retrain (oracle) + NPO.

Runs 3 experiment types across multiple (author, forget_set_type) combos:
  1. SFT on all data (shared, runs once)
  2. Per-combo: Retrain SFT on retain-only data (oracle baseline)
  3. Per-combo: NPO on SFT checkpoint (unlearning)

Each step runs in a separate subprocess for GPU isolation.
All models are evaluated on the same forget_eval and retain_eval samples per combo.

Usage:
    cd /home/hxue/Projects/originblame
    python benchmarks/scripts/pipeline_qa/pilot_unified_06b.py --config config_06b.yaml
    # Or run a single step:
    python benchmarks/scripts/pipeline_qa/pilot_unified_06b.py --config config_06b.yaml --step sft
    python benchmarks/scripts/pipeline_qa/pilot_unified_06b.py --config config_06b.yaml --step retrain --author InternetArchiveBot --forget-type line
    python benchmarks/scripts/pipeline_qa/pilot_unified_06b.py --config config_06b.yaml --step npo --author InternetArchiveBot --forget-type line
    python benchmarks/scripts/pipeline_qa/pilot_unified_06b.py --config config_06b.yaml --step compare
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import yaml


# ── Config helpers ──────────────────────────────────────────────────────


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_pilot_dir(cfg):
    output_dir = cfg.get("output_dir", "benchmarks/results/pipeline_qa/pilot")
    return Path(output_dir)


def get_retain_size(cfg):
    return cfg.get("data", {}).get("retain_size", None)


def get_eval_sample_size(cfg):
    return cfg.get("eval", {}).get("eval_sample_size", 50)


# ── Split construction ──────────────────────────────────────────────────

def build_combo_splits(cfg, author, forget_type):
    data_file = Path(cfg["data"]["data_file"])
    forget_sets_file = cfg["data"].get(
        "forget_sets_file",
        str(Path(cfg["data"]["ob_data_dir"]).parent / "results" / "forget_sets.json"),
    )
    seed = cfg["seed"]
    retain_size = get_retain_size(cfg)
    eval_sample_size = get_eval_sample_size(cfg)
    forget_max_size = cfg.get("data", {}).get("forget_max_size", None)

    all_data_ordered = []
    with open(data_file) as f:
        for line in f:
            all_data_ordered.append(json.loads(line))
    total = len(all_data_ordered)

    with open(forget_sets_file) as f:
        forget_sets = json.load(f)

    forget_indices = sorted(forget_sets[author][forget_type]["indices"])
    if forget_max_size and len(forget_indices) > forget_max_size:
        random.seed(seed)
        forget_indices = sorted(random.sample(forget_indices, forget_max_size))
    forget_indices = set(forget_indices)

    forget_subset = [
        all_data_ordered[i]
        for i in sorted(forget_indices)
        if i < total
    ]

    retain_indices = [i for i in range(total) if i not in forget_indices]
    random.seed(seed)
    random.shuffle(retain_indices)
    retain_subset = [all_data_ordered[i] for i in retain_indices[:retain_size]] if retain_size else [all_data_ordered[i] for i in retain_indices]

    random.seed(seed)
    forget_eval = random.sample(forget_subset, min(eval_sample_size, len(forget_subset)))
    random.seed(seed)
    retain_eval = random.sample(retain_subset, min(eval_sample_size, len(retain_subset)))

    pilot_dir = get_pilot_dir(cfg)
    combo_dir = pilot_dir / "combos" / author / forget_type
    combo_dir.mkdir(parents=True, exist_ok=True)

    splits_path = combo_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump({
            "author": author,
            "forget_type": forget_type,
            "forget_subset": forget_subset,
            "retain_subset": retain_subset,
            "forget_eval": forget_eval,
            "retain_eval": retain_eval,
        }, f)

    print(f"  [{author}/{forget_type}] Forget: {len(forget_subset)}, Retain: {len(retain_subset)}, "
          f"Forget eval: {len(forget_eval)}, Retain eval: {len(retain_eval)}")
    return splits_path


def load_splits(combo_dir):
    """Load splits from combo directory."""
    with open(combo_dir / "splits.json") as f:
        d = json.load(f)
    return d["forget_subset"], d["retain_subset"], d["forget_eval"], d["retain_eval"]


# ── Step functions (each runs in a clean subprocess) ────────────────────


def run_sft(config_path):
    """SFT on ALL data (shared across combos). Runs once."""
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

    cfg = load_config(config_path)
    seed = cfg["seed"]
    pilot_dir = get_pilot_dir(cfg)
    max_seq = cfg["model"]["max_seq_length"]
    model_path = cfg["model"]["base_path"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_file = Path(cfg["data"]["data_file"])
    all_data = []
    with open(data_file) as f:
        for line in f:
            all_data.append(json.loads(line))

    random.seed(seed)
    random.shuffle(all_data)

    print(f"\n{'='*60}")
    print(f"STEP: SFT on ALL data ({len(all_data)} samples)")
    print(f"{'='*60}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    sft_dir = pilot_dir / "sft"

    def tokenize_fn(examples):
        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                 for m in examples["messages"]]
        tok = tokenizer(texts, truncation=True, max_length=max_seq, padding=False)
        tok["labels"] = [x[:] for x in tok["input_ids"]]
        return tok

    train_ds = Dataset.from_dict({"messages": [d["messages"] for d in all_data]})
    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["messages"])

    sft_cfg = cfg["sft"]
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(sft_dir),
            num_train_epochs=sft_cfg["num_train_epochs"],
            per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
            learning_rate=sft_cfg["learning_rate"],
            warmup_ratio=sft_cfg["warmup_ratio"],
            weight_decay=sft_cfg["weight_decay"],
            lr_scheduler_type=sft_cfg["lr_scheduler_type"],
            bf16=sft_cfg["bf16"],
            logging_steps=sft_cfg["logging_steps"],
            save_steps=sft_cfg["save_steps"],
            eval_strategy="no",
            report_to="none",
            seed=seed,
            dataloader_num_workers=0,
        ),
        train_dataset=train_tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    t0 = time.time()
    trainer.train()
    sft_time = time.time() - t0
    print(f"  SFT done in {sft_time:.0f}s ({sft_time/60:.1f}min)")

    sft_final = sft_dir / "final"
    trainer.save_model(str(sft_final))
    tokenizer.save_pretrained(str(sft_final))

    model.eval()

    def compute_ppl(data, tag):
        total_nll, n_tokens = 0.0, 0
        with torch.no_grad():
            for item in data:
                text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq).to(model.device)
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_nll += outputs.loss.item() * inputs["input_ids"].shape[1]
                n_tokens += inputs["input_ids"].shape[1]
        ppl = math.exp(total_nll / n_tokens) if n_tokens > 0 else float("inf")
        print(f"  {tag} PPL: {ppl:.2f}")
        return ppl

    # Evaluate on all combos
    results = {}
    for author_info in cfg["authors"]:
        author = author_info["name"]
        for ft in cfg["forget_set_types"]:
            combo_dir = pilot_dir / "combos" / author / ft
            if not (combo_dir / "splits.json").exists():
                continue
            _, _, forget_eval, retain_eval = load_splits(combo_dir)
            sft_forget_ppl = compute_ppl(forget_eval, f"SFT forget [{author}/{ft}]")
            sft_retain_ppl = compute_ppl(retain_eval, f"SFT retain [{author}/{ft}]")
            results[f"{author}/{ft}"] = {
                "author": author,
                "forget_type": ft,
                "training_samples": len(all_data),
                "forget_ppl": sft_forget_ppl,
                "retain_ppl": sft_retain_ppl,
                "training_time_s": sft_time,
                "max_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            }

    with open(pilot_dir / "sft_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved SFT results")


def run_retrain(config_path, author, forget_type):
    """Retrain SFT on retain-only data (oracle baseline) for one combo."""
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

    cfg = load_config(config_path)
    seed = cfg["seed"]
    pilot_dir = get_pilot_dir(cfg)
    max_seq = cfg["model"]["max_seq_length"]
    model_path = cfg["model"]["base_path"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    combo_dir = pilot_dir / "combos" / author / forget_type
    if not (combo_dir / "splits.json").exists():
        build_combo_splits(cfg, author, forget_type)
    forget_subset, retain_subset, forget_eval, retain_eval = load_splits(combo_dir)

    print(f"\n{'='*60}")
    print(f"STEP: Retrain [{author}/{forget_type}] on RETAIN ONLY ({len(retain_subset)} samples)")
    print(f"{'='*60}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    )

    retrain_train_data = retain_subset[:]
    random.seed(seed)
    random.shuffle(retrain_train_data)
    print(f"  Training samples: {len(retrain_train_data)} (retain ONLY)")

    retrain_dir = combo_dir / "retrain"

    def tokenize_fn(examples):
        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                 for m in examples["messages"]]
        tok = tokenizer(texts, truncation=True, max_length=max_seq, padding=False)
        tok["labels"] = [x[:] for x in tok["input_ids"]]
        return tok

    train_ds = Dataset.from_dict({"messages": [d["messages"] for d in retrain_train_data]})
    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["messages"])

    sft_cfg = cfg["sft"]
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(retrain_dir),
            num_train_epochs=sft_cfg["num_train_epochs"],
            per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
            learning_rate=sft_cfg["learning_rate"],
            warmup_ratio=sft_cfg["warmup_ratio"],
            weight_decay=sft_cfg["weight_decay"],
            lr_scheduler_type=sft_cfg["lr_scheduler_type"],
            bf16=sft_cfg["bf16"],
            logging_steps=sft_cfg["logging_steps"],
            save_steps=sft_cfg["save_steps"],
            eval_strategy="no",
            report_to="none",
            seed=seed,
            dataloader_num_workers=0,
        ),
        train_dataset=train_tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    t0 = time.time()
    trainer.train()
    retrain_time = time.time() - t0
    print(f"  Retrain done in {retrain_time:.0f}s ({retrain_time/60:.1f}min)")

    retrain_final = retrain_dir / "final"
    trainer.save_model(str(retrain_final))
    tokenizer.save_pretrained(str(retrain_final))

    model.eval()

    def compute_ppl(data, tag):
        total_nll, n_tokens = 0.0, 0
        with torch.no_grad():
            for item in data:
                text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq).to(model.device)
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_nll += outputs.loss.item() * inputs["input_ids"].shape[1]
                n_tokens += inputs["input_ids"].shape[1]
        ppl = math.exp(total_nll / n_tokens) if n_tokens > 0 else float("inf")
        print(f"  {tag} PPL: {ppl:.2f}")
        return ppl

    retrain_forget_ppl = compute_ppl(forget_eval, "Retrain forget")
    retrain_retain_ppl = compute_ppl(retain_eval, "Retrain retain")

    results = {
        "author": author,
        "forget_type": forget_type,
        "training_samples": len(retrain_train_data),
        "forget_ppl": retrain_forget_ppl,
        "retain_ppl": retrain_retain_ppl,
        "training_time_s": retrain_time,
        "max_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    with open(combo_dir / "retrain_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved Retrain results to {combo_dir / 'retrain_results.json'}")


def run_npo(config_path, author, forget_type):
    """NPO on SFT checkpoint for one combo."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config(config_path)
    seed = cfg["seed"]
    pilot_dir = get_pilot_dir(cfg)
    max_seq = cfg["model"]["max_seq_length"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    combo_dir = pilot_dir / "combos" / author / forget_type
    if not (combo_dir / "splits.json").exists():
        build_combo_splits(cfg, author, forget_type)
    forget_subset, retain_subset, forget_eval, retain_eval = load_splits(combo_dir)
    sft_checkpoint = str(pilot_dir / "sft" / "final")

    npo_cfg = cfg["npo"]
    npo_beta = npo_cfg["beta"]
    npo_lr = npo_cfg["learning_rate"]
    npo_num_epochs = npo_cfg["num_train_epochs"]
    npo_batch_size = npo_cfg["per_device_train_batch_size"]
    npo_grad_accum = npo_cfg["gradient_accumulation_steps"]
    npo_warmup_ratio = npo_cfg["warmup_ratio"]
    npo_weight_decay = npo_cfg["weight_decay"]
    npo_max_grad_norm = npo_cfg["max_grad_norm"]
    npo_lr_scheduler = npo_cfg["lr_scheduler_type"]
    npo_retain_weight = npo_cfg.get("retain_weight", 1.0)

    print(f"\n{'='*60}")
    print(f"STEP: NPO [{author}/{forget_type}] on SFT checkpoint")
    print(f"{'='*60}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Forget: {len(forget_subset)} | Retain: {len(retain_subset)}")

    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    npo_model = AutoModelForCausalLM.from_pretrained(
        sft_checkpoint, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    npo_model.config.use_cache = False
    npo_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    npo_model.lm_head.weight = torch.nn.Parameter(npo_model.lm_head.weight.clone())
    print(f"  VRAM after trainable: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    ref_model = AutoModelForCausalLM.from_pretrained(
        sft_checkpoint, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print(f"  VRAM after ref: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, records, tok, max_len):
            self.texts = []
            for r in records:
                if "messages" in r:
                    self.texts.append(tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False))
                else:
                    self.texts.append(r.get("text", ""))
            self.tok = tok
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tok(self.texts[idx], truncation=True, max_length=self.max_len,
                           padding="max_length", return_tensors="pt")
            ids = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)
            labels = ids.clone()
            labels[mask == 0] = -100
            return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    forget_loader = DataLoader(TextDataset(forget_subset, tokenizer, max_seq),
                               batch_size=npo_batch_size, shuffle=True, drop_last=True)
    retain_loader = DataLoader(TextDataset(retain_subset, tokenizer, max_seq),
                               batch_size=npo_batch_size, shuffle=True, drop_last=True)

    trainable_params = [p for p in npo_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=npo_lr, weight_decay=npo_weight_decay)

    total_steps = len(forget_loader) // npo_grad_accum * npo_num_epochs
    warmup_steps = int(total_steps * npo_warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        if npo_lr_scheduler == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"  Steps: {total_steps}, warmup: {warmup_steps}")

    def compute_log_probs(model, input_ids, attention_mask, no_grad=False):
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = outputs.logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:]
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_lp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            return (token_lp * shift_mask).sum(dim=1)

    def compute_npo_loss(model, ref, ids, mask, beta):
        log_pi = compute_log_probs(model, ids, mask, no_grad=False)
        log_ref = compute_log_probs(ref, ids, mask, no_grad=True)
        return (-F.logsigmoid(-beta * (log_pi - log_ref)) * 2.0 / beta).mean()

    def compute_nll_loss(model, ids, mask, labels):
        logits = model(input_ids=ids, attention_mask=mask).logits
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fn(logits[:, :-1].contiguous().view(-1, logits.size(-1)), labels[:, 1:].contiguous().view(-1))

    device = next(npo_model.parameters()).device
    global_step = 0
    running_npo, running_retain, running_total = 0.0, 0.0, 0.0
    log_interval = 10
    t0 = time.time()

    for epoch in range(npo_num_epochs):
        npo_model.train()
        epoch_npo, epoch_retain, n_batches = 0.0, 0.0, 0
        retain_iter = iter(retain_loader)

        for step, f_batch in enumerate(forget_loader):
            f_ids = f_batch["input_ids"].to(device)
            f_mask = f_batch["attention_mask"].to(device)
            try:
                r_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                r_batch = next(retain_iter)
            r_ids = r_batch["input_ids"].to(device)
            r_mask = r_batch["attention_mask"].to(device)
            r_labels = r_batch["labels"].to(device)

            npo_loss = compute_npo_loss(npo_model, ref_model, f_ids, f_mask, npo_beta)
            retain_loss = compute_nll_loss(npo_model, r_ids, r_mask, r_labels)
            total_loss = (npo_loss + npo_retain_weight * retain_loss) / npo_grad_accum
            total_loss.backward()

            if (step + 1) % npo_grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, npo_max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                running_npo += npo_loss.item() * npo_grad_accum
                running_retain += retain_loss.item() * npo_grad_accum * npo_retain_weight
                running_total += total_loss.item() * npo_grad_accum

                if global_step % log_interval == 0:
                    print(f"[Epoch {epoch+1}/{npo_num_epochs} | Step {global_step}] "
                          f"npo={running_npo/log_interval:.4f} retain={running_retain/log_interval:.4f} "
                          f"total={running_total/log_interval:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
                    running_npo, running_retain, running_total = 0.0, 0.0, 0.0

            epoch_npo += npo_loss.item()
            epoch_retain += retain_loss.item()
            n_batches += 1

        print(f"\n=== Epoch {epoch+1}/{npo_num_epochs} === "
              f"npo={epoch_npo/max(1,n_batches):.4f} retain={epoch_retain/max(1,n_batches):.4f}\n")

    npo_time = time.time() - t0
    print(f"  NPO done in {npo_time:.0f}s ({npo_time/60:.1f}min)")

    npo_final = combo_dir / "npo" / "final"
    npo_model.save_pretrained(str(npo_final), safe_serialization=True)
    tokenizer.save_pretrained(str(npo_final))

    npo_model.eval()

    def compute_ppl(data, tag):
        total_nll, n_tokens = 0.0, 0
        with torch.no_grad():
            for item in data:
                text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq).to(npo_model.device)
                outputs = npo_model(**inputs, labels=inputs["input_ids"])
                total_nll += outputs.loss.item() * inputs["input_ids"].shape[1]
                n_tokens += inputs["input_ids"].shape[1]
        ppl = math.exp(total_nll / n_tokens) if n_tokens > 0 else float("inf")
        print(f"  {tag} PPL: {ppl:.2f}")
        return ppl

    npo_forget_ppl = compute_ppl(forget_eval, "NPO forget")
    npo_retain_ppl = compute_ppl(retain_eval, "NPO retain")

    results = {
        "author": author,
        "forget_type": forget_type,
        "forget_ppl": npo_forget_ppl,
        "retain_ppl": npo_retain_ppl,
        "training_time_s": npo_time,
        "max_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    with open(combo_dir / "npo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved NPO results to {combo_dir / 'npo_results.json'}")


# ── Comparison table ────────────────────────────────────────────────────


def print_comparison(config_path):
    cfg = load_config(config_path)
    pilot_dir = get_pilot_dir(cfg)
    sft_results_path = pilot_dir / "sft_results.json"
    if not sft_results_path.exists():
        print("ERROR: sft_results.json not found. Run SFT first.")
        return

    with open(sft_results_path) as f:
        sft_results = json.load(f)

    all_combos = []
    combos_dir = pilot_dir / "combos"
    if not combos_dir.exists():
        print("ERROR: No combos directory found.")
        return

    for author_dir in sorted(combos_dir.iterdir()):
        if not author_dir.is_dir():
            continue
        author = author_dir.name
        for ft_dir in sorted(author_dir.iterdir()):
            if not ft_dir.is_dir():
                continue
            ft = ft_dir.name

            retrain_path = ft_dir / "retrain_results.json"
            npo_path = ft_dir / "npo_results.json"

            retrain = None
            npo = None
            if retrain_path.exists():
                with open(retrain_path) as f:
                    retrain = json.load(f)
            if npo_path.exists():
                with open(npo_path) as f:
                    npo = json.load(f)

            all_combos.append({
                "author": author,
                "forget_type": ft,
                "retrain": retrain,
                "npo": npo,
            })

    if not all_combos:
        print("ERROR: No combo results found.")
        return

    print("\n" + "=" * 100)
    print("=== Config-Driven Pilot: SFT vs Retrain vs NPO ===")
    print(f"{'Author':<22} {'Type':<16} {'Method':<18} "
          f"{'Forget PPL':>12} {'Retain PPL':>12} {'Forget×':>10} {'Retain×':>10}")
    print("-" * 100)

    unified = {"model": "Qwen3-0.6B", "combos": {}}

    for combo in all_combos:
        author = combo["author"]
        ft = combo["forget_type"]
        key = f"{author}/{ft}"

        sft_data = sft_results.get(key)
        if sft_data is None:
            print(f"  WARNING: No SFT results for {key}, skipping comparison.")
            continue

        sft_fp, sft_rp = sft_data["forget_ppl"], sft_data["retain_ppl"]

        print(f"{author:<22} {ft:<16} {'SFT':<18} "
              f"{sft_fp:>12.2f} {sft_rp:>12.2f} {'1.00x':>10} {'1.00x':>10}")

        combo_result = {"sft": sft_data}

        rt_fp = rt_rp = None
        if combo["retrain"]:
            rt_fp = combo["retrain"]["forget_ppl"]
            rt_rp = combo["retrain"]["retain_ppl"]
            rt_fr = rt_fp / sft_fp if sft_fp > 0 else float("inf")
            rt_rr = rt_rp / sft_rp if sft_rp > 0 else float("inf")
            print(f"{'':<22} {'':<16} {'Retrain':<18} "
                  f"{rt_fp:>12.2f} {rt_rp:>12.2f} {rt_fr:>9.2f}x {rt_rr:>9.2f}x")
            combo_result["retrain"] = combo["retrain"]
            combo_result["retrain"]["forget_ratio"] = rt_fr
            combo_result["retrain"]["retain_ratio"] = rt_rr

        if combo["npo"]:
            np_fp = combo["npo"]["forget_ppl"]
            np_rp = combo["npo"]["retain_ppl"]
            np_fr = np_fp / sft_fp if sft_fp > 0 else float("inf")
            np_rr = np_rp / sft_rp if sft_rp > 0 else float("inf")
            print(f"{'':<22} {'':<16} {'NPO':<18} "
                  f"{np_fp:>12.2f} {np_rp:>12.2f} {np_fr:>9.2f}x {np_rr:>9.2f}x")
            combo_result["npo"] = combo["npo"]
            combo_result["npo"]["forget_ratio"] = np_fr
            combo_result["npo"]["retain_ratio"] = np_rr
            combo_result["npo"]["selectivity"] = np_fr / np_rr if np_rr > 0 else float("inf")

            if rt_fp is not None and rt_rp is not None:
                sft_gap = rt_fp - sft_fp
                npo_gap = np_fp - sft_fp
                npo_closeness = npo_gap / sft_gap if sft_gap > 0 else float("inf")
                npo_retain_vs_retrain = np_rp / rt_rp if rt_rp > 0 else float("inf")
                combo_result["comparison"] = {
                    "npo_vs_retrain_forgetting": npo_closeness,
                    "npo_retain_vs_retrain": npo_retain_vs_retrain,
                }
                print(f"{'':*<22} {'':<16} {'':<18} "
                      f"  NPO achieves {npo_closeness*100:.1f}% of retrain forgetting, "
                      f"retain {npo_retain_vs_retrain:.2f}x retrain")

        unified["combos"][key] = combo_result
        print()

    print("=" * 100)

    results_path = pilot_dir / "unified_results.json"
    with open(results_path, "w") as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")


# ── Main orchestrator ───────────────────────────────────────────────────


def get_combos(cfg, author_filter=None, forget_type_filter=None):
    """Return list of (author, forget_type) tuples from config, with optional filters."""
    combos = []
    for author_info in cfg["authors"]:
        author = author_info["name"]
        if author_filter and author != author_filter:
            continue
        for ft in cfg["forget_set_types"]:
            if forget_type_filter and ft != forget_type_filter:
                continue
            combos.append((author, ft))
    return combos


def main():
    parser = argparse.ArgumentParser(description="Config-driven pilot: SFT + Retrain + NPO")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file (e.g. config_06b.yaml)")
    parser.add_argument("--step", choices=["all", "sft", "retrain", "npo", "compare"],
                        default="all",
                        help="Which step to run (default: all)")
    parser.add_argument("--author", type=str, default=None,
                        help="Filter to specific author (for retrain/npo steps)")
    parser.add_argument("--forget-type", type=str, default=None,
                        help="Filter to specific forget set type (for retrain/npo steps)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pilot_dir = get_pilot_dir(cfg)
    pilot_dir.mkdir(parents=True, exist_ok=True)

    script = os.path.abspath(__file__)
    venv_python = str(Path("benchmarks/.venv/bin/python").resolve())

    if args.step == "all":
        print("=" * 70)
        print("CONFIG-DRIVEN UNIFIED PILOT: SFT + Retrain (oracle) + NPO")
        combos = get_combos(cfg)
        print(f"  Config: {args.config}")
        print(f"  Combos: {len(combos)} ({', '.join(f'{a}/{t}' for a,t in combos)})")
        print(f"  Each step runs in a clean subprocess for GPU isolation")
        print("=" * 70)

        t_total = time.time()

        # Step 0: Build splits for all combos
        print("\n[0] Building splits for all combos...")
        for author, ft in combos:
            combo_dir = pilot_dir / "combos" / author / ft
            splits_path = combo_dir / "splits.json"
            if splits_path.exists() and not args.force:
                print(f"  [{author}/{ft}] splits.json exists, skipping (use --force to rebuild)")
            else:
                build_combo_splits(cfg, author, ft)

        # Step 1: SFT (shared, runs once)
        if (pilot_dir / "sft_results.json").exists() and not args.force:
            print("\n[1] SFT results exist, skipping (use --force to rerun)")
        else:
            print("\n[1] Running SFT (subprocess)...")
            subprocess.run(
                [venv_python, script, "--config", args.config, "--step", "sft"],
                cwd=os.getcwd(), check=True,
            )

        # Step 2: Retrain per combo
        for i, (author, ft) in enumerate(combos):
            combo_dir = pilot_dir / "combos" / author / ft
            if (combo_dir / "retrain_results.json").exists() and not args.force:
                print(f"\n[2.{i+1}] Retrain [{author}/{ft}] exists, skipping")
            else:
                print(f"\n[2.{i+1}] Running Retrain [{author}/{ft}] (subprocess)...")
                subprocess.run(
                    [venv_python, script, "--config", args.config,
                     "--step", "retrain", "--author", author, "--forget-type", ft],
                    cwd=os.getcwd(), check=True,
                )

        # Step 3: NPO per combo
        for i, (author, ft) in enumerate(combos):
            combo_dir = pilot_dir / "combos" / author / ft
            if (combo_dir / "npo_results.json").exists() and not args.force:
                print(f"\n[3.{i+1}] NPO [{author}/{ft}] exists, skipping")
            else:
                print(f"\n[3.{i+1}] Running NPO [{author}/{ft}] (subprocess)...")
                subprocess.run(
                    [venv_python, script, "--config", args.config,
                     "--step", "npo", "--author", author, "--forget-type", ft],
                    cwd=os.getcwd(), check=True,
                )

        # Step 4: Comparison
        print("\n[4] Generating comparison table...")
        print_comparison(args.config)

        total_time = time.time() - t_total
        print(f"\nTotal pipeline time: {total_time:.0f}s ({total_time/60:.1f}min)")

    elif args.step == "sft":
        run_sft(args.config)

    elif args.step == "retrain":
        if not args.author or not args.forget_type:
            print("ERROR: --author and --forget-type required for retrain step")
            sys.exit(1)
        run_retrain(args.config, args.author, args.forget_type)

    elif args.step == "npo":
        if not args.author or not args.forget_type:
            print("ERROR: --author and --forget-type required for npo step")
            sys.exit(1)
        run_npo(args.config, args.author, args.forget_type)

    elif args.step == "compare":
        print_comparison(args.config)


if __name__ == "__main__":
    main()
