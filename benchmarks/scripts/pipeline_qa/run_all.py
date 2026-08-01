#!/usr/bin/env python3
"""Orchestrate the full MU experiment pipeline.

Execution order (when --phase all):
  Phase 2: build_forget_sets.py   — extract forget/retain splits from ob provenance
  Phase 1: train_sft.py           — Full FT fine-tuning on full dataset
  Phase 3: train_npo/rmu — 18 unlearning runs (2 algos × 3 authors × 3 forget types)
  Phase 4: evaluate.py --eval-all — compute PPL + ROUGE for every checkpoint

Resume: if a checkpoint already exists for a given (algorithm, author, forget_type),
        that run is skipped.

Usage:
    PYTHONPATH=src python3 benchmarks/scripts/pipeline_qa/run_all.py
    PYTHONPATH=src python3 benchmarks/scripts/pipeline_qa/run_all.py --config benchmarks/scripts/pipeline_qa/config.yaml
    PYTHONPATH=src python3 benchmarks/scripts/pipeline_qa/run_all.py --phase 3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from threading import Thread

import yaml

# ── helpers ───────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_subprocess(
    cmd: list[str],
    log_file: Path | None = None,
    env: dict | None = None,
) -> int:
    """Run *cmd* as a subprocess, stream stdout to console and log file in real time."""
    print(f"\n  $ {' '.join(cmd)}", flush=True)
    merged_env = os.environ.copy()
    # Preserve existing PYTHONPATH and add project roots for ob package
    _existing_pp = merged_env.get("PYTHONPATH", "")
    _project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
    _ob_parts = [_project_root]
    # Detect rust-originblame python src from OB_RUST_BIN or sibling directory
    _ob_rust_bin = merged_env.get("OB_RUST_BIN", "")
    if _ob_rust_bin:
        _ob_parts.append(str(Path(_ob_rust_bin).parent.parent / "python" / "src"))
    else:
        _sibling = Path(_project_root).parent / "rust-originblame" / "python" / "src"
        if _sibling.is_dir():
            _ob_parts.append(str(_sibling))
    merged_env["PYTHONPATH"] = ":".join(_ob_parts)
    merged_env["PYTHONUNBUFFERED"] = "1"
    if env:
        merged_env.update(env)

    log_fh = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_file, "a", encoding="utf-8")
        log_fh.write(f"\n{'=' * 60}\n")
        log_fh.write(f"Command: {' '.join(cmd)}\n")
        log_fh.write(f"{'=' * 60}\n")
        log_fh.flush()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=merged_env,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if log_fh is not None:
            log_fh.write(line)
            log_fh.flush()

    proc.wait()

    if log_fh is not None:
        log_fh.write(f"\nExit code: {proc.returncode}\n")
        log_fh.close()

    return proc.returncode


# ── checkpoint existence (resume support) ─────────────────────────────────────


def sft_checkpoint_exists(cfg: dict) -> bool:
    sft_dir = Path(cfg["sft"]["output_dir"])
    return sft_dir.is_dir() and (sft_dir / "config.json").exists()


def unlearn_checkpoint_exists(
    cfg: dict, algorithm: str, author: str, forget_type: str
) -> bool:
    algo_dir_map = {
        "npo": cfg["npo"]["output_dir"],
        "rmu": cfg["rmu"]["output_dir"],
    }
    if algorithm not in algo_dir_map:
        return False
    ckpt = Path(algo_dir_map[algorithm]) / f"{author}_{forget_type}" / "final"
    return ckpt.is_dir() and (ckpt / "config.json").exists()


def retrain_checkpoint_exists(cfg: dict, author: str, forget_type: str) -> bool:
    retrain_cfg = cfg.get("retrain", {})
    if "output_dir" in retrain_cfg:
        base = retrain_cfg["output_dir"]
    else:
        base = cfg["sft"]["output_dir"].replace("sft", "retrain")
    ckpt = Path(base) / f"{author}_{forget_type}" / "final"
    return ckpt.is_dir() and (ckpt / "config.json").exists()


# ── forget-set flattening ─────────────────────────────────────────────────────


def copy_forget_sets(nested_path: Path, dest_path: Path) -> None:
    """Copy forget_sets.json to the data directory for train scripts."""
    import shutil

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(nested_path, dest_path)
    print(f"  Forget sets copied to {dest_path}")


# ── phase runners ─────────────────────────────────────────────────────────────


def phase_build_forget_sets(
    config_path: str, log_dir: Path, extra_args: list[str] | None = None
) -> int:
    print("\n" + "=" * 60)
    print("PHASE 2: Build Forget Sets")
    print("=" * 60)

    cmd = [
        sys.executable,
        "benchmarks/scripts/pipeline_qa/build_forget_sets.py",
        "--config",
        config_path,
    ]
    if extra_args:
        cmd.extend(extra_args)
    rc = run_subprocess(cmd, log_dir / "build_forget_sets.log")

    if rc == 0:
        cfg = load_config(config_path)
        project_root = Path(config_path).resolve().parent.parent.parent.parent
        data_dir = Path(cfg["data"]["data_file"]).resolve().parent
        nested = data_dir / "forget_sets.json"

        if nested.exists():
            print(f"  Forget sets ready at {nested}")
        else:
            print(f"  WARNING: forget_sets.json not found at {nested}")

    return rc


def phase_train_sft(config_path: str, cfg: dict, log_dir: Path) -> int:
    print("\n" + "=" * 60)
    print("PHASE 1: SFT Fine-tuning")
    print("=" * 60)

    if sft_checkpoint_exists(cfg):
        print("  SFT checkpoint already exists — skipping.")
        return 0

    cmd = [sys.executable, "benchmarks/scripts/pipeline_qa/train_sft.py", "--config", config_path]
    return run_subprocess(cmd, log_dir / "train_sft.log")


def _make_env() -> dict[str, str]:
    merged_env = os.environ.copy()
    _existing_pp = merged_env.get("PYTHONPATH", "")
    _project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
    _ob_parts = [_project_root]
    _ob_rust_bin = merged_env.get("OB_RUST_BIN", "")
    if _ob_rust_bin:
        _ob_parts.append(str(Path(_ob_rust_bin).parent.parent / "python" / "src"))
    else:
        _sibling = Path(_project_root).parent / "rust-originblame" / "python" / "src"
        if _sibling.is_dir():
            _ob_parts.append(str(_sibling))
    merged_env["PYTHONPATH"] = ":".join(_ob_parts)
    merged_env["PYTHONUNBUFFERED"] = "1"
    return merged_env


def _launch_process(
    cmd: list[str], log_file: Path, env: dict[str, str]
) -> subprocess.Popen[str]:
    """Launch subprocess with stdout teed to console and log file (non-blocking)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_file, "a", encoding="utf-8")
    log_fh.write(f"\n{'=' * 60}\n")
    log_fh.write(f"Command: {' '.join(cmd)}\n")
    log_fh.write(f"{'=' * 60}\n")
    log_fh.flush()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    def _tee() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_fh.write(line)
            log_fh.flush()
        log_fh.write(f"\nExit code: {proc.returncode}\n")
        log_fh.close()

    t = Thread(target=_tee, daemon=True)
    t.start()
    proc._log_thread = t  # type: ignore[attr-defined]
    return proc


def phase_unlearn(
    config_path: str, cfg: dict, log_dir: Path, *, max_workers: int = 1
) -> int:
    algorithms = [
        ("npo", "train_npo.py"),
        ("rmu", "train_rmu.py"),
    ]
    authors = [a["name"] for a in cfg["authors"]]
    forget_types = cfg["forget_set_types"]
    total = len(algorithms) * len(authors) * len(forget_types)

    print("\n" + "=" * 60)
    print(f"PHASE 3: Unlearning Training ({total} runs, max_workers={max_workers})")
    print("=" * 60)

    pending: list[tuple[str, str, str, str, str]] = []
    skipped = 0
    for algo, script in algorithms:
        for author in authors:
            for ft in forget_types:
                label = f"{algo}/{author}/{ft}"
                if unlearn_checkpoint_exists(cfg, algo, author, ft):
                    print(f"  SKIP {label} (checkpoint exists)")
                    skipped += 1
                    continue
                pending.append((algo, script, author, ft, label))

    if not pending:
        print(f"  All {total} runs skipped (checkpoints exist).")
        return 0

    to_run = list(pending)
    n_run = len(to_run)
    env = _make_env()

    if max_workers <= 1:
        done_count = 0
        failed = 0
        for algo, script, author, ft, label in to_run:
            done_count += 1
            print(f"\n  [{done_count}/{n_run}] {label}")
            cmd = [
                sys.executable,
                f"benchmarks/scripts/pipeline_qa/{script}",
                "--config",
                config_path,
                "--author",
                author,
                "--forget-type",
                ft,
            ]
            rc = run_subprocess(cmd, log_dir / f"train_{algo}_{author}_{ft}.log")
            if rc != 0:
                print(f"  ERROR: {label} failed (exit {rc})")
                failed += 1
        print(
            f"\nPhase 3 complete: {n_run - failed} trained, "
            f"{skipped} skipped, {failed} failed"
        )
        return 0 if failed == 0 else 1

    running: dict[subprocess.Popen[str], tuple[str, str]] = {}
    results: list[tuple[str, int]] = []
    failed = 0
    done_count = 0
    idx = 0

    def _print_progress() -> None:
        running_labels = [lbl for _, (_, lbl) in running.items()]
        running_str = ", ".join(running_labels) if running_labels else "none"
        print(
            f"  [{done_count}/{n_run} DONE] "
            f"[{len(running)} RUNNING: {running_str}] "
            f"[{n_run - done_count - len(running)} PENDING]"
        )

    while idx < len(to_run) or running:
        while len(running) < max_workers and idx < len(to_run):
            algo, script, author, ft, label = to_run[idx]
            idx += 1
            cmd = [
                sys.executable,
                f"benchmarks/scripts/pipeline_qa/{script}",
                "--config",
                config_path,
                "--author",
                author,
                "--forget-type",
                ft,
            ]
            log_file = log_dir / f"train_{algo}_{author}_{ft}.log"
            print(f"  LAUNCH {label}")
            proc = _launch_process(cmd, log_file, env)
            running[proc] = (label, algo)
            _print_progress()
            # Stagger launches so device_map="auto" sees updated VRAM
            # after the previous process finishes model loading.
            if len(running) < max_workers and idx < len(to_run):
                time.sleep(30)

        if not running:
            break

        finished = []
        for proc in list(running):
            ret = proc.poll()
            if ret is not None:
                finished.append(proc)

        if not finished:
            time.sleep(2)
            continue

        for proc in finished:
            label, algo = running.pop(proc)
            proc.wait()
            done_count += 1
            results.append((label, proc.returncode))
            status = f"exit {proc.returncode}"
            if proc.returncode != 0:
                failed += 1
                print(f"  [{done_count}/{n_run} DONE] {label} ({status}) FAILED")
            else:
                print(f"  [{done_count}/{n_run} DONE] {label} ({status})")
            _print_progress()

    print(
        f"\nPhase 3 complete: {n_run - failed} trained, "
        f"{skipped} skipped, {failed} failed"
    )
    return 0 if failed == 0 else 1


def phase_retrain(
    config_path: str, cfg: dict, log_dir: Path, *, max_workers: int = 1
) -> int:
    authors = [a["name"] for a in cfg["authors"]]
    forget_types = cfg["forget_set_types"]
    total = len(authors) * len(forget_types)

    print("\n" + "=" * 60)
    print(f"PHASE 3R: Retrain Oracle ({total} runs, max_workers={max_workers})")
    print("=" * 60)

    to_run: list[tuple[str, str, str]] = []
    skipped = 0
    for author in authors:
        for ft in forget_types:
            label = f"retrain/{author}/{ft}"
            if retrain_checkpoint_exists(cfg, author, ft):
                print(f"  SKIP {label} (checkpoint exists)")
                skipped += 1
                continue
            to_run.append((author, ft, label))

    if not to_run:
        print(f"  All {total} retrain runs skipped (checkpoints exist).")
        return 0

    n_run = len(to_run)

    if max_workers <= 1:
        done_count = 0
        failed = 0
        for author, ft, label in to_run:
            done_count += 1
            print(f"\n  [{done_count}/{n_run}] {label}")
            cmd = [
                sys.executable,
                "benchmarks/scripts/pipeline_qa/train_retrain.py",
                "--config", config_path,
                "--author", author,
                "--forget-type", ft,
            ]
            rc = run_subprocess(cmd, log_dir / f"retrain_{author}_{ft}.log")
            if rc != 0:
                print(f"  ERROR: {label} failed (exit {rc})")
                failed += 1
        print(f"\nPhase 3R complete: {n_run - failed} trained, {skipped} skipped, {failed} failed")
        return 0 if failed == 0 else 1

    env = _make_env()
    running: dict[subprocess.Popen[str], tuple[str, str]] = {}
    failed = 0
    done_count = 0
    idx = 0

    def _print_progress() -> None:
        running_labels = [lbl for _, (_, lbl) in running.items()]
        running_str = ", ".join(running_labels) if running_labels else "none"
        print(
            f"  [{done_count}/{n_run} DONE] "
            f"[{len(running)} RUNNING: {running_str}] "
            f"[{n_run - done_count - len(running)} PENDING]"
        )

    while idx < len(to_run) or running:
        while len(running) < max_workers and idx < len(to_run):
            author, ft, label = to_run[idx]
            idx += 1
            cmd = [
                sys.executable,
                "benchmarks/scripts/pipeline_qa/train_retrain.py",
                "--config", config_path,
                "--author", author,
                "--forget-type", ft,
            ]
            log_file = log_dir / f"retrain_{author}_{ft}.log"
            print(f"  LAUNCH {label}")
            proc = _launch_process(cmd, log_file, env)
            running[proc] = (label, author)
            _print_progress()
            if len(running) < max_workers and idx < len(to_run):
                time.sleep(30)

        if not running:
            break

        finished = []
        for proc in list(running):
            ret = proc.poll()
            if ret is not None:
                finished.append(proc)

        if not finished:
            time.sleep(2)
            continue

        for proc in finished:
            label, _ = running.pop(proc)
            proc.wait()
            done_count += 1
            if proc.returncode != 0:
                failed += 1
                print(f"  [{done_count}/{n_run} DONE] {label} FAILED")
            else:
                print(f"  [{done_count}/{n_run} DONE] {label}")
            _print_progress()

    print(f"\nPhase 3R complete: {n_run - failed} trained, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


def phase_evaluate(
    config_path: str, cfg: dict, log_dir: Path, *, max_workers: int = 1
) -> int:
    print("\n" + "=" * 60)
    print(f"PHASE 4: Evaluation (max_workers={max_workers})")
    print("=" * 60)

    results_dir = log_dir.parent
    results_file = results_dir / "eval_results.json"

    if results_file.exists():
        with open(results_file, encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {}

    env = _make_env()

    # 1) Base model — always sequential (needs reference split)
    if "base" not in existing:
        print("  Evaluating BASE model...")
        cmd = [
            sys.executable,
            "benchmarks/scripts/pipeline_qa/evaluate.py",
            "--config", config_path,
            "--author", cfg["authors"][0]["name"],
            "--forget-type", "line",
            "--metrics", "all",
        ]
        rc = run_subprocess(cmd, log_dir / "eval_base.log")
        if rc != 0:
            print("  ERROR: base model eval failed")
            return 1
    else:
        print("  SKIP base (already evaluated)")

    # 2) Discover checkpoints and build per-checkpoint eval commands
    ALGO_NAMES = sorted(["retrain", "npo", "rmu"], key=len, reverse=True)

    checkpoints: dict[str, str] = {}

    sft_dir = Path(cfg["sft"]["output_dir"])
    if sft_dir.is_dir() and (sft_dir / "config.json").exists():
        checkpoints["sft"] = str(sft_dir)
    else:
        print(f"  NOTE: SFT dir not found or incomplete: {sft_dir}")

    algo_dirs = {
        "npo": cfg["npo"]["output_dir"],
        "rmu": cfg["rmu"]["output_dir"],
        "retrain": cfg.get("retrain", {}).get("output_dir", cfg["sft"]["output_dir"].replace("sft", "retrain")),
    }

    for algo_name, algo_dir in algo_dirs.items():
        algo_path = Path(algo_dir)
        if not algo_path.is_dir():
            print(f"  NOTE: {algo_name} dir not found: {algo_path}")
            continue
        for child in sorted(algo_path.iterdir()):
            if not child.is_dir():
                continue
            final_dir = child / "final"
            adapter_dir = final_dir if final_dir.is_dir() and (final_dir / "config.json").exists() else child
            if (adapter_dir / "config.json").exists():
                checkpoints[f"{algo_name}_{child.name}"] = str(adapter_dir)

    def _parse_key(key: str) -> tuple[str | None, str | None, str | None]:
        for algo in ALGO_NAMES:
            if key.startswith(algo + "_"):
                rest = key[len(algo) + 1:]
                for ft in ("random", "line", "page_prototype"):
                    if rest.endswith("_" + ft):
                        author = rest[: -(len(ft) + 1)]
                        return algo, author, ft
        return None, None, None
    print(f"\n  Discovered {len(checkpoints)} checkpoint(s), {len(existing)} already evaluated:")
    for key, ckpt_path in checkpoints.items():
        status = "DONE" if key in existing else "PENDING"
        print(f"    {status} {key} -> {ckpt_path}")

    to_run: list[tuple[str, list[str], str]] = []
    skipped = 0
    for key, ckpt_path in checkpoints.items():
        if key in existing:
            print(f"  SKIP {key} (already evaluated)")
            skipped += 1
            continue
        if key == "sft":
            cmd = [
                sys.executable,
                "benchmarks/scripts/pipeline_qa/evaluate.py",
                "--config", config_path,
                "--author", cfg["authors"][0]["name"],
                "--forget-type", "line",
                "--metrics", "all",
                "--full-model",
                "--checkpoint-path", ckpt_path,
            ]
        else:
            algo, author, ft = _parse_key(key)
            if not author or not ft:
                print(f"  SKIP {key} (cannot parse key)")
                skipped += 1
                continue
            cmd = [
                sys.executable,
                "benchmarks/scripts/pipeline_qa/evaluate.py",
                "--config", config_path,
                "--author", author,
                "--forget-type", ft,
                "--algorithm", algo,
                "--metrics", "all",
                "--full-model",
                "--checkpoint-path", ckpt_path,
            ]
        to_run.append((key, cmd, ckpt_path))
        print(f"  PENDING {key}")

    if not to_run:
        print(f"\n  All {len(checkpoints)} checkpoints already evaluated.")
        return 0

    n_run = len(to_run)

    if max_workers <= 1:
        done_count = 0
        failed = 0
        for key, cmd, _ in to_run:
            done_count += 1
            print(f"\n  [{done_count}/{n_run}] {key}")
            rc = run_subprocess(cmd, log_dir / f"eval_{key}.log")
            if rc != 0:
                print(f"  ERROR: {key} failed (exit {rc})")
                failed += 1
        print(f"\nPhase 4 complete: {n_run - failed} evaluated, {skipped} skipped, {failed} failed")
        return 0 if failed == 0 else 1

    running: dict[subprocess.Popen[str], str] = {}
    failed = 0
    done_count = 0
    idx = 0

    def _print_progress() -> None:
        running_labels = list(running.values())
        running_str = ", ".join(running_labels) if running_labels else "none"
        print(
            f"  [{done_count}/{n_run} DONE] "
            f"[{len(running)} RUNNING: {running_str}] "
            f"[{n_run - done_count - len(running)} PENDING]"
        )

    while idx < len(to_run) or running:
        while len(running) < max_workers and idx < len(to_run):
            key, cmd, _ = to_run[idx]
            idx += 1
            log_file = log_dir / f"eval_{key}.log"
            print(f"  LAUNCH {key}")
            proc = _launch_process(cmd, log_file, env)
            running[proc] = key
            _print_progress()
            if len(running) < max_workers and idx < len(to_run):
                time.sleep(10)

        if not running:
            break

        finished = []
        for proc in list(running):
            ret = proc.poll()
            if ret is not None:
                finished.append(proc)

        if not finished:
            time.sleep(2)
            continue

        for proc in finished:
            key = running.pop(proc)
            proc.wait()
            done_count += 1
            if proc.returncode != 0:
                failed += 1
                print(f"  [{done_count}/{n_run} DONE] {key} FAILED")
            else:
                print(f"  [{done_count}/{n_run} DONE] {key}")
            _print_progress()

    print(f"\nPhase 4 complete: {n_run - failed} evaluated, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="MU experiment pipeline orchestrator")
    parser.add_argument(
        "--config",
        default="benchmarks/scripts/pipeline_qa/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "3", "3r", "4", "all"],
        default="all",
        help="Which phase(s) to run (default: all)",
    )
    parser.add_argument(
        "--embedding-api", default=None, help="OpenAI-compatible embedding API URL"
    )
    parser.add_argument(
        "--skip-page-prototype",
        action="store_true",
        help="Skip page prototype embedding forget sets",
    )
    parser.add_argument(
        "--skip-ngram", action="store_true", help="Skip ngram forget sets"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        choices=range(1, 4),
        metavar="1-3",
        help="Max parallel unlearning jobs (default: 1)",
    )
    args = parser.parse_args()

    config_path = str(Path(args.config).resolve())
    cfg = load_config(config_path)
    project_root = Path(config_path).resolve().parent.parent.parent.parent
    log_dir = project_root / "benchmarks" / "results" / "pipeline_qa" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    phases: list[str]
    if args.phase == "all":
        phases = ["2", "1", "3", "3r", "4"]
    else:
        phases = [args.phase]

    print(f"Config:  {config_path}")
    print(f"Log dir: {log_dir}")
    print(f"Phases:  {phases}")

    forget_set_args: list[str] = []
    if args.embedding_api:
        forget_set_args.extend(["--embedding-api", args.embedding_api])
    if args.skip_page_prototype:
        forget_set_args.append("--skip-page-prototype")
    if args.skip_ngram:
        forget_set_args.append("--skip-ngram")

    phase_funcs = {
        "2": lambda: phase_build_forget_sets(config_path, log_dir, forget_set_args),
        "1": lambda: phase_train_sft(config_path, cfg, log_dir),
        "3": lambda: phase_unlearn(config_path, cfg, log_dir, max_workers=args.max_workers),
        "3r": lambda: phase_retrain(config_path, cfg, log_dir, max_workers=args.max_workers),
        "4": lambda: phase_evaluate(config_path, cfg, log_dir, max_workers=args.max_workers),
    }

    start_time = time.time()
    phase_results: dict[str, dict] = {}

    for phase in phases:
        t0 = time.time()
        rc = phase_funcs[phase]()
        elapsed = time.time() - t0
        phase_results[f"phase_{phase}"] = {
            "return_code": rc,
            "duration_seconds": round(elapsed, 1),
        }
        if rc != 0:
            print(f"\nPhase {phase} FAILED (exit {rc}). Aborting pipeline.")
            break

    total_time = time.time() - start_time

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    for phase, info in phase_results.items():
        status = "OK" if info["return_code"] == 0 else "FAILED"
        duration = str(timedelta(seconds=int(info["duration_seconds"])))
        print(f"  {phase}: {status}  ({duration})")

    print(f"\n  Total: {str(timedelta(seconds=int(total_time)))}")

    eval_info = phase_results.get("phase_4")
    if eval_info and eval_info["return_code"] == 0:
        eval_file = Path(cfg["eval"]["output_dir"]) / "eval_results.json"
        if eval_file.exists():
            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)
            with open(eval_file, encoding="utf-8") as f:
                eval_data = json.load(f)
            for key, m in eval_data.items():
                print(
                    f"  {key:<28} "
                    f"f_ppl={m['forget_ppl']:>8.1f}  "
                    f"r_ppl={m['retain_ppl']:>8.1f}  "
                    f"f_rouge={m['forget_rouge_l']:.4f}  "
                    f"r_rouge={m['retain_rouge_l']:.4f}"
                )

    # ── Save pipeline log ────────────────────────────────────────────────
    pipeline_log = {
        "config": config_path,
        "phases": phases,
        "total_duration_seconds": round(total_time, 1),
        "phase_results": phase_results,
    }
    log_path = log_dir / "run_all.log"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_log, f, indent=2, ensure_ascii=False)
    print(f"\nPipeline log: {log_path}")


if __name__ == "__main__":
    main()
