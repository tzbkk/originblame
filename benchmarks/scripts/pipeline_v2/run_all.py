#!/usr/bin/env python3
from __future__ import annotations

"""Run all 28 pipeline configurations sequentially.

Generates every (framework, data_source, scale, with_ob) tuple and executes
the appropriate orchestrator script via subprocess.

  2 frameworks × (4 zhwiki scales + 3 kernel scales) × 2 ob configs = 28 runs

Usage:
    python benchmarks/scripts/pipeline_v2/run_all.py
    python benchmarks/scripts/pipeline_v2/run_all.py --dry-run
    python benchmarks/scripts/pipeline_v2/run_all.py --output-dir results/pipeline_v2
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

FRAMEWORKS = ["datatrove", "huggingface"]
DATA_SOURCES = {
    "zhwiki": ["1k", "10k", "100k", "all"],
    "kernel": ["1k", "10k", "all"],
}
OB_CONFIGS = [True, False]  # with_ob, no_ob

SCRIPT_DIR = Path(__file__).resolve().parent


def script_for_framework(framework: str) -> Path:
    """Return the orchestrator script path for a framework."""
    return SCRIPT_DIR / f"run_{framework}.py"


# ═══════════════════════════════════════════════════════════════════════════════
# Run config
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RunConfig:
    framework: str
    data_source: str
    scale: str
    with_ob: bool

    @property
    def ob_label(self) -> str:
        return "ob" if self.with_ob else "baseline"

    @property
    def run_id(self) -> str:
        ob_suffix = "-ob" if self.with_ob else ""
        return f"{self.framework}-{self.data_source}-{self.scale}{ob_suffix}"


def generate_all_configs() -> list[RunConfig]:
    """Generate all 28 pipeline configurations."""
    configs: list[RunConfig] = []
    for framework in FRAMEWORKS:
        for data_source, scales in DATA_SOURCES.items():
            for scale in scales:
                for with_ob in OB_CONFIGS:
                    configs.append(
                        RunConfig(
                            framework=framework,
                            data_source=data_source,
                            scale=scale,
                            with_ob=with_ob,
                        )
                    )
    return configs


# ═══════════════════════════════════════════════════════════════════════════════
# Result tracking
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RunResult:
    config: RunConfig
    success: bool
    returncode: int = 0
    elapsed_s: float = 0.0
    error: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run all 28 pipeline configurations",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/pipeline_v2"),
        help="Base output directory (default: benchmarks/results/pipeline_v2)",
    )
    p.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("benchmarks/raw_data"),
        help="Directory containing raw data dumps",
    )
    p.add_argument(
        "--linux-dir",
        type=Path,
        default=Path("benchmarks/raw_data/linux"),
        help="Path to Linux kernel source tree",
    )
    p.add_argument(
        "--tokenizer",
        default="gpt2",
        help="Tokenizer name or path (default: gpt2)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all commands without executing",
    )
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Command building
# ═══════════════════════════════════════════════════════════════════════════════


def build_command(
    config: RunConfig,
    output_dir: Path,
    raw_data_dir: Path,
    linux_dir: Path,
    tokenizer: str,
) -> list[str]:
    """Build the subprocess command for a given configuration."""
    script_path = script_for_framework(config.framework)
    ob_flag = "--with-ob" if config.with_ob else "--no-ob"
    cmd = [
        sys.executable,
        str(script_path),
        "--data-source", config.data_source,
        "--scale", config.scale,
        ob_flag,
        "--output-dir", str(output_dir),
        "--raw-data-dir", str(raw_data_dir),
        "--linux-dir", str(linux_dir),
    ]
    if config.framework == "datatrove":
        cmd.extend(["--tokenizer", tokenizer])
    return cmd


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════


def print_summary_table(results: list[RunResult]) -> None:
    """Print a summary table of all runs."""
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"{'#':<4} {'Run ID':<35} {'Status':<10} {'RC':<4} {'Time(s)':<8}")
    print(f"{'-' * 72}")

    passed = 0
    failed = 0
    for i, r in enumerate(results, 1):
        status = "PASS" if r.success else "FAIL"
        if r.success:
            passed += 1
        else:
            failed += 1
        print(
            f"{i:<4} {r.config.run_id:<35} {status:<10} "
            f"{r.returncode:<4} {r.elapsed_s:<8.1f}"
        )

    print(f"{'-' * 72}")
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print(f"{'=' * 72}\n")


def write_summary_json(results: list[RunResult], output_dir: Path) -> None:
    """Write summary.json to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "runs": [
            {
                "run_id": r.config.run_id,
                "framework": r.config.framework,
                "data_source": r.config.data_source,
                "scale": r.config.scale,
                "with_ob": r.config.with_ob,
                "success": r.success,
                "returncode": r.returncode,
                "elapsed_s": r.elapsed_s,
                "error": r.error,
            }
            for r in results
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summary written to {summary_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    configs = generate_all_configs()
    total = len(configs)
    print(f"Generated {total} pipeline configurations\n")

    results: list[RunResult] = []

    for i, config in enumerate(configs, 1):
        cmd = build_command(
            config=config,
            output_dir=args.output_dir,
            raw_data_dir=args.raw_data_dir,
            linux_dir=args.linux_dir,
            tokenizer=args.tokenizer,
        )

        # Print progress
        label = f"{config.framework} {config.data_source} {config.scale} {config.ob_label}"
        print(f"[{i}/{total}] {label}...")

        if args.dry_run:
            print(f"  Command: {' '.join(cmd)}")
            results.append(
                RunResult(config=config, success=True, returncode=0, elapsed_s=0.0)
            )
            continue

        # Execute
        start = time.monotonic()
        try:
            env = os.environ.copy()
            project_root = str(Path(__file__).resolve().parent.parent.parent)
            env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)
            elapsed = time.monotonic() - start

            if proc.returncode == 0:
                print(f"  OK ({elapsed:.1f}s)")
                results.append(
                    RunResult(
                        config=config,
                        success=True,
                        returncode=proc.returncode,
                        elapsed_s=round(elapsed, 1),
                    )
                )
            else:
                # Print error but continue
                stderr = proc.stderr.strip()
                print(f"  FAILED (rc={proc.returncode}, {elapsed:.1f}s)")
                if stderr:
                    # Show last few lines of stderr
                    for line in stderr.splitlines()[-5:]:
                        print(f"    {line}")
                results.append(
                    RunResult(
                        config=config,
                        success=False,
                        returncode=proc.returncode,
                        elapsed_s=round(elapsed, 1),
                        error=stderr[-500:] if stderr else "",
                    )
                )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            print(f"  TIMEOUT ({elapsed:.1f}s)")
            results.append(
                RunResult(
                    config=config,
                    success=False,
                    returncode=-1,
                    elapsed_s=round(elapsed, 1),
                    error="Timeout after 7200s",
                )
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            print(f"  ERROR: {exc}")
            results.append(
                RunResult(
                    config=config,
                    success=False,
                    returncode=-1,
                    elapsed_s=round(elapsed, 1),
                    error=str(exc),
                )
            )

    # Summary
    print_summary_table(results)

    if not args.dry_run:
        write_summary_json(results, args.output_dir)

    return 1 if any(not r.success for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
