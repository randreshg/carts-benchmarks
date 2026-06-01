#!/usr/bin/env python3
"""Run the 64-thread 1-node to 2-node validation gate one case at a time."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


CONFIG_NAME = "all-enabled-1-to-2-64t-validation"
DEFAULT_RESULTS_DIR = Path("external/carts-benchmarks/results/serial-1-to-2-64t")
PASS_STATUS = "pass"


@dataclass(frozen=True)
class Phase:
    name: str
    size: str
    threads: str
    nodes: str
    timeout: int
    runs: int
    rdma: bool
    compile_args: Optional[str]


@dataclass
class PhaseResult:
    benchmark: str
    phase: str
    status: str
    metric_time_sec: Optional[float]
    result_dir: Path
    results_json: Optional[Path]
    runtime_warning: bool = False
    runtime_warning_reasons: list[str] | None = None
    error: Optional[str] = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def config_path(root: Path) -> Path:
    return (
        root
        / "external"
        / "carts-benchmarks"
        / "configs"
        / "experiments"
        / f"{CONFIG_NAME}.json"
    )


def load_config(root: Path) -> tuple[list[str], list[Phase]]:
    payload = json.loads(config_path(root).read_text())
    steps = payload.get("steps", [])
    if not steps:
        raise ValueError(f"{CONFIG_NAME} has no steps")

    benchmarks = steps[0].get("benchmarks") or []
    if not benchmarks:
        raise ValueError(f"{CONFIG_NAME} first step must list benchmarks")

    phases: list[Phase] = []
    for step in steps:
        phases.append(
            Phase(
                name=str(step["name"]),
                size=str(step["size"]),
                threads=str(step["threads"]),
                nodes=str(step["nodes"]),
                timeout=int(step["timeout"]),
                runs=int(step.get("runs", 1)),
                rdma=bool(step.get("rdma", True)),
                compile_args=step.get("compile_args"),
            )
        )
    return list(benchmarks), phases


def dekk_path(root: Path, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    local = root / ".dekk" / "env" / "bin" / "dekk"
    return str(local) if local.exists() else "dekk"


def result_dirs(base: Path) -> set[Path]:
    if not base.exists():
        return set()
    return {path for path in base.iterdir() if path.is_dir()}


def newest_result_dir(base: Path, before: set[Path]) -> Path:
    after = result_dirs(base)
    created = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if created:
        return created[-1]
    existing = sorted(after, key=lambda path: path.stat().st_mtime)
    if existing:
        return existing[-1]
    return base


def load_rows(result_dir: Path) -> tuple[list[dict[str, Any]], Optional[Path]]:
    results_json = result_dir / "results.json"
    if results_json.exists():
        payload = json.loads(results_json.read_text())
        rows = payload.get("results", [])
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)], results_json

    rows: list[dict[str, Any]] = []
    for path in sorted(result_dir.rglob("result.json")):
        try:
            row = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows, results_json if results_json.exists() else None


def sum_timing_map(value: Any) -> Optional[float]:
    if not isinstance(value, dict) or not value:
        return None
    total = 0.0
    seen = False
    for item in value.values():
        try:
            total += float(item)
            seen = True
        except (TypeError, ValueError):
            continue
    return total if seen else None


def metric_time(row: dict[str, Any]) -> Optional[float]:
    arts = row.get("arts")
    if isinstance(arts, dict):
        for key in ("e2e_timings", "kernel_timings"):
            value = sum_timing_map(arts.get(key))
            if value is not None:
                return value
        try:
            return float(arts["duration_sec"])
        except (KeyError, TypeError, ValueError):
            pass
    try:
        return float(row["duration_sec"])
    except (KeyError, TypeError, ValueError):
        return None


def runtime_warning(row: dict[str, Any]) -> tuple[bool, list[str]]:
    diagnostics = row.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return False, []
    warning = diagnostics.get("runtime_warning")
    if not isinstance(warning, dict):
        return False, []
    reasons = warning.get("reasons")
    reason_list = [str(item) for item in reasons] if isinstance(reasons, list) else []
    return bool(warning.get("has_warning")), reason_list


def summarize_phase(
    benchmark: str,
    phase: Phase,
    result_dir: Path,
    *,
    fail_on_runtime_warning: bool,
) -> PhaseResult:
    rows, results_json = load_rows(result_dir)
    if not rows:
        return PhaseResult(
            benchmark=benchmark,
            phase=phase.name,
            status="missing",
            metric_time_sec=None,
            result_dir=result_dir,
            results_json=results_json,
            error="no result rows found",
        )

    statuses = {str(row.get("status", "")).lower() for row in rows}
    warnings: list[str] = []
    has_runtime_warning = False
    times = [metric_time(row) for row in rows]
    metric = next((time for time in times if time is not None), None)

    for row in rows:
        row_has_warning, row_reasons = runtime_warning(row)
        if row_has_warning:
            has_runtime_warning = True
            warnings.extend(row_reasons)

    status = PASS_STATUS if statuses == {PASS_STATUS} else ",".join(sorted(statuses))
    error = None
    if status != PASS_STATUS:
        error = f"non-pass status: {status}"
    elif metric is not None and metric > phase.timeout:
        status = "timeout-budget"
        error = f"metric time {metric:.3f}s exceeded {phase.timeout}s"
    elif fail_on_runtime_warning and has_runtime_warning:
        status = "runtime-warning"
        error = "runtime warning: " + ", ".join(warnings or ["unknown"])

    return PhaseResult(
        benchmark=benchmark,
        phase=phase.name,
        status=status,
        metric_time_sec=metric,
        result_dir=result_dir,
        results_json=results_json,
        runtime_warning=has_runtime_warning,
        runtime_warning_reasons=warnings,
        error=error,
    )


def phase_command(
    *,
    dekk: str,
    benchmark: str,
    phase: Phase,
    partition: str,
    results_dir: Path,
    dry_run: bool,
    nodelist: Optional[str],
    exclude_nodes: Optional[str],
    cpu_pinning: Optional[str],
    time_limit: Optional[str],
    quiet: bool,
) -> list[str]:
    command = [
        dekk,
        "carts",
        "benchmarks",
        "run",
        benchmark,
        "--slurm",
        "--partition",
        partition,
        "--max-jobs",
        "1",
        "--results-dir",
        str(results_dir),
        "--size",
        phase.size,
        "--threads",
        phase.threads,
        "--nodes",
        phase.nodes,
        "--timeout",
        str(phase.timeout),
        "--time-limit",
        time_limit or "00:05:00",
        "--runs",
        str(phase.runs),
    ]
    command.append("--rdma" if phase.rdma else "--no-rdma")
    if phase.compile_args:
        command.extend(["--compile-args", phase.compile_args])
    if dry_run:
        command.append("--dry-run")
    if nodelist:
        command.extend(["--nodelist", nodelist])
    if exclude_nodes:
        command.extend(["--exclude-nodes", exclude_nodes])
    if cpu_pinning:
        command.extend(["--cpu-pinning", cpu_pinning])
    if quiet:
        command.append("--quiet")
    return command


def benchmark_token(benchmark: str) -> str:
    return benchmark.replace("/", "__")


def check_scaling(
    benchmark: str,
    phase_results: list[PhaseResult],
    min_two_node_speedup: float,
) -> Optional[str]:
    by_phase = {result.phase: result for result in phase_results}
    reference = by_phase.get("single-node-reference")
    if reference is None or reference.metric_time_sec is None:
        return "missing single-node reference timing"

    failures: list[str] = []
    for phase_name in ("two-node-baseline", "two-node-distributed-db"):
        candidate = by_phase.get(phase_name)
        if candidate is None or candidate.metric_time_sec is None:
            failures.append(f"{phase_name}: missing 2-node timing")
            continue
        if candidate.metric_time_sec <= 0:
            failures.append(f"{phase_name}: invalid 2-node timing")
            continue
        speedup = reference.metric_time_sec / candidate.metric_time_sec
        if speedup < min_two_node_speedup:
            failures.append(
                f"{phase_name}: speedup {speedup:.3f}x < {min_two_node_speedup:.3f}x"
            )
    if failures:
        return f"{benchmark} did not scale 1n->2n: " + "; ".join(failures)
    return None


def write_summary(results_dir: Path, rows: Iterable[PhaseResult]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "serial_1_to_2_summary.json"
    payload = [
        {
            "benchmark": row.benchmark,
            "phase": row.phase,
            "status": row.status,
            "metric_time_sec": row.metric_time_sec,
            "result_dir": str(row.result_dir),
            "results_json": str(row.results_json) if row.results_json else None,
            "runtime_warning": row.runtime_warning,
            "runtime_warning_reasons": row.runtime_warning_reasons or [],
            "error": row.error,
        }
        for row in rows
    ]
    summary_path.write_text(json.dumps(payload, indent=2))
    return summary_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run every enabled benchmark through the 64-thread 1n->2n gate "
            "serially, stopping on the first failure or non-scaling case."
        )
    )
    parser.add_argument("benchmarks", nargs="*", help="Optional benchmark subset")
    parser.add_argument("-p", "--partition", required=True, help="SLURM partition")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Base results directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument("--dekk", help="Path to the dekk executable")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts only")
    parser.add_argument("--nodelist", help="SLURM nodelist restriction")
    parser.add_argument("--exclude-nodes", help="Comma-separated SLURM node exclusion")
    parser.add_argument("--cpu-pinning", help="CPU pinning mode passed to the runner")
    parser.add_argument(
        "--time-limit",
        default="00:05:00",
        help="SLURM wall time per job. Default keeps the whole example under five minutes.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Run remaining cases after a failure instead of stopping immediately",
    )
    parser.add_argument(
        "--min-two-node-speedup",
        type=float,
        default=1.0,
        help="Minimum 1n/2n ARTS timing speedup required for both 2-node phases",
    )
    parser.add_argument(
        "--allow-runtime-warnings",
        action="store_true",
        help="Do not fail a passing result only because runtime warnings were detected",
    )
    parser.add_argument("--quiet", action="store_true", help="Pass --quiet to runner")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = repo_root()
    configured_benchmarks, phases = load_config(root)
    selected = args.benchmarks or configured_benchmarks
    unknown = sorted(set(selected) - set(configured_benchmarks))
    if unknown:
        print("Unknown benchmark(s): " + ", ".join(unknown), file=sys.stderr)
        return 2

    dekk = dekk_path(root, args.dekk)
    env = os.environ.copy()
    env.setdefault("CARTS_DIR", str(root))

    all_results: list[PhaseResult] = []
    args.results_dir.mkdir(parents=True, exist_ok=True)

    for benchmark in selected:
        print(f"\n=== {benchmark} ===", flush=True)
        current_benchmark: list[PhaseResult] = []
        for phase in phases:
            phase_dir = args.results_dir / benchmark_token(benchmark) / phase.name
            before = result_dirs(phase_dir)
            command = phase_command(
                dekk=dekk,
                benchmark=benchmark,
                phase=phase,
                partition=args.partition,
                results_dir=phase_dir,
                dry_run=args.dry_run,
                nodelist=args.nodelist,
                exclude_nodes=args.exclude_nodes,
                cpu_pinning=args.cpu_pinning,
                time_limit=args.time_limit,
                quiet=args.quiet,
            )
            print("$ " + " ".join(command), flush=True)
            completed = subprocess.run(command, cwd=root, env=env, check=False)
            result_dir = newest_result_dir(phase_dir, before)

            if args.dry_run:
                phase_result = PhaseResult(
                    benchmark=benchmark,
                    phase=phase.name,
                    status="dry-run",
                    metric_time_sec=None,
                    result_dir=result_dir,
                    results_json=result_dir / "results.json",
                )
            elif completed.returncode != 0:
                phase_result = PhaseResult(
                    benchmark=benchmark,
                    phase=phase.name,
                    status="runner-failed",
                    metric_time_sec=None,
                    result_dir=result_dir,
                    results_json=(result_dir / "results.json"),
                    error=f"runner exited with {completed.returncode}",
                )
            else:
                phase_result = summarize_phase(
                    benchmark,
                    phase,
                    result_dir,
                    fail_on_runtime_warning=not args.allow_runtime_warnings,
                )

            all_results.append(phase_result)
            current_benchmark.append(phase_result)
            metric = (
                f"{phase_result.metric_time_sec:.3f}s"
                if phase_result.metric_time_sec is not None
                else "n/a"
            )
            print(
                f"{benchmark} {phase.name}: {phase_result.status} ({metric})",
                flush=True,
            )

            if phase_result.error and not args.continue_on_failure:
                summary = write_summary(args.results_dir, all_results)
                print(f"Stopped at {benchmark} / {phase.name}: {phase_result.error}")
                print(f"Summary: {summary}")
                return 1

        if not args.dry_run:
            scaling_error = check_scaling(
                benchmark,
                current_benchmark,
                args.min_two_node_speedup,
            )
            if scaling_error:
                all_results.append(
                    PhaseResult(
                        benchmark=benchmark,
                        phase="scaling-check",
                        status="scale-fail",
                        metric_time_sec=None,
                        result_dir=args.results_dir / benchmark_token(benchmark),
                        results_json=None,
                        error=scaling_error,
                    )
                )
                if not args.continue_on_failure:
                    summary = write_summary(args.results_dir, all_results)
                    print(scaling_error)
                    print(f"Summary: {summary}")
                    return 1

    summary = write_summary(args.results_dir, all_results)
    print(f"\nSummary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
