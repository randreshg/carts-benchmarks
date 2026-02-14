#!/usr/bin/env python3
"""
CARTS Benchmark Analyzer

Post-run analysis tool for benchmark results. Works with both
``results.json`` (from ``carts benchmarks run``) and
``aggregated_results.json`` (from ``carts benchmarks slurm-run``).

Usage:
    carts analyze summary <results_dir> [OPTIONS]
    carts analyze export  <results_dir> [OPTIONS]
    carts analyze compare <baseline_dir> <candidate_dir> [OPTIONS]
"""

from __future__ import annotations

import csv
import io
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

import typer

from carts_styles import (
    console,
    print_header,
    print_error,
    print_info,
    print_success,
    print_warning,
)

from benchmark_common import (
    get_result_config,
    get_result_status,
    get_result_timing,
    load_experiment,
)

app = typer.Typer(
    name="carts-analyze",
    help="Analyze benchmark results from completed experiments.",
    add_completion=False,
    no_args_is_help=True,
)


# ============================================================================
# Helpers
# ============================================================================

def _geometric_mean(values: List[float]) -> Optional[float]:
    """Geometric mean of positive values, or None."""
    positives = [v for v in values if v > 0]
    if not positives:
        return None
    return math.exp(sum(math.log(v) for v in positives) / len(positives))


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def _fmt(v: Optional[float], decimals: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{decimals}f}"


def _group_results(
    experiment: Dict[str, Any],
    benchmark_filter: Optional[str] = None,
) -> Dict[Tuple[str, int, int], List[Dict[str, Any]]]:
    """Group results by (benchmark, threads, nodes), optionally filtered."""
    source = experiment["source"]
    metadata = experiment.get("metadata", {})
    groups: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = defaultdict(list)

    for r in experiment.get("results", []):
        name, threads, nodes = get_result_config(r, source, metadata)
        if benchmark_filter and benchmark_filter.lower() not in name.lower():
            continue
        groups[(name, threads, nodes)].append(r)

    return groups


# ============================================================================
# summary command
# ============================================================================

@app.command()
def summary(
    results_dir: Path = typer.Argument(..., help="Experiment results directory"),
    benchmark: Optional[str] = typer.Option(
        None, "--benchmark", "-b", help="Filter by benchmark name (substring match)"),
    sort: str = typer.Option(
        "name", "--sort", help="Sort by: name, speedup"),
):
    """Display a summary table of benchmark results."""
    try:
        experiment = load_experiment(results_dir)
    except FileNotFoundError as e:
        print_error(str(e))
        raise typer.Exit(1)

    source = experiment["source"]
    groups = _group_results(experiment, benchmark)

    if not groups:
        print_warning("No matching results found.")
        raise typer.Exit(0)

    print_header("Benchmark Results Summary", str(results_dir))

    from rich.table import Table

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Benchmark", style="white", no_wrap=True)
    table.add_column("Threads", justify="right")
    table.add_column("Nodes", justify="right")
    table.add_column("Runs", justify="right")
    table.add_column("ARTS E2E (s)", justify="right")
    table.add_column("OMP E2E (s)", justify="right")
    table.add_column("ARTS Kernel (s)", justify="right")
    table.add_column("OMP Kernel (s)", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Status", justify="center")

    rows = []
    all_speedups = []
    total_pass = 0
    total_fail = 0

    for (name, threads, nodes), results in groups.items():
        arts_e2e = [v for r in results if (v := get_result_timing(r, source, "arts_e2e")) is not None]
        omp_e2e = [v for r in results if (v := get_result_timing(r, source, "omp_e2e")) is not None]
        arts_kernel = [v for r in results if (v := get_result_timing(r, source, "arts_kernel")) is not None]
        omp_kernel = [v for r in results if (v := get_result_timing(r, source, "omp_kernel")) is not None]
        speedups = [v for r in results if (v := get_result_timing(r, source, "speedup")) is not None and v > 0]

        arts_e2e_mean, _ = _mean_std(arts_e2e)
        omp_e2e_mean, _ = _mean_std(omp_e2e)
        arts_kernel_mean, _ = _mean_std(arts_kernel)
        omp_kernel_mean, _ = _mean_std(omp_kernel)
        sp_mean, sp_std = _mean_std(speedups)

        passed = sum(1 for r in results if get_result_status(r, source) == "PASS")
        failed = len(results) - passed
        total_pass += passed
        total_fail += failed

        if sp_mean is not None:
            all_speedups.append(sp_mean)
            if sp_mean >= 1.0:
                sp_str = f"[green]{sp_mean:.2f}x[/]"
            else:
                sp_str = f"[red]{sp_mean:.2f}x[/]"
            if sp_std and sp_std > 0:
                sp_str += f" [dim]\u00b1{sp_std:.2f}[/]"
        else:
            sp_str = "[dim]-[/]"

        status_str = f"[green]{passed}[/]" if failed == 0 else f"[green]{passed}[/]/[red]{failed}[/]"

        rows.append((
            name, threads, nodes, len(results),
            arts_e2e_mean, omp_e2e_mean, arts_kernel_mean, omp_kernel_mean,
            sp_mean, sp_str, status_str,
        ))

    # Sort
    if sort == "speedup":
        rows.sort(key=lambda r: r[8] if r[8] is not None else -1, reverse=True)
    else:
        rows.sort(key=lambda r: (r[0], r[1], r[2]))

    for row in rows:
        table.add_row(
            row[0], str(row[1]), str(row[2]), str(row[3]),
            _fmt(row[4]), _fmt(row[5]),
            _fmt(row[6]), _fmt(row[7]),
            row[9], row[10],
        )

    console.print(table)

    # Summary footer
    geomean = _geometric_mean(all_speedups)
    parts = []
    if geomean is not None:
        parts.append(f"Geometric mean speedup: [bold]{geomean:.2f}x[/]")
    parts.append(f"[green]{total_pass}[/] passed, [red]{total_fail}[/] failed")
    console.print()
    console.print("  ".join(parts))


# ============================================================================
# export command
# ============================================================================

@app.command()
def export(
    results_dir: Path = typer.Argument(..., help="Experiment results directory"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output CSV file (default: stdout)"),
    benchmark: Optional[str] = typer.Option(
        None, "--benchmark", "-b", help="Filter by benchmark name (substring match)"),
):
    """Export timing data to CSV."""
    try:
        experiment = load_experiment(results_dir)
    except FileNotFoundError as e:
        print_error(str(e))
        raise typer.Exit(1)

    source = experiment["source"]
    metadata = experiment.get("metadata", {})

    fieldnames = [
        "benchmark", "threads", "nodes", "run",
        "arts_e2e_sec", "omp_e2e_sec",
        "arts_kernel_sec", "omp_kernel_sec",
        "arts_init_sec", "omp_init_sec",
        "speedup", "status",
    ]

    buf = io.StringIO() if output is None else None
    fh = open(output, "w", newline="") if output else buf

    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()

    for r in experiment.get("results", []):
        name, threads, nodes = get_result_config(r, source, metadata)
        if benchmark and benchmark.lower() not in name.lower():
            continue

        run_number = r.get("run_number", 1)
        writer.writerow({
            "benchmark": name,
            "threads": threads,
            "nodes": nodes,
            "run": run_number,
            "arts_e2e_sec": _fmt(get_result_timing(r, source, "arts_e2e"), 6),
            "omp_e2e_sec": _fmt(get_result_timing(r, source, "omp_e2e"), 6),
            "arts_kernel_sec": _fmt(get_result_timing(r, source, "arts_kernel"), 6),
            "omp_kernel_sec": _fmt(get_result_timing(r, source, "omp_kernel"), 6),
            "arts_init_sec": _fmt(get_result_timing(r, source, "arts_init"), 6),
            "omp_init_sec": _fmt(get_result_timing(r, source, "omp_init"), 6),
            "speedup": _fmt(get_result_timing(r, source, "speedup"), 4),
            "status": get_result_status(r, source),
        })

    if output:
        fh.close()
        print_success(f"Exported to {output}")
    else:
        sys.stdout.write(buf.getvalue())


# ============================================================================
# compare command
# ============================================================================

@app.command()
def compare(
    baseline_dir: Path = typer.Argument(..., help="Baseline experiment directory"),
    candidate_dir: Path = typer.Argument(..., help="Candidate experiment directory"),
    threshold: float = typer.Option(
        0.05, "--threshold", help="Relative change threshold for verdict (default: 5%)"),
    benchmark: Optional[str] = typer.Option(
        None, "--benchmark", "-b", help="Filter by benchmark name (substring match)"),
):
    """Compare two experiments side-by-side."""
    try:
        base_exp = load_experiment(baseline_dir)
        cand_exp = load_experiment(candidate_dir)
    except FileNotFoundError as e:
        print_error(str(e))
        raise typer.Exit(1)

    base_groups = _group_results(base_exp, benchmark)
    cand_groups = _group_results(cand_exp, benchmark)

    all_keys = sorted(set(base_groups.keys()) | set(cand_groups.keys()))
    if not all_keys:
        print_warning("No matching results found.")
        raise typer.Exit(0)

    print_header("Experiment Comparison")
    print_info(f"Baseline:  {baseline_dir}")
    print_info(f"Candidate: {candidate_dir}")
    print_info(f"Threshold: {threshold:.0%}")
    console.print()

    from rich.table import Table

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Benchmark", style="white", no_wrap=True)
    table.add_column("Config", justify="center")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Verdict", justify="center")

    improved = 0
    regressed = 0
    unchanged = 0
    new_benchmarks = 0
    removed_benchmarks = 0

    base_source = base_exp["source"]
    cand_source = cand_exp["source"]

    for key in all_keys:
        name, threads, nodes = key
        config_str = f"{threads}t {nodes}n"

        base_results = base_groups.get(key)
        cand_results = cand_groups.get(key)

        if base_results is None:
            table.add_row(name, config_str, "[dim]-[/]", "[dim]new[/]", "-", "[cyan]NEW[/]")
            new_benchmarks += 1
            continue

        if cand_results is None:
            table.add_row(name, config_str, "[dim]present[/]", "[dim]-[/]", "-", "[yellow]REMOVED[/]")
            removed_benchmarks += 1
            continue

        base_speedups = [v for r in base_results if (v := get_result_timing(r, base_source, "speedup")) is not None and v > 0]
        cand_speedups = [v for r in cand_results if (v := get_result_timing(r, cand_source, "speedup")) is not None and v > 0]

        base_mean, _ = _mean_std(base_speedups)
        cand_mean, _ = _mean_std(cand_speedups)

        if base_mean is None or cand_mean is None:
            table.add_row(name, config_str, _fmt(base_mean, 2), _fmt(cand_mean, 2), "-", "[dim]N/A[/]")
            unchanged += 1
            continue

        delta = cand_mean - base_mean
        rel_change = delta / base_mean if base_mean != 0 else 0.0

        if rel_change > threshold:
            verdict = "[green]IMPROVED[/]"
            improved += 1
        elif rel_change < -threshold:
            verdict = "[red]REGRESSED[/]"
            regressed += 1
        else:
            verdict = "[dim]SAME[/]"
            unchanged += 1

        delta_str = f"{delta:+.2f} ({rel_change:+.0%})"
        if rel_change > threshold:
            delta_str = f"[green]{delta_str}[/]"
        elif rel_change < -threshold:
            delta_str = f"[red]{delta_str}[/]"

        table.add_row(
            name, config_str,
            f"{base_mean:.2f}x", f"{cand_mean:.2f}x",
            delta_str, verdict,
        )

    console.print(table)
    console.print()

    parts = []
    if improved:
        parts.append(f"[green]{improved} improved[/]")
    if regressed:
        parts.append(f"[red]{regressed} regressed[/]")
    if unchanged:
        parts.append(f"[dim]{unchanged} unchanged[/]")
    if new_benchmarks:
        parts.append(f"[cyan]{new_benchmarks} new[/]")
    if removed_benchmarks:
        parts.append(f"[yellow]{removed_benchmarks} removed[/]")
    console.print(", ".join(parts))


if __name__ == "__main__":
    app()
