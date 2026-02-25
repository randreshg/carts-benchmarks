"""Excel report generation for benchmark runs."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple

from benchmark_common import parse_all_counters

if TYPE_CHECKING:
    from benchmark_models import BenchmarkResult, ExperimentStep

try:
    from openpyxl import Workbook
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import Font
    from openpyxl.styles import PatternFill
except ImportError:  # pragma: no cover - runtime dependency check
    Workbook = None  # type: ignore[assignment]
    CellIsRule = None  # type: ignore[assignment]
    Font = None  # type: ignore[assignment]
    PatternFill = None  # type: ignore[assignment]


RESULTS_COLUMNS = [
    "benchmark",
    "suite",
    "size",
    "threads",
    "nodes",
    "run",
    "run_phase",
    "status",
    "speedup_basis",
    "arts_e2e_sec",
    "omp_e2e_sec",
    "arts_kernel_sec",
    "omp_kernel_sec",
    "arts_init_sec",
    "omp_init_sec",
    "arts_total_sec",
    "omp_total_sec",
    "speedup",
    "parallel_efficiency",
    "init_overhead_pct",
    "cache_references",
    "cache_misses",
    "cache_miss_rate",
    "l1d_loads",
    "l1d_load_misses",
    "l1d_load_miss_rate",
    "num_edts_created",
    "num_edts_finished",
    "num_dbs_created",
    "memory_footprint_bytes",
    "remote_bytes_sent",
    "remote_bytes_received",
    "edt_running_time_ms",
    "initialization_time_ms",
    "end_to_end_time_ms",
    "task_throughput",
    "avg_task_time_us",
    "memory_per_edt",
    "comm_bytes_per_edt",
]

SUMMARY_COLUMNS = [
    "benchmark",
    "suite",
    "size",
    "threads",
    "nodes",
    "run_phase",
    "num_runs",
    "arts_e2e_mean",
    "arts_e2e_std",
    "arts_e2e_cv_pct",
    "omp_e2e_mean",
    "omp_e2e_std",
    "arts_kernel_mean",
    "omp_kernel_mean",
    "arts_init_mean",
    "omp_init_mean",
    "speedup_mean",
    "speedup_std",
    "speedup_min",
    "speedup_max",
    "parallel_efficiency_mean",
    "init_overhead_pct_mean",
    "pass_count",
    "fail_count",
]

COUNTER_FIELD_MAP = {
    "num_edts_created": "numEdtsCreated",
    "num_edts_finished": "numEdtsFinished",
    "num_dbs_created": "numDbsCreated",
    "memory_footprint_bytes": "memoryFootprint",
    "remote_bytes_sent": "remoteBytesSent",
    "remote_bytes_received": "remoteBytesReceived",
    "edt_running_time_ms": "edtRunningTime",
    "initialization_time_ms": "initializationTime",
    "end_to_end_time_ms": "endToEndTime",
}

INT_FIELDS = {
    "threads",
    "nodes",
    "run",
    "cache_references",
    "cache_misses",
    "l1d_loads",
    "l1d_load_misses",
    "num_edts_created",
    "num_edts_finished",
    "num_dbs_created",
    "memory_footprint_bytes",
    "remote_bytes_sent",
    "remote_bytes_received",
    "num_runs",
    "pass_count",
    "fail_count",
}

RATIO_FIELDS = {
    "cache_miss_rate",
    "l1d_load_miss_rate",
    "parallel_efficiency",
    "parallel_efficiency_mean",
}

PCT_POINT_FIELDS = {
    "init_overhead_pct",
    "arts_e2e_cv_pct",
    "init_overhead_pct_mean",
}


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            f = float(text)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except ValueError:
            return None
    return None


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def _geomean(values: Iterable[Optional[float]]) -> Optional[float]:
    positives = [v for v in values if v is not None and v > 0]
    if not positives:
        return None
    return math.exp(sum(math.log(v) for v in positives) / len(positives))


def _status_text(value: Any) -> str:
    raw = value.value if hasattr(value, "value") else value
    if raw is None:
        return ""
    return str(raw).upper()


def _phase_name(value: Any) -> str:
    phase = str(value or "default").strip()
    return phase or "default"


def _counter_dir_from_artifacts(artifacts: Any) -> Optional[Path]:
    counter_dir: Optional[str] = None
    if isinstance(artifacts, dict):
        counter_dir = artifacts.get("counters_dir")
        if not counter_dir:
            counter_files = artifacts.get("counter_files") or []
            if counter_files:
                counter_dir = str(Path(counter_files[0]).parent)
    else:
        counter_dir = getattr(artifacts, "counters_dir", None)
        if not counter_dir:
            counter_files = getattr(artifacts, "counter_files", None) or []
            if counter_files:
                counter_dir = str(Path(counter_files[0]).parent)

    if not counter_dir:
        return None
    path = Path(counter_dir)
    return path if path.exists() else None


def _perf_dict(run: Any) -> Dict[str, Any]:
    if isinstance(run, dict):
        perf = run.get("perf_metrics")
    else:
        perf = getattr(run, "perf_metrics", None)

    if perf is None:
        return {}
    if isinstance(perf, dict):
        return perf
    if is_dataclass(perf):
        return asdict(perf)
    return {}


def _apply_derived_fields(row: Dict[str, Any]) -> None:
    speedup = _to_float(row.get("speedup"))
    threads = _to_float(row.get("threads"))
    arts_init = _to_float(row.get("arts_init_sec"))
    arts_e2e = _to_float(row.get("arts_e2e_sec"))

    num_edts_finished = _to_float(row.get("num_edts_finished"))
    num_edts_created = _to_float(row.get("num_edts_created"))
    end_to_end_time_ms = _to_float(row.get("end_to_end_time_ms"))
    edt_running_time_ms = _to_float(row.get("edt_running_time_ms"))
    memory_footprint_bytes = _to_float(row.get("memory_footprint_bytes"))
    remote_bytes_sent = _to_float(row.get("remote_bytes_sent"))
    remote_bytes_received = _to_float(row.get("remote_bytes_received"))

    row["parallel_efficiency"] = _safe_div(speedup, threads)

    init_overhead = _safe_div(arts_init, arts_e2e)
    row["init_overhead_pct"] = (
        init_overhead * 100.0 if init_overhead is not None else None
    )

    e2e_sec = _safe_div(end_to_end_time_ms, 1000.0)
    row["task_throughput"] = _safe_div(num_edts_finished, e2e_sec)

    avg_task_ms = _safe_div(edt_running_time_ms, num_edts_finished)
    row["avg_task_time_us"] = avg_task_ms * 1000.0 if avg_task_ms is not None else None

    row["memory_per_edt"] = _safe_div(memory_footprint_bytes, num_edts_created)

    if (
        num_edts_finished is None
        or num_edts_finished == 0
        or remote_bytes_sent is None
        or remote_bytes_received is None
    ):
        row["comm_bytes_per_edt"] = None
    else:
        row["comm_bytes_per_edt"] = (
            remote_bytes_sent + remote_bytes_received
        ) / num_edts_finished


def _empty_result_row() -> Dict[str, Any]:
    return {column: None for column in RESULTS_COLUMNS}


def _flatten_result_dataclass(result: BenchmarkResult) -> Dict[str, Any]:
    row = _empty_result_row()

    row.update(
        {
            "benchmark": result.name,
            "suite": result.suite,
            "size": result.size,
            "threads": result.config.arts_threads,
            "nodes": result.config.arts_nodes,
            "run": result.run_number,
            "run_phase": _phase_name(getattr(result, "run_phase", None)),
            "status": _status_text(result.run_arts.status),
            "speedup_basis": result.timing.speedup_basis,
            "arts_e2e_sec": result.timing.arts_e2e_sec,
            "omp_e2e_sec": result.timing.omp_e2e_sec,
            "arts_kernel_sec": result.timing.arts_kernel_sec,
            "omp_kernel_sec": result.timing.omp_kernel_sec,
            "arts_init_sec": result.timing.arts_init_sec,
            "omp_init_sec": result.timing.omp_init_sec,
            "arts_total_sec": result.timing.arts_total_sec,
            "omp_total_sec": result.timing.omp_total_sec,
            "speedup": result.timing.speedup,
        }
    )

    perf = _perf_dict(result.run_arts)
    row["cache_references"] = perf.get("cache_references")
    row["cache_misses"] = perf.get("cache_misses")
    row["cache_miss_rate"] = perf.get("cache_miss_rate")
    row["l1d_loads"] = perf.get("l1d_loads")
    row["l1d_load_misses"] = perf.get("l1d_load_misses")
    row["l1d_load_miss_rate"] = perf.get("l1d_load_miss_rate")

    counter_dir = _counter_dir_from_artifacts(result.artifacts)
    counters = parse_all_counters(counter_dir) if counter_dir else {}
    for field, counter_key in COUNTER_FIELD_MAP.items():
        row[field] = counters.get(counter_key)

    _apply_derived_fields(row)
    return row


def _build_summary_rows(result_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, Any, Any, Any, Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        key = (
            row.get("benchmark"),
            row.get("suite"),
            row.get("size"),
            row.get("threads"),
            row.get("nodes"),
            row.get("run_phase"),
        )
        grouped[key].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key in sorted(
        grouped.keys(),
        key=lambda k: (
            str(k[0]),
            str(k[1]),
            str(k[2]),
            int(k[3] or 0),
            int(k[4] or 0),
            str(k[5] or ""),
        ),
    ):
        benchmark, suite, size, threads, nodes, run_phase = key
        runs = grouped[key]

        def collect(field: str) -> List[float]:
            values = [_to_float(r.get(field)) for r in runs]
            return [v for v in values if v is not None]

        arts_e2e_values = collect("arts_e2e_sec")
        omp_e2e_values = collect("omp_e2e_sec")
        arts_kernel_values = collect("arts_kernel_sec")
        omp_kernel_values = collect("omp_kernel_sec")
        arts_init_values = collect("arts_init_sec")
        omp_init_values = collect("omp_init_sec")
        speedup_values = collect("speedup")
        efficiency_values = collect("parallel_efficiency")
        init_overhead_values = collect("init_overhead_pct")

        arts_e2e_mean, arts_e2e_std = _mean_std(arts_e2e_values)
        omp_e2e_mean, omp_e2e_std = _mean_std(omp_e2e_values)
        arts_kernel_mean, _ = _mean_std(arts_kernel_values)
        omp_kernel_mean, _ = _mean_std(omp_kernel_values)
        arts_init_mean, _ = _mean_std(arts_init_values)
        omp_init_mean, _ = _mean_std(omp_init_values)
        speedup_mean, speedup_std = _mean_std(speedup_values)
        efficiency_mean, _ = _mean_std(efficiency_values)
        init_overhead_mean, _ = _mean_std(init_overhead_values)

        if arts_e2e_mean is None or arts_e2e_std is None:
            arts_e2e_cv = None
        elif arts_e2e_mean == 0:
            arts_e2e_cv = 0.0
        else:
            arts_e2e_cv = (arts_e2e_std / arts_e2e_mean) * 100.0

        pass_count = sum(1 for r in runs if str(r.get("status", "")).upper() == "PASS")
        fail_count = len(runs) - pass_count

        summary_rows.append(
            {
                "benchmark": benchmark,
                "suite": suite,
                "size": size,
                "threads": threads,
                "nodes": nodes,
                "run_phase": run_phase,
                "num_runs": len(runs),
                "arts_e2e_mean": arts_e2e_mean,
                "arts_e2e_std": arts_e2e_std,
                "arts_e2e_cv_pct": arts_e2e_cv,
                "omp_e2e_mean": omp_e2e_mean,
                "omp_e2e_std": omp_e2e_std,
                "arts_kernel_mean": arts_kernel_mean,
                "omp_kernel_mean": omp_kernel_mean,
                "arts_init_mean": arts_init_mean,
                "omp_init_mean": omp_init_mean,
                "speedup_mean": speedup_mean,
                "speedup_std": speedup_std,
                "speedup_min": min(speedup_values) if speedup_values else None,
                "speedup_max": max(speedup_values) if speedup_values else None,
                "parallel_efficiency_mean": efficiency_mean,
                "init_overhead_pct_mean": init_overhead_mean,
                "pass_count": pass_count,
                "fail_count": fail_count,
            }
        )

    phases = sorted(
        {
            _phase_name(row.get("run_phase"))
            for row in summary_rows
            if row.get("benchmark") != "GEOMEAN"
        }
    )
    for phase in phases:
        phase_rows = [
            row
            for row in summary_rows
            if row.get("benchmark") != "GEOMEAN" and _phase_name(row.get("run_phase")) == phase
        ]
        geomean_speedup = _geomean(row.get("speedup_mean") for row in phase_rows)
        if geomean_speedup is None:
            continue
        footer = {column: None for column in SUMMARY_COLUMNS}
        footer["benchmark"] = "GEOMEAN"
        footer["run_phase"] = phase
        footer["speedup_mean"] = geomean_speedup
        summary_rows.append(footer)

    return summary_rows


def _build_comparison_sheet(
    workbook: Workbook,
    result_rows: List[Dict[str, Any]],
    steps: Optional[List["ExperimentStep"]] = None,
) -> None:
    all_phases = sorted({_phase_name(row.get("run_phase")) for row in result_rows})
    if len(all_phases) <= 1:
        return

    # Determine phase order: use steps definition order, fall back to sorted names.
    if steps:
        ordered_phases: List[str] = []
        for idx, step in enumerate(steps, start=1):
            name = _phase_name(getattr(step, "name", None) or f"step_{idx}")
            if name in all_phases and name not in ordered_phases:
                ordered_phases.append(name)
        for phase in all_phases:
            if phase not in ordered_phases:
                ordered_phases.append(phase)
    else:
        ordered_phases = list(all_phases)

    # Group result rows by (config_key, phase).
    counter_fields = [
        "num_edts_created",
        "num_edts_finished",
        "num_dbs_created",
        "memory_footprint_bytes",
        "remote_bytes_sent",
        "remote_bytes_received",
        "edt_running_time_ms",
        "initialization_time_ms",
        "end_to_end_time_ms",
        "task_throughput",
        "avg_task_time_us",
        "memory_per_edt",
        "comm_bytes_per_edt",
    ]
    perf_fields = [
        "cache_references",
        "cache_misses",
        "cache_miss_rate",
        "l1d_loads",
        "l1d_load_misses",
        "l1d_load_miss_rate",
    ]

    grouped: Dict[Tuple[Any, Any, Any, Any, Any, str], List[Dict[str, Any]]] = defaultdict(list)
    config_keys: Set[Tuple[Any, Any, Any, Any, Any]] = set()
    for row in result_rows:
        phase = _phase_name(row.get("run_phase"))
        config_key = (
            row.get("benchmark"),
            row.get("suite"),
            row.get("size"),
            row.get("threads"),
            row.get("nodes"),
        )
        config_keys.add(config_key)
        grouped[(*config_key, phase)].append(row)

    # Detect which phases have counter / perf data in any row.
    phases_with_counters: Set[str] = set()
    phases_with_perf: Set[str] = set()
    for row in result_rows:
        phase = _phase_name(row.get("run_phase"))
        if any(row.get(f) is not None for f in counter_fields):
            phases_with_counters.add(phase)
        if any(row.get(f) is not None for f in perf_fields):
            phases_with_perf.add(phase)

    # Build dynamic column list.
    columns: List[str] = ["benchmark", "suite", "size", "threads", "nodes"]
    for phase in ordered_phases:
        columns.extend([
            f"{phase}_arts_e2e_mean",
            f"{phase}_speedup_mean",
        ])
    for phase in ordered_phases:
        if phase in phases_with_counters:
            columns.extend(f"{phase}_{f}" for f in counter_fields)
    for phase in ordered_phases:
        if phase in phases_with_perf:
            columns.extend(f"{phase}_{f}" for f in perf_fields)

    def collect(rows: List[Dict[str, Any]], field: str) -> List[float]:
        values = [_to_float(r.get(field)) for r in rows]
        return [v for v in values if v is not None]

    ws = workbook.create_sheet(title="Comparison")
    ws.append(columns)
    _style_header(ws)
    ws.freeze_panes = "A2"

    for config_key in sorted(
        config_keys,
        key=lambda k: (
            str(k[0]),
            str(k[1]),
            str(k[2]),
            int(k[3] or 0),
            int(k[4] or 0),
        ),
    ):
        benchmark, suite, size, threads, nodes = config_key
        row_values: List[Any] = [benchmark, suite, size, threads, nodes]

        # Timing columns for each phase.
        for phase in ordered_phases:
            phase_rows = grouped.get((*config_key, phase), [])
            arts_e2e_mean, _ = _mean_std(collect(phase_rows, "arts_e2e_sec"))
            speedup_mean, _ = _mean_std(collect(phase_rows, "speedup"))
            row_values.extend([arts_e2e_mean, speedup_mean])

        # Counter columns for phases that have counter data.
        for phase in ordered_phases:
            if phase in phases_with_counters:
                phase_rows = grouped.get((*config_key, phase), [])
                for field in counter_fields:
                    row_values.append(_mean_std(collect(phase_rows, field))[0])

        # Perf columns for phases that have perf data.
        for phase in ordered_phases:
            if phase in phases_with_perf:
                phase_rows = grouped.get((*config_key, phase), [])
                for field in perf_fields:
                    row_values.append(_mean_std(collect(phase_rows, field))[0])

        ws.append(row_values)

    _apply_table_formats(ws, columns)
    _apply_speedup_rules(ws, columns)
    ws.auto_filter.ref = ws.dimensions
    _autosize_columns(ws)


def _build_overview_sheet(
    workbook: Workbook,
    result_rows: List[Dict[str, Any]],
    steps: Optional[List["ExperimentStep"]],
) -> None:
    if not steps:
        return

    ws = workbook.create_sheet(title="Overview")
    columns = [
        "step",
        "threads",
        "nodes",
        "runs",
        "perf",
        "profile",
        "benchmarks_run",
        "passed",
        "failed",
    ]
    ws.append(columns)
    _style_header(ws)
    ws.freeze_panes = "A2"

    by_phase: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        by_phase[_phase_name(row.get("run_phase"))].append(row)

    for idx, step in enumerate(steps, start=1):
        step_name = _phase_name(getattr(step, "name", None) or f"step_{idx}")
        phase_rows = by_phase.get(step_name, [])
        statuses = [str(r.get("status") or "").upper() for r in phase_rows]
        passed = sum(1 for s in statuses if s == "PASS")
        failed = sum(1 for s in statuses if s in {"FAIL", "CRASH", "TIMEOUT"})
        benchmarks = getattr(step, "benchmarks", None)
        benchmarks_run = ", ".join(benchmarks) if benchmarks else "all"

        ws.append(
            [
                step_name,
                getattr(step, "threads", None),
                getattr(step, "nodes", None),
                getattr(step, "runs", None),
                bool(getattr(step, "perf", False)),
                getattr(step, "profile", None),
                benchmarks_run,
                passed,
                failed,
            ]
        )

    _apply_table_formats(ws, columns)
    ws.auto_filter.ref = ws.dimensions
    _autosize_columns(ws)


def _append_table_sheet(workbook: Workbook, title: str, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    ws = workbook.create_sheet(title=title)
    ws.append(columns)
    _style_header(ws)
    ws.freeze_panes = "A2"

    for row in rows:
        ws.append([row.get(column) for column in columns])

    _apply_table_formats(ws, columns)
    _apply_speedup_rules(ws, columns)
    _apply_status_fill(ws, columns)
    ws.auto_filter.ref = ws.dimensions
    _autosize_columns(ws)


def _autosize_columns(worksheet: Any, max_width: int = 48) -> None:
    for column_cells in worksheet.columns:
        values = [cell.value for cell in column_cells if cell.value is not None]
        if not values:
            continue
        width = max(len(str(v)) for v in values) + 2
        width = min(max_width, max(10, width))
        worksheet.column_dimensions[column_cells[0].column_letter].width = width


def _style_header(worksheet: Any) -> None:
    if Font is None:
        return
    fill = PatternFill(fill_type="solid", fgColor="E7EDF7") if PatternFill is not None else None
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
        if fill is not None:
            cell.fill = fill


TIME_SUFFIXES = {"_e2e", "_e2e_mean", "_e2e_std", "_kernel", "_kernel_mean", "_init", "_init_mean"}


def _classify_format(field: str) -> Optional[str]:
    """Return number format for a field, using suffix matching for step-prefixed columns."""
    for int_field in INT_FIELDS:
        if field == int_field or field.endswith(f"_{int_field}"):
            return "#,##0"
    for ratio_field in RATIO_FIELDS:
        if field == ratio_field or field.endswith(f"_{ratio_field}"):
            return "0.00%"
    for pct_field in PCT_POINT_FIELDS:
        if field == pct_field or field.endswith(f"_{pct_field}"):
            return "0.00"
    for suffix in TIME_SUFFIXES:
        if field == suffix.lstrip("_") or field.endswith(suffix):
            return "0.000"
    return None


def _apply_table_formats(worksheet: Any, columns: List[str]) -> None:
    max_row = worksheet.max_row
    if max_row < 2:
        return

    for idx, field in enumerate(columns, start=1):
        fmt = _classify_format(field)
        for row in range(2, max_row + 1):
            cell = worksheet.cell(row=row, column=idx)
            value = cell.value
            if value is None or value == "":
                continue

            if fmt is not None:
                cell.number_format = fmt
            elif isinstance(value, (float, int)):
                cell.number_format = "0.000000"


def _apply_speedup_rules(worksheet: Any, columns: List[str]) -> None:
    if CellIsRule is None or PatternFill is None:
        return

    speedup_cols = [
        idx for idx, name in enumerate(columns, start=1)
        if "speedup" in name and name not in {"peak_speedup"}
    ]
    if not speedup_cols or worksheet.max_row < 2:
        return

    green_fill = PatternFill(fill_type="solid", fgColor="E7F5E7")
    red_fill = PatternFill(fill_type="solid", fgColor="FDECEC")

    for idx in speedup_cols:
        col = worksheet.cell(row=1, column=idx).column_letter
        rng = f"{col}2:{col}{worksheet.max_row}"
        worksheet.conditional_formatting.add(
            rng, CellIsRule(operator="greaterThanOrEqual", formula=["1"], fill=green_fill)
        )
        worksheet.conditional_formatting.add(
            rng, CellIsRule(operator="lessThan", formula=["1"], fill=red_fill)
        )


def _apply_status_fill(worksheet: Any, columns: List[str]) -> None:
    if PatternFill is None:
        return
    if "status" not in columns or worksheet.max_row < 2:
        return

    idx = columns.index("status") + 1
    pass_fill = PatternFill(fill_type="solid", fgColor="E7F5E7")
    fail_fill = PatternFill(fill_type="solid", fgColor="FDECEC")
    warn_fill = PatternFill(fill_type="solid", fgColor="FFF4D6")

    for row in range(2, worksheet.max_row + 1):
        cell = worksheet.cell(row=row, column=idx)
        text = str(cell.value or "").upper()
        if not text:
            continue
        if text == "PASS":
            cell.fill = pass_fill
        elif text in {"FAIL", "CRASH", "TIMEOUT"}:
            cell.fill = fail_fill
        elif text == "SKIP":
            cell.fill = warn_fill


def _build_scaling_sheet(workbook: Workbook, summary_rows: List[Dict[str, Any]]) -> None:
    ws = workbook.create_sheet(title="Scaling")

    data_rows = [r for r in summary_rows if r.get("benchmark") != "GEOMEAN"]
    phase_threads: Dict[str, Set[int]] = defaultdict(set)
    phase_nodes: Dict[str, Set[int]] = defaultdict(set)
    for row in data_rows:
        phase = _phase_name(row.get("run_phase"))
        threads_value = _to_float(row.get("threads"))
        nodes_value = _to_float(row.get("nodes"))
        if threads_value is not None:
            phase_threads[phase].add(int(threads_value))
        if nodes_value is not None:
            phase_nodes[phase].add(int(nodes_value))

    sweep_phases = sorted(
        {
            phase
            for phase in set(list(phase_threads.keys()) + list(phase_nodes.keys()))
            if len(phase_threads.get(phase, set())) > 1 or len(phase_nodes.get(phase, set())) > 1
        }
    )
    has_thread_sweep = any(len(values) > 1 for values in phase_threads.values())
    has_node_sweep = any(len(values) > 1 for values in phase_nodes.values())
    if sweep_phases:
        data_rows = [
            row for row in data_rows if _phase_name(row.get("run_phase")) in sweep_phases
        ]

    threads = sorted({int(r["threads"]) for r in data_rows if r.get("threads") is not None})
    nodes = sorted({int(r["nodes"]) for r in data_rows if r.get("nodes") is not None})

    if not has_thread_sweep and not has_node_sweep:
        ws["A1"] = "No thread/node sweep detected. Scaling view is empty for this run."
        _autosize_columns(ws)
        return

    indexed: Dict[Tuple[str, str, str, str], Dict[Tuple[int, int], Dict[str, Any]]] = defaultdict(dict)
    for row in data_rows:
        benchmark = str(row.get("benchmark") or "")
        suite = str(row.get("suite") or "")
        size = str(row.get("size") or "")
        run_phase = _phase_name(row.get("run_phase"))
        threads_value = int(row.get("threads") or 0)
        nodes_value = int(row.get("nodes") or 0)
        indexed[(benchmark, suite, size, run_phase)][(threads_value, nodes_value)] = row

    headers: List[str]
    matrix_rows: List[List[Any]] = []

    if has_thread_sweep and not has_node_sweep:
        headers = ["benchmark", "suite", "run_phase"]
        for t in threads:
            headers.extend([f"T{t}_arts_e2e", f"T{t}_speedup", f"T{t}_efficiency"])
        headers.extend(["peak_speedup", "peak_threads", "arts_self_scaling"])

        for (benchmark, suite, _size, run_phase), configs in sorted(indexed.items()):
            row_values: List[Any] = [benchmark, suite, run_phase]
            speedup_candidates: List[Tuple[float, int]] = []
            phase_nodes = sorted({n for (_t, n) in configs.keys()})
            fixed_node = phase_nodes[0] if phase_nodes else nodes[0]

            for t in threads:
                entry = configs.get((t, fixed_node), {})
                arts_e2e = entry.get("arts_e2e_mean")
                speedup = entry.get("speedup_mean")
                efficiency = entry.get("parallel_efficiency_mean")
                row_values.extend([arts_e2e, speedup, efficiency])
                speedup_f = _to_float(speedup)
                if speedup_f is not None and speedup_f > 0:
                    speedup_candidates.append((speedup_f, t))

            peak_speedup = max(speedup_candidates, key=lambda x: x[0])[0] if speedup_candidates else None
            peak_threads = max(speedup_candidates, key=lambda x: x[0])[1] if speedup_candidates else None

            base_t1 = configs.get((1, fixed_node), {}).get("arts_e2e_mean")
            max_t = max(threads)
            max_t_value = configs.get((max_t, fixed_node), {}).get("arts_e2e_mean")
            arts_self_scaling = _safe_div(_to_float(base_t1), _to_float(max_t_value))

            row_values.extend([peak_speedup, peak_threads, arts_self_scaling])
            matrix_rows.append(row_values)

    elif has_node_sweep and not has_thread_sweep:
        headers = ["benchmark", "suite", "run_phase"]
        for n in nodes:
            headers.extend([f"N{n}_arts_e2e", f"N{n}_speedup"])
            if n != 1:
                headers.append(f"N{n}_scaling_vs_N1")
        headers.extend(["peak_speedup", "peak_nodes"])

        for (benchmark, suite, _size, run_phase), configs in sorted(indexed.items()):
            row_values = [benchmark, suite, run_phase]
            speedup_candidates: List[Tuple[float, int]] = []
            phase_threads = sorted({t for (t, _n) in configs.keys()})
            fixed_thread = phase_threads[0] if phase_threads else threads[0]
            baseline_e2e = _to_float(configs.get((fixed_thread, 1), {}).get("arts_e2e_mean"))

            for n in nodes:
                entry = configs.get((fixed_thread, n), {})
                arts_e2e = entry.get("arts_e2e_mean")
                speedup = entry.get("speedup_mean")
                row_values.extend([arts_e2e, speedup])

                speedup_f = _to_float(speedup)
                if speedup_f is not None and speedup_f > 0:
                    speedup_candidates.append((speedup_f, n))

                if n != 1:
                    scale = _safe_div(baseline_e2e, _to_float(arts_e2e))
                    row_values.append(scale)

            peak_speedup = max(speedup_candidates, key=lambda x: x[0])[0] if speedup_candidates else None
            peak_nodes = max(speedup_candidates, key=lambda x: x[0])[1] if speedup_candidates else None
            row_values.extend([peak_speedup, peak_nodes])
            matrix_rows.append(row_values)

    else:
        headers = ["benchmark", "suite", "run_phase"]
        for t in threads:
            for n in nodes:
                prefix = f"T{t}N{n}"
                headers.extend(
                    [
                        f"{prefix}_arts_e2e",
                        f"{prefix}_speedup",
                        f"{prefix}_efficiency",
                    ]
                )
        headers.extend(["peak_speedup", "peak_threads", "peak_nodes"])

        for (benchmark, suite, _size, run_phase), configs in sorted(indexed.items()):
            row_values = [benchmark, suite, run_phase]
            peak_tuple: Optional[Tuple[float, int, int]] = None

            for t in threads:
                for n in nodes:
                    entry = configs.get((t, n), {})
                    arts_e2e = entry.get("arts_e2e_mean")
                    speedup = entry.get("speedup_mean")
                    efficiency = entry.get("parallel_efficiency_mean")
                    row_values.extend([arts_e2e, speedup, efficiency])

                    speedup_f = _to_float(speedup)
                    if speedup_f is not None and speedup_f > 0:
                        candidate = (speedup_f, t, n)
                        if peak_tuple is None or speedup_f > peak_tuple[0]:
                            peak_tuple = candidate

            if peak_tuple is None:
                row_values.extend([None, None, None])
            else:
                row_values.extend([peak_tuple[0], peak_tuple[1], peak_tuple[2]])

            matrix_rows.append(row_values)

    ws.append(headers)
    _style_header(ws)
    ws.freeze_panes = "A2"

    for values in matrix_rows:
        ws.append(values)

    if matrix_rows:
        phase_groups: Dict[str, List[List[Any]]] = defaultdict(list)
        for row in matrix_rows:
            phase_groups[row[2]].append(row)

        for phase in sorted(phase_groups.keys()):
            rows = phase_groups[phase]
            geomean_row: List[Any] = ["GEOMEAN", "", phase]
            for idx, header in enumerate(headers[3:], start=3):
                if header.startswith("peak_"):
                    geomean_row.append(None)
                    continue
                values = [_to_float(r[idx]) for r in rows]
                values = [v for v in values if v is not None and v > 0]
                geomean_row.append(_geomean(values))
            ws.append(geomean_row)

    _apply_table_formats(ws, headers)
    _apply_speedup_rules(ws, headers)
    ws.auto_filter.ref = ws.dimensions
    _autosize_columns(ws)


def _load_results_metadata(experiment_dir: Path) -> Dict[str, Any]:
    results_json = experiment_dir / "results.json"
    if not results_json.exists():
        return {}

    try:
        with open(results_json) as f:
            data = json.load(f)
        return data.get("metadata", {})
    except (OSError, json.JSONDecodeError, TypeError, AttributeError):
        return {}


def _load_manifest_command(experiment_dir: Path) -> Optional[str]:
    manifest_json = experiment_dir / "manifest.json"
    if not manifest_json.exists():
        return None

    try:
        with open(manifest_json) as f:
            data = json.load(f)
        command = data.get("command")
        if command is None:
            return None
        return str(command)
    except (OSError, json.JSONDecodeError, TypeError, AttributeError):
        return None


def _metadata_rows(
    metadata: Dict[str, Any],
    command: Optional[str],
    report_summary: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Any]]:
    rows: List[Tuple[str, Any]] = []

    def add(key: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (dict, list)):
            rows.append((key, json.dumps(value, sort_keys=True)))
        else:
            rows.append((key, value))

    add("timestamp", metadata.get("timestamp"))
    add("hostname", metadata.get("hostname"))
    add("size", metadata.get("size"))
    add("duration_seconds", metadata.get("total_duration_seconds"))
    add("runs_per_config", metadata.get("runs_per_config"))

    add("thread_sweep", metadata.get("thread_sweep"))
    add("node_sweep", metadata.get("node_sweep"))
    add("fixed_threads", metadata.get("fixed_threads"))
    add("fixed_nodes", metadata.get("fixed_nodes"))
    add("launcher", metadata.get("launcher"))
    add("weak_scaling", metadata.get("weak_scaling"))

    repro = metadata.get("reproducibility") if isinstance(metadata, dict) else {}
    if isinstance(repro, dict):
        commits = repro.get("git_commits", {})
        if isinstance(commits, dict):
            add("git_carts", commits.get("carts"))
            add("git_arts", commits.get("arts"))
            add("git_carts_benchmarks", commits.get("carts_benchmarks"))

        compilers = repro.get("compilers", {})
        if isinstance(compilers, dict):
            add("compiler_clang", compilers.get("clang"))
            add("compiler_gcc", compilers.get("gcc"))

        cpu = repro.get("cpu", {})
        if isinstance(cpu, dict):
            add("cpu_model", cpu.get("model"))
            add("cpu_cores", cpu.get("cores"))
            add("cpu_physical_cores", cpu.get("physical_cores"))

    if isinstance(report_summary, dict):
        add("report_generated_at", report_summary.get("generated_at"))
        add("report_total_rows", report_summary.get("total_rows"))
        add("report_pass_count", report_summary.get("pass_count"))
        add("report_fail_count", report_summary.get("fail_count"))
        add("report_skipped_count", report_summary.get("skip_count"))
        add("report_geomean_speedup", report_summary.get("geomean_speedup"))
        add("report_rows_with_counters", report_summary.get("rows_with_counters"))
        add("report_rows_with_perf", report_summary.get("rows_with_perf"))

    add("command", command)
    return rows


def _append_metadata_sheet(
    workbook: Workbook,
    metadata: Dict[str, Any],
    command: Optional[str],
    report_summary: Optional[Dict[str, Any]] = None,
) -> None:
    ws = workbook.create_sheet(title="Metadata")
    ws.append(["key", "value"])
    _style_header(ws)

    for key, value in _metadata_rows(metadata, command, report_summary=report_summary):
        ws.append([key, value])

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    _autosize_columns(ws)


def _build_report_summary(result_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    statuses = [str(r.get("status") or "").upper() for r in result_rows]
    pass_count = sum(1 for s in statuses if s == "PASS")
    fail_count = sum(1 for s in statuses if s in {"FAIL", "CRASH", "TIMEOUT"})
    skip_count = sum(1 for s in statuses if s == "SKIP")

    phases = sorted({_phase_name(r.get("run_phase")) for r in result_rows})
    geomean_speedup: Dict[str, Optional[float]] = {}
    for phase in phases:
        phase_speedups = [
            _to_float(r.get("speedup"))
            for r in result_rows
            if _phase_name(r.get("run_phase")) == phase
        ]
        geomean_speedup[phase] = _geomean(
            v for v in phase_speedups if v is not None and v > 0
        )

    rows_with_counters = sum(
        1
        for r in result_rows
        if any(r.get(field) is not None for field in COUNTER_FIELD_MAP.keys())
    )
    rows_with_perf = sum(
        1
        for r in result_rows
        if any(
            r.get(field) is not None
            for field in (
                "cache_references",
                "cache_misses",
                "cache_miss_rate",
                "l1d_loads",
                "l1d_load_misses",
                "l1d_load_miss_rate",
            )
        )
    )

    return {
        "generated_at": datetime.now().isoformat(),
        "total_rows": len(result_rows),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "skip_count": skip_count,
        "geomean_speedup": geomean_speedup,
        "rows_with_counters": rows_with_counters,
        "rows_with_perf": rows_with_perf,
    }


def _write_report(
    result_rows: List[Dict[str, Any]],
    experiment_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
    command: Optional[str] = None,
    steps: Optional[List["ExperimentStep"]] = None,
) -> Optional[Path]:
    if Workbook is None:
        return None

    report_path = experiment_dir / "report.xlsx"
    metadata = metadata or _load_results_metadata(experiment_dir)

    workbook = Workbook()
    workbook.remove(workbook.active)

    _build_overview_sheet(workbook, result_rows, steps)
    _append_table_sheet(workbook, "Results", RESULTS_COLUMNS, result_rows)

    summary_rows = _build_summary_rows(result_rows)
    _append_table_sheet(workbook, "Summary", SUMMARY_COLUMNS, summary_rows)
    _build_scaling_sheet(workbook, summary_rows)
    _build_comparison_sheet(workbook, result_rows, steps=steps)

    effective_command = command or _load_manifest_command(experiment_dir)
    report_summary = _build_report_summary(result_rows)
    _append_metadata_sheet(
        workbook,
        metadata,
        effective_command,
        report_summary=report_summary,
    )

    workbook.save(report_path)
    return report_path


def generate_report(
    results: List["BenchmarkResult"],
    experiment_dir: Path,
    quiet: bool = False,
    steps: Optional[List["ExperimentStep"]] = None,
) -> Optional[Path]:
    """Generate report.xlsx from in-memory benchmark results."""
    del quiet  # kept for compatibility with caller API

    result_rows = [_flatten_result_dataclass(result) for result in results]

    command = "carts benchmarks " + " ".join(sys.argv[1:])
    return _write_report(
        result_rows,
        Path(experiment_dir),
        command=command,
        steps=steps,
    )
