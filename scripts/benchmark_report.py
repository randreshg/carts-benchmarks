"""Excel report generation for benchmark runs."""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple

from benchmark_common import parse_all_counters, parse_perf_csv

if TYPE_CHECKING:
    from benchmark_models import BenchmarkResult, ExperimentStep

try:
    from openpyxl import Workbook
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import Font
    from openpyxl.styles import PatternFill
    from openpyxl.worksheet.table import Table, TableStyleInfo
except ImportError:  # pragma: no cover - runtime dependency check
    Workbook = None  # type: ignore[assignment]
    CellIsRule = None  # type: ignore[assignment]
    Font = None  # type: ignore[assignment]
    PatternFill = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    TableStyleInfo = None  # type: ignore[assignment]


RESULTS_COLUMNS = [
    "benchmark",
    "suite",
    "size",
    "threads",
    "nodes",
    "run",
    "run_phase",
    "compile_args",
    "profile",
    "perf_enabled",
    "perf_interval",
    "status",
    "status_detail",
    "verified",
    "has_counters",
    "has_perf",
    "verification_note",
    "arts_checksum",
    "omp_checksum",
    "reference_checksum",
    "reference_source",
    "runtime_warning",
    "slurm_job_id",
    "slurm_state",
    "slurm_exit_code",
    "slurm_nodelist",
    "srun_error_count",
    "broken_pipe_count",
    "counter_timeout_warnings",
    "remote_send_hard_timeout_count",
    "connection_refused_count",
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
    "counter_source",
    "counter_files_found",
    "counter_files_valid",
    "counter_expected_nodes",
    "counter_complete",
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
    "artifact_run_dir",
    "artifact_run_config",
    "artifact_result_json",
    "artifact_slurm_out",
    "artifact_slurm_err",
    "artifact_build_dir",
    "artifact_counter_dir",
    "artifact_perf_dir",
    "artifact_perf_file_count",
]

SUMMARY_COLUMNS = [
    "benchmark",
    "suite",
    "size",
    "threads",
    "nodes",
    "run_phase",
    "compile_args",
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
    "verified_count",
    "pass_count",
    "fail_count",
    "warn_count",
    "rows_with_counters",
    "rows_with_perf",
]

NODE_COUNTER_SUMMARY_COLUMNS = [
    "benchmark",
    "suite",
    "size",
    "threads",
    "nodes",
    "run",
    "run_phase",
    "compile_args",
    "counter_name",
    "capture_mode",
    "capture_level",
    "reduce_method",
    "metric_kind",
    "nodes_reported",
    "metric_total",
    "metric_mean",
    "metric_min",
    "metric_max",
    "metric_std",
    "metric_cv_pct",
    "max_to_min_ratio",
]

NODE_COUNTER_COLUMNS = [
    "benchmark",
    "suite",
    "size",
    "threads",
    "nodes",
    "run",
    "run_phase",
    "compile_args",
    "node_id",
    "total_threads",
    "counter_name",
    "capture_mode",
    "capture_level",
    "reduce_method",
    "metric_kind",
    "metric_value",
    "value",
    "value_ms",
    "history_points",
    "nodes_reported",
    "metric_total",
    "metric_mean",
    "node_fraction",
    "imbalance_vs_mean_pct",
    "counter_file",
]

PERF_FILE_COLUMNS = [
    "benchmark",
    "suite",
    "size",
    "threads",
    "nodes",
    "run",
    "run_phase",
    "compile_args",
    "perf_role",
    "perf_rank",
    "perf_file",
    "cache_references",
    "cache_misses",
    "cache_miss_rate",
    "l1d_loads",
    "l1d_load_misses",
    "l1d_load_miss_rate",
]

PATH_LIKE_FIELDS = {
    "reference_source",
    "artifact_run_dir",
    "artifact_run_config",
    "artifact_result_json",
    "artifact_slurm_out",
    "artifact_slurm_err",
    "artifact_build_dir",
    "artifact_counter_dir",
    "artifact_perf_dir",
    "counter_file",
    "perf_file",
}

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
    "srun_error_count",
    "broken_pipe_count",
    "counter_timeout_warnings",
    "remote_send_hard_timeout_count",
    "connection_refused_count",
    "counter_files_found",
    "counter_files_valid",
    "counter_expected_nodes",
    "artifact_perf_file_count",
    "memory_footprint_bytes",
    "remote_bytes_sent",
    "remote_bytes_received",
    "num_runs",
    "pass_count",
    "fail_count",
    "warn_count",
    "verified_count",
    "nodes_reported",
    "node_id",
    "total_threads",
    "history_points",
}

RATIO_FIELDS = {
    "cache_miss_rate",
    "l1d_load_miss_rate",
    "parallel_efficiency",
    "parallel_efficiency_mean",
    "node_fraction",
}

PCT_POINT_FIELDS = {
    "init_overhead_pct",
    "arts_e2e_cv_pct",
    "init_overhead_pct_mean",
    "imbalance_vs_mean_pct",
    "metric_cv_pct",
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
        counter_dir = artifacts.get("counters_dir") or artifacts.get("counter_dir")
        if not counter_dir:
            counter_files = artifacts.get("counter_files") or []
            if counter_files:
                counter_dir = str(Path(counter_files[0]).parent)
    else:
        counter_dir = getattr(artifacts, "counters_dir", None) or getattr(artifacts, "counter_dir", None)
        if not counter_dir:
            counter_files = getattr(artifacts, "counter_files", None) or []
            if counter_files:
                counter_dir = str(Path(counter_files[0]).parent)

    if not counter_dir:
        return None
    path = Path(counter_dir)
    return path if path.exists() else None


def _counter_value(entry: Any) -> Optional[float]:
    if not isinstance(entry, dict):
        return None
    raw = entry.get("value_ms")
    if raw is None:
        raw = entry.get("value")
    return _to_float(raw)


def _count_valid_node_counter_files(counter_dir: Path) -> Tuple[int, int]:
    files = sorted(counter_dir.glob("n*.json"))
    valid = 0
    for path in files:
        try:
            with open(path) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            valid += 1
    return len(files), valid


def _aggregate_node_counter_files(counter_dir: Path) -> Tuple[Dict[str, float], int, int]:
    aggregated: Dict[str, float] = {}
    files = sorted(counter_dir.glob("n*.json"))
    valid = 0
    for path in files:
        try:
            with open(path) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        counters = payload.get("counters")
        if not isinstance(counters, dict):
            continue

        valid += 1
        for name, entry in counters.items():
            value = _counter_value(entry)
            if value is None:
                continue
            aggregated[name] = aggregated.get(name, 0.0) + value
    return aggregated, len(files), valid


def _collect_counters(
    counter_dir: Optional[Path],
    expected_nodes: Optional[int],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "counter_source": None,
        "counter_files_found": 0,
        "counter_files_valid": 0,
        "counter_expected_nodes": expected_nodes,
        "counter_complete": None,
    }
    if counter_dir is None:
        return {}, meta

    cluster_counters = parse_all_counters(counter_dir)
    node_counters, found, valid = _aggregate_node_counter_files(counter_dir)
    meta["counter_files_found"] = found
    meta["counter_files_valid"] = valid

    combined_counters = dict(cluster_counters)
    node_added_fields = False
    for name, value in node_counters.items():
        if name in combined_counters:
            continue
        combined_counters[name] = value
        node_added_fields = True

    if cluster_counters and node_added_fields:
        meta["counter_source"] = "cluster+node"
        if expected_nodes is not None:
            meta["counter_complete"] = valid >= expected_nodes
    elif cluster_counters:
        meta["counter_source"] = "cluster"
        if expected_nodes is not None:
            meta["counter_complete"] = True
    elif node_counters:
        meta["counter_source"] = "node_fallback"
        if expected_nodes is not None:
            meta["counter_complete"] = valid >= expected_nodes

    return combined_counters, meta


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


def _path_if_exists(value: Any) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    return path if path.exists() else None


def _base_result_identity(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "benchmark": row.get("benchmark"),
        "suite": row.get("suite"),
        "size": row.get("size"),
        "threads": row.get("threads"),
        "nodes": row.get("nodes"),
        "run": row.get("run"),
        "run_phase": row.get("run_phase"),
        "compile_args": row.get("compile_args"),
    }


def _build_node_counter_rows(result_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for result_row in result_rows:
        counter_dir = _path_if_exists(result_row.get("artifact_counter_dir"))
        if counter_dir is None:
            continue

        for counter_file in sorted(counter_dir.glob("n*.json")):
            try:
                payload = json.loads(counter_file.read_text())
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            counters = payload.get("counters")
            if not isinstance(counters, dict):
                continue

            node_id = metadata.get("nodeId")
            if node_id is None:
                match = re.match(r"n(\d+)\.json$", counter_file.name)
                node_id = int(match.group(1)) if match else None

            total_threads = metadata.get("totalThreads")

            for counter_name, entry in counters.items():
                if not isinstance(entry, dict):
                    continue
                capture_level = str(entry.get("captureLevel") or "")
                if capture_level.upper() == "CLUSTER":
                    continue

                value = _to_float(entry.get("value"))
                value_ms = _to_float(entry.get("value_ms"))
                metric_value = value_ms if value_ms is not None else value
                metric_kind = "value_ms" if value_ms is not None else "value"
                if metric_value is None:
                    continue

                history = entry.get("captureHistory")
                history_points = len(history) if isinstance(history, list) else 0

                row = _base_result_identity(result_row)
                row.update(
                    {
                        "node_id": node_id,
                        "total_threads": total_threads,
                        "counter_name": counter_name,
                        "capture_mode": entry.get("captureMode"),
                        "capture_level": entry.get("captureLevel"),
                        "reduce_method": entry.get("reduceMethod"),
                        "metric_kind": metric_kind,
                        "metric_value": metric_value,
                        "value": value,
                        "value_ms": value_ms,
                        "history_points": history_points,
                        "counter_file": str(counter_file),
                    }
                )
                rows.append(row)

    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("benchmark"),
            row.get("suite"),
            row.get("size"),
            row.get("threads"),
            row.get("nodes"),
            row.get("run"),
            row.get("run_phase"),
            row.get("compile_args"),
            row.get("counter_name"),
            row.get("capture_mode"),
            row.get("capture_level"),
            row.get("reduce_method"),
            row.get("metric_kind"),
        )
        grouped[key].append(row)

    for grouped_rows in grouped.values():
        values = [_to_float(row.get("metric_value")) for row in grouped_rows]
        numeric_values = [value for value in values if value is not None]
        if not numeric_values:
            continue
        total_value = sum(numeric_values)
        mean_value = mean(numeric_values)
        min_value = min(numeric_values)
        max_value = max(numeric_values)
        for row in grouped_rows:
            metric_value = _to_float(row.get("metric_value"))
            row["nodes_reported"] = len(grouped_rows)
            row["metric_total"] = total_value
            row["metric_mean"] = mean_value
            row["node_fraction"] = _safe_div(metric_value, total_value)
            row["imbalance_vs_mean_pct"] = (
                ((metric_value - mean_value) / mean_value) * 100.0
                if metric_value is not None and mean_value not in (None, 0)
                else None
            )

    return rows


def _build_node_counter_summary_rows(result_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    detail_rows = _build_node_counter_rows(result_rows)
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        key = (
            row.get("benchmark"),
            row.get("suite"),
            row.get("size"),
            row.get("threads"),
            row.get("nodes"),
            row.get("run"),
            row.get("run_phase"),
            row.get("compile_args"),
            row.get("counter_name"),
            row.get("capture_mode"),
            row.get("capture_level"),
            row.get("reduce_method"),
            row.get("metric_kind"),
        )
        grouped[key].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped.keys(), key=lambda item: tuple(str(part) for part in item)):
        values = [
            _to_float(row.get("metric_value"))
            for row in grouped[key]
            if _to_float(row.get("metric_value")) is not None
        ]
        if not values:
            continue

        metric_mean, metric_std = _mean_std(values)
        metric_min = min(values)
        metric_max = max(values)
        metric_cv_pct = None
        if metric_mean not in (None, 0) and metric_std is not None:
            metric_cv_pct = (metric_std / metric_mean) * 100.0

        max_to_min_ratio = None
        if metric_min != 0:
            max_to_min_ratio = metric_max / metric_min

        (
            benchmark,
            suite,
            size,
            threads,
            nodes,
            run,
            run_phase,
            compile_args,
            counter_name,
            capture_mode,
            capture_level,
            reduce_method,
            metric_kind,
        ) = key

        summary_rows.append(
            {
                "benchmark": benchmark,
                "suite": suite,
                "size": size,
                "threads": threads,
                "nodes": nodes,
                "run": run,
                "run_phase": run_phase,
                "compile_args": compile_args,
                "counter_name": counter_name,
                "capture_mode": capture_mode,
                "capture_level": capture_level,
                "reduce_method": reduce_method,
                "metric_kind": metric_kind,
                "nodes_reported": len(values),
                "metric_total": sum(values),
                "metric_mean": metric_mean,
                "metric_min": metric_min,
                "metric_max": metric_max,
                "metric_std": metric_std,
                "metric_cv_pct": metric_cv_pct,
                "max_to_min_ratio": max_to_min_ratio,
            }
        )

    return summary_rows


def _build_perf_file_rows(result_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for result_row in result_rows:
        perf_dir = _path_if_exists(result_row.get("artifact_perf_dir"))
        if perf_dir is None:
            continue

        perf_files = sorted(perf_dir.glob("arts_node_*.csv"))
        omp_file = perf_dir / "omp.csv"
        if omp_file.exists():
            perf_files.append(omp_file)

        for perf_file in perf_files:
            perf_metrics = parse_perf_csv(perf_file)
            row = _base_result_identity(result_row)
            rank_match = re.match(r"arts_node_(\d+)\.csv$", perf_file.name)
            perf_role = "omp" if perf_file.name == "omp.csv" else "arts"
            perf_rank = int(rank_match.group(1)) if rank_match else None
            row.update(
                {
                    "perf_role": perf_role,
                    "perf_rank": perf_rank,
                    "perf_file": str(perf_file),
                    "cache_references": perf_metrics.get("cache_references") if perf_metrics else None,
                    "cache_misses": perf_metrics.get("cache_misses") if perf_metrics else None,
                    "cache_miss_rate": perf_metrics.get("cache_miss_rate") if perf_metrics else None,
                    "l1d_loads": perf_metrics.get("l1d_loads") if perf_metrics else None,
                    "l1d_load_misses": perf_metrics.get("l1d_load_misses") if perf_metrics else None,
                    "l1d_load_miss_rate": perf_metrics.get("l1d_load_miss_rate") if perf_metrics else None,
                }
            )
            rows.append(row)

    return rows


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


def _verification_state(
    status: Any,
    verification_note: Any,
    arts_checksum: Any,
    omp_checksum: Any,
    reference_checksum: Any,
) -> Optional[bool]:
    status_text = _status_text(status)
    if status_text != "PASS":
        return False

    note_text = str(verification_note or "").strip().lower()
    if "mismatch" in note_text or "failed" in note_text:
        return False

    if arts_checksum is not None and (omp_checksum is not None or reference_checksum is not None):
        return True

    return None


def _empty_result_row() -> Dict[str, Any]:
    return {column: None for column in RESULTS_COLUMNS}


def _flatten_result_dataclass(result: BenchmarkResult) -> Dict[str, Any]:
    row = _empty_result_row()
    verification = result.verification
    artifacts = result.artifacts
    arts_perf_path = result.run_arts.perf_csv_path
    arts_perf_dir = str(Path(arts_perf_path).parent) if arts_perf_path else None

    row.update(
        {
            "benchmark": result.name,
            "suite": result.suite,
            "size": result.size,
            "threads": result.config.arts_threads,
            "nodes": result.config.arts_nodes,
            "run": result.run_number,
            "run_phase": _phase_name(getattr(result, "run_phase", None)),
            "compile_args": getattr(result, "compile_args", None),
            "profile": None,
            "perf_enabled": bool(result.run_arts.perf_metrics or arts_perf_path),
            "perf_interval": None,
            "status": _status_text(result.run_arts.status),
            "status_detail": _status_text(result.run_arts.status),
            "verified": _verification_state(
                result.run_arts.status,
                verification.note,
                verification.arts_checksum,
                verification.omp_checksum,
                verification.reference_checksum,
            ),
            "verification_note": verification.note,
            "arts_checksum": verification.arts_checksum,
            "omp_checksum": verification.omp_checksum,
            "reference_checksum": verification.reference_checksum,
            "reference_source": verification.reference_source,
            "runtime_warning": False,
            "slurm_job_id": None,
            "slurm_state": None,
            "slurm_exit_code": None,
            "slurm_nodelist": None,
            "srun_error_count": None,
            "broken_pipe_count": None,
            "counter_timeout_warnings": None,
            "remote_send_hard_timeout_count": None,
            "connection_refused_count": None,
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
            "artifact_run_dir": artifacts.run_dir,
            "artifact_run_config": str(Path(artifacts.run_dir) / "run_config.json") if artifacts.run_dir else None,
            "artifact_result_json": str(Path(artifacts.run_dir) / "result.json") if artifacts.run_dir else None,
            "artifact_slurm_out": str(Path(artifacts.run_dir) / "slurm.out") if artifacts.run_dir else None,
            "artifact_slurm_err": str(Path(artifacts.run_dir) / "slurm.err") if artifacts.run_dir else None,
            "artifact_build_dir": artifacts.build_dir,
            "artifact_counter_dir": artifacts.counters_dir,
            "artifact_perf_dir": arts_perf_dir,
            "artifact_perf_file_count": 1 if arts_perf_path else 0,
        }
    )

    perf = _perf_dict(result.run_arts)
    row["cache_references"] = perf.get("cache_references")
    row["cache_misses"] = perf.get("cache_misses")
    row["cache_miss_rate"] = perf.get("cache_miss_rate")
    row["l1d_loads"] = perf.get("l1d_loads")
    row["l1d_load_misses"] = perf.get("l1d_load_misses")
    row["l1d_load_miss_rate"] = perf.get("l1d_load_miss_rate")

    expected_nodes = int(result.config.arts_nodes) if result.config.arts_nodes else None
    counter_dir = _counter_dir_from_artifacts(result.artifacts)
    counters, counter_meta = _collect_counters(counter_dir, expected_nodes=expected_nodes)
    for field, counter_key in COUNTER_FIELD_MAP.items():
        row[field] = counters.get(counter_key)
    row.update(counter_meta)
    row["has_counters"] = bool(counter_meta.get("counter_source"))
    row["has_perf"] = bool(result.run_arts.perf_metrics or arts_perf_path)

    _apply_derived_fields(row)
    return row


def _first_timing_value(payload: Any) -> Optional[float]:
    if isinstance(payload, dict) and payload:
        first = next(iter(payload.values()))
        return _to_float(first)
    return None


def _flatten_result_serialized(
    result: Dict[str, Any],
    experiment_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    row = _empty_result_row()

    benchmark = result.get("benchmark")
    config = result.get("config") or {}
    arts = result.get("arts") or {}
    omp = result.get("omp") or {}
    slurm = result.get("slurm") or {}
    diagnostics = result.get("diagnostics") or {}
    slurm_stderr = diagnostics.get("slurm_stderr") if isinstance(diagnostics, dict) else {}
    if not isinstance(slurm_stderr, dict):
        slurm_stderr = {}
    runtime_warning = diagnostics.get("runtime_warning") if isinstance(diagnostics, dict) else {}
    if not isinstance(runtime_warning, dict):
        runtime_warning = {}

    suite: Optional[str] = None
    if isinstance(benchmark, str) and "/" in benchmark:
        suite = benchmark.split("/", 1)[0]

    status = _status_text(result.get("status"))
    status_detail = _status_text(result.get("status_detail")) or status
    detected_runtime_warning = bool(runtime_warning.get("has_warning"))
    if not detected_runtime_warning:
        detected_runtime_warning = any(
            int(slurm_stderr.get(key) or 0) > 0
            for key in (
                "srun_error_count",
                "broken_pipe_count",
                "counter_timeout_warnings",
                "remote_send_hard_timeout_count",
                "connection_refused_count",
            )
        )
    if status == "PASS" and detected_runtime_warning and status_detail == "PASS":
        status_detail = "WARN"

    row.update(
        {
            "benchmark": benchmark,
            "suite": suite,
            "size": result.get("size"),
            "threads": result.get("threads") or config.get("arts_threads"),
            "nodes": result.get("nodes") or config.get("arts_nodes"),
            "run": result.get("run_number"),
            "run_phase": _phase_name(result.get("run_phase")),
            "compile_args": result.get("compile_args"),
            "profile": result.get("profile"),
            "perf_enabled": result.get("perf"),
            "perf_interval": result.get("perf_interval"),
            "status": status,
            "status_detail": status_detail,
            "verified": _verification_state(
                status,
                (result.get("verification") or {}).get("note"),
                arts.get("checksum"),
                omp.get("checksum"),
                (result.get("verification") or {}).get("reference_checksum"),
            ),
            "verification_note": (result.get("verification") or {}).get("note"),
            "arts_checksum": arts.get("checksum"),
            "omp_checksum": omp.get("checksum"),
            "reference_checksum": (result.get("verification") or {}).get("reference_checksum"),
            "reference_source": (result.get("verification") or {}).get("reference_source"),
            "runtime_warning": detected_runtime_warning,
            "slurm_job_id": slurm.get("job_id"),
            "slurm_state": slurm.get("state"),
            "slurm_exit_code": slurm.get("exit_code"),
            "slurm_nodelist": slurm.get("nodelist"),
            "srun_error_count": slurm_stderr.get("srun_error_count"),
            "broken_pipe_count": slurm_stderr.get("broken_pipe_count"),
            "counter_timeout_warnings": slurm_stderr.get("counter_timeout_warnings"),
            "remote_send_hard_timeout_count": slurm_stderr.get("remote_send_hard_timeout_count"),
            "connection_refused_count": slurm_stderr.get("connection_refused_count"),
            "speedup_basis": None,
            "arts_e2e_sec": _to_float(arts.get("e2e_sec")) or _first_timing_value(arts.get("e2e_timings")),
            "omp_e2e_sec": _to_float(omp.get("e2e_sec")) or _first_timing_value(omp.get("e2e_timings")),
            "arts_kernel_sec": _first_timing_value(arts.get("kernel_timings")),
            "omp_kernel_sec": _first_timing_value(omp.get("kernel_timings")),
            "arts_init_sec": _to_float(arts.get("init_sec")) or _first_timing_value(arts.get("init_timings")),
            "omp_init_sec": _first_timing_value(omp.get("init_timings")),
            "arts_total_sec": _to_float(arts.get("duration_sec")),
            "omp_total_sec": _to_float(omp.get("duration_sec")),
            "speedup": _to_float(result.get("speedup")),
            "artifact_run_dir": (result.get("artifacts") or {}).get("run_dir"),
            "artifact_run_config": (result.get("artifacts") or {}).get("run_config"),
            "artifact_result_json": (result.get("artifacts") or {}).get("result_json"),
            "artifact_slurm_out": (result.get("artifacts") or {}).get("slurm_out"),
            "artifact_slurm_err": (result.get("artifacts") or {}).get("slurm_err"),
            "artifact_build_dir": (result.get("artifacts") or {}).get("build_dir"),
            "artifact_counter_dir": (result.get("artifacts") or {}).get("counter_dir"),
            "artifact_perf_dir": (result.get("artifacts") or {}).get("perf_dir"),
        }
    )
    perf_files = (result.get("artifacts") or {}).get("perf_files")
    if isinstance(perf_files, list):
        row["artifact_perf_file_count"] = len(perf_files)
    elif row.get("artifact_perf_dir"):
        row["artifact_perf_file_count"] = 1

    perf = _perf_dict(arts)
    row["cache_references"] = perf.get("cache_references")
    row["cache_misses"] = perf.get("cache_misses")
    row["cache_miss_rate"] = perf.get("cache_miss_rate")
    row["l1d_loads"] = perf.get("l1d_loads")
    row["l1d_load_misses"] = perf.get("l1d_load_misses")
    row["l1d_load_miss_rate"] = perf.get("l1d_load_miss_rate")

    counter_dir: Optional[Path] = None
    candidate_dirs: List[Path] = []
    run_dir = result.get("_run_dir")
    if isinstance(run_dir, str) and run_dir:
        candidate_dirs.append(Path(run_dir) / "counters")

    artifact_counter_dir = _counter_dir_from_artifacts(result.get("artifacts"))
    if artifact_counter_dir is not None:
        candidate_dirs.append(artifact_counter_dir)

    if experiment_dir is not None:
        bench_name = result.get("benchmark")
        threads_value = _to_float(result.get("threads") or config.get("arts_threads"))
        nodes_value = _to_float(result.get("nodes") or config.get("arts_nodes"))
        run_number_value = _to_float(result.get("run_number"))
        if (
            isinstance(bench_name, str)
            and threads_value is not None
            and nodes_value is not None
            and run_number_value is not None
        ):
            phase = _phase_name(result.get("run_phase"))
            local_counter_dir = (
                Path(experiment_dir)
                / phase
                / bench_name
                / f"{int(threads_value)}t_{int(nodes_value)}n"
                / f"run_{int(run_number_value)}"
                / "counters"
            )
            candidate_dirs.append(local_counter_dir)

    for candidate in candidate_dirs:
        if candidate.exists():
            counter_dir = candidate
            break

    expected_nodes: Optional[int] = None
    nodes_value = _to_float(row.get("nodes"))
    if nodes_value is not None:
        expected_nodes = int(nodes_value)

    counters, counter_meta = _collect_counters(counter_dir, expected_nodes=expected_nodes)
    for field, counter_key in COUNTER_FIELD_MAP.items():
        row[field] = counters.get(counter_key)
    row.update(counter_meta)
    row["has_counters"] = bool(counter_meta.get("counter_source"))
    row["has_perf"] = bool(row.get("perf_enabled")) or bool(perf)

    _apply_derived_fields(row)
    return row


def _build_summary_rows(result_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, Any, Any, Any, Any, Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        key = (
            row.get("benchmark"),
            row.get("suite"),
            row.get("size"),
            row.get("threads"),
            row.get("nodes"),
            row.get("run_phase"),
            row.get("compile_args"),
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
            str(k[6] or ""),
        ),
    ):
        benchmark, suite, size, threads, nodes, run_phase, compile_args = key
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
        warn_count = sum(1 for r in runs if r.get("runtime_warning") is True)
        verified_count = sum(1 for r in runs if r.get("verified") is True)
        rows_with_counters = sum(1 for r in runs if r.get("has_counters") is True)
        rows_with_perf = sum(1 for r in runs if r.get("has_perf") is True)

        summary_rows.append(
            {
                "benchmark": benchmark,
                "suite": suite,
                "size": size,
                "threads": threads,
                "nodes": nodes,
                "run_phase": run_phase,
                "compile_args": compile_args,
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
                "verified_count": verified_count,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "warn_count": warn_count,
                "rows_with_counters": rows_with_counters,
                "rows_with_perf": rows_with_perf,
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

    grouped: Dict[Tuple[Any, Any, Any, Any, Any, Any, str], List[Dict[str, Any]]] = defaultdict(list)
    config_keys: Set[Tuple[Any, Any, Any, Any, Any, Any]] = set()
    for row in result_rows:
        phase = _phase_name(row.get("run_phase"))
        config_key = (
            row.get("benchmark"),
            row.get("suite"),
            row.get("size"),
            row.get("threads"),
            row.get("nodes"),
            row.get("compile_args"),
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
    columns: List[str] = ["benchmark", "suite", "size", "threads", "nodes", "compile_args"]
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
            str(k[5] or ""),
        ),
    ):
        benchmark, suite, size, threads, nodes, compile_args = config_key
        row_values: List[Any] = [benchmark, suite, size, threads, nodes, compile_args]

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

    _finalize_sheet(ws, columns)


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
        "size",
        "threads",
        "nodes",
        "runs",
        "compile_args",
        "debug",
        "perf",
        "perf_interval",
        "profile",
        "benchmarks_run",
        "rows",
        "passed",
        "verified",
        "warn",
        "failed",
        "rows_with_counters",
        "rows_with_complete_counters",
        "rows_with_perf",
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
        verified = sum(1 for r in phase_rows if r.get("verified") is True)
        failed = sum(1 for s in statuses if s in {"FAIL", "CRASH", "TIMEOUT"})
        warn = sum(1 for r in phase_rows if _status_text(r.get("status_detail")) == "WARN")
        rows_with_counters = sum(
            1
            for r in phase_rows
            if any(r.get(field) is not None for field in COUNTER_FIELD_MAP.keys())
        )
        rows_with_complete_counters = sum(1 for r in phase_rows if r.get("counter_complete") is True)
        rows_with_perf = sum(
            1
            for r in phase_rows
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
        benchmarks = getattr(step, "benchmarks", None)
        benchmarks_run = ", ".join(benchmarks) if benchmarks else "all"

        ws.append(
            [
                step_name,
                getattr(step, "size", None),
                getattr(step, "threads", None),
                getattr(step, "nodes", None),
                getattr(step, "runs", None),
                getattr(step, "compile_args", None),
                getattr(step, "debug", None),
                bool(getattr(step, "perf", False)),
                getattr(step, "perf_interval", None) if getattr(step, "perf", False) else None,
                getattr(step, "profile", None),
                benchmarks_run,
                len(phase_rows),
                passed,
                verified,
                warn,
                failed,
                rows_with_counters,
                rows_with_complete_counters,
                rows_with_perf,
            ]
        )

    _finalize_sheet(ws, columns)


def _append_table_sheet(workbook: Workbook, title: str, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    ws = workbook.create_sheet(title=title)
    ws.append(columns)
    _style_header(ws)
    ws.freeze_panes = "A2"

    for row in rows:
        ws.append([row.get(column) for column in columns])

    _finalize_sheet(ws, columns)


def _build_issues_sheet(workbook: Workbook, result_rows: List[Dict[str, Any]]) -> None:
    issue_rows = [
        row
        for row in result_rows
        if (
            _status_text(row.get("status")) not in {"PASS", "SKIP"}
            or _status_text(row.get("status_detail")) == "WARN"
            or row.get("runtime_warning") is True
            or (row.get("has_counters") is True and row.get("counter_complete") is False)
            or row.get("verified") is False
        )
    ]
    if not issue_rows:
        return

    _append_table_sheet(workbook, "Issues", RESULTS_COLUMNS, issue_rows)


def _append_optional_table_sheet(
    workbook: Workbook,
    title: str,
    columns: List[str],
    rows: List[Dict[str, Any]],
) -> None:
    if not rows:
        return
    _append_table_sheet(workbook, title, columns, rows)


def _apply_path_hyperlinks(worksheet: Any, columns: List[str]) -> None:
    path_columns = [
        idx
        for idx, field in enumerate(columns, start=1)
        if field in PATH_LIKE_FIELDS
    ]
    if not path_columns or worksheet.max_row < 2:
        return

    for idx in path_columns:
        for row in range(2, worksheet.max_row + 1):
            cell = worksheet.cell(row=row, column=idx)
            value = cell.value
            if not isinstance(value, str):
                continue
            text = value.strip()
            if not text:
                continue

            if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", text):
                target = text
            else:
                path = Path(text)
                if not path.is_absolute():
                    continue
                try:
                    target = path.as_uri()
                except ValueError:
                    continue

            cell.hyperlink = target
            cell.style = "Hyperlink"


def _excel_table_name(title: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_]", "_", title)
    if not base:
        base = "Sheet"
    if not re.match(r"^[A-Za-z_]", base):
        base = f"T_{base}"
    return f"tbl_{base}"[:255]


def _finalize_sheet(worksheet: Any, columns: List[str]) -> None:
    _apply_table_formats(worksheet, columns)
    _apply_speedup_rules(worksheet, columns)
    _apply_status_fill(worksheet, columns)
    _apply_path_hyperlinks(worksheet, columns)
    worksheet.auto_filter.ref = worksheet.dimensions

    if (
        Table is not None
        and TableStyleInfo is not None
        and worksheet.max_row >= 2
        and worksheet.max_column >= 1
    ):
        table = Table(
            displayName=_excel_table_name(worksheet.title),
            ref=worksheet.dimensions,
        )
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        worksheet.add_table(table)

    _autosize_columns(worksheet)


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

    indexed: Dict[Tuple[str, str, str, str, str], Dict[Tuple[int, int], Dict[str, Any]]] = defaultdict(dict)
    for row in data_rows:
        benchmark = str(row.get("benchmark") or "")
        suite = str(row.get("suite") or "")
        size = str(row.get("size") or "")
        run_phase = _phase_name(row.get("run_phase"))
        compile_args = str(row.get("compile_args") or "")
        threads_value = int(row.get("threads") or 0)
        nodes_value = int(row.get("nodes") or 0)
        indexed[(benchmark, suite, size, run_phase, compile_args)][(threads_value, nodes_value)] = row

    headers: List[str]
    matrix_rows: List[List[Any]] = []

    if has_thread_sweep and not has_node_sweep:
        headers = ["benchmark", "suite", "size", "run_phase", "compile_args"]
        for t in threads:
            headers.extend([f"T{t}_arts_e2e", f"T{t}_speedup", f"T{t}_efficiency"])
        headers.extend(["peak_speedup", "peak_threads", "arts_self_scaling"])

        for (benchmark, suite, size, run_phase, compile_args), configs in sorted(indexed.items()):
            row_values: List[Any] = [benchmark, suite, size, run_phase, compile_args]
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
        headers = ["benchmark", "suite", "size", "run_phase", "compile_args"]
        for n in nodes:
            headers.extend([f"N{n}_arts_e2e", f"N{n}_speedup"])
            if n != 1:
                headers.append(f"N{n}_scaling_vs_N1")
        headers.extend(["peak_speedup", "peak_nodes"])

        for (benchmark, suite, size, run_phase, compile_args), configs in sorted(indexed.items()):
            row_values = [benchmark, suite, size, run_phase, compile_args]
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
        headers = ["benchmark", "suite", "size", "run_phase", "compile_args"]
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

        for (benchmark, suite, size, run_phase, compile_args), configs in sorted(indexed.items()):
            row_values = [benchmark, suite, size, run_phase, compile_args]
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
        phase_groups: Dict[Tuple[str, str, str], List[List[Any]]] = defaultdict(list)
        for row in matrix_rows:
            phase_groups[(str(row[2]), str(row[3]), str(row[4]))].append(row)

        for size, phase, compile_args in sorted(phase_groups.keys()):
            rows = phase_groups[(size, phase, compile_args)]
            geomean_row: List[Any] = ["GEOMEAN", "", size, phase, compile_args]
            for idx, header in enumerate(headers[5:], start=5):
                if header.startswith("peak_"):
                    geomean_row.append(None)
                    continue
                values = [_to_float(r[idx]) for r in rows]
                values = [v for v in values if v is not None and v > 0]
                geomean_row.append(_geomean(values))
            ws.append(geomean_row)

    _finalize_sheet(ws, headers)


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
    add("total_jobs", metadata.get("total_jobs"))
    add("submitted_jobs", metadata.get("submitted_jobs"))
    add("failed_submissions", metadata.get("failed_submissions"))
    add("partition", metadata.get("partition"))
    add("time_limit", metadata.get("time_limit"))
    add("profile", metadata.get("profile"))
    add("perf", metadata.get("perf"))
    add("perf_interval", metadata.get("perf_interval"))

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
        add("report_warn_count", report_summary.get("warn_count"))
        add("report_skipped_count", report_summary.get("skip_count"))
        add("report_verified_count", report_summary.get("verified_count"))
        add("report_geomean_speedup", report_summary.get("geomean_speedup"))
        add("report_rows_with_counters", report_summary.get("rows_with_counters"))
        add("report_rows_with_complete_counters", report_summary.get("rows_with_complete_counters"))
        add("report_rows_with_partial_counters", report_summary.get("rows_with_partial_counters"))
        add("report_rows_with_unknown_slurm_state", report_summary.get("rows_with_unknown_slurm_state"))
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
    _finalize_sheet(ws, ["key", "value"])


def _build_report_summary(result_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    statuses = [str(r.get("status") or "").upper() for r in result_rows]
    pass_count = sum(1 for s in statuses if s == "PASS")
    fail_count = sum(1 for s in statuses if s in {"FAIL", "CRASH", "TIMEOUT"})
    skip_count = sum(1 for s in statuses if s == "SKIP")
    warn_count = sum(1 for r in result_rows if r.get("runtime_warning") is True)
    verified_count = sum(1 for r in result_rows if r.get("verified") is True)

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
    rows_with_complete_counters = sum(1 for r in result_rows if r.get("counter_complete") is True)
    rows_with_partial_counters = sum(1 for r in result_rows if r.get("counter_source") == "node_fallback")
    rows_with_unknown_slurm_state = sum(
        1
        for r in result_rows
        if str(r.get("slurm_state") or "").upper() == "UNKNOWN"
    )
    rows_with_perf = sum(
        1
        for r in result_rows
        if r.get("has_perf") is True
    )

    return {
        "generated_at": datetime.now().isoformat(),
        "total_rows": len(result_rows),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "skip_count": skip_count,
        "verified_count": verified_count,
        "geomean_speedup": geomean_speedup,
        "rows_with_counters": rows_with_counters,
        "rows_with_complete_counters": rows_with_complete_counters,
        "rows_with_partial_counters": rows_with_partial_counters,
        "rows_with_unknown_slurm_state": rows_with_unknown_slurm_state,
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
    _build_issues_sheet(workbook, result_rows)

    summary_rows = _build_summary_rows(result_rows)
    _append_table_sheet(workbook, "Summary", SUMMARY_COLUMNS, summary_rows)
    _build_scaling_sheet(workbook, summary_rows)
    _build_comparison_sheet(workbook, result_rows, steps=steps)
    _append_optional_table_sheet(
        workbook,
        "NodeCounterSummary",
        NODE_COUNTER_SUMMARY_COLUMNS,
        _build_node_counter_summary_rows(result_rows),
    )
    _append_optional_table_sheet(
        workbook,
        "NodeCounters",
        NODE_COUNTER_COLUMNS,
        _build_node_counter_rows(result_rows),
    )
    _append_optional_table_sheet(
        workbook,
        "PerfFiles",
        PERF_FILE_COLUMNS,
        _build_perf_file_rows(result_rows),
    )

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
    steps: Optional[List["ExperimentStep"]] = None,
) -> Optional[Path]:
    """Generate report.xlsx from in-memory benchmark results."""

    result_rows = [_flatten_result_dataclass(result) for result in results]

    command = "carts benchmarks " + " ".join(sys.argv[1:])
    return _write_report(
        result_rows,
        Path(experiment_dir),
        command=command,
        steps=steps,
    )


def generate_report_from_rows(
    result_rows: List[Dict[str, Any]],
    experiment_dir: Path,
    steps: Optional[List["ExperimentStep"]] = None,
) -> Optional[Path]:
    """Generate report.xlsx from already-serialized result rows."""

    normalized_rows: List[Dict[str, Any]] = []
    experiment_dir_path = Path(experiment_dir)
    for result in result_rows:
        if isinstance(result, dict) and "arts" in result:
            normalized_rows.append(
                _flatten_result_serialized(result, experiment_dir=experiment_dir_path)
            )
            continue

        row = _empty_result_row()
        if isinstance(result, dict):
            for key in RESULTS_COLUMNS:
                if key in result:
                    row[key] = result.get(key)
        _apply_derived_fields(row)
        normalized_rows.append(row)

    command = "carts benchmarks " + " ".join(sys.argv[1:])
    return _write_report(
        normalized_rows,
        experiment_dir_path,
        command=command,
        steps=steps,
    )
