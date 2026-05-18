from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPORT_DIRNAME = "carts-report"
PHASE_THREAD = "single_node"
PHASE_NODE = "multinode"
PHASE_NODE_DB = "multinode_distributed_db"


@dataclass(frozen=True)
class ReportArtifact:
    output_dir: Path
    index_html: Path
    data_js: Path


def generate_carts_report(
    results_dir: Path,
    output_dir: Path | None = None,
    extra_results: Iterable[Path] | None = None,
) -> ReportArtifact:
    """Generate a self-contained static report for a benchmark results directory."""
    results_dir = Path(results_dir).resolve()
    results_json = results_dir / "results.json"
    if not results_json.exists():
        raise FileNotFoundError(f"results.json not found under {results_dir}")

    with results_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise ValueError(f"{results_json} does not contain a results list")

    rows = [_normalize_result(result, results_dir) for result in raw_results if isinstance(result, dict)]
    source_results = [results_json]
    for extra_dir in extra_results or []:
        extra_dir = Path(extra_dir).resolve()
        extra_json = extra_dir / "results.json"
        if not extra_json.exists():
            raise FileNotFoundError(f"extra results.json not found under {extra_dir}")
        with extra_json.open("r", encoding="utf-8") as f:
            extra_payload = json.load(f)
        extra_raw_results = extra_payload.get("results")
        if not isinstance(extra_raw_results, list):
            raise ValueError(f"{extra_json} does not contain a results list")
        rows.extend(
            _normalize_result(result, extra_dir)
            for result in extra_raw_results
            if isinstance(result, dict)
        )
        source_results.append(extra_json)
    report_data = _build_report_data(payload.get("metadata") or {}, rows, source_results)

    if output_dir is None:
        output_dir = results_dir / "presentation" / REPORT_DIRNAME
    output_dir = Path(output_dir).resolve()
    assets_dir = output_dir / "assets"
    data_dir = output_dir / "data"
    src_dir = output_dir / "src"

    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "index.html").write_text(_INDEX_HTML, encoding="utf-8")
    (assets_dir / "styles.css").write_text(_STYLES_CSS, encoding="utf-8")
    (assets_dir / "app.js").write_text(_APP_JS, encoding="utf-8")
    (src_dir / "app.ts").write_text(_APP_TS, encoding="utf-8")
    (assets_dir / "data.js").write_text(
        "window.CARTS_REPORT_DATA = "
        + json.dumps(report_data, sort_keys=True, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )

    _write_csv(data_dir / "results_flat.csv", report_data["rows"])
    _write_csv(data_dir / "phase_summary.csv", report_data["tables"]["phase_summary"])
    _write_csv(data_dir / "thread_scaling.csv", report_data["tables"]["thread_scaling"])
    _write_csv(data_dir / "node_scaling.csv", report_data["tables"]["node_scaling"])
    _write_csv(data_dir / "issues.csv", report_data["tables"]["issues"])
    _write_csv(data_dir / "communication.csv", report_data["tables"]["communication"])
    _write_csv(data_dir / "communication_summary.csv", report_data["tables"]["communication_summary"])
    _write_csv(data_dir / "family_thread_summary.csv", report_data["tables"]["family_thread_summary"])
    _write_csv(data_dir / "family_node_summary.csv", report_data["tables"]["family_node_summary"])
    _write_csv(data_dir / "timing_summary.csv", report_data["tables"]["timing_summary"])
    _write_csv(data_dir / "top_examples.csv", report_data["tables"]["top_examples"])
    _write_csv(data_dir / "family_examples.csv", report_data["tables"]["family_examples"])

    (output_dir / "README.md").write_text(_readme(report_data), encoding="utf-8")
    return ReportArtifact(
        output_dir=output_dir,
        index_html=output_dir / "index.html",
        data_js=assets_dir / "data.js",
    )


def _normalize_result(result: dict[str, Any], results_dir: Path) -> dict[str, Any]:
    benchmark = str(result.get("benchmark") or result.get("name") or "unknown")
    arts = _dict(result.get("arts") or result.get("run_arts"))
    omp = _dict(result.get("omp") or result.get("run_omp"))
    diagnostics = _dict(result.get("diagnostics"))
    artifacts = _dict(result.get("artifacts"))
    verification = _dict(result.get("verification"))
    config = _dict(result.get("config"))
    slurm = _dict(result.get("slurm"))
    phase = str(result.get("run_phase") or "default")
    status = str(result.get("status") or arts.get("status") or "UNKNOWN").upper()
    counter_dir = _path_or_none(artifacts.get("counter_dir"))
    counters = _read_cluster_counters(counter_dir, results_dir)
    stderr_diag = _dict(diagnostics.get("slurm_stderr"))
    runtime_warning = _dict(diagnostics.get("runtime_warning"))
    threads = _int(result.get("threads") or config.get("arts_threads")) or 0
    nodes = _int(result.get("nodes") or config.get("arts_nodes")) or 0
    arts_e2e = _timing(arts, "e2e_timings") or _float(arts.get("duration_sec"))
    kernel = _timing(arts, "kernel_timings")
    startup = _timing_sum(arts, "startup_timings")
    startup_share = (_safe_div(startup, arts_e2e) or 0.0) * 100.0 if startup is not None else None
    kernel_share = (_safe_div(kernel, arts_e2e) or 0.0) * 100.0 if kernel is not None else None
    overhead_sec = None
    if arts_e2e is not None and kernel is not None:
        overhead_sec = max(0.0, arts_e2e - kernel - (startup or 0.0))
    omp_e2e = None if omp.get("skipped") is True else (_timing(omp, "e2e_timings") or _float(omp.get("duration_sec")))
    speedup_vs_omp = _safe_div(omp_e2e, arts_e2e)

    return {
        "benchmark": benchmark,
        "family": benchmark.split("/", 1)[0] if "/" in benchmark else benchmark,
        "phase": phase,
        "phase_kind": _phase_kind(phase),
        "phase_label": _phase_label(_phase_kind(phase)),
        "size": str(result.get("size") or ""),
        "threads": threads,
        "nodes": nodes,
        "run": _int(result.get("run_number")) or 1,
        "status": status,
        "verified": _verification_ok(verification, status),
        "arts_e2e_sec": arts_e2e,
        "arts_kernel_sec": kernel,
        "startup_sec": startup,
        "startup_share_pct": startup_share,
        "kernel_share_pct": kernel_share,
        "overhead_sec": overhead_sec,
        "omp_e2e_sec": omp_e2e,
        "speedup_vs_openmp": speedup_vs_omp,
        "e2e_self_speedup": None,
        "kernel_self_speedup": None,
        "e2e_efficiency": None,
        "kernel_efficiency": None,
        "compile_args": result.get("compile_args"),
        "slurm_state": slurm.get("state"),
        "slurm_job_id": slurm.get("job_id"),
        "runtime_warning": bool(runtime_warning.get("has_warning")),
        "srun_error_count": _int(stderr_diag.get("srun_error_count")) or 0,
        "broken_pipe_count": _int(stderr_diag.get("broken_pipe_count")) or 0,
        "counter_timeout_warnings": _int(stderr_diag.get("counter_timeout_warnings")) or 0,
        "remote_send_hard_timeout_count": _int(stderr_diag.get("remote_send_hard_timeout_count")) or 0,
        "connection_refused_count": _int(stderr_diag.get("connection_refused_count")) or 0,
        "counter_available": bool(counters),
        "remote_bytes_sent": _counter(counters, "BYTES_REMOTE_SENT", "remoteBytesSent"),
        "remote_bytes_received": _counter(counters, "BYTES_REMOTE_RECEIVED", "remoteBytesReceived"),
        "remote_sends": _counter(counters, "NUM_REMOTE_SEND", "numRemoteSend"),
        "remote_receives": _counter(counters, "NUM_REMOTE_RECEIVE", "numRemoteReceive"),
        "connect_attempts": _counter(counters, "NUM_REMOTE_CONNECT_ATTEMPT", "numRemoteConnectAttempt"),
        "connect_success": _counter(counters, "NUM_REMOTE_CONNECT_SUCCESS", "numRemoteConnectSuccess"),
        "connect_fail": _counter(counters, "NUM_REMOTE_CONNECT_FAIL", "numRemoteConnectFail"),
        "memory_footprint_bytes": _counter(counters, "BYTES_MEMORY_FOOTPRINT", "memoryFootprint"),
        "time_init_ms": _counter(counters, "TIME_INIT", "initializationTime", prefer_ms=True),
        "time_total_ms": _counter(counters, "TIME_TOTAL", "endToEndTime", prefer_ms=True),
        "artifact_run_dir": artifacts.get("run_dir"),
    }


def _build_report_data(metadata: dict[str, Any], rows: list[dict[str, Any]], source_results: list[Path]) -> dict[str, Any]:
    _attach_self_scaling(rows)
    phase_summary = _phase_summary(rows)
    thread_scaling = _thread_scaling(rows)
    node_scaling = _node_scaling(rows)
    issues = _issues(rows)
    openmp = _openmp_rows(rows)
    communication = _communication_rows(rows)
    communication_summary = _communication_summary(communication)
    family_thread_summary = _family_thread_summary(rows)
    family_summary = _family_node_summary(rows)
    timing_summary = _timing_summary(rows)
    top_examples = _top_examples(rows)
    family_examples = _family_examples(rows)
    optimizations = _optimization_recommendations(rows, node_scaling, communication)
    headline = _headline(rows, thread_scaling, node_scaling, issues, communication)
    metadata_out = {
        "title": "CARTS Report",
        "source_results": [str(path) for path in source_results],
        "timestamp": metadata.get("timestamp"),
        "experiment_name": metadata.get("experiment_name"),
        "total_jobs": metadata.get("total_jobs") or len(rows),
        "timeout_sec": _infer_timeout(metadata, rows),
        "transport_note": (
            "Multinode CARTS runs use the ARTS runtime transport selected by the benchmark configuration. "
            "The report frames results around CARTS compiler/runtime behavior and keeps protocol details in metrics."
        ),
    }
    return {
        "metadata": metadata_out,
        "headline": headline,
        "rows": rows,
        "tables": {
            "phase_summary": phase_summary,
            "thread_scaling": thread_scaling,
            "node_scaling": node_scaling,
            "issues": issues,
            "openmp": openmp,
            "communication": communication,
            "communication_summary": communication_summary,
            "family_thread_summary": family_thread_summary,
            "family_node_summary": family_summary,
            "timing_summary": timing_summary,
            "top_examples": top_examples,
            "family_examples": family_examples,
        },
        "benchmarks": sorted({row["benchmark"] for row in rows}),
        "families": sorted({row["family"] for row in rows}),
        "optimizations": optimizations,
    }


def _attach_self_scaling(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row["status"] != "PASS":
            continue
        dimension = "threads" if row["phase_kind"] == PHASE_THREAD else "nodes"
        key = (row["phase"], row["benchmark"], dimension)
        if (dimension == "threads" and row["nodes"] == 1 and row["threads"] == 1) or (
            dimension == "nodes" and row["nodes"] == 1
        ):
            baselines.setdefault(key, row)

    for row in rows:
        if row["status"] != "PASS":
            continue
        dimension = "threads" if row["phase_kind"] == PHASE_THREAD else "nodes"
        key = (row["phase"], row["benchmark"], dimension)
        base = baselines.get(key)
        if not base:
            continue
        e2e_speedup = _safe_div(base.get("arts_e2e_sec"), row.get("arts_e2e_sec"))
        kernel_speedup = _safe_div(base.get("arts_kernel_sec"), row.get("arts_kernel_sec"))
        scale = row["threads"] if dimension == "threads" else row["nodes"]
        row["e2e_self_speedup"] = e2e_speedup
        row["kernel_self_speedup"] = kernel_speedup
        row["e2e_efficiency"] = _safe_div(e2e_speedup, scale)
        row["kernel_efficiency"] = _safe_div(kernel_speedup, scale)


def _phase_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["phase"]].append(row)
    summary = []
    for phase, phase_rows in sorted(grouped.items()):
        passed = sum(1 for row in phase_rows if row["status"] == "PASS")
        verified = sum(1 for row in phase_rows if row["verified"])
        summary.append({
            "phase": phase,
            "phase_kind": _phase_kind(phase),
            "total": len(phase_rows),
            "pass": passed,
            "fail": len(phase_rows) - passed,
            "verified": verified,
            "pass_pct": _pct(passed, len(phase_rows)),
        })
    return summary


def _thread_scaling(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    thread_values = sorted({row["threads"] for row in rows if row["phase_kind"] == PHASE_THREAD and row["nodes"] == 1})
    for threads in thread_values:
        subset = [row for row in rows if row["phase_kind"] == PHASE_THREAD and row["nodes"] == 1 and row["threads"] == threads]
        passed = [row for row in subset if row["status"] == "PASS"]
        out.append({
            "threads": threads,
            "total": len(subset),
            "pass": len(passed),
            "pass_pct": _pct(len(passed), len(subset)),
            "e2e_geomean_speedup": _geomean(row.get("e2e_self_speedup") for row in passed),
            "kernel_geomean_speedup": _geomean(row.get("kernel_self_speedup") for row in passed),
            "e2e_geomean_efficiency": _geomean(row.get("e2e_efficiency") for row in passed),
            "kernel_geomean_efficiency": _geomean(row.get("kernel_efficiency") for row in passed),
            "comparable": sum(1 for row in passed if row.get("e2e_self_speedup")),
        })
    return out


def _node_scaling(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for phase_kind in (PHASE_NODE, PHASE_NODE_DB):
        phase_rows = [row for row in rows if row["phase_kind"] == phase_kind]
        for nodes in sorted({row["nodes"] for row in phase_rows}):
            subset = [row for row in phase_rows if row["nodes"] == nodes]
            passed = [row for row in subset if row["status"] == "PASS"]
            out.append({
                "phase_kind": phase_kind,
                "phase_label": _phase_label(phase_kind),
                "nodes": nodes,
                "total": len(subset),
                "pass": len(passed),
                "pass_pct": _pct(len(passed), len(subset)),
                "e2e_geomean_speedup": _geomean(row.get("e2e_self_speedup") for row in passed),
                "kernel_geomean_speedup": _geomean(row.get("kernel_self_speedup") for row in passed),
                "e2e_geomean_efficiency": _geomean(row.get("e2e_efficiency") for row in passed),
                "kernel_geomean_efficiency": _geomean(row.get("kernel_efficiency") for row in passed),
                "comparable": sum(1 for row in passed if row.get("e2e_self_speedup")),
            })
    return out


def _issues(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues = []
    for row in rows:
        if row["status"] == "PASS" and row["verified"] and not row["runtime_warning"]:
            continue
        issues.append({
            "benchmark": row["benchmark"],
            "phase": row["phase"],
            "threads": row["threads"],
            "nodes": row["nodes"],
            "status": row["status"],
            "verified": row["verified"],
            "runtime_warning": row["runtime_warning"],
            "srun_error_count": row["srun_error_count"],
            "artifact_run_dir": row["artifact_run_dir"],
        })
    return sorted(issues, key=lambda item: (item["status"] == "PASS", item["phase"], item["benchmark"], item["nodes"], item["threads"]))


def _openmp_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows
        if row["nodes"] == 1 and row["status"] == "PASS" and row.get("omp_e2e_sec") and row.get("arts_e2e_sec")
    ]
    max_threads = max((row["threads"] for row in candidates), default=0)
    out = []
    for row in sorted((r for r in candidates if r["threads"] == max_threads), key=lambda r: r["benchmark"]):
        out.append({
            "benchmark": row["benchmark"],
            "family": row["family"],
            "threads": row["threads"],
            "arts_e2e_sec": row["arts_e2e_sec"],
            "openmp_e2e_sec": row["omp_e2e_sec"],
            "carts_vs_openmp": row["speedup_vs_openmp"],
        })
    return out


def _communication_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if not row["counter_available"]:
            continue
        sent = _float(row.get("remote_bytes_sent")) or 0.0
        received = _float(row.get("remote_bytes_received")) or 0.0
        sends = _float(row.get("remote_sends")) or 0.0
        receives = _float(row.get("remote_receives")) or 0.0
        total_bytes = sent + received
        total_messages = sends + receives
        e2e_sec = _float(row.get("arts_e2e_sec"))
        out.append({
            "benchmark": row["benchmark"],
            "family": row["family"],
            "phase": row["phase"],
            "phase_kind": row["phase_kind"],
            "threads": row["threads"],
            "nodes": row["nodes"],
            "remote_bytes_total": total_bytes,
            "remote_messages_total": total_messages,
            "remote_bytes_per_message": _safe_div(total_bytes, total_messages),
            "remote_bytes_per_sec": _safe_div(total_bytes, e2e_sec),
            "remote_messages_per_sec": _safe_div(total_messages, e2e_sec),
            "connect_success_pct": _pct(_float(row.get("connect_success")) or 0.0, _float(row.get("connect_attempts")) or 0.0),
            "memory_footprint_bytes": row.get("memory_footprint_bytes"),
            "time_init_ms": row.get("time_init_ms"),
            "time_total_ms": row.get("time_total_ms"),
        })
    return sorted(out, key=lambda row: (row["phase"], row["nodes"], row["benchmark"]))


def _communication_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["nodes"])].append(row)
    out = []
    for (family, nodes), subset in sorted(grouped.items()):
        out.append({
            "family": family,
            "nodes": nodes,
            "counter_rows": len(subset),
            "median_remote_bytes_total": _median(row.get("remote_bytes_total") for row in subset),
            "median_remote_messages_total": _median(row.get("remote_messages_total") for row in subset),
            "median_remote_bytes_per_message": _median(row.get("remote_bytes_per_message") for row in subset),
            "median_remote_bytes_per_sec": _median(row.get("remote_bytes_per_sec") for row in subset),
            "median_connect_success_pct": _median(row.get("connect_success_pct") for row in subset),
        })
    return out


def _family_thread_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["phase_kind"] == PHASE_THREAD:
            grouped[(row["family"], row["threads"])].append(row)
    out = []
    for (family, threads), subset in sorted(grouped.items()):
        passed = [row for row in subset if row["status"] == "PASS"]
        out.append({
            "family": family,
            "threads": threads,
            "total": len(subset),
            "pass": len(passed),
            "pass_pct": _pct(len(passed), len(subset)),
            "e2e_geomean_speedup": _geomean(row.get("e2e_self_speedup") for row in passed),
            "kernel_geomean_speedup": _geomean(row.get("kernel_self_speedup") for row in passed),
            "e2e_geomean_efficiency": _geomean(row.get("e2e_efficiency") for row in passed),
            "median_startup_share_pct": _median(row.get("startup_share_pct") for row in passed),
        })
    return out


def _family_node_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["phase_kind"] in {PHASE_NODE, PHASE_NODE_DB}:
            grouped[(row["family"], row["phase_kind"], row["nodes"])].append(row)
    out = []
    for (family, phase_kind, nodes), subset in sorted(grouped.items()):
        passed = [row for row in subset if row["status"] == "PASS"]
        out.append({
            "family": family,
            "phase_kind": phase_kind,
            "phase_label": _phase_label(phase_kind),
            "nodes": nodes,
            "total": len(subset),
            "pass": len(passed),
            "pass_pct": _pct(len(passed), len(subset)),
            "e2e_geomean_speedup": _geomean(row.get("e2e_self_speedup") for row in passed),
            "kernel_geomean_speedup": _geomean(row.get("kernel_self_speedup") for row in passed),
            "e2e_geomean_efficiency": _geomean(row.get("e2e_efficiency") for row in passed),
            "median_startup_share_pct": _median(row.get("startup_share_pct") for row in passed),
        })
    return out


def _timing_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scale = row["threads"] if row["phase_kind"] == PHASE_THREAD else row["nodes"]
        grouped[(row["phase_kind"], _phase_label(row["phase_kind"]), scale)].append(row)
    out = []
    for (phase_kind, phase_label, scale), subset in sorted(grouped.items()):
        passed = [row for row in subset if row["status"] == "PASS"]
        out.append({
            "phase_kind": phase_kind,
            "phase_label": phase_label,
            "scale": scale,
            "scale_label": "threads" if phase_kind == PHASE_THREAD else "nodes",
            "total": len(subset),
            "pass": len(passed),
            "median_e2e_sec": _median(row.get("arts_e2e_sec") for row in passed),
            "median_kernel_sec": _median(row.get("arts_kernel_sec") for row in passed),
            "median_startup_sec": _median(row.get("startup_sec") for row in passed),
            "median_overhead_sec": _median(row.get("overhead_sec") for row in passed),
            "median_startup_share_pct": _median(row.get("startup_share_pct") for row in passed),
            "median_kernel_share_pct": _median(row.get("kernel_share_pct") for row in passed),
            "srun_error_count": sum(row.get("srun_error_count") or 0 for row in subset),
            "runtime_warning_count": sum(1 for row in subset if row.get("runtime_warning")),
        })
    return out


def _top_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    thread_rows = [row for row in rows if row["phase_kind"] == PHASE_THREAD and row["status"] == "PASS"]
    max_threads = max((row["threads"] for row in thread_rows), default=0)
    for row in sorted(
        (r for r in thread_rows if r["threads"] == max_threads and r.get("e2e_self_speedup")),
        key=lambda r: r["e2e_self_speedup"],
        reverse=True,
    )[:12]:
        out.append(_example_row("single-node", row, "threads", max_threads))

    node_rows = [row for row in rows if row["phase_kind"] in {PHASE_NODE, PHASE_NODE_DB} and row["status"] == "PASS"]
    max_nodes = max((row["nodes"] for row in node_rows), default=0)
    for row in sorted(
        (r for r in node_rows if r["nodes"] == max_nodes and r.get("e2e_self_speedup")),
        key=lambda r: r["e2e_self_speedup"],
        reverse=True,
    )[:16]:
        out.append(_example_row("multinode", row, "nodes", max_nodes))
    return out


def _family_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    specs = [
        ("single-node", PHASE_THREAD, "threads"),
        ("multinode", PHASE_NODE, "nodes"),
        ("multinode distributed DB", PHASE_NODE_DB, "nodes"),
    ]
    for kind, phase_kind, scale_key in specs:
        phase_rows = [row for row in rows if row["phase_kind"] == phase_kind]
        max_scale = max((row[scale_key] for row in phase_rows), default=0)
        if not max_scale:
            continue
        for family in sorted({row["family"] for row in phase_rows}):
            family_rows = [row for row in phase_rows if row["family"] == family and row[scale_key] == max_scale]
            passed = [row for row in family_rows if row["status"] == "PASS"]
            comparable = [row for row in passed if row.get("e2e_self_speedup") is not None]
            best = max(comparable, key=lambda row: row["e2e_self_speedup"], default=None)
            worst = min(comparable, key=lambda row: row["e2e_self_speedup"], default=None)
            out.append({
                "kind": kind,
                "family": family,
                "scale_label": scale_key,
                "scale": max_scale,
                "pass": len(passed),
                "total": len(family_rows),
                "pass_pct": _pct(len(passed), len(family_rows)),
                "geomean_e2e_speedup": _geomean(row.get("e2e_self_speedup") for row in comparable),
                "geomean_kernel_speedup": _geomean(row.get("kernel_self_speedup") for row in comparable),
                "median_e2e_sec": _median(row.get("arts_e2e_sec") for row in passed),
                "median_startup_share_pct": _median(row.get("startup_share_pct") for row in passed),
                "best_benchmark": best.get("benchmark") if best else None,
                "best_e2e_speedup": best.get("e2e_self_speedup") if best else None,
                "worst_benchmark": worst.get("benchmark") if worst else None,
                "worst_e2e_speedup": worst.get("e2e_self_speedup") if worst else None,
            })
    return out


def _example_row(kind: str, row: dict[str, Any], scale_label: str, scale: int) -> dict[str, Any]:
    return {
        "kind": kind,
        "benchmark": row["benchmark"],
        "family": row["family"],
        "phase": row["phase"],
        "scale_label": scale_label,
        "scale": scale,
        "e2e_speedup": row.get("e2e_self_speedup"),
        "kernel_speedup": row.get("kernel_self_speedup"),
        "e2e_efficiency": row.get("e2e_efficiency"),
        "startup_share_pct": row.get("startup_share_pct"),
        "arts_e2e_sec": row.get("arts_e2e_sec"),
    }


def _optimization_recommendations(
    rows: list[dict[str, Any]],
    node_scaling: list[dict[str, Any]],
    communication: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    recs = []
    worst_node = min(
        (row for row in node_scaling if row["nodes"] > 1 and row.get("e2e_geomean_efficiency") is not None),
        key=lambda row: row["e2e_geomean_efficiency"],
        default=None,
    )
    if worst_node:
        recs.append({
            "area": "Grain size and owner locality",
            "priority": 1,
            "evidence": f"{worst_node['phase_label']} reaches {worst_node['e2e_geomean_efficiency']:.2f} E2E efficiency at {worst_node['nodes']} nodes.",
            "proposal": "Coarsen tiny owner-local codelets and place work near DB ownership before adding more communication.",
        })
    if communication:
        recs.append({
            "area": "Communication aggregation",
            "priority": 2,
            "evidence": "Counter-enabled rows expose remote bytes, messages, and connection success by benchmark and node count.",
            "proposal": "Batch small remote signals and reuse runtime connections where the dependency contract allows it.",
        })
    failures = Counter(row["benchmark"] for row in rows if row["status"] != "PASS")
    if failures:
        bench, count = failures.most_common(1)[0]
        recs.append({
            "area": "Targeted correctness/performance triage",
            "priority": 3,
            "evidence": f"{bench} has {count} non-passing run(s) in this study.",
            "proposal": "Promote the smallest failing case to a regression and capture its stage dumps under .carts/outputs.",
        })
    recs.append({
        "area": "Weak scaling follow-up",
        "priority": 4,
        "evidence": "This report is a fixed-size scaling study; high node counts can become overhead dominated.",
        "proposal": "Add a weak-scaling companion run to separate runtime overhead from insufficient work per node.",
    })
    return recs


def _headline(
    rows: list[dict[str, Any]],
    thread_scaling: list[dict[str, Any]],
    node_scaling: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    communication: list[dict[str, Any]],
) -> dict[str, Any]:
    scaling_rows = [
        row for row in rows
        if row["phase_kind"] in {PHASE_THREAD, PHASE_NODE, PHASE_NODE_DB}
    ] or rows
    passed = sum(1 for row in scaling_rows if row["status"] == "PASS")
    thread_max = max(thread_scaling, key=lambda row: row["threads"], default={})
    node64 = [row for row in node_scaling if row["nodes"] == 64]
    gemm64 = [
        row for row in rows
        if row["benchmark"] == "polybench/gemm" and row["nodes"] == 64 and row["status"] == "PASS"
    ]
    return {
        "overall_pass_rate_pct": _pct(passed, len(scaling_rows)),
        "overall_passes": passed,
        "overall_total": len(scaling_rows),
        "single_node_max_threads": thread_max.get("threads"),
        "single_node_geomean_e2e_speedup": thread_max.get("e2e_geomean_speedup"),
        "node64_baseline_geomean_e2e_speedup": _first(
            row.get("e2e_geomean_speedup") for row in node64 if row["phase_kind"] == PHASE_NODE
        ),
        "node64_distributed_db_geomean_e2e_speedup": _first(
            row.get("e2e_geomean_speedup") for row in node64 if row["phase_kind"] == PHASE_NODE_DB
        ),
        "gemm_64n_best_e2e_speedup": max((row.get("e2e_self_speedup") or 0.0 for row in gemm64), default=None),
        "issue_count": len(issues),
        "counter_rows": len(communication),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_cluster_counters(counter_dir: Path | None, results_dir: Path) -> dict[str, Any]:
    if counter_dir is None:
        return {}
    candidates = [counter_dir / "cluster.json"]
    if counter_dir.is_absolute():
        try:
            relative = counter_dir.relative_to(results_dir)
            candidates.append(results_dir / relative / "cluster.json")
        except ValueError:
            pass
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            counters = payload.get("counters") if isinstance(payload, dict) else None
            return counters if isinstance(counters, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _counter(counters: dict[str, Any], *names: str, prefer_ms: bool = False) -> float | None:
    for name in names:
        payload = counters.get(name)
        if isinstance(payload, dict):
            value = payload.get("value_ms") if prefer_ms and payload.get("value_ms") is not None else payload.get("value")
            parsed = _float(value)
            if parsed is not None:
                return parsed
    return None


def _phase_kind(phase: str) -> str:
    lowered = phase.lower()
    if "thread" in lowered:
        return PHASE_THREAD
    if "distributed" in lowered or "dist-db" in lowered or "db" in lowered:
        return PHASE_NODE_DB
    if "node" in lowered or "multinode" in lowered:
        return PHASE_NODE
    return "other"


def _phase_label(phase_kind: str) -> str:
    return {
        PHASE_THREAD: "CARTS single node",
        PHASE_NODE: "CARTS multinode",
        PHASE_NODE_DB: "CARTS multinode + distributed DB",
    }.get(phase_kind, phase_kind)


def _verification_ok(verification: dict[str, Any], status: str) -> bool:
    if status != "PASS":
        return False
    if verification.get("correct") is False:
        return False
    note = str(verification.get("note") or "").lower()
    return "fail" not in note and "mismatch" not in note


def _timing(payload: dict[str, Any], field: str) -> float | None:
    timings = payload.get(field)
    if isinstance(timings, dict):
        return _first(_float(value) for value in timings.values())
    return None


def _timing_sum(payload: dict[str, Any], field: str) -> float | None:
    timings = payload.get(field)
    if not isinstance(timings, dict):
        return None
    values = [_float(value) for value in timings.values()]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _path_or_none(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value))


def _int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _first(values: Iterable[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _safe_div(numerator: Any, denominator: Any) -> float | None:
    num = _float(numerator)
    den = _float(denominator)
    if num is None or den in (None, 0.0):
        return None
    return num / den


def _pct(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * numerator / denominator


def _geomean(values: Iterable[Any]) -> float | None:
    positives = [_float(value) for value in values]
    positives = [value for value in positives if value is not None and value > 0.0]
    if not positives:
        return None
    return math.exp(sum(math.log(value) for value in positives) / len(positives))


def _median(values: Iterable[Any]) -> float | None:
    numbers = sorted(value for value in (_float(value) for value in values) if value is not None)
    if not numbers:
        return None
    midpoint = len(numbers) // 2
    if len(numbers) % 2:
        return numbers[midpoint]
    return (numbers[midpoint - 1] + numbers[midpoint]) / 2.0


def _infer_timeout(metadata: dict[str, Any], rows: list[dict[str, Any]]) -> int | None:
    for key in ("timeout", "timeout_sec"):
        parsed = _int(metadata.get(key))
        if parsed:
            return parsed
    durations = [_float(row.get("arts_e2e_sec")) for row in rows if row["status"] != "PASS"]
    durations = [value for value in durations if value]
    return int(max(durations)) if durations else None


def _readme(data: dict[str, Any]) -> str:
    headline = data["headline"]
    return "\n".join([
        "# CARTS Report",
        "",
        "Open `index.html` in a browser. The report is static and reads only local files in this directory.",
        "Figures can be downloaded as SVG from the controls next to each plot.",
        "",
        "## Headline",
        "",
        f"- CARTS scaling pass rate: {headline['overall_pass_rate_pct']:.1f}% ({headline['overall_passes']}/{headline['overall_total']}).",
        f"- Max single-node thread speedup: {_fmt(headline.get('single_node_geomean_e2e_speedup'))}x.",
        f"- 64-node baseline geomean speedup: {_fmt(headline.get('node64_baseline_geomean_e2e_speedup'))}x.",
        f"- 64-node distributed-DB geomean speedup: {_fmt(headline.get('node64_distributed_db_geomean_e2e_speedup'))}x.",
        "",
        "## Data",
        "",
        "- `assets/data.js`: report payload.",
        "- `data/results_flat.csv`: flattened run rows.",
        "- `data/thread_scaling.csv`: single-node scaling summary.",
        "- `data/node_scaling.csv`: multinode scaling summary.",
        "- `data/family_thread_summary.csv`: per-family single-node subplots source.",
        "- `data/family_node_summary.csv`: per-family multinode subplots source.",
        "- `data/family_examples.csv`: best and worst examples per family at the largest scale.",
        "- `data/communication_summary.csv`: family-level communication counter medians.",
        "- `data/timing_summary.csv`: startup/kernel/overhead metrics by scale.",
        "- `data/issues.csv`: non-passing or warning rows.",
    ]) + "\n"


def _fmt(value: Any) -> str:
    parsed = _float(value)
    return "n/a" if parsed is None else f"{parsed:.2f}"


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>CARTS Report</title>
<link rel="stylesheet" href="assets/styles.css" />
<script src="assets/data.js"></script>
<script defer src="assets/app.js"></script>
</head>
<body>
<header class="topbar">
  <div>
    <h1>CARTS Report</h1>
    <p>Compiler-managed task execution from OpenMP-style source programs, measured from one node through the full multinode sweep.</p>
  </div>
  <nav>
    <a href="#overview">Overview</a>
    <a href="#tables">Tables</a>
    <a href="#single">Single Node</a>
    <a href="#multi">Multinode</a>
    <a href="#openmp">OpenMP</a>
    <a href="#metrics">Metrics</a>
    <a href="#explorer">Explorer</a>
  </nav>
</header>
<main>
  <section id="overview">
    <h2>Evidence Overview</h2>
    <div id="kpis" class="kpis"></div>
    <div class="grid two">
      <article class="panel wide"><h3>Study Matrix</h3><div id="study-matrix" class="table"></div></article>
    </div>
    <div class="grid two">
      <article class="panel"><h3>Pass Rate By Phase</h3><div id="phase-pass" class="chart"></div></article>
      <article class="panel"><h3>Issues</h3><div id="issues" class="table"></div></article>
    </div>
  </section>
  <section id="tables">
    <h2>Report Tables And Exports</h2>
    <div id="export-links" class="export-grid"></div>
    <div class="grid two">
      <article class="panel wide"><h3>Scaling Summary</h3><div id="scaling-summary-table" class="table"></div></article>
      <article class="panel"><h3>Single-Node Family Summary</h3><div id="family-thread-table" class="table"></div></article>
      <article class="panel"><h3>Multinode Family Summary</h3><div id="family-node-table" class="table"></div></article>
    </div>
  </section>
  <section id="single">
    <h2>Single Node Thread Scaling</h2>
    <div class="grid two">
      <article class="panel"><h3>Geomean Speedup</h3><div id="thread-speedup" class="chart"></div></article>
      <article class="panel"><h3>Geomean Efficiency</h3><div id="thread-efficiency" class="chart"></div></article>
      <article class="panel wide"><h3>Top 64-Thread Examples</h3><div id="single-examples" class="table"></div></article>
      <article class="panel wide"><h3>Family 64-Thread Best/Worst Examples</h3><div id="single-family-examples" class="table"></div></article>
    </div>
    <h3 class="subhead">Per-Family Benchmark Subplots</h3>
    <div class="controls compact">
      <label>Family metric <select id="single-family-metric-select">
        <option value="e2e_self_speedup">E2E self-speedup</option>
        <option value="kernel_self_speedup">Kernel self-speedup</option>
        <option value="e2e_efficiency">E2E efficiency</option>
        <option value="kernel_efficiency">Kernel efficiency</option>
        <option value="arts_e2e_sec">E2E seconds</option>
        <option value="startup_share_pct">Startup share</option>
      </select></label>
    </div>
    <div id="single-family-plots" class="family-grid"></div>
  </section>
  <section id="multi">
    <h2>Multinode CARTS Scaling</h2>
    <div class="grid two">
      <article class="panel"><h3>Node Speedup</h3><div id="node-speedup" class="chart"></div></article>
      <article class="panel"><h3>Node Efficiency</h3><div id="node-efficiency" class="chart"></div></article>
      <article class="panel wide"><h3>Family Efficiency Heatmap</h3><div id="family-heatmap" class="chart"></div></article>
      <article class="panel wide"><h3>GEMM Node Scaling</h3><div id="gemm" class="chart"></div></article>
      <article class="panel wide"><h3>Top 64-Node Examples</h3><div id="multi-examples" class="table"></div></article>
      <article class="panel wide"><h3>Family 64-Node Best/Worst Examples</h3><div id="multi-family-examples" class="table"></div></article>
    </div>
    <h3 class="subhead">Per-Family Multinode Subplots</h3>
    <div class="controls compact">
      <label>Family metric <select id="multi-family-metric-select">
        <option value="e2e_self_speedup">E2E self-speedup</option>
        <option value="kernel_self_speedup">Kernel self-speedup</option>
        <option value="e2e_efficiency">E2E efficiency</option>
        <option value="kernel_efficiency">Kernel efficiency</option>
        <option value="arts_e2e_sec">E2E seconds</option>
        <option value="startup_share_pct">Startup share</option>
      </select></label>
    </div>
    <div id="multi-family-plots" class="family-grid"></div>
  </section>
  <section id="openmp">
    <h2>Single-Node OpenMP Baseline</h2>
    <div class="grid two">
      <article class="panel"><h3>CARTS / OpenMP Ratio</h3><div id="openmp-bars" class="chart"></div></article>
      <article class="panel"><h3>OpenMP Comparison Rows</h3><div id="openmp-table" class="table"></div></article>
    </div>
  </section>
  <section id="metrics">
    <h2>Runtime Metrics</h2>
    <div class="grid two">
      <article class="panel"><h3>Median Startup Share</h3><div id="startup-share" class="chart"></div></article>
      <article class="panel"><h3>Median E2E Time</h3><div id="median-e2e" class="chart"></div></article>
      <article class="panel"><h3>Runtime Diagnostics</h3><div id="diagnostics" class="chart"></div></article>
      <article class="panel"><h3>Timing Metrics</h3><div id="timing-table" class="table"></div></article>
      <article class="panel"><h3>Remote Bytes</h3><div id="comm-bytes" class="chart"></div></article>
      <article class="panel"><h3>Connection Success</h3><div id="comm-connect" class="chart"></div></article>
      <article class="panel wide"><h3>Communication Summary By Family</h3><div id="comm-family-table" class="table"></div></article>
      <article class="panel wide"><h3>Optimization Roadmap</h3><div id="optimizations" class="cards"></div></article>
    </div>
  </section>
  <section id="explorer">
    <h2>Benchmark Explorer</h2>
    <div class="controls">
      <label>Benchmark <select id="benchmark-select"></select></label>
      <label>Metric <select id="metric-select">
        <option value="e2e_self_speedup">E2E self-speedup</option>
        <option value="kernel_self_speedup">Kernel self-speedup</option>
        <option value="e2e_efficiency">E2E efficiency</option>
        <option value="arts_e2e_sec">E2E seconds</option>
        <option value="arts_kernel_sec">Kernel seconds</option>
      </select></label>
    </div>
    <div class="grid two">
      <article class="panel"><h3>Single Node</h3><div id="explorer-single" class="chart"></div></article>
      <article class="panel"><h3>Multinode</h3><div id="explorer-multi" class="chart"></div></article>
    </div>
  </section>
</main>
</body>
</html>
"""


_STYLES_CSS = """
:root{color-scheme:light;--ink:#18212f;--muted:#637083;--line:#d9e1ea;--panel:#fff;--bg:#f5f7fa;--blue:#2867b2;--green:#187a5b;--red:#b64242;--gold:#9b6b17}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:inherit}
.topbar{position:sticky;top:0;z-index:5;background:#101827;color:white;padding:18px 28px;border-bottom:1px solid #263246}.topbar h1{margin:0 0 4px;font-size:28px;letter-spacing:0}.topbar p{margin:0;color:#cbd5e1;max-width:980px}.topbar nav{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}.topbar a{padding:6px 10px;border:1px solid #3d4b63;border-radius:6px;text-decoration:none;color:#e2e8f0}
main{max-width:1480px;margin:0 auto;padding:26px}section{margin-bottom:34px}h2{font-size:22px;margin:0 0 14px}h3{font-size:15px;margin:0 0 10px;color:#334155}.subhead{margin:20px 0 12px}.grid{display:grid;gap:16px}.grid.two{grid-template-columns:repeat(2,minmax(0,1fr))}.wide{grid-column:1/-1}.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;box-shadow:0 1px 2px rgba(16,24,40,.04)}.chart{min-height:300px;width:100%;overflow:hidden}.small-chart{min-height:235px;width:100%;overflow:hidden}.figure-tools{display:flex;gap:8px;align-items:center;justify-content:flex-end;margin-top:8px}.figure-tools button,.export-card a{border:1px solid var(--line);border-radius:6px;background:white;color:#334155;padding:6px 9px;font:12px/1.2 inherit;text-decoration:none;cursor:pointer}.figure-tools button:hover,.export-card a:hover{border-color:#94a3b8;background:#f8fafc}.family-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.family-card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;box-shadow:0 1px 2px rgba(16,24,40,.04)}.family-card h4{font-size:15px;margin:0 0 4px;color:#1f2937}.family-meta{color:var(--muted);font-size:12px;margin-bottom:8px}.family-meta span{display:inline-block;margin-right:8px}.kpis{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-bottom:16px}.kpi{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}.kpi .label{color:var(--muted);font-size:12px;text-transform:uppercase}.kpi .value{font-size:25px;font-weight:720;margin-top:4px}.kpi .note{font-size:12px;color:var(--muted);margin-top:3px}.axis text{fill:var(--muted);font-size:11px}.axis line,.axis path{stroke:var(--line)}.legend{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;color:var(--muted);font-size:12px}.family-card .legend{max-height:44px;overflow:auto}.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}.table{overflow:auto;max-height:420px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{position:sticky;top:0;background:#f8fafc;color:#475569}.controls{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:14px}.controls.compact{margin-top:-4px}.controls label{display:flex;gap:8px;align-items:center;color:var(--muted)}select{border:1px solid var(--line);border-radius:6px;background:white;padding:7px 9px}.cards,.export-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.export-grid{grid-template-columns:repeat(4,minmax(0,1fr));margin-bottom:16px}.export-card{border:1px solid var(--line);border-radius:8px;padding:12px;background:#fff}.export-card b{display:block;margin-bottom:4px}.export-card p{margin:0 0 8px;color:var(--muted);font-size:12px}.roadmap{border:1px solid var(--line);border-radius:8px;padding:12px;background:#fbfdff}.roadmap b{display:block;margin-bottom:5px}.roadmap p{margin:4px 0;color:#475569}
@media(max-width:900px){main{padding:16px}.grid.two,.kpis,.cards,.family-grid,.export-grid{grid-template-columns:1fr}.topbar{position:static}.chart{min-height:260px}}
"""


_APP_TS = """// TypeScript reference source for the generated CARTS report.
// The runtime report uses assets/app.js directly so it opens from file:// without a build step.
declare global { interface Window { CARTS_REPORT_DATA: any; } }
export {};
"""


_APP_JS = """
(function(){
const data=window.CARTS_REPORT_DATA;
const colors=["#2867b2","#187a5b","#9b6b17","#7b3fb3","#b64242","#0f766e"];
const dataFiles=[
 ["Thread scaling","Single-node thread sweep aggregates","thread_scaling.csv"],
 ["Node scaling","Multinode node sweep aggregates","node_scaling.csv"],
 ["Family thread summary","Single-node per-family subplot source","family_thread_summary.csv"],
 ["Family node summary","Multinode per-family subplot source","family_node_summary.csv"],
 ["Family examples","Best and worst examples per family","family_examples.csv"],
 ["Timing metrics","Startup, kernel, E2E, diagnostics by scale","timing_summary.csv"],
 ["Communication","Run-level communication counters","communication.csv"],
 ["Communication summary","Family-level communication medians","communication_summary.csv"],
 ["Issues","Non-passing and warning rows","issues.csv"],
 ["Flat results","Merged row-level report data","results_flat.csv"]
];
const metricLabels={
 e2e_self_speedup:"E2E self-speedup",
 kernel_self_speedup:"Kernel self-speedup",
 e2e_efficiency:"E2E efficiency",
 kernel_efficiency:"Kernel efficiency",
 arts_e2e_sec:"E2E seconds",
 arts_kernel_sec:"Kernel seconds",
 startup_share_pct:"Startup share",
 remote_bytes_total:"Remote bytes",
 remote_messages_total:"Remote messages"
};
const fmt=(v,d=2)=>v===null||v===undefined||Number.isNaN(Number(v))?"n/a":Number(v).toFixed(d);
const pct=(v)=>v===null||v===undefined?"n/a":`${fmt(v,1)}%`;
const by=(rows,key)=>rows.reduce((m,r)=>{const k=typeof key==="function"?key(r):r[key];(m[k]??=[]).push(r);return m},{});
function el(id){return document.getElementById(id)}
function kpis(){
 const h=data.headline;
 el("kpis").innerHTML=[
  ["Scaling pass rate",pct(h.overall_pass_rate_pct),`${h.overall_passes}/${h.overall_total} CARTS scaling runs`],
  ["Single-node speedup",`${fmt(h.single_node_geomean_e2e_speedup)}x`,`${h.single_node_max_threads||"max"} threads`],
  ["64-node baseline",`${fmt(h.node64_baseline_geomean_e2e_speedup)}x`,"CARTS multinode"],
  ["64-node distributed DB",`${fmt(h.node64_distributed_db_geomean_e2e_speedup)}x`,"CARTS runtime config"],
  ["Counter rows",h.counter_rows||0,"communication metrics"]
 ].map(k=>`<div class="kpi"><div class="label">${k[0]}</div><div class="value">${k[1]}</div><div class="note">${k[2]}</div></div>`).join("");
}
function svgRoot(w,h,title="CARTS Report figure"){return `<svg viewBox="0 0 ${w} ${h}" width="100%" height="100%" role="img" aria-label="${esc(title)}" xmlns="http://www.w3.org/2000/svg"><title>${esc(title)}</title><rect x="0" y="0" width="${w}" height="${h}" fill="#ffffff"/>`}
function lineChart(target,series,xKey,yKey,opts={}){
 const w=opts.width||720,h=opts.height||300,p={l:opts.left||54,r:18,t:18,b:44}; const all=series.flatMap(s=>s.rows.map(r=>({row:r,key:s.yKey||yKey}))).filter(x=>num(x.row[xKey])!==null&&num(x.row[x.key])!==null);
 if(!all.length){el(target).innerHTML="<p>No comparable rows.</p>";return}
 const xs=[...new Set(all.map(x=>num(x.row[xKey])))].sort((a,b)=>a-b); const ys=all.map(x=>num(x.row[x.key]));
 const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=opts.zero?0:Math.min(0,...ys),ymax=Math.max(...ys,opts.ymax||0);
 const sx=x=>p.l+((x-xmin)/(xmax-xmin||1))*(w-p.l-p.r); const sy=y=>h-p.b-((y-ymin)/(ymax-ymin||1))*(h-p.t-p.b);
 let out=svgRoot(w,h,opts.title||metricName(yKey));
 out+=axis(w,h,p,xs,ymin,ymax,sx,sy,opts.yLabel||"");
 series.forEach((s,i)=>{const key=s.yKey||yKey; const rows=s.rows.filter(r=>num(r[xKey])!==null&&num(r[key])!==null).sort((a,b)=>num(a[xKey])-num(b[xKey])); const pts=rows.map(r=>`${sx(num(r[xKey]))},${sy(num(r[key]))}`).join(" "); const c=s.color||colors[i%colors.length]; out+=`<polyline points="${pts}" fill="none" stroke="${c}" stroke-width="3"/>`; rows.forEach(r=>out+=`<circle cx="${sx(num(r[xKey]))}" cy="${sy(num(r[key]))}" r="4" fill="${c}"><title>${s.name}: ${r[xKey]}=${num(r[xKey])}, ${key}=${fmt(num(r[key]))}</title></circle>`)}); 
 out+="</svg>"+legend(series)+figureTools(target); el(target).innerHTML=out;
}
function barChart(target,rows,labelKey,valueKey,opts={}){
 const w=720,h=300,p={l:150,r:18,t:18,b:34}; const vals=rows.map(r=>num(r[valueKey])).filter(v=>v!==null);
 if(!vals.length){el(target).innerHTML="<p>No data.</p>";return}
 const max=Math.max(...vals,opts.max||0); const barH=Math.max(14,(h-p.t-p.b)/rows.length-6);
 let out=svgRoot(w,h,opts.title||metricName(valueKey)); rows.forEach((r,i)=>{const y=p.t+i*(barH+6); const v=num(r[valueKey])||0; const bw=((w-p.l-p.r)*v/(max||1)); const c=opts.color||colors[i%colors.length]; out+=`<text x="${p.l-8}" y="${y+barH*.7}" text-anchor="end" font-size="11" fill="#637083">${short(r[labelKey])}</text><rect x="${p.l}" y="${y}" width="${bw}" height="${barH}" rx="3" fill="${c}"><title>${esc(r[labelKey])}: ${fmt(v)}</title></rect><text x="${p.l+bw+5}" y="${y+barH*.7}" font-size="11" fill="#334155">${opts.percent?pct(v):fmt(v)}</text>`});
 out+="</svg>"+figureTools(target); el(target).innerHTML=out;
}
function heatmap(target,rows,xKey,yKey,valueKey){
 const xs=[...new Set(rows.map(r=>r[xKey]))].sort((a,b)=>Number(a)-Number(b)); const ys=[...new Set(rows.map(r=>r[yKey]))].sort();
 if(!xs.length||!ys.length){el(target).innerHTML="<p>No heatmap data.</p>";return}
 const cellW=70,cellH=26,w=150+cellW*xs.length,h=42+cellH*ys.length;
 const vals=rows.map(r=>num(r[valueKey])).filter(v=>v!==null); const max=Math.max(...vals,1),min=Math.min(...vals,0);
 const m=new Map(rows.map(r=>[`${r[yKey]}|${r[xKey]}`,num(r[valueKey])]));
 let out=svgRoot(w,h,metricName(valueKey)); xs.forEach((x,i)=>out+=`<text x="${150+i*cellW+cellW/2}" y="20" text-anchor="middle" font-size="11" fill="#637083">${x}</text>`);
 ys.forEach((y,j)=>{out+=`<text x="142" y="${42+j*cellH+17}" text-anchor="end" font-size="11" fill="#637083">${short(y,28)}</text>`; xs.forEach((x,i)=>{const v=m.get(`${y}|${x}`); const t=v===null||v===undefined?0:(v-min)/(max-min||1); const fill=v===null||v===undefined?"#eef2f7":mix([231,239,248],[24,122,91],t); out+=`<rect x="${150+i*cellW}" y="${30+j*cellH}" width="${cellW-2}" height="${cellH-2}" rx="2" fill="${fill}"><title>${y} ${x}: ${fmt(v)}</title></rect>`; if(v!==null&&v!==undefined) out+=`<text x="${150+i*cellW+cellW/2}" y="${47+j*cellH}" text-anchor="middle" font-size="10" fill="${t>.55?"white":"#18212f"}">${fmt(v,2)}</text>`})});
 out+="</svg>"+figureTools(target); el(target).innerHTML=out;
}
function table(target,rows,cols,limit=12,csvFile=null){
 if(!rows.length){el(target).innerHTML="<p>No rows.</p>";return}
 const body=rows.slice(0,limit).map(r=>`<tr>${cols.map(c=>`<td>${cell(r[c])}</td>`).join("")}</tr>`).join("");
 const link=csvFile?`<div class="figure-tools"><a href="data/${csvFile}" download>Download CSV</a></div>`:"";
 el(target).innerHTML=`<table><thead><tr>${cols.map(c=>`<th>${c.replaceAll("_"," ")}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table>${link}`;
}
function familySubplots(target,phaseKinds,xKey,yKey){
 const kinds=Array.isArray(phaseKinds)?phaseKinds:[phaseKinds];
 const families=data.families.filter(f=>data.rows.some(r=>r.family===f&&kinds.includes(r.phase_kind)&&r.status==="PASS"));
 el(target).innerHTML=families.map(f=>`<article class="family-card"><h4>${f}</h4><div class="family-meta" id="${safe(target+"-"+f+"-meta")}"></div><div id="${safe(target+"-"+f)}" class="small-chart"></div></article>`).join("");
 families.forEach((family,fi)=>{
  const rows=data.rows.filter(r=>r.family===family&&kinds.includes(r.phase_kind)&&r.status==="PASS"&&num(r[yKey])!==null);
  const passed=rows.length,total=data.rows.filter(r=>r.family===family&&kinds.includes(r.phase_kind)).length;
  const scale=max(rows.map(r=>r[xKey]));
  const atMax=rows.filter(r=>r[xKey]===scale);
  const best=atMax.slice().sort((a,b)=>(num(b[yKey])||0)-(num(a[yKey])||0))[0];
  const worst=atMax.slice().sort((a,b)=>(num(a[yKey])||0)-(num(b[yKey])||0))[0];
  el(safe(target+"-"+family+"-meta")).innerHTML=[
   `<span>${passed}/${total} passing points</span>`,
   `<span>max ${xKey}: ${scale||"n/a"}</span>`,
   `<span>${metricName(yKey)}</span>`,
   best?`<span>best ${short(best.benchmark,30)} ${fmt(best[yKey])}</span>`:"",
   worst&&worst!==best?`<span>worst ${short(worst.benchmark,30)} ${fmt(worst[yKey])}</span>`:""
  ].filter(Boolean).join(" | ");
  const groups=Object.entries(by(rows,r=>kinds.length>1?`${r.benchmark} | ${r.phase_label}`:r.benchmark)).map(([name,rs],i)=>({name,rows:rs,color:colors[i%colors.length]}));
  lineChart(safe(target+"-"+family),groups,xKey,yKey,{zero:metricZero(yKey),width:640,height:235,left:46,ymax:metricMax(yKey)});
 });
}
function setupFamilyControls(){
 const single=el("single-family-metric-select"),multi=el("multi-family-metric-select");
 function drawSingle(){familySubplots("single-family-plots","single_node","threads",single.value)}
 function drawMulti(){familySubplots("multi-family-plots",["multinode","multinode_distributed_db"],"nodes",multi.value)}
 single.addEventListener("change",drawSingle); multi.addEventListener("change",drawMulti);
 drawSingle(); drawMulti();
}
function exportLinks(){
 el("export-links").innerHTML=dataFiles.map(([name,desc,file])=>`<div class="export-card"><b>${name}</b><p>${desc}</p><a href="data/${file}" download>Download CSV</a></div>`).join("");
}
function studyMatrix(){
 const rows=[
  {study:"Single-node CARTS thread sweep",size:"large",scale:"1,2,4,8,16,32,64 threads",benchmarks:data.benchmarks.length,status:`${findScale(data.tables.thread_scaling,"threads",64)?.pass||0}/${findScale(data.tables.thread_scaling,"threads",64)?.total||0} at 64 threads`},
  {study:"Multinode CARTS node sweep",size:"extralarge",scale:"1,2,4,8,16,32,64 nodes",benchmarks:data.benchmarks.length,status:`${findNode("multinode")?.pass||0}/${findNode("multinode")?.total||0} at 64 nodes`},
  {study:"Multinode CARTS distributed DB sweep",size:"extralarge",scale:"1,2,4,8,16,32,64 nodes",benchmarks:data.benchmarks.length,status:`${findNode("multinode_distributed_db")?.pass||0}/${findNode("multinode_distributed_db")?.total||0} at 64 nodes`},
  {study:"Single-node OpenMP comparison",size:"large",scale:"64 threads",benchmarks:data.tables.openmp.length,status:`${data.tables.openmp.length} comparison rows`},
  {study:"Communication counter profile",size:"extralarge",scale:"1,2,4,8 nodes",benchmarks:data.tables.communication_summary.length,status:`${data.tables.communication.length} counter-backed rows`}
 ];
 table("study-matrix",rows,["study","size","scale","benchmarks","status"],20);
}
function scalingSummary(){
 const thread64=findScale(data.tables.thread_scaling,"threads",64)||{};
 const node64=data.tables.node_scaling.filter(r=>r.nodes===64);
 const rows=[
  {view:"Single-node 64-thread",pass:`${thread64.pass||0}/${thread64.total||0}`,e2e_geomean_speedup:thread64.e2e_geomean_speedup,kernel_geomean_speedup:thread64.kernel_geomean_speedup,e2e_efficiency:thread64.e2e_geomean_efficiency},
  ...node64.map(r=>({view:`64-node ${r.phase_label}`,pass:`${r.pass}/${r.total}`,e2e_geomean_speedup:r.e2e_geomean_speedup,kernel_geomean_speedup:r.kernel_geomean_speedup,e2e_efficiency:r.e2e_geomean_efficiency}))
 ];
 table("scaling-summary-table",rows,["view","pass","e2e_geomean_speedup","kernel_geomean_speedup","e2e_efficiency"],10);
 table("family-thread-table",data.tables.family_thread_summary.filter(r=>r.threads===64),["family","pass","total","pass_pct","e2e_geomean_speedup","kernel_geomean_speedup","e2e_geomean_efficiency","median_startup_share_pct"],50,"family_thread_summary.csv");
 table("family-node-table",data.tables.family_node_summary.filter(r=>r.nodes===64),["family","phase_label","pass","total","pass_pct","e2e_geomean_speedup","kernel_geomean_speedup","e2e_geomean_efficiency","median_startup_share_pct"],80,"family_node_summary.csv");
}
function setupDownloads(){
 if(!document.addEventListener||window.__cartsReportDownloads)return;
 window.__cartsReportDownloads=true;
 document.addEventListener("click",ev=>{const btn=ev.target.closest&&ev.target.closest("[data-svg-target]"); if(btn) downloadSvg(btn.getAttribute("data-svg-target"));});
}
function render(){
 setupDownloads();
 kpis();
 exportLinks();
 studyMatrix();
 scalingSummary();
 barChart("phase-pass",data.tables.phase_summary,"phase","pass_pct",{percent:true,color:"#187a5b",max:100,title:"Pass rate by phase"});
 table("issues",data.tables.issues,["benchmark","phase","threads","nodes","status","runtime_warning"],50,"issues.csv");
 lineChart("thread-speedup",[{name:"E2E",rows:data.tables.thread_scaling},{name:"Kernel",rows:data.tables.thread_scaling,yKey:"kernel_geomean_speedup",color:"#187a5b"}],"threads","e2e_geomean_speedup",{zero:true,title:"Single-node geomean speedup"});
 lineChart("thread-efficiency",[{name:"E2E",rows:data.tables.thread_scaling},{name:"Kernel",rows:data.tables.thread_scaling,yKey:"kernel_geomean_efficiency",color:"#187a5b"}],"threads","e2e_geomean_efficiency",{zero:true,ymax:1,title:"Single-node geomean efficiency"});
 table("single-examples",data.tables.top_examples.filter(r=>r.kind==="single-node"),["benchmark","family","scale","e2e_speedup","kernel_speedup","e2e_efficiency","arts_e2e_sec"],30,"top_examples.csv");
 table("single-family-examples",data.tables.family_examples.filter(r=>r.kind==="single-node"),["family","scale","pass","total","pass_pct","geomean_e2e_speedup","best_benchmark","best_e2e_speedup","worst_benchmark","worst_e2e_speedup","median_startup_share_pct"],50,"family_examples.csv");
 const nodeBy=by(data.tables.node_scaling,"phase_label");
 lineChart("node-speedup",Object.entries(nodeBy).map(([name,rows],i)=>({name,rows,color:colors[i]})),"nodes","e2e_geomean_speedup",{zero:true,title:"Multinode geomean speedup"});
 lineChart("node-efficiency",Object.entries(nodeBy).map(([name,rows],i)=>({name,rows,color:colors[i]})),"nodes","e2e_geomean_efficiency",{zero:true,ymax:1,title:"Multinode geomean efficiency"});
 heatmap("family-heatmap",data.tables.family_node_summary,"nodes","family","e2e_geomean_efficiency");
 const gemm=data.rows.filter(r=>r.benchmark==="polybench/gemm"&&r.phase_kind!=="single_node");
 lineChart("gemm",Object.entries(by(gemm,"phase")).map(([name,rows],i)=>({name,rows,color:colors[i]})),"nodes","e2e_self_speedup",{zero:true,title:"GEMM multinode speedup"});
 table("multi-examples",data.tables.top_examples.filter(r=>r.kind==="multinode"),["benchmark","family","phase","scale","e2e_speedup","kernel_speedup","e2e_efficiency","arts_e2e_sec"],40,"top_examples.csv");
 table("multi-family-examples",data.tables.family_examples.filter(r=>r.kind!=="single-node"),["kind","family","scale","pass","total","pass_pct","geomean_e2e_speedup","best_benchmark","best_e2e_speedup","worst_benchmark","worst_e2e_speedup","median_startup_share_pct"],80,"family_examples.csv");
 barChart("openmp-bars",data.tables.openmp.slice().sort((a,b)=>(b.carts_vs_openmp||0)-(a.carts_vs_openmp||0)).slice(0,12),"benchmark","carts_vs_openmp",{color:"#2867b2",title:"CARTS over OpenMP ratio"});
 table("openmp-table",data.tables.openmp,["benchmark","threads","arts_e2e_sec","openmp_e2e_sec","carts_vs_openmp"],50,"results_flat.csv");
 const timingBy=by(data.tables.timing_summary,"phase_label");
 lineChart("startup-share",Object.entries(timingBy).map(([name,rows],i)=>({name,rows,color:colors[i]})),"scale","median_startup_share_pct",{zero:true,ymax:100,title:"Median startup share"});
 lineChart("median-e2e",Object.entries(timingBy).map(([name,rows],i)=>({name,rows,color:colors[i]})),"scale","median_e2e_sec",{zero:true,title:"Median E2E time"});
 const diag=data.tables.timing_summary.map(r=>({...r,diag_total:(r.srun_error_count||0)+(r.runtime_warning_count||0),label:`${r.phase_label} ${r.scale} ${r.scale_label}`}));
 barChart("diagnostics",diag.filter(r=>r.diag_total>0).slice(-18),"label","diag_total",{color:"#b64242"});
 table("timing-table",data.tables.timing_summary,["phase_label","scale","scale_label","pass","total","median_e2e_sec","median_startup_share_pct","median_kernel_share_pct","srun_error_count","runtime_warning_count"],40,"timing_summary.csv");
 const commByNodes=Object.values(by(data.tables.communication,"nodes")).map(rows=>({nodes:rows[0].nodes,remote_bytes_total:median(rows.map(r=>r.remote_bytes_total)),connect_success_pct:median(rows.map(r=>r.connect_success_pct))})).sort((a,b)=>a.nodes-b.nodes);
 lineChart("comm-bytes",[{name:"median remote bytes",rows:commByNodes}],"nodes","remote_bytes_total",{zero:true,title:"Median remote bytes"});
 lineChart("comm-connect",[{name:"median connect success",rows:commByNodes,color:"#187a5b"}],"nodes","connect_success_pct",{zero:true,ymax:100,title:"Connection success"});
 table("comm-family-table",data.tables.communication_summary,["family","nodes","counter_rows","median_remote_bytes_total","median_remote_messages_total","median_remote_bytes_per_message","median_remote_bytes_per_sec","median_connect_success_pct"],80,"communication_summary.csv");
 el("optimizations").innerHTML=data.optimizations.map(r=>`<div class="roadmap"><b>${r.priority}. ${r.area}</b><p>${r.evidence}</p><p>${r.proposal}</p></div>`).join("");
 setupFamilyControls();
 setupExplorer();
}
function setupExplorer(){
 const b=el("benchmark-select"),m=el("metric-select"); b.innerHTML=data.benchmarks.map(x=>`<option>${x}</option>`).join("");
 if(!b.value&&data.benchmarks.length)b.value=data.benchmarks[0];
 if(!m.value)m.value="e2e_self_speedup";
 function draw(){const bench=b.value,metric=m.value; const rows=data.rows.filter(r=>r.benchmark===bench&&r.status==="PASS"); lineChart("explorer-single",[{name:bench,rows:rows.filter(r=>r.phase_kind==="single_node")}],"threads",metric,{zero:metric.includes("speedup")||metric.includes("efficiency")}); const groups=Object.entries(by(rows.filter(r=>r.phase_kind!=="single_node"),"phase")).map(([name,rows],i)=>({name,rows,color:colors[i]})); lineChart("explorer-multi",groups,"nodes",metric,{zero:metric.includes("speedup")||metric.includes("efficiency")});}
 b.addEventListener("change",draw); m.addEventListener("change",draw); draw();
}
function figureTools(target){return `<div class="figure-tools"><button type="button" data-svg-target="${target}">Download SVG</button></div>`}
function downloadSvg(target){
 const host=el(target),svg=host&&host.querySelector?host.querySelector("svg"):null;
 if(!svg)return;
 const text=new XMLSerializer().serializeToString(svg);
 const blob=new Blob([text],{type:"image/svg+xml"});
 const url=URL.createObjectURL(blob);
 const a=document.createElement("a");
 a.href=url; a.download=`${safe(target)}.svg`; document.body.appendChild(a); a.click(); a.remove();
 URL.revokeObjectURL(url);
}
function axis(w,h,p,xs,ymin,ymax,sx,sy,label){let out=""; for(let i=0;i<=4;i++){const yv=ymin+(ymax-ymin)*i/4,y=sy(yv); out+=`<line x1="${p.l}" x2="${w-p.r}" y1="${y}" y2="${y}" stroke="#e2e8f0"/><text x="${p.l-8}" y="${y+4}" text-anchor="end" font-size="11" fill="#637083">${fmt(yv,1)}</text>`} xs.forEach(x=>out+=`<text x="${sx(x)}" y="${h-18}" text-anchor="middle" font-size="11" fill="#637083">${x}</text>`); out+=`<line x1="${p.l}" x2="${w-p.r}" y1="${h-p.b}" y2="${h-p.b}" stroke="#cbd5e1"/><line x1="${p.l}" x2="${p.l}" y1="${p.t}" y2="${h-p.b}" stroke="#cbd5e1"/>`; return out}
function legend(series){return `<div class="legend">${series.map((s,i)=>`<span><span class="swatch" style="background:${s.color||colors[i%colors.length]}"></span>${s.name}</span>`).join("")}</div>`}
function num(v){const n=Number(v);return Number.isFinite(n)?n:null}
function short(v,n=22){v=String(v);return v.length>n?v.slice(0,n-1)+"...":v}
function cell(v){if(typeof v==="number")return fmt(v); if(v===null||v===undefined)return ""; return String(v)}
function metricName(k){return metricLabels[k]||k.replaceAll("_"," ")}
function metricZero(k){return k.includes("speedup")||k.includes("efficiency")||k.includes("sec")||k.includes("share")}
function metricMax(k){return k.includes("efficiency")?1:(k.includes("share")?100:undefined)}
function findScale(rows,key,value){return rows.find(r=>r[key]===value)}
function findNode(kind){return data.tables.node_scaling.find(r=>r.phase_kind===kind&&r.nodes===64)}
function median(vals){vals=vals.map(num).filter(v=>v!==null).sort((a,b)=>a-b); if(!vals.length)return null; const i=Math.floor(vals.length/2); return vals.length%2?vals[i]:(vals[i-1]+vals[i])/2}
function mix(a,b,t){const c=a.map((x,i)=>Math.round(x+(b[i]-x)*t));return `rgb(${c[0]},${c[1]},${c[2]})`}
function max(vals){vals=vals.map(num).filter(v=>v!==null);return vals.length?Math.max(...vals):null}
function safe(v){return String(v).replace(/[^a-zA-Z0-9_-]/g,"_")}
function esc(v){return String(v).replace(/[&<>"']/g,c=>c==="&"?"&amp;":c==="<"?"&lt;":c===">"?"&gt;":c==='"'?"&quot;":"&#39;")}
render();
})();
"""
