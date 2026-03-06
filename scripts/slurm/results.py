"""SLURM result collection and failure-row synthesis for benchmark runs."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from benchmark_common import aggregate_perf_csvs, parse_perf_csv

from .models import SlurmJobStatus, SubmissionFailure


def _parse_key_value_tokens(text: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key] = value
    return parsed


def _run_snapshot_cmd(command: List[str], timeout: int) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "command": " ".join(command),
        "ok": False,
    }
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        snapshot["return_code"] = proc.returncode
        snapshot["stdout"] = (proc.stdout or "").strip()
        snapshot["stderr"] = (proc.stderr or "").strip()
        snapshot["ok"] = proc.returncode == 0
    except FileNotFoundError:
        snapshot["error"] = f"{command[0]} not found"
    except subprocess.TimeoutExpired:
        snapshot["error"] = f"{command[0]} timed out after {timeout}s"
    except Exception as exc:
        snapshot["error"] = str(exc)
    return snapshot


def _load_run_config(run_dir: Path) -> Dict[str, Any]:
    run_config_file = run_dir / "run_config.json"
    if not run_config_file.exists():
        return {}
    try:
        payload = json.loads(run_config_file.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _apply_run_config(result: Dict[str, Any], run_config: Dict[str, Any]) -> None:
    if not run_config:
        return
    result["threads"] = run_config.get("threads")
    result["nodes"] = run_config.get("nodes")
    result["run_phase"] = run_config.get("run_phase")
    if "profile" in run_config:
        result["profile"] = run_config.get("profile")
    if "perf" in run_config:
        result["perf"] = run_config.get("perf")
    if "perf_interval" in run_config:
        result["perf_interval"] = run_config.get("perf_interval")
    if "size" in run_config:
        result["size"] = run_config.get("size")
    if "compile_args" in run_config:
        result["compile_args"] = run_config.get("compile_args")
    if "cflags" in run_config:
        result["cflags"] = run_config.get("cflags")
    if "config" in run_config and isinstance(run_config["config"], dict):
        result["config"] = run_config["config"]
    if "reference" in run_config and isinstance(run_config["reference"], dict):
        verification = result.setdefault("verification", {})
        reference = run_config["reference"]
        if "checksum" in reference:
            verification.setdefault("reference_checksum", reference.get("checksum"))
        if "source" in reference:
            verification.setdefault("reference_source", reference.get("source"))


def _apply_compile_artifact_paths(result: Dict[str, Any], run_config: Dict[str, Any]) -> None:
    """Attach compile-time artifact locations from run_config when available."""
    if not run_config:
        return

    arts_cfg_source = run_config.get("arts_cfg_source")
    if not arts_cfg_source:
        return

    try:
        arts_cfg_path = Path(str(arts_cfg_source)).resolve()
    except Exception:
        return

    artifacts = result.setdefault("artifacts", {})
    artifacts.setdefault("arts_config", str(arts_cfg_path))
    artifacts.setdefault("build_dir", str(arts_cfg_path.parent))

    benchmark_name = str(run_config.get("benchmark") or "")
    example_name = benchmark_name.split("/")[-1] if benchmark_name else ""
    if example_name:
        arts_exe = arts_cfg_path.parent / f"{example_name}_arts"
        omp_exe = arts_cfg_path.parent / f"{example_name}_omp"
        if arts_exe.exists():
            artifacts.setdefault("executable_arts", str(arts_exe))
        if omp_exe.exists():
            artifacts.setdefault("executable_omp", str(omp_exe))


def _apply_perf_artifacts(result: Dict[str, Any], run_dir: Path) -> None:
    perf_dir = run_dir / "perf"
    if not perf_dir.exists():
        return

    arts_perf_files = sorted(perf_dir.glob("arts_node_*.csv"))
    omp_perf_file = perf_dir / "omp.csv"

    artifacts = result.setdefault("artifacts", {})
    artifacts["perf_dir"] = str(perf_dir)
    if arts_perf_files:
        artifacts["perf_files"] = [str(path) for path in arts_perf_files]
    if omp_perf_file.exists():
        artifacts["perf_omp_file"] = str(omp_perf_file)

    if arts_perf_files:
        perf_metrics = aggregate_perf_csvs(arts_perf_files)
        if perf_metrics:
            arts = result.setdefault("arts", {})
            arts["perf_metrics"] = perf_metrics
            arts["perf_csv_path"] = str(perf_dir)
    if omp_perf_file.exists():
        omp_perf_metrics = parse_perf_csv(omp_perf_file)
        if omp_perf_metrics:
            omp = result.setdefault("omp", {})
            omp["perf_metrics"] = omp_perf_metrics
            omp["perf_csv_path"] = str(omp_perf_file)


def _summarize_log(log_path: Path, tail_lines: int = 40) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "path": str(log_path),
        "exists": log_path.exists(),
    }
    if not log_path.exists():
        return summary
    try:
        text = log_path.read_text(errors="replace")
    except Exception as exc:
        summary["read_error"] = str(exc)
        return summary

    lines = text.splitlines()
    summary["line_count"] = len(lines)
    summary["tail"] = lines[-tail_lines:]
    if log_path.name.endswith(".err"):
        summary["broken_pipe_count"] = len(re.findall(r"Broken pipe", text))
        summary["srun_error_count"] = len(
            re.findall(r"^srun: error:", text, flags=re.MULTILINE)
        )
        summary["counter_timeout_warnings"] = len(
            re.findall(r"Could not read counter file", text)
        )
        summary["remote_send_hard_timeout_count"] = len(
            re.findall(r"Remote send hard-timeout", text)
        )
        summary["connection_refused_count"] = len(
            re.findall(r"Connection refused", text)
        )
    return summary


def _slurm_err_summary(diagnostics: Any) -> Dict[str, Any]:
    if not isinstance(diagnostics, dict):
        return {}
    slurm_stderr = diagnostics.get("slurm_stderr")
    if isinstance(slurm_stderr, dict):
        return slurm_stderr
    slurm_err = diagnostics.get("slurm_err")
    if isinstance(slurm_err, dict):
        return slurm_err
    return {}


def _runtime_warning_reasons(diagnostics: Any) -> List[str]:
    slurm_err = _slurm_err_summary(diagnostics)
    reasons: List[str] = []
    if not slurm_err:
        return reasons

    warning_counts = {
        "srun_error_count": int(slurm_err.get("srun_error_count") or 0),
        "broken_pipe_count": int(slurm_err.get("broken_pipe_count") or 0),
        "counter_timeout_warnings": int(slurm_err.get("counter_timeout_warnings") or 0),
        "remote_send_hard_timeout_count": int(
            slurm_err.get("remote_send_hard_timeout_count") or 0
        ),
        "connection_refused_count": int(slurm_err.get("connection_refused_count") or 0),
    }
    for key, value in warning_counts.items():
        if value > 0:
            reasons.append(f"{key}={value}")
    return reasons


class SlurmResultCollector:
    """Collect and normalize per-job results for a SLURM experiment."""

    def __init__(self, job_statuses: Dict[str, SlurmJobStatus]) -> None:
        self.job_statuses = job_statuses
        self.snapshot_cache: Dict[str, Dict[str, Any]] = {}

    def _collect_slurm_snapshot(self, job_id: str) -> Dict[str, Any]:
        if job_id in self.snapshot_cache:
            return self.snapshot_cache[job_id]

        snapshot: Dict[str, Any] = {
            "captured_at": datetime.now().isoformat(),
            "job_id": job_id,
        }

        sacct_cmd = [
            "sacct",
            f"--jobs={job_id}",
            "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList,Start,End,Reason",
            "--parsable2",
            "--noheader",
        ]
        sacct = _run_snapshot_cmd(sacct_cmd, timeout=20)
        if sacct.get("ok") and sacct.get("stdout"):
            lines = [line for line in str(sacct["stdout"]).splitlines() if line.strip()]
            primary = None
            for line in lines:
                parts = line.split("|")
                if parts and parts[0] == job_id:
                    primary = line
                    break
            if primary is None:
                primary = lines[0] if lines else ""
            if primary:
                parts = primary.split("|")
                if len(parts) >= 8:
                    exit_parts = parts[2].split(":", 1)
                    sacct["parsed"] = {
                        "job_id_raw": parts[0],
                        "state": parts[1],
                        "exit_code": parts[2],
                        "elapsed": parts[3],
                        "nodelist": parts[4],
                        "start": parts[5],
                        "end": parts[6],
                        "reason": parts[7],
                        "exit_status": int(exit_parts[0]) if exit_parts[0].isdigit() else None,
                        "exit_signal": (
                            int(exit_parts[1])
                            if len(exit_parts) > 1 and exit_parts[1].isdigit()
                            else None
                        ),
                    }
            sacct["lines"] = lines[-20:]
        snapshot["sacct"] = sacct

        scontrol_cmd = ["scontrol", "show", "job", job_id]
        scontrol = _run_snapshot_cmd(scontrol_cmd, timeout=20)
        if scontrol.get("ok") and scontrol.get("stdout"):
            parsed = _parse_key_value_tokens(str(scontrol["stdout"]))
            keys = [
                "JobId",
                "JobName",
                "JobState",
                "Reason",
                "ExitCode",
                "RunTime",
                "SubmitTime",
                "StartTime",
                "EndTime",
                "NodeList",
                "BatchHost",
                "NumNodes",
                "NumCPUs",
                "NumTasks",
                "CPUs/Task",
            ]
            scontrol["parsed"] = {k: parsed.get(k) for k in keys if k in parsed}
            scontrol["stdout_tail"] = str(scontrol["stdout"]).splitlines()[-60:]
        snapshot["scontrol"] = scontrol

        self.snapshot_cache[job_id] = snapshot
        return snapshot

    def _build_failure_result(
        self,
        status: SlurmJobStatus,
        error: str,
        run_dir: Path,
        run_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        snapshot = self._collect_slurm_snapshot(status.job_id)
        parsed = snapshot.get("scontrol", {}).get("parsed", {})
        snapshot_nodelist = parsed.get("NodeList")
        resolved_nodelist = status.node_list or snapshot_nodelist
        result_status = "TIMEOUT" if status.state == "TIMEOUT" else "FAIL"
        failure_error = error
        if status.state == "TIMEOUT" and error.startswith("No result.json found"):
            failure_error = (
                f"SLURM job timed out before result.json was written: {run_dir} "
                "(increase --time-limit or reduce workload)."
            )
        failure: Dict[str, Any] = {
            "benchmark": status.benchmark_name,
            "run_number": status.run_number,
            "status": result_status,
            "slurm": {
                "job_id": status.job_id,
                "state": status.state,
                "exit_code": status.exit_code,
                "elapsed": status.elapsed,
                "nodelist": resolved_nodelist,
            },
            "error": failure_error,
            "_run_dir": str(run_dir),
            "artifacts": {
                "run_dir": str(run_dir),
                "run_config": str(run_dir / "run_config.json"),
                "result_json": str(run_dir / "result.json"),
                "slurm_out": str(run_dir / "slurm.out"),
                "slurm_err": str(run_dir / "slurm.err"),
            },
            "diagnostics": {
                "slurm_out": _summarize_log(run_dir / "slurm.out"),
                "slurm_err": _summarize_log(run_dir / "slurm.err"),
                "slurm_snapshot": snapshot,
            },
        }
        _apply_run_config(failure, run_config)
        _apply_compile_artifact_paths(failure, run_config)
        _apply_perf_artifacts(failure, run_dir)
        warning_reasons = _runtime_warning_reasons(failure.get("diagnostics"))
        if warning_reasons:
            failure.setdefault("diagnostics", {})["runtime_warning"] = {
                "has_warning": True,
                "reasons": warning_reasons,
            }
        return failure

    def collect(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        for job_id, status in self.job_statuses.items():
            if status.state == "DRY_RUN":
                continue

            if not status.run_dir:
                results.append({
                    "benchmark": status.benchmark_name,
                    "run_number": status.run_number,
                    "status": "FAIL",
                    "slurm": {
                        "job_id": job_id,
                        "state": status.state,
                        "exit_code": status.exit_code,
                    },
                    "error": "Missing run_dir in SLURM job status",
                    "diagnostics": {
                        "slurm_snapshot": self._collect_slurm_snapshot(job_id),
                    },
                })
                continue

            result_file = status.run_dir / "result.json"
            run_config = _load_run_config(status.run_dir)

            if result_file.exists():
                try:
                    result = json.loads(result_file.read_text())
                    slurm_info = result.setdefault("slurm", {})
                    effective_state = status.state
                    if effective_state == "UNKNOWN" and result.get("status") == "PASS":
                        effective_state = "COMPLETED"
                        result.setdefault("diagnostics", {}).setdefault(
                            "slurm_state_inference",
                            {
                                "state": "COMPLETED",
                                "reason": (
                                    "Inferred from a successful result.json because "
                                    "SLURM accounting did not return a final state."
                                ),
                            },
                        )
                    slurm_info["state"] = effective_state
                    if status.exit_code is not None:
                        slurm_info["exit_code"] = status.exit_code
                    if status.elapsed is not None:
                        slurm_info["elapsed"] = status.elapsed
                    if status.node_list:
                        slurm_info["nodelist"] = status.node_list
                    result["_run_dir"] = str(status.run_dir)
                    _apply_run_config(result, run_config)
                    result.setdefault("artifacts", {}).update({
                        "run_dir": str(status.run_dir),
                        "run_config": str(status.run_dir / "run_config.json"),
                        "result_json": str(result_file),
                        "slurm_out": str(status.run_dir / "slurm.out"),
                        "slurm_err": str(status.run_dir / "slurm.err"),
                    })
                    _apply_compile_artifact_paths(result, run_config)
                    _apply_perf_artifacts(result, status.run_dir)

                    warning_reasons = _runtime_warning_reasons(result.get("diagnostics"))
                    if warning_reasons:
                        diagnostics = result.setdefault("diagnostics", {})
                        diagnostics["runtime_warning"] = {
                            "has_warning": True,
                            "reasons": warning_reasons,
                        }
                        if str(result.get("status", "")).upper() == "PASS":
                            result["status_detail"] = "WARN"

                    if result.get("status") != "PASS" or warning_reasons:
                        result.setdefault("diagnostics", {}).setdefault(
                            "slurm_snapshot",
                            self._collect_slurm_snapshot(status.job_id),
                        )
                    results.append(result)
                except json.JSONDecodeError:
                    results.append(
                        self._build_failure_result(
                            status,
                            "Failed to parse result.json",
                            status.run_dir,
                            run_config,
                        )
                    )
            else:
                results.append(
                    self._build_failure_result(
                        status,
                        f"No result.json found in {status.run_dir}",
                        status.run_dir,
                        run_config,
                    )
                )

        return results


def collect_results(
    job_statuses: Dict[str, SlurmJobStatus],
    experiment_dir: Path,
) -> List[Dict[str, Any]]:
    """Collect results from completed SLURM jobs."""
    _ = experiment_dir
    return SlurmResultCollector(job_statuses).collect()


def build_submission_failure_results(
    failures: List[SubmissionFailure],
) -> List[Dict[str, Any]]:
    """Build synthetic result rows for sbatch submission failures."""
    results: List[Dict[str, Any]] = []

    for failure in failures:
        run_dir = failure.run_dir
        run_config = _load_run_config(run_dir)

        result: Dict[str, Any] = {
            "benchmark": failure.benchmark_name,
            "run_number": failure.run_number,
            "status": "FAIL",
            "status_detail": "SUBMIT_FAILED",
            "error": f"SLURM submission failed: {failure.error}",
            "_run_dir": str(run_dir),
            "slurm": {
                "job_id": None,
                "state": "SUBMIT_FAILED",
                "exit_code": None,
                "elapsed": None,
                "nodelist": None,
            },
            "artifacts": {
                "run_dir": str(run_dir),
                "run_config": str(run_dir / "run_config.json"),
                "result_json": str(run_dir / "result.json"),
                "slurm_out": str(run_dir / "slurm.out"),
                "slurm_err": str(run_dir / "slurm.err"),
                "sbatch_script": str(failure.script_path),
            },
            "diagnostics": {
                "submission_error": failure.error,
                "sbatch_script": str(failure.script_path),
            },
        }

        _apply_run_config(result, run_config)
        _apply_compile_artifact_paths(result, run_config)
        _apply_perf_artifacts(result, run_dir)
        results.append(result)

    return results
