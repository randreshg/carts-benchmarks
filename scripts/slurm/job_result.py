#!/usr/bin/env python3
"""
SLURM Job Result Generator

Standalone script executed inside each SLURM job to parse benchmark output
and generate a result.json file. This script is called at the end of each
sbatch job to capture timing, checksums, and counter data.

Usage:
    python3 slurm_job_result.py \\
        --benchmark "polybench/gemm" \\
        --run-number 1 \\
        --size "medium" \\
        --arts-exit 0 \\
        --arts-duration 1.234 \\
        --omp-exit -1 \\
        --omp-duration 0 \\
        --counter-dir /path/to/counters \\
        --output /path/to/result.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure benchmark_common is importable (script may run from arbitrary CWD in SLURM).
# benchmark_common.py lives in the parent scripts/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_common import (
    parse_checksum,
    parse_kernel_timings,
    parse_e2e_timings,
    parse_init_timings,
    parse_counter_json,
)
from benchmark_models import Status, VerificationResult
from benchmark_verification import verify_against_omp, verify_against_reference


def read_slurm_output(output_dir: Path, job_id: str) -> Tuple[str, str]:
    """Read SLURM stdout and stderr files.

    Looks for slurm.out/slurm.err first (matching sbatch template), then
    falls back to slurm-{job_id}.out/slurm-{job_id}.err for backward compat.

    Args:
        output_dir: Directory containing SLURM output files
        job_id: SLURM job ID

    Returns:
        Tuple of (stdout, stderr) contents
    """
    stdout_candidates = [output_dir / "slurm.out", output_dir / f"slurm-{job_id}.out"]
    stderr_candidates = [output_dir / "slurm.err", output_dir / f"slurm-{job_id}.err"]

    stdout = ""
    for candidate in stdout_candidates:
        if candidate.exists():
            try:
                stdout = candidate.read_text()
            except Exception:
                pass
            break

    stderr = ""
    for candidate in stderr_candidates:
        if candidate.exists():
            try:
                stderr = candidate.read_text()
            except Exception:
                pass
            break

    return stdout, stderr


def determine_status(
    arts_exit: int,
    omp_exit: int,
    arts_checksum: Optional[str],
    omp_checksum: Optional[str],
    reference_checksum: Optional[str] = None,
    reference_source: Optional[str] = None,
    reference_omp_threads: Optional[int] = None,
    tolerance: float = 0.01,
) -> Tuple[str, VerificationResult]:
    """Determine overall status and verification result.

    Args:
        arts_exit: ARTS exit code
        omp_exit: OpenMP exit code (-1 if skipped)
        arts_checksum: ARTS checksum
        omp_checksum: OpenMP checksum
        tolerance: Tolerance for checksum comparison

    Returns:
        Tuple of (status, verification_result)
    """
    arts_status = Status.PASS if arts_exit == 0 else Status.FAIL
    if omp_exit == -1:
        verification = verify_against_reference(
            arts_status,
            arts_checksum,
            reference_checksum,
            tolerance,
            reference_source=reference_source,
            reference_omp_threads=reference_omp_threads,
        )
    else:
        omp_status = Status.PASS if omp_exit == 0 else Status.FAIL
        verification = verify_against_omp(
            arts_status,
            arts_checksum,
            omp_status,
            omp_checksum,
            tolerance,
        )

    return ("PASS" if verification.correct else "FAIL", verification)


def summarize_slurm_logs(stdout: str, stderr: str, include_tails: bool) -> Dict[str, Any]:
    """Summarize SLURM log content for debugging failed runs."""
    stdout_lines = stdout.splitlines()
    stderr_lines = stderr.splitlines()

    slurm_stderr_summary: Dict[str, Any] = {
        "line_count": len(stderr_lines),
        "srun_error_count": len(re.findall(r"^srun: error:", stderr, flags=re.MULTILINE)),
        "broken_pipe_count": len(re.findall(r"Broken pipe", stderr)),
        "counter_timeout_warnings": len(re.findall(r"Could not read counter file", stderr)),
        "remote_send_hard_timeout_count": len(re.findall(r"Remote send hard-timeout", stderr)),
        "connection_refused_count": len(re.findall(r"Connection refused", stderr)),
    }

    warning_reasons: List[str] = []
    if slurm_stderr_summary["srun_error_count"] > 0:
        warning_reasons.append(f"srun_error_count={slurm_stderr_summary['srun_error_count']}")
    if slurm_stderr_summary["broken_pipe_count"] > 0:
        warning_reasons.append(f"broken_pipe_count={slurm_stderr_summary['broken_pipe_count']}")
    if slurm_stderr_summary["counter_timeout_warnings"] > 0:
        warning_reasons.append(
            f"counter_timeout_warnings={slurm_stderr_summary['counter_timeout_warnings']}"
        )
    if slurm_stderr_summary["remote_send_hard_timeout_count"] > 0:
        warning_reasons.append(
            "remote_send_hard_timeout_count="
            f"{slurm_stderr_summary['remote_send_hard_timeout_count']}"
        )
    if slurm_stderr_summary["connection_refused_count"] > 0:
        warning_reasons.append(
            f"connection_refused_count={slurm_stderr_summary['connection_refused_count']}"
        )

    summary: Dict[str, Any] = {
        "slurm_stdout": {
            "line_count": len(stdout_lines),
        },
        "slurm_stderr": slurm_stderr_summary,
        "runtime_warning": {
            "has_warning": bool(warning_reasons),
            "reasons": warning_reasons,
        },
    }

    if include_tails or warning_reasons:
        summary["slurm_stdout"]["tail"] = stdout_lines[-40:]
        summary["slurm_stderr"]["tail"] = stderr_lines[-40:]

    return summary


def generate_result(
    benchmark: str,
    run_number: int,
    size: str,
    arts_exit: int,
    arts_duration: float,
    omp_exit: int,
    omp_duration: float,
    counter_dir: Optional[Path],
    slurm_job_id: str,
    slurm_nodelist: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Generate a complete result dictionary.

    Args:
        benchmark: Benchmark name
        run_number: Run number
        size: Dataset size
        arts_exit: ARTS exit code
        arts_duration: ARTS wall clock duration
        omp_exit: OpenMP exit code (-1 if skipped)
        omp_duration: OpenMP wall clock duration
        counter_dir: Directory containing counter files
        slurm_job_id: SLURM job ID
        slurm_nodelist: SLURM node list
        output_dir: Job output directory

    Returns:
        Result dictionary
    """
    run_config: Dict[str, Any] = {}
    run_config_file = output_dir / "run_config.json"
    if run_config_file.exists():
        try:
            payload = json.loads(run_config_file.read_text())
            if isinstance(payload, dict):
                run_config = payload
        except Exception:
            run_config = {}

    reference_payload = run_config.get("reference")
    if not isinstance(reference_payload, dict):
        reference_payload = {}
    reference_checksum = reference_payload.get("checksum")
    reference_source = reference_payload.get("source")
    reference_omp_threads = reference_payload.get("omp_threads")

    # Read SLURM output for parsing
    stdout, stderr = read_slurm_output(output_dir, slurm_job_id)

    # Split stdout into ARTS and OpenMP sections
    # Format: [ARTS] ... [OpenMP] ...
    arts_section = stdout
    omp_section = ""
    if "[OpenMP]" in stdout:
        parts = stdout.split("[OpenMP]", 1)
        arts_section = parts[0]
        omp_section = parts[1] if len(parts) > 1 else ""

    # Parse ARTS output (from ARTS section)
    arts_checksum = parse_checksum(arts_section)
    arts_kernel = parse_kernel_timings(arts_section)
    arts_e2e = parse_e2e_timings(arts_section)
    arts_init = parse_init_timings(arts_section)

    # Parse counter data if available
    counter_init_sec = None
    counter_e2e_sec = None
    if counter_dir and counter_dir.exists():
        counter_init_sec, counter_e2e_sec = parse_counter_json(counter_dir)

    # OpenMP results (only if it ran, parse from OMP section)
    omp_checksum = None
    omp_kernel = {}
    omp_e2e = {}
    omp_init = {}
    if omp_exit != -1 and omp_section:
        omp_checksum = parse_checksum(omp_section)
        omp_kernel = parse_kernel_timings(omp_section)
        omp_e2e = parse_e2e_timings(omp_section)
        omp_init = parse_init_timings(omp_section)

    # Determine status
    status, verification_result = determine_status(
        arts_exit,
        omp_exit,
        arts_checksum,
        omp_checksum,
        reference_checksum=(
            str(reference_checksum) if reference_checksum is not None else None
        ),
        reference_source=(
            str(reference_source) if reference_source is not None else None
        ),
        reference_omp_threads=(
            int(reference_omp_threads) if reference_omp_threads is not None else None
        ),
    )
    diagnostics = summarize_slurm_logs(stdout, stderr, include_tails=(status != "PASS"))

    # Build result
    result = {
        "benchmark": benchmark,
        "run_number": run_number,
        "size": size,
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "verification": {
            "note": verification_result.note,
            "arts_checksum": verification_result.arts_checksum,
            "omp_checksum": verification_result.omp_checksum,
            "reference_checksum": verification_result.reference_checksum,
            "reference_source": verification_result.reference_source,
            "mode": verification_result.mode,
            "reference_omp_threads": verification_result.reference_omp_threads,
        },
        "arts": {
            "exit_code": arts_exit,
            "duration_sec": arts_duration,
            "checksum": arts_checksum,
            "init_sec": counter_init_sec,
            "e2e_sec": counter_e2e_sec,
            "kernel_timings": arts_kernel,
            "e2e_timings": arts_e2e,
            "init_timings": arts_init,
        },
        "omp": {
            "exit_code": omp_exit,
            "duration_sec": omp_duration if omp_exit != -1 else None,
            "checksum": omp_checksum if omp_exit != -1 else None,
            "kernel_timings": omp_kernel if omp_exit != -1 else {},
            "e2e_timings": omp_e2e if omp_exit != -1 else {},
            "init_timings": omp_init if omp_exit != -1 else {},
            "skipped": omp_exit == -1,
        },
        "slurm": {
            "job_id": slurm_job_id,
            "nodelist": slurm_nodelist,
        },
        "artifacts": {
            "run_dir": str(output_dir),
            "slurm_out": str(output_dir / "slurm.out"),
            "slurm_err": str(output_dir / "slurm.err"),
            "counter_dir": str(counter_dir) if counter_dir else None,
        },
        "diagnostics": diagnostics,
    }

    if status == "PASS" and diagnostics.get("runtime_warning", {}).get("has_warning"):
        result["status_detail"] = "WARN"

    # Compute speedup if both ran
    if omp_exit != -1 and omp_duration > 0 and arts_duration > 0:
        result["speedup"] = omp_duration / arts_duration
    else:
        result["speedup"] = None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate result.json from SLURM job output"
    )
    parser.add_argument("--benchmark", required=True, help="Benchmark name")
    parser.add_argument("--run-number", type=int, required=True, help="Run number")
    parser.add_argument("--size", required=True, help="Dataset size")
    parser.add_argument("--arts-exit", type=int, required=True, help="ARTS exit code")
    parser.add_argument("--arts-duration", type=float, required=True, help="ARTS duration")
    parser.add_argument("--omp-exit", type=int, required=True, help="OpenMP exit code (-1 if skipped)")
    parser.add_argument("--omp-duration", type=float, required=True, help="OpenMP duration")
    parser.add_argument("--counter-dir", type=Path, help="Counter directory")
    parser.add_argument("--slurm-job-id", default="", help="SLURM job ID")
    parser.add_argument("--slurm-nodelist", default="", help="SLURM node list")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")

    args = parser.parse_args()

    # Generate result
    result = generate_result(
        benchmark=args.benchmark,
        run_number=args.run_number,
        size=args.size,
        arts_exit=args.arts_exit,
        arts_duration=args.arts_duration,
        omp_exit=args.omp_exit,
        omp_duration=args.omp_duration,
        counter_dir=args.counter_dir,
        slurm_job_id=args.slurm_job_id,
        slurm_nodelist=args.slurm_nodelist,
        output_dir=args.output.parent,
    )

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Result written to: {args.output}")
    print(f"Status: {result['status']}")


if __name__ == "__main__":
    main()
