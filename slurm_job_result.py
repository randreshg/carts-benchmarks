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


# ============================================================================
# Checksum and Timing Patterns
# ============================================================================

CHECKSUM_PATTERNS = [
    r"checksum[:\s]*=?\s*([0-9.eE+-]+)",
    r"result[:\s]*=?\s*([0-9.eE+-]+)",
    r"sum[:\s]*=?\s*([0-9.eE+-]+)",
    r"total[:\s]*=?\s*([0-9.eE+-]+)",
    r"RMS error[:\s]*\(?\s*([0-9.eE+-]+)",
    r"^([0-9.eE+-]+)\s*$",
]

KERNEL_TIME_PATTERN = r"^\s*kernel\.([^:]+):\s*([0-9.eE+-]+)s?\s*$"
E2E_TIME_PATTERN = r"^\s*e2e\.([^:]+):\s*([0-9.eE+-]+)s?\s*$"
INIT_TIME_PATTERN = r"^\s*init\.([^:]+):\s*([0-9.eE+-]+)s?\s*$"


# ============================================================================
# Parsing Functions
# ============================================================================


def parse_checksum(output: str) -> Optional[str]:
    """Extract checksum from benchmark output.

    Args:
        output: Benchmark stdout

    Returns:
        Checksum string or None if not found
    """
    for pattern in CHECKSUM_PATTERNS:
        match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1)
    return None


def parse_kernel_timings(output: str) -> Dict[str, float]:
    """Extract kernel timing values from output.

    Args:
        output: Benchmark stdout

    Returns:
        Dict mapping kernel name -> time in seconds
    """
    timings = {}
    for match in re.finditer(KERNEL_TIME_PATTERN, output, re.MULTILINE):
        name, value = match.groups()
        try:
            timings[name] = float(value)
        except ValueError:
            pass
    return timings


def parse_e2e_timings(output: str) -> Dict[str, float]:
    """Extract end-to-end timing values from output.

    Args:
        output: Benchmark stdout

    Returns:
        Dict mapping name -> time in seconds
    """
    timings = {}
    for match in re.finditer(E2E_TIME_PATTERN, output, re.MULTILINE):
        name, value = match.groups()
        try:
            timings[name] = float(value)
        except ValueError:
            pass
    return timings


def parse_init_timings(output: str) -> Dict[str, float]:
    """Extract initialization timing values from output.

    Args:
        output: Benchmark stdout

    Returns:
        Dict mapping name -> time in seconds
    """
    timings = {}
    for match in re.finditer(INIT_TIME_PATTERN, output, re.MULTILINE):
        name, value = match.groups()
        try:
            timings[name] = float(value)
        except ValueError:
            pass
    return timings


def parse_counter_json(counter_dir: Path) -> Tuple[Optional[float], Optional[float]]:
    """Parse cluster.json to extract initializationTime and endToEndTime in seconds.

    Args:
        counter_dir: Directory containing counter JSON files.

    Returns:
        Tuple of (init_sec, e2e_sec), either may be None if not found.
    """
    cluster_file = counter_dir / "cluster.json"
    if not cluster_file.exists():
        return None, None

    try:
        with open(cluster_file) as f:
            data = json.load(f)
        counters = data.get("counters", {})
        init_ms = counters.get("initializationTime", {}).get("value_ms")
        e2e_ms = counters.get("endToEndTime", {}).get("value_ms")
        init_sec = init_ms / 1000.0 if init_ms is not None else None
        e2e_sec = e2e_ms / 1000.0 if e2e_ms is not None else None
        return init_sec, e2e_sec
    except (json.JSONDecodeError, KeyError, TypeError):
        return None, None


def read_slurm_output(output_dir: Path, job_id: str) -> Tuple[str, str]:
    """Read SLURM stdout and stderr files.

    Args:
        output_dir: Directory containing SLURM output files
        job_id: SLURM job ID

    Returns:
        Tuple of (stdout, stderr) contents
    """
    stdout_file = output_dir / f"slurm-{job_id}.out"
    stderr_file = output_dir / f"slurm-{job_id}.err"

    stdout = ""
    stderr = ""

    if stdout_file.exists():
        try:
            stdout = stdout_file.read_text()
        except Exception:
            pass

    if stderr_file.exists():
        try:
            stderr = stderr_file.read_text()
        except Exception:
            pass

    return stdout, stderr


def determine_status(
    arts_exit: int,
    omp_exit: int,
    arts_checksum: Optional[str],
    omp_checksum: Optional[str],
    tolerance: float = 0.01,
) -> Tuple[str, str]:
    """Determine overall status and verification result.

    Args:
        arts_exit: ARTS exit code
        omp_exit: OpenMP exit code (-1 if skipped)
        arts_checksum: ARTS checksum
        omp_checksum: OpenMP checksum
        tolerance: Tolerance for checksum comparison

    Returns:
        Tuple of (status, verification_note)
    """
    # Check for crash/failure
    if arts_exit != 0:
        return "FAIL", f"ARTS exited with code {arts_exit}"

    # If OMP was skipped, just check ARTS ran successfully
    if omp_exit == -1:
        if arts_checksum:
            return "PASS", "ARTS completed (OpenMP skipped for multi-node)"
        else:
            return "PASS", "ARTS completed, no checksum found"

    # OMP ran - check its status
    if omp_exit != 0:
        return "FAIL", f"OpenMP exited with code {omp_exit}"

    # Both ran successfully - compare checksums if available
    if arts_checksum and omp_checksum:
        try:
            arts_val = float(arts_checksum)
            omp_val = float(omp_checksum)

            if abs(arts_val - omp_val) / max(abs(omp_val), 1e-10) <= tolerance:
                return "PASS", f"Checksums match within {tolerance*100:.1f}% tolerance"
            else:
                return "FAIL", f"Checksum mismatch: ARTS={arts_checksum}, OMP={omp_checksum}"
        except ValueError:
            # Non-numeric checksums - exact match
            if arts_checksum == omp_checksum:
                return "PASS", "Checksums match exactly"
            else:
                return "FAIL", f"Checksum mismatch: ARTS={arts_checksum}, OMP={omp_checksum}"

    return "PASS", "Completed (no checksums to verify)"


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

    # Parse ARTS output (only from ARTS section)
    # Note: For ARTS, only use counter data for timing (stdout timing is inaccurate)
    arts_checksum = parse_checksum(arts_section)

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
    status, verification_note = determine_status(
        arts_exit, omp_exit, arts_checksum, omp_checksum
    )

    # Build result
    result = {
        "benchmark": benchmark,
        "run_number": run_number,
        "size": size,
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "verification": {
            "note": verification_note,
            "arts_checksum": arts_checksum,
            "omp_checksum": omp_checksum,
        },
        "arts": {
            "exit_code": arts_exit,
            "duration_sec": arts_duration,
            "checksum": arts_checksum,
            "init_sec": counter_init_sec,
            "e2e_sec": counter_e2e_sec,
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
    }

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
