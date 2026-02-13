"""
Shared constants and parsing functions for CARTS benchmarks.

Used by benchmark_runner.py, slurm_batch.py, and slurm_job_result.py.
Stdlib-only — no dependencies on the three scripts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Constants
# ============================================================================

DEFAULT_TIMEOUT = 60
DEFAULT_SIZE = "small"
DEFAULT_TOLERANCE = 1e-2  # 1% tolerance for FP operation ordering differences
DEFAULT_ARTS_PORT = 34739

SKIP_DIRS = {"common", "include", "src", "utilities",
             ".git", ".svn", ".hg", "build", "logs"}

# Perf cache events for hardware counter profiling
PERF_CACHE_EVENTS = [
    "cache-references",
    "cache-misses",
    "L1-dcache-loads",
    "L1-dcache-load-misses",
]

BENCHMARK_CLEAN_DIR_NAMES = [
    "build",
    "logs",
    "counters",
    "counter",
    "results",
    "perfs",
    "introspection",
]

BENCHMARK_CLEAN_DIR_GLOBS = [
    "counters.prev.*",
]

BENCHMARK_CLEAN_FILE_GLOBS = [
    "*.mlir",
    "*.ll",
    "*.o",
    "*.bc",
    "*.s",
    "*.tmp",
    "*.log",
    "*.out",
    "*.err",
    "*-metadata.json",
    ".carts-metadata.json",
    ".artsPrintLock",
    "core",
    "core.*",
    "vgcore.*",
]

BENCHMARK_SHARED_CLEAN_DIR_NAMES = [
    "results",
    "counters",
    "perfs",
    ".generated_configs",
]


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

    Uses the LAST checksum found in output to support benchmarks that
    print multiple intermediate checksums followed by a final combined one.

    Args:
        output: Benchmark stdout

    Returns:
        Checksum string or None if not found
    """
    for pattern in CHECKSUM_PATTERNS:
        matches = re.findall(pattern, output, re.MULTILINE | re.IGNORECASE)
        if matches:
            return matches[-1]  # Return the LAST match

    # Fallback: last non-empty line that looks numeric
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if re.match(r"^-?[0-9.]+(?:[eE][+-]?[0-9]+)?$", line):
            return line

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
            timings[name.strip()] = float(value)
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
            timings[name.strip()] = float(value)
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
            timings[name.strip()] = float(value)
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


def filter_benchmark_output(output: str) -> str:
    """Extract only CARTS benchmark output lines (init/e2e/kernel timing, parallel/task timing, checksum).

    Filters out verbose ARTS runtime debug logs and keeps only benchmark-relevant output.
    """
    if not output:
        return ""
    prefixes = ("kernel.", "e2e.", "init.", "parallel.", "task.", "checksum:", "tmp_checksum:")
    return "\n".join(
        line for line in output.splitlines()
        if line.startswith(prefixes) or "checksum:" in line.lower()
    )
