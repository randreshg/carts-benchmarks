"""
Shared constants and parsing functions for CARTS benchmarks.

Used by runner.py, slurm/batch.py,
and slurm/job_result.py.
Stdlib-only — no external dependencies.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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
    "counters_*",
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
    "arts.cfg",
    "core",
    "core.*",
    "vgcore.*",
]

BENCHMARK_SHARED_CLEAN_DIR_NAMES = [
    "build",
    "results",
    "counters",
    "perfs",
    ".generated_configs",
]


# ============================================================================
# Variant and Basis Identifiers
# ============================================================================

VARIANT_ARTS = "arts"
VARIANT_OMP = "omp"
VARIANT_OPENMP = "openmp"

SPEEDUP_BASIS_KERNEL = "kernel"
SPEEDUP_BASIS_E2E = "e2e"
SPEEDUP_BASIS_TOTAL = "total"
SPEEDUP_BASIS_NA = "n/a"

# ============================================================================
# Serialization / Dict Key Constants
# ============================================================================

# Top-level result fields
KEY_BUILD_ARTS = "build_arts"
KEY_BUILD_OMP = "build_omp"
KEY_RUN_ARTS = "run_arts"
KEY_RUN_OMP = "run_omp"
KEY_TIMING = "timing"
KEY_VERIFICATION = "verification"
KEY_CONFIG = "config"
KEY_STATUS = "status"
KEY_RUN_NUMBER = "run_number"
KEY_RUN_PHASE = "run_phase"
KEY_SPEEDUP = "speedup"
KEY_NAME = "name"
KEY_SIZE = "size"

# Summary / robust statistics keys
KEY_ARTS_E2E_SEC = "arts_e2e_sec"
KEY_OMP_E2E_SEC = "omp_e2e_sec"
KEY_ARTS_OUTLIERS = "arts_outliers"
KEY_OMP_OUTLIERS = "omp_outliers"
KEY_ARTS_RAW_COUNT = "arts_raw_count"
KEY_OMP_RAW_COUNT = "omp_raw_count"
KEY_PAIRED_RAW_COUNT = "paired_raw_count"
KEY_ARTS_FILTERED_COUNT = "arts_filtered_count"
KEY_OMP_FILTERED_COUNT = "omp_filtered_count"
KEY_PAIRED_FILTERED_COUNT = "paired_filtered_count"

# Outlier detail keys
KEY_IS_OUTLIER = "is_outlier"

# ============================================================================
# Status Strings (used across SLURM results, report, and job_result)
# ============================================================================

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_CRASH = "CRASH"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_WARN = "WARN"
STATUS_SKIP = "SKIP"

# Statuses that indicate a failed/non-passing result
FAIL_STATUSES = frozenset({STATUS_FAIL, STATUS_CRASH, STATUS_TIMEOUT})

# ============================================================================
# Well-known Filenames
# ============================================================================

RESULTS_FILENAME = "results.json"
RESULT_JSON_FILENAME = "result.json"
RUN_CONFIG_JSON_FILENAME = "run_config.json"
MANIFEST_JSON_FILENAME = "manifest.json"
JOB_MANIFEST_JSON_FILENAME = "job_manifest.json"
SLURM_OUT_FILENAME = "slurm.out"
SLURM_ERR_FILENAME = "slurm.err"
ARTS_CFG_FILENAME = "arts.cfg"
REFERENCE_JSON_FILENAME = "reference.json"
COUNTERS_DIR_NAME = "counters"
PERF_DIR_NAME = "perf"
STARTUP_OUTLIER_DIAGNOSTICS_FILENAME = "startup_outlier_diagnostics.json"


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
STARTUP_TIME_PATTERN = r"^\s*startup\.([^:]+):\s*([0-9.eE+-]+)s?\s*$"
VERIFICATION_TIME_PATTERN = r"^\s*verification\.([^:]+):\s*([0-9.eE+-]+)s?\s*$"
CLEANUP_TIME_PATTERN = r"^\s*cleanup\.([^:]+):\s*([0-9.eE+-]+)s?\s*$"


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


def parse_startup_timings(output: str) -> Dict[str, float]:
    """Extract startup timing values from output."""
    timings = {}
    for match in re.finditer(STARTUP_TIME_PATTERN, output, re.MULTILINE):
        name, value = match.groups()
        try:
            timings[name.strip()] = float(value)
        except ValueError:
            pass
    return timings


def parse_verification_timings(output: str) -> Dict[str, float]:
    """Extract verification timing values from output."""
    timings = {}
    for match in re.finditer(VERIFICATION_TIME_PATTERN, output, re.MULTILINE):
        name, value = match.groups()
        try:
            timings[name.strip()] = float(value)
        except ValueError:
            pass
    return timings


def parse_cleanup_timings(output: str) -> Dict[str, float]:
    """Extract cleanup timing values from output."""
    timings = {}
    for match in re.finditer(CLEANUP_TIME_PATTERN, output, re.MULTILINE):
        name, value = match.groups()
        try:
            timings[name.strip()] = float(value)
        except ValueError:
            pass
    return timings


def parse_all_counters(counter_dir: Path) -> Dict[str, float]:
    """Parse cluster.json and return all counters as a flat map.

    For each counter in ``cluster.json``:
    - use ``value_ms`` when present (timing counters),
    - otherwise use ``value`` (count/size counters).

    Returns:
        Dict of ``counter_name -> numeric_value``.
        Returns empty dict on missing file or parse errors.
    """
    cluster_file = counter_dir / "cluster.json"
    if not cluster_file.exists():
        return {}

    parsed: Dict[str, float] = {}
    try:
        with open(cluster_file) as f:
            data = json.load(f)

        counters = data.get("counters", {})
        if not isinstance(counters, dict):
            return {}

        for name, entry in counters.items():
            if not isinstance(entry, dict):
                continue
            raw_value = entry.get("value_ms")
            if raw_value is None:
                raw_value = entry.get("value")
            if raw_value is None:
                continue

            try:
                parsed[name] = float(raw_value)
            except (TypeError, ValueError):
                continue

        return parsed
    except (json.JSONDecodeError, OSError, TypeError, AttributeError):
        return {}


def _parse_perf_event_totals(perf_output: Path) -> Tuple[Dict[str, int], bool]:
    """Parse one ``perf stat`` CSV and return raw event totals."""
    if not perf_output.exists():
        return {}, False

    event_totals: Dict[str, int] = {}
    found_any = False
    try:
        with open(perf_output, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split(",")
                if len(parts) < 4:
                    continue

                try:
                    if parts[0] and parts[0].replace(".", "").isdigit():
                        value_str = parts[1]
                        event_name = parts[3] if len(parts) > 3 else parts[2]
                    else:
                        value_str = parts[0]
                        event_name = parts[2] if len(parts) > 2 else ""

                    if not value_str or value_str.startswith("<"):
                        continue

                    value = int(float(value_str))
                    event_name = event_name.strip()
                    if not event_name:
                        continue

                    event_totals[event_name] = event_totals.get(event_name, 0) + value
                    found_any = True
                except (ValueError, IndexError):
                    continue
    except OSError:
        return {}, False

    return event_totals, found_any


def _perf_metrics_from_event_totals(event_totals: Dict[str, int]) -> Dict[str, float]:
    """Convert raw perf event totals into reportable metrics."""
    cache_references = event_totals.get("cache-references", 0)
    cache_misses = event_totals.get("cache-misses", 0)
    l1d_loads = event_totals.get("L1-dcache-loads", 0)
    l1d_load_misses = event_totals.get("L1-dcache-load-misses", 0)

    metrics: Dict[str, float] = {
        "cache_references": float(cache_references),
        "cache_misses": float(cache_misses),
        "l1d_loads": float(l1d_loads),
        "l1d_load_misses": float(l1d_load_misses),
        "cache_miss_rate": 0.0,
        "l1d_load_miss_rate": 0.0,
    }
    if cache_references > 0:
        metrics["cache_miss_rate"] = cache_misses / cache_references
    if l1d_loads > 0:
        metrics["l1d_load_miss_rate"] = l1d_load_misses / l1d_loads
    return metrics


def parse_perf_csv(perf_output: Path) -> Optional[Dict[str, float]]:
    """Parse one perf CSV file into cache metrics."""
    event_totals, found_any = _parse_perf_event_totals(perf_output)
    if not found_any:
        return None
    return _perf_metrics_from_event_totals(event_totals)


def aggregate_perf_csvs(perf_outputs: Iterable[Path]) -> Optional[Dict[str, float]]:
    """Aggregate multiple perf CSV files into one cache-metrics record."""
    merged: Dict[str, int] = {}
    found_any = False

    for perf_output in perf_outputs:
        event_totals, file_found_any = _parse_perf_event_totals(perf_output)
        if not file_found_any:
            continue
        found_any = True
        for event_name, value in event_totals.items():
            merged[event_name] = merged.get(event_name, 0) + value

    if not found_any:
        return None
    return _perf_metrics_from_event_totals(merged)


def filter_benchmark_output(output: str) -> str:
    """Extract only CARTS benchmark output lines (section/e2e/kernel timing, parallel/task timing, checksum).

    Filters out verbose ARTS runtime debug logs and keeps only benchmark-relevant output.
    """
    if not output:
        return ""
    prefixes = (
        "startup.", "kernel.", "verification.", "cleanup.",
        "e2e.", "parallel.", "task.", "checksum:", "tmp_checksum:",
    )
    return "\n".join(
        line for line in output.splitlines()
        if line.startswith(prefixes) or "checksum:" in line.lower()
    )
