#!/usr/bin/env python3
"""
CARTS Unified Benchmark Runner

A powerful CLI tool for building, running, verifying, and reporting on CARTS benchmarks.
Provides rich terminal output, correctness verification, performance timing, and JSON export.

Usage:
    carts benchmarks list [--suite SUITE] [--format FORMAT]
    carts benchmarks run [BENCHMARKS...] [OPTIONS]
    carts benchmarks build [BENCHMARKS...] [--size SIZE]
    carts benchmarks clean [BENCHMARKS...] [--all]
    carts benchmarks report [--format FORMAT] [--output FILE]
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import threading

logger = logging.getLogger(__name__)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import typer

from carts_styles import (
    console as _shared_console,
    Colors, Symbols,
    print_header, print_footer, print_step, print_success, print_error,
    print_warning, print_info, print_debug as _print_debug,
    create_progress as _create_progress, create_results_table,
    create_summary_panel as _create_summary_panel_styles,
    format_summary_line, format_passed, format_failed, format_skipped,
    status_symbol as _status_symbol_str,
)

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

# SLURM batch support
import slurm_batch

# Shared constants and parsing
from benchmark_common import (
    CHECKSUM_PATTERNS,
    PERF_CACHE_EVENTS,
    SKIP_DIRS,
    BENCHMARK_CLEAN_DIR_NAMES,
    BENCHMARK_CLEAN_DIR_GLOBS,
    BENCHMARK_CLEAN_FILE_GLOBS,
    BENCHMARK_SHARED_CLEAN_DIR_NAMES,
    DEFAULT_TIMEOUT,
    DEFAULT_SIZE,
    DEFAULT_TOLERANCE,
    DEFAULT_ARTS_PORT,
    KERNEL_TIME_PATTERN,
    E2E_TIME_PATTERN,
    INIT_TIME_PATTERN,
    parse_checksum,
    parse_kernel_timings,
    parse_e2e_timings,
    parse_init_timings,
    parse_counter_json,
    filter_benchmark_output,
)

# ============================================================================
# Constants (local-only — shared constants imported from benchmark_common)
# ============================================================================

DEFAULT_ARTS_CONFIG = Path(__file__).parent / "arts.cfg"


# ============================================================================
# Data Classes
# ============================================================================


class Status(str, Enum):
    """Status of a build or run operation."""
    PENDING = "pending"
    BUILDING = "building"
    RUNNING = "running"
    PASS = "pass"
    FAIL = "fail"
    CRASH = "crash"
    TIMEOUT = "timeout"
    SKIP = "skip"


class Phase(str, Enum):
    """Current phase of benchmark execution."""
    PENDING = "pending"
    BUILD_ARTS = "build_arts"
    BUILD_OMP = "build_omp"
    RUN_ARTS = "run_arts"
    RUN_OMP = "run_omp"
    DONE = "done"


@dataclass
class BuildResult:
    """Result of building a benchmark."""
    status: Status
    duration_sec: float
    output: str
    executable: Optional[str] = None


@dataclass
class WorkerTiming:
    """Timing data for a single worker."""
    worker_id: int
    time_sec: float


@dataclass
class ParallelTaskTiming:
    """Parallel region and task timing data for analyzing delayed optimization impact.

    See docs/hypothesis.md for the experimental design this supports.
    """
    # Parallel region timings per worker
    parallel_timings: Dict[str, List[WorkerTiming]
                           ] = field(default_factory=dict)
    # Task (kernel) timings per worker
    task_timings: Dict[str, List[WorkerTiming]] = field(default_factory=dict)

    def get_parallel_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a parallel region."""
        return self._compute_stats(self.parallel_timings.get(name, []))

    def get_task_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a task."""
        return self._compute_stats(self.task_timings.get(name, []))

    def _compute_stats(self, timings: List[WorkerTiming]) -> Dict[str, float]:
        """Compute mean, min, max, stddev for a list of timings."""
        if not timings:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "stddev": 0.0, "count": 0}

        times = [t.time_sec for t in timings]
        n = len(times)
        mean = sum(times) / n
        variance = sum((t - mean) ** 2 for t in times) / n if n > 1 else 0.0

        return {
            "mean": mean,
            "min": min(times),
            "max": max(times),
            "stddev": variance ** 0.5,
            "count": n,
        }

    def compute_overhead(self, parallel_name: str, task_name: str) -> Dict[str, float]:
        """Compute overhead = parallel_time - task_time per worker."""
        parallel = {t.worker_id: t.time_sec for t in self.parallel_timings.get(
            parallel_name, [])}
        task = {t.worker_id: t.time_sec for t in self.task_timings.get(
            task_name, [])}

        overheads = []
        for worker_id in parallel:
            if worker_id in task:
                overheads.append(parallel[worker_id] - task[worker_id])

        if not overheads:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}

        return {
            "mean": sum(overheads) / len(overheads),
            "min": min(overheads),
            "max": max(overheads),
        }


@dataclass
class PerfCacheMetrics:
    """Cache metrics from perf stat profiling."""
    cache_references: int = 0
    cache_misses: int = 0
    l1d_loads: int = 0
    l1d_load_misses: int = 0
    cache_miss_rate: float = 0.0
    l1d_load_miss_rate: float = 0.0


@dataclass
class RunResult:
    """Result of running a benchmark."""
    status: Status
    duration_sec: float
    exit_code: int
    stdout: str
    stderr: str
    checksum: Optional[str] = None
    kernel_timings: Dict[str, float] = field(default_factory=dict)
    e2e_timings: Dict[str, float] = field(default_factory=dict)
    init_timings: Dict[str, float] = field(default_factory=dict)
    parallel_task_timing: Optional[ParallelTaskTiming] = None
    # Counter-based timing from ARTS introspection JSON (cluster.json)
    counter_init_sec: Optional[float] = None
    counter_e2e_sec: Optional[float] = None
    # Perf cache metrics
    perf_metrics: Optional[PerfCacheMetrics] = None
    perf_csv_path: Optional[str] = None


@dataclass
class TimingResult:
    """Timing comparison between ARTS and OpenMP."""
    arts_time_sec: float  # Basis used for speedup (e2e if available, else kernel, else total)
    omp_time_sec: float
    speedup: float  # omp_time / arts_time (>1 = ARTS faster)
    note: str
    # Additional context
    arts_kernel_sec: Optional[float] = None
    omp_kernel_sec: Optional[float] = None
    arts_e2e_sec: Optional[float] = None
    omp_e2e_sec: Optional[float] = None
    arts_init_sec: Optional[float] = None
    omp_init_sec: Optional[float] = None
    arts_total_sec: float = 0.0
    omp_total_sec: float = 0.0
    speedup_basis: str = "total"  # "e2e", "kernel", or "total"


@dataclass
class VerificationResult:
    """Result of correctness verification."""
    correct: bool
    arts_checksum: Optional[str]
    omp_checksum: Optional[str]
    tolerance_used: float
    note: str


@dataclass
class Artifacts:
    """Paths to generated artifacts."""
    # Source location
    benchmark_dir: str

    # Per-config build artifacts (in results/{experiment}/{benchmark}/build/{config}/)
    build_dir: Optional[str] = None
    carts_metadata: Optional[str] = None       # .carts-metadata.json
    arts_metadata_mlir: Optional[str] = None   # *_arts_metadata.mlir
    executable_arts: Optional[str] = None
    executable_omp: Optional[str] = None
    arts_config: Optional[str] = None          # arts.cfg

    # Per-run artifacts (in results/{experiment}/{benchmark}/build/{config}/runs/{N}/)
    run_dir: Optional[str] = None
    arts_log: Optional[str] = None
    omp_log: Optional[str] = None
    counters_dir: Optional[str] = None
    counter_files: List[str] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    arts_threads: int
    arts_nodes: int
    omp_threads: int
    launcher: str


@dataclass
class BenchmarkResult:
    """Complete result for a single benchmark."""
    name: str
    suite: str
    size: str
    config: BenchmarkConfig
    run_number: int
    build_arts: BuildResult
    build_omp: BuildResult
    run_arts: RunResult
    run_omp: RunResult
    timing: TimingResult
    verification: VerificationResult
    artifacts: Artifacts
    timestamp: str
    total_duration_sec: float
    # Actual CFLAGS used for this size (e.g., "-DNI=2000 -DNJ=2000")
    size_params: Optional[str] = None


# ============================================================================
# CLI Application
# ============================================================================

app = typer.Typer(
    name="carts-bench",
    help="CARTS Benchmark Runner - Build, run, verify, and report on benchmarks.",
    add_completion=False,
    no_args_is_help=True,
)
console = _shared_console


def get_carts_dir() -> Path:
    """Get the CARTS root directory."""
    script_dir = Path(__file__).parent.resolve()
    # Navigate up from external/carts-benchmarks to carts root
    carts_dir = script_dir.parent.parent
    if not (carts_dir / "tools" / "carts").exists():
        # Fallback: try CARTS_DIR environment variable
        env_dir = os.environ.get("CARTS_DIR")
        if env_dir:
            carts_dir = Path(env_dir)
    return carts_dir


def get_benchmarks_dir() -> Path:
    """Get the benchmarks directory."""
    return Path(__file__).parent.resolve()


# ============================================================================
# Helper Functions
# ============================================================================


def parse_threads(spec: str) -> List[int]:
    """Parse thread specification into list of thread counts.

    Supports comma-separated: "1,2,4,8"
    Supports range format: "1:16:2" (start:stop:step)
    """
    if ',' in spec:
        return [int(t.strip()) for t in spec.split(',') if t.strip()]
    elif ':' in spec:
        parts = [int(p.strip()) for p in spec.split(':') if p.strip()]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError(f"Invalid thread range format: {spec}")
        return list(range(start, stop + 1, step))
    else:
        # Single thread count
        return [int(spec)]


# ============================================================================
# Weak Scaling Support
# ============================================================================

# Benchmark-specific size parameters for weak scaling
# Maps benchmark names to their size parameters and work complexity
BENCHMARK_SIZE_PARAMS = {
    # PolyBench - Linear Algebra
    "polybench/gemm": {"params": ["NI", "NJ", "NK"], "complexity": "3d"},
    "polybench/2mm": {"params": ["NI", "NJ", "NK", "NL"], "complexity": "3d"},
    "polybench/3mm": {"params": ["NI", "NJ", "NK", "NL", "NM"], "complexity": "3d"},
    "polybench/syrk": {"params": ["N", "M"], "complexity": "3d"},
    "polybench/syr2k": {"params": ["N", "M"], "complexity": "3d"},
    "polybench/trmm": {"params": ["M", "N"], "complexity": "3d"},
    "polybench/lu": {"params": ["N"], "complexity": "3d"},
    "polybench/cholesky": {"params": ["N"], "complexity": "3d"},
    "polybench/mvt": {"params": ["N"], "complexity": "2d"},
    "polybench/atax": {"params": ["M", "N"], "complexity": "2d"},
    "polybench/bicg": {"params": ["M", "N"], "complexity": "2d"},
    "polybench/gesummv": {"params": ["N"], "complexity": "2d"},
    "polybench/doitgen": {"params": ["NQ", "NR", "NP"], "complexity": "3d"},
    # PolyBench - Stencils (2D work complexity)
    "polybench/jacobi2d": {"params": ["N"], "complexity": "2d", "extra": ["TSTEPS"]},
    "polybench/fdtd-2d": {"params": ["NX", "NY"], "complexity": "2d", "extra": ["TMAX"]},
    "polybench/heat-3d": {"params": ["N"], "complexity": "3d", "extra": ["TSTEPS"]},
    "polybench/seidel-2d": {"params": ["N"], "complexity": "2d", "extra": ["TSTEPS"]},
    # PolyBench - Data Mining
    "polybench/correlation": {"params": ["M", "N"], "complexity": "2d"},
    "polybench/covariance": {"params": ["M", "N"], "complexity": "2d"},
    # KaStORS benchmarks
    "kastors-jacobi/jacobi-block-for": {"params": ["SIZE"], "complexity": "2d"},
    "kastors-jacobi/jacobi-for": {"params": ["SIZE"], "complexity": "2d"},
    "kastors-jacobi/jacobi-task-dep": {"params": ["SIZE"], "complexity": "2d"},
}


def compute_weak_scaled_size(
    base_size: int,
    base_parallelism: int,
    target_parallelism: int,
    work_complexity: str = "2d",
) -> int:
    """Compute problem size for weak scaling (constant work per core).

    For constant work/core:
    - 2D problems (stencils, 2D FFT): N(p) = N0 * sqrt(p/p0)
    - 3D problems (GEMM, 3D stencils): N(p) = N0 * cbrt(p/p0)
    - Linear problems: N(p) = N0 * (p/p0)

    Args:
        base_size: Problem size at base parallelism level.
        base_parallelism: Starting parallelism (usually 1 thread/node).
        target_parallelism: Target parallelism (threads * nodes).
        work_complexity: "2d" (N²), "3d" (N³), or "linear" (N).

    Returns:
        Scaled problem size for constant work/core.
    """
    import math

    ratio = target_parallelism / base_parallelism
    if work_complexity == "2d":
        return int(base_size * math.sqrt(ratio))
    elif work_complexity == "3d":
        return int(base_size * (ratio ** (1 / 3)))
    else:  # linear
        return int(base_size * ratio)


def get_weak_scaling_cflags(
    benchmark: str,
    base_size: int,
    threads: int,
    nodes: int = 1,
    base_parallelism: int = 1,
) -> str:
    """Generate CFLAGS for weak scaling a specific benchmark.

    Args:
        benchmark: Benchmark name (e.g., "polybench/gemm").
        base_size: Base problem size at base_parallelism.
        threads: Number of threads per node.
        nodes: Number of nodes.
        base_parallelism: Reference parallelism for base_size.

    Returns:
        CFLAGS string with scaled size parameters.
    """
    if benchmark not in BENCHMARK_SIZE_PARAMS:
        # Unknown benchmark - return empty (use default size)
        return ""

    config = BENCHMARK_SIZE_PARAMS[benchmark]
    target_parallelism = threads * nodes
    scaled_size = compute_weak_scaled_size(
        base_size, base_parallelism, target_parallelism, config["complexity"]
    )

    # Build CFLAGS for all size parameters
    cflags_parts = [f"-D{param}={scaled_size}" for param in config["params"]]

    return " ".join(cflags_parts)


def append_perf_to_main_csv(
    temp_perf_file: Path,
    main_perf_file: Path,
    run_number: int,
) -> None:
    """Append perf stat data from temp file to main CSV with run column.

    For the first run (or if main file doesn't exist), creates the main file
    with a header. For subsequent runs, appends data rows with run number.

    Args:
        temp_perf_file: Temporary perf output file from current run.
        main_perf_file: Main CSV file to append to.
        run_number: Run number to add as first column.
    """
    if not temp_perf_file.exists():
        return

    is_first_run = not main_perf_file.exists() or main_perf_file.stat().st_size == 0

    with open(temp_perf_file, "r") as temp_f:
        lines = temp_f.readlines()

    # Filter out comment lines (# ...) and blank lines
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            data_lines.append(stripped)

    if not data_lines:
        return

    mode = "w" if is_first_run else "a"
    with open(main_perf_file, mode) as main_f:
        if is_first_run:
            # Write header for the run column
            main_f.write("# Columns: run,timestamp,value,unit,event,...\n")

        for line in data_lines:
            main_f.write(f"{run_number},{line}\n")

    # Remove temp file after successful append
    temp_perf_file.unlink(missing_ok=True)


def generate_arts_config(
    base_path: Path,
    threads: int,
    counter_dir: Optional[Path] = None,
    launcher: str = "ssh",
    nodes_override: Optional[int] = None,
) -> Path:
    """Generate temporary arts.cfg with specific configuration from a template.

    The base_path must be a valid config file that serves as a template.
    This ensures proper node hostnames are preserved for the SSH launcher.

    Args:
        base_path: Template config file (required). Must contain nodes= with hostnames.
        threads: Worker thread count per node.
        counter_dir: Optional directory for counter output.
        launcher: ARTS launcher type (ssh, slurm, lsf, local).
        nodes_override: If set, use only the first N nodes from the template's nodes= list.

    Note: For Slurm, nodeCount in config is IGNORED - ARTS reads SLURM_NNODES
    from environment (set by srun). The launcher controls HOW we invoke the
    executable, not just what's in the config.
    """
    if not base_path.exists():
        raise ValueError(f"Config template not found: {base_path}")

    content = base_path.read_text()

    # Update threads
    if re.search(r'^threads\s*=', content, re.MULTILINE):
        content = re.sub(
            r'^threads\s*=\s*\d+', f'threads={threads}', content, flags=re.MULTILINE)
    else:
        content = content.replace('[ARTS]', f'[ARTS]\nthreads={threads}')

    # Handle nodes override: truncate the nodes= list and update nodeCount
    if nodes_override is not None:
        all_nodes = get_arts_cfg_nodes(base_path)
        if nodes_override > len(all_nodes):
            raise ValueError(
                f"Requested --nodes {nodes_override} but config '{base_path}' only has {len(all_nodes)} node(s).\n"
                f"Use --arts-config to specify a config template with sufficient nodes.\n"
                f"Example: carts benchmarks run --arts-config /opt/carts/docker/arts-docker.cfg --nodes {nodes_override}"
            )
        truncated = all_nodes[:nodes_override]
        truncated_str = ",".join(truncated)

        # Update nodes= line
        if re.search(r'^nodes\s*=', content, re.MULTILINE):
            content = re.sub(
                r'^nodes\s*=.*$', f'nodes={truncated_str}', content, flags=re.MULTILINE)
        else:
            content = content.replace(
                '[ARTS]', f'[ARTS]\nnodes={truncated_str}')

        # Update nodeCount
        if re.search(r'^nodeCount\s*=', content, re.MULTILINE):
            content = re.sub(
                r'^nodeCount\s*=\s*\d+', f'nodeCount={nodes_override}', content, flags=re.MULTILINE)
        else:
            content = content.replace(
                '[ARTS]', f'[ARTS]\nnodeCount={nodes_override}')

        # Update masterNode to first node in truncated list
        if re.search(r'^masterNode\s*=', content, re.MULTILINE):
            content = re.sub(
                r'^masterNode\s*=.*$', f'masterNode={truncated[0]}', content, flags=re.MULTILINE)

    # Update launcher
    if re.search(r'^launcher\s*=', content, re.MULTILINE):
        content = re.sub(r'^launcher\s*=\s*\w+',
                         f'launcher={launcher}', content, flags=re.MULTILINE)
    else:
        content = content.replace('[ARTS]', f'[ARTS]\nlauncher={launcher}')

    # Add counter settings if requested
    if counter_dir:
        # Remove any existing counterFolder/counterStartPoint lines to avoid duplicates
        content = re.sub(r'^counterFolder\s*=.*\n?', '', content, flags=re.MULTILINE)
        content = re.sub(r'^counterStartPoint\s*=.*\n?', '', content, flags=re.MULTILINE)
        content += f"\ncounterFolder={counter_dir}\ncounterStartPoint=1\n"

    # Determine node count for filename
    node_count = nodes_override if nodes_override else get_arts_cfg_int(
        base_path, "nodeCount") or 1

    # Write to shared directory (NOT /tmp which is node-local in multi-node setups)
    # The carts-benchmarks directory is shared across all nodes via mounted volume
    generated_configs_dir = Path(__file__).parent / ".generated_configs"
    generated_configs_dir.mkdir(exist_ok=True)
    temp_path = generated_configs_dir / f"arts_{threads}t_{node_count}n.cfg"
    temp_path.write_text(content)
    return temp_path


def parse_arts_cfg(path: Optional[Path]) -> Dict[str, str]:
    """Parse an ARTS config file (arts.cfg) into a key/value dict.

    This parser is intentionally simple: it ignores section headers and
    supports `key=value` lines, stripping whitespace and inline comments.
    """
    if not path or not path.exists():
        return {}

    values: Dict[str, str] = {}
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            # Strip common inline comments.
            value = value.split("#", 1)[0].split(";", 1)[0].strip()
            if key:
                values[key] = value
    except Exception:
        logger.debug("Failed to parse arts.cfg: %s", path, exc_info=True)
        return {}
    return values


def get_arts_cfg_int(path: Optional[Path], key: str) -> Optional[int]:
    vals = parse_arts_cfg(path)
    if key not in vals:
        return None
    try:
        return int(vals[key])
    except Exception:
        logger.debug("Failed to convert arts.cfg key '%s' to int", key, exc_info=True)
        return None


def get_arts_cfg_str(path: Optional[Path], key: str) -> Optional[str]:
    vals = parse_arts_cfg(path)
    v = vals.get(key)
    return v if v else None


def parse_node_spec(spec: str) -> List[int]:
    """Parse node count specification.

    Supports:
        "4"         -> [4]
        "1-15"      -> [1, 2, 3, ..., 15]
        "1,2,4,8"   -> [1, 2, 4, 8]
        "1-4,8,16"  -> [1, 2, 3, 4, 8, 16]

    Args:
        spec: Node count specification string

    Returns:
        Sorted list of unique node counts

    Raises:
        ValueError: If specification is invalid
    """
    result = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            # Range: "1-15"
            try:
                start, end = part.split("-", 1)
                start_val = int(start.strip())
                end_val = int(end.strip())
                if start_val < 1 or end_val < start_val:
                    raise ValueError(f"Invalid range: {part}")
                result.update(range(start_val, end_val + 1))
            except ValueError as e:
                raise ValueError(f"Invalid range '{part}': {e}")
        else:
            # Single value: "4"
            try:
                val = int(part)
                if val < 1:
                    raise ValueError(f"Node count must be >= 1: {val}")
                result.add(val)
            except ValueError:
                raise ValueError(f"Invalid node count: {part}")

    return sorted(result)


def get_arts_cfg_nodes(path: Optional[Path]) -> List[str]:
    """Parse nodes= field from arts.cfg into list of hostnames.

    Returns list of hostnames from the nodes= field, or ["localhost"] if not found.
    """
    cfg = parse_arts_cfg(path)
    nodes_str = cfg.get("nodes", "localhost")
    return [n.strip() for n in nodes_str.split(",") if n.strip()]






# ============================================================================
# Artifact Manager
# ============================================================================


class ArtifactManager:
    """Manages a self-contained artifact directory for a benchmark experiment.

    Every execution produces a single timestamped directory with all artifacts,
    logs, and a manifest.  Both ``run`` and ``slurm-run`` share the same
    canonical layout:

        {base_results_dir}/{timestamp}/
          manifest.json
          results.json
          {benchmark_name}/
            {threads}t_{nodes}n/
              artifacts/          # build outputs (shared across runs)
              run_1/              # per-run outputs
                arts.log
                omp.log
                counters/         # if counters enabled
                perf/             # if perf enabled
              run_2/
    """

    def __init__(self, base_results_dir: Path, timestamp: str):
        self.experiment_dir = base_results_dir / timestamp
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.results_json_path = self.experiment_dir / "results.json"
        self.manifest_path = self.experiment_dir / "manifest.json"
        self._manifest_benchmarks: Dict[str, Dict] = {}

    # -- directory getters (create on first access) --------------------------

    def get_config_dir(self, benchmark_name: str, config: BenchmarkConfig) -> Path:
        config_label = f"{config.arts_threads}t_{config.arts_nodes}n"
        d = self.experiment_dir / benchmark_name / config_label
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_artifacts_dir(self, benchmark_name: str, config: BenchmarkConfig) -> Path:
        d = self.get_config_dir(benchmark_name, config) / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_run_dir(self, benchmark_name: str, config: BenchmarkConfig, run_number: int) -> Path:
        d = self.get_config_dir(benchmark_name, config) / f"run_{run_number}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_counter_dir(self, benchmark_name: str, config: BenchmarkConfig, run_number: int) -> Path:
        d = self.get_run_dir(benchmark_name, config, run_number) / "counters"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_perf_dir(self, benchmark_name: str, config: BenchmarkConfig, run_number: int) -> Path:
        d = self.get_run_dir(benchmark_name, config, run_number) / "perf"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- artifact operations -------------------------------------------------

    def copy_build_artifacts(
        self,
        bench_path: Path,
        benchmark_name: str,
        config: BenchmarkConfig,
        arts_cfg_used: Optional[Path] = None,
    ) -> Dict[str, Optional[str]]:
        """Copy build artifacts into the experiment's ``artifacts/`` directory."""
        artifacts_dir = self.get_artifacts_dir(benchmark_name, config)
        paths: Dict[str, Optional[str]] = {}

        # arts.cfg (build INPUT)
        cfg_src = (
            arts_cfg_used if arts_cfg_used and arts_cfg_used.exists()
            else (bench_path / "arts.cfg")
        )
        if cfg_src.exists():
            dest = artifacts_dir / "arts.cfg"
            shutil.copy2(cfg_src, dest)
            paths["arts_config"] = str(dest)

        # .carts-metadata.json (compiler metadata)
        metadata = bench_path / ".carts-metadata.json"
        if metadata.exists():
            dest = artifacts_dir / ".carts-metadata.json"
            shutil.copy2(metadata, dest)
            paths["carts_metadata"] = str(dest)

        # *_arts_metadata.mlir
        for mlir in bench_path.glob("*_arts_metadata.mlir"):
            dest = artifacts_dir / mlir.name
            shutil.copy2(mlir, dest)
            paths["arts_metadata_mlir"] = str(dest)

        # Other MLIR files
        for mlir in bench_path.glob("*.mlir"):
            if "_metadata" not in mlir.name:
                shutil.copy2(mlir, artifacts_dir / mlir.name)

        # LLVM IR
        for ll in bench_path.glob("*-arts.ll"):
            shutil.copy2(ll, artifacts_dir / ll.name)

        # Executables
        for exe in bench_path.glob("*_arts"):
            if exe.is_file():
                dest = artifacts_dir / exe.name
                shutil.copy2(exe, dest)
                paths["executable_arts"] = str(dest)
        for exe in list(bench_path.glob("*_omp")) + list(
            (bench_path / "build").glob("*_omp")
        ):
            if exe.is_file():
                dest = artifacts_dir / exe.name
                shutil.copy2(exe, dest)
                paths["executable_omp"] = str(dest)

        # Build logs
        logs_dir = bench_path / "logs"
        if logs_dir.exists():
            for log in ["build_arts.log", "build_openmp.log"]:
                log_file = logs_dir / log
                if log_file.exists():
                    shutil.copy2(log_file, artifacts_dir / log)

        return paths

    def record_run(
        self,
        benchmark_name: str,
        config: BenchmarkConfig,
        run_number: int,
        has_counters: bool = False,
        has_perf: bool = False,
    ):
        """Track a completed run for the manifest."""
        config_label = f"{config.arts_threads}t_{config.arts_nodes}n"
        if benchmark_name not in self._manifest_benchmarks:
            self._manifest_benchmarks[benchmark_name] = {"configs": {}}
        configs = self._manifest_benchmarks[benchmark_name]["configs"]
        if config_label not in configs:
            configs[config_label] = {
                "artifacts": str(Path(benchmark_name) / config_label / "artifacts"),
                "runs": {},
            }
        configs[config_label]["runs"][str(run_number)] = {
            "path": str(Path(benchmark_name) / config_label / f"run_{run_number}"),
            "has_counters": has_counters,
            "has_perf": has_perf,
        }

    def write_manifest(
        self,
        results: List[BenchmarkResult],
        command: str,
        total_duration: float,
    ) -> Path:
        """Write ``manifest.json`` — a structure index and quick summary."""
        import math
        from collections import Counter as _Counter

        passed = sum(
            1 for r in results
            if r.run_arts.status == Status.PASS and r.verification.correct
        )
        failed = sum(
            1 for r in results
            if r.run_arts.status in (Status.FAIL, Status.CRASH)
        )
        total_benchmarks = len(set(r.name for r in results))
        total_configs = len(
            set((r.name, r.config.arts_threads, r.config.arts_nodes) for r in results)
        )
        config_counts = _Counter(
            (r.name, r.config.arts_threads, r.config.arts_nodes) for r in results
        )
        runs_per_config = max(config_counts.values()) if config_counts else 1
        speedups = [r.timing.speedup for r in results if r.timing.speedup > 0]
        geomean = (
            math.exp(sum(math.log(s) for s in speedups) / len(speedups))
            if speedups else 0.0
        )

        carts_dir = get_carts_dir()
        benchmarks_dir = get_benchmarks_dir()
        repro = get_reproducibility_metadata(carts_dir, benchmarks_dir)

        manifest = {
            "version": 1,
            "created": datetime.now().isoformat(),
            "command": command,
            "layout": {
                "results_json": "results.json",
                "benchmarks": self._manifest_benchmarks,
            },
            "summary": {
                "total_benchmarks": total_benchmarks,
                "total_configs": total_configs,
                "runs_per_config": runs_per_config,
                "passed": passed,
                "failed": failed,
                "geometric_mean_speedup": round(geomean, 4),
                "total_duration_sec": round(total_duration, 1),
            },
            "reproducibility": repro,
        }

        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        return self.manifest_path


# ============================================================================
# Benchmark Runner Class
# ============================================================================


class BenchmarkRunner:
    """Main class for running benchmarks."""

    def __init__(
        self,
        console: Console,
        verbose: bool = False,
        quiet: bool = False,
        trace: bool = False,
        clean: bool = True,
        debug: int = 0,
        artifact_manager: Optional[ArtifactManager] = None,
    ):
        self.console = console
        self.verbose = verbose
        self.quiet = quiet
        self.trace = trace
        self.clean = clean
        self.debug = debug
        self.artifact_manager = artifact_manager
        self.carts_dir = get_carts_dir()
        self.benchmarks_dir = get_benchmarks_dir()
        self.results: List[BenchmarkResult] = []

    def discover_benchmarks(self, suite: Optional[str] = None) -> List[str]:
        """Find all benchmarks by looking for Makefiles with source files."""
        benchmarks = []

        for makefile in self.benchmarks_dir.rglob("Makefile"):
            bench_dir = makefile.parent
            rel_path = bench_dir.relative_to(self.benchmarks_dir)

            # Skip excluded directories
            if any(part in SKIP_DIRS for part in rel_path.parts):
                continue

            # Skip if no source files
            has_source = any(bench_dir.glob(
                "*.c")) or any(bench_dir.glob("*.cpp"))
            if not has_source:
                continue

            # Skip disabled benchmarks
            if (bench_dir / ".disabled").exists():
                continue

            bench_name = str(rel_path)

            # Filter by suite if specified
            if suite and not bench_name.startswith(suite):
                continue

            benchmarks.append(bench_name)

        return sorted(benchmarks)

    def _find_source_file(self, bench_path: Path) -> Optional[Path]:
        """Find the source file for a benchmark."""
        # Try to read from Makefile
        makefile = bench_path / "Makefile"
        if makefile.exists():
            content = makefile.read_text()
            # Look for SRC := <filename>
            for line in content.splitlines():
                if line.startswith("SRC"):
                    parts = line.split(":=")
                    if len(parts) == 2:
                        src = parts[1].strip()
                        src_path = bench_path / src
                        if src_path.exists():
                            return src_path

        # Fallback: look for .c files
        c_files = list(bench_path.glob("*.c"))
        if c_files:
            # Prefer file matching directory name
            bench_name = bench_path.name
            for f in c_files:
                if f.stem == bench_name:
                    return f
            return c_files[0]

        return None

    def get_size_params(self, bench_path: Path, size: str) -> Optional[str]:
        """Extract the CFLAGS for a given size from the Makefile.

        Returns the actual -D flags used (e.g., "-DNI=2000 -DNJ=2000") or None if not found.
        """
        makefile = bench_path / "Makefile"
        if not makefile.exists():
            return None

        content = makefile.read_text()

        # Map size to CFLAGS variable name
        size_var_map = {
            "small": "SMALL_CFLAGS",
            "medium": "MEDIUM_CFLAGS",
            "large": "LARGE_CFLAGS",
            "extralarge": "EXTRALARGE_CFLAGS",
            "mini": "MINI_CFLAGS",
            "standard": "STANDARD_CFLAGS",
        }

        var_name = size_var_map.get(size.lower())
        if not var_name:
            return None

        return self._extract_make_var(content, var_name)

    def get_run_args(self, bench_path: Path, size: str) -> List[str]:
        """Extract run-time arguments for a given size from the Makefile."""
        makefile = bench_path / "Makefile"
        if not makefile.exists():
            return []

        content = makefile.read_text()

        # Map size to args variable name
        size_var_map = {
            "small": "SMALL_ARGS",
            "medium": "MEDIUM_ARGS",
            "large": "LARGE_ARGS",
            "extralarge": "EXTRALARGE_ARGS",
            "mini": "MINI_ARGS",
            "standard": "STANDARD_ARGS",
        }

        var_name = size_var_map.get(size.lower())
        value = self._extract_make_var(content, var_name) if var_name else None
        if not value:
            value = self._extract_make_var(content, "RUN_ARGS")
        if not value:
            return []

        return shlex.split(value)

    def get_verify_tolerance(self, bench_path: Path) -> float:
        """Extract a per-benchmark verification tolerance from the Makefile."""
        makefile = bench_path / "Makefile"
        if not makefile.exists():
            return DEFAULT_TOLERANCE

        content = makefile.read_text()
        value = self._extract_make_var(content, "VERIFY_TOLERANCE")
        if not value:
            value = self._extract_make_var(content, "TOLERANCE")
        if not value:
            return DEFAULT_TOLERANCE

        try:
            return float(value)
        except ValueError:
            return DEFAULT_TOLERANCE

    def _extract_make_var(self, content: str, var_name: str) -> Optional[str]:
        """Return the value of a Makefile variable if present."""
        pattern = rf'^{re.escape(var_name)}\s*[?:]?=\s*(.+)$'
        for line in content.splitlines():
            match = re.match(pattern, line.strip())
            if match:
                return match.group(1).strip()
        return None

    def get_executable_paths(self, bench_path: Path) -> Tuple[Path, Path]:
        """Get the ARTS and OpenMP executable paths for a benchmark.

        Based on carts.mk:
            ARTS_BINARY := $(EXAMPLE_NAME)_arts       # in benchmark root
            OMP_BINARY := $(BUILD_DIR)/$(EXAMPLE_NAME)_omp  # in build/

        Args:
            bench_path: Path to the benchmark directory

        Returns:
            Tuple of (arts_exe_path, omp_exe_path)
        """
        makefile = bench_path / "Makefile"
        example_name = None

        if makefile.exists():
            content = makefile.read_text()
            example_name = self._extract_make_var(content, "EXAMPLE_NAME")

        # Fallback to directory name if EXAMPLE_NAME not found
        if not example_name:
            example_name = bench_path.name

        arts_exe = bench_path / f"{example_name}_arts"
        omp_exe = bench_path / "build" / f"{example_name}_omp"

        return arts_exe, omp_exe

    def build_benchmark(
        self,
        name: str,
        size: str,
        variant: str = "arts",
        arts_config: Optional[Path] = None,
        cflags: str = "",
        arts_exec_args: Optional[str] = None,
    ) -> BuildResult:
        """Build a single benchmark using make."""
        bench_path = self.benchmarks_dir / name

        if not bench_path.exists():
            return BuildResult(
                status=Status.FAIL,
                duration_sec=0.0,
                output=f"Benchmark directory not found: {bench_path}",
            )

        # Check that Makefile exists
        makefile = bench_path / "Makefile"
        if not makefile.exists():
            return BuildResult(
                status=Status.FAIL,
                duration_sec=0.0,
                output=f"No Makefile found in {bench_path}",
            )

        # Map size parameter to make target
        size_targets = {
            "small": "small",
            "medium": "medium",
            "large": "large",
            "extralarge": "extralarge",
            "mini": "mini",
            "standard": "standard",
        }

        # Build command using make
        # Use granular targets ({size}-arts, {size}-openmp) defined in common/carts.mk
        # CRITICAL: Provide explicit path to carts executable (not in PATH during non-interactive shells)
        carts_exe = self.carts_dir / "tools" / "carts"
        if variant == "openmp":
            # Build only OpenMP variant using granular target
            if size in size_targets:
                cmd = ["make", f"{size}-openmp", f"CARTS={carts_exe}"]
            else:
                cmd = ["make", "openmp", f"CARTS={carts_exe}"]
            if cflags:
                cmd.append(f"CFLAGS={cflags}")
        else:
            # Build ARTS variant (full pipeline)
            if size in size_targets:
                # Use granular size-arts target for ARTS-only builds
                cmd = ["make", f"{size}-arts", f"CARTS={carts_exe}"]
            else:
                cmd = ["make", "all", f"CARTS={carts_exe}"]
            if cflags:
                cmd.append(f"CFLAGS={cflags}")

        # Add ARTS config override if provided
        if arts_config and variant != "openmp":
            cmd.append(f"ARTS_CFG={arts_config}")
        if arts_exec_args and variant != "openmp":
            escaped_args = arts_exec_args.replace("\\", "\\\\").replace(" ", "\\ ")
            cmd.append(f"EXECUTE_ARGS={escaped_args}")

        # Debug output level 1: show commands
        if self.debug >= 1:
            self.console.print(f"[dim]$ cd {bench_path} && {' '.join(cmd)}[/]")

        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=bench_path,
            )
            duration = time.time() - start

            # Debug output level 2: write build log to file
            if self.debug >= 2:
                logs_dir = bench_path / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                log_file = logs_dir / f"build_{variant}.log"
                with open(log_file, "w") as f:
                    f.write(f"# Command: {' '.join(cmd)}\n")
                    f.write(f"# Duration: {duration:.3f}s\n")
                    f.write(f"# Exit code: {result.returncode}\n\n")
                    if result.stdout:
                        f.write(result.stdout)
                    if result.stderr:
                        f.write("=== STDERR ===\n")
                        f.write(result.stderr)
                self.console.print(f"[dim]  Log: {log_file}[/]")

            if result.returncode == 0:
                executable = self._find_executable(bench_path, variant)
                return BuildResult(
                    status=Status.PASS,
                    duration_sec=duration,
                    output=result.stdout + result.stderr,
                    executable=executable,
                )
            else:
                return BuildResult(
                    status=Status.FAIL,
                    duration_sec=duration,
                    output=result.stdout + result.stderr,
                )
        except subprocess.TimeoutExpired:
            return BuildResult(
                status=Status.TIMEOUT,
                duration_sec=300.0,
                output="Build timed out after 300 seconds",
            )
        except Exception as e:
            return BuildResult(
                status=Status.FAIL,
                duration_sec=time.time() - start,
                output=str(e),
            )

    def _find_executable(self, bench_path: Path, variant: str) -> Optional[str]:
        """Find the generated executable."""
        suffix = "_arts" if variant == "arts" else "_omp"

        # First check in the benchmark directory itself
        for exe in bench_path.glob(f"*{suffix}"):
            if exe.is_file() and os.access(exe, os.X_OK):
                return str(exe)

        # Then check in the build directory
        build_dir = bench_path / "build"
        if build_dir.exists():
            for exe in build_dir.glob(f"*{suffix}"):
                if exe.is_file() and os.access(exe, os.X_OK):
                    return str(exe)

        return None

    def run_with_thread_sweep(
        self,
        name: str,
        size: str,
        threads_list: List[Optional[int]],
        base_config: Optional[Path],
        cflags: str = "",
        counter_dir: Optional[Path] = None,
        timeout: int = DEFAULT_TIMEOUT,
        omp_threads: Optional[int] = None,
        launcher: Optional[str] = None,
        node_counts: Optional[List[Optional[int]]] = None,
        weak_scaling: bool = False,
        base_size: Optional[int] = None,
        runs: int = 1,
        arts_exec_args: Optional[str] = None,
        perf_enabled: bool = False,
        perf_interval: float = 0.1,
    ) -> List[BenchmarkResult]:
        """Run benchmark with multiple thread configurations.

        Directory management is delegated to ``self.artifact_manager`` when
        present.  Logs are always captured into the run directory.

        Returns:
            List of BenchmarkResult objects.
        """
        results = []
        am = self.artifact_manager  # shorthand (may be None)

        # Clean benchmark directory to avoid stale artifacts
        if self.clean:
            self.clean_benchmark(name)

        bench_path = self.benchmarks_dir / name
        run_args = self.get_run_args(bench_path, size)
        verify_tolerance = self.get_verify_tolerance(bench_path)

        # Determine effective config template
        effective_config = base_config
        if effective_config is None:
            candidate = bench_path / "arts.cfg"
            if candidate.exists():
                effective_config = candidate
            else:
                effective_config = DEFAULT_ARTS_CONFIG

        base_nodes = get_arts_cfg_int(effective_config, "nodeCount") or 1
        base_threads = get_arts_cfg_int(effective_config, "threads") or 1
        base_launcher = get_arts_cfg_str(effective_config, "launcher") or "ssh"
        desired_launcher = launcher if launcher is not None else base_launcher

        effective_node_counts = [n if n is not None else base_nodes
                                 for n in node_counts] if node_counts else [base_nodes]

        for desired_nodes in effective_node_counts:
            # Skip benchmarks disabled for multi-node
            if desired_nodes > 1 and (bench_path / ".disable-multinode").exists():
                first_threads = threads_list[0] if threads_list[0] is not None else base_threads
                skip_config = BenchmarkConfig(
                    arts_threads=first_threads,
                    arts_nodes=desired_nodes,
                    omp_threads=first_threads,
                    launcher=desired_launcher,
                )
                results.append(self._make_skip_result(
                    name, size,
                    "Benchmark disabled for multi-node (has .disable-multinode marker)",
                    skip_config
                ))
                continue

            for threads_or_none in threads_list:
                threads = threads_or_none if threads_or_none is not None else base_threads
                actual_omp_threads = omp_threads if omp_threads else threads
                config = BenchmarkConfig(
                    arts_threads=threads,
                    arts_nodes=desired_nodes,
                    omp_threads=actual_omp_threads,
                    launcher=desired_launcher,
                )

                # Counter directory: artifact_manager path or explicit --counter-dir
                run_counter_dir: Optional[Path] = None
                if am:
                    # Always set counterFolder so counters land in place
                    run_counter_dir = am.get_counter_dir(name, config, 1)
                elif counter_dir is not None:
                    run_counter_dir = counter_dir

                # Generate arts.cfg with thread count, launcher, node count, counter dir
                arts_cfg = generate_arts_config(
                    effective_config, threads, run_counter_dir,
                    desired_launcher, desired_nodes
                )

                # Compute effective cflags (may include weak scaling size overrides)
                effective_cflags = cflags
                if weak_scaling and base_size:
                    weak_cflags = get_weak_scaling_cflags(
                        name, base_size, threads, desired_nodes, base_parallelism=1
                    )
                    if weak_cflags:
                        effective_cflags = f"{cflags} {weak_cflags}".strip()

                env = {"OMP_NUM_THREADS": str(actual_omp_threads)}
                if "OMP_WAIT_POLICY" not in os.environ:
                    env["OMP_WAIT_POLICY"] = "ACTIVE"
                if "CARTS_BENCHMARKS_REPORT_INIT" not in os.environ:
                    env["CARTS_BENCHMARKS_REPORT_INIT"] = "1"

                # Build ARTS variant
                build_arts = self.build_benchmark(
                    name, size, "arts", arts_cfg, effective_cflags, arts_exec_args)

                # Build OpenMP variant (skip for multi-node)
                if desired_nodes == 1:
                    build_omp = self.build_benchmark(
                        name, size, "openmp", None, effective_cflags)
                else:
                    build_omp = BuildResult(
                        status=Status.SKIP, duration_sec=0.0,
                        output="Skipped: OpenMP not applicable for multi-node",
                        executable=None)

                # Copy build artifacts ONCE per config (via artifact manager)
                artifact_paths: Dict[str, Optional[str]] = {}
                if am:
                    artifact_paths = am.copy_build_artifacts(
                        bench_path, name, config, arts_cfg_used=arts_cfg)

                # Run multiple times per configuration
                for run_num in range(1, runs + 1):
                    arts_log: Optional[Path] = None
                    omp_log: Optional[Path] = None
                    run_counter_dir: Optional[Path] = None

                    if am:
                        run_counter_dir = am.get_counter_dir(name, config, run_num)
                        run_dir = am.get_run_dir(name, config, run_num)
                        arts_log = run_dir / "arts.log"
                        omp_log = run_dir / "omp.log"
                    elif self.debug >= 2:
                        logs_dir = bench_path / "logs"
                        run_suffix = f"_r{run_num}" if runs > 1 else ""
                        arts_log = logs_dir / f"arts_{threads}t{run_suffix}.log"
                        omp_log = logs_dir / f"omp_{threads}t{run_suffix}.log"

                    # Perf setup
                    arts_perf_name = omp_perf_name = None
                    perf_output_dir: Optional[Path] = None
                    if perf_enabled and am:
                        perf_output_dir = am.get_perf_dir(name, config, run_num)
                        arts_perf_name = "arts_cache.csv"
                        omp_perf_name = "omp_cache.csv"

                    # Clean up stale ARTS port before run
                    self._cleanup_port()

                    # Run ARTS version
                    if build_arts.status == Status.PASS and build_arts.executable:
                        run_arts = self.run_benchmark(
                            build_arts.executable,
                            timeout,
                            env=env,
                            launcher=desired_launcher,
                            node_count=desired_nodes,
                            threads=threads,
                            args=run_args,
                            log_file=arts_log,
                            perf_enabled=perf_enabled,
                            perf_interval=perf_interval,
                            perf_output_name=arts_perf_name or "perf_cache_arts.csv",
                            perf_output_dir=perf_output_dir,
                            counter_dir=run_counter_dir,
                        )
                    else:
                        run_arts = RunResult(
                            status=Status.SKIP, duration_sec=0.0,
                            exit_code=-1, stdout="", stderr="Build failed",
                        )

                    # Run OpenMP version
                    if build_omp.status == Status.PASS and build_omp.executable:
                        run_omp = self.run_benchmark(
                            build_omp.executable,
                            timeout,
                            env=env,
                            args=run_args,
                            log_file=omp_log,
                            perf_enabled=perf_enabled,
                            perf_interval=perf_interval,
                            perf_output_name=omp_perf_name or "perf_cache_omp.csv",
                            perf_output_dir=perf_output_dir,
                        )
                    else:
                        run_omp = RunResult(
                            status=Status.SKIP, duration_sec=0.0,
                            exit_code=-1, stdout="", stderr="Build failed",
                        )

                    timing = self.calculate_timing(run_arts, run_omp)
                    verification = self.verify_correctness(
                        run_arts, run_omp, tolerance=verify_tolerance)
                    artifacts = self.collect_artifacts(bench_path)

                    # Update artifacts with paths inside experiment directory
                    has_counters = False
                    has_perf = False
                    if am:
                        run_dir = am.get_run_dir(name, config, run_num)
                        artifacts_dir = am.get_artifacts_dir(name, config)

                        # Build artifact paths
                        for key, attr in [
                            ("arts_config", "arts_config"),
                            ("carts_metadata", "carts_metadata"),
                            ("arts_metadata_mlir", "arts_metadata_mlir"),
                            ("executable_arts", "executable_arts"),
                            ("executable_omp", "executable_omp"),
                        ]:
                            if artifact_paths.get(key):
                                setattr(artifacts, attr, artifact_paths[key])
                        artifacts.build_dir = str(artifacts_dir)
                        artifacts.run_dir = str(run_dir)
                        artifacts.arts_log = str(arts_log) if arts_log else None
                        artifacts.omp_log = str(omp_log) if omp_log else None

                        # Counter files (already landed in place via counterFolder)
                        counter_path = am.get_counter_dir(name, config, run_num)
                        counter_files = sorted(counter_path.glob("*.json"))
                        if counter_files:
                            has_counters = True
                            artifacts.counters_dir = str(counter_path)
                            artifacts.counter_files = [str(f) for f in counter_files]
                            init_sec, e2e_sec = parse_counter_json(counter_path)
                            run_arts.counter_init_sec = init_sec
                            run_arts.counter_e2e_sec = e2e_sec

                        # Perf
                        if perf_enabled and perf_output_dir and perf_output_dir.exists():
                            perf_files = list(perf_output_dir.glob("*.csv"))
                            if perf_files:
                                has_perf = True

                        am.record_run(name, config, run_num,
                                      has_counters=has_counters, has_perf=has_perf)

                    result = BenchmarkResult(
                        name=name,
                        suite=name.split("/")[0] if "/" in name else "",
                        size=size,
                        config=config,
                        run_number=run_num,
                        build_arts=build_arts,
                        build_omp=build_omp,
                        run_arts=run_arts,
                        run_omp=run_omp,
                        timing=timing,
                        verification=verification,
                        artifacts=artifacts,
                        timestamp=datetime.now().isoformat(),
                        total_duration_sec=0.0,
                        size_params=self.get_size_params(bench_path, size),
                    )
                    results.append(result)

        return results

    def run_benchmark(
        self,
        executable: str,
        timeout: int = DEFAULT_TIMEOUT,
        env: Optional[Dict[str, str]] = None,
        launcher: str = "ssh",
        node_count: int = 1,
        threads: int = 1,
        args: Optional[List[str]] = None,
        log_file: Optional[Path] = None,
        perf_enabled: bool = False,
        perf_interval: float = 0.1,
        perf_output_name: str = "perf_cache.csv",
        perf_output_dir: Optional[Path] = None,
        counter_dir: Optional[Path] = None,
    ) -> RunResult:
        """Execute a benchmark and capture output.

        Args:
            executable: Path to the benchmark executable.
            timeout: Maximum execution time in seconds.
            env: Environment variables to set.
            launcher: ARTS launcher type. For 'slurm', wraps executable in srun.
            node_count: Number of nodes for distributed execution (slurm only).
            threads: Number of threads per node (for srun --cpus-per-task).
            args: Optional list of command-line arguments to pass to the executable.
            log_file: Optional path to write full output (for debug=2).
            perf_enabled: Enable perf stat profiling for cache metrics.
            perf_interval: Interval in seconds for perf stat sampling (default: 0.1).
            perf_output_name: Filename for perf CSV output (default: perf_cache.csv).
            perf_output_dir: Directory for perf CSV output (default: executable's parent).
            counter_dir: Optional path to override the embedded counterFolder config value
                via the ARTS per-variable env var mechanism.
        """
        if not executable or not os.path.exists(executable):
            return RunResult(
                status=Status.SKIP,
                duration_sec=0.0,
                exit_code=-1,
                stdout="",
                stderr="Executable not found",
            )

        # Merge with current environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        if counter_dir:
            run_env["counterFolder"] = str(counter_dir)

        # Build command based on launcher
        if launcher == "slurm" and node_count > 1:
            # For Slurm multi-node: wrap in srun
            # ARTS reads SLURM_NNODES and SLURM_CPUS_PER_TASK from environment
            cmd = [
                "srun",
                f"-N{node_count}",
                "--ntasks-per-node=1",
                f"--cpus-per-task={threads}",
                executable,
            ]
        else:
            # Local or SSH launcher: run directly (ARTS handles distribution for SSH)
            cmd = [executable]
        if args:
            cmd.extend(args)

        # Wrap with perf stat if enabled and available
        perf_output: Optional[Path] = None
        perf_available = False
        if perf_enabled:
            # Check if perf is available and has permission
            try:
                test_result = subprocess.run(
                    ["perf", "stat", "-e", "cycles", "--", "/bin/true"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if test_result.returncode == 0 or "counted" in test_result.stderr.lower():
                    perf_available = True
                else:
                    if self.verbose or self.debug >= 1:
                        self.console.print(
                            "[yellow]Warning: perf not available (permission denied or not installed). "
                            "Running without perf profiling.[/]"
                        )
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                if self.verbose or self.debug >= 1:
                    self.console.print(
                        "[yellow]Warning: perf not found. Running without perf profiling.[/]"
                    )

        if perf_enabled and perf_available:
            output_dir = perf_output_dir if perf_output_dir else Path(executable).parent
            perf_output = output_dir / perf_output_name
            events = ",".join(PERF_CACHE_EVENTS)
            interval_ms = int(perf_interval * 1000)
            # perf stat: -e events, -I interval_ms, -x separator, -o output file
            perf_cmd = [
                "perf", "stat",
                "-e", events,
                "-I", str(interval_ms),
                "-x", ",",
                "-o", str(perf_output),
                "--"
            ]
            cmd = perf_cmd + cmd

        # Debug output level 1: show commands
        if self.debug >= 1:
            env_str = " ".join(f"{k}={v}" for k, v in (env or {}).items())
            if env_str:
                self.console.print(f"[dim]$ {env_str} {' '.join(cmd)}[/]")
            else:
                self.console.print(f"[dim]$ {' '.join(cmd)}[/]")

        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path(executable).parent,
                env=run_env,
            )
            duration = time.time() - start

            # Always write log file when path is provided
            if log_file:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(log_file, "w") as f:
                    f.write(f"# Command: {' '.join(cmd)}\n")
                    f.write(f"# Duration: {duration:.3f}s\n")
                    f.write(f"# Exit code: {result.returncode}\n\n")
                    if result.stdout:
                        f.write("=== STDOUT ===\n")
                        f.write(result.stdout)
                        f.write("\n")
                    if result.stderr:
                        f.write("=== STDERR ===\n")
                        f.write(result.stderr)
                        f.write("\n")
                if self.debug >= 2:
                    self.console.print(f"[dim]  Log: {log_file}[/]")
            elif self.debug >= 2:
                # No log file specified, print summary only
                lines = (result.stdout + result.stderr).strip().split('\n')
                if len(lines) > 10:
                    self.console.print(
                        f"[dim]  ({len(lines)} lines of output, use log_file to capture)[/]")
                elif lines and lines[0]:
                    for line in lines[:10]:
                        self.console.print(f"[dim]  {line}[/]")

            # Determine status based on exit code
            if result.returncode == 0:
                status = Status.PASS
            elif result.returncode in (139, 134, 136):  # SEGV, ABRT, FPE
                status = Status.CRASH
            else:
                status = Status.FAIL

            checksum = self.extract_checksum(result.stdout)
            kernel_timings = self.extract_kernel_timings(result.stdout)
            e2e_timings = self.extract_e2e_timings(result.stdout)
            init_timings = self.extract_init_timings(result.stdout)
            parallel_task_timing = self.extract_parallel_task_timings(
                result.stdout)

            # Parse perf metrics if enabled
            perf_metrics = None
            perf_csv_path = None
            if perf_enabled and perf_output and perf_output.exists():
                perf_metrics = self.parse_perf_csv(perf_output)
                perf_csv_path = str(perf_output)

            return RunResult(
                status=status,
                duration_sec=duration,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                checksum=checksum,
                kernel_timings=kernel_timings,
                e2e_timings=e2e_timings,
                init_timings=init_timings,
                parallel_task_timing=parallel_task_timing,
                perf_metrics=perf_metrics,
                perf_csv_path=perf_csv_path,
            )
        except subprocess.TimeoutExpired:
            return RunResult(
                status=Status.TIMEOUT,
                duration_sec=float(timeout),
                exit_code=124,
                stdout="",
                stderr=f"Execution timed out after {timeout} seconds",
            )
        except Exception as e:
            return RunResult(
                status=Status.FAIL,
                duration_sec=time.time() - start,
                exit_code=-1,
                stdout="",
                stderr=str(e),
            )

    def extract_checksum(self, output: str) -> Optional[str]:
        """Extract checksum/result from benchmark output.

        Delegates to shared parse_checksum() which uses the LAST match.
        """
        return parse_checksum(output)

    def extract_kernel_timings(self, output: str) -> Dict[str, float]:
        """Extract kernel timing info from benchmark output."""
        return parse_kernel_timings(output)

    def extract_e2e_timings(self, output: str) -> Dict[str, float]:
        """Extract end-to-end timing info from benchmark output."""
        return parse_e2e_timings(output)

    def extract_init_timings(self, output: str) -> Dict[str, float]:
        """Extract runtime initialization timing info from benchmark output."""
        return parse_init_timings(output)

    def extract_parallel_task_timings(self, output: str) -> Optional[ParallelTaskTiming]:
        """Extract parallel region and task timing info from benchmark output.

        Parses lines like:
            'parallel.gemm[worker=0]: 0.001234s'
            'task.gemm:kernel[worker=0]: 0.001100s'

        Used for analyzing the impact of delayed MLIR optimizations.
        See docs/hypothesis.md for details.
        """
        result = ParallelTaskTiming()
        found_any = False

        # Pattern for parallel timings: parallel.<name>[worker=<id>]: <time>s
        parallel_pattern = r"parallel\.([^\[]+)\[worker=(\d+)\]:\s*([0-9.]+)s"
        for match in re.finditer(parallel_pattern, output):
            name = match.group(1)
            worker_id = int(match.group(2))
            time_sec = float(match.group(3))

            if name not in result.parallel_timings:
                result.parallel_timings[name] = []
            result.parallel_timings[name].append(
                WorkerTiming(worker_id, time_sec))
            found_any = True

        # Pattern for task timings: task.<name>[worker=<id>]: <time>s
        task_pattern = r"task\.([^\[]+)\[worker=(\d+)\]:\s*([0-9.]+)s"
        for match in re.finditer(task_pattern, output):
            name = match.group(1)
            worker_id = int(match.group(2))
            time_sec = float(match.group(3))

            if name not in result.task_timings:
                result.task_timings[name] = []
            result.task_timings[name].append(WorkerTiming(worker_id, time_sec))
            found_any = True

        return result if found_any else None

    def parse_perf_csv(self, perf_output: Path) -> Optional[PerfCacheMetrics]:
        """Parse perf stat CSV output and return aggregated cache metrics.

        perf stat -x, -I <ms> format outputs lines like:
            <interval>,<value>,<unit>,<event>,<variance>,...
        For aggregate (final) stats, interval is empty or shows total.

        We aggregate all interval values into totals.
        """
        if not perf_output.exists():
            return None

        metrics = PerfCacheMetrics()
        event_totals: Dict[str, int] = {}

        try:
            with open(perf_output, "r") as f:
                for line in f:
                    line = line.strip()
                    # Skip comment lines and empty lines
                    if not line or line.startswith("#"):
                        continue

                    parts = line.split(",")
                    if len(parts) < 4:
                        continue

                    # Format: interval,value,unit,event,...
                    # or for some perf versions: value,unit,event,...
                    try:
                        # Try interval format first (perf stat -I)
                        if parts[0] and parts[0].replace(".", "").isdigit():
                            # Has interval timestamp
                            value_str = parts[1]
                            event_name = parts[3] if len(parts) > 3 else parts[2]
                        else:
                            # No interval or empty interval
                            value_str = parts[0]
                            event_name = parts[2] if len(parts) > 2 else ""

                        # Skip if value is not a number or is <not counted>
                        if not value_str or value_str.startswith("<"):
                            continue

                        value = int(float(value_str))
                        event_name = event_name.strip()

                        if event_name:
                            event_totals[event_name] = event_totals.get(event_name, 0) + value
                    except (ValueError, IndexError):
                        continue

            # Map event totals to metrics
            metrics.cache_references = event_totals.get("cache-references", 0)
            metrics.cache_misses = event_totals.get("cache-misses", 0)
            metrics.l1d_loads = event_totals.get("L1-dcache-loads", 0)
            metrics.l1d_load_misses = event_totals.get("L1-dcache-load-misses", 0)

            if metrics.cache_references > 0:
                metrics.cache_miss_rate = metrics.cache_misses / metrics.cache_references
            if metrics.l1d_loads > 0:
                metrics.l1d_load_miss_rate = metrics.l1d_load_misses / metrics.l1d_loads

            return metrics

        except Exception:
            return None

    def verify_correctness(
        self,
        arts_result: RunResult,
        omp_result: RunResult,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> VerificationResult:
        """Compare ARTS output against OpenMP reference."""
        if arts_result.status != Status.PASS or omp_result.status != Status.PASS:
            return VerificationResult(
                correct=False,
                arts_checksum=arts_result.checksum,
                omp_checksum=omp_result.checksum,
                tolerance_used=tolerance,
                note="Cannot verify: one or both runs failed",
            )

        if arts_result.checksum is None or omp_result.checksum is None:
            return VerificationResult(
                correct=False,
                arts_checksum=arts_result.checksum,
                omp_checksum=omp_result.checksum,
                tolerance_used=tolerance,
                note="Cannot verify: checksum not found in output",
            )

        try:
            arts_val = float(arts_result.checksum)
            omp_val = float(omp_result.checksum)

            if omp_val == 0:
                correct = abs(arts_val) < tolerance
            else:
                correct = abs((arts_val - omp_val) / omp_val) < tolerance

            if correct:
                note = "Checksums match within tolerance"
            else:
                note = f"Checksums differ: ARTS={arts_val}, OMP={omp_val}"

            return VerificationResult(
                correct=correct,
                arts_checksum=arts_result.checksum,
                omp_checksum=omp_result.checksum,
                tolerance_used=tolerance,
                note=note,
            )
        except ValueError:
            # String comparison fallback
            correct = arts_result.checksum.strip() == omp_result.checksum.strip()
            return VerificationResult(
                correct=correct,
                arts_checksum=arts_result.checksum,
                omp_checksum=omp_result.checksum,
                tolerance_used=tolerance,
                note="String comparison" if correct else "String mismatch",
            )

    def calculate_timing(
        self,
        arts_result: RunResult,
        omp_result: RunResult,
    ) -> TimingResult:
        """Calculate speedup preferring E2E timings when available."""
        arts_kernel = get_kernel_time(arts_result)
        omp_kernel = get_kernel_time(omp_result)
        # Prefer counter-based E2E time from ARTS introspection JSON
        arts_e2e = arts_result.counter_e2e_sec if arts_result.counter_e2e_sec is not None else get_e2e_time(arts_result)
        omp_e2e = get_e2e_time(omp_result)
        # Prefer counter-based init time from ARTS introspection JSON
        arts_init = arts_result.counter_init_sec if arts_result.counter_init_sec is not None else get_init_time(arts_result)
        omp_init = get_init_time(omp_result)
        arts_total = arts_result.duration_sec
        omp_total = omp_result.duration_sec

        if arts_result.status != Status.PASS or omp_result.status != Status.PASS:
            return TimingResult(
                arts_time_sec=arts_total,
                omp_time_sec=omp_total,
                speedup=0.0,
                note="Cannot calculate: one or both runs failed",
                arts_kernel_sec=arts_kernel,
                omp_kernel_sec=omp_kernel,
                arts_e2e_sec=arts_e2e,
                omp_e2e_sec=omp_e2e,
                arts_init_sec=arts_init,
                omp_init_sec=omp_init,
                arts_total_sec=arts_total,
                omp_total_sec=omp_total,
                speedup_basis=(
                    "e2e"
                    if (arts_e2e is not None and omp_e2e is not None)
                    else "kernel"
                    if (arts_kernel is not None and omp_kernel is not None)
                    else "total"
                ),
            )

        # Prefer E2E timings when both are available, otherwise fall back to kernel timings,
        # otherwise fall back to total process duration.
        if arts_e2e is not None and omp_e2e is not None:
            arts_time = arts_e2e
            omp_time = omp_e2e
            speedup_basis = "e2e"
        elif arts_kernel is not None and omp_kernel is not None:
            arts_time = arts_kernel
            omp_time = omp_kernel
            speedup_basis = "kernel"
        else:
            arts_time = arts_total
            omp_time = omp_total
            speedup_basis = "total"

        if arts_time == 0:
            speedup = 0.0
            note = f"ARTS {speedup_basis} time is zero"
        else:
            speedup = omp_time / arts_time
            if speedup > 1:
                note = f"ARTS is {speedup:.2f}x faster ({speedup_basis})"
            elif speedup < 1:
                note = f"OpenMP is {1/speedup:.2f}x faster ({speedup_basis})"
            else:
                note = f"Same performance ({speedup_basis})"

        return TimingResult(
            arts_time_sec=arts_time,
            omp_time_sec=omp_time,
            speedup=speedup,
            note=note,
            arts_kernel_sec=arts_kernel,
            omp_kernel_sec=omp_kernel,
            arts_e2e_sec=arts_e2e,
            omp_e2e_sec=omp_e2e,
            arts_init_sec=arts_init,
            omp_init_sec=omp_init,
            arts_total_sec=arts_total,
            omp_total_sec=omp_total,
            speedup_basis=speedup_basis,
        )

    def collect_artifacts(self, bench_path: Path) -> Artifacts:
        """Collect all artifact paths for a benchmark."""
        build_dir = bench_path / "build"
        counters_dir = bench_path / "counters"

        artifacts = Artifacts(benchmark_dir=str(bench_path))

        if build_dir.exists():
            artifacts.build_dir = str(build_dir)

        # Find executables
        for exe in bench_path.glob("*_arts"):
            if exe.is_file() and os.access(exe, os.X_OK):
                artifacts.executable_arts = str(exe)
                break

        # OpenMP reference binaries usually live under build/ (common/carts.mk).
        for exe in list(bench_path.glob("*_omp")) + list(build_dir.glob("*_omp") if build_dir.exists() else []):
            if exe.is_file() and os.access(exe, os.X_OK):
                artifacts.executable_omp = str(exe)
                break

        # Find CARTS metadata JSON (compiler-generated analysis)
        carts_meta = bench_path / ".carts-metadata.json"
        if carts_meta.exists():
            artifacts.carts_metadata = str(carts_meta)

        # Find ARTS metadata MLIR (MLIR with embedded metadata attributes)
        for mlir in bench_path.glob("*_arts_metadata.mlir"):
            artifacts.arts_metadata_mlir = str(mlir)
            break

        # Find arts.cfg (ARTS runtime configuration)
        arts_cfg = bench_path / "arts.cfg"
        if arts_cfg.exists():
            artifacts.arts_config = str(arts_cfg)

        # Collect counter files
        if counters_dir.exists():
            artifacts.counters_dir = str(counters_dir)
            artifacts.counter_files = sorted(
                str(f) for f in counters_dir.glob("*.json")
            )

        return artifacts

    def run_single(
        self,
        name: str,
        size: str = DEFAULT_SIZE,
        timeout: int = DEFAULT_TIMEOUT,
        verify: bool = True,
        arts_config: Optional[Path] = None,
        threads_override: Optional[int] = None,
        nodes_override: Optional[int] = None,
        launcher_override: Optional[str] = None,
        omp_threads_override: Optional[int] = None,
        counter_dir: Optional[Path] = None,
        arts_exec_args: Optional[str] = None,
        phase_callback: Optional[Callable[[Phase], None]] = None,
        partial_results: Optional[Dict[str, Any]] = None,
        perf_enabled: bool = False,
        perf_interval: float = 0.1,
        run_number: int = 1,
        run_timestamp: str = "",
        perf_dir: Optional[Path] = None,
    ) -> BenchmarkResult:
        """Run complete pipeline for a single benchmark.

        Args:
            phase_callback: Optional callback invoked when phase changes.
                           Used by run_all to update live display.
            partial_results: Optional dict to store partial results as phases complete.
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        bench_path = self.benchmarks_dir / name
        suite = name.split("/")[0] if "/" in name else ""

        # Determine effective config template:
        # 1. Use explicitly provided arts_config (resolve to absolute path for make CWD)
        # 2. Fall back to benchmark's own arts.cfg
        # 3. Fall back to default carts-benchmarks config
        effective_config = arts_config.resolve() if arts_config else None
        if effective_config is None:
            candidate = bench_path / "arts.cfg"
            if candidate.exists():
                effective_config = candidate
            else:
                effective_config = DEFAULT_ARTS_CONFIG

        base_threads = get_arts_cfg_int(effective_config, "threads") or 1
        base_nodes = get_arts_cfg_int(effective_config, "nodeCount") or 1
        base_launcher = get_arts_cfg_str(effective_config, "launcher") or "ssh"

        desired_threads = threads_override if threads_override is not None else base_threads
        desired_nodes = nodes_override if nodes_override is not None else base_nodes
        desired_launcher = launcher_override if launcher_override is not None else base_launcher

        # Compute config early (needed by artifact_manager)
        actual_omp_threads = (
            omp_threads_override if omp_threads_override is not None else desired_threads
        )
        config = BenchmarkConfig(
            arts_threads=desired_threads,
            arts_nodes=desired_nodes,
            omp_threads=actual_omp_threads,
            launcher=desired_launcher,
        )
        am = self.artifact_manager  # shorthand (may be None)

        # Skip benchmarks disabled for multi-node when running with nodeCount > 1
        if desired_nodes > 1 and (bench_path / ".disable-multinode").exists():
            return self._make_skip_result(
                name, size,
                "Benchmark disabled for multi-node (has .disable-multinode marker)",
                config,
            )

        # Counter directory: artifact_manager path or explicit --counter-dir
        safe_bench_name = name.replace("/", "_")
        run_counter_dir: Optional[Path] = None
        if am:
            run_counter_dir = am.get_counter_dir(name, config, run_number)
        elif counter_dir is not None:
            run_counter_dir = counter_dir / run_timestamp / f"{safe_bench_name}_{run_number}"
            run_counter_dir.mkdir(parents=True, exist_ok=True)

        # Generate config with overrides only if values actually differ from base
        need_generated = False
        if threads_override is not None and threads_override != base_threads:
            need_generated = True
        if nodes_override is not None and nodes_override != base_nodes:
            need_generated = True
        if launcher_override is not None and launcher_override != base_launcher:
            need_generated = True
        if run_counter_dir is not None:
            need_generated = True

        effective_arts_cfg: Path
        if need_generated:
            effective_arts_cfg = generate_arts_config(
                effective_config, desired_threads, run_counter_dir, desired_launcher, nodes_override
            )
        else:
            effective_arts_cfg = effective_config

        # Clean before building to avoid stale artifacts
        if self.clean:
            self.clean_benchmark(name)

        # Build ARTS version
        if phase_callback:
            phase_callback(Phase.BUILD_ARTS)
        build_arts = self.build_benchmark(
            name, size, "arts", effective_arts_cfg, "", arts_exec_args)
        if partial_results is not None:
            partial_results["build_arts"] = build_arts

        # Build OpenMP version - skip for multi-node since OMP is single-node only
        if phase_callback:
            phase_callback(Phase.BUILD_OMP)
        if desired_nodes == 1:
            build_omp = self.build_benchmark(name, size, "openmp", None)
        else:
            build_omp = BuildResult(
                status=Status.SKIP,
                duration_sec=0.0,
                output="Skipped: OpenMP does not support multi-node execution",
                executable=None,
            )
        if partial_results is not None:
            partial_results["build_omp"] = build_omp

        # Copy build artifacts via artifact manager (ONCE per config)
        artifact_paths: Dict[str, Optional[str]] = {}
        if am:
            artifact_paths = am.copy_build_artifacts(
                bench_path, name, config, arts_cfg_used=effective_arts_cfg)

        # Setup log files
        arts_log: Optional[Path] = None
        omp_log: Optional[Path] = None
        if am:
            run_dir = am.get_run_dir(name, config, run_number)
            arts_log = run_dir / "arts.log"
            omp_log = run_dir / "omp.log"
        elif self.debug >= 2:
            logs_dir = bench_path / "logs"
            arts_log = logs_dir / "arts.log"
            omp_log = logs_dir / "omp.log"

        run_args = self.get_run_args(bench_path, size)
        verify_tolerance = self.get_verify_tolerance(bench_path)

        # Ensure init timing is reported and align OMP wait/spin policy with ARTS.
        common_env: Dict[str, str] = {}
        if "CARTS_BENCHMARKS_REPORT_INIT" not in os.environ:
            common_env["CARTS_BENCHMARKS_REPORT_INIT"] = "1"

        # Setup perf output paths
        perf_output_dir: Optional[Path] = None
        arts_perf_name = omp_perf_name = None
        arts_perf_main = arts_perf_temp = omp_perf_main = omp_perf_temp = None

        if perf_enabled and am:
            # Perf CSVs land inside the run directory
            perf_output_dir = am.get_perf_dir(name, config, run_number)
            arts_perf_name = "arts_cache.csv"
            omp_perf_name = "omp_cache.csv"
        elif perf_enabled:
            effective_perf_dir = perf_dir or Path("./perfs")
            if run_timestamp:
                perf_timestamp_dir = effective_perf_dir / run_timestamp
                perf_timestamp_dir.mkdir(parents=True, exist_ok=True)
                perf_output_dir = perf_timestamp_dir
                arts_perf_main = perf_timestamp_dir / f"{safe_bench_name}_arts.csv"
                arts_perf_temp = perf_timestamp_dir / f"_temp_{safe_bench_name}_arts_{run_number}.csv"
                omp_perf_main = perf_timestamp_dir / f"{safe_bench_name}_omp.csv"
                omp_perf_temp = perf_timestamp_dir / f"_temp_{safe_bench_name}_omp_{run_number}.csv"
                arts_perf_name = arts_perf_temp.name
                omp_perf_name = omp_perf_temp.name
            else:
                effective_perf_dir.mkdir(parents=True, exist_ok=True)
                perf_output_dir = effective_perf_dir

        # Clean up stale ARTS port before run
        self._cleanup_port()

        # Run ARTS version (config is embedded at compile time;
        # counter_dir overrides embedded counterFolder via env var)
        if phase_callback:
            phase_callback(Phase.RUN_ARTS)
        if build_arts.status == Status.PASS and build_arts.executable:
            run_arts = self.run_benchmark(
                build_arts.executable,
                timeout,
                env=common_env,
                launcher=desired_launcher,
                node_count=desired_nodes,
                threads=desired_threads,
                args=run_args,
                log_file=arts_log,
                perf_enabled=perf_enabled,
                perf_interval=perf_interval,
                perf_output_name=arts_perf_name or "perf_cache_arts.csv",
                perf_output_dir=perf_output_dir,
                counter_dir=run_counter_dir,
            )
            # Append temp perf data to main CSV (legacy path, not used with artifact_manager)
            if perf_enabled and arts_perf_temp and arts_perf_main:
                append_perf_to_main_csv(arts_perf_temp, arts_perf_main, run_number)
        else:
            run_arts = RunResult(
                status=Status.SKIP, duration_sec=0.0,
                exit_code=-1, stdout="", stderr="Build failed",
            )

        # Parse counter JSON for timing data from run-specific counter directory
        if run_counter_dir and run_counter_dir.exists():
            init_sec, e2e_sec = parse_counter_json(run_counter_dir)
            run_arts.counter_init_sec = init_sec
            run_arts.counter_e2e_sec = e2e_sec

        if partial_results is not None:
            partial_results["run_arts"] = run_arts

        # Run OpenMP version
        if phase_callback:
            phase_callback(Phase.RUN_OMP)
        if build_omp.status == Status.PASS and build_omp.executable:
            omp_env = dict(common_env)
            omp_env["OMP_NUM_THREADS"] = str(actual_omp_threads)
            if "OMP_WAIT_POLICY" not in os.environ:
                omp_env["OMP_WAIT_POLICY"] = "ACTIVE"
            run_omp = self.run_benchmark(
                build_omp.executable,
                timeout,
                env=omp_env,
                args=run_args,
                log_file=omp_log,
                perf_enabled=perf_enabled,
                perf_interval=perf_interval,
                perf_output_name=omp_perf_name or "perf_cache_omp.csv",
                perf_output_dir=perf_output_dir,
            )
            # Append temp perf data to main CSV (legacy path)
            if perf_enabled and omp_perf_temp and omp_perf_main:
                append_perf_to_main_csv(omp_perf_temp, omp_perf_main, run_number)
        else:
            skip_reason = (
                "Skipped: OpenMP does not support multi-node execution"
                if desired_nodes > 1
                else "Build failed"
            )
            run_omp = RunResult(
                status=Status.SKIP, duration_sec=0.0,
                exit_code=-1, stdout="", stderr=skip_reason,
            )

        # Print trace output if enabled
        if self.trace:
            arts_output = filter_benchmark_output(run_arts.stdout)
            omp_output = filter_benchmark_output(run_omp.stdout)

            self.console.print(
                f"\n[bold cyan]═══ CARTS Output ({name}) ═══[/]")
            self.console.print(arts_output or "[dim](no benchmark output)[/]")

            self.console.print(f"\n[bold green]═══ OMP Output ({name}) ═══[/]")
            self.console.print(omp_output or "[dim](no benchmark output)[/]")
            self.console.print()

        # Calculate timing
        timing = self.calculate_timing(run_arts, run_omp)

        # Verify correctness
        if verify:
            verification = self.verify_correctness(
                run_arts, run_omp, tolerance=verify_tolerance
            )
        else:
            verification = VerificationResult(
                correct=False,
                arts_checksum=run_arts.checksum,
                omp_checksum=run_omp.checksum,
                tolerance_used=0.0,
                note="Verification disabled",
            )

        # Collect artifacts
        artifacts = self.collect_artifacts(bench_path)

        # Update artifacts with experiment directory paths when using artifact_manager
        has_counters = False
        has_perf = False
        if am:
            run_dir = am.get_run_dir(name, config, run_number)
            artifacts_dir = am.get_artifacts_dir(name, config)
            for key, attr in [
                ("arts_config", "arts_config"),
                ("carts_metadata", "carts_metadata"),
                ("arts_metadata_mlir", "arts_metadata_mlir"),
                ("executable_arts", "executable_arts"),
                ("executable_omp", "executable_omp"),
            ]:
                if artifact_paths.get(key):
                    setattr(artifacts, attr, artifact_paths[key])
            artifacts.build_dir = str(artifacts_dir)
            artifacts.run_dir = str(run_dir)
            artifacts.arts_log = str(arts_log) if arts_log else None
            artifacts.omp_log = str(omp_log) if omp_log else None

            # Counter files (already landed in place via counterFolder)
            counter_path = am.get_counter_dir(name, config, run_number)
            counter_files = sorted(counter_path.glob("*.json"))
            if counter_files:
                has_counters = True
                artifacts.counters_dir = str(counter_path)
                artifacts.counter_files = [str(f) for f in counter_files]

            # Perf
            if perf_enabled and perf_output_dir and perf_output_dir.exists():
                if list(perf_output_dir.glob("*.csv")):
                    has_perf = True

            am.record_run(name, config, run_number,
                          has_counters=has_counters, has_perf=has_perf)
        else:
            if effective_arts_cfg:
                artifacts.arts_config = str(effective_arts_cfg)
            if arts_log is not None:
                artifacts.arts_log = str(arts_log)
            if omp_log is not None:
                artifacts.omp_log = str(omp_log)

        total_duration = time.time() - start_time

        return BenchmarkResult(
            name=name,
            suite=suite,
            size=size,
            config=config,
            run_number=run_number,
            build_arts=build_arts,
            build_omp=build_omp,
            run_arts=run_arts,
            run_omp=run_omp,
            timing=timing,
            verification=verification,
            artifacts=artifacts,
            timestamp=timestamp,
            total_duration_sec=total_duration,
            size_params=self.get_size_params(bench_path, size),
        )

    def run_all(
        self,
        benchmarks: List[str],
        size: str = DEFAULT_SIZE,
        timeout: int = DEFAULT_TIMEOUT,
        verify: bool = True,
        arts_config: Optional[Path] = None,
        threads_override: Optional[int] = None,
        nodes_override: Optional[int] = None,
        launcher_override: Optional[str] = None,
        omp_threads_override: Optional[int] = None,
        counter_dir: Optional[Path] = None,
        arts_exec_args: Optional[str] = None,
        perf_enabled: bool = False,
        perf_interval: float = 0.1,
        runs: int = 1,
        run_timestamp: str = "",
        perf_dir: Optional[Path] = None,
    ) -> List[BenchmarkResult]:
        """Run benchmark suite.
        """
        results_dict: Dict[str, List[BenchmarkResult]] = {}
        results_list: List[BenchmarkResult] = []
        start_time = time.time()

        if self.quiet:
            # Quiet mode - no live display
            for bench in benchmarks:
                for run_num in range(1, runs + 1):
                    result = self.run_single(
                        bench,
                        size,
                        timeout,
                        verify,
                        arts_config,
                        threads_override,
                        nodes_override,
                        launcher_override,
                        omp_threads_override,
                        counter_dir,
                        arts_exec_args,
                        perf_enabled=perf_enabled,
                        perf_interval=perf_interval,
                        run_number=run_num,
                        run_timestamp=run_timestamp,
                        perf_dir=perf_dir,
                    )
                    results_list.append(result)
            self.results = results_list
            return results_list

        # Live display mode - show table that updates as benchmarks complete
        # Track current phase for live display updates
        current_phase: List[Optional[Phase]] = [
            None]  # Use list for mutability in closure
        current_bench: List[Optional[str]] = [None]
        # Store partial results for current benchmark to show kernel times during RUN_OMP
        current_partial: List[Optional[Dict[str, Any]]] = [None]

        def phase_callback(phase: Phase) -> None:
            """Update display when phase changes."""
            current_phase[0] = phase
            elapsed = time.time() - start_time
            live.update(create_live_display(
                benchmarks, results_dict, current_bench[0], elapsed, phase, current_partial[0], total_runs=runs))

        with Live(
            create_live_display(benchmarks, results_dict, None, 0, None, None, total_runs=runs),
            console=self.console,
            refresh_per_second=4,
        ) as live:
            for bench in benchmarks:
                for run_num in range(1, runs + 1):
                    current_bench[0] = f"{bench} (run {run_num}/{runs})" if runs > 1 else bench
                    current_partial[0] = {}
                    # Update display to show current benchmark as in-progress
                    elapsed = time.time() - start_time
                    live.update(create_live_display(
                        benchmarks, results_dict, bench, elapsed, Phase.BUILD_ARTS, current_partial[0], total_runs=runs))

                    # Run benchmark with phase callback and partial results storage
                    try:
                        result = self.run_single(
                            bench,
                            size,
                            timeout,
                            verify,
                            arts_config,
                            threads_override,
                            nodes_override,
                            launcher_override,
                            omp_threads_override,
                            counter_dir,
                            arts_exec_args,
                            phase_callback,
                            current_partial[0],
                            perf_enabled=perf_enabled,
                            perf_interval=perf_interval,
                            run_number=run_num,
                            run_timestamp=run_timestamp,
                            perf_dir=perf_dir,
                        )
                    except Exception as e:
                        # Log error and continue to next benchmark
                        result = self._make_error_result(bench, size, str(e))

                    # Update results and refresh display
                    results_dict.setdefault(bench, []).append(result)
                    results_list.append(result)
                    current_bench[0] = None
                    current_partial[0] = None
                    elapsed = time.time() - start_time
                    live.update(create_live_display(
                        benchmarks, results_dict, None, elapsed, None, None, total_runs=runs))

        self.results = results_list
        return results_list

    def _run_parallel(
        self,
        benchmarks: List[str],
        size: str,
        timeout: int,
        n_workers: int,
        verify: bool,
        arts_config: Optional[Path],
    ) -> List[BenchmarkResult]:
        """Execute benchmarks in parallel using process pool."""
        results_dict: Dict[str, BenchmarkResult] = {}
        results_list: List[BenchmarkResult] = []
        start_time = time.time()
        # All benchmarks start as in-progress
        in_progress: set = set(benchmarks)

        if self.quiet:
            # Quiet mode - no live display
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        _run_single_worker,
                        str(self.benchmarks_dir),
                        bench,
                        size,
                        timeout,
                        verify,
                        str(arts_config) if arts_config else None,
                        clean=self.clean,
                    ): bench
                    for bench in benchmarks
                }

                for future in as_completed(futures):
                    bench = futures[future]
                    try:
                        result = future.result()
                        results_list.append(result)
                    except Exception as e:
                        results_list.append(
                            self._make_error_result(bench, size, str(e)))

            self.results = results_list
            return results_list

        # Live display mode - show table that updates as benchmarks complete
        with Live(
            create_live_display(benchmarks, results_dict,
                                f"[parallel={n_workers}]", 0),
            console=self.console,
            refresh_per_second=4,
        ) as live:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        _run_single_worker,
                        str(self.benchmarks_dir),
                        bench,
                        size,
                        timeout,
                        verify,
                        str(arts_config) if arts_config else None,
                        clean=self.clean,
                    ): bench
                    for bench in benchmarks
                }

                for future in as_completed(futures):
                    bench = futures[future]
                    try:
                        result = future.result()
                        results_dict[bench] = result
                        results_list.append(result)
                    except Exception as e:
                        error_result = self._make_error_result(
                            bench, size, str(e))
                        results_dict[bench] = error_result
                        results_list.append(error_result)

                    in_progress.remove(bench)
                    elapsed = time.time() - start_time

                    # Show one of the remaining in-progress benchmarks
                    current_in_progress = next(iter(in_progress), None)
                    live.update(create_live_display(
                        benchmarks, results_dict, current_in_progress, elapsed))

        self.results = results_list
        return results_list

    def _make_error_result(
        self,
        name: str,
        size: str,
        error: str,
    ) -> BenchmarkResult:
        """Create an error result for a failed benchmark."""
        bench_path = self.benchmarks_dir / name
        suite = name.split("/")[0] if "/" in name else ""

        failed_build = BuildResult(
            status=Status.FAIL,
            duration_sec=0.0,
            output=error,
        )
        failed_run = RunResult(
            status=Status.SKIP,
            duration_sec=0.0,
            exit_code=-1,
            stdout="",
            stderr=error,
        )

        # Default config for error results
        config = BenchmarkConfig(
            arts_threads=1,
            arts_nodes=1,
            omp_threads=1,
            launcher="local",
        )

        return BenchmarkResult(
            name=name,
            suite=suite,
            size=size,
            config=config,
            run_number=1,
            build_arts=failed_build,
            build_omp=failed_build,
            run_arts=failed_run,
            run_omp=failed_run,
            timing=TimingResult(0.0, 0.0, 0.0, "Error"),
            verification=VerificationResult(False, None, None, 0.0, error),
            artifacts=Artifacts(benchmark_dir=str(bench_path)),
            timestamp=datetime.now().isoformat(),
            total_duration_sec=0.0,
            size_params=self.get_size_params(bench_path, size),
        )

    def _make_skip_result(
        self,
        name: str,
        size: str,
        reason: str,
        config: Optional[BenchmarkConfig] = None,
    ) -> BenchmarkResult:
        """Create a skip result for a benchmark that should not run."""
        bench_path = self.benchmarks_dir / name
        suite = name.split("/")[0] if "/" in name else ""

        skip_build = BuildResult(status=Status.SKIP, duration_sec=0.0, output=reason)
        skip_run = RunResult(
            status=Status.SKIP, duration_sec=0.0, exit_code=-1, stdout="", stderr=reason
        )

        if config is None:
            config = BenchmarkConfig(
                arts_threads=1, arts_nodes=1, omp_threads=1, launcher="local"
            )

        return BenchmarkResult(
            name=name,
            suite=suite,
            size=size,
            config=config,
            run_number=1,
            build_arts=skip_build,
            build_omp=skip_build,
            run_arts=skip_run,
            run_omp=skip_run,
            timing=TimingResult(0.0, 0.0, 0.0, reason),
            verification=VerificationResult(False, None, None, 0.0, reason),
            artifacts=Artifacts(benchmark_dir=str(bench_path)),
            timestamp=datetime.now().isoformat(),
            total_duration_sec=0.0,
            size_params=None,
        )

    def _cleanup_port(self, port: int = DEFAULT_ARTS_PORT) -> None:
        """Kill any process holding the ARTS port."""
        try:
            subprocess.run(
                ["fuser", "-k", f"{port}/tcp"],
                capture_output=True, timeout=5, check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    def clean_benchmark(self, name: str) -> bool:
        """Clean build artifacts for a benchmark."""
        bench_path = self.benchmarks_dir / name

        if not bench_path.exists():
            return False

        cleaned = False

        for dirname in BENCHMARK_CLEAN_DIR_NAMES:
            dirpath = bench_path / dirname
            if dirpath.exists():
                shutil.rmtree(dirpath)
                cleaned = True

        for pattern in BENCHMARK_CLEAN_DIR_GLOBS:
            for dirpath in bench_path.glob(pattern):
                if dirpath.is_dir():
                    shutil.rmtree(dirpath)
                    cleaned = True

        for pattern in BENCHMARK_CLEAN_FILE_GLOBS:
            for f in bench_path.glob(pattern):
                if not f.is_file():
                    continue
                f.unlink()
                cleaned = True

        for exe in bench_path.glob("*_arts"):
            if exe.is_file():
                exe.unlink()
                cleaned = True
        for exe in bench_path.glob("*_omp"):
            if exe.is_file():
                exe.unlink()
                cleaned = True

        return cleaned

    def clean_shared_artifacts(self) -> int:
        """Clean benchmark-runner shared artifacts under external/carts-benchmarks."""
        removed = 0

        for dirname in BENCHMARK_SHARED_CLEAN_DIR_NAMES:
            dirpath = self.benchmarks_dir / dirname
            if dirpath.exists():
                shutil.rmtree(dirpath)
                removed += 1

        for pattern in ["core", "core.*", "vgcore.*", "*.log", "*.tmp"]:
            for f in self.benchmarks_dir.glob(pattern):
                if not f.is_file():
                    continue
                f.unlink()
                removed += 1

        return removed


# Worker function for parallel execution (must be at module level for pickling)
def _run_single_worker(
    benchmarks_dir: str,
    name: str,
    size: str,
    timeout: int,
    verify: bool,
    arts_config: Optional[str],
    threads_override: Optional[int] = None,
    nodes_override: Optional[int] = None,
    launcher_override: Optional[str] = None,
    omp_threads_override: Optional[int] = None,
    counter_dir: Optional[str] = None,
    clean: bool = True,
) -> BenchmarkResult:
    """Worker function for parallel benchmark execution."""
    runner = BenchmarkRunner(
        Console(force_terminal=False), quiet=True, clean=clean)
    runner.benchmarks_dir = Path(benchmarks_dir)
    return runner.run_single(
        name,
        size,
        timeout,
        verify,
        Path(arts_config) if arts_config else None,
        threads_override,
        nodes_override,
        launcher_override,
        omp_threads_override,
        Path(counter_dir) if counter_dir else None,
    )


# ============================================================================
# Output Helpers
# ============================================================================


def status_text(status: Status) -> Text:
    """Create colored text for a status."""
    if status == Status.PASS:
        return Text("PASS", style="bold green")
    elif status == Status.FAIL:
        return Text("FAIL", style="bold red")
    elif status == Status.CRASH:
        return Text("CRASH", style="bold red")
    elif status == Status.TIMEOUT:
        return Text("TIMEOUT", style="bold yellow")
    elif status == Status.SKIP:
        return Text("SKIP", style="dim")
    else:
        return Text("N/A", style="dim")


def status_symbol(status: Status) -> str:
    """Get symbol for a status."""
    if status == Status.PASS:
        return "[green]\u2713[/]"
    elif status == Status.FAIL:
        return "[red]\u2717[/]"
    elif status == Status.CRASH:
        return "[red]\u2717[/]"
    elif status == Status.TIMEOUT:
        return "[yellow]\u23f1[/]"
    elif status == Status.SKIP:
        return "[dim]\u25cb[/]"
    else:
        return "[dim]-[/]"


def format_duration(seconds: float) -> str:
    """Format duration for display."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def get_kernel_time(run_result: RunResult) -> Optional[float]:
    """Get total kernel time from run result (sum of all kernel timings)."""
    if run_result.kernel_timings:
        return sum(run_result.kernel_timings.values())
    return None


def get_e2e_time(run_result: RunResult) -> Optional[float]:
    """Get total end-to-end time from run result (sum of all e2e timings)."""
    if run_result.e2e_timings:
        return sum(run_result.e2e_timings.values())
    return None


def get_init_time(run_result: RunResult) -> Optional[float]:
    """Get total runtime init time from run result (sum of all init timings)."""
    if run_result.init_timings:
        # Prefer the canonical keys when present to avoid accidental double counting
        # if additional init signals are added (e.g., init.arts_runtime).
        if "arts" in run_result.init_timings:
            return run_result.init_timings["arts"]
        if "omp" in run_result.init_timings:
            return run_result.init_timings["omp"]
        return sum(run_result.init_timings.values())
    return None


def format_kernel_time(run_result: RunResult) -> Tuple[Optional[float], str]:
    """Format kernel time for display. Returns (total_time, display_string).

    For single kernel: returns (time, "0.1234s")
    For multiple kernels: returns (sum, "0.5678s [3]") where [3] is kernel count
    """
    if not run_result.kernel_timings:
        return None, ""

    total = sum(run_result.kernel_timings.values())
    count = len(run_result.kernel_timings)

    if count == 1:
        return total, f"{total:.4f}s"
    else:
        return total, f"{total:.4f}s [{count}]"


def format_e2e_time(run_result: RunResult) -> Tuple[Optional[float], str]:
    """Format end-to-end time for display. Returns (total_time, display_string)."""
    if not run_result.e2e_timings:
        return None, ""
    total = sum(run_result.e2e_timings.values())
    count = len(run_result.e2e_timings)
    if count == 1:
        return total, f"{total:.4f}s"
    return total, f"{total:.4f}s [{count}]"


def format_init_time(run_result: RunResult) -> Tuple[Optional[float], str]:
    """Format runtime init time for display. Returns (total_time, display_string)."""
    if not run_result.init_timings:
        return None, ""
    total = sum(run_result.init_timings.values())
    count = len(run_result.init_timings)
    if count == 1:
        return total, f"{total:.4f}s"
    return total, f"{total:.4f}s [{count}]"


def create_results_table(results: List[BenchmarkResult]) -> Table:
    """Create a rich table from benchmark results."""
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")

    table.add_column("Benchmark", style="cyan", no_wrap=True)
    table.add_column("Build", justify="center")
    table.add_column("ARTS Init", justify="right")
    table.add_column("ARTS E2E", justify="right")
    table.add_column("OMP Init", justify="right")
    table.add_column("OMP E2E", justify="right")
    table.add_column("Correct", justify="center")
    table.add_column("Speedup", justify="right")

    has_fallback = False
    has_multi_kernel = False
    has_multi_e2e = False
    for r in results:
        # Build status (combined)
        if r.build_arts.status == Status.PASS and r.build_omp.status == Status.PASS:
            build = f"[green]\u2713[/] {r.build_arts.duration_sec + r.build_omp.duration_sec:.1f}s"
        else:
            build = f"[red]\u2717[/] {r.build_arts.status.value}/{r.build_omp.status.value}"

        # ARTS Init time from counter JSON
        if r.run_arts.counter_init_sec is not None:
            arts_init = f"{r.run_arts.counter_init_sec:.4f}s"
        else:
            arts_init = "[dim]-[/]"

        # ARTS E2E time: prefer counter JSON, fall back to parsed stdout
        if r.run_arts.counter_e2e_sec is not None:
            arts_e2e = r.run_arts.counter_e2e_sec
            arts_e2e_str = f"{arts_e2e:.4f}s"
        else:
            arts_e2e, arts_e2e_str = format_e2e_time(r.run_arts)

        omp_e2e, omp_e2e_str = format_e2e_time(r.run_omp)

        # OMP Init time from parsed stdout
        omp_init = r.run_omp.init_timings.get("omp")
        if omp_init is not None:
            omp_init_str = f"{omp_init:.4f}s"
        else:
            omp_init_str = "[dim]-[/]"

        # Track if any benchmark has multiple kernels / e2e segments
        if r.run_arts.kernel_timings and len(r.run_arts.kernel_timings) > 1:
            has_multi_kernel = True
        if r.run_arts.e2e_timings and len(r.run_arts.e2e_timings) > 1:
            has_multi_e2e = True

        # Run status with e2e time (fall back to kernel, then total duration)
        if r.run_arts.status == Status.PASS:
            if arts_e2e is not None:
                run_arts = f"{status_symbol(r.run_arts.status)} {arts_e2e_str}"
            else:
                arts_kernel, arts_kernel_str = format_kernel_time(r.run_arts)
                if arts_kernel is not None:
                    run_arts = f"{status_symbol(r.run_arts.status)} {arts_kernel_str}*"
                else:
                    run_arts = f"{status_symbol(r.run_arts.status)} {r.run_arts.duration_sec:.2f}s*"
                has_fallback = True
        else:
            run_arts = f"{status_symbol(r.run_arts.status)} {r.run_arts.status.value}"

        if r.run_omp.status == Status.PASS:
            if omp_e2e is not None:
                run_omp = f"{status_symbol(r.run_omp.status)} {omp_e2e_str}"
            else:
                omp_kernel, omp_kernel_str = format_kernel_time(r.run_omp)
                if omp_kernel is not None:
                    run_omp = f"{status_symbol(r.run_omp.status)} {omp_kernel_str}*"
                else:
                    run_omp = f"{status_symbol(r.run_omp.status)} {r.run_omp.duration_sec:.2f}s*"
                has_fallback = True
        else:
            run_omp = f"{status_symbol(r.run_omp.status)} {r.run_omp.status.value}"

        # Correctness
        if r.verification.correct:
            correct = "[green]\u2713 YES[/]"
        elif r.verification.note == "Verification disabled":
            correct = "[dim]- N/A[/]"
        elif r.run_arts.status != Status.PASS or r.run_omp.status != Status.PASS:
            correct = "[dim]- N/A[/]"
        else:
            correct = "[red]\u2717 NO[/]"

        # Speedup (basis chosen in calculate_timing)
        if r.timing.speedup > 0:
            if r.timing.speedup >= 1.0:
                speedup = f"[green]{r.timing.speedup:.2f}x[/]"
            elif r.timing.speedup >= 0.8:
                speedup = f"[yellow]{r.timing.speedup:.2f}x[/]"
            else:
                speedup = f"[red]{r.timing.speedup:.2f}x[/]"
            if r.timing.speedup_basis != "e2e":
                speedup += "*"
                has_fallback = True
        else:
            speedup = "[dim]-[/]"

        table.add_row(
            r.name,
            build,
            arts_init,
            run_arts,
            omp_init_str,
            run_omp,
            correct,
            speedup,
        )

    # Build caption based on what notations are used
    captions = []
    if has_multi_kernel:
        captions.append("[N] = sum of N kernels")
    if has_multi_e2e:
        captions.append("[N] = sum of N e2e segments")
    if has_fallback:
        captions.append("* = speedup/time not based on e2e")
    if captions:
        table.caption = "[dim]" + "  |  ".join(captions) + "[/]"

    return table


def create_summary_panel(results: List[BenchmarkResult], duration: float) -> Panel:
    """Create a summary panel."""
    passed = sum(1 for r in results if r.run_arts.status ==
                 Status.PASS and r.verification.correct)
    failed = sum(1 for r in results if r.run_arts.status in (Status.FAIL, Status.CRASH) or
                 (r.run_arts.status == Status.PASS and not r.verification.correct))
    skipped = sum(1 for r in results if r.run_arts.status == Status.SKIP)

    # Calculate geometric mean speedup based on e2e time (preferred)
    import math
    speedups = []
    for r in results:
        if r.timing.speedup > 0:
            speedups.append(r.timing.speedup)

    if speedups:
        bases = {r.timing.speedup_basis for r in results if r.timing.speedup > 0}
        basis_label = next(iter(bases)) if len(bases) == 1 else "mixed"
        geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        speedup_text = f"Geometric mean speedup ({basis_label}): [cyan]{geomean:.2f}x[/]"
    else:
        speedup_text = ""

    content = (
        f"[green]\u2713 {passed}[/] passed  "
        f"[red]\u2717 {failed}[/] failed  "
        f"[dim]\u25cb {skipped}[/] skipped  "
        f"[cyan]\u23f1 {format_duration(duration)}[/]"
    )

    if speedup_text:
        content += f"\n\n{speedup_text}"

    return Panel(content, title="Summary", border_style="blue")


def _render_init_e2e_bar(init_sec: Optional[float], e2e_sec: Optional[float], max_total: float, width: int = 30) -> Text:
    """Render a stacked bar: init (red) + e2e (green)."""
    init_val = float(init_sec or 0.0)
    e2e_val = float(e2e_sec or 0.0)
    total = init_val + e2e_val

    if max_total <= 0.0:
        return Text("(no timing)", style="dim")

    total_len = int(round(width * (total / max_total)))
    init_len = int(round(width * (init_val / max_total)))
    init_len = min(init_len, total_len)
    e2e_len = max(0, total_len - init_len)

    bar = Text()
    if init_len > 0:
        bar.append("█" * init_len, style="red")
    if e2e_len > 0:
        bar.append("█" * e2e_len, style="green")
    if total_len < width:
        bar.append(" " * (width - total_len), style="dim")

    label = f" {total:.4f}s"
    if init_sec is not None and e2e_sec is not None:
        label += f" (init {init_val:.4f}s + e2e {e2e_val:.4f}s)"
    elif e2e_sec is not None:
        label += f" (e2e {e2e_val:.4f}s)"
    elif init_sec is not None:
        label += f" (init {init_val:.4f}s)"
    else:
        label += " (n/a)"
    bar.append(label, style="dim")
    return bar


def create_init_e2e_bar_chart(results: List[BenchmarkResult], width: int = 30) -> Table:
    """Create a stacked bar chart per benchmark: init + e2e (ARTS vs OMP)."""
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Benchmark", style="cyan", no_wrap=True)
    table.add_column("ARTS init+e2e", justify="left")
    table.add_column("OMP init+e2e", justify="left")

    # Scale bars by max (init+e2e) across all rows/runtimes.
    max_total = 0.0
    for r in results:
        for init_sec, e2e_sec in (
            (r.timing.arts_init_sec, r.timing.arts_e2e_sec),
            (r.timing.omp_init_sec, r.timing.omp_e2e_sec),
        ):
            init_val = float(init_sec or 0.0)
            e2e_val = float(e2e_sec or 0.0)
            max_total = max(max_total, init_val + e2e_val)

    for r in results:
        table.add_row(
            r.name,
            _render_init_e2e_bar(r.timing.arts_init_sec,
                                 r.timing.arts_e2e_sec, max_total, width),
            _render_init_e2e_bar(r.timing.omp_init_sec,
                                 r.timing.omp_e2e_sec, max_total, width),
        )

    table.caption = "[dim]Bar colors: init=red, e2e=green[/]"
    return table


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "benchmark"


def _svg_escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace("\"", "&quot;"))


def _bar_segment_palette(n: int) -> List[str]:
    # Tableau-ish distinct colors (good in print, ok for colorblind readers).
    base = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    ]
    if n <= len(base):
        return base[:n]
    out = list(base)
    # Repeat with a slight offset in luminance by alternating a neutral gray.
    while len(out) < n:
        out.append("#777777")
    return out[:n]


def _write_per_benchmark_timeline_svg(
    benchmark: str,
    arts_init: Dict[str, float],
    arts_e2e: Dict[str, float],
    arts_kernel: Dict[str, float],
    omp_init: Dict[str, float],
    omp_e2e: Dict[str, float],
    omp_kernel: Dict[str, float],
    *,
    speedup: Optional[float],
    speedup_basis: str,
    svg_path: Path,
) -> None:
    """Write a per-benchmark SVG with its own local x-scale (paper-friendly)."""
    arts_init_total = sum(arts_init.values()) if arts_init else 0.0
    omp_init_total = sum(omp_init.values()) if omp_init else 0.0
    arts_e2e_total = sum(arts_e2e.values()) if arts_e2e else 0.0
    omp_e2e_total = sum(omp_e2e.values()) if omp_e2e else 0.0
    arts_kernel_total = sum(arts_kernel.values()) if arts_kernel else 0.0
    omp_kernel_total = sum(omp_kernel.values()) if omp_kernel else 0.0
    arts_other_total = max(arts_e2e_total - arts_kernel_total, 0.0)
    omp_other_total = max(omp_e2e_total - omp_kernel_total, 0.0)

    # Segment ordering: preserve observed order in ARTS, then append unseen OMP segments.
    seg_order: List[str] = list(arts_e2e.keys())
    for k in omp_e2e.keys():
        if k not in seg_order:
            seg_order.append(k)

    # Total used for scaling (per runtime bars)
    arts_total = arts_init_total + arts_e2e_total
    omp_total = omp_init_total + omp_e2e_total
    max_total = max(arts_total, omp_total, 1e-12)

    # Layout
    pad = 18
    label_w = 220
    bar_w = 640
    right_w = 260
    w = pad + label_w + bar_w + right_w + pad
    title_h = 46
    row_h = 34
    bar_h = 14
    h = title_h + row_h * 4 + 78

    bar_x = pad + label_w
    arts_y = title_h + 10
    omp_y = arts_y + row_h
    arts_b_y = omp_y + row_h + 12
    omp_b_y = arts_b_y + row_h
    axis_y = omp_b_y + 28

    def xscale(v: float) -> float:
        return bar_w * (v / max_total)

    init_color = "#D62728"
    other_color = "#9AA0A6"
    kernel_color = "#1F77B4"
    e2e_colors = _bar_segment_palette(max(1, len(seg_order)))
    seg_color = {name: e2e_colors[i] for i, name in enumerate(seg_order)}

    title = _svg_escape(benchmark)
    lines: List[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    lines.append("<style>")
    lines.append(
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; }')
    lines.append(
        '.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }')
    lines.append(".dim { fill: #555; }")
    lines.append(".light { fill: #888; }")
    lines.append("</style>")

    # Title
    sp = ""
    if speedup is not None and speedup > 0:
        sp = f"  |  speedup={speedup:.2f}x ({_svg_escape(speedup_basis)})"
    lines.append(f'<text x="{pad}" y="{22}" font-size="16">{title}{sp}</text>')
    lines.append(
        f'<text x="{pad}" y="{40}" class="dim mono">local x-scale: 0–{max_total:.6f}s</text>')

    # Legend (init + kernel/other + e2e segments)
    legend_x = pad
    legend_y = h - 22
    lines.append(
        f'<rect x="{legend_x}" y="{legend_y-10}" width="12" height="8" fill="{init_color}" />')
    lines.append(
        f'<text x="{legend_x+18}" y="{legend_y-3}" class="dim mono">init</text>')
    lines.append(
        f'<rect x="{legend_x+70}" y="{legend_y-10}" width="12" height="8" fill="{kernel_color}" />')
    lines.append(
        f'<text x="{legend_x+88}" y="{legend_y-3}" class="dim mono">kernel</text>')
    lines.append(
        f'<rect x="{legend_x+160}" y="{legend_y-10}" width="12" height="8" fill="{other_color}" />')
    lines.append(
        f'<text x="{legend_x+178}" y="{legend_y-3}" class="dim mono">other(e2e-kernel)</text>')

    lx = legend_x + 360
    for i, seg in enumerate(seg_order[:6]):  # keep legend compact
        c = seg_color[seg]
        lines.append(
            f'<rect x="{lx}" y="{legend_y-10}" width="12" height="8" fill="{c}" />')
        lines.append(
            f'<text x="{lx+18}" y="{legend_y-3}" class="dim mono">{_svg_escape(seg)}</text>')
        lx += 160
        if lx > w - 240:
            break

    # Axis ticks (independent scale)
    lines.append(
        f'<line x1="{bar_x}" y1="{axis_y}" x2="{bar_x+bar_w}" y2="{axis_y}" stroke="#bbb" stroke-width="1" />')
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = bar_x + bar_w * t
        lines.append(
            f'<line x1="{x}" y1="{axis_y-4}" x2="{x}" y2="{axis_y+4}" stroke="#bbb" stroke-width="1" />')
        lines.append(
            f'<text x="{x}" y="{axis_y+18}" text-anchor="middle" class="light mono">{(max_total*t):.3f}</text>')

    def draw_runtime_row(y: int, label: str, init_map: Dict[str, float], e2e_map: Dict[str, float]) -> None:
        init_v = sum(init_map.values()) if init_map else 0.0
        e2e_total = sum(e2e_map.values()) if e2e_map else 0.0
        total_v = init_v + e2e_total

        lines.append(
            f'<text x="{pad}" y="{y+12}" class="mono">{_svg_escape(label)}</text>')
        # Background bar
        lines.append(
            f'<rect x="{bar_x}" y="{y}" width="{bar_w}" height="{bar_h}" fill="#f1f1f1" />')

        x0 = bar_x
        if init_v > 0:
            iw = xscale(init_v)
            lines.append(
                f'<rect x="{x0}" y="{y}" width="{iw:.2f}" height="{bar_h}" fill="{init_color}"><title>init: {init_v:.6f}s</title></rect>')
            x0 += iw

        for seg in seg_order:
            v = float(e2e_map.get(seg, 0.0))
            if v <= 0:
                continue
            sw = xscale(v)
            c = seg_color[seg]
            pct = (v / total_v * 100.0) if total_v > 0 else 0.0
            title_txt = f"{seg}: {v:.6f}s ({pct:.1f}%)"
            lines.append(
                f'<rect x="{x0:.2f}" y="{y}" width="{sw:.2f}" height="{bar_h}" fill="{c}"><title>{_svg_escape(title_txt)}</title></rect>')
            # Inline label only if it fits well (reduce clutter)
            if sw >= 70:
                lines.append(
                    f'<text x="{x0 + sw/2:.2f}" y="{y+11}" text-anchor="middle" fill="#000" class="mono">{v:.3f}s</text>')
            x0 += sw

        # Totals on the right
        tx = bar_x + bar_w + 10
        lines.append(
            f'<text x="{tx}" y="{y+12}" class="dim mono">total {total_v:.6f}s</text>')
        lines.append(
            f'<text x="{tx}" y="{y+26}" class="light mono">init {init_v:.6f}s + e2e {e2e_total:.6f}s</text>')

    draw_runtime_row(arts_y, "ARTS", arts_init, arts_e2e)
    draw_runtime_row(omp_y, "OMP", omp_init, omp_e2e)

    # Breakdown rows: init + other(e2e-kernel) + kernel (consistent across benchmarks).
    def draw_breakdown_row(y: int, label: str, init_v: float, other_v: float, kernel_v: float) -> None:
        total_v = init_v + other_v + kernel_v
        lines.append(
            f'<text x="{pad}" y="{y+12}" class="mono">{_svg_escape(label)}</text>')
        lines.append(
            f'<rect x="{bar_x}" y="{y}" width="{bar_w}" height="{bar_h}" fill="#f1f1f1" />')

        x0 = bar_x
        if init_v > 0:
            iw = xscale(init_v)
            lines.append(
                f'<rect x="{x0}" y="{y}" width="{iw:.2f}" height="{bar_h}" fill="{init_color}"><title>init: {init_v:.6f}s</title></rect>')
            x0 += iw
        if other_v > 0:
            ow = xscale(other_v)
            lines.append(
                f'<rect x="{x0:.2f}" y="{y}" width="{ow:.2f}" height="{bar_h}" fill="{other_color}"><title>other (e2e-kernel): {other_v:.6f}s</title></rect>')
            x0 += ow
        if kernel_v > 0:
            kw = xscale(kernel_v)
            lines.append(
                f'<rect x="{x0:.2f}" y="{y}" width="{kw:.2f}" height="{bar_h}" fill="{kernel_color}"><title>kernel: {kernel_v:.6f}s</title></rect>')
            x0 += kw

        tx = bar_x + bar_w + 10
        lines.append(
            f'<text x="{tx}" y="{y+12}" class="dim mono">total {total_v:.6f}s</text>')
        lines.append(
            f'<text x="{tx}" y="{y+26}" class="light mono">init {init_v:.6f}s | other {other_v:.6f}s | kernel {kernel_v:.6f}s</text>')

    draw_breakdown_row(arts_b_y, "ARTS breakdown",
                       arts_init_total, arts_other_total, arts_kernel_total)
    draw_breakdown_row(omp_b_y, "OMP breakdown", omp_init_total,
                       omp_other_total, omp_kernel_total)

    lines.append("</svg>")
    svg_path.write_text("\n".join(lines))


def _write_small_multiples_svg(groups: List[Dict[str, Any]], svg_path: Path) -> None:
    """Write a multi-panel SVG: each benchmark panel has its own local x-scale."""
    if not groups:
        svg_path.write_text("")
        return

    pad = 18
    panel_h = 120
    w = 1200
    h = 60 + panel_h * len(groups) + 30
    lines: List[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    lines.append("<style>")
    lines.append(
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; }')
    lines.append(
        '.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }')
    lines.append(".dim { fill: #555; }")
    lines.append(".light { fill: #888; }")
    lines.append("</style>")
    lines.append(
        f'<text x="{pad}" y="24" font-size="16">Init + E2E timeline (per-benchmark local scale)</text>')
    lines.append(
        f'<text x="{pad}" y="44" class="dim mono">Note: each panel has an independent x-axis scale for readability.</text>')

    y0 = 60
    for i, g in enumerate(groups):
        y = y0 + i * panel_h
        # Reuse per-benchmark writer by inlining a simplified rendering into a group <g>
        bench = str(g["benchmark"])
        sp = g.get("speedup")
        sp_basis = str(g.get("speedup_basis", "total"))
        scale_max = float(g["scale_max"])

        label_w = 240
        bar_w = 720
        right_w = 200
        bar_x = pad + label_w
        arts_y = y + 30
        omp_y = y + 64
        axis_y = y + 92

        def xscale(v: float) -> float:
            return bar_w * (v / max(scale_max, 1e-12))

        # Panel background
        lines.append(
            f'<rect x="{pad}" y="{y}" width="{w-2*pad}" height="{panel_h-6}" fill="#fafafa" stroke="#eee" />')

        sp_txt = ""
        if sp is not None and float(sp) > 0:
            sp_txt = f"  speedup={float(sp):.2f}x ({_svg_escape(sp_basis)})"
        lines.append(
            f'<text x="{pad+6}" y="{y+18}" class="mono">{_svg_escape(bench)}{sp_txt}</text>')
        lines.append(
            f'<text x="{pad+6}" y="{y+34}" class="light mono">scale: 0–{scale_max:.4f}s</text>')

        # Axis
        lines.append(
            f'<line x1="{bar_x}" y1="{axis_y}" x2="{bar_x+bar_w}" y2="{axis_y}" stroke="#bbb" stroke-width="1" />')
        for t in [0.0, 0.5, 1.0]:
            x = bar_x + bar_w * t
            lines.append(
                f'<line x1="{x}" y1="{axis_y-4}" x2="{x}" y2="{axis_y+4}" stroke="#bbb" stroke-width="1" />')
            lines.append(
                f'<text x="{x}" y="{axis_y+18}" text-anchor="middle" class="light mono">{(scale_max*t):.3f}</text>')

        # Draw rows using consistent segments (init + other + kernel)
        init_color = "#D62728"
        other_color = "#9AA0A6"
        kernel_color = "#1F77B4"

        def draw_row(y_row: int, label: str, segments: List[Tuple[str, float]], total: float) -> None:
            lines.append(
                f'<text x="{pad+6}" y="{y_row+12}" class="mono">{_svg_escape(label)}</text>')
            lines.append(
                f'<rect x="{bar_x}" y="{y_row}" width="{bar_w}" height="14" fill="#f1f1f1" />')
            xcur = bar_x
            for name, val in segments:
                if val <= 0:
                    continue
                sw = xscale(val)
                if name == "__init__":
                    c = init_color
                    title_txt = "init"
                elif name == "__kernel__":
                    c = kernel_color
                    title_txt = "kernel"
                else:
                    c = other_color
                    title_txt = "other (e2e-kernel)"
                pct = (val / total * 100.0) if total > 0 else 0.0
                lines.append(
                    f'<rect x="{xcur:.2f}" y="{y_row}" width="{sw:.2f}" height="14" fill="{c}"><title>{_svg_escape(title_txt)}: {val:.6f}s ({pct:.1f}%)</title></rect>')
                xcur += sw
            lines.append(
                f'<text x="{bar_x+bar_w+10}" y="{y_row+12}" class="dim mono">{total:.4f}s</text>')

        draw_row(arts_y, "ARTS", g["arts_segments"], g["arts_total"])
        draw_row(omp_y, "OMP", g["omp_segments"], g["omp_total"])

    lines.append("</svg>")
    svg_path.write_text("\n".join(lines))


def _write_speedup_bar_chart(rows: List[Dict[str, Any]], svg_path: Path) -> None:
    """Write a single bar chart with all benchmarks/configs."""
    valid = [
        r
        for r in rows
        if r.get("speedup_mean") is not None and float(r["speedup_mean"]) > 0
    ]
    if not valid:
        svg_path.write_text("")
        return

    width = 1200
    bar_height = 30
    gap = 12
    pad = 150
    height = 60 + len(valid) * (bar_height + gap)

    sorted_rows = sorted(valid, key=lambda r: float(
        r["speedup_mean"]), reverse=True)
    max_speedup = max(float(r["speedup_mean"]) for r in sorted_rows)
    max_speedup = max(max_speedup, 1.2)

    def x_pos(value: float) -> float:
        return pad + (width - pad - 80) * min(value / max_speedup, 1.0)

    lines: List[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append("<style>")
    lines.append(
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px; }'
    )
    lines.append(
        '.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }'
    )
    lines.append(".dim { fill: #555; }")
    lines.append("</style>")
    lines.append(
        f'<text x="{pad}" y="24" font-size="16">Speedup (OMP / ARTS) per benchmark/configuration</text>')
    lines.append(
        f'<text x="{pad}" y="42" class="dim mono">Bars > 1.0x → CARTS faster; < 1.0x → OpenMP wins. Multi-run mean ± std shown.</text>')
    lines.append(
        f'<line x1="{pad}" y1="48" x2="{width-60}" y2="48" stroke="#ddd" stroke-width="1" />')

    for idx, r in enumerate(sorted_rows):
        y = 60 + idx * (bar_height + gap)
        label = f"{r['benchmark']} (t={r['threads']}, n={r['nodes']}, omp={r['omp_threads']})"
        sp = float(r["speedup_mean"])
        sp_std = r.get("speedup_std") or 0.0
        color = "#2CA02C" if sp >= 1.0 else "#D62728"
        lines.append(
            f'<text x="8" y="{y + bar_height/2 + 4:.1f}" class="mono" text-anchor="start">{_svg_escape(label)}</text>')
        bar_start = pad
        bar_end = x_pos(sp)
        lines.append(
            f'<rect x="{bar_start}" y="{y}" width="{max(bar_end - bar_start, 1):.2f}" height="{bar_height}" fill="{color}" opacity="0.9">'
        )
        lines.append("</rect>")
        if sp_std > 0:
            err_start = x_pos(max(sp - float(sp_std), 0.0))
            err_end = x_pos(sp + float(sp_std))
            lines.append(
                f'<line x1="{err_start:.2f}" y1="{y + bar_height/2:.2f}" x2="{err_end:.2f}" y2="{y + bar_height/2:.2f}" stroke="{color}" stroke-width="1.2" />'
            )
        lines.append(
            f'<text x="{bar_end + 8:.2f}" y="{y + bar_height/2 + 4:.1f}" class="dim mono">{sp:.2f}x ± {float(sp_std):.2f}x</text>'
        )

    # x-axis
    axis_y = height - 16
    lines.append(
        f'<line x1="{pad}" y1="{axis_y}" x2="{width-80}" y2="{axis_y}" stroke="#bbb" stroke-width="1" />')
    for t in [0.2, 0.5, 1.0, 2.0, max_speedup]:
        if t > max_speedup:
            continue
        x = x_pos(t)
        lines.append(
            f'<line x1="{x:.2f}" y1="{axis_y-4}" x2="{x:.2f}" y2="{axis_y+4}" stroke="#bbb" stroke-width="1" />')
        lines.append(
            f'<text x="{x:.2f}" y="{axis_y+16}" text-anchor="middle" class="light mono">{t:.2f}x</text>')
    if max_speedup >= 1.0:
        x1 = x_pos(1.0)
        lines.append(
            f'<line x1="{x1:.2f}" y1="48" x2="{x1:.2f}" y2="{axis_y}" stroke="#999" stroke-dasharray="4,3" stroke-width="1" />')
        lines.append(f'<text x="{x1:.2f}" y="60" class="dim mono">1.0x</text>')

    lines.append("</svg>")
    svg_path.write_text("\n".join(lines))


def generate_markdown_report(
    results: List[BenchmarkResult],
    report_md_path: Path,
    chart_svg_path: Path,
    *,
    size: str,
) -> Tuple[Path, Path]:
    """Generate a Markdown report + companion SVG charts (paper-friendly)."""
    now = datetime.now().isoformat(timespec="seconds")
    figures_dir = report_md_path.parent / f"{report_md_path.stem}_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    speedup_svg_path = report_md_path.parent / \
        f"{report_md_path.stem}_speedup_bar_chart.svg"

    # Build per-benchmark groups (config-aware) and per-benchmark SVGs.
    groups: Dict[Tuple, List[BenchmarkResult]] = {}
    for r in results:
        key = (r.name, r.config.arts_threads, r.config.arts_nodes,
               r.config.omp_threads, r.config.launcher)
        groups.setdefault(key, []).append(r)

    panel_groups: List[Dict[str, Any]] = []
    per_svgs: List[Tuple[Tuple, Path]] = []

    def _mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
        if not vals:
            return None, None
        if len(vals) == 1:
            return vals[0], 0.0
        from statistics import mean, stdev
        return mean(vals), stdev(vals)

    def _fmt_ms(v: Optional[float]) -> str:
        if v is None:
            return "-"
        return f"{v * 1e3:.3f}"

    def _fmt_s(v: Optional[float]) -> str:
        if v is None:
            return "-"
        return f"{v:.6f}"

    def _fmt_mean_std_s(mean_v: Optional[float], std_v: Optional[float]) -> str:
        if mean_v is None:
            return "-"
        if std_v is None:
            return f"{mean_v:.6f}"
        return f"{mean_v:.6f} ± {std_v:.6f}"

    def _kv_lines(d: Dict[str, float]) -> str:
        if not d:
            return "-"
        parts = [f"`{k}`={v:.6f}s" for k, v in sorted(d.items())]
        return ", ".join(parts)

    report_rows: List[Dict[str, Any]] = []

    for key, runs in sorted(groups.items(), key=lambda kv: kv[0][0]):
        r0 = runs[0]
        bench = r0.name

        # Aggregate timing stats across runs (mean±std) for paper reporting.
        arts_init_vals = [float(r.run_arts.init_timings.get(
            "arts")) for r in runs if r.run_arts.init_timings.get("arts") is not None]
        omp_init_vals = [float(r.run_omp.init_timings.get("omp"))
                         for r in runs if r.run_omp.init_timings.get("omp") is not None]
        arts_e2e_vals = [sum(r.run_arts.e2e_timings.values())
                         for r in runs if r.run_arts.e2e_timings]
        omp_e2e_vals = [sum(r.run_omp.e2e_timings.values())
                        for r in runs if r.run_omp.e2e_timings]
        arts_kernel_vals = [sum(r.run_arts.kernel_timings.values())
                            for r in runs if r.run_arts.kernel_timings]
        omp_kernel_vals = [sum(r.run_omp.kernel_timings.values())
                           for r in runs if r.run_omp.kernel_timings]

        arts_init_mean, arts_init_std = _mean_std(arts_init_vals)
        omp_init_mean, omp_init_std = _mean_std(omp_init_vals)
        arts_e2e_mean, arts_e2e_std = _mean_std(arts_e2e_vals)
        omp_e2e_mean, omp_e2e_std = _mean_std(omp_e2e_vals)
        arts_kernel_mean, arts_kernel_std = _mean_std(arts_kernel_vals)
        omp_kernel_mean, omp_kernel_std = _mean_std(omp_kernel_vals)

        arts_other_mean = None if arts_e2e_mean is None or arts_kernel_mean is None else max(
            arts_e2e_mean - arts_kernel_mean, 0.0)
        omp_other_mean = None if omp_e2e_mean is None or omp_kernel_mean is None else max(
            omp_e2e_mean - omp_kernel_mean, 0.0)

        arts_total_mean = None if arts_init_mean is None or arts_e2e_mean is None else (
            arts_init_mean + arts_e2e_mean)
        omp_total_mean = None if omp_init_mean is None or omp_e2e_mean is None else (
            omp_init_mean + omp_e2e_mean)
        scale_max = max(float(arts_total_mean or 0.0),
                        float(omp_total_mean or 0.0), 1e-12)

        # Use segment maps from the first run for labeling. (If multiple runs, segment keys should match.)
        arts_init_map = dict(r0.run_arts.init_timings)
        omp_init_map = dict(r0.run_omp.init_timings)
        arts_e2e_map = dict(r0.run_arts.e2e_timings)
        omp_e2e_map = dict(r0.run_omp.e2e_timings)
        arts_kernel_map = dict(r0.run_arts.kernel_timings)
        omp_kernel_map = dict(r0.run_omp.kernel_timings)

        # Per-benchmark SVG (local scale, detailed tooltips).
        svg_name = f"{_sanitize_filename(bench)}_{r0.config.arts_threads}t_{r0.config.arts_nodes}n.svg"
        svg_path = figures_dir / svg_name
        _write_per_benchmark_timeline_svg(
            bench,
            arts_init_map,
            arts_e2e_map,
            arts_kernel_map,
            omp_init_map,
            omp_e2e_map,
            omp_kernel_map,
            speedup=(r0.timing.speedup if r0.timing.speedup > 0 else None),
            speedup_basis=r0.timing.speedup_basis,
            svg_path=svg_path,
        )
        per_svgs.append((key, svg_path))

        # Multi-panel summary data (init + other(e2e-kernel) + kernel): consistent meaning across benchmarks.
        arts_segments: List[Tuple[str, float]] = [
            ("__init__", float(arts_init_mean or 0.0)),
            ("__other__", float(arts_other_mean or 0.0)),
            ("__kernel__", float(arts_kernel_mean or 0.0)),
        ]
        omp_segments: List[Tuple[str, float]] = [
            ("__init__", float(omp_init_mean or 0.0)),
            ("__other__", float(omp_other_mean or 0.0)),
            ("__kernel__", float(omp_kernel_mean or 0.0)),
        ]

        panel_groups.append({
            "benchmark": bench,
            "speedup": (r0.timing.speedup if r0.timing.speedup > 0 else None),
            "speedup_basis": r0.timing.speedup_basis,
            "scale_max": scale_max,
            "arts_segments": arts_segments,
            "omp_segments": omp_segments,
            "arts_total": sum(v for _, v in arts_segments),
            "omp_total": sum(v for _, v in omp_segments),
        })

        # Report row (mean±std)
        run_count = len(runs)
        correct_count = sum(1 for r in runs if r.verification.correct)
        arts_pass = sum(1 for r in runs if r.run_arts.status == Status.PASS)
        omp_pass = sum(1 for r in runs if r.run_omp.status == Status.PASS)
        speedup_vals = [r.timing.speedup for r in runs if r.timing.speedup > 0]
        sp_mean, sp_std = _mean_std(speedup_vals)

        report_rows.append({
            "key": key,
            "benchmark": bench,
            "size_params": r0.size_params,
            "threads": r0.config.arts_threads,
            "nodes": r0.config.arts_nodes,
            "omp_threads": r0.config.omp_threads,
            "launcher": r0.config.launcher,
            "run_count": run_count,
            "arts_pass": arts_pass,
            "omp_pass": omp_pass,
            "correct_count": correct_count,
            "speedup_basis": r0.timing.speedup_basis,
            "speedup_mean": sp_mean,
            "speedup_std": sp_std,
            "arts_init_mean": arts_init_mean,
            "arts_init_std": arts_init_std,
            "arts_e2e_mean": arts_e2e_mean,
            "arts_e2e_std": arts_e2e_std,
            "arts_kernel_mean": arts_kernel_mean,
            "arts_kernel_std": arts_kernel_std,
            "arts_other_mean": arts_other_mean,
            "omp_init_mean": omp_init_mean,
            "omp_init_std": omp_init_std,
            "omp_e2e_mean": omp_e2e_mean,
            "omp_e2e_std": omp_e2e_std,
            "omp_kernel_mean": omp_kernel_mean,
            "omp_kernel_std": omp_kernel_std,
            "omp_other_mean": omp_other_mean,
            # First-run raw breakdowns (for segment listing)
            "arts_init_map": arts_init_map,
            "arts_e2e_map": arts_e2e_map,
            "arts_kernel_map": arts_kernel_map,
            "omp_init_map": omp_init_map,
            "omp_e2e_map": omp_e2e_map,
            "omp_kernel_map": omp_kernel_map,
            "arts_status": r0.run_arts.status.value,
            "omp_status": r0.run_omp.status.value,
            "correct_note": r0.verification.note,
        })

    # Write the summary small-multiples SVG (each panel has its own scale).
    _write_small_multiples_svg(panel_groups, chart_svg_path)

    # Write a speedup scaling SVG (each benchmark has its own y-axis range).
    _write_speedup_bar_chart(report_rows, speedup_svg_path)

    # Markdown
    lines: List[str] = []
    lines.append("# CARTS Benchmarks Report")
    lines.append("")
    lines.append(f"- Timestamp: {now}")
    lines.append(f"- Size: {size}")
    lines.append(f"- Config groups: {len(groups)}")
    lines.append(f"- Figures: `{figures_dir.name}/`")
    lines.append("")
    lines.append("## Summary Figure (local scale per benchmark)")
    lines.append("")
    lines.append(f"![init+e2e small multiples]({chart_svg_path.name})")
    lines.append("")
    lines.append("## Speedup Figure (local scale per benchmark)")
    lines.append("")
    lines.append(f"![speedup small multiples]({speedup_svg_path.name})")
    lines.append("")
    lines.append("## Results (mean ± std, seconds)")
    lines.append("")
    lines.append("| Benchmark | t | n | ARTS status | OMP status | ARTS init | ARTS e2e | ARTS kernel | ARTS other | OMP init | OMP e2e | OMP kernel | OMP other | Speedup | Correct |")
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for row in report_rows:
        speedup_txt = "-"
        if row["speedup_mean"] is not None:
            speedup_txt = _fmt_mean_std_s(
                row["speedup_mean"], row["speedup_std"]) + f" ({row['speedup_basis']})"

        correct_txt = "-"
        if row["correct_note"] == "Verification disabled":
            correct_txt = "N/A"
        else:
            correct_txt = f"{row['correct_count']}/{row['run_count']}"

        lines.append(
            "| "
            + f"{row['benchmark']} "
            + f"| {row['threads']} "
            + f"| {row['nodes']} "
            + f"| {row['arts_status']} "
            + f"| {row['omp_status']} "
            + f"| {_fmt_mean_std_s(row['arts_init_mean'], row['arts_init_std'])} "
            + f"| {_fmt_mean_std_s(row['arts_e2e_mean'], row['arts_e2e_std'])} "
            + f"| {_fmt_mean_std_s(row['arts_kernel_mean'], row['arts_kernel_std'])} "
            + f"| {_fmt_s(row['arts_other_mean'])} "
            + f"| {_fmt_mean_std_s(row['omp_init_mean'], row['omp_init_std'])} "
            + f"| {_fmt_mean_std_s(row['omp_e2e_mean'], row['omp_e2e_std'])} "
            + f"| {_fmt_mean_std_s(row['omp_kernel_mean'], row['omp_kernel_std'])} "
            + f"| {_fmt_s(row['omp_other_mean'])} "
            + f"| {speedup_txt} "
            + f"| {correct_txt} |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- `other = e2e - kernel` (clamped at 0).")
    lines.append(
        "- Figures use independent x-axis scaling per benchmark to preserve within-benchmark structure.")
    lines.append("")
    lines.append("## Missing Signals")
    lines.append("")
    lines.append("| Benchmark | Missing | Likely cause |")
    lines.append("|---|---|---|")
    any_missing = False
    for row in report_rows:
        missing: List[str] = []
        if not row["arts_init_map"].get("arts"):
            missing.append("ARTS init.arts")
        if not row["arts_e2e_map"]:
            missing.append("ARTS e2e.*")
        if not row["omp_init_map"].get("omp"):
            missing.append("OMP init.omp")
        if not row["omp_e2e_map"]:
            missing.append("OMP e2e.*")
        if not missing:
            continue
        any_missing = True

        cause = []
        if row["arts_status"] != Status.PASS.value:
            cause.append("ARTS run failed/crashed before printing timers")
        if row["omp_status"] != Status.PASS.value:
            cause.append("OMP run failed/crashed before printing timers")
        if not cause:
            cause.append("timing macros not enabled or benchmark not rebuilt")

        lines.append(
            f"| {row['benchmark']} | {', '.join(missing)} | {', '.join(cause)} |")
    if not any_missing:
        lines.append("| (none) | - | - |")
    lines.append("")
    lines.append("## Per-Benchmark Figures")
    lines.append("")
    per_svg_lookup = {k: p for k, p in per_svgs}
    for row in report_rows:
        key = row["key"]
        svg_path = per_svg_lookup.get(key)
        bench = row["benchmark"]
        lines.append(f"### {bench}")
        lines.append("")
        if svg_path is not None:
            lines.append(f"![{bench}]({figures_dir.name}/{svg_path.name})")
            lines.append("")
        lines.append(
            f"- Config: `threads={row['threads']}`, `nodes={row['nodes']}`, `omp_threads={row['omp_threads']}`, `launcher={row['launcher']}`, `runs={row['run_count']}`")
        lines.append(f"- Size params: `{row['size_params'] or 'N/A'}`")
        lines.append(
            f"- Status: `ARTS={row['arts_status']}`, `OMP={row['omp_status']}`")
        lines.append(
            f"- ARTS init breakdown: {_kv_lines(row['arts_init_map'])}")
        lines.append(f"- ARTS e2e breakdown: {_kv_lines(row['arts_e2e_map'])}")
        lines.append(
            f"- ARTS kernel breakdown: {_kv_lines(row['arts_kernel_map'])}")
        lines.append(f"- OMP init breakdown: {_kv_lines(row['omp_init_map'])}")
        lines.append(f"- OMP e2e breakdown: {_kv_lines(row['omp_e2e_map'])}")
        lines.append(
            f"- OMP kernel breakdown: {_kv_lines(row['omp_kernel_map'])}")
        lines.append("")

    report_md_path.write_text("\n".join(lines))
    return chart_svg_path, speedup_svg_path


def create_live_table(
    benchmarks: List[str],
    results: Dict[str, List[BenchmarkResult]],
    in_progress: Optional[str] = None,
    current_phase: Optional[Phase] = None,
    current_partial: Optional[Dict[str, Any]] = None,
    total_runs: int = 1,
) -> Table:
    """Create a live-updating table showing benchmark progress with running statistics."""
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")

    table.add_column("Benchmark", style="cyan", no_wrap=True)
    table.add_column("CARTS Build", justify="center")
    table.add_column("ARTS Init", justify="right")
    table.add_column("ARTS E2E", justify="right")
    table.add_column("OMP Init", justify="right")
    table.add_column("OMP E2E", justify="right")
    table.add_column("Correct", justify="center")
    table.add_column("Speedup", justify="right")

    has_fallback = False
    has_multi_kernel = False
    for bench in benchmarks:
        if bench in results and results[bench]:
            # Completed runs - show running statistics
            runs_list = results[bench]
            run_count = len(runs_list)
            r = runs_list[-1]  # Latest result for status checks

            # CARTS Build time (only ARTS build, with statistics)
            arts_build_times = [run.build_arts.duration_sec for run in runs_list
                                if run.build_arts.status == Status.PASS]
            if arts_build_times:
                stats = compute_stats(arts_build_times)
                stddev = stats.get('stddev', 0.0)
                build = f"[green]\u2713[/] {stats['mean']:.1f}s ({stddev:.2f}s) [{run_count}]"
            else:
                build = f"[red]\u2717[/] {r.build_arts.status.value}"

            # ARTS Init time with statistics
            arts_init_times = [run.run_arts.counter_init_sec for run in runs_list
                               if run.run_arts.counter_init_sec is not None]
            if arts_init_times:
                stats = compute_stats(arts_init_times)
                stddev = stats.get('stddev', 0.0)
                arts_init = f"{stats['mean']:.4f}s ({stddev:.4f}s) [{run_count}]"
            else:
                arts_init = "[dim]-[/]"

            # ARTS E2E time with statistics
            arts_e2e_times = []
            for run in runs_list:
                if run.run_arts.counter_e2e_sec is not None:
                    arts_e2e_times.append(run.run_arts.counter_e2e_sec)
                elif run.run_arts.e2e_timings:
                    arts_e2e_times.append(sum(run.run_arts.e2e_timings.values()))
            if arts_e2e_times:
                stats = compute_stats(arts_e2e_times)
                stddev = stats.get('stddev', 0.0)
                run_arts = f"[green]\u2713[/] {stats['mean']:.4f}s ({stddev:.4f}s) [{run_count}]"
            elif r.run_arts.status == Status.PASS:
                # Fallback to kernel times
                arts_kernel, arts_kernel_str = format_kernel_time(r.run_arts)
                if arts_kernel is not None:
                    run_arts = f"[green]\u2713[/] {arts_kernel_str}*"
                else:
                    run_arts = f"[green]\u2713[/] {r.run_arts.duration_sec:.2f}s*"
                has_fallback = True
            else:
                run_arts = f"{status_symbol(r.run_arts.status)} {r.run_arts.status.value}"

            # OMP Init time with statistics
            omp_init_times = [run.run_omp.init_timings.get("omp") for run in runs_list
                              if run.run_omp.init_timings.get("omp") is not None]
            if omp_init_times:
                stats = compute_stats(omp_init_times)
                stddev = stats.get('stddev', 0.0)
                omp_init_str = f"{stats['mean']:.4f}s ({stddev:.4f}s) [{run_count}]"
            else:
                omp_init_str = "[dim]-[/]"

            # OMP E2E time with statistics
            omp_e2e_times = []
            for run in runs_list:
                if run.run_omp.e2e_timings:
                    omp_e2e_times.append(sum(run.run_omp.e2e_timings.values()))
            if omp_e2e_times:
                stats = compute_stats(omp_e2e_times)
                stddev = stats.get('stddev', 0.0)
                run_omp = f"[green]\u2713[/] {stats['mean']:.4f}s ({stddev:.4f}s) [{run_count}]"
            elif r.run_omp.status == Status.PASS:
                # Fallback to kernel times
                omp_kernel, omp_kernel_str = format_kernel_time(r.run_omp)
                if omp_kernel is not None:
                    run_omp = f"[green]\u2713[/] {omp_kernel_str}*"
                else:
                    run_omp = f"[green]\u2713[/] {r.run_omp.duration_sec:.2f}s*"
                has_fallback = True
            else:
                run_omp = f"{status_symbol(r.run_omp.status)} {r.run_omp.status.value}"

            # Correctness (based on latest run)
            if r.verification.correct:
                correct = "[green]\u2713 YES[/]"
            elif r.verification.note == "Verification disabled":
                correct = "[dim]- N/A[/]"
            elif r.run_arts.status != Status.PASS or r.run_omp.status != Status.PASS:
                correct = "[dim]- N/A[/]"
            else:
                correct = "[red]\u2717 NO[/]"

            # Speedup with statistics
            speedups = [run.timing.speedup for run in runs_list if run.timing.speedup > 0]
            if speedups:
                stats = compute_stats(speedups)
                mean_speedup = stats['mean']
                stddev = stats.get('stddev', 0.0)
                if mean_speedup >= 1.0:
                    speedup = f"[green]{mean_speedup:.2f}x ({stddev:.2f}) [{run_count}][/]"
                elif mean_speedup >= 0.8:
                    speedup = f"[yellow]{mean_speedup:.2f}x ({stddev:.2f}) [{run_count}][/]"
                else:
                    speedup = f"[red]{mean_speedup:.2f}x ({stddev:.2f}) [{run_count}][/]"
                if r.timing.speedup_basis != "e2e":
                    speedup = speedup.replace(f"[{run_count}]", f"[{run_count}]*")
                    has_fallback = True
            else:
                speedup = "[dim]-[/]"

            # Track if any benchmark has multiple kernels
            if r.run_arts.kernel_timings and len(r.run_arts.kernel_timings) > 1:
                has_multi_kernel = True

            table.add_row(bench, build, arts_init, run_arts, omp_init_str, run_omp, correct, speedup)

        elif bench == in_progress:
            # Currently running - show phase-specific indicator
            arts_init = "[dim]-[/]"  # Default for in-progress phases
            omp_init_str = "[dim]-[/]"  # Default for in-progress phases
            if current_phase == Phase.BUILD_ARTS:
                build = "[yellow]⏳ ARTS...[/]"
                run_arts = "[dim]-[/]"
                run_omp = "[dim]-[/]"
                correct = "[dim]-[/]"
                speedup = "[dim]-[/]"
            elif current_phase == Phase.BUILD_OMP:
                # Show ARTS build time if available (CARTS Build = ARTS only)
                if current_partial and "build_arts" in current_partial:
                    build_arts_result = current_partial["build_arts"]
                    if build_arts_result.status == Status.PASS:
                        build = f"[green]✓[/] {build_arts_result.duration_sec:.1f}s (0.00s) [1]"
                    else:
                        build = f"[red]✗[/] {build_arts_result.status.value}"
                else:
                    build = "[yellow]⏳ building...[/]"
                run_arts = "[dim]-[/]"
                run_omp = "[dim]-[/]"
                correct = "[dim]-[/]"
                speedup = "[dim]-[/]"
            elif current_phase == Phase.RUN_ARTS:
                # Show CARTS build time (ARTS only)
                if current_partial and "build_arts" in current_partial:
                    build_arts_result = current_partial["build_arts"]
                    if build_arts_result.status == Status.PASS:
                        build = f"[green]✓[/] {build_arts_result.duration_sec:.1f}s (0.00s) [1]"
                    else:
                        build = f"[red]✗[/] {build_arts_result.status.value}"
                else:
                    build = "[green]✓[/]"
                run_arts = "[yellow]⏳ running...[/]"
                run_omp = "[dim]-[/]"
                correct = "[dim]-[/]"
                speedup = "[dim]-[/]"
            elif current_phase == Phase.RUN_OMP:
                # Show CARTS build time (ARTS only)
                if current_partial and "build_arts" in current_partial:
                    build_arts_result = current_partial["build_arts"]
                    if build_arts_result.status == Status.PASS:
                        build = f"[green]✓[/] {build_arts_result.duration_sec:.1f}s (0.00s) [1]"
                    else:
                        build = f"[red]✗[/] {build_arts_result.status.value}"
                else:
                    build = "[green]✓[/]"
                # ARTS run completed, show e2e time if available in partial results
                if current_partial and "run_arts" in current_partial:
                    run_arts_result = current_partial["run_arts"]
                    # Prefer counter-based timing
                    if run_arts_result.counter_init_sec is not None:
                        arts_init = f"{run_arts_result.counter_init_sec:.4f}s"
                    if run_arts_result.counter_e2e_sec is not None:
                        arts_e2e = run_arts_result.counter_e2e_sec
                        arts_e2e_str = f"{arts_e2e:.4f}s"
                    else:
                        arts_e2e, arts_e2e_str = format_e2e_time(run_arts_result)
                    if run_arts_result.status == Status.PASS:
                        if arts_e2e is not None:
                            run_arts = f"{status_symbol(run_arts_result.status)} {arts_e2e_str}"
                        else:
                            arts_kernel, arts_kernel_str = format_kernel_time(
                                run_arts_result)
                            if arts_kernel is not None:
                                run_arts = f"{status_symbol(run_arts_result.status)} {arts_kernel_str}*"
                            else:
                                run_arts = f"{status_symbol(run_arts_result.status)} {run_arts_result.duration_sec:.2f}s*"
                    else:
                        run_arts = f"{status_symbol(run_arts_result.status)} {run_arts_result.status.value}"
                else:
                    run_arts = "[green]✓[/]"
                run_omp = "[yellow]⏳ running...[/]"
                correct = "[dim]-[/]"
                speedup = "[dim]-[/]"
            else:
                build = "[yellow]⏳...[/]"
                run_arts = "[dim]-[/]"
                run_omp = "[dim]-[/]"
                correct = "[dim]-[/]"
                speedup = "[dim]-[/]"
            table.add_row(
                f"[bold]{bench}[/]",
                build,
                arts_init,
                run_arts,
                omp_init_str,
                run_omp,
                correct,
                speedup,
            )
        else:
            # Pending - show placeholder
            table.add_row(
                f"[dim]{bench}[/]",
                "[dim]-[/]",
                "[dim]-[/]",
                "[dim]-[/]",
                "[dim]-[/]",
                "[dim]-[/]",
                "[dim]-[/]",
                "[dim]-[/]",
            )

    # Build caption based on what notations are used
    captions = []
    if has_multi_kernel:
        captions.append("[N] = sum of N kernels")
    if has_fallback:
        captions.append("* = speedup/time not based on e2e")
    if captions:
        table.caption = "[dim]" + "  |  ".join(captions) + "[/]"

    return table


def create_live_summary(
    results: Dict[str, List[BenchmarkResult]],
    total: int,
    elapsed: float,
) -> Text:
    """Create a one-line summary for live display."""
    # Count passed/failed using latest result from each benchmark
    passed = sum(1 for runs in results.values()
                 if runs and runs[-1].run_arts.status == Status.PASS
                 and runs[-1].verification.correct)
    failed = sum(1 for runs in results.values()
                 if runs and (runs[-1].run_arts.status in (Status.FAIL, Status.CRASH)
                              or (runs[-1].run_arts.status == Status.PASS
                                  and not runs[-1].verification.correct)))
    pending = total - len(results)

    text = Text()
    text.append(f"\u2713 {passed} passed  ", style="green")
    text.append(f"\u2717 {failed} failed  ", style="red")
    text.append(f"\u25cb {pending} pending  ", style="dim")
    text.append(f"\u23f1 {elapsed:.1f}s", style="dim")
    return text


def create_live_display(
    benchmarks: List[str],
    results: Dict[str, List[BenchmarkResult]],
    in_progress: Optional[str],
    elapsed: float,
    current_phase: Optional[Phase] = None,
    current_partial: Optional[Dict[str, Any]] = None,
    total_runs: int = 1,
) -> Group:
    """Create the complete live display (status panel + table + summary)."""
    components = []

    # Add running status panel if benchmark in progress
    if in_progress:
        phase_text = current_phase.value.replace("_", " ").title() if current_phase else "Starting"
        status_text = f"[bold cyan]\u25b6 Running:[/] [white]{in_progress}[/] [dim]({phase_text})[/]"
        status_panel = Panel(status_text, box=box.ROUNDED, style="blue", padding=(0, 1))
        components.append(status_panel)

    table = create_live_table(
        benchmarks, results, in_progress, current_phase, current_partial, total_runs)
    summary = create_live_summary(results, len(benchmarks), elapsed)

    components.extend([table, summary])
    return Group(*components)


# ============================================================================
# JSON Export
# ============================================================================


def get_git_hash(repo_path: Path) -> Optional[str]:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        logger.debug("Failed to get git hash for %s", repo_path, exc_info=True)
    return None


def get_compiler_version() -> Dict[str, Optional[str]]:
    """Get compiler version information.

    Prioritizes CARTS-installed LLVM/clang over system compilers,
    since CARTS builds LLVM from source.
    """
    compilers = {}
    carts_dir = get_carts_dir()

    # Try CARTS-installed LLVM clang first (built from source)
    carts_clang = carts_dir / ".install" / "llvm" / "bin" / "clang"
    clang_paths = [str(carts_clang), "clang"]

    for clang_path in clang_paths:
        try:
            result = subprocess.run(
                [clang_path, "--version"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                first_line = result.stdout.split('\n')[0]
                compilers["clang"] = first_line
                # Record which clang was found
                if clang_path != "clang":
                    compilers["clang_path"] = clang_path
                break
        except Exception:
            logger.debug("Failed to get clang version from %s", clang_path, exc_info=True)
            continue

    # Try gcc
    try:
        result = subprocess.run(
            ["gcc", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            first_line = result.stdout.split('\n')[0]
            compilers["gcc"] = first_line
    except Exception:
        logger.debug("Failed to get gcc version", exc_info=True)

    return compilers


def get_cpu_info() -> Dict[str, Any]:
    """Get CPU information for reproducibility."""
    cpu_info = {}

    system = platform.system().lower()

    if system == "darwin":
        # macOS: use sysctl
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                cpu_info["model"] = result.stdout.strip()

            result = subprocess.run(
                ["sysctl", "-n", "hw.ncpu"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                cpu_info["cores"] = int(result.stdout.strip())

            result = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                cpu_info["physical_cores"] = int(result.stdout.strip())
        except Exception:
            logger.debug("Failed to get macOS CPU info", exc_info=True)
    elif system == "linux":
        # Linux: parse /proc/cpuinfo
        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read()
            for line in content.split('\n'):
                if line.startswith("model name"):
                    cpu_info["model"] = line.split(":")[1].strip()
                    break

            result = subprocess.run(["nproc"], capture_output=True, text=True)
            if result.returncode == 0:
                cpu_info["cores"] = int(result.stdout.strip())
        except Exception:
            logger.debug("Failed to get Linux CPU info", exc_info=True)

    return cpu_info


def get_reproducibility_metadata(carts_dir: Path, benchmarks_dir: Path) -> Dict[str, Any]:
    """Collect comprehensive reproducibility metadata.

    This captures all information needed to reproduce benchmark results:
    - Git commit hashes for all repositories
    - Compiler versions
    - CPU and system information
    - Relevant environment variables
    """
    metadata = {}

    # Git hashes
    metadata["git_commits"] = {
        "carts": get_git_hash(carts_dir) or "unknown",
        "carts_benchmarks": get_git_hash(benchmarks_dir) or "unknown",
    }

    # Check for ARTS submodule
    arts_dir = carts_dir / "external" / "arts"
    if arts_dir.exists():
        metadata["git_commits"]["arts"] = get_git_hash(arts_dir) or "unknown"

    # Compiler versions
    metadata["compilers"] = get_compiler_version()

    # CPU info
    metadata["cpu"] = get_cpu_info()

    # System info
    metadata["system"] = {
        "os": platform.system(),
        "os_version": platform.release(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }

    # Relevant environment variables
    env_vars_to_capture = [
        "OMP_NUM_THREADS",
        "OMP_PROC_BIND",
        "OMP_PLACES",
        "CARTS_DIR",
        "ARTS_DIR",
        "CC",
        "CXX",
        "CFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
    ]
    metadata["environment"] = {
        var: os.environ.get(var) for var in env_vars_to_capture if os.environ.get(var)
    }

    return metadata


def _serialize_parallel_task_timing(timing: Optional[ParallelTaskTiming]) -> Optional[Dict]:
    """Serialize ParallelTaskTiming to JSON-compatible dict."""
    if timing is None:
        return None

    return {
        "parallel_timings": {
            name: [{"worker_id": t.worker_id, "time_sec": t.time_sec}
                   for t in timings]
            for name, timings in timing.parallel_timings.items()
        },
        "task_timings": {
            name: [{"worker_id": t.worker_id, "time_sec": t.time_sec}
                   for t in timings]
            for name, timings in timing.task_timings.items()
        },
    }


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, stddev, min, max for a list of values."""
    if not values:
        return {"mean": 0.0}
    if len(values) == 1:
        return {"mean": values[0]}
    from statistics import mean, stdev
    return {
        "mean": mean(values),
        "stddev": stdev(values),
        "min": min(values),
        "max": max(values),
    }


def calculate_statistics(results: List[BenchmarkResult]) -> Dict[str, Dict]:
    """Calculate statistics for multiple runs grouped by config."""
    from collections import defaultdict

    # Group by config (name + threads + nodes)
    groups: Dict[Tuple, List[BenchmarkResult]] = defaultdict(list)
    for r in results:
        key = (r.name, r.config.arts_threads, r.config.arts_nodes)
        groups[key].append(r)

    stats = {}
    for key, runs in groups.items():
        _name, threads, nodes = key
        # Extract timings
        arts_build_times = []
        omp_build_times = []
        arts_init_times = []
        omp_init_times = []
        arts_e2e_times = []
        omp_e2e_times = []
        # Keep kernel times as optional context
        arts_kernel_times = []
        omp_kernel_times = []
        for r in runs:
            # Collect build times
            if r.build_arts.status == Status.PASS:
                arts_build_times.append(r.build_arts.duration_sec)
            if r.build_omp.status == Status.PASS:
                omp_build_times.append(r.build_omp.duration_sec)

            arts_init = get_init_time(r.run_arts)
            omp_init = get_init_time(r.run_omp)
            if arts_init is not None:
                arts_init_times.append(arts_init)
            if omp_init is not None:
                omp_init_times.append(omp_init)

            arts_e2e = get_e2e_time(r.run_arts)
            omp_e2e = get_e2e_time(r.run_omp)
            if arts_e2e is not None:
                arts_e2e_times.append(arts_e2e)
            if omp_e2e is not None:
                omp_e2e_times.append(omp_e2e)

            if r.run_arts.kernel_timings:
                # Use the first kernel timing (most benchmarks have one)
                first_key = next(iter(r.run_arts.kernel_timings))
                arts_kernel_times.append(r.run_arts.kernel_timings[first_key])
            if r.run_omp.kernel_timings:
                first_key = next(iter(r.run_omp.kernel_timings))
                omp_kernel_times.append(r.run_omp.kernel_timings[first_key])

        speedups = [r.timing.speedup for r in runs if r.timing.speedup > 0]

        config_key = f"{threads}_threads"
        if nodes > 1:
            config_key = f"{threads}_threads_{nodes}_nodes"

        stats[config_key] = {
            "arts_build_time": compute_stats(arts_build_times),
            "omp_build_time": compute_stats(omp_build_times),
            "arts_init_time": compute_stats(arts_init_times),
            "omp_init_time": compute_stats(omp_init_times),
            "arts_e2e_time": compute_stats(arts_e2e_times),
            "omp_e2e_time": compute_stats(omp_e2e_times),
            "arts_kernel_time": compute_stats(arts_kernel_times),
            "omp_kernel_time": compute_stats(omp_kernel_times),
            "speedup": compute_stats(speedups),
            "run_count": len(runs),
        }

    return stats


def export_json(
    results: List[BenchmarkResult],
    output_path: Path,
    size: str,
    total_duration: float,
    threads_list: Optional[List[int]] = None,
    cflags: Optional[str] = None,
    launcher: Optional[str] = None,
    weak_scaling: bool = False,
    base_size: Optional[int] = None,
    runs_per_config: int = 1,
    artifacts_directory: Optional[str] = None,
    fixed_threads: Optional[int] = None,
    fixed_nodes: Optional[int] = None,
    omp_threads_override: Optional[int] = None,
    arts_config_override: Optional[str] = None,
) -> None:
    """Export results to JSON file with comprehensive reproducibility metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    carts_dir = get_carts_dir()
    benchmarks_dir = get_benchmarks_dir()

    # Collect comprehensive reproducibility metadata
    repro_metadata = get_reproducibility_metadata(carts_dir, benchmarks_dir)

    # Collect experiment metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "hostname": platform.node(),
        "size": size,
        "total_duration_seconds": total_duration,
        "runs_per_config": runs_per_config,
        # Include reproducibility bundle
        "reproducibility": repro_metadata,
    }

    # Add experiment configuration
    if threads_list:
        metadata["thread_sweep"] = threads_list
    if fixed_threads is not None:
        metadata["fixed_threads"] = fixed_threads
    if cflags:
        metadata["cflags"] = cflags
    if launcher:
        metadata["launcher"] = launcher
    if fixed_nodes is not None:
        metadata["fixed_nodes"] = fixed_nodes
    if omp_threads_override is not None:
        metadata["omp_threads_override"] = omp_threads_override
    if arts_config_override is not None:
        metadata["arts_config_override"] = arts_config_override
    if weak_scaling:
        metadata["weak_scaling"] = {
            "enabled": True,
            "base_size": base_size,
        }
    if artifacts_directory:
        metadata["artifacts_directory"] = artifacts_directory

    # Calculate summary
    passed = sum(1 for r in results if r.run_arts.status ==
                 Status.PASS and r.verification.correct)
    failed = sum(1 for r in results if r.run_arts.status in (
        Status.FAIL, Status.CRASH))
    skipped = sum(1 for r in results if r.run_arts.status == Status.SKIP)
    total = len(results)

    # Count unique configs
    unique_configs = len(set((r.name, r.config.arts_threads, r.config.arts_nodes)
                             for r in results))

    speedups = [r.timing.speedup for r in results if r.timing.speedup > 0]
    if speedups:
        import math
        avg_speedup = sum(speedups) / len(speedups)
        geomean_speedup = math.exp(sum(math.log(s)
                                   for s in speedups) / len(speedups))
    else:
        avg_speedup = 0.0
        geomean_speedup = 0.0

    summary: Dict[str, Any] = {
        "total_configs": unique_configs,
        "total_runs": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "pass_rate": passed / total if total > 0 else 0.0,
        "avg_speedup": avg_speedup,
        "geometric_mean_speedup": geomean_speedup,
    }

    # Add statistics when multiple runs
    if runs_per_config > 1:
        summary["statistics"] = calculate_statistics(results)

    # Convert results to dict
    def result_to_dict(r: BenchmarkResult) -> Dict[str, Any]:
        return {
            "name": r.name,
            "suite": r.suite,
            "size": r.size,
            "size_params": r.size_params,
            "config": {
                "arts_threads": r.config.arts_threads,
                "arts_nodes": r.config.arts_nodes,
                "omp_threads": r.config.omp_threads,
                "launcher": r.config.launcher,
            },
            "run_number": r.run_number,
            "build_arts": {
                "status": r.build_arts.status.value,
                "duration_sec": r.build_arts.duration_sec,
            },
            "build_omp": {
                "status": r.build_omp.status.value,
                "duration_sec": r.build_omp.duration_sec,
            },
            "run_arts": {
                "status": r.run_arts.status.value,
                "duration_sec": r.run_arts.duration_sec,
                "exit_code": r.run_arts.exit_code,
                "checksum": r.run_arts.checksum,
                "kernel_timings": r.run_arts.kernel_timings,
                "e2e_timings": r.run_arts.e2e_timings,
                "init_timings": r.run_arts.init_timings,
                "parallel_task_timing": _serialize_parallel_task_timing(r.run_arts.parallel_task_timing),
                "perf_metrics": asdict(r.run_arts.perf_metrics) if r.run_arts.perf_metrics else None,
                "perf_csv_path": r.run_arts.perf_csv_path,
            },
            "run_omp": {
                "status": r.run_omp.status.value,
                "duration_sec": r.run_omp.duration_sec,
                "exit_code": r.run_omp.exit_code,
                "checksum": r.run_omp.checksum,
                "kernel_timings": r.run_omp.kernel_timings,
                "e2e_timings": r.run_omp.e2e_timings,
                "init_timings": r.run_omp.init_timings,
                "parallel_task_timing": _serialize_parallel_task_timing(r.run_omp.parallel_task_timing),
                "perf_metrics": asdict(r.run_omp.perf_metrics) if r.run_omp.perf_metrics else None,
                "perf_csv_path": r.run_omp.perf_csv_path,
            },
            "timing": {
                "arts_time_sec": r.timing.arts_time_sec,
                "omp_time_sec": r.timing.omp_time_sec,
                "speedup": r.timing.speedup,
                "speedup_basis": r.timing.speedup_basis,
                "arts_kernel_sec": r.timing.arts_kernel_sec,
                "omp_kernel_sec": r.timing.omp_kernel_sec,
                "arts_e2e_sec": r.timing.arts_e2e_sec,
                "omp_e2e_sec": r.timing.omp_e2e_sec,
                "arts_init_sec": r.timing.arts_init_sec,
                "omp_init_sec": r.timing.omp_init_sec,
                "arts_total_sec": r.timing.arts_total_sec,
                "omp_total_sec": r.timing.omp_total_sec,
                "note": r.timing.note,
            },
            "verification": {
                "correct": r.verification.correct,
                "tolerance": r.verification.tolerance_used,
                "note": r.verification.note,
            },
            "artifacts": asdict(r.artifacts),
            "timestamp": r.timestamp,
        }

    # Collect failures
    failures = []
    for r in results:
        if r.run_arts.status in (Status.FAIL, Status.CRASH, Status.TIMEOUT):
            failures.append({
                "name": r.name,
                "config": {
                    "arts_threads": r.config.arts_threads,
                    "arts_nodes": r.config.arts_nodes,
                },
                "run_number": r.run_number,
                "phase": "run_arts",
                "error": r.run_arts.status.value,
                "exit_code": r.run_arts.exit_code,
                "stderr": r.run_arts.stderr[:500] if r.run_arts.stderr else "",
                "artifacts": {
                    "benchmark_dir": r.artifacts.benchmark_dir,
                    "run_dir": r.artifacts.run_dir,
                    "arts_log": r.artifacts.arts_log,
                },
            })
        elif r.build_arts.status == Status.FAIL:
            failures.append({
                "name": r.name,
                "config": {
                    "arts_threads": r.config.arts_threads,
                    "arts_nodes": r.config.arts_nodes,
                },
                "run_number": r.run_number,
                "phase": "build_arts",
                "error": "build_failed",
                "output": r.build_arts.output[:500],
                "artifacts": {
                    "benchmark_dir": r.artifacts.benchmark_dir,
                    "build_dir": r.artifacts.build_dir,
                },
            })

    export_data = {
        "metadata": metadata,
        "summary": summary,
        "results": [result_to_dict(r) for r in results],
        "failures": failures,
    }

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)


# ============================================================================
# CLI Commands
# ============================================================================


@app.command(name="list")
def list_benchmarks(
    suite: Optional[str] = typer.Option(
        None, "--suite", "-s", help="Filter by suite"),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json, plain"),
):
    """List all available benchmarks."""
    runner = BenchmarkRunner(console)
    benchmarks = runner.discover_benchmarks(suite)

    if format == "json":
        console.print_json(data=benchmarks)
    elif format == "plain":
        for bench in benchmarks:
            console.print(bench)
    else:
        # Group by suite
        suites: Dict[str, List[str]] = {}
        for bench in benchmarks:
            parts = bench.split("/")
            suite_name = parts[0] if len(parts) > 1 else ""
            if suite_name not in suites:
                suites[suite_name] = []
            suites[suite_name].append(bench)

        console.print(
            f"\n[bold]Available CARTS Benchmarks[/] ({len(benchmarks)} total)\n")

        for suite_name in sorted(suites.keys()):
            if suite_name:
                console.print(f"[cyan]{suite_name}:[/]")
            for bench in sorted(suites[suite_name]):
                console.print(f"  {bench}")
            console.print()


@app.command()
def run(
    benchmarks: Optional[List[str]] = typer.Argument(
        None, help="Specific benchmarks to run"),
    size: str = typer.Option(DEFAULT_SIZE, "--size",
                             "-s", help="Dataset size: small, medium, large, extralarge"),
    timeout: int = typer.Option(
        DEFAULT_TIMEOUT, "--timeout", "-t", help="Execution timeout in seconds"),
    no_verify: bool = typer.Option(
        False, "--no-verify", help="Disable correctness verification"),
    no_clean: bool = typer.Option(
        False, "--no-clean", help="Skip cleaning before build (faster, but may use stale artifacts)"),
    arts_config: Optional[Path] = typer.Option(
        None, "--arts-config", help="Custom arts.cfg file"),
    suite: Optional[str] = typer.Option(
        None, "--suite", help="Filter by suite"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Minimal output (CI mode)"),
    trace: bool = typer.Option(
        False, "--trace", help="Show benchmark output (kernel timing and checksum)"),
    threads: Optional[str] = typer.Option(
        None, "--threads", help="Thread counts: '1,2,4,8' or '1:16:2' for thread sweep"),
    omp_threads: Optional[int] = typer.Option(
        None, "--omp-threads", help="OpenMP thread count (default: same as ARTS threads)"),
    launcher: Optional[str] = typer.Option(
        None, "--launcher", "-l", help="Override ARTS launcher: ssh, slurm, lsf, local (default: from arts.cfg)"),
    nodes: Optional[str] = typer.Option(
        None, "--nodes", "-n", help="Node counts: single (2), list (1,2,4), range (1:8:2)"),
    weak_scaling: bool = typer.Option(
        False, "--weak-scaling", help="Enable weak scaling: auto-scale problem size with parallelism"),
    base_size: Optional[int] = typer.Option(
        None, "--base-size", help="Base problem size for weak scaling (at base parallelism)"),
    cflags: Optional[str] = typer.Option(
        None, "--cflags", help="Additional CFLAGS: '-DNI=500 -DNJ=500'"),
    arts_exec_args: Optional[str] = typer.Option(
        None, "--arts-exec-args", help="Extra carts compile args (e.g., '--partition-fallback=fine')"),
    debug_level: int = typer.Option(
        0, "--debug", "-d", help="Debug level: 0=off, 1=commands, 2=verbose console output"),
    profile: Optional[Path] = typer.Option(
        None, "--profile", help="Custom counter profile file. Triggers ARTS rebuild with this configuration."),
    runs: int = typer.Option(
        1, "--runs", "-r", help="Number of times to run each benchmark for statistical significance"),
    report: bool = typer.Option(
        False, "--report", help="Generate a Markdown+SVG report artifact after running"),
    report_dir: Optional[Path] = typer.Option(
        None, "--report-dir", help="Directory for report outputs (default: alongside results)"),
    perf: bool = typer.Option(
        False, "--perf", help="Enable perf stat profiling for cache metrics"),
    perf_interval: float = typer.Option(
        0.1, "--perf-interval", help="Perf stat sampling interval in seconds"),
    results_dir: Optional[Path] = typer.Option(
        None, "--results-dir", help="Base directory for experiment output (default: carts-benchmarks/results/)"),
    exclude: Optional[List[str]] = typer.Option(
        None, "--exclude", "-e",
        help="Benchmarks to exclude (substring match, repeatable)"),
):
    """Run benchmarks with verification and timing."""
    verify = not no_verify
    clean = not no_clean

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve results_dir — default to carts-benchmarks/results/
    if results_dir is None:
        results_dir = get_benchmarks_dir() / "results"
    results_dir = Path(results_dir).resolve()

    # Create ArtifactManager — every run gets a self-contained timestamped directory
    am = ArtifactManager(results_dir, run_timestamp)

    # Create runner with artifact manager
    runner = BenchmarkRunner(
        console, verbose=verbose, quiet=quiet, trace=trace, clean=clean,
        debug=debug_level, artifact_manager=am,
    )

    # Parse thread specification
    threads_list = None
    if threads:
        threads_list = parse_threads(threads)

    # Parse node specification
    node_counts = None
    if nodes:
        node_counts = parse_threads(nodes)

    # Discover or use provided benchmarks
    if benchmarks:
        bench_list = list(benchmarks)
    else:
        bench_list = runner.discover_benchmarks(suite)

    # Apply --exclude filter
    if exclude:
        before = len(bench_list)
        bench_list = [b for b in bench_list
                      if not any(ex in b for ex in exclude)]
        excluded = before - len(bench_list)
        if excluded:
            console.print(f"[dim]  ({excluded} benchmarks excluded via --exclude)[/]")

    if not bench_list:
        console.print("[yellow]No benchmarks found.[/]")
        raise typer.Exit(1)

    # Handle custom counter profile (triggers ARTS rebuild)
    if profile:
        if not profile.exists():
            console.print(f"[red]Error: Profile not found: {profile}[/]")
            raise typer.Exit(1)

        console.print(f"[yellow]Rebuilding ARTS with profile: {profile}[/]")
        import subprocess
        result = subprocess.run(
            ["carts", "build", "--arts", f"--profile={profile}"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            console.print(f"[red]ARTS rebuild failed:[/]\n{result.stderr}")
            raise typer.Exit(1)
        console.print("[green]ARTS rebuild complete[/]")

    # Print header
    if not quiet:
        config_items = [f"size={size}", f"timeout={timeout}s",
                        f"verify={verify}", f"clean={clean}"]
        if threads_list:
            config_items.append(f"threads={threads}")
        if launcher is not None:
            config_items.append(f"launcher={launcher}")
        if node_counts is not None:
            config_items.append(f"nodes={nodes}")
        if runs > 1:
            config_items.append(f"runs={runs}")
        if cflags:
            config_items.append(f"cflags={cflags}")
        if arts_exec_args:
            config_items.append(f"arts-exec-args={arts_exec_args}")
        if debug_level > 0:
            config_items.append(f"debug={debug_level}")
        if profile:
            config_items.append(f"profile={profile.name}")
        if perf:
            config_items.append("perf=on")
        subtitle = f"Config: {', '.join(config_items)}"
        print_header("CARTS Benchmark Runner", subtitle)

        # Show effective ARTS configuration
        if bench_list:
            # Multiple benchmarks without custom config: each uses its own local config
            if len(bench_list) > 1 and not arts_config:
                console.print("ARTS Config: using local")
            else:
                # Single benchmark or custom config: show specific values
                sample_bench = bench_list[0]
                bench_path = runner.benchmarks_dir / sample_bench

                # Determine effective config file and source in one pass
                if arts_config:
                    effective_config = arts_config
                    config_source = "custom"
                else:
                    bench_cfg = bench_path / "arts.cfg"
                    if bench_cfg.exists():
                        effective_config = bench_cfg
                        config_source = "local"
                    else:
                        effective_config = DEFAULT_ARTS_CONFIG
                        config_source = "default"

                # Parse ARTS config values once
                cfg = parse_arts_cfg(effective_config)
                arts_threads = int(cfg.get("threads", "1"))
                arts_nodes = int(cfg.get("nodeCount", "1"))
                arts_launcher = cfg.get("launcher", "ssh")
                arts_nodes_list = [n.strip() for n in cfg.get(
                    "nodes", "localhost").split(",") if n.strip()]

                # Build config display items efficiently
                arts_config_items = [f"arts-threads={arts_threads}",
                                     f"arts-nodes={arts_nodes}",
                                     f"arts-launcher={arts_launcher}"]

                # Always show node list (shows configured nodes including ports)
                # Escape brackets to prevent Rich from interpreting as markup
                if arts_nodes_list:
                    node_display = ','.join(arts_nodes_list[:3])
                    if len(arts_nodes_list) > 3:
                        node_display += "..."
                    arts_config_items.append(
                        f"arts-node-list=\\[{node_display}]")

                console.print(
                    f"ARTS Config ({config_source}): {', '.join(arts_config_items)}")

        console.print(f"Benchmarks: {len(bench_list)}\n")

    # Run benchmarks
    start_time = time.time()

    # Validate multi-benchmark restrictions
    threads_override: Optional[int] = None
    if len(bench_list) > 1:
        if threads_list and len(threads_list) > 1:
            console.print(
                "[red]Error:[/] For multiple benchmarks, `--threads` must specify a single value (e.g., `--threads 8`)."
            )
            raise typer.Exit(2)
        if node_counts and len(node_counts) > 1:
            console.print(
                "[red]Error:[/] For multiple benchmarks, `--nodes` must specify a single value (e.g., `--nodes 2`)."
            )
            raise typer.Exit(2)
        if threads_list:
            threads_override = int(threads_list[0])
            threads_list = None

    # Determine if sweep mode
    has_thread_sweep = threads_list and len(threads_list) > 1
    has_node_sweep = node_counts and len(node_counts) > 1

    try:
        if len(bench_list) == 1 and (has_thread_sweep or has_node_sweep):
            # Sweep mode for single benchmark
            effective_threads = threads_list if threads_list else [None]
            effective_nodes = node_counts if node_counts else [None]
            results = runner.run_with_thread_sweep(
                bench_list[0],
                size,
                effective_threads,
                arts_config,
                cflags or "",
                None,  # counter_dir (auto-managed by ArtifactManager)
                timeout,
                omp_threads,
                launcher,
                node_counts=effective_nodes,
                weak_scaling=weak_scaling,
                base_size=base_size,
                runs=runs,
                arts_exec_args=arts_exec_args,
                perf_enabled=perf,
                perf_interval=perf_interval,
            )
        else:
            # Standard mode
            if threads_list and len(threads_list) == 1:
                threads_override = int(threads_list[0])
                threads_list = None
            nodes_int = int(node_counts[0]) if node_counts else None
            results = runner.run_all(
                bench_list,
                size=size,
                timeout=timeout,
                verify=verify,
                arts_config=arts_config,
                threads_override=threads_override,
                nodes_override=nodes_int,
                launcher_override=launcher,
                omp_threads_override=omp_threads,
                arts_exec_args=arts_exec_args,
                perf_enabled=perf,
                perf_interval=perf_interval,
                runs=runs,
                run_timestamp=run_timestamp,
            )
    except ValueError as e:
        console.print(f"\n[red]Error:[/] {e}")
        raise typer.Exit(1)

    total_duration = time.time() - start_time

    # Display results
    if not quiet:
        # Table was already shown via Live display, just show the summary panel
        console.print()
        console.print(create_summary_panel(results, total_duration))

    # Write results.json into the experiment directory
    export_json(
        results,
        am.results_json_path,
        size,
        total_duration,
        threads_list,
        cflags,
        launcher,
        weak_scaling,
        base_size,
        runs,
        ".",  # artifacts are in same directory
        fixed_threads=threads_override,
        fixed_nodes=nodes,
        omp_threads_override=omp_threads,
        arts_config_override=str(arts_config) if arts_config else None,
    )

    # Write manifest.json
    command_str = "carts benchmarks " + " ".join(sys.argv[1:])
    am.write_manifest(results, command_str, total_duration)

    if report:
        out_dir = report_dir or am.experiment_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        report_md = out_dir / f"benchmark_report_{run_timestamp}.md"
        report_svg = out_dir / \
            f"benchmark_report_{run_timestamp}_init_e2e_small_multiples.svg"
        report_chart, report_speedup = generate_markdown_report(
            results, report_md, report_svg, size=size)
        if not quiet:
            console.print()
            console.print(f"[green]Report:[/] {report_md}")
            console.print(f"[green]Chart:[/]  {report_chart}")
            console.print(f"[green]Speedup:[/] {report_speedup}")

    # Show single results path
    if not quiet:
        console.print()
        console.print(f"[green]Results:[/] {am.experiment_dir}")

    # Exit with error if any failures
    failed = sum(1 for r in results if r.run_arts.status in (
        Status.FAIL, Status.CRASH))
    if failed > 0:
        raise typer.Exit(1)


@app.command()
def build(
    benchmarks: Optional[List[str]] = typer.Argument(
        None, help="Specific benchmarks to build"),
    size: str = typer.Option(DEFAULT_SIZE, "--size",
                             "-s", help="Dataset size: small, medium, large, extralarge"),
    openmp: bool = typer.Option(
        False, "--openmp", help="Build OpenMP version only"),
    arts: bool = typer.Option(False, "--arts", help="Build ARTS version only"),
    suite: Optional[str] = typer.Option(
        None, "--suite", help="Filter by suite"),
    arts_config: Optional[Path] = typer.Option(
        None, "--arts-config", help="Custom arts.cfg file"),
):
    """Build benchmarks without running."""
    runner = BenchmarkRunner(console)

    # Discover or use provided benchmarks
    if benchmarks:
        bench_list = list(benchmarks)
    else:
        bench_list = runner.discover_benchmarks(suite)

    if not bench_list:
        console.print("[yellow]No benchmarks found.[/]")
        raise typer.Exit(1)

    # Determine variants to build
    variants = []
    if openmp:
        variants.append("openmp")
    elif arts:
        variants.append("arts")
    else:
        variants = ["arts", "openmp"]

    print_header("Build Benchmarks", f"{len(bench_list)} benchmarks, size={size}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Building...", total=len(bench_list) * len(variants))

        for bench in bench_list:
            for variant in variants:
                progress.update(
                    task, description=f"[cyan]{bench}[/] ({variant})")
                result = runner.build_benchmark(
                    bench, size, variant, arts_config)
                status = status_symbol(result.status)
                if result.status != Status.PASS:
                    console.print(
                        f"  {status} {bench} ({variant}): {result.status.value}")
                progress.advance(task)

    print_success("Build complete!")


@app.command()
def clean(
    benchmarks: Optional[List[str]] = typer.Argument(
        None, help="Specific benchmarks to clean"),
    all: bool = typer.Option(False, "--all", "-a",
                             help="Clean all benchmarks"),
    suite: Optional[str] = typer.Option(
        None, "--suite", help="Filter by suite"),
):
    """Clean build artifacts."""
    runner = BenchmarkRunner(console)

    if all:
        bench_list = runner.discover_benchmarks()
    elif benchmarks:
        bench_list = list(benchmarks)
    else:
        bench_list = runner.discover_benchmarks(suite)

    if not bench_list:
        console.print("[yellow]No benchmarks found.[/]")
        raise typer.Exit(1)

    print_header("Clean Benchmarks", f"{len(bench_list)} benchmarks")

    cleaned = 0
    for bench in bench_list:
        if runner.clean_benchmark(bench):
            console.print(f"  [{Colors.PASS}]{Symbols.PASS}[/{Colors.PASS}] {bench}")
            cleaned += 1
        else:
            console.print(f"  [{Colors.DIM}]{Symbols.SKIP}[/{Colors.DIM}] {bench} (nothing to clean)")

    # Clean shared artifacts
    shared_removed = runner.clean_shared_artifacts()
    if shared_removed:
        console.print(f"  [{Colors.PASS}]{Symbols.PASS}[/{Colors.PASS}] Shared artifacts ({shared_removed} items)")

    print_success(f"Cleaned {cleaned} benchmarks!")


# ============================================================================
# SLURM Batch Command
# ============================================================================


@app.command(name="slurm-run")
def slurm_run(
    benchmarks: Optional[List[str]] = typer.Argument(
        None, help="Specific benchmarks to run (default: all)"),
    nodes: str = typer.Option(
        ..., "--nodes", "-n",
        help="Node counts: single (4), range (1-15), or list (1,2,4,8,16)"),
    size: str = typer.Option(
        DEFAULT_SIZE, "--size", "-s",
        help="Dataset size: small, medium, large, extralarge"),
    runs: int = typer.Option(
        1, "--runs", "-r", help="Number of runs per benchmark"),
    partition: Optional[str] = typer.Option(
        None, "--partition", "-p",
        help="SLURM partition (uses cluster default if not specified)"),
    time_limit: str = typer.Option(
        "01:00:00", "--time", "-t", help="Time limit per job (HH:MM:SS)"),
    account: Optional[str] = typer.Option(
        None, "--account", "-A", help="SLURM account (if required)"),
    arts_config: Optional[Path] = typer.Option(
        None, "--arts-config",
        help="Base arts.cfg file (for threads and other settings)"),
    threads: Optional[int] = typer.Option(
        None, "--threads",
        help="Thread count for OpenMP comparison (default: from arts.cfg or 8)"),
    output_dir: Path = typer.Option(
        Path("./results"), "--results-dir",
        help="Base directory for experiment output"),
    suite: Optional[str] = typer.Option(
        None, "--suite", help="Filter by suite"),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Generate scripts but don't submit to SLURM"),
    no_build: bool = typer.Option(
        False, "--no-build",
        help="Skip build phase (assumes executables already exist)"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Verbose output"),
    cflags: Optional[str] = typer.Option(
        None, "--cflags", help="Additional compiler flags"),
    gdb: bool = typer.Option(
        False, "--gdb",
        help="Wrap executable with gdb for backtrace on crash"),
    profile: Optional[Path] = typer.Option(
        None, "--profile",
        help="Custom counter profile file. Triggers ARTS rebuild with this configuration."),
    perf: bool = typer.Option(
        False, "--perf",
        help="Enable perf stat profiling for cache metrics"),
    perf_interval: float = typer.Option(
        0.1, "--perf-interval",
        help="Perf stat sampling interval in seconds"),
    exclude_nodes: Optional[str] = typer.Option(
        None, "--exclude-nodes", "-X",
        help="SLURM nodes to exclude (comma-separated, e.g. j006,j007)"),
    exclude: Optional[List[str]] = typer.Option(
        None, "--exclude", "-e",
        help="Benchmarks to exclude (substring match, repeatable)"),
):
    """Submit benchmarks as SLURM batch jobs.

    This command:
    1. Builds all benchmarks locally (reusable across jobs)
    2. Generates sbatch scripts for each (benchmark x node_count x run)
    3. Submits all jobs to SLURM queue
    4. Monitors job progress
    5. Collects and aggregates results

    Example:
        carts benchmarks slurm-run --nodes=1-15 --runs 10
        carts benchmarks slurm-run polybench/gemm --nodes=4 -t 00:30:00
        carts benchmarks slurm-run --nodes=1,2,4,8,16 --partition compute

    Key Features:
    - Submit all jobs at once (no sequential waiting)
    - Exclusive node allocation for resource isolation
    - Each job has isolated counter directory (no data collision)
    - Sweep across multiple node counts with --nodes=1-15
    """
    # Initialize
    slurm_start_time = time.time()
    runner = BenchmarkRunner(console, verbose, False, False, False, 0)

    # Parse node counts from --nodes parameter
    try:
        node_counts = parse_node_spec(nodes)
    except ValueError as e:
        console.print(f"[red]Error parsing --nodes: {e}[/]")
        raise typer.Exit(1)

    # Determine base arts config (for threads and other settings)
    # CRITICAL: Resolve to absolute path for SLURM jobs running from different directories
    base_config = (arts_config.resolve() if arts_config else DEFAULT_ARTS_CONFIG.resolve())
    if not base_config.exists():
        console.print(f"[red]Error: arts.cfg not found: {base_config}[/]")
        raise typer.Exit(1)

    # Get threads from config or default
    if threads is None:
        threads = get_arts_cfg_int(base_config, "threads") or 8

    # Format node counts for display
    if len(node_counts) == 1:
        nodes_display = str(node_counts[0])
    elif len(node_counts) <= 5:
        nodes_display = ", ".join(str(n) for n in node_counts)
    else:
        nodes_display = f"{node_counts[0]}-{node_counts[-1]} ({len(node_counts)} values)"

    subtitle_parts = [
        f"Config: {base_config}",
        f"Nodes: {nodes_display}, Threads: {threads}",
        f"Runs per benchmark: {runs}, Size: {size}",
    ]
    if profile:
        subtitle_parts.append(f"Profile: {profile}")
    if perf:
        subtitle_parts.append(f"Perf: enabled (interval={perf_interval}s)")
    if exclude_nodes:
        subtitle_parts.append(f"Excluding SLURM nodes: {exclude_nodes}")
    if exclude:
        subtitle_parts.append(f"Excluding benchmarks: {', '.join(exclude)}")
    print_header("SLURM Batch Submission", "\n".join(subtitle_parts))

    # Discover benchmarks
    if benchmarks:
        bench_list = benchmarks
    else:
        bench_list = runner.discover_benchmarks(suite)

    # Apply --exclude filter for benchmarks
    if exclude:
        before = len(bench_list)
        bench_list = [b for b in bench_list
                      if not any(ex in b for ex in exclude)]
        excluded_count = before - len(bench_list)
        if excluded_count:
            console.print(f"[{Colors.DIM}]  ({excluded_count} benchmarks excluded via --exclude)[/{Colors.DIM}]")

    if not bench_list:
        print_warning("No benchmarks to run.")
        raise typer.Exit(0)

    # Check which benchmarks support multinode
    multinode_disabled = set()
    for bench in bench_list:
        bench_path = runner.benchmarks_dir / bench
        if (bench_path / ".disable-multinode").exists():
            multinode_disabled.add(bench)

    # Calculate total jobs: sum over node counts
    # (benchmarks with multinode disabled only run with node_count=1)
    total_jobs = 0
    for nc in node_counts:
        if nc == 1:
            total_jobs += len(bench_list) * runs
        else:
            valid_benchmarks = len(bench_list) - len(multinode_disabled)
            total_jobs += valid_benchmarks * runs

    console.print(f"\n[bold]Found {len(bench_list)} benchmarks, {len(node_counts)} node counts, {total_jobs} total jobs[/]")
    if multinode_disabled and max(node_counts) > 1:
        console.print(f"[dim]  ({len(multinode_disabled)} benchmarks disabled for multi-node)[/]")

    # Create directories:
    # - build/: shared across experiments (reusable executables)
    # - results/slurm_{timestamp}/: experiment-specific (jobs, counters, results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    build_dir = output_dir / "build"  # Shared, reusable
    experiment_dir = output_dir / f"slurm_{timestamp}"  # Experiment-specific
    experiment_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir = experiment_dir / "jobs"

    console.print(f"Build directory: {build_dir} (shared)")
    console.print(f"Experiment directory: {experiment_dir}")

    # Handle custom counter profile (triggers ARTS rebuild)
    if profile:
        if not profile.exists():
            console.print(f"[red]Error: Profile not found: {profile}[/]")
            raise typer.Exit(1)

        console.print(f"[yellow]Rebuilding ARTS with profile: {profile}[/]")
        result = subprocess.run(
            ["carts", "build", "--arts", f"--profile={profile}"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            console.print(f"[red]ARTS rebuild failed:[/]\n{result.stderr}")
            raise typer.Exit(1)
        console.print("[green]ARTS rebuild complete[/]")

    # Phase 1: Build per (benchmark, node_count) into shared build/ directory
    # CRITICAL: ARTS executable embeds nodeCount at compile time via arts.cfg
    # So we must build separately for each node_count
    # Executables are REUSABLE across experiments (same binary for same benchmark+node_count)
    import shutil

    num_workers = min(os.cpu_count() or 1, len(bench_list))
    console.print(f"\n[bold]Phase 1: Building benchmarks per node count ({num_workers} workers)...[/]")

    # build_results: {(bench, node_count): (arts_exe, omp_exe, build_arts_cfg)}
    # Note: build_arts_cfg is the compile-time config in build/ directory
    build_results: Dict[Tuple[str, int], Tuple[Path, Optional[Path], Path]] = {}
    print_lock = threading.Lock()

    def _build_one_bench(bench: str) -> List[Tuple[Tuple[str, int], Tuple[Path, Optional[Path], Path]]]:
        """Build all node_count variants for one benchmark (sequential within)."""
        safe_name = bench.replace("/", "_")
        bench_path = runner.benchmarks_dir / bench
        src_arts, src_omp = runner.get_executable_paths(bench_path)
        results = []

        for node_count in node_counts:
            if node_count > 1 and bench in multinode_disabled:
                continue

            # Build directory: build/{benchmark}/nodes_{N}/{T}T/ (shared, reusable)
            build_node_dir = build_dir / safe_name / f"nodes_{node_count}" / f"{threads}T"
            build_node_dir.mkdir(parents=True, exist_ok=True)

            dst_arts = build_node_dir / src_arts.name
            dst_omp = build_node_dir / src_omp.name if node_count == 1 else None
            build_arts_cfg = build_node_dir / "arts.cfg"

            # Skip if ARTS executable already exists (reuse from previous experiment)
            if dst_arts.exists() and build_arts_cfg.exists():
                with print_lock:
                    console.print(f"  {bench} (nodes={node_count}, threads={threads})... [cyan]SKIP (exists)[/]")
                if node_count == 1 and dst_omp and not dst_omp.exists():
                    build_omp = runner.build_benchmark(
                        bench, size, variant="openmp",
                        cflags=cflags or "",
                    )
                    if build_omp.status == Status.PASS and src_omp.exists():
                        shutil.copy2(src_omp, dst_omp)
                results.append(((bench, node_count), (dst_arts, dst_omp if dst_omp and dst_omp.exists() else None, build_arts_cfg)))
                continue

            if no_build:
                with print_lock:
                    console.print(f"  {bench} (nodes={node_count}, threads={threads})... [red]MISSING (--no-build)[/]")
                continue

            # Generate arts.cfg for compilation (counterFolder is placeholder)
            build_arts_cfg = slurm_batch.generate_arts_config_for_node(
                base_config, build_node_dir, node_count, threads
            )

            # Build ARTS with this node-specific arts.cfg
            build_arts = runner.build_benchmark(
                bench, size, variant="arts",
                arts_config=build_arts_cfg,
                cflags=cflags or "",
            )

            if build_arts.status != Status.PASS:
                with print_lock:
                    console.print(f"  {bench} (nodes={node_count}, threads={threads})... [red]FAILED[/]")
                    if verbose:
                        console.print(f"    {build_arts.output[:200]}...")
                continue

            # Copy ARTS executable and LLVM IR to build directory
            shutil.copy2(src_arts, dst_arts)
            for ll_file in bench_path.glob("*-arts.ll"):
                shutil.copy2(ll_file, build_node_dir / ll_file.name)

            # Build and copy OMP (only for node_count=1)
            dst_omp = None
            if node_count == 1:
                build_omp = runner.build_benchmark(
                    bench, size, variant="openmp",
                    cflags=cflags or "",
                )
                if build_omp.status == Status.PASS and src_omp.exists():
                    dst_omp = build_node_dir / src_omp.name
                    shutil.copy2(src_omp, dst_omp)

            with print_lock:
                console.print(f"  {bench} (nodes={node_count}, threads={threads})... [green]OK[/]")
            results.append(((bench, node_count), (dst_arts, dst_omp, build_arts_cfg)))

        return results

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_build_one_bench, bench): bench for bench in bench_list}
        for future in as_completed(futures):
            for key, value in future.result():
                build_results[key] = value

    # Phase 2: Generate sbatch scripts
    # Directory structure:
    #   build/{benchmark}/nodes_{N}/{T}T/       <- shared, reusable
    #   ├── arts.cfg                           <- compile-time config
    #   ├── {name}_arts, {name}_omp            <- executables
    #   ├── {name}-arts.ll                     <- LLVM IR (for debugging)
    #
    #   results/slurm_{timestamp}/jobs/{benchmark}/nodes_{N}/  <- experiment-specific
    #   ├── job_1.sbatch, job_2.sbatch, ...
    #   ├── arts_1.cfg, arts_2.cfg, ...        <- runtime config (different counterFolder)
    #   └── run_1/                             <- per-run directory
    #       ├── result.json
    #       ├── counters/
    #       ├── perf/                          <- if --perf
    #       ├── slurm.out
    #       └── slurm.err
    console.print("\n[bold]Phase 2: Generating job scripts...[/]")

    slurm_job_result_script = Path(__file__).parent / "slurm_job_result.py"

    job_configs: List[Tuple[slurm_batch.SlurmJobConfig, Path]] = []

    for (bench, node_count), (arts_exe, omp_exe, build_arts_cfg) in build_results.items():
        # Create job directory: jobs/{benchmark}/nodes_{N}/ (experiment-specific)
        safe_name = bench.replace("/", "_")
        job_node_dir = jobs_dir / safe_name / f"nodes_{node_count}"
        job_node_dir.mkdir(parents=True, exist_ok=True)

        for run_num in range(1, runs + 1):
            # Create job config - executables come from build/, output goes to jobs/
            config = slurm_batch.SlurmJobConfig(
                benchmark_name=bench,
                run_number=run_num,
                node_count=node_count,
                time_limit=time_limit,
                partition=partition,
                account=account,
                executable_arts=arts_exe,           # From build/ (shared)
                executable_omp=omp_exe,             # From build/ (shared)
                arts_config_path=build_arts_cfg,    # From build/ (template for runtime cfg)
                output_dir=job_node_dir,            # To jobs/ (experiment-specific)
                size=size,
                threads=threads,
                gdb=gdb,
                perf=perf,
                perf_interval=perf_interval,
                exclude_nodes=exclude_nodes,
            )

            # Generate sbatch script in job directory
            script_path = job_node_dir / f"job_{run_num}.sbatch"
            slurm_batch.generate_sbatch_script(
                config, script_path, slurm_job_result_script
            )

            job_configs.append((config, script_path))

    console.print(f"  Generated {len(job_configs)} job scripts")

    # Phase 3: Submit jobs
    console.print("\n[bold]Phase 3: Submitting jobs...[/]")

    job_statuses = slurm_batch.submit_all_jobs(job_configs, console, dry_run)

    # Write job manifest
    metadata = {
        "timestamp": timestamp,
        "size": size,
        "node_counts": node_counts,
        "threads": threads,
        "runs_per_benchmark": runs,
        "total_jobs": len(job_configs),
        "partition": partition,
        "time_limit": time_limit,
        "arts_config": str(base_config),
        "dry_run": dry_run,
        "profile": str(profile) if profile else None,
        "perf": perf,
        "perf_interval": perf_interval if perf else None,
    }

    manifest_path = slurm_batch.write_job_manifest(
        experiment_dir, job_statuses, metadata
    )
    console.print(f"  Job manifest: {manifest_path}")

    if dry_run:
        console.print("\n[yellow]Dry run complete. Scripts generated but not submitted.[/]")
        console.print(f"To submit manually, run sbatch on scripts in: {jobs_dir}")
        raise typer.Exit(0)

    # Phase 4: Monitor jobs
    console.print("\n[bold]Phase 4: Monitoring jobs...[/]")

    job_statuses = slurm_batch.wait_for_jobs_completion(
        job_statuses, console, poll_interval=10
    )

    # Update manifest with final job states
    slurm_batch.write_job_manifest(experiment_dir, job_statuses, metadata)

    # Phase 5: Collect results
    console.print("\n[bold]Phase 5: Collecting results...[/]")

    results = slurm_batch.collect_results(job_statuses, experiment_dir)

    results_path = slurm_batch.write_aggregated_results(
        experiment_dir, results, metadata
    )

    # Write manifest.json (same schema as standard run)
    total_duration = time.time() - slurm_start_time
    slurm_command = "carts benchmarks " + " ".join(sys.argv[1:])
    carts_dir = get_carts_dir()
    benchmarks_dir = get_benchmarks_dir()
    repro = get_reproducibility_metadata(carts_dir, benchmarks_dir)
    slurm_manifest = slurm_batch.write_slurm_manifest(
        experiment_dir, results, metadata, slurm_command, total_duration, repro
    )

    # Summary
    successful = sum(1 for r in results if r.get("status") == "PASS")
    failed = sum(1 for r in results if r.get("status") in ("FAIL", "CRASH", "TIMEOUT"))

    summary_content = (
        f"{format_summary_line(successful, failed, len(results) - successful - failed)}\n\n"
        f"Total jobs: {len(results)}\n"
        f"Results: {results_path}\n"
        f"Manifest: {slurm_manifest}"
    )
    style = "green" if failed == 0 else "red"
    print_footer(f"Experiment Complete — {summary_content}", style=style)


if __name__ == "__main__":
    # Enable debug logging when --verbose/-v is in args
    if "--verbose" in sys.argv or "-v" in sys.argv:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    app()
