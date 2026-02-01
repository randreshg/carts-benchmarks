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
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import typer
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

# ============================================================================
# Constants
# ============================================================================

VERSION = "0.1.0"
DEFAULT_TIMEOUT = 60
DEFAULT_SIZE = "small"
DEFAULT_TOLERANCE = 1e-2  # 1% tolerance for FP operation ordering differences
DEFAULT_ARTS_CONFIG = Path(__file__).parent / "arts.cfg"

CHECKSUM_PATTERNS = [
    r"checksum[:\s]*=?\s*([0-9.eE+-]+)",
    r"result[:\s]*=?\s*([0-9.eE+-]+)",
    r"sum[:\s]*=?\s*([0-9.eE+-]+)",
    r"total[:\s]*=?\s*([0-9.eE+-]+)",
    r"RMS error[:\s]*\(?\s*([0-9.eE+-]+)",
    r"^([0-9.eE+-]+)\s*$",
]

SKIP_DIRS = {"common", "include", "src", "utilities",
             ".git", ".svn", ".hg", "build", "logs"}


def filter_benchmark_output(output: str) -> str:
    """Extract only CARTS benchmark output lines (init/e2e/kernel timing, parallel/task timing, checksum).

    Filters out verbose ARTS runtime debug logs and keeps only benchmark-relevant output.
    """
    if not output:
        return ""
    lines = []
    for line in output.splitlines():
        # Keep lines that match benchmark output patterns
        if (line.startswith("kernel.") or
            line.startswith("e2e.") or
            line.startswith("init.") or
            line.startswith("parallel.") or
            line.startswith("task.") or
            line.startswith("checksum:") or
            line.startswith("tmp_checksum:") or
                "checksum:" in line.lower()):
            lines.append(line)
    return "\n".join(lines)


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
console = Console()


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
        return {}
    return values


def get_arts_cfg_int(path: Optional[Path], key: str) -> Optional[int]:
    vals = parse_arts_cfg(path)
    if key not in vals:
        return None
    try:
        return int(vals[key])
    except Exception:
        return None


def get_arts_cfg_str(path: Optional[Path], key: str) -> Optional[str]:
    vals = parse_arts_cfg(path)
    v = vals.get(key)
    return v if v else None


def get_arts_cfg_nodes(path: Optional[Path]) -> List[str]:
    """Parse nodes= field from arts.cfg into list of hostnames.

    Returns list of hostnames from the nodes= field, or ["localhost"] if not found.
    """
    cfg = parse_arts_cfg(path)
    nodes_str = cfg.get("nodes", "localhost")
    return [n.strip() for n in nodes_str.split(",") if n.strip()]


# Counter level mapping
COUNTER_LEVELS = {
    0: "off",       # No counters (counter.profile-none.cfg)
    1: "artsid",    # ArtsID metrics (counter.profile-artsid-only.cfg)
    2: "deep",      # Deep captures (counter.profile-deep.cfg)
}


def check_arts_counter_support() -> int:
    """Check what counter level ARTS was built with.

    Returns:
        0 = counters disabled (counter.profile-none.cfg)
        1 = artsid metrics (counter.profile-artsid-only.cfg)
        2 = deep captures (counter.profile-deep.cfg)
    """
    preamble = Path(get_carts_dir()) / \
        ".install/arts/include/arts/introspection/Preamble.h"
    if not preamble.exists():
        return 0

    content = preamble.read_text()

    # Check for deep captures (level 2)
    if "ENABLE_artsIdEdtCaptures 1" in content:
        return 2

    # Check for artsid metrics (level 1)
    if "ENABLE_artsIdEdtMetrics 1" in content:
        return 1

    return 0


# ============================================================================
# Artifact Directory Management
# ============================================================================


def create_experiment_dir(output_base: Path) -> Tuple[Path, Path]:
    """Create timestamped experiment directory with JSON inside.

    This creates a self-contained experiment directory where both the
    JSON results and all build/run artifacts are stored together.

    Args:
        output_base: Base path for the experiment (e.g., results/single_rank/foo).
                     The .json extension is optional and will be stripped.

    Returns:
        Tuple of (experiment_dir, json_path):
        - experiment_dir: results/single_rank/foo_20251214_143052/
        - json_path: results/single_rank/foo_20251214_143052/foo.json

    Example:
        >>> create_experiment_dir(Path("results/single_rank/foo"))
        (Path("results/single_rank/foo_20251214_143052/"),
         Path("results/single_rank/foo_20251214_143052/foo.json"))

        >>> create_experiment_dir(Path("results/foo.json"))
        (Path("results/foo_20251214_143052/"),
         Path("results/foo_20251214_143052/foo.json"))
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Strip .json extension if provided
    name = output_base.stem  # "foo" from "foo.json" or "foo"
    parent = output_base.parent  # "results/single_rank"

    experiment_dir = parent / f"{name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    json_path = experiment_dir / f"{name}.json"

    return experiment_dir, json_path


def create_config_directory(
    artifacts_base_dir: Path,
    benchmark_name: str,
    config: BenchmarkConfig,
) -> Tuple[Path, Path, Path]:
    """Create config directory structure.

    Returns:
        Tuple of (config_dir, artifacts_dir, runs_dir)

    Creates:
        {artifacts_base_dir}/{benchmark_name}/build/{threads}t_{nodes}n/
        {artifacts_base_dir}/{benchmark_name}/build/{threads}t_{nodes}n/artifacts/
        {artifacts_base_dir}/{benchmark_name}/build/{threads}t_{nodes}n/runs/
    """
    config_dir = artifacts_base_dir / benchmark_name / \
        "build" / f"{config.arts_threads}t_{config.arts_nodes}n"
    artifacts_dir = config_dir / "artifacts"
    runs_dir = config_dir / "runs"

    config_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)
    runs_dir.mkdir(exist_ok=True)

    return config_dir, artifacts_dir, runs_dir


def create_run_directory(runs_dir: Path, run_number: int) -> Path:
    """Create numbered run directory.

    Example: {runs_dir}/1/, {runs_dir}/2/, etc.
    """
    run_dir = runs_dir / str(run_number)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def copy_build_artifacts(
    bench_path: Path,
    config_dir: Path,
    artifacts_dir: Path,
    *,
    arts_cfg_used: Optional[Path] = None,
) -> Dict[str, Optional[str]]:
    """Copy build artifacts to results directory.

    Copies:
    - arts.cfg (used) -> config_dir/ (build INPUT)
    - .carts-metadata.json -> artifacts_dir/ (compiler metadata)
    - *_arts_metadata.mlir -> artifacts_dir/ (MLIR with metadata)
    - *.mlir -> artifacts_dir/ (other MLIR files)
    - *-arts.ll -> artifacts_dir/ (LLVM IR)
    - *_arts, *_omp -> artifacts_dir/ (executables)
    - logs/build_*.log -> artifacts_dir/ (build logs)

    Returns:
        Dict with paths to key artifacts in the results directory.
    """
    paths: Dict[str, Optional[str]] = {}

    # Copy arts.cfg to config root (it's the build INPUT).
    # Prefer the effective config actually used for this build/run.
    cfg_src = arts_cfg_used if arts_cfg_used and arts_cfg_used.exists() else (
        bench_path / "arts.cfg")
    if cfg_src.exists():
        dest = config_dir / "arts.cfg"
        shutil.copy2(cfg_src, dest)
        paths["arts_config"] = str(dest)

    # Copy compiler metadata to artifacts/
    metadata = bench_path / ".carts-metadata.json"
    if metadata.exists():
        dest = artifacts_dir / ".carts-metadata.json"
        shutil.copy2(metadata, dest)
        paths["carts_metadata"] = str(dest)

    # Copy *_arts_metadata.mlir to artifacts/
    for mlir in bench_path.glob("*_arts_metadata.mlir"):
        dest = artifacts_dir / mlir.name
        shutil.copy2(mlir, dest)
        paths["arts_metadata_mlir"] = str(dest)

    # Copy other MLIR files to artifacts/
    for mlir in bench_path.glob("*.mlir"):
        if "_metadata" not in mlir.name:
            shutil.copy2(mlir, artifacts_dir / mlir.name)

    # Copy LLVM IR to artifacts/
    for ll in bench_path.glob("*-arts.ll"):
        shutil.copy2(ll, artifacts_dir / ll.name)

    # Copy executables to artifacts/
    for exe in bench_path.glob("*_arts"):
        if exe.is_file():
            dest = artifacts_dir / exe.name
            shutil.copy2(exe, dest)
            paths["executable_arts"] = str(dest)
    # OpenMP reference binaries usually live under build/ (common/carts.mk).
    for exe in list(bench_path.glob("*_omp")) + list((bench_path / "build").glob("*_omp")):
        if exe.is_file():
            dest = artifacts_dir / exe.name
            shutil.copy2(exe, dest)
            paths["executable_omp"] = str(dest)

    # Copy build logs to artifacts/
    logs_dir = bench_path / "logs"
    if logs_dir.exists():
        for log in ["build_arts.log", "build_openmp.log"]:
            log_file = logs_dir / log
            if log_file.exists():
                shutil.copy2(log_file, artifacts_dir / log)

    return paths


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
    ):
        self.console = console
        self.verbose = verbose
        self.quiet = quiet
        self.trace = trace
        self.clean = clean
        self.debug = debug
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
        if variant == "openmp":
            # Build only OpenMP variant using granular target
            if size in size_targets:
                cmd = ["make", f"{size}-openmp"]
            else:
                cmd = ["make", "openmp"]
            if cflags:
                cmd.append(f"CFLAGS={cflags}")
        else:
            # Build ARTS variant (full pipeline)
            if size in size_targets:
                # Use granular size-arts target for ARTS-only builds
                cmd = ["make", f"{size}-arts"]
            else:
                cmd = ["make", "all"]
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
        threads_list: List[int],
        base_config: Optional[Path],
        cflags: str = "",
        counter_dir: Optional[Path] = None,
        timeout: int = DEFAULT_TIMEOUT,
        omp_threads: Optional[int] = None,
        launcher: Optional[str] = None,
        nodes: Optional[int] = None,
        weak_scaling: bool = False,
        base_size: Optional[int] = None,
        runs: int = 1,
        output_path: Optional[Path] = None,
        arts_exec_args: Optional[str] = None,
    ) -> Tuple[List[BenchmarkResult], Optional[Path], Optional[Path]]:
        """Run benchmark with multiple thread configurations.

        Args:
            omp_threads: If specified, use this fixed thread count for OpenMP
                        instead of matching ARTS thread count. Useful for
                        comparing ARTS scaling against OpenMP at its optimal.
            launcher: Optional override for launcher (default: from arts.cfg).
                      For Slurm, this controls how the executable is invoked (via srun).
            nodes: Optional override for nodeCount (default: from arts.cfg).
            weak_scaling: If True, scale problem size with parallelism.
            base_size: Base problem size for weak scaling.
            runs: Number of times to run each configuration.
            output_path: If specified, create experiment directory for reproducibility.

        Returns:
            Tuple of (results, experiment_dir, json_path). Both paths are None if
            output_path was not specified.
        """
        results = []

        # Clean benchmark directory to avoid stale artifacts
        if self.clean:
            self.clean_benchmark(name)

        bench_path = self.benchmarks_dir / name
        run_args = self.get_run_args(bench_path, size)
        verify_tolerance = self.get_verify_tolerance(bench_path)

        # Determine effective config template:
        # 1. Use explicitly provided base_config
        # 2. Fall back to benchmark's own arts.cfg
        # 3. Fall back to default carts-benchmarks config
        effective_config = base_config
        if effective_config is None:
            candidate = bench_path / "arts.cfg"
            if candidate.exists():
                effective_config = candidate
            else:
                effective_config = DEFAULT_ARTS_CONFIG

        base_nodes = get_arts_cfg_int(effective_config, "nodeCount") or 1
        base_launcher = get_arts_cfg_str(effective_config, "launcher") or "ssh"
        desired_nodes = nodes if nodes is not None else base_nodes
        desired_launcher = launcher if launcher is not None else base_launcher

        # Create timestamped experiment directory ONCE per experiment
        # Both JSON results and artifacts will be stored together
        experiment_dir = None
        json_path = None
        if output_path:
            experiment_dir, json_path = create_experiment_dir(output_path)

        for threads in threads_list:
            # Generate arts.cfg with thread count, launcher, and node count
            # Pass nodes (CLI override) instead of desired_nodes to avoid unnecessary nodes= modification
            arts_cfg = generate_arts_config(
                effective_config, threads, counter_dir, desired_launcher, nodes
            )

            # Compute effective cflags (may include weak scaling size overrides)
            effective_cflags = cflags
            if weak_scaling and base_size:
                # Note: complexity is determined inside get_weak_scaling_cflags
                # based on BENCHMARK_SIZE_PARAMS lookup
                weak_cflags = get_weak_scaling_cflags(
                    name, base_size, threads, desired_nodes, base_parallelism=1
                )
                if weak_cflags:
                    effective_cflags = f"{cflags} {weak_cflags}".strip()

            # Set OMP_NUM_THREADS for OpenMP variant
            # Use omp_threads if specified, otherwise match ARTS threads
            actual_omp_threads = omp_threads if omp_threads else threads
            env = {"OMP_NUM_THREADS": str(actual_omp_threads)}
            if "OMP_WAIT_POLICY" not in os.environ:
                env["OMP_WAIT_POLICY"] = "ACTIVE"
            if "CARTS_BENCHMARKS_REPORT_INIT" not in os.environ:
                env["CARTS_BENCHMARKS_REPORT_INIT"] = "1"

            # Build both variants ONCE per thread config (with potentially different cflags per thread count)
            build_arts = self.build_benchmark(
                name, size, "arts", arts_cfg, effective_cflags, arts_exec_args)
            build_omp = self.build_benchmark(
                name, size, "openmp", None, effective_cflags)

            # Create config object
            config = BenchmarkConfig(
                arts_threads=threads,
                arts_nodes=desired_nodes,
                omp_threads=actual_omp_threads,
                launcher=desired_launcher,
            )

            # Create artifact directories if output specified (ONCE per config)
            config_dir = result_artifacts_dir = runs_dir = None
            artifact_paths: Dict[str, Optional[str]] = {}
            if experiment_dir:
                config_dir, result_artifacts_dir, runs_dir = create_config_directory(
                    experiment_dir, name, config
                )
                # Copy build artifacts ONCE per config
                artifact_paths = copy_build_artifacts(
                    bench_path, config_dir, result_artifacts_dir, arts_cfg_used=arts_cfg)

            # Run multiple times per configuration
            for run_num in range(1, runs + 1):
                # Create run directory if output specified
                run_dir = None
                arts_log_path: Optional[str] = None
                omp_log_path: Optional[str] = None

                if runs_dir:
                    run_dir = create_run_directory(runs_dir, run_num)
                    # Logs go to run directory
                    arts_log = run_dir / "arts.log"
                    omp_log = run_dir / "omp.log"
                    arts_log_path = str(arts_log)
                    omp_log_path = str(omp_log)
                elif self.debug >= 2:
                    # Fallback to benchmark logs directory for debug mode without output
                    logs_dir = bench_path / "logs"
                    run_suffix = f"_r{run_num}" if runs > 1 else ""
                    arts_log = logs_dir / f"arts_{threads}t{run_suffix}.log"
                    omp_log = logs_dir / f"omp_{threads}t{run_suffix}.log"
                else:
                    arts_log = None
                    omp_log = None

                # Run ARTS version - MUST pass artsConfig env var so runtime uses same config as compile
                # ARTS reads config from: 1) artsConfig env var, 2) ./arts.cfg in CWD
                # Without this, runtime may use wrong config (thread count mismatch)
                if build_arts.status == Status.PASS and build_arts.executable:
                    arts_env = dict(env)
                    arts_env["artsConfig"] = str(arts_cfg)
                    run_arts = self.run_benchmark(
                        build_arts.executable,
                        timeout,
                        env=arts_env,
                        launcher=desired_launcher,
                        node_count=desired_nodes,
                        threads=threads,
                        args=run_args,
                        log_file=arts_log,
                    )
                else:
                    run_arts = RunResult(
                        status=Status.SKIP,
                        duration_sec=0.0,
                        exit_code=-1,
                        stdout="",
                        stderr="Build failed",
                    )

                # Run OpenMP version (needs OMP_NUM_THREADS env var)
                if build_omp.status == Status.PASS and build_omp.executable:
                    run_omp = self.run_benchmark(
                        build_omp.executable,
                        timeout,
                        env=env,
                        args=run_args,
                        log_file=omp_log,
                    )
                else:
                    run_omp = RunResult(
                        status=Status.SKIP,
                        duration_sec=0.0,
                        exit_code=-1,
                        stdout="",
                        stderr="Build failed",
                    )

                # Calculate timing
                timing = self.calculate_timing(run_arts, run_omp)

                # Verify correctness for all thread counts
                verification = self.verify_correctness(
                    run_arts, run_omp, tolerance=verify_tolerance
                )

                # Collect artifacts from benchmark source directory
                artifacts = self.collect_artifacts(bench_path)

                # If output specified, update artifacts with paths in results directory
                if experiment_dir and run_dir:
                    # Update build artifacts from copied locations
                    if artifact_paths.get("arts_config"):
                        artifacts.arts_config = artifact_paths["arts_config"]
                    if artifact_paths.get("carts_metadata"):
                        artifacts.carts_metadata = artifact_paths["carts_metadata"]
                    if artifact_paths.get("arts_metadata_mlir"):
                        artifacts.arts_metadata_mlir = artifact_paths["arts_metadata_mlir"]
                    if artifact_paths.get("executable_arts"):
                        artifacts.executable_arts = artifact_paths["executable_arts"]
                    if artifact_paths.get("executable_omp"):
                        artifacts.executable_omp = artifact_paths["executable_omp"]
                    if result_artifacts_dir:
                        artifacts.build_dir = str(result_artifacts_dir)

                    # Update run artifacts
                    artifacts.run_dir = str(run_dir)
                    artifacts.arts_log = arts_log_path
                    artifacts.omp_log = omp_log_path

                    # Handle counter files - check explicit counter_dir or ARTS default ./counter
                    effective_counter_dir = None
                    if counter_dir and counter_dir.exists():
                        effective_counter_dir = counter_dir
                    else:
                        default_counter_dir = bench_path / "counters"
                        if default_counter_dir.exists():
                            effective_counter_dir = default_counter_dir

                    if effective_counter_dir and run_dir:
                        run_counters_dir = run_dir / "counters"
                        run_counters_dir.mkdir(exist_ok=True)
                        for counter_file in effective_counter_dir.glob("*.json"):
                            shutil.copy2(
                                counter_file, run_counters_dir / counter_file.name)
                        artifacts.counters_dir = str(run_counters_dir)
                        artifacts.counter_files = sorted(
                            str(f) for f in run_counters_dir.glob("*.json")
                        )
                        # Parse counter JSON for timing data
                        init_sec, e2e_sec = parse_counter_json(run_counters_dir)
                        run_arts.counter_init_sec = init_sec
                        run_arts.counter_e2e_sec = e2e_sec

                # Store with config and run number
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
                    total_duration_sec=0.0,  # Will be calculated later
                    size_params=self.get_size_params(bench_path, size),
                )

                results.append(result)

        return results, experiment_dir, json_path

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

            # Debug output level 2: write to log file (ARTS output can be huge)
            if self.debug >= 2:
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
                    self.console.print(f"[dim]  Log: {log_file}[/]")
                else:
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

        Uses the LAST checksum found in output to support benchmarks that
        print multiple intermediate checksums followed by a final combined one.
        """
        # Find all checksums and use the last one
        # This handles benchmarks with multiple kernels that print individual
        # checksums followed by a final combined checksum for verification
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

    def extract_kernel_timings(self, output: str) -> Dict[str, float]:
        """Extract kernel timing info from benchmark output.

        Parses lines like 'kernel.relu: 0.001234s' into a dict.
        """
        timings = {}
        pattern = r"^\s*kernel\.([^:]+):\s*([0-9.]+)s\s*$"
        for match in re.finditer(pattern, output, re.MULTILINE):
            kernel_name = match.group(1).strip()
            kernel_time = float(match.group(2))
            timings[kernel_name] = kernel_time
        return timings

    def extract_e2e_timings(self, output: str) -> Dict[str, float]:
        """Extract end-to-end timing info from benchmark output.

        Parses lines like 'e2e.gemm: 0.001234567s' into a dict.
        """
        timings = {}
        pattern = r"^\s*e2e\.([^:]+):\s*([0-9.]+)s\s*$"
        for match in re.finditer(pattern, output, re.MULTILINE):
            name = match.group(1).strip()
            time_sec = float(match.group(2))
            timings[name] = time_sec
        return timings

    def extract_init_timings(self, output: str) -> Dict[str, float]:
        """Extract runtime initialization timing info from benchmark output.

        Parses lines like 'init.omp: 0.000123456s' or 'init.arts: 0.123456789s'.
        """
        timings = {}
        pattern = r"^\s*init\.([^:]+):\s*([0-9.]+)s\s*$"
        for match in re.finditer(pattern, output, re.MULTILINE):
            name = match.group(1).strip()
            time_sec = float(match.group(2))
            timings[name] = time_sec
        return timings

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

        # Default counter directory if not specified - ensures counter collection always works
        if counter_dir is None:
            counter_dir = bench_path / "counters"

        # Generate config with overrides only if values actually differ from base
        need_generated = False
        if threads_override is not None and threads_override != base_threads:
            need_generated = True
        if nodes_override is not None and nodes_override != base_nodes:
            need_generated = True
        if launcher_override is not None and launcher_override != base_launcher:
            need_generated = True
        if counter_dir is not None:
            need_generated = True

        effective_arts_cfg: Path
        if need_generated:
            effective_arts_cfg = generate_arts_config(
                effective_config, desired_threads, counter_dir, desired_launcher, nodes_override
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

        # Build OpenMP version
        if phase_callback:
            phase_callback(Phase.BUILD_OMP)
        build_omp = self.build_benchmark(name, size, "openmp", None)
        if partial_results is not None:
            partial_results["build_omp"] = build_omp

        # Setup log files for debug=2
        logs_dir = bench_path / "logs"
        arts_log = logs_dir / "arts.log" if self.debug >= 2 else None
        omp_log = logs_dir / "omp.log" if self.debug >= 2 else None
        run_args = self.get_run_args(bench_path, size)
        verify_tolerance = self.get_verify_tolerance(bench_path)

        # Ensure init timing is reported and align OMP wait/spin policy with ARTS.
        common_env: Dict[str, str] = {}
        if "CARTS_BENCHMARKS_REPORT_INIT" not in os.environ:
            common_env["CARTS_BENCHMARKS_REPORT_INIT"] = "1"

        # Run ARTS version - pass artsConfig env var so runtime uses the effective config.
        if phase_callback:
            phase_callback(Phase.RUN_ARTS)
        if build_arts.status == Status.PASS and build_arts.executable:
            arts_env = dict(common_env)
            if effective_arts_cfg:
                arts_env["artsConfig"] = str(effective_arts_cfg)
            run_arts = self.run_benchmark(
                build_arts.executable,
                timeout,
                env=arts_env,
                launcher=desired_launcher,
                node_count=desired_nodes,
                threads=desired_threads,
                args=run_args,
                log_file=arts_log,
            )
        else:
            run_arts = RunResult(
                status=Status.SKIP,
                duration_sec=0.0,
                exit_code=-1,
                stdout="",
                stderr="Build failed",
            )

        # Parse counter JSON for timing data
        # Check explicit counter_dir first, then fall back to ARTS default ./counter location
        effective_counter_dir = None
        if counter_dir and counter_dir.exists():
            effective_counter_dir = counter_dir
        else:
            default_counter_dir = bench_path / "counters"
            if default_counter_dir.exists():
                effective_counter_dir = default_counter_dir
        if effective_counter_dir:
            init_sec, e2e_sec = parse_counter_json(effective_counter_dir)
            run_arts.counter_init_sec = init_sec
            run_arts.counter_e2e_sec = e2e_sec

        if partial_results is not None:
            partial_results["run_arts"] = run_arts

        # Run OpenMP version
        if phase_callback:
            phase_callback(Phase.RUN_OMP)
        if build_omp.status == Status.PASS and build_omp.executable:
            omp_env = dict(common_env)
            actual_omp_threads = (
                omp_threads_override if omp_threads_override is not None else desired_threads
            )
            omp_env["OMP_NUM_THREADS"] = str(actual_omp_threads)
            if "OMP_WAIT_POLICY" not in os.environ:
                omp_env["OMP_WAIT_POLICY"] = "ACTIVE"
            run_omp = self.run_benchmark(
                build_omp.executable,
                timeout,
                env=omp_env,
                args=run_args,
                log_file=omp_log,
            )
        else:
            run_omp = RunResult(
                status=Status.SKIP,
                duration_sec=0.0,
                exit_code=-1,
                stdout="",
                stderr="Build failed",
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
        if effective_arts_cfg:
            artifacts.arts_config = str(effective_arts_cfg)
        if arts_log is not None:
            artifacts.arts_log = str(arts_log)
        if omp_log is not None:
            artifacts.omp_log = str(omp_log)

        total_duration = time.time() - start_time

        # Default config for single runs (no thread sweep)
        config = BenchmarkConfig(
            arts_threads=desired_threads,
            arts_nodes=desired_nodes,
            omp_threads=(
                omp_threads_override if omp_threads_override is not None else desired_threads
            ),
            launcher=desired_launcher,
        )

        return BenchmarkResult(
            name=name,
            suite=suite,
            size=size,
            config=config,
            run_number=1,
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
    ) -> List[BenchmarkResult]:
        """Run benchmark suite.
        """
        results_dict: Dict[str, BenchmarkResult] = {}
        results_list: List[BenchmarkResult] = []
        start_time = time.time()

        if self.quiet:
            # Quiet mode - no live display
            for bench in benchmarks:
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
                benchmarks, results_dict, current_bench[0], elapsed, phase, current_partial[0]))

        with Live(
            create_live_display(benchmarks, results_dict, None, 0, None, None),
            console=self.console,
            refresh_per_second=4,
        ) as live:
            for bench in benchmarks:
                current_bench[0] = bench
                current_partial[0] = {}
                # Update display to show current benchmark as in-progress
                elapsed = time.time() - start_time
                live.update(create_live_display(
                    benchmarks, results_dict, bench, elapsed, Phase.BUILD_ARTS, current_partial[0]))

                # Run benchmark with phase callback and partial results storage
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
                )

                # Update results and refresh display
                results_dict[bench] = result
                results_list.append(result)
                current_bench[0] = None
                current_partial[0] = None
                elapsed = time.time() - start_time
                live.update(create_live_display(
                    benchmarks, results_dict, None, elapsed, None, None))

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

    def clean_benchmark(self, name: str) -> bool:
        """Clean build artifacts for a benchmark."""
        bench_path = self.benchmarks_dir / name

        if not bench_path.exists():
            return False

        cleaned = False

        # Remove directories (legacy from make-based builds)
        for dirname in ["build", "logs", "counters"]:
            dirpath = bench_path / dirname
            if dirpath.exists():
                shutil.rmtree(dirpath)
                cleaned = True

        # Remove files
        for pattern in ["*.mlir", "*.ll", "*.o", "*-metadata.json", ".carts-metadata.json"]:
            for f in bench_path.glob(pattern):
                f.unlink()
                cleaned = True

        # Remove executables
        for exe in bench_path.glob("*_arts"):
            if exe.is_file():
                exe.unlink()
                cleaned = True
        for exe in bench_path.glob("*_omp"):
            if exe.is_file():
                exe.unlink()
                cleaned = True

        return cleaned


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


def generate_markdown_report_from_json(
    data: Dict[str, Any],
    *,
    input_path: Path,
    output_md: Path,
) -> Tuple[Path, Path, Path]:
    """Generate a Markdown+SVG report from an exported results JSON."""
    output_md.parent.mkdir(parents=True, exist_ok=True)
    figures_dir = output_md.parent / f"{output_md.stem}_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    chart_svg_path = output_md.parent / \
        f"{output_md.stem}_init_e2e_small_multiples.svg"
    out_speedup_svg = output_md.parent / \
        f"{output_md.stem}_speedup_bar_chart.svg"

    results = data.get("results", [])
    if not results:
        output_md.write_text("# CARTS Benchmarks Report\n\n(no results)\n")
        _write_small_multiples_svg([], chart_svg_path)
        _write_speedup_bar_chart([], out_speedup_svg)
        return chart_svg_path, out_speedup_svg, figures_dir

    # Group by config (name + threads + nodes + omp_threads + launcher)
    grouped: Dict[Tuple, List[Dict[str, Any]]] = {}
    for r in results:
        cfg = r.get("config", {})
        key = (
            r.get("name", ""),
            cfg.get("arts_threads"),
            cfg.get("arts_nodes"),
            cfg.get("omp_threads"),
            cfg.get("launcher"),
        )
        grouped.setdefault(key, []).append(r)

    def mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
        if not vals:
            return None, None
        if len(vals) == 1:
            return vals[0], 0.0
        from statistics import mean, stdev
        return mean(vals), stdev(vals)

    def sum_map(m: Any) -> Optional[float]:
        if not isinstance(m, dict) or not m:
            return None
        return sum(float(v) for v in m.values())

    def fmt_mean_std(m: Optional[float], s: Optional[float]) -> str:
        if m is None:
            return "-"
        if s is None:
            return f"{m:.6f}"
        return f"{m:.6f} ± {s:.6f}"

    def fmt_s(m: Optional[float]) -> str:
        return "-" if m is None else f"{m:.6f}"

    def kv_lines(m: Any) -> str:
        if not isinstance(m, dict) or not m:
            return "-"
        return ", ".join(f"`{k}`={float(v):.6f}s" for k, v in sorted(m.items()))

    panel_groups: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []

    for key, runs in sorted(grouped.items(), key=lambda kv: kv[0][0]):
        name, arts_t, arts_n, omp_t, launcher_v = key
        first = runs[0]
        timing0 = first.get("timing", {})
        run_arts0 = first.get("run_arts", {})
        run_omp0 = first.get("run_omp", {})
        speedup_vals = [
            float(r.get("timing", {}).get("speedup") or 0.0)
            for r in runs
            if float(r.get("timing", {}).get("speedup") or 0.0) > 0.0
        ]
        sp_mean, sp_std = mean_std(speedup_vals)

        arts_init_vals = [
            float(r.get("run_arts", {}).get("init_timings", {}).get("arts"))
            for r in runs
            if r.get("run_arts", {}).get("init_timings", {}).get("arts") is not None
        ]
        omp_init_vals = [
            float(r.get("run_omp", {}).get("init_timings", {}).get("omp"))
            for r in runs
            if r.get("run_omp", {}).get("init_timings", {}).get("omp") is not None
        ]

        arts_e2e_vals = [
            sum_map(r.get("run_arts", {}).get("e2e_timings")) for r in runs]
        arts_e2e_vals = [v for v in arts_e2e_vals if v is not None]
        omp_e2e_vals = [
            sum_map(r.get("run_omp", {}).get("e2e_timings")) for r in runs]
        omp_e2e_vals = [v for v in omp_e2e_vals if v is not None]

        arts_kernel_vals = [
            sum_map(r.get("run_arts", {}).get("kernel_timings")) for r in runs]
        arts_kernel_vals = [v for v in arts_kernel_vals if v is not None]
        omp_kernel_vals = [
            sum_map(r.get("run_omp", {}).get("kernel_timings")) for r in runs]
        omp_kernel_vals = [v for v in omp_kernel_vals if v is not None]

        arts_init_mean, arts_init_std = mean_std(arts_init_vals)
        omp_init_mean, omp_init_std = mean_std(omp_init_vals)
        arts_e2e_mean, arts_e2e_std = mean_std(arts_e2e_vals)
        omp_e2e_mean, omp_e2e_std = mean_std(omp_e2e_vals)
        arts_kernel_mean, arts_kernel_std = mean_std(arts_kernel_vals)
        omp_kernel_mean, omp_kernel_std = mean_std(omp_kernel_vals)

        arts_other = None if arts_e2e_mean is None or arts_kernel_mean is None else max(
            arts_e2e_mean - arts_kernel_mean, 0.0)
        omp_other = None if omp_e2e_mean is None or omp_kernel_mean is None else max(
            omp_e2e_mean - omp_kernel_mean, 0.0)

        arts_total = float((arts_init_mean or 0.0) + (arts_e2e_mean or 0.0))
        omp_total = float((omp_init_mean or 0.0) + (omp_e2e_mean or 0.0))
        scale_max = max(arts_total, omp_total, 1e-12)

        arts_segments = [
            ("__init__", float(arts_init_mean or 0.0)),
            ("__other__", float(arts_other or 0.0)),
            ("__kernel__", float(arts_kernel_mean or 0.0)),
        ]
        omp_segments = [
            ("__init__", float(omp_init_mean or 0.0)),
            ("__other__", float(omp_other or 0.0)),
            ("__kernel__", float(omp_kernel_mean or 0.0)),
        ]
        panel_groups.append({
            "benchmark": name,
            "speedup": sp_mean,
            "speedup_basis": timing0.get("speedup_basis", "total"),
            "scale_max": scale_max,
            "arts_segments": arts_segments,
            "omp_segments": omp_segments,
            "arts_total": sum(v for _, v in arts_segments),
            "omp_total": sum(v for _, v in omp_segments),
        })

        fig_svg = figures_dir / \
            f"{_sanitize_filename(name)}_{arts_t}t_{arts_n}n.svg"
        _write_per_benchmark_timeline_svg(
            name,
            dict(run_arts0.get("init_timings") or {}),
            dict(run_arts0.get("e2e_timings") or {}),
            dict(run_arts0.get("kernel_timings") or {}),
            dict(run_omp0.get("init_timings") or {}),
            dict(run_omp0.get("e2e_timings") or {}),
            dict(run_omp0.get("kernel_timings") or {}),
            speedup=sp_mean,
            speedup_basis=str(timing0.get("speedup_basis", "total")),
            svg_path=fig_svg,
        )

        report_rows.append({
            "name": name,
            "benchmark": name,
            "arts_threads": arts_t,
            "arts_nodes": arts_n,
            "omp_threads": omp_t,
            "launcher": launcher_v,
            "arts_status": run_arts0.get("status", ""),
            "omp_status": run_omp0.get("status", ""),
            "arts_init": (arts_init_mean, arts_init_std),
            "arts_e2e": (arts_e2e_mean, arts_e2e_std),
            "arts_kernel": (arts_kernel_mean, arts_kernel_std),
            "arts_other": arts_other,
            "omp_init": (omp_init_mean, omp_init_std),
            "omp_e2e": (omp_e2e_mean, omp_e2e_std),
            "omp_kernel": (omp_kernel_mean, omp_kernel_std),
            "omp_other": omp_other,
            "speedup_mean": sp_mean,
            "speedup_std": sp_std,
            "speedup_basis": timing0.get("speedup_basis", "total"),
            "verification_note": first.get("verification", {}).get("note", ""),
            "correct": bool(first.get("verification", {}).get("correct", False)),
            "arts_init_map": run_arts0.get("init_timings") or {},
            "arts_e2e_map": run_arts0.get("e2e_timings") or {},
            "arts_kernel_map": run_arts0.get("kernel_timings") or {},
            "omp_init_map": run_omp0.get("init_timings") or {},
            "omp_e2e_map": run_omp0.get("e2e_timings") or {},
            "omp_kernel_map": run_omp0.get("kernel_timings") or {},
            "figure": fig_svg.name,
        })

    _write_small_multiples_svg(panel_groups, chart_svg_path)
    _write_speedup_bar_chart(report_rows, out_speedup_svg)

    meta = data.get("metadata", {})
    lines: List[str] = []
    lines.append("# CARTS Benchmarks Report")
    lines.append("")
    lines.append(f"- Source JSON: `{input_path}`")
    lines.append(f"- Timestamp: {meta.get('timestamp', '-')}")
    lines.append(f"- Size: {meta.get('size', '-')}")
    lines.append(f"- Runs per config: {meta.get('runs_per_config', '-')}")
    lines.append(f"- Figures: `{figures_dir.name}/`")
    lines.append("")
    lines.append("## Summary Figure (local scale per benchmark)")
    lines.append("")
    lines.append(f"![init+e2e small multiples]({chart_svg_path.name})")
    lines.append("")
    lines.append("## Speedup Figure (local scale per benchmark)")
    lines.append("")
    lines.append(f"![speedup small multiples]({out_speedup_svg.name})")
    lines.append("")
    lines.append("## Results (mean ± std, seconds)")
    lines.append("")
    lines.append("| Benchmark | t | n | ARTS status | OMP status | ARTS init | ARTS e2e | ARTS kernel | ARTS other | OMP init | OMP e2e | OMP kernel | OMP other | Speedup | Correct |")
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for r in report_rows:
        speed_txt = "-"
        if r.get("speedup_mean") is not None:
            sp_m = float(r["speedup_mean"])
            sp_s = r.get("speedup_std")
            if sp_s is None:
                speed_txt = f"{sp_m:.2f}x ({r['speedup_basis']})"
            else:
                speed_txt = f"{sp_m:.2f} ± {float(sp_s):.2f}x ({r['speedup_basis']})"
        if r["verification_note"] == "Verification disabled":
            correct_txt = "N/A"
        else:
            correct_txt = "YES" if r["correct"] else "NO"
        lines.append(
            f"| {r['name']} | {r['arts_threads']} | {r['arts_nodes']} | {r['arts_status']} | {r['omp_status']} | "
            f"{fmt_mean_std(*r['arts_init'])} | {fmt_mean_std(*r['arts_e2e'])} | {fmt_mean_std(*r['arts_kernel'])} | {fmt_s(r['arts_other'])} | "
            f"{fmt_mean_std(*r['omp_init'])} | {fmt_mean_std(*r['omp_e2e'])} | {fmt_mean_std(*r['omp_kernel'])} | {fmt_s(r['omp_other'])} | "
            f"{speed_txt} | {correct_txt} |"
        )
    lines.append("")
    lines.append("## Per-Benchmark Figures")
    lines.append("")
    for r in report_rows:
        lines.append(f"### {r['name']}")
        lines.append("")
        lines.append(f"![{r['name']}]({figures_dir.name}/{r['figure']})")
        lines.append("")
        lines.append(
            f"- Config: `threads={r['arts_threads']}`, `nodes={r['arts_nodes']}`, `omp_threads={r['omp_threads']}`, `launcher={r['launcher']}`")
        lines.append(f"- ARTS init breakdown: {kv_lines(r['arts_init_map'])}")
        lines.append(f"- ARTS e2e breakdown: {kv_lines(r['arts_e2e_map'])}")
        lines.append(
            f"- ARTS kernel breakdown: {kv_lines(r['arts_kernel_map'])}")
        lines.append(f"- OMP init breakdown: {kv_lines(r['omp_init_map'])}")
        lines.append(f"- OMP e2e breakdown: {kv_lines(r['omp_e2e_map'])}")
        lines.append(
            f"- OMP kernel breakdown: {kv_lines(r['omp_kernel_map'])}")
        lines.append("")

    output_md.write_text("\n".join(lines))
    return chart_svg_path, out_speedup_svg, figures_dir


def create_live_table(
    benchmarks: List[str],
    results: Dict[str, BenchmarkResult],
    in_progress: Optional[str] = None,
    current_phase: Optional[Phase] = None,
    current_partial: Optional[Dict[str, Any]] = None,
) -> Table:
    """Create a live-updating table showing benchmark progress."""
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
    for bench in benchmarks:
        if bench in results:
            # Completed - show full results
            r = results[bench]

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

            # Run status with e2e time
            if r.run_arts.status == Status.PASS:
                if arts_e2e is not None:
                    run_arts = f"{status_symbol(r.run_arts.status)} {arts_e2e_str}"
                else:
                    arts_kernel, arts_kernel_str = format_kernel_time(
                        r.run_arts)
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
                # Show ARTS build time if available
                if current_partial and "build_arts" in current_partial:
                    build_arts_result = current_partial["build_arts"]
                    if build_arts_result.status == Status.PASS:
                        build = f"[green]✓[/] {build_arts_result.duration_sec:.1f}s [yellow]⏳ OMP...[/]"
                    else:
                        build = f"[red]✗[/] {build_arts_result.status.value} [yellow]⏳ OMP...[/]"
                else:
                    build = "[yellow]⏳ OMP...[/]"
                run_arts = "[dim]-[/]"
                run_omp = "[dim]-[/]"
                correct = "[dim]-[/]"
                speedup = "[dim]-[/]"
            elif current_phase == Phase.RUN_ARTS:
                # Show build time if available
                if current_partial and "build_arts" in current_partial and "build_omp" in current_partial:
                    build_arts_result = current_partial["build_arts"]
                    build_omp_result = current_partial["build_omp"]
                    if build_arts_result.status == Status.PASS and build_omp_result.status == Status.PASS:
                        build = f"[green]✓[/] {build_arts_result.duration_sec + build_omp_result.duration_sec:.1f}s"
                    else:
                        build = f"[red]✗[/] {build_arts_result.status.value}/{build_omp_result.status.value}"
                else:
                    build = "[green]✓[/]"
                run_arts = "[yellow]⏳ running...[/]"
                run_omp = "[dim]-[/]"
                correct = "[dim]-[/]"
                speedup = "[dim]-[/]"
            elif current_phase == Phase.RUN_OMP:
                # Show build time if available
                if current_partial and "build_arts" in current_partial and "build_omp" in current_partial:
                    build_arts_result = current_partial["build_arts"]
                    build_omp_result = current_partial["build_omp"]
                    if build_arts_result.status == Status.PASS and build_omp_result.status == Status.PASS:
                        build = f"[green]✓[/] {build_arts_result.duration_sec + build_omp_result.duration_sec:.1f}s"
                    else:
                        build = f"[red]✗[/] {build_arts_result.status.value}/{build_omp_result.status.value}"
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
    results: Dict[str, BenchmarkResult],
    total: int,
    elapsed: float,
) -> Text:
    """Create a one-line summary for live display."""
    passed = sum(1 for r in results.values() if r.run_arts.status ==
                 Status.PASS and r.verification.correct)
    failed = sum(1 for r in results.values() if r.run_arts.status in (Status.FAIL, Status.CRASH) or
                 (r.run_arts.status == Status.PASS and not r.verification.correct))
    pending = total - len(results)

    text = Text()
    text.append(f"\u2713 {passed} passed  ", style="green")
    text.append(f"\u2717 {failed} failed  ", style="red")
    text.append(f"\u25cb {pending} pending  ", style="dim")
    text.append(f"\u23f1 {elapsed:.1f}s", style="dim")
    return text


def create_live_display(
    benchmarks: List[str],
    results: Dict[str, BenchmarkResult],
    in_progress: Optional[str],
    elapsed: float,
    current_phase: Optional[Phase] = None,
    current_partial: Optional[Dict[str, Any]] = None,
) -> Group:
    """Create the complete live display (table + summary)."""
    table = create_live_table(
        benchmarks, results, in_progress, current_phase, current_partial)
    summary = create_live_summary(results, len(benchmarks), elapsed)
    return Group(table, summary)


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
        pass
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
        pass

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
            pass
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
            pass

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
        arts_init_times = []
        omp_init_times = []
        arts_e2e_times = []
        omp_e2e_times = []
        # Keep kernel times as optional context
        arts_kernel_times = []
        omp_kernel_times = []
        for r in runs:
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
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Export results to JSON file"),
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
    nodes: Optional[int] = typer.Option(
        None, "--nodes", "-n", help="Override nodeCount from config (uses first N nodes)"),
    weak_scaling: bool = typer.Option(
        False, "--weak-scaling", help="Enable weak scaling: auto-scale problem size with parallelism"),
    base_size: Optional[int] = typer.Option(
        None, "--base-size", help="Base problem size for weak scaling (at base parallelism)"),
    cflags: Optional[str] = typer.Option(
        None, "--cflags", help="Additional CFLAGS: '-DNI=500 -DNJ=500'"),
    arts_exec_args: Optional[str] = typer.Option(
        None, "--arts-exec-args", help="Extra carts execute args (e.g., '--partition-fallback=fine')"),
    debug_level: int = typer.Option(
        0, "--debug", "-d", help="Debug level: 0=off, 1=commands, 2=full output"),
    counters: int = typer.Option(
        0, "--counters", help="Counter level: 0=off, 1=artsid metrics, 2=deep captures. Requires ARTS built with matching --counters level."),
    counter_dir: Optional[Path] = typer.Option(
        None, "--counter-dir", help="Directory for ARTS counter output (n0_t*.json files)"),
    runs: int = typer.Option(
        1, "--runs", "-r", help="Number of times to run each benchmark configuration (for statistical significance)"),
    report: bool = typer.Option(
        False, "--report", help="Generate a Markdown+SVG report artifact after running"),
    report_dir: Optional[Path] = typer.Option(
        None, "--report-dir", help="Directory for report outputs (default: alongside JSON output or CWD)"),
):
    """Run benchmarks with verification and timing."""
    verify = not no_verify
    clean = not no_clean
    runner = BenchmarkRunner(
        console, verbose=verbose, quiet=quiet, trace=trace, clean=clean, debug=debug_level)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parse thread specification
    threads_list = None
    if threads:
        threads_list = parse_threads(threads)

    # Discover or use provided benchmarks
    if benchmarks:
        bench_list = list(benchmarks)
    else:
        bench_list = runner.discover_benchmarks(suite)

    if not bench_list:
        console.print("[yellow]No benchmarks found.[/]")
        raise typer.Exit(1)

    # Setup counter collection
    if counters > 0:
        # Check if ARTS was built with counter support
        arts_level = check_arts_counter_support()
        if arts_level < counters:
            console.print(
                f"[yellow]Warning: ARTS built with counters={arts_level}, requested counters={counters}[/]")
            console.print(
                f"[dim]Rebuild ARTS: carts build --arts --counters={counters}[/]")
        if not counter_dir:
            counter_dir = Path(f"./counters/{size}")
    if counter_dir:
        counter_dir.mkdir(parents=True, exist_ok=True)

    # Print header
    if not quiet:
        console.print(f"\n[bold]CARTS Benchmark Runner v{VERSION}[/]")
        console.print("\u2501" * 30)

        # Show benchmark runner config
        config_items = [f"size={size}", f"timeout={timeout}s",
                        f"verify={verify}", f"clean={clean}"]
        if threads_list:
            config_items.append(f"threads={threads}")
        if launcher is not None:
            config_items.append(f"launcher={launcher}")
        if nodes is not None:
            config_items.append(f"nodes={nodes}")
        if runs > 1:
            config_items.append(f"runs={runs}")
        if cflags:
            config_items.append(f"cflags={cflags}")
        if arts_exec_args:
            config_items.append(f"arts-exec-args={arts_exec_args}")
        if debug_level > 0:
            config_items.append(f"debug={debug_level}")
        if counters > 0:
            config_items.append(
                f"counters={counters} ({COUNTER_LEVELS.get(counters, 'unknown')})")
        console.print(f"Config: {', '.join(config_items)}")

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

                # Only show node list if it's not just localhost
                if arts_nodes_list != ["localhost"]:
                    node_display = ','.join(arts_nodes_list[:3])
                    if len(arts_nodes_list) > 3:
                        node_display += "..."
                    arts_config_items.append(
                        f"arts-node-list=[{node_display}]")

                console.print(
                    f"ARTS Config ({config_source}): {', '.join(arts_config_items)}")

        console.print(f"Benchmarks: {len(bench_list)}\n")

    # Run benchmarks
    start_time = time.time()
    experiment_dir: Optional[Path] = None
    actual_json_path: Optional[Path] = None

    threads_override: Optional[int] = None
    if threads_list and len(bench_list) > 1:
        if len(threads_list) != 1:
            console.print(
                "[red]Error:[/] For multiple benchmarks, `--threads` must specify a single value (e.g., `--threads 8`)."
            )
            raise typer.Exit(2)
        threads_override = int(threads_list[0])
        threads_list = None

    try:
        if threads_list and len(bench_list) == 1 and len(threads_list) > 1:
            # Thread sweep mode for single benchmark
            results, experiment_dir, actual_json_path = runner.run_with_thread_sweep(
                bench_list[0],
                size,
                threads_list,
                arts_config,
                cflags or "",
                counter_dir,
                timeout,
                omp_threads,
                launcher,
                nodes,
                weak_scaling,
                base_size,
                runs,
                output,  # Pass output_path for artifact organization
                arts_exec_args,
            )
        else:
            # Standard mode
            if threads_list and len(threads_list) == 1:
                threads_override = int(threads_list[0])
                threads_list = None
            results = runner.run_all(
                bench_list,
                size=size,
                timeout=timeout,
                verify=verify,
                arts_config=arts_config,
                threads_override=threads_override,
                nodes_override=nodes,
                launcher_override=launcher,
                omp_threads_override=omp_threads,
                counter_dir=counter_dir,
                arts_exec_args=arts_exec_args,
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

    # Always export a JSON artifact (for paper-quality reporting + reproducibility).
    # If --output is not provided, write `benchmark_results_<timestamp>.json` to CWD.
    json_output_path = (
        (actual_json_path if actual_json_path else output)
        if output
        else (get_benchmarks_dir() / "results" / f"benchmark_results_{run_timestamp}.json")
    )

    artifacts_dir_name = "." if experiment_dir else None
    export_json(
        results,
        json_output_path,
        size,
        total_duration,
        threads_list,
        cflags,
        launcher,
        weak_scaling,
        base_size,
        runs,
        artifacts_dir_name,
        fixed_threads=threads_override,
        fixed_nodes=nodes,
        omp_threads_override=omp_threads,
        arts_config_override=str(arts_config) if arts_config else None,
    )

    if report:
        if report_dir:
            out_dir = report_dir
        elif experiment_dir:
            out_dir = experiment_dir
        elif output:
            out_dir = output.parent
        else:
            out_dir = get_benchmarks_dir() / "reports"

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

    # Show JSON artifact path last (after live table + optional report).
    if not quiet:
        console.print()
        if experiment_dir:
            console.print(f"[green]Experiment folder:[/] {experiment_dir}")
        console.print(f"[green]Results JSON:[/]      {json_output_path}")

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

    console.print(
        f"\n[bold]Building {len(bench_list)} benchmarks[/] (size={size})\n")

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

    console.print("\n[bold green]Build complete![/]")


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

    console.print(f"\n[bold]Cleaning {len(bench_list)} benchmarks[/]\n")

    cleaned = 0
    for bench in bench_list:
        if runner.clean_benchmark(bench):
            console.print(f"  [green]\u2713[/] {bench}")
            cleaned += 1
        else:
            console.print(f"  [dim]\u25cb[/] {bench} (nothing to clean)")

    console.print(f"\n[bold green]Cleaned {cleaned} benchmarks![/]")


@app.command()
def report(
    input: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Input JSON file"),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, bars, detailed, json, csv"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file"),
):
    """View or export benchmark results."""
    if not input:
        # Look for most recent results file in common locations (CWD and
        # carts-benchmarks/results).
        candidates: List[Path] = []
        candidates.extend(Path(".").glob("benchmark_results_*.json"))
        candidates.extend(
            (get_benchmarks_dir() / "results").glob("benchmark_results_*.json"))
        candidates = [p for p in candidates if p.exists()]
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            input = candidates[0]
        else:
            console.print(
                "[yellow]No results file found. Run benchmarks first.[/]")
            raise typer.Exit(1)

    with open(input) as f:
        data = json.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = output.parent if output else input.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    report_md = report_dir / f"benchmark_report_{timestamp}.md"
    report_chart, report_speedup, figures_dir = generate_markdown_report_from_json(
        data,
        input_path=input,
        output_md=report_md,
    )

    if format == "json":
        if output:
            with open(output, "w") as f:
                json.dump(data, f, indent=2)
        else:
            console.print_json(data=data)
    elif format == "bars":
        results = data.get("results", [])
        if not results:
            console.print("[yellow]No results to display.[/]")
            raise typer.Exit(1)

        max_total = 0.0
        for r in results:
            timing = r.get("timing", {})
            for init_key, e2e_key in (
                ("arts_init_sec", "arts_e2e_sec"),
                ("omp_init_sec", "omp_e2e_sec"),
            ):
                init_val = float(timing.get(init_key) or 0.0)
                e2e_val = float(timing.get(e2e_key) or 0.0)
                max_total = max(max_total, init_val + e2e_val)

        chart = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        chart.add_column("Benchmark", style="cyan", no_wrap=True)
        chart.add_column("ARTS init+e2e")
        chart.add_column("OMP init+e2e")
        for r in results:
            timing = r.get("timing", {})
            chart.add_row(
                r.get("name", ""),
                _render_init_e2e_bar(timing.get(
                    "arts_init_sec"), timing.get("arts_e2e_sec"), max_total),
                _render_init_e2e_bar(timing.get(
                    "omp_init_sec"), timing.get("omp_e2e_sec"), max_total),
            )
        chart.caption = "[dim]Bar colors: init=red, e2e=green[/]"
        console.print(chart)
    elif format == "detailed":
        results = data.get("results", [])
        if not results:
            console.print("[yellow]No results to display.[/]")
            raise typer.Exit(1)

        def _sum_and_count(d: Any) -> Tuple[Optional[float], int]:
            if not isinstance(d, dict) or not d:
                return None, 0
            vals = []
            for v in d.values():
                try:
                    vals.append(float(v))
                except Exception:
                    pass
            return (sum(vals), len(vals)) if vals else (None, 0)

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
        table.add_column("Benchmark", style="cyan", no_wrap=True)
        table.add_column("ARTS init", justify="right")
        table.add_column("ARTS e2e", justify="right")
        table.add_column("ARTS kernel", justify="right")
        table.add_column("OMP init", justify="right")
        table.add_column("OMP e2e", justify="right")
        table.add_column("OMP kernel", justify="right")
        table.add_column("Speedup", justify="right")

        for r in results:
            timing = r.get("timing", {})
            arts = r.get("run_arts", {})
            omp = r.get("run_omp", {})

            arts_init, arts_init_n = _sum_and_count(arts.get("init_timings"))
            arts_e2e, arts_e2e_n = _sum_and_count(arts.get("e2e_timings"))
            arts_kernel, arts_kernel_n = _sum_and_count(
                arts.get("kernel_timings"))
            omp_init, omp_init_n = _sum_and_count(omp.get("init_timings"))
            omp_e2e, omp_e2e_n = _sum_and_count(omp.get("e2e_timings"))
            omp_kernel, omp_kernel_n = _sum_and_count(
                omp.get("kernel_timings"))

            def _fmt(v: Optional[float], n: int) -> str:
                if v is None:
                    return "[dim]-[/]"
                if n > 1:
                    return f"{v:.4f}s [{n}]"
                return f"{v:.4f}s"

            sp = timing.get("speedup")
            basis = timing.get("speedup_basis", "total")
            if sp is None or sp == 0:
                sp_str = "[dim]-[/]"
            else:
                sp_val = float(sp)
                sp_str = f"{sp_val:.2f}x ({basis})"

            table.add_row(
                r.get("name", ""),
                _fmt(arts_init, arts_init_n),
                _fmt(arts_e2e, arts_e2e_n),
                _fmt(arts_kernel, arts_kernel_n),
                _fmt(omp_init, omp_init_n),
                _fmt(omp_e2e, omp_e2e_n),
                _fmt(omp_kernel, omp_kernel_n),
                sp_str,
            )

        console.print(table)
    elif format == "csv":
        import csv
        rows = []
        for r in data["results"]:
            timing = r["timing"]
            rows.append({
                "name": r["name"],
                "suite": r["suite"],
                "size": r["size"],
                "build_arts": r["build_arts"]["status"],
                "build_omp": r["build_omp"]["status"],
                "run_arts": r["run_arts"]["status"],
                "run_omp": r["run_omp"]["status"],
                "arts_time": timing.get("arts_time_sec"),
                "omp_time": timing.get("omp_time_sec"),
                "speedup_basis": timing.get("speedup_basis", "total"),
                "arts_kernel": timing.get("arts_kernel_sec"),
                "omp_kernel": timing.get("omp_kernel_sec"),
                "arts_init": timing.get("arts_init_sec"),
                "omp_init": timing.get("omp_init_sec"),
                "arts_e2e": timing.get("arts_e2e_sec"),
                "omp_e2e": timing.get("omp_e2e_sec"),
                "arts_total": timing.get("arts_total_sec"),
                "omp_total": timing.get("omp_total_sec"),
                "speedup": timing.get("speedup"),
                "correct": r["verification"]["correct"],
            })

        if output:
            with open(output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            console.print(f"[dim]Results exported to: {output}[/]")
        else:
            console.print("[yellow]Use --output to export CSV[/]")
    else:
        # Table format
        console.print(f"\n[bold]Results from {input}[/]\n")
        console.print(f"Timestamp: {data['metadata']['timestamp']}")
        console.print(f"Size: {data['metadata']['size']}")
        console.print()

        summary = data["summary"]
        console.print(Panel(
            f"[green]\u2713 {summary['passed']}[/] passed  "
            f"[red]\u2717 {summary['failed']}[/] failed  "
            f"[dim]\u25cb {summary['skipped']}[/] skipped\n\n"
            f"Geometric mean speedup: [cyan]{summary['geometric_mean_speedup']:.2f}x[/]",
            title="Summary",
        ))

    console.print()
    console.print(f"[green]Report:[/] {report_md}")
    console.print(f"[green]Chart:[/]  {report_chart}")
    console.print(f"[green]Speedup:[/] {report_speedup}")
    return


@app.command()
def analyze(
    input: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Input JSON file with benchmark results"),
    benchmark: Optional[str] = typer.Option(
        None, "--benchmark", "-b", help="Specific benchmark to analyze"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed per-worker timings"),
):
    """Analyze parallel region and task timing results.

    Computes efficiency ratios to determine if delayed MLIR optimizations
    (CSE, DCE) impact sequential kernel performance. See docs/hypothesis.md.

    The key metric is:
        η = T_task(CARTS) / T_task(OpenMP)

    Where:
        η ≈ 1.0  → LLVM -O3 recovers optimizations well
        η > 1.0  → CARTS sequential code is slower (delayed opts hurt)
    """
    if not input:
        # Look for most recent results file
        results_files = sorted(Path(".").glob(
            "benchmark_results_*.json"), reverse=True)
        if results_files:
            input = results_files[0]
        else:
            console.print(
                "[yellow]No results file found. Run benchmarks first with --output.[/]")
            raise typer.Exit(1)

    with open(input) as f:
        data = json.load(f)

    console.print(f"\n[bold]Parallel/Task Timing Analysis[/]")
    console.print(f"Source: {input}")
    console.print("=" * 60)

    # Filter by benchmark if specified
    results = data.get("results", [])
    if benchmark:
        results = [r for r in results if benchmark in r["name"]]

    if not results:
        console.print("[yellow]No matching benchmarks found.[/]")
        raise typer.Exit(1)

    # Analyze each benchmark
    analysis_results = []

    for r in results:
        name = r["name"]

        # Check if we have parallel/task timing data
        arts_pt = r.get("run_arts", {}).get("parallel_task_timing")
        omp_pt = r.get("run_omp", {}).get("parallel_task_timing")

        if not arts_pt and not omp_pt:
            # No timing data for this benchmark
            continue

        console.print(f"\n[cyan]{name}[/]")
        console.print("-" * 40)

        # Analyze CARTS (ARTS) timings
        if arts_pt:
            console.print("\n[bold]CARTS Timings:[/]")
            for task_name, timings in arts_pt.get("task_timings", {}).items():
                times = [t["time_sec"] for t in timings]
                if times:
                    mean = sum(times) / len(times)
                    console.print(
                        f"  task.{task_name}: mean={mean:.6f}s (n={len(times)} workers)")
                    if verbose:
                        for t in timings:
                            console.print(
                                f"    worker {t['worker_id']}: {t['time_sec']:.6f}s")

            for par_name, timings in arts_pt.get("parallel_timings", {}).items():
                times = [t["time_sec"] for t in timings]
                if times:
                    mean = sum(times) / len(times)
                    console.print(
                        f"  parallel.{par_name}: mean={mean:.6f}s (n={len(times)} workers)")

        # Analyze OpenMP timings
        if omp_pt:
            console.print("\n[bold]OpenMP Timings:[/]")
            for task_name, timings in omp_pt.get("task_timings", {}).items():
                times = [t["time_sec"] for t in timings]
                if times:
                    mean = sum(times) / len(times)
                    console.print(
                        f"  task.{task_name}: mean={mean:.6f}s (n={len(times)} workers)")
                    if verbose:
                        for t in timings:
                            console.print(
                                f"    worker {t['worker_id']}: {t['time_sec']:.6f}s")

            for par_name, timings in omp_pt.get("parallel_timings", {}).items():
                times = [t["time_sec"] for t in timings]
                if times:
                    mean = sum(times) / len(times)
                    console.print(
                        f"  parallel.{par_name}: mean={mean:.6f}s (n={len(times)} workers)")

        # Compute efficiency ratio if we have both
        if arts_pt and omp_pt:
            console.print("\n[bold]Efficiency Analysis:[/]")

            # Find matching task names
            arts_tasks = arts_pt.get("task_timings", {})
            omp_tasks = omp_pt.get("task_timings", {})

            for task_name in set(arts_tasks.keys()) & set(omp_tasks.keys()):
                arts_times = [t["time_sec"] for t in arts_tasks[task_name]]
                omp_times = [t["time_sec"] for t in omp_tasks[task_name]]

                if arts_times and omp_times:
                    arts_mean = sum(arts_times) / len(arts_times)
                    omp_mean = sum(omp_times) / len(omp_times)

                    if omp_mean > 0:
                        eta = arts_mean / omp_mean
                        analysis_results.append({
                            "benchmark": name,
                            "task": task_name,
                            "eta": eta,
                            "arts_mean": arts_mean,
                            "omp_mean": omp_mean,
                        })

                        # Interpret the result
                        if 0.95 <= eta <= 1.05:
                            interpretation = "[green]LLVM -O3 recovers optimizations well[/]"
                        elif eta < 0.95:
                            interpretation = "[blue]CARTS faster (unexpected)[/]"
                        elif eta <= 1.5:
                            interpretation = "[yellow]Partial recovery - some opts lost[/]"
                        else:
                            interpretation = "[red]Significant degradation - delayed opts hurt[/]"

                        console.print(f"  task.{task_name}:")
                        console.print(
                            f"    η = {eta:.3f} (CARTS={arts_mean:.6f}s / OMP={omp_mean:.6f}s)")
                        console.print(f"    {interpretation}")

            # Compute overhead if we have matching parallel/task names
            arts_parallel = arts_pt.get("parallel_timings", {})
            omp_parallel = omp_pt.get("parallel_timings", {})

            # Try to find matching parallel and task names for overhead calculation
            for par_name in set(arts_parallel.keys()) & set(omp_parallel.keys()):
                # Find a task that might correspond (e.g., "gemm" -> "gemm:kernel")
                matching_tasks = [
                    t for t in arts_tasks.keys() if t.startswith(par_name)]
                if matching_tasks:
                    task_name = matching_tasks[0]
                    if task_name in arts_tasks and task_name in omp_tasks:
                        arts_par_times = [t["time_sec"]
                                          for t in arts_parallel[par_name]]
                        arts_task_times = [t["time_sec"]
                                           for t in arts_tasks[task_name]]
                        omp_par_times = [t["time_sec"]
                                         for t in omp_parallel[par_name]]
                        omp_task_times = [t["time_sec"]
                                          for t in omp_tasks[task_name]]

                        if arts_par_times and arts_task_times and omp_par_times and omp_task_times:
                            arts_overhead = sum(
                                arts_par_times) / len(arts_par_times) - sum(arts_task_times) / len(arts_task_times)
                            omp_overhead = sum(
                                omp_par_times) / len(omp_par_times) - sum(omp_task_times) / len(omp_task_times)

                            console.print(
                                f"\n  Overhead Analysis (parallel.{par_name} - task.{task_name}):")
                            console.print(
                                f"    CARTS overhead: {arts_overhead:.6f}s")
                            console.print(
                                f"    OMP overhead:   {omp_overhead:.6f}s")

    # Summary
    if analysis_results:
        console.print("\n" + "=" * 60)
        console.print("[bold]Summary[/]")
        console.print("-" * 40)

        # Compute aggregate statistics
        etas = [r["eta"] for r in analysis_results]
        avg_eta = sum(etas) / len(etas)

        # Geometric mean
        import math
        geomean_eta = math.exp(sum(math.log(e) for e in etas) / len(etas))

        console.print(f"Benchmarks analyzed: {len(analysis_results)}")
        console.print(f"Average η:          {avg_eta:.3f}")
        console.print(f"Geometric mean η:   {geomean_eta:.3f}")

        # Overall interpretation
        console.print("\n[bold]Interpretation:[/]")
        if geomean_eta <= 1.05:
            console.print(
                "[green]✓ Delayed optimizations have minimal impact.[/]")
            console.print(
                "  LLVM -O3 successfully recovers most optimizations.")
        elif geomean_eta <= 1.2:
            console.print(
                "[yellow]⚠ Minor performance impact from delayed optimizations.[/]")
            console.print(
                "  Some optimization opportunities are lost at MLIR level.")
        else:
            console.print(
                "[red]✗ Significant performance degradation detected.[/]")
            console.print(
                "  Delayed MLIR optimizations hurt sequential kernel performance.")
            console.print(
                "  Consider investigating optimization ordering in CARTS pipeline.")
    else:
        console.print(
            "\n[yellow]No parallel/task timing data found in results.[/]")
        console.print(
            "Make sure benchmarks use CARTS_PARALLEL_TIMER_* and CARTS_TASK_TIMER_* macros.")


if __name__ == "__main__":
    app()
