#!/usr/bin/env python3
"""
CARTS Unified Benchmark Runner

A powerful CLI tool for building, running, and verifying CARTS benchmarks.
Provides rich terminal output, correctness verification, performance timing, and JSON export.

Usage:
    carts benchmarks list [--suite SUITE] [--format FORMAT]
    carts benchmarks run [BENCHMARKS...] [OPTIONS]
    carts benchmarks build [BENCHMARKS...] [--size SIZE]
    carts benchmarks clean [BENCHMARKS...] [--all]
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import hashlib
import signal
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from statistics import mean, median, stdev

logger = logging.getLogger(__name__)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import typer

from dekk import (
    Colors, Symbols,
    console as _shared_console,
    print_header, print_step, print_success, print_error,
    print_warning, print_info, print_debug as _print_debug,
)
from scripts import format_passed, format_failed, format_skipped
from scripts.arts_config import (
    parse_arts_cfg,
    EMBEDDED_KEYS,
    KEY_COUNTER_FOLDER,
    KEY_LAUNCHER,
    KEY_MASTER_NODE,
    KEY_NODE_COUNT,
    KEY_NODES,
    KEY_WORKER_THREADS,
    get_cfg_int as get_arts_cfg_int,
    get_cfg_str as get_arts_cfg_str,
    get_cfg_nodes as get_arts_cfg_nodes,
    upsert_cfg_value as _upsert_arts_cfg_value,
    extract_embedded_cfg as _extract_embedded_arts_cfg,
    validate_embedded_cfg as _validate_embedded_arts_cfg,
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

from slurm.experiment import (
    SlurmBatchExecutor,
    SlurmBatchRequest,
    SlurmExecutorDependencies,
    count_total_slurm_jobs,
    find_multinode_disabled_benchmarks,
    format_node_counts_display,
    require_slurm_commands,
    validate_requested_node_counts,
)

# Shared constants and parsing
from common import (
    CHECKSUM_PATTERNS,
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
    VARIANT_ARTS,
    VARIANT_OMP,
    VARIANT_OPENMP,
    SPEEDUP_BASIS_KERNEL,
    SPEEDUP_BASIS_E2E,
    SPEEDUP_BASIS_TOTAL,
    SPEEDUP_BASIS_NA,
    KEY_BUILD_ARTS,
    KEY_BUILD_OMP,
    KEY_RUN_ARTS,
    KEY_RUN_OMP,
    KEY_TIMING,
    KEY_VERIFICATION,
    KEY_CONFIG,
    KEY_STATUS,
    KEY_RUN_NUMBER,
    KEY_RUN_PHASE,
    KEY_SPEEDUP,
    KEY_NAME,
    KEY_SIZE,
    KEY_ARTS_E2E_SEC,
    KEY_OMP_E2E_SEC,
    KEY_ARTS_OUTLIERS,
    KEY_OMP_OUTLIERS,
    KEY_ARTS_RAW_COUNT,
    KEY_OMP_RAW_COUNT,
    KEY_PAIRED_RAW_COUNT,
    KEY_ARTS_FILTERED_COUNT,
    KEY_OMP_FILTERED_COUNT,
    KEY_PAIRED_FILTERED_COUNT,
    KEY_IS_OUTLIER,
    RESULTS_FILENAME,
    STARTUP_OUTLIER_DIAGNOSTICS_FILENAME,
    parse_checksum,
    parse_kernel_timings,
    parse_e2e_timings,
    parse_startup_timings,
    parse_verification_timings,
    parse_cleanup_timings,
    parse_perf_csv as _shared_parse_perf_csv,
)

# ============================================================================
# Constants (local-only — shared constants imported from benchmark_common)
# ============================================================================

# Data models (enums + dataclasses)
from models import (
    Status, Phase,
    BuildResult, WorkerTiming, ParallelTaskTiming, PerfCacheMetrics,
    RunResult, TimingResult, VerificationResult, ReferenceChecksum, ExperimentStep,
    Artifacts, BenchmarkConfig, BenchmarkResult,
)
from execution import (
    BenchmarkExecutionContext,
    BenchmarkProcessRequest,
    BenchmarkProcessRunner,
    BenchmarkRunFiles,
)
from verification import verify_against_omp
from orchestration import (
    LocalStepExecutionRequest,
    ResolvedStepConfig,
    SlurmStepExecutionRequest,
    StepCliDefaults,
    StepExecutionOrchestrator,
    StepResolver,
)
from pipeline import (
    ConfigExecutionExecutor,
    ConfigExecutionPlan,
    ExecutionHooks,
)

# Artifact management
from artifacts import ArtifactManager
from report import generate_report

# Reproducibility metadata
from metadata import (
    get_git_hash, get_compiler_version, get_cpu_info,
    get_reproducibility_metadata, _serialize_parallel_task_timing,
)


# ============================================================================
# CLI Application
# ============================================================================

app = typer.Typer(
    name="carts-bench",
    help="CARTS Benchmark Runner - Build, run, and verify benchmarks.",
    add_completion=False,
    no_args_is_help=True,
)
console = _shared_console


def get_carts_dir() -> Path:
    """Get the CARTS root directory."""
    script_dir = Path(__file__).parent.resolve()
    # Navigate up from external/carts-benchmarks/scripts to carts root
    carts_dir = script_dir.parent.parent.parent
    if not (carts_dir / "tools" / "carts").exists():
        # Fallback: try CARTS_DIR environment variable
        env_dir = os.environ.get("CARTS_DIR")
        if env_dir:
            carts_dir = Path(env_dir)
    return carts_dir


def arts_runtime_is_installed(carts_dir: Optional[Path] = None) -> bool:
    """Return True when the installed ARTS runtime is present and linkable."""
    root = carts_dir or get_carts_dir()
    install_dir = root / ".install" / "arts"
    lib_dir = install_dir / "lib"
    cmake_config = lib_dir / "cmake" / "ARTS" / "ARTSConfig.cmake"
    public_header = install_dir / "include" / "arts.h"
    has_library = any(path.is_file() for path in lib_dir.glob("libarts*"))
    return cmake_config.is_file() and public_header.is_file() and has_library


BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BENCHMARKS_DIR / "configs"
PROFILES_DIR = CONFIGS_DIR / "profiles"
DEFAULT_ARTS_CONFIG = CONFIGS_DIR / "local.cfg"
SUPPORTED_SIZES: Tuple[str, ...] = (
    "small",
    "medium",
    "large",
    "extralarge",
    "mini",
    "standard",
)
SIZE_ALIASES: Dict[str, str] = {
    "extra-large": "extralarge",
    "extra_large": "extralarge",
    "xlarge": "extralarge",
}
SIZE_HELP = (
    "Dataset size: small, medium, large, extralarge, mini, standard"
)
DEFAULT_REPORTING_MODE = "median"
DEFAULT_STARTUP_OUTLIER_POLICY: Dict[str, Any] = {
    "enabled": True,
    "z_threshold": 3.5,
    "min_runs": 3,
    "min_startup_sec": 0.05,
    "min_relative_multiplier": 1.25,
}
DEFAULT_PERF_GATE_POLICY = (
    CONFIGS_DIR / "perf-gates" / "openmp-surpass-stable-subset.json"
)


# ============================================================================
# Helper Functions
# ============================================================================


def parse_threads(spec: str) -> List[int]:
    """Parse thread specification into list of thread counts.

    Supports comma-separated: "1,2,4,8"
    Supports range format: "1:16:2" (start:stop:step)
    """
    if ',' in spec:
        values = [int(t.strip()) for t in spec.split(',') if t.strip()]
    elif ':' in spec:
        parts = [int(p.strip()) for p in spec.split(':') if p.strip()]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError(f"Invalid thread range format: {spec}")
        values = list(range(start, stop + 1, step))
    else:
        # Single thread count
        values = [int(spec)]

    if any(v < 1 for v in values):
        raise ValueError(f"Thread counts must be >= 1: {spec}")
    return values


def parse_size(value: str, field_name: str = "size") -> str:
    """Normalize and validate dataset size tokens."""
    raw = str(value).strip().lower()
    if not raw:
        raise ValueError(f"{field_name} cannot be empty")
    normalized = SIZE_ALIASES.get(raw, raw)
    if normalized not in SUPPORTED_SIZES:
        allowed = ", ".join(SUPPORTED_SIZES)
        raise ValueError(
            f"Invalid {field_name}: '{value}'. Supported sizes: {allowed}"
        )
    return normalized


def parse_slurm_time_limit_seconds(value: str) -> int:
    """Convert a SLURM HH:MM:SS time limit into seconds."""
    parts = [part.strip() for part in str(value).split(":")]
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError(f"Invalid SLURM time limit: {value}")
    hours, minutes, seconds = (int(part) for part in parts)
    return hours * 3600 + minutes * 60 + seconds


def format_slurm_time_limit(seconds: int) -> str:
    """Format a wall-clock limit in seconds as HH:MM:SS."""
    total = max(1, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def resolve_slurm_time_limit(
    timeout_seconds: int,
    explicit_time_limit: Optional[str],
) -> str:
    """Resolve the scheduler wall time for a benchmark run."""
    if explicit_time_limit is not None:
        parse_slurm_time_limit_seconds(explicit_time_limit)
        return explicit_time_limit
    return format_slurm_time_limit(timeout_seconds + 30)


def normalize_requested_benchmarks(
    benchmarks: Optional[List[str]],
) -> Tuple[Optional[List[str]], bool]:
    """Normalize user-provided benchmark names.

    Returns:
        (normalized_list_or_none, dropped_blank_args)
    """
    if not benchmarks:
        return None, False

    normalized: List[str] = []
    seen = set()
    dropped_blank = False
    for bench in benchmarks:
        bench_name = bench.strip()
        if not bench_name:
            dropped_blank = True
            continue
        if bench_name in seen:
            continue
        normalized.append(bench_name)
        seen.add(bench_name)

    return (normalized or None), dropped_blank


def find_invalid_benchmarks(
    runner: "BenchmarkRunner",
    benchmarks: List[str],
) -> List[str]:
    """Return benchmark names that do not exist in the benchmark tree."""
    available = set(runner.discover_benchmarks())
    return [name for name in benchmarks if name not in available]


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


def _sanitize_config_token(value: str, fallback: str = "default") -> str:
    """Create a filesystem-safe token for generated config names."""
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return token or fallback


def generate_arts_config(
    base_path: Path,
    threads: int,
    counter_dir: Optional[Path] = None,
    launcher: str = "ssh",
    nodes_override: Optional[int] = None,
    benchmark_name: Optional[str] = None,
) -> Path:
    """Generate temporary arts.cfg with specific configuration from a template.

    The base_path must be a valid config file that serves as a template.
    For SSH launcher templates we preserve and truncate the configured
    hostnames. For local and Slurm launchers, node_count is sufficient.

    Args:
        base_path: Template config file (required).
        threads: Worker thread count per node.
        counter_dir: Optional directory for counter output.
        launcher: ARTS launcher type (ssh, slurm, lsf, local).
        nodes_override: If set, override node_count in the generated config.

    Note: For Slurm, node_count in config is IGNORED - ARTS reads SLURM_NNODES
    from environment (set by srun). The launcher controls HOW we invoke the
    executable, not just what's in the config.
    """
    if not base_path.exists():
        raise ValueError(f"Config template not found: {base_path}")

    content = base_path.read_text()

    # CLI --threads maps to ARTS worker_threads in the v2 runtime schema.
    content = _upsert_arts_cfg_value(content, KEY_WORKER_THREADS, threads)

    # Handle node override.
    if nodes_override is not None:
        content = _upsert_arts_cfg_value(content, KEY_NODE_COUNT, nodes_override)
        if launcher == "ssh":
            all_nodes = get_arts_cfg_nodes(base_path)
            if nodes_override > len(all_nodes):
                raise ValueError(
                    f"Requested --nodes {nodes_override} but SSH config '{base_path}' only has {len(all_nodes)} node(s).\n"
                    f"Use --arts-config to specify a config template with sufficient nodes.\n"
                    f"Example: carts benchmarks run --arts-config /opt/carts/docker/arts-docker.cfg --nodes {nodes_override}"
                )
            truncated = all_nodes[:nodes_override]
            content = _upsert_arts_cfg_value(content, KEY_NODES, ",".join(truncated))
            content = _upsert_arts_cfg_value(content, KEY_MASTER_NODE, truncated[0])

    # Update launcher
    content = _upsert_arts_cfg_value(content, KEY_LAUNCHER, launcher)

    # Add counter settings if requested
    if counter_dir:
        content = _upsert_arts_cfg_value(content, KEY_COUNTER_FOLDER, counter_dir)

    # Determine node count for filename
    node_count = nodes_override if nodes_override else get_arts_cfg_int(
        base_path, KEY_NODE_COUNT) or 1

    # Write to shared directory (NOT /tmp which is node-local in multi-node setups)
    # The carts-benchmarks directory is shared across all nodes via mounted volume.
    # Filename encodes the effective combination so configs are not overwritten
    # when different benchmarks/launchers/templates are built in one run.
    generated_configs_dir = Path(__file__).parent / ".generated_configs"
    generated_configs_dir.mkdir(exist_ok=True)
    source_hash = hashlib.sha1(str(base_path.resolve()).encode("utf-8")).hexdigest()[:8]
    counter_tag = (
        hashlib.sha1(str(counter_dir.resolve()).encode("utf-8")).hexdigest()[:8]
        if counter_dir is not None
        else "nocounter"
    )
    bench_tag = _sanitize_config_token(benchmark_name or "global")
    launcher_tag = _sanitize_config_token(launcher)
    temp_path = (
        generated_configs_dir
        / f"arts_{bench_tag}_{launcher_tag}_{threads}t_{node_count}n_{source_hash}_{counter_tag}.cfg"
    )
    temp_path.write_text(content)
    return temp_path


def _resolve_effective_arts_config(
    bench_path: Path,
    override_config: Optional[Path] = None,
) -> Path:
    """Resolve the arts.cfg template for a benchmark."""
    if override_config is not None:
        return override_config.resolve()

    for candidate in [
        bench_path / "arts.cfg",
        bench_path.parent / "arts.cfg",
        DEFAULT_ARTS_CONFIG,
    ]:
        if candidate.exists():
            return candidate.resolve()

    return DEFAULT_ARTS_CONFIG.resolve()




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
        self.benchmarks_dir = BENCHMARKS_DIR
        self.results: List[BenchmarkResult] = []
        self._reference_cache: Dict[Tuple[str, str, str, int], ReferenceChecksum] = {}

    def _reference_cache_key(
        self,
        name: str,
        size: str,
        cflags: str,
        omp_threads: int,
    ) -> Tuple[str, str, str, int]:
        return (name, size, cflags.strip(), omp_threads)

    def ensure_omp_reference(
        self,
        name: str,
        size: str,
        cflags: str,
        omp_threads: int,
        timeout: int,
    ) -> ReferenceChecksum:
        """Build and run a stored OpenMP reference for multi-node checksum verification."""
        key = self._reference_cache_key(name, size, cflags, omp_threads)
        cached = self._reference_cache.get(key)
        if cached is not None:
            return cached

        am = self.artifact_manager
        if am is not None:
            persisted = am.load_reference_result(name, size, omp_threads, cflags)
            if persisted:
                try:
                    reference = ReferenceChecksum(
                        status=Status(str(persisted.get("status", Status.FAIL))),
                        checksum=(
                            str(persisted["checksum"])
                            if persisted.get("checksum") is not None
                            else None
                        ),
                        omp_threads=int(persisted.get("omp_threads", omp_threads)),
                        note=str(persisted.get("note", "")),
                        source=str(persisted.get("source", "")),
                        executable_omp=(
                            str(persisted["executable_omp"])
                            if persisted.get("executable_omp") is not None
                            else None
                        ),
                        log_path=(
                            str(persisted["log_path"])
                            if persisted.get("log_path") is not None
                            else None
                        ),
                        run_dir=(
                            str(persisted["run_dir"])
                            if persisted.get("run_dir") is not None
                            else None
                        ),
                    )
                    self._reference_cache[key] = reference
                    return reference
                except Exception:
                    pass

        bench_path = self.benchmarks_dir / name
        run_args = self.get_run_args(bench_path, size)

        reference_artifacts_dir: Optional[Path] = None
        reference_run_dir: Optional[Path] = None
        reference_log: Optional[Path] = None
        reference_source = "multinode_omp_reference"
        if am is not None:
            reference_root = am.get_reference_root(
                name, size, omp_threads, cflags
            )
            reference_artifacts_dir = am.get_reference_artifacts_dir(
                name, size, omp_threads, cflags
            )
            reference_run_dir = am.get_reference_run_dir(
                name, size, omp_threads, cflags
            )
            reference_log = reference_run_dir / "omp_reference.log"
            reference_source = str(reference_root / "reference.json")
        else:
            tmp_root = Path(tempfile.mkdtemp(prefix="carts_bench_reference_"))
            reference_artifacts_dir = tmp_root / "artifacts"
            reference_run_dir = tmp_root / "run_1"
            reference_artifacts_dir.mkdir(parents=True, exist_ok=True)
            reference_run_dir.mkdir(parents=True, exist_ok=True)
            reference_log = reference_run_dir / "omp_reference.log"

        build_omp = self.build_benchmark(
            name,
            size,
            variant=VARIANT_OPENMP,
            cflags=cflags,
            build_output_dir=reference_artifacts_dir,
        )
        if build_omp.status != Status.PASS or not build_omp.executable:
            reference = ReferenceChecksum(
                status=Status.FAIL,
                checksum=None,
                omp_threads=omp_threads,
                note="Failed to build multi-node OpenMP reference",
                source=reference_source,
                executable_omp=build_omp.executable,
                log_path=str(reference_log) if reference_log else None,
                run_dir=str(reference_run_dir) if reference_run_dir else None,
            )
        else:
            reference_env = {
                "OMP_NUM_THREADS": str(omp_threads),
                "OMP_WAIT_POLICY": "ACTIVE",
            }
            run_omp = self.run_benchmark(
                build_omp.executable,
                timeout,
                env=reference_env,
                args=run_args,
                log_file=reference_log,
            )
            if run_omp.status == Status.PASS and run_omp.checksum is not None:
                note = "Stored multi-node OpenMP reference checksum captured"
                ref_status = Status.PASS
            elif run_omp.status == Status.PASS:
                note = "Stored multi-node OpenMP reference completed without checksum"
                ref_status = Status.FAIL
            else:
                note = f"Stored multi-node OpenMP reference failed ({run_omp.status.value})"
                ref_status = run_omp.status
            reference = ReferenceChecksum(
                status=ref_status,
                checksum=run_omp.checksum,
                omp_threads=omp_threads,
                note=note,
                source=reference_source,
                executable_omp=build_omp.executable,
                log_path=str(reference_log) if reference_log else None,
                run_dir=str(reference_run_dir) if reference_run_dir else None,
            )

        if am is not None:
            am.save_reference_result(name, size, omp_threads, cflags, asdict(reference))

        self._reference_cache[key] = reference
        return reference

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

        Based on carts.mk defaults:
            ARTS_BINARY := $(EXAMPLE_NAME)_arts       # in benchmark root
            OMP_BINARY := $(BUILD_DIR)/$(EXAMPLE_NAME)_omp  # in build/

        Note: artifact-managed builds may override OMP_BINARY to place
        executable directly in artifacts/ (no nested build/).

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
        variant: str = VARIANT_ARTS,
        arts_config: Optional[Path] = None,
        cflags: str = "",
        compile_args: Optional[str] = None,
        build_output_dir: Optional[Path] = None,
    ) -> BuildResult:
        """Build a single benchmark using make."""
        try:
            size = parse_size(size, "size")
        except ValueError as e:
            return BuildResult(
                status=Status.FAIL,
                duration_sec=0.0,
                output=str(e),
            )

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
        # Build command using make
        # Use granular targets ({size}-arts, {size}-openmp) defined in common/carts.mk
        # CRITICAL: Provide explicit path to carts executable (not in PATH during non-interactive shells)
        carts_exe = self.carts_dir / "tools" / "carts"
        arts_exe_default, omp_exe_default = self.get_executable_paths(bench_path)
        output_root = build_output_dir.resolve() if build_output_dir else bench_path
        # Keep build outputs flat under the selected output root.
        build_dir_override = output_root
        logs_dir_override = output_root / "logs"
        output_root.mkdir(parents=True, exist_ok=True)
        build_dir_override.mkdir(parents=True, exist_ok=True)
        logs_dir_override.mkdir(parents=True, exist_ok=True)
        arts_output_path = output_root / arts_exe_default.name
        omp_output_path = output_root / omp_exe_default.name

        env_overrides: Dict[str, str] = {}
        effective_arts_config = arts_config
        if variant != VARIANT_OPENMP and effective_arts_config is None:
            # Keep build behavior independent of current working directory.
            effective_arts_config = _resolve_effective_arts_config(bench_path)
        if variant != VARIANT_OPENMP and effective_arts_config is not None:
            effective_arts_config = effective_arts_config.resolve()
            if build_output_dir is not None:
                # Keep the exact compile-time config alongside the build artifacts.
                local_cfg = output_root / "arts.cfg"
                if effective_arts_config != local_cfg.resolve():
                    shutil.copy2(effective_arts_config, local_cfg)
                effective_arts_config = local_cfg

        if variant == VARIANT_OPENMP:
            # Build only OpenMP variant using granular target
            cmd = [
                "make",
                f"{size}-openmp",
                f"CARTS={carts_exe}",
                f"BUILD_DIR={build_dir_override}",
                f"LOG_DIR={logs_dir_override}",
                f"OMP_BINARY={omp_output_path}",
            ]
            if cflags:
                cmd.append(f"CFLAGS={cflags}")
        else:
            # Build ARTS variant (full pipeline)
            # Use granular size-arts target for ARTS-only builds
            cmd = [
                "make",
                f"{size}-arts",
                f"CARTS={carts_exe}",
                f"BUILD_DIR={build_dir_override}",
                f"LOG_DIR={logs_dir_override}",
                f"ARTS_BINARY={arts_output_path}",
            ]
            if cflags:
                cmd.append(f"CFLAGS={cflags}")
            # Keep compiler intermediates out of source benchmark directories.
            env_overrides["CARTS_COMPILE_WORKDIR"] = str(output_root)

        # Add ARTS config override if provided
        if effective_arts_config and variant != VARIANT_OPENMP:
            cmd.append(f"ARTS_CFG={effective_arts_config.resolve()}")
        if compile_args and variant != VARIANT_OPENMP:
            escaped_args = compile_args.replace("\\", "\\\\").replace(" ", "\\ ")
            cmd.append(f"COMPILE_ARGS={escaped_args}")

        # Debug output level 1: show commands
        if self.debug >= 1:
            env_prefix = " ".join(f"{k}={v}" for k, v in env_overrides.items())
            if env_prefix:
                self.console.print(
                    f"[{Colors.DEBUG}]$ cd {bench_path} && {env_prefix} {' '.join(cmd)}[/{Colors.DEBUG}]"
                )
            else:
                self.console.print(f"[{Colors.DEBUG}]$ cd {bench_path} && {' '.join(cmd)}[/{Colors.DEBUG}]")

        start = time.time()

        try:
            env = os.environ.copy()
            env.update(env_overrides)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=bench_path,
                env=env,
            )
            duration = time.time() - start

            # Debug output level 2: write build log to file
            if self.debug >= 2:
                log_file = logs_dir_override / f"build_{variant}.log"
                with open(log_file, "w") as f:
                    f.write(f"# Command: {' '.join(cmd)}\n")
                    f.write(f"# Duration: {duration:.3f}s\n")
                    f.write(f"# Exit code: {result.returncode}\n\n")
                    if result.stdout:
                        f.write(result.stdout)
                    if result.stderr:
                        f.write("=== STDERR ===\n")
                        f.write(result.stderr)
                self.console.print(f"[{Colors.DEBUG}]  Log: {log_file}[/{Colors.DEBUG}]")

            if result.returncode == 0:
                expected_exe = arts_output_path if variant != VARIANT_OPENMP else omp_output_path
                executable = None
                if expected_exe.is_file() and os.access(expected_exe, os.X_OK):
                    executable = str(expected_exe)
                else:
                    executable = self._find_executable(bench_path, variant)
                if variant != VARIANT_OPENMP:
                    config_error = _validate_embedded_arts_cfg(
                        output_root, effective_arts_config
                    )
                    if config_error is not None:
                        combined_output = (result.stdout + result.stderr).strip()
                        output = (
                            f"{combined_output}\n{config_error}".strip()
                            if combined_output
                            else config_error
                        )
                        return BuildResult(
                            status=Status.FAIL,
                            duration_sec=duration,
                            output=output,
                        )
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
        except subprocess.TimeoutExpired as e:
            if isinstance(e.stdout, bytes):
                partial_stdout = e.stdout.decode("utf-8", errors="replace")
            elif isinstance(e.stdout, str):
                partial_stdout = e.stdout
            else:
                partial_stdout = ""

            if isinstance(e.stderr, bytes):
                partial_stderr = e.stderr.decode("utf-8", errors="replace")
            elif isinstance(e.stderr, str):
                partial_stderr = e.stderr
            else:
                partial_stderr = ""

            partial_out = (partial_stdout + partial_stderr).strip()
            timeout_msg = "Build timed out after 300 seconds"
            output = f"{partial_out}\n{timeout_msg}".strip() if partial_out else timeout_msg
            return BuildResult(
                status=Status.TIMEOUT,
                duration_sec=300.0,
                output=output,
            )
        except Exception as e:
            return BuildResult(
                status=Status.FAIL,
                duration_sec=time.time() - start,
                output=str(e),
            )

    def _find_executable(self, bench_path: Path, variant: str) -> Optional[str]:
        """Find the generated executable."""
        suffix = "_arts" if variant == VARIANT_ARTS else "_omp"

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

    def _index_build_artifacts(
        self,
        artifacts_dir: Path,
        arts_cfg_used: Optional[Path] = None,
    ) -> Dict[str, Optional[str]]:
        """Return discovered build artifact paths from an already-built directory."""
        paths: Dict[str, Optional[str]] = {}

        cfg_src = artifacts_dir / "arts.cfg"
        if not cfg_src.exists() and arts_cfg_used is not None and arts_cfg_used.exists():
            cfg_src = arts_cfg_used
        if cfg_src.exists():
            paths["arts_config"] = str(cfg_src.resolve())

        metadata = artifacts_dir / ".carts-metadata.json"
        if metadata.exists():
            paths["carts_metadata"] = str(metadata.resolve())

        arts_metadata_files = sorted(artifacts_dir.glob("*_arts_metadata.mlir"))
        if arts_metadata_files:
            paths["arts_metadata_mlir"] = str(arts_metadata_files[0].resolve())

        arts_bins = [p for p in sorted(artifacts_dir.glob("*_arts")) if p.is_file()]
        if arts_bins:
            paths["executable_arts"] = str(arts_bins[0].resolve())

        omp_bins = [p for p in sorted(artifacts_dir.glob("*_omp")) if p.is_file()]
        if omp_bins:
            paths["executable_omp"] = str(omp_bins[0].resolve())

        return paths

    def _make_process_runner(self) -> BenchmarkProcessRunner:
        """Create the shared process runner used by local benchmark variants."""
        return BenchmarkProcessRunner(
            self.console,
            verbose=self.verbose,
            debug=self.debug,
        )

    def _create_execution_context(
        self,
        *,
        name: str,
        size: str,
        bench_path: Path,
        config: BenchmarkConfig,
        effective_arts_cfg: Path,
        desired_threads: int,
        desired_nodes: int,
        desired_launcher: str,
        actual_omp_threads: int,
        effective_cflags: str,
        build_output_dir: Optional[Path] = None,
        artifact_paths: Optional[Dict[str, Optional[str]]] = None,
    ) -> BenchmarkExecutionContext:
        """Create the resolved execution contract for one benchmark config."""
        return BenchmarkExecutionContext(
            name=name,
            suite=name.split("/")[0] if "/" in name else "",
            size=size,
            bench_path=bench_path,
            config=config,
            effective_arts_cfg=effective_arts_cfg,
            desired_threads=desired_threads,
            desired_nodes=desired_nodes,
            desired_launcher=desired_launcher,
            actual_omp_threads=actual_omp_threads,
            effective_cflags=effective_cflags,
            run_args=self.get_run_args(bench_path, size),
            verify_tolerance=self.get_verify_tolerance(bench_path),
            build_output_dir=build_output_dir,
            artifact_paths=artifact_paths or {},
        )

    def _create_run_files(
        self,
        *,
        name: str,
        bench_path: Path,
        config: BenchmarkConfig,
        desired_threads: int,
        run_number: int,
        runs: int,
        counter_dir: Optional[Path],
        perf_enabled: bool,
        perf_dir: Optional[Path] = None,
        run_timestamp: str = "",
        sweep_log_names: bool = False,
    ) -> BenchmarkRunFiles:
        """Resolve per-run logs, counters, and perf output locations."""
        am = self.artifact_manager
        safe_bench_name = name.replace("/", "_")

        run_dir: Optional[Path] = None
        arts_log: Optional[Path] = None
        omp_log: Optional[Path] = None
        run_counter_dir: Optional[Path] = None
        perf_output_dir: Optional[Path] = None
        arts_perf_name = omp_perf_name = None
        arts_perf_main = arts_perf_temp = omp_perf_main = omp_perf_temp = None

        if am:
            run_dir = am.get_run_dir(name, config, run_number)
            arts_log = run_dir / "arts.log"
            omp_log = run_dir / "omp.log"
            run_counter_dir = am.get_counter_dir(name, config, run_number)
            if perf_enabled:
                perf_output_dir = am.get_perf_dir(name, config, run_number)
                arts_perf_name = "arts_cache.csv"
                omp_perf_name = "omp_cache.csv"
        else:
            if counter_dir is not None:
                if run_timestamp:
                    run_counter_dir = counter_dir / run_timestamp / f"{safe_bench_name}_{run_number}"
                else:
                    run_counter_dir = counter_dir
                run_counter_dir.mkdir(parents=True, exist_ok=True)

            if self.debug >= 2:
                logs_dir = bench_path / "logs"
                if sweep_log_names:
                    run_suffix = f"_r{run_number}" if runs > 1 else ""
                    arts_log = logs_dir / f"arts_{desired_threads}t{run_suffix}.log"
                    omp_log = logs_dir / f"omp_{desired_threads}t{run_suffix}.log"
                else:
                    arts_log = logs_dir / "arts.log"
                    omp_log = logs_dir / "omp.log"

            if perf_enabled:
                effective_perf_dir = perf_dir or Path("./perfs")
                if run_timestamp:
                    perf_timestamp_dir = effective_perf_dir / run_timestamp
                    perf_timestamp_dir.mkdir(parents=True, exist_ok=True)
                    perf_output_dir = perf_timestamp_dir
                    arts_perf_main = perf_timestamp_dir / f"{safe_bench_name}_arts.csv"
                    arts_perf_temp = (
                        perf_timestamp_dir / f"_temp_{safe_bench_name}_arts_{run_number}.csv"
                    )
                    omp_perf_main = perf_timestamp_dir / f"{safe_bench_name}_omp.csv"
                    omp_perf_temp = (
                        perf_timestamp_dir / f"_temp_{safe_bench_name}_omp_{run_number}.csv"
                    )
                    arts_perf_name = arts_perf_temp.name
                    omp_perf_name = omp_perf_temp.name
                elif not sweep_log_names:
                    effective_perf_dir.mkdir(parents=True, exist_ok=True)
                    perf_output_dir = effective_perf_dir

        return BenchmarkRunFiles(
            run_number=run_number,
            run_dir=run_dir,
            arts_log=arts_log,
            omp_log=omp_log,
            counter_dir=run_counter_dir,
            perf_output_dir=perf_output_dir,
            arts_perf_name=arts_perf_name,
            omp_perf_name=omp_perf_name,
            arts_perf_main=arts_perf_main,
            arts_perf_temp=arts_perf_temp,
            omp_perf_main=omp_perf_main,
            omp_perf_temp=omp_perf_temp,
        )

    def _run_process_request(self, request: BenchmarkProcessRequest) -> RunResult:
        """Execute one process request and parse the benchmark-specific outputs."""
        outcome = self._make_process_runner().execute(request)

        perf_metrics = None
        perf_csv_path = None
        if request.perf_enabled and outcome.perf_output and outcome.perf_output.exists():
            perf_metrics = self.parse_perf_csv(outcome.perf_output)
            perf_csv_path = str(outcome.perf_output)

        return RunResult(
            status=outcome.status,
            duration_sec=outcome.duration_sec,
            exit_code=outcome.exit_code,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            checksum=self.extract_checksum(outcome.stdout),
            kernel_timings=self.extract_kernel_timings(outcome.stdout),
            e2e_timings=self.extract_e2e_timings(outcome.stdout),
            startup_timings=self.extract_startup_timings(outcome.stdout),
            verification_timings=self.extract_verification_timings(outcome.stdout),
            cleanup_timings=self.extract_cleanup_timings(outcome.stdout),
            startup_diagnostics=dict(outcome.startup_diagnostics),
            parallel_task_timing=self.extract_parallel_task_timings(outcome.stdout),
            perf_metrics=perf_metrics,
            perf_csv_path=perf_csv_path,
        )

    def _create_common_env(self) -> Dict[str, str]:
        """Return environment overrides shared by local benchmark runs."""
        return {}

    def _create_execution_plan(
        self,
        *,
        execution: BenchmarkExecutionContext,
        timeout: int,
        run_numbers: Tuple[int, ...],
        compile_args: Optional[str],
        perf_enabled: bool,
        perf_interval: float,
        counter_dir: Optional[Path],
        perf_dir: Optional[Path],
        run_timestamp: str,
        sweep_log_names: bool,
        report_speedup: bool,
        env_overrides: Dict[str, str],
        persisted_env_overrides: Optional[Dict[str, str]] = None,
        variant: Optional[str] = None,
    ) -> ConfigExecutionPlan:
        """Create the shared execution plan for one resolved benchmark config."""
        return ConfigExecutionPlan(
            execution=execution,
            timeout=timeout,
            run_numbers=run_numbers,
            compile_args=compile_args,
            perf_enabled=perf_enabled,
            perf_interval=perf_interval,
            counter_dir=counter_dir,
            perf_dir=perf_dir,
            run_timestamp=run_timestamp,
            sweep_log_names=sweep_log_names,
            report_speedup=report_speedup,
            env_overrides=dict(env_overrides),
            persisted_env_overrides=(
                dict(persisted_env_overrides)
                if persisted_env_overrides is not None
                else None
            ),
            variant=variant,
        )

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
        compile_args: Optional[str] = None,
        perf_enabled: bool = False,
        perf_interval: float = 0.1,
        variant: Optional[str] = None,
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
        # Determine effective config template
        effective_config = _resolve_effective_arts_config(bench_path, base_config)

        base_nodes = get_arts_cfg_int(effective_config, KEY_NODE_COUNT) or 1
        base_threads = get_arts_cfg_int(effective_config, KEY_WORKER_THREADS) or 1
        base_launcher = get_arts_cfg_str(effective_config, KEY_LAUNCHER) or "ssh"
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

                # Generate arts.cfg with thread count, launcher, and node count.
                arts_cfg = generate_arts_config(
                    effective_config, threads, None,
                    desired_launcher, desired_nodes, benchmark_name=name
                )

                # Compute effective cflags (may include weak scaling size overrides)
                effective_cflags = cflags
                if weak_scaling and base_size:
                    weak_cflags = get_weak_scaling_cflags(
                        name, base_size, threads, desired_nodes, base_parallelism=1
                    )
                    if weak_cflags:
                        effective_cflags = f"{cflags} {weak_cflags}".strip()

                env = self._create_common_env()
                env["OMP_NUM_THREADS"] = str(actual_omp_threads)
                if "OMP_WAIT_POLICY" not in os.environ:
                    env["OMP_WAIT_POLICY"] = "ACTIVE"

                build_output_dir: Optional[Path] = None
                if am:
                    # Build directly in artifacts directory to keep outputs self-contained.
                    build_output_dir = am.get_artifacts_dir(name, config)
                    build_output_dir.mkdir(parents=True, exist_ok=True)

                execution = self._create_execution_context(
                    name=name,
                    size=size,
                    bench_path=bench_path,
                    config=config,
                    effective_arts_cfg=arts_cfg,
                    desired_threads=threads,
                    desired_nodes=desired_nodes,
                    desired_launcher=desired_launcher,
                    actual_omp_threads=actual_omp_threads,
                    effective_cflags=effective_cflags,
                    build_output_dir=build_output_dir,
                )
                plan = self._create_execution_plan(
                    execution=execution,
                    timeout=timeout,
                    run_numbers=tuple(range(1, runs + 1)),
                    compile_args=compile_args,
                    perf_enabled=perf_enabled,
                    perf_interval=perf_interval,
                    counter_dir=counter_dir,
                    perf_dir=None,
                    run_timestamp="",
                    sweep_log_names=True,
                    report_speedup=(desired_nodes == 1),
                    env_overrides=env,
                    persisted_env_overrides=env,
                    variant=variant,
                )
                results.extend(ConfigExecutionExecutor(self, plan).execute())

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
            counter_dir: Optional path to override the embedded counter_folder config value
                via the ARTS per-variable env var mechanism.
        """
        result = self._run_process_request(
            BenchmarkProcessRequest(
                executable=executable,
                timeout=timeout,
                env=dict(env or {}),
                launcher=launcher,
                node_count=node_count,
                threads=threads,
                args=list(args or []),
                log_file=log_file,
                perf_enabled=perf_enabled,
                perf_interval=perf_interval,
                perf_output_name=perf_output_name,
                perf_output_dir=perf_output_dir,
                counter_dir=counter_dir,
            )
        )
        if result.status == Status.TIMEOUT:
            self._cleanup_port()
            time.sleep(0.5)
        return result

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

    def extract_startup_timings(self, output: str) -> Dict[str, float]:
        """Extract startup timing info from benchmark output."""
        return parse_startup_timings(output)

    def extract_verification_timings(self, output: str) -> Dict[str, float]:
        """Extract verification timing info from benchmark output."""
        return parse_verification_timings(output)

    def extract_cleanup_timings(self, output: str) -> Dict[str, float]:
        """Extract cleanup timing info from benchmark output."""
        return parse_cleanup_timings(output)

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
        """Parse perf stat CSV output and return aggregated cache metrics."""
        parsed = _shared_parse_perf_csv(perf_output)
        if parsed is None:
            return None
        return PerfCacheMetrics(
            cache_references=int(parsed.get("cache_references", 0)),
            cache_misses=int(parsed.get("cache_misses", 0)),
            l1d_loads=int(parsed.get("l1d_loads", 0)),
            l1d_load_misses=int(parsed.get("l1d_load_misses", 0)),
            cache_miss_rate=float(parsed.get("cache_miss_rate", 0.0)),
            l1d_load_miss_rate=float(parsed.get("l1d_load_miss_rate", 0.0)),
        )

    def verify_correctness(
        self,
        arts_result: RunResult,
        omp_result: RunResult,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> VerificationResult:
        """Compare ARTS output against OpenMP reference."""
        return verify_against_omp(
            arts_result.status,
            arts_result.checksum,
            omp_result.status,
            omp_result.checksum,
            tolerance,
        )

    def calculate_timing(
        self,
        arts_result: RunResult,
        omp_result: RunResult,
        report_speedup: bool = True,
    ) -> TimingResult:
        """Calculate speedup preferring kernel timings when available."""
        arts_kernel = get_kernel_time(arts_result)
        omp_kernel = get_kernel_time(omp_result)
        arts_e2e = get_e2e_time(arts_result)
        omp_e2e = get_e2e_time(omp_result)
        arts_startup = get_startup_time(arts_result)
        omp_startup = get_startup_time(omp_result)
        arts_verification = get_verification_time(arts_result)
        omp_verification = get_verification_time(omp_result)
        arts_cleanup = get_cleanup_time(arts_result)
        omp_cleanup = get_cleanup_time(omp_result)
        arts_total = arts_result.duration_sec
        omp_total = omp_result.duration_sec

        section_fields = dict(
            arts_startup_sec=arts_startup,
            omp_startup_sec=omp_startup,
            arts_verification_sec=arts_verification,
            omp_verification_sec=omp_verification,
            arts_cleanup_sec=arts_cleanup,
            omp_cleanup_sec=omp_cleanup,
        )

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
                arts_total_sec=arts_total,
                omp_total_sec=omp_total,
                speedup_basis=(
                    SPEEDUP_BASIS_E2E
                    if (arts_e2e is not None and omp_e2e is not None)
                    else SPEEDUP_BASIS_KERNEL
                    if (arts_kernel is not None and omp_kernel is not None)
                    else SPEEDUP_BASIS_TOTAL
                ),
                **section_fields,
            )

        # Prefer kernel timings when both are available, otherwise fall back to E2E timings,
        # otherwise fall back to total process duration.
        if arts_kernel is not None and omp_kernel is not None:
            arts_time = arts_kernel
            omp_time = omp_kernel
            speedup_basis = SPEEDUP_BASIS_KERNEL
        elif arts_e2e is not None and omp_e2e is not None:
            arts_time = arts_e2e
            omp_time = omp_e2e
            speedup_basis = SPEEDUP_BASIS_E2E
        else:
            arts_time = arts_total
            omp_time = omp_total
            speedup_basis = SPEEDUP_BASIS_TOTAL

        if not report_speedup:
            speedup = 0.0
            note = "Speedup hidden for distributed runs (unfair comparison)"
            speedup_basis = SPEEDUP_BASIS_NA
        elif arts_time == 0:
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
            arts_total_sec=arts_total,
            omp_total_sec=omp_total,
            speedup_basis=speedup_basis,
            **section_fields,
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
        arts_config: Optional[Path] = None,
        threads_override: Optional[int] = None,
        nodes_override: Optional[int] = None,
        launcher_override: Optional[str] = None,
        omp_threads_override: Optional[int] = None,
        counter_dir: Optional[Path] = None,
        compile_args: Optional[str] = None,
        phase_callback: Optional[Callable[[Phase], None]] = None,
        partial_results: Optional[Dict[str, Any]] = None,
        perf_enabled: bool = False,
        perf_interval: float = 0.1,
        run_number: int = 1,
        run_timestamp: str = "",
        perf_dir: Optional[Path] = None,
        cflags: str = "",
        variant: Optional[str] = None,
    ) -> BenchmarkResult:
        """Run complete pipeline for a single benchmark.

        Args:
            phase_callback: Optional callback invoked when phase changes.
                           Used by run_all to update live display.
            partial_results: Optional dict to store partial results as phases complete.
            variant: None=both, "arts"=ARTS only, "openmp"=OpenMP only.
        """
        bench_path = self.benchmarks_dir / name

        # Determine effective config template.
        effective_config = _resolve_effective_arts_config(bench_path, arts_config)

        base_threads = get_arts_cfg_int(effective_config, KEY_WORKER_THREADS) or 1
        base_nodes = get_arts_cfg_int(effective_config, KEY_NODE_COUNT) or 1
        base_launcher = get_arts_cfg_str(effective_config, KEY_LAUNCHER) or "ssh"

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

        # Skip benchmarks disabled for multi-node when running with node_count > 1.
        if desired_nodes > 1 and (bench_path / ".disable-multinode").exists():
            return self._make_skip_result(
                name, size,
                "Benchmark disabled for multi-node (has .disable-multinode marker)",
                config,
            )
        run_counter_dir: Optional[Path] = None
        if am:
            run_counter_dir = am.get_counter_dir(name, config, run_number)
        elif counter_dir is not None:
            run_counter_dir = counter_dir

        # Generate config with overrides only if values actually differ from base
        need_generated = False
        if threads_override is not None and threads_override != base_threads:
            need_generated = True
        if nodes_override is not None and nodes_override != base_nodes:
            need_generated = True
        if launcher_override is not None and launcher_override != base_launcher:
            need_generated = True
        effective_arts_cfg: Path
        if need_generated:
            effective_arts_cfg = generate_arts_config(
                effective_config,
                desired_threads,
                None,
                desired_launcher,
                nodes_override,
                benchmark_name=name,
            )
        else:
            effective_arts_cfg = effective_config

        # Clean before building to avoid stale artifacts
        if self.clean:
            self.clean_benchmark(name)

        build_output_dir: Optional[Path] = None
        if am:
            # Build directly in artifacts directory to keep outputs self-contained.
            build_output_dir = am.get_artifacts_dir(name, config)
            build_output_dir.mkdir(parents=True, exist_ok=True)

        execution = self._create_execution_context(
            name=name,
            size=size,
            bench_path=bench_path,
            config=config,
            effective_arts_cfg=effective_arts_cfg,
            desired_threads=desired_threads,
            desired_nodes=desired_nodes,
            desired_launcher=desired_launcher,
            actual_omp_threads=actual_omp_threads,
            effective_cflags=cflags,
            build_output_dir=build_output_dir,
        )
        plan = self._create_execution_plan(
            execution=execution,
            timeout=timeout,
            run_numbers=(run_number,),
            compile_args=compile_args,
            perf_enabled=perf_enabled,
            perf_interval=perf_interval,
            counter_dir=counter_dir,
            perf_dir=perf_dir,
            run_timestamp=run_timestamp,
            sweep_log_names=False,
            report_speedup=(desired_nodes == 1),
            env_overrides=self._create_common_env(),
            variant=variant,
        )
        hooks = ExecutionHooks(
            phase_callback=phase_callback,
            partial_results=partial_results,
        )
        return ConfigExecutionExecutor(self, plan).execute(hooks)[0]

    def run_all(
        self,
        benchmarks: List[str],
        size: str = DEFAULT_SIZE,
        timeout: int = DEFAULT_TIMEOUT,
        arts_config: Optional[Path] = None,
        threads_override: Optional[int] = None,
        nodes_override: Optional[int] = None,
        launcher_override: Optional[str] = None,
        omp_threads_override: Optional[int] = None,
        counter_dir: Optional[Path] = None,
        compile_args: Optional[str] = None,
        perf_enabled: bool = False,
        perf_interval: float = 0.1,
        runs: int = 1,
        run_timestamp: str = "",
        perf_dir: Optional[Path] = None,
        cflags: str = "",
        variant: Optional[str] = None,
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
                        arts_config,
                        threads_override,
                        nodes_override,
                        launcher_override,
                        omp_threads_override,
                        counter_dir,
                        compile_args,
                        perf_enabled=perf_enabled,
                        perf_interval=perf_interval,
                        run_number=run_num,
                        run_timestamp=run_timestamp,
                        perf_dir=perf_dir,
                        cflags=cflags,
                        variant=variant,
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
                            arts_config,
                            threads_override,
                            nodes_override,
                            launcher_override,
                            omp_threads_override,
                            counter_dir,
                            compile_args,
                            phase_callback,
                            current_partial[0],
                            perf_enabled=perf_enabled,
                            perf_interval=perf_interval,
                            run_number=run_num,
                            run_timestamp=run_timestamp,
                            perf_dir=perf_dir,
                            cflags=cflags,
                            variant=variant,
                        )
                    except Exception as e:
                        # Log error and continue to next benchmark
                        self.console.print(
                            f"[{Colors.ERROR}]Error running {bench}:[/{Colors.ERROR}] {e}")
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
                        self.console.print(
                            f"[{Colors.ERROR}]Error running {bench}:[/{Colors.ERROR}] {e}")
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
                        self.console.print(
                            f"[{Colors.ERROR}]Error running {bench}:[/{Colors.ERROR}] {e}")
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

        # macOS fallback: terminate any process IDs bound to tcp:{port}
        try:
            lsof_result = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return
        except Exception:
            return

        pids: List[int] = []
        for line in (lsof_result.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                pid = int(line)
            except ValueError:
                continue
            if pid != os.getpid():
                pids.append(pid)

        if not pids:
            return

        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                continue
            except Exception:
                continue

        deadline = time.time() + 2.0
        alive = set(pids)
        while alive and time.time() < deadline:
            remaining = set()
            for pid in alive:
                try:
                    os.kill(pid, 0)
                    remaining.add(pid)
                except ProcessLookupError:
                    continue
                except PermissionError:
                    continue
            if not remaining:
                break
            alive = remaining
            time.sleep(0.1)

        for pid in alive:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                continue
            except Exception:
                continue

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


_STATUS_TEXT_STYLES: dict[Status, tuple[str, str]] = {
    Status.PASS:    (Colors.SUCCESS, "PASS"),
    Status.FAIL:    (Colors.ERROR,   "FAIL"),
    Status.CRASH:   (Colors.ERROR,   "CRASH"),
    Status.TIMEOUT: (Colors.WARNING, "TIMEOUT"),
    Status.SKIP:    (Colors.DEBUG,   "SKIP"),
}


def status_text(status: Status) -> Text:
    """Create colored text for a status."""
    entry = _STATUS_TEXT_STYLES.get(status)
    if entry is None:
        return Text("N/A", style=Colors.DEBUG)
    style, label = entry
    return Text(label, style=style)


_STATUS_SYMBOL_STYLES: dict[Status, tuple[str, str]] = {
    Status.PASS:    (Colors.SUCCESS, Symbols.PASS),
    Status.FAIL:    (Colors.ERROR,   Symbols.FAIL),
    Status.CRASH:   (Colors.ERROR,   Symbols.FAIL),
    Status.TIMEOUT: (Colors.WARNING, Symbols.TIMEOUT),
    Status.SKIP:    (Colors.DEBUG,   Symbols.SKIP),
}


def status_symbol(status: Status) -> str:
    """Get symbol for a status."""
    entry = _STATUS_SYMBOL_STYLES.get(status)
    if entry is None:
        return f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"
    color, symbol = entry
    return f"[{color}]{symbol}[/{color}]"


def benchmark_result_passed(result: BenchmarkResult) -> bool:
    """Return True only when the full benchmark run completed and verified."""
    return (
        result.build_arts.status == Status.PASS
        and result.build_omp.status == Status.PASS
        and result.run_arts.status == Status.PASS
        and result.run_omp.status == Status.PASS
        and result.verification.correct
    )


def benchmark_result_failed(result: BenchmarkResult) -> bool:
    """Return True when any build/run/verification stage failed."""
    if result.build_arts.status == Status.FAIL or result.build_omp.status == Status.FAIL:
        return True
    if result.run_arts.status in (Status.FAIL, Status.CRASH, Status.TIMEOUT):
        return True
    if result.run_omp.status in (Status.FAIL, Status.CRASH, Status.TIMEOUT):
        return True
    if result.run_arts.status == Status.PASS and result.run_omp.status == Status.PASS:
        return not result.verification.correct
    return False


def benchmark_result_skipped(result: BenchmarkResult) -> bool:
    """Return True when the benchmark was intentionally skipped."""
    return (
        result.build_arts.status == Status.SKIP
        or result.build_omp.status == Status.SKIP
        or result.run_arts.status == Status.SKIP
        or result.run_omp.status == Status.SKIP
    ) and not benchmark_result_failed(result)


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


def get_startup_time(run_result: RunResult) -> Optional[float]:
    """Get total startup time from run result."""
    if run_result.startup_timings:
        return sum(run_result.startup_timings.values())
    return None


def get_verification_time(run_result: RunResult) -> Optional[float]:
    """Get total verification time from run result."""
    if run_result.verification_timings:
        return sum(run_result.verification_timings.values())
    return None


def get_cleanup_time(run_result: RunResult) -> Optional[float]:
    """Get total cleanup time from run result."""
    if run_result.cleanup_timings:
        return sum(run_result.cleanup_timings.values())
    return None


def _median(values: List[float]) -> Optional[float]:
    """Return the median of a non-empty value list."""
    return float(median(values)) if values else None


def detect_startup_outliers(
    values: List[float],
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Detect high startup outliers using a robust modified z-score policy."""
    effective_policy = dict(DEFAULT_STARTUP_OUTLIER_POLICY)
    if policy:
        effective_policy.update(policy)

    sample_count = len(values)
    flags: List[bool] = [False] * sample_count
    z_scores: List[float] = [0.0] * sample_count
    median_value: Optional[float] = _median(values)
    mad: Optional[float] = None
    threshold: Optional[float] = None

    if median_value is not None and (
        effective_policy.get("enabled", True)
        and sample_count >= int(effective_policy["min_runs"])
    ):
        deviations = [abs(v - median_value) for v in values]
        mad = _median(deviations) or 0.0
        min_startup = float(effective_policy["min_startup_sec"])
        min_relative = float(effective_policy["min_relative_multiplier"])
        floor_threshold = max(min_startup, median_value * min_relative)

        if mad <= 1e-12:
            threshold = floor_threshold
            for idx, value in enumerate(values):
                flags[idx] = value > threshold
                z_scores[idx] = float("inf") if flags[idx] else 0.0
        else:
            z_threshold = float(effective_policy["z_threshold"])
            robust_threshold = median_value + (z_threshold * mad / 0.6745)
            threshold = max(robust_threshold, floor_threshold)
            for idx, value in enumerate(values):
                z_score = (0.6745 * (value - median_value)) / mad
                z_scores[idx] = z_score
                flags[idx] = value > threshold and z_score > z_threshold
    elif median_value is not None:
        mad = 0.0

    return {
        "outliers": flags,
        "z_scores": z_scores,
        "median": median_value,
        "mad": mad,
        "threshold": threshold,
        "policy": effective_policy,
    }


def _collect_variant_outlier_map(
    runs: List[BenchmarkResult],
    *,
    variant: str,
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Return outlier classification for one variant keyed by run index."""
    sample_indices: List[int] = []
    startup_values: List[float] = []
    for idx, run in enumerate(runs):
        run_result = run.run_arts if variant == VARIANT_ARTS else run.run_omp
        startup = get_startup_time(run_result)
        if startup is None:
            continue
        sample_indices.append(idx)
        startup_values.append(float(startup))

    analysis = detect_startup_outliers(startup_values, policy=policy)
    details: Dict[int, Dict[str, Any]] = {}
    for sample_idx, run_idx in enumerate(sample_indices):
        details[run_idx] = {
            "variant": variant,
            "startup_sec": startup_values[sample_idx],
            KEY_IS_OUTLIER: bool(analysis["outliers"][sample_idx]),
            "z_score": analysis["z_scores"][sample_idx],
        }
    return details


def _compute_robust_summary(
    items: list,
    get_arts_e2e: Callable[[Any], Optional[float]],
    get_omp_e2e: Callable[[Any], Optional[float]],
    get_speedup: Callable[[Any], Optional[float]],
    get_is_outlier: Callable[[int, Any], Tuple[bool, bool]],
) -> Dict[str, Any]:
    """Shared core for robust median summarization with outlier filtering.

    Parameters
    ----------
    items : list
        Ordered collection of run data (BenchmarkResult objects or dicts).
    get_arts_e2e : callable
        ``(item) -> Optional[float]`` returning the ARTS e2e time.
    get_omp_e2e : callable
        ``(item) -> Optional[float]`` returning the OMP e2e time.
    get_speedup : callable
        ``(item) -> Optional[float]`` returning the speedup value.
    get_is_outlier : callable
        ``(index, item) -> (arts_is_outlier, omp_is_outlier)``

    Returns
    -------
    dict with keys: arts_value, omp_value, speedup_value, and per-metric
    raw / filtered counts plus outlier index sets.
    """
    arts_outlier_indices: Set[int] = set()
    omp_outlier_indices: Set[int] = set()
    for idx, item in enumerate(items):
        arts_out, omp_out = get_is_outlier(idx, item)
        if arts_out:
            arts_outlier_indices.add(idx)
        if omp_out:
            omp_outlier_indices.add(idx)

    arts_e2e_all: List[float] = []
    omp_e2e_all: List[float] = []
    arts_e2e_filtered: List[float] = []
    omp_e2e_filtered: List[float] = []
    speedup_all: List[float] = []
    speedup_filtered: List[float] = []

    for idx, item in enumerate(items):
        arts_val = get_arts_e2e(item)
        if arts_val is not None:
            arts_e2e_all.append(float(arts_val))
            if idx not in arts_outlier_indices:
                arts_e2e_filtered.append(float(arts_val))

        omp_val = get_omp_e2e(item)
        if omp_val is not None:
            omp_e2e_all.append(float(omp_val))
            if idx not in omp_outlier_indices:
                omp_e2e_filtered.append(float(omp_val))

        spd = get_speedup(item)
        if spd is not None and spd > 0.0:
            speedup_all.append(float(spd))
            if idx not in arts_outlier_indices and idx not in omp_outlier_indices:
                speedup_filtered.append(float(spd))

    arts_value = _median(arts_e2e_filtered) if arts_e2e_filtered else _median(arts_e2e_all)
    omp_value = _median(omp_e2e_filtered) if omp_e2e_filtered else _median(omp_e2e_all)
    speedup_value = (
        _median(speedup_filtered) if speedup_filtered else _median(speedup_all)
    )

    return {
        "arts_value": arts_value,
        "omp_value": omp_value,
        "speedup_value": speedup_value,
        "arts_raw_count": len(arts_e2e_all),
        "omp_raw_count": len(omp_e2e_all),
        "paired_raw_count": len(speedup_all),
        "arts_filtered_count": len(arts_e2e_filtered),
        "omp_filtered_count": len(omp_e2e_filtered),
        "paired_filtered_count": len(speedup_filtered),
        "arts_outlier_indices": arts_outlier_indices,
        "omp_outlier_indices": omp_outlier_indices,
    }


def summarize_runs_robust(
    runs: List[BenchmarkResult],
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute robust run-level aggregates for display/export."""
    if not runs:
        return {
            KEY_ARTS_E2E_SEC: None,
            KEY_OMP_E2E_SEC: None,
            KEY_SPEEDUP: None,
            KEY_ARTS_RAW_COUNT: 0,
            KEY_OMP_RAW_COUNT: 0,
            KEY_PAIRED_RAW_COUNT: 0,
            KEY_ARTS_FILTERED_COUNT: 0,
            KEY_OMP_FILTERED_COUNT: 0,
            KEY_PAIRED_FILTERED_COUNT: 0,
            "run_count": 0,
            KEY_ARTS_OUTLIERS: {},
            KEY_OMP_OUTLIERS: {},
        }

    arts_outliers = _collect_variant_outlier_map(runs, variant=VARIANT_ARTS, policy=policy)
    omp_outliers = _collect_variant_outlier_map(runs, variant=VARIANT_OMP, policy=policy)

    def _is_outlier(idx: int, _item: Any) -> Tuple[bool, bool]:
        a = arts_outliers.get(idx, {}).get(KEY_IS_OUTLIER, False)
        o = omp_outliers.get(idx, {}).get(KEY_IS_OUTLIER, False)
        return (bool(a), bool(o))

    core = _compute_robust_summary(
        items=runs,
        get_arts_e2e=lambda run: get_e2e_time(run.run_arts),
        get_omp_e2e=lambda run: get_e2e_time(run.run_omp),
        get_speedup=lambda run: (
            float(run.timing.speedup) if run.timing.speedup > 0 else None
        ),
        get_is_outlier=_is_outlier,
    )

    return {
        KEY_ARTS_E2E_SEC: core["arts_value"],
        KEY_OMP_E2E_SEC: core["omp_value"],
        KEY_SPEEDUP: core["speedup_value"],
        KEY_ARTS_RAW_COUNT: core["arts_raw_count"],
        KEY_OMP_RAW_COUNT: core["omp_raw_count"],
        KEY_PAIRED_RAW_COUNT: core["paired_raw_count"],
        KEY_ARTS_FILTERED_COUNT: core["arts_filtered_count"],
        KEY_OMP_FILTERED_COUNT: core["omp_filtered_count"],
        KEY_PAIRED_FILTERED_COUNT: core["paired_filtered_count"],
        "run_count": len(runs),
        KEY_ARTS_OUTLIERS: arts_outliers,
        KEY_OMP_OUTLIERS: omp_outliers,
    }


def _config_key(result: BenchmarkResult) -> Tuple[str, int, int, str]:
    """Return the canonical grouping key for a benchmark result.

    Groups by (name, arts_threads, arts_nodes, run_phase) so that runs
    with different phases are never mixed into the same statistical group.
    """
    return (
        result.name,
        result.config.arts_threads,
        result.config.arts_nodes,
        result.run_phase or "",
    )


def annotate_startup_outliers(
    results: List[BenchmarkResult],
    *,
    write_artifacts: bool = True,
) -> Dict[str, int]:
    """Annotate run results with startup outlier metadata and optional artifacts."""
    from collections import defaultdict

    grouped: Dict[Tuple[str, int, int, str], List[BenchmarkResult]] = defaultdict(list)
    for result in results:
        grouped[_config_key(result)].append(result)

    total_arts = 0
    total_omp = 0
    for _key, runs in grouped.items():
        runs_sorted = sorted(runs, key=lambda r: r.run_number)
        summary = summarize_runs_robust(runs_sorted)
        arts_outliers = summary[KEY_ARTS_OUTLIERS]
        omp_outliers = summary[KEY_OMP_OUTLIERS]

        for idx, run in enumerate(runs_sorted):
            arts_detail = arts_outliers.get(idx)
            omp_detail = omp_outliers.get(idx)
            run.run_arts.startup_outlier = arts_detail
            run.run_omp.startup_outlier = omp_detail
            arts_is_outlier = bool(arts_detail and arts_detail.get(KEY_IS_OUTLIER))
            omp_is_outlier = bool(omp_detail and omp_detail.get(KEY_IS_OUTLIER))
            if arts_is_outlier:
                total_arts += 1
            if omp_is_outlier:
                total_omp += 1
            run.run_arts.startup_diagnostics = _prepare_startup_diagnostics_for_persistence(
                run.run_arts.startup_diagnostics,
                keep=arts_is_outlier,
            )
            run.run_omp.startup_diagnostics = _prepare_startup_diagnostics_for_persistence(
                run.run_omp.startup_diagnostics,
                keep=omp_is_outlier,
            )
            if write_artifacts:
                _write_startup_outlier_artifacts(run)

    return {"arts_outliers": total_arts, "omp_outliers": total_omp}


def _truncate_lines(
    lines: Any,
    max_lines: int,
) -> Tuple[Any, Optional[int]]:
    """Truncate a list of lines, returning (truncated_list, dropped_count or None)."""
    if not isinstance(lines, list) or len(lines) <= max_lines:
        return lines, None
    return lines[:max_lines], len(lines) - max_lines


def _prepare_startup_diagnostics_for_persistence(
    diagnostics: Dict[str, Any],
    *,
    keep: bool,
) -> Dict[str, Any]:
    """Persist diagnostics for outliers only, trimming large snapshots for readability."""
    if not keep or not diagnostics:
        return {}

    payload = dict(diagnostics)
    for key in (
        "network_snapshot_pre",
        "network_snapshot_post",
        "process_snapshot_pre",
        "process_snapshot_post",
    ):
        snapshot = payload.get(key)
        if not isinstance(snapshot, dict):
            continue
        snap = dict(snapshot)
        snap["stdout"], dropped = _truncate_lines(snap.get("stdout"), 80)
        if dropped is not None:
            snap["stdout_truncated"] = dropped
        snap["stderr"], dropped = _truncate_lines(snap.get("stderr"), 40)
        if dropped is not None:
            snap["stderr_truncated"] = dropped
        payload[key] = snap

    # stdout_preview and stderr_preview are already capped at 20 lines by
    # _preview_lines in benchmark_execution.py, so this is a no-op safety net.
    for preview_key in ("stdout_preview", "stderr_preview"):
        truncated, dropped = _truncate_lines(payload.get(preview_key), 20)
        payload[preview_key] = truncated
        if dropped is not None:
            payload[f"{preview_key}_truncated"] = dropped

    return payload


def _write_startup_outlier_artifacts(result: BenchmarkResult) -> None:
    """Write per-run startup diagnostics artifacts when outliers were flagged."""
    if not (
        (result.run_arts.startup_outlier and result.run_arts.startup_outlier.get(KEY_IS_OUTLIER))
        or (result.run_omp.startup_outlier and result.run_omp.startup_outlier.get(KEY_IS_OUTLIER))
    ):
        return

    run_dir = result.artifacts.run_dir
    if not run_dir:
        return

    path = Path(run_dir)
    if not path.exists():
        return

    payload: Dict[str, Any] = {
        "benchmark": result.name,
        "run_number": result.run_number,
        "size": result.size,
        "threads": result.config.arts_threads,
        "nodes": result.config.arts_nodes,
        "run_phase": result.run_phase,
        "timestamp": datetime.now().isoformat(),
        "startup_outliers": {
            VARIANT_ARTS: result.run_arts.startup_outlier,
            VARIANT_OMP: result.run_omp.startup_outlier,
        },
        "startup_timings": {
            VARIANT_ARTS: result.run_arts.startup_timings,
            VARIANT_OMP: result.run_omp.startup_timings,
        },
        "diagnostics": {
            VARIANT_ARTS: result.run_arts.startup_diagnostics,
            VARIANT_OMP: result.run_omp.startup_diagnostics,
        },
    }

    out_path = path / STARTUP_OUTLIER_DIAGNOSTICS_FILENAME
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)


def format_kernel_time(run_result: RunResult) -> Tuple[Optional[float], str]:
    """Format kernel time for display. Returns (total_time, display_string).

    For single kernel: returns (time, "0.1234s")
    For multiple kernels: returns (sum, "0.5678s [3]") where [3] is kernel count
    """
    if not run_result.kernel_timings:
        return None, ""

    total = sum(run_result.kernel_timings.values())
    count = len(run_result.kernel_timings)

    return total, f"{total:.2f}s"


def format_e2e_time(run_result: RunResult) -> Tuple[Optional[float], str]:
    """Format end-to-end time for display. Returns (total_time, display_string)."""
    if not run_result.e2e_timings:
        return None, ""
    total = sum(run_result.e2e_timings.values())
    return total, f"{total:.2f}s"


def create_results_table(results: List[BenchmarkResult]) -> Table:
    """Create a rich table from benchmark results."""
    table = Table(box=box.ROUNDED, show_header=True, header_style=Colors.HIGHLIGHT)

    table.add_column("Benchmark", style=Colors.INFO, no_wrap=True)
    table.add_column("ARTS E2E", justify="right")
    table.add_column("OMP E2E", justify="right")
    table.add_column("A.Startup", justify="right")
    table.add_column("O.Startup", justify="right")
    table.add_column("A.Kernel", justify="right")
    table.add_column("O.Kernel", justify="right")
    table.add_column("A.Verify", justify="right")
    table.add_column("O.Verify", justify="right")
    table.add_column("A.Cleanup", justify="right")
    table.add_column("O.Cleanup", justify="right")
    table.add_column("Correct", justify="center")
    table.add_column("Speedup", justify="right")

    has_fallback = False
    for r in results:
        arts_e2e, arts_e2e_str = format_e2e_time(r.run_arts)
        omp_e2e, omp_e2e_str = format_e2e_time(r.run_omp)

        # ARTS E2E: skip → compile failure → runtime failure → success
        if r.build_arts.status == Status.SKIP:
            run_arts = f"[{Colors.DEBUG}]- Skip[/{Colors.DEBUG}]"
        elif r.build_arts.status != Status.PASS:
            run_arts = f"[{Colors.ERROR}]{Symbols.FAIL} Compile[/{Colors.ERROR}]"
        elif r.run_arts.status == Status.PASS:
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
            run_arts = f"[{Colors.ERROR}]{Symbols.FAIL} Runtime[/{Colors.ERROR}]"

        # OMP E2E: skip → compile failure → runtime failure → success
        if r.build_omp.status == Status.SKIP:
            run_omp = f"[{Colors.DEBUG}]- Skip[/{Colors.DEBUG}]"
        elif r.build_omp.status != Status.PASS:
            run_omp = f"[{Colors.ERROR}]{Symbols.FAIL} Compile[/{Colors.ERROR}]"
        elif r.run_omp.status == Status.PASS:
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
            run_omp = f"[{Colors.ERROR}]{Symbols.FAIL} Runtime[/{Colors.ERROR}]"

        # Correctness
        if r.verification.correct:
            correct = f"[{Colors.SUCCESS}]{Symbols.PASS} YES[/{Colors.SUCCESS}]"
        elif r.run_arts.status != Status.PASS or r.run_omp.status != Status.PASS:
            correct = f"[{Colors.DEBUG}]- N/A[/{Colors.DEBUG}]"
        else:
            correct = f"[{Colors.ERROR}]{Symbols.FAIL} NO[/{Colors.ERROR}]"

        # Section times (both ARTS and OMP)
        def _fmt_sec(val: "Optional[float]") -> str:
            return f"{val:.2f}s" if val is not None else f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"

        startup_arts = _fmt_sec(r.timing.arts_startup_sec)
        startup_omp = _fmt_sec(r.timing.omp_startup_sec)
        kernel_arts = _fmt_sec(r.timing.arts_kernel_sec)
        kernel_omp = _fmt_sec(r.timing.omp_kernel_sec)
        verify_arts = _fmt_sec(r.timing.arts_verification_sec)
        verify_omp = _fmt_sec(r.timing.omp_verification_sec)
        cleanup_arts = _fmt_sec(r.timing.arts_cleanup_sec)
        cleanup_omp = _fmt_sec(r.timing.omp_cleanup_sec)

        # Speedup (basis chosen in calculate_timing)
        if r.timing.speedup > 0:
            if r.timing.speedup >= 1.0:
                speedup = f"[{Colors.SUCCESS}]{r.timing.speedup:.2f}x[/{Colors.SUCCESS}]"
            elif r.timing.speedup >= 0.8:
                speedup = f"[{Colors.WARNING}]{r.timing.speedup:.2f}x[/{Colors.WARNING}]"
            else:
                speedup = f"[{Colors.ERROR}]{r.timing.speedup:.2f}x[/{Colors.ERROR}]"
            if r.timing.speedup_basis != SPEEDUP_BASIS_KERNEL:
                speedup += "*"
                has_fallback = True
        else:
            speedup = f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"

        table.add_row(
            r.name,
            run_arts,
            run_omp,
            startup_arts,
            startup_omp,
            kernel_arts,
            kernel_omp,
            verify_arts,
            verify_omp,
            cleanup_arts,
            cleanup_omp,
            correct,
            speedup,
        )

    if has_fallback:
        table.caption = f"[{Colors.DEBUG}]* = speedup not based on kernel[/{Colors.DEBUG}]"

    return table


def create_summary_panel(results: List[BenchmarkResult], duration: float) -> Panel:
    """Create a summary panel."""
    passed = sum(1 for r in results if benchmark_result_passed(r))
    failed = sum(1 for r in results if benchmark_result_failed(r))
    skipped = sum(1 for r in results if benchmark_result_skipped(r))

    # Calculate geometric mean from per-config robust median speedups.
    from collections import defaultdict
    import math

    grouped: Dict[Tuple[str, int, int, str], List[BenchmarkResult]] = defaultdict(list)
    for result in results:
        grouped[_config_key(result)].append(result)

    speedups: List[float] = []
    for runs in grouped.values():
        robust = summarize_runs_robust(sorted(runs, key=lambda r: r.run_number))
        if robust[KEY_SPEEDUP] and robust[KEY_SPEEDUP] > 0:
            speedups.append(float(robust[KEY_SPEEDUP]))

    if speedups:
        bases = {r.timing.speedup_basis for r in results if r.timing.speedup > 0}
        basis_label = next(iter(bases)) if len(bases) == 1 else "mixed"
        geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        speedup_text = (
            f"Geometric mean speedup ({basis_label}, {DEFAULT_REPORTING_MODE} filtered): "
            f"[{Colors.INFO}]{geomean:.2f}x[/{Colors.INFO}]"
        )
    else:
        speedup_text = ""

    arts_outliers = sum(
        1
        for r in results
        if r.run_arts.startup_outlier and r.run_arts.startup_outlier.get(KEY_IS_OUTLIER)
    )
    omp_outliers = sum(
        1
        for r in results
        if r.run_omp.startup_outlier and r.run_omp.startup_outlier.get(KEY_IS_OUTLIER)
    )

    content = (
        f"{format_passed(passed)}  "
        f"{format_failed(failed)}  "
        f"{format_skipped(skipped)}  "
        f"[{Colors.INFO}]{Symbols.TIMEOUT} {format_duration(duration)}[/{Colors.INFO}]"
    )

    content += f"\n\nReporting: [{Colors.INFO}]{DEFAULT_REPORTING_MODE}-of-N[/{Colors.INFO}] with startup outlier filtering"
    content += f"\nStartup outliers: ARTS={arts_outliers}, OpenMP={omp_outliers}"

    if speedup_text:
        content += f"\n\n{speedup_text}"

    return Panel(content, title="Summary", border_style=Colors.STEP)


# NOTE: SVG/report generation code was removed.
# Benchmark reports are generated automatically into each results directory.



def _format_with_filter_marker(
    value_str: str,
    filtered_count: int,
    raw_count: int,
    marker: str = "\u2020",
) -> Tuple[str, bool]:
    """Append a filtered-marker to *value_str* when outliers were removed.

    Returns ``(decorated_string, marker_was_appended)``.
    """
    if filtered_count > 0 and raw_count > filtered_count:
        return value_str + marker, True
    return value_str, False


def create_live_table(
    benchmarks: List[str],
    results: Dict[str, List[BenchmarkResult]],
    in_progress: Optional[str] = None,
    current_phase: Optional[Phase] = None,
    current_partial: Optional[Dict[str, Any]] = None,
    total_runs: int = 1,
) -> Table:
    """Create a live-updating table showing benchmark progress with running statistics."""
    table = Table(box=box.ROUNDED, show_header=True, header_style=Colors.HIGHLIGHT)

    table.add_column("Benchmark", style=Colors.INFO, no_wrap=True)
    table.add_column("ARTS E2E", justify="right")
    table.add_column("OMP E2E", justify="right")
    table.add_column("A.Startup", justify="right")
    table.add_column("O.Startup", justify="right")
    table.add_column("A.Kernel", justify="right")
    table.add_column("O.Kernel", justify="right")
    table.add_column("A.Verify", justify="right")
    table.add_column("O.Verify", justify="right")
    table.add_column("A.Cleanup", justify="right")
    table.add_column("O.Cleanup", justify="right")
    table.add_column("Correct", justify="center")
    table.add_column("Speedup", justify="right")

    has_fallback = False
    has_filtered = False
    for bench in benchmarks:
        if bench in results and results[bench]:
            # Completed runs - show running statistics
            runs_list = results[bench]
            r = runs_list[-1]  # Latest result for status checks
            robust = summarize_runs_robust(runs_list)
            filtered_used = False

            # ARTS E2E: skip → compile failure → runtime failure → robust median
            if r.build_arts.status == Status.SKIP:
                run_arts = f"[{Colors.DEBUG}]- Skip[/{Colors.DEBUG}]"
            elif r.build_arts.status != Status.PASS:
                run_arts = f"[{Colors.ERROR}]{Symbols.FAIL} Compile[/{Colors.ERROR}]"
            else:
                arts_e2e = robust[KEY_ARTS_E2E_SEC]
                if arts_e2e is not None:
                    run_arts, arts_marked = _format_with_filter_marker(
                        f"[{Colors.SUCCESS}]{Symbols.PASS}[/{Colors.SUCCESS}] {arts_e2e:.2f}s",
                        robust[KEY_ARTS_FILTERED_COUNT],
                        robust[KEY_ARTS_RAW_COUNT],
                    )
                    if arts_marked:
                        filtered_used = True
                elif r.run_arts.status == Status.PASS:
                    arts_kernel, arts_kernel_str = format_kernel_time(r.run_arts)
                    if arts_kernel is not None:
                        run_arts = f"[{Colors.SUCCESS}]{Symbols.PASS}[/{Colors.SUCCESS}] {arts_kernel_str}*"
                    else:
                        run_arts = f"[{Colors.SUCCESS}]{Symbols.PASS}[/{Colors.SUCCESS}] {r.run_arts.duration_sec:.2f}s*"
                    has_fallback = True
                else:
                    run_arts = f"[{Colors.ERROR}]{Symbols.FAIL} Runtime[/{Colors.ERROR}]"

            # OMP E2E: skip → compile failure → runtime failure → robust median
            if r.build_omp.status == Status.SKIP:
                run_omp = f"[{Colors.DEBUG}]- Skip[/{Colors.DEBUG}]"
            elif r.build_omp.status != Status.PASS:
                run_omp = f"[{Colors.ERROR}]{Symbols.FAIL} Compile[/{Colors.ERROR}]"
            else:
                omp_e2e = robust[KEY_OMP_E2E_SEC]
                if omp_e2e is not None:
                    run_omp, omp_marked = _format_with_filter_marker(
                        f"[{Colors.SUCCESS}]{Symbols.PASS}[/{Colors.SUCCESS}] {omp_e2e:.2f}s",
                        robust[KEY_OMP_FILTERED_COUNT],
                        robust["omp_raw_count"],
                    )
                    if omp_marked:
                        filtered_used = True
                elif r.run_omp.status == Status.PASS:
                    omp_kernel, omp_kernel_str = format_kernel_time(r.run_omp)
                    if omp_kernel is not None:
                        run_omp = f"[{Colors.SUCCESS}]{Symbols.PASS}[/{Colors.SUCCESS}] {omp_kernel_str}*"
                    else:
                        run_omp = f"[{Colors.SUCCESS}]{Symbols.PASS}[/{Colors.SUCCESS}] {r.run_omp.duration_sec:.2f}s*"
                    has_fallback = True
                else:
                    run_omp = f"[{Colors.ERROR}]{Symbols.FAIL} Runtime[/{Colors.ERROR}]"

            # Correctness (based on latest run)
            if r.verification.correct:
                correct = f"[{Colors.SUCCESS}]{Symbols.PASS} YES[/{Colors.SUCCESS}]"
            elif r.run_arts.status != Status.PASS or r.run_omp.status != Status.PASS:
                correct = f"[{Colors.DEBUG}]- N/A[/{Colors.DEBUG}]"
            else:
                correct = f"[{Colors.ERROR}]{Symbols.FAIL} NO[/{Colors.ERROR}]"

            # Speedup (median on startup-filtered paired runs)
            speedup_median = robust[KEY_SPEEDUP]
            if speedup_median:
                if speedup_median >= 1.0:
                    speedup = f"[{Colors.SUCCESS}]{speedup_median:.2f}x[/{Colors.SUCCESS}]"
                elif speedup_median >= 0.8:
                    speedup = f"[{Colors.WARNING}]{speedup_median:.2f}x[/{Colors.WARNING}]"
                else:
                    speedup = f"[{Colors.ERROR}]{speedup_median:.2f}x[/{Colors.ERROR}]"
                speedup, spd_marked = _format_with_filter_marker(
                    speedup,
                    robust["paired_filtered_count"],
                    robust["paired_raw_count"],
                )
                if spd_marked:
                    filtered_used = True
                if r.timing.speedup_basis != SPEEDUP_BASIS_KERNEL:
                    speedup += "*"
                    has_fallback = True
            else:
                speedup = f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"

            # Section times (both ARTS and OMP, latest run)
            def _fmt_sec(val: "Optional[float]") -> str:
                return f"{val:.2f}s" if val is not None else f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"

            startup_arts = _fmt_sec(r.timing.arts_startup_sec)
            startup_omp = _fmt_sec(r.timing.omp_startup_sec)
            kernel_arts = _fmt_sec(r.timing.arts_kernel_sec)
            kernel_omp = _fmt_sec(r.timing.omp_kernel_sec)
            verify_arts = _fmt_sec(r.timing.arts_verification_sec)
            verify_omp = _fmt_sec(r.timing.omp_verification_sec)
            cleanup_arts = _fmt_sec(r.timing.arts_cleanup_sec)
            cleanup_omp = _fmt_sec(r.timing.omp_cleanup_sec)

            table.add_row(bench, run_arts, run_omp,
                          startup_arts, startup_omp,
                          kernel_arts, kernel_omp,
                          verify_arts, verify_omp,
                          cleanup_arts, cleanup_omp,
                          correct, speedup)
            if filtered_used:
                has_filtered = True

        elif bench == in_progress:
            # Currently running - show phase-specific indicator
            if current_phase == Phase.BUILD_ARTS:
                run_arts = f"[{Colors.WARNING}]{Symbols.RUNNING} Building...[/{Colors.WARNING}]"
                run_omp = f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"
                correct = f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"
                speedup = f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"
            elif current_phase == Phase.BUILD_OMP:
                # ARTS build done or skipped
                if current_partial and "build_arts" in current_partial:
                    ba = current_partial["build_arts"]
                    if ba.status == Status.SKIP:
                        run_arts = f"[{Colors.DEBUG}]- Skip[/{Colors.DEBUG}]"
                    elif ba.status == Status.PASS:
                        run_arts = f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"
                    else:
                        run_arts = f"[{Colors.ERROR}]{Symbols.FAIL} Compile[/{Colors.ERROR}]"
                else:
                    run_arts = f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"
                run_omp = f"[{Colors.WARNING}]{Symbols.RUNNING} Building...[/{Colors.WARNING}]"
                correct = f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"
                speedup = f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"
            elif current_phase == Phase.RUN_ARTS:
                run_arts = f"[{Colors.WARNING}]{Symbols.RUNNING} Running...[/{Colors.WARNING}]"
                run_omp = f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"
                correct = f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"
                speedup = f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"
            elif current_phase == Phase.RUN_OMP:
                # ARTS run completed, show e2e time if available in partial results
                if current_partial and "run_arts" in current_partial:
                    run_arts_result = current_partial["run_arts"]
                    if run_arts_result.status == Status.SKIP:
                        run_arts = f"[{Colors.DEBUG}]- Skip[/{Colors.DEBUG}]"
                    elif run_arts_result.status == Status.PASS:
                        arts_e2e, arts_e2e_str = format_e2e_time(run_arts_result)
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
                        run_arts = f"[{Colors.ERROR}]{Symbols.FAIL} Runtime[/{Colors.ERROR}]"
                else:
                    run_arts = f"[{Colors.SUCCESS}]{Symbols.PASS}[/{Colors.SUCCESS}]"
                run_omp = f"[{Colors.WARNING}]{Symbols.RUNNING} Running...[/{Colors.WARNING}]"
                correct = f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"
                speedup = f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"
            else:
                run_arts = f"[{Colors.WARNING}]{Symbols.RUNNING}...[/{Colors.WARNING}]"
                run_omp = f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"
                correct = f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"
                speedup = f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]"
            table.add_row(
                f"[{Colors.HIGHLIGHT}]{bench}[/{Colors.HIGHLIGHT}]",
                run_arts,
                run_omp,
                f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]", f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]", f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]", f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]",
                f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]", f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]", f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]", f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]",
                correct,
                speedup,
            )
        else:
            # Pending - show placeholder
            table.add_row(
                f"[{Colors.DEBUG}]{bench}[/{Colors.DEBUG}]",
                f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]",
                f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]",
                f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]", f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]", f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]", f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]",
                f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]", f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]", f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]", f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]",
                f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]",
                f"[{Colors.DEBUG}]-[/{Colors.DEBUG}]",
            )

    caption_notes: List[str] = []
    if has_fallback:
        caption_notes.append("* = speedup not based on kernel")
    if has_filtered:
        caption_notes.append("\u2020 = startup outliers filtered (median-of-N)")
    if caption_notes:
        table.caption = f"[{Colors.DEBUG}]" + " | ".join(caption_notes) + f"[/{Colors.DEBUG}]"

    return table


def create_live_summary(
    results: Dict[str, List[BenchmarkResult]],
    total: int,
    elapsed: float,
) -> Text:
    """Create a one-line summary for live display."""
    # Count passed/failed using latest result from each benchmark
    passed = sum(1 for runs in results.values()
                 if runs and benchmark_result_passed(runs[-1]))
    failed = sum(1 for runs in results.values()
                 if runs and benchmark_result_failed(runs[-1]))
    pending = total - len(results)

    text = Text()
    text.append(f"{Symbols.PASS} {passed} passed  ", style=Colors.SUCCESS)
    text.append(f"{Symbols.FAIL} {failed} failed  ", style=Colors.ERROR)
    text.append(f"{Symbols.SKIP} {pending} pending  ", style=Colors.DEBUG)
    text.append(f"{Symbols.TIMEOUT} {elapsed:.1f}s", style=Colors.DEBUG)
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
        status_text = f"[{Colors.HEADER}]{Symbols.RUNNING} Running:[/{Colors.HEADER}] [{Colors.HIGHLIGHT}]{in_progress}[/{Colors.HIGHLIGHT}] [{Colors.DEBUG}]({phase_text})[/{Colors.DEBUG}]"
        status_panel = Panel(status_text, box=box.ROUNDED, style=Colors.STEP, padding=(0, 1))
        components.append(status_panel)

    table = create_live_table(
        benchmarks, results, in_progress, current_phase, current_partial, total_runs)
    summary = create_live_summary(results, len(benchmarks), elapsed)

    components.extend([table, summary])
    return Group(*components)


# ============================================================================
# JSON Export
# ============================================================================


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

    # Group by config (name + threads + nodes + run_phase)
    groups: Dict[Tuple[str, int, int, str], List[BenchmarkResult]] = defaultdict(list)
    for r in results:
        groups[_config_key(r)].append(r)

    stats = {}
    for key, runs in groups.items():
        _name, threads, nodes, run_phase = key
        robust = summarize_runs_robust(sorted(runs, key=lambda r: r.run_number))
        # Extract timings
        arts_build_times = []
        omp_build_times = []
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
        if run_phase:
            config_key = f"{config_key}_{run_phase}"

        stats[config_key] = {
            "reporting_mode": DEFAULT_REPORTING_MODE,
            "arts_build_time": compute_stats(arts_build_times),
            "omp_build_time": compute_stats(omp_build_times),
            "arts_e2e_time": compute_stats(arts_e2e_times),
            "omp_e2e_time": compute_stats(omp_e2e_times),
            "arts_e2e_median_filtered": robust[KEY_ARTS_E2E_SEC],
            "omp_e2e_median_filtered": robust[KEY_OMP_E2E_SEC],
            "speedup_median_filtered": robust[KEY_SPEEDUP],
            "arts_e2e_filtered_count": robust[KEY_ARTS_FILTERED_COUNT],
            "omp_e2e_filtered_count": robust[KEY_OMP_FILTERED_COUNT],
            "paired_speedup_filtered_count": robust["paired_filtered_count"],
            "arts_e2e_raw_count": robust[KEY_ARTS_RAW_COUNT],
            "omp_e2e_raw_count": robust["omp_raw_count"],
            "paired_speedup_raw_count": robust["paired_raw_count"],
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
    nodes_list: Optional[List[int]] = None,
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
    experiment_name: Optional[str] = None,
    experiment_description: Optional[str] = None,
    experiment_steps: Optional[List[Dict[str, Any]]] = None,
    startup_outlier_counts: Optional[Dict[str, int]] = None,
) -> None:
    """Export results to JSON file with comprehensive reproducibility metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    carts_dir = get_carts_dir()
    benchmarks_dir = BENCHMARKS_DIR

    # Collect comprehensive reproducibility metadata
    repro_metadata = get_reproducibility_metadata(carts_dir, benchmarks_dir)

    # Collect experiment metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "hostname": platform.node(),
        "size": size,
        "total_duration_seconds": total_duration,
        "runs_per_config": runs_per_config,
        "reporting": {
            "mode": DEFAULT_REPORTING_MODE,
            "startup_outlier_filter": "see top-level startup_outlier_policy",
            "startup_diagnostics": {
                "capture": "outlier_runs",
                "artifact": STARTUP_OUTLIER_DIAGNOSTICS_FILENAME,
            },
        },
        # Include reproducibility bundle
        "reproducibility": repro_metadata,
    }

    # Add experiment configuration
    if threads_list:
        metadata["thread_sweep"] = threads_list
    if nodes_list:
        metadata["node_sweep"] = nodes_list
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
    if experiment_name:
        metadata["experiment_name"] = experiment_name
    if experiment_description:
        metadata["experiment_description"] = experiment_description
    if experiment_steps:
        metadata["experiment_steps"] = experiment_steps
    if weak_scaling:
        metadata["weak_scaling"] = {
            "enabled": True,
            "base_size": base_size,
        }
    if artifacts_directory:
        metadata["artifacts_directory"] = artifacts_directory

    # Calculate summary
    passed = sum(1 for r in results if benchmark_result_passed(r))
    failed = sum(1 for r in results if benchmark_result_failed(r))
    skipped = sum(1 for r in results if benchmark_result_skipped(r))
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
        "reporting_mode": DEFAULT_REPORTING_MODE,
    }
    if startup_outlier_counts:
        summary["startup_outliers"] = dict(startup_outlier_counts)

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
            "run_phase": r.run_phase,
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
                "startup_timings": r.run_arts.startup_timings,
                "startup_total_sec": get_startup_time(r.run_arts),
                "startup_outlier": r.run_arts.startup_outlier,
                "startup_diagnostics": r.run_arts.startup_diagnostics,
                "verification_timings": r.run_arts.verification_timings,
                "cleanup_timings": r.run_arts.cleanup_timings,
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
                "startup_timings": r.run_omp.startup_timings,
                "startup_total_sec": get_startup_time(r.run_omp),
                "startup_outlier": r.run_omp.startup_outlier,
                "startup_diagnostics": r.run_omp.startup_diagnostics,
                "verification_timings": r.run_omp.verification_timings,
                "cleanup_timings": r.run_omp.cleanup_timings,
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
                "arts_startup_sec": r.timing.arts_startup_sec,
                "omp_startup_sec": r.timing.omp_startup_sec,
                "arts_verification_sec": r.timing.arts_verification_sec,
                "omp_verification_sec": r.timing.omp_verification_sec,
                "arts_cleanup_sec": r.timing.arts_cleanup_sec,
                "omp_cleanup_sec": r.timing.omp_cleanup_sec,
                "arts_total_sec": r.timing.arts_total_sec,
                "omp_total_sec": r.timing.omp_total_sec,
                "note": r.timing.note,
            },
            "verification": {
                "correct": r.verification.correct,
                "tolerance": r.verification.tolerance_used,
                "note": r.verification.note,
                "arts_checksum": r.verification.arts_checksum,
                "omp_checksum": r.verification.omp_checksum,
                "reference_checksum": r.verification.reference_checksum,
                "reference_source": r.verification.reference_source,
                "mode": r.verification.mode,
                "reference_omp_threads": r.verification.reference_omp_threads,
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
        "startup_outlier_policy": dict(DEFAULT_STARTUP_OUTLIER_POLICY),
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
            f"\n[{Colors.HIGHLIGHT}]Available CARTS Benchmarks[/{Colors.HIGHLIGHT}] ({len(benchmarks)} total)\n")

        for suite_name in sorted(suites.keys()):
            if suite_name:
                console.print(f"[{Colors.INFO}]{suite_name}:[/{Colors.INFO}]")
            for bench in sorted(suites[suite_name]):
                console.print(f"  {bench}")
            console.print()


def _parse_bool_flag(value: Any) -> bool:
    """Parse common bool-like values from JSON/CLI step definitions."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _parse_step_benchmarks(raw: Any) -> Optional[List[str]]:
    """Parse optional step benchmark selection."""
    if raw is None:
        return None

    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        parts = [p.strip() for p in re.split(r"[,\|;]", text) if p.strip()]
        return parts or None

    if isinstance(raw, list):
        parsed = [str(item).strip() for item in raw if str(item).strip()]
        return parsed or None

    raise ValueError("Step field `benchmarks` must be a list or string")


KNOWN_STEP_KEYS = {
    "name",
    "benchmarks",
    "profile",
    "debug",
    "runs",
    "perf",
    "perf_interval",
    "size",
    "threads",
    "nodes",
    "timeout",
    "cflags",
    "compile_args",
    "exclude_nodes",
    "arts_config",
    "launcher",
    "description",
}


def _make_experiment_step(
    data: Dict[str, Any],
    default_name: str,
    base_dir: Optional[Path] = None,
) -> ExperimentStep:
    """Create an ExperimentStep from a dictionary payload."""
    normalized = {str(k).replace("-", "_"): v for k, v in data.items()}
    step_name = str(normalized.get("name", default_name))

    def _resolve_path(raw: Any, field: str) -> Optional[str]:
        if raw is None or str(raw).strip() == "":
            return None
        p = Path(str(raw).strip())
        if p.is_absolute():
            return str(p)

        if base_dir:
            base_candidate = (base_dir / p).resolve()
            if base_candidate.exists():
                return str(base_candidate)

        search_dirs: List[Path] = []
        if field == "arts_config":
            # Benchmark experiments live in the carts-benchmarks submodule, but
            # the Docker runtime config lives in the parent CARTS repo.
            search_dirs = [get_carts_dir(), CONFIGS_DIR]
        elif field == "profile":
            search_dirs = [PROFILES_DIR]

        for search_dir in search_dirs:
            candidate = (search_dir / p).resolve()
            if candidate.exists():
                return str(candidate)

        if p.exists():
            return str(p.resolve())

        raise ValueError(f"Cannot resolve {field} '{raw}' - file not found")

    unknown = sorted(set(normalized.keys()) - KNOWN_STEP_KEYS)
    if unknown:
        console.print(
            f"[{Colors.WARNING}]Warning:[/{Colors.WARNING}] Unknown step keys: {', '.join(unknown)}"
        )

    step = ExperimentStep(
        name=step_name,
        description=(
            str(normalized["description"])
            if normalized.get("description") is not None
            else None
        ),
        benchmarks=_parse_step_benchmarks(normalized.get("benchmarks")),
        profile=_resolve_path(normalized.get("profile"), "profile"),
        debug=int(normalized.get("debug", 0) or 0),
        runs=int(normalized.get("runs", 1) or 1),
        perf=_parse_bool_flag(normalized.get("perf", False)),
        perf_interval=float(normalized.get("perf_interval", 0.1) or 0.1),
        size=parse_size(str(normalized["size"]), f"step '{step_name}' size")
        if normalized.get("size") is not None else None,
        threads=str(normalized["threads"]) if normalized.get("threads") is not None else None,
        nodes=str(normalized["nodes"]) if normalized.get("nodes") is not None else None,
        timeout=int(normalized["timeout"]) if normalized.get("timeout") is not None else None,
        cflags=str(normalized["cflags"]) if normalized.get("cflags") is not None else None,
        compile_args=str(normalized["compile_args"]) if normalized.get("compile_args") is not None else None,
        exclude_nodes=str(normalized["exclude_nodes"]) if normalized.get("exclude_nodes") is not None else None,
        arts_config=_resolve_path(normalized.get("arts_config"), "arts_config"),
        launcher=str(normalized["launcher"]) if normalized.get("launcher") is not None else None,
    )
    setattr(step, "_has_runs", "runs" in normalized and normalized.get("runs") is not None)
    setattr(step, "_has_perf", "perf" in normalized and normalized.get("perf") is not None)
    setattr(
        step,
        "_has_perf_interval",
        "perf_interval" in normalized and normalized.get("perf_interval") is not None,
    )
    setattr(step, "_has_size", "size" in normalized and normalized.get("size") is not None)
    setattr(step, "_has_threads", "threads" in normalized and normalized.get("threads") is not None)
    setattr(step, "_has_nodes", "nodes" in normalized and normalized.get("nodes") is not None)
    setattr(step, "_has_timeout", "timeout" in normalized and normalized.get("timeout") is not None)
    setattr(step, "_has_cflags", "cflags" in normalized and normalized.get("cflags") is not None)
    setattr(
        step,
        "_has_compile_args",
        "compile_args" in normalized and normalized.get("compile_args") is not None,
    )
    setattr(
        step,
        "_has_exclude_nodes",
        "exclude_nodes" in normalized and normalized.get("exclude_nodes") is not None,
    )
    setattr(
        step,
        "_has_arts_config",
        "arts_config" in normalized and normalized.get("arts_config") is not None,
    )
    setattr(
        step,
        "_has_profile",
        "profile" in normalized and normalized.get("profile") is not None,
    )
    setattr(
        step,
        "_has_benchmarks",
        "benchmarks" in normalized and normalized.get("benchmarks") is not None,
    )
    setattr(
        step,
        "_has_launcher",
        "launcher" in normalized and normalized.get("launcher") is not None,
    )
    return step


def _rebuild_arts(
    console: Console,
    debug: int = 0,
    profile: Path = PROFILES_DIR / "profile-none.cfg",
) -> None:
    """Rebuild ARTS runtime/compiler with requested instrumentation profile."""
    if not profile.exists():
        print_error(f"Profile not found: {profile}")
        raise typer.Exit(1)
    if debug < 0 or debug > 3:
        print_error(f"Invalid ARTS debug level: {debug} (expected 0..3)")
        raise typer.Exit(1)

    cmd = [
        "carts", "build", "--arts",
        f"--profile={profile}",
        f"--debug={debug}",
    ]

    details: List[str] = []
    if debug > 0:
        details.append(f"debug={debug}")
    details.append(f"profile={profile}")

    detail_text = ", ".join(details)
    print_warning(f"Rebuilding ARTS ({detail_text})")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else "(no stderr)"
        print_error(f"ARTS rebuild failed:\n{stderr}")
        raise typer.Exit(1)
    print_success("ARTS rebuild complete")

def _load_experiment(
    experiment: str,
    configs_dir: Path,
) -> List[ExperimentStep]:
    """Load experiment step definitions from a JSON config."""
    exp_path: Optional[Path] = None
    exp_arg = Path(experiment)
    if exp_arg.exists():
        exp_path = exp_arg.resolve()
    else:
        candidates = [
            (configs_dir / "experiments" / f"{experiment}.json").resolve(),
            (configs_dir / "experiments" / experiment).resolve(),
        ]
        exp_path = next((p for p in candidates if p.exists()), None)

    if exp_path is None:
        raise ValueError(f"Experiment config not found: {experiment}")

    with open(exp_path, "r") as f:
        payload = json.load(f)

    experiment_name: Optional[str] = None
    experiment_description: Optional[str] = None

    if isinstance(payload, dict):
        if "setup_commands" in payload:
            raise ValueError(
                "Experiment field `setup_commands` is no longer supported."
            )
        experiment_name = (
            str(payload.get("name")).strip()
            if payload.get("name") is not None and str(payload.get("name")).strip()
            else exp_path.stem
        )
        experiment_description = (
            str(payload.get("description")).strip()
            if payload.get("description") is not None
            and str(payload.get("description")).strip()
            else None
        )
        known_top_keys = {"name", "description", "steps"} | KNOWN_STEP_KEYS
        unknown_top = sorted(set(payload.keys()) - known_top_keys)
        if unknown_top:
            console.print(
                f"[{Colors.WARNING}]Warning:[/{Colors.WARNING}] Unknown experiment keys: {', '.join(unknown_top)}"
            )
        raw_steps = payload.get("steps")
        defaults = {
            k: v
            for k, v in payload.items()
            if k in KNOWN_STEP_KEYS and k not in {"name", "description"}
        }
    elif isinstance(payload, list):
        raw_steps = payload
        defaults = {}
        experiment_name = exp_path.stem
    else:
        raise ValueError(f"Invalid experiment format in {exp_path}")

    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"Experiment '{experiment}' has no steps")

    steps: List[ExperimentStep] = []
    for idx, item in enumerate(raw_steps, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Step {idx} in {exp_path} must be an object")
        merged = {**defaults, **item}
        step = _make_experiment_step(
            merged,
            default_name=f"step_{idx}",
            base_dir=exp_path.parent,
        )
        setattr(step, "_experiment_name", experiment_name)
        setattr(step, "_experiment_description", experiment_description)
        setattr(step, "_experiment_path", str(exp_path))
        steps.append(step)
    return steps


def _serialize_experiment_steps(
    steps: Optional[List[ExperimentStep]],
) -> Optional[List[Dict[str, Any]]]:
    if not steps:
        return None

    serialized: List[Dict[str, Any]] = []
    for step in steps:
        serialized.append(
            {
                "name": step.name,
                "description": step.description,
                "benchmarks": list(step.benchmarks) if step.benchmarks else None,
                "size": step.size,
                "threads": step.threads,
                "nodes": step.nodes,
                "runs": step.runs,
                "compile_args": step.compile_args,
                "debug": step.debug,
                "perf": step.perf,
                "perf_interval": step.perf_interval if step.perf else None,
                "profile": step.profile,
            }
        )
    return serialized


def _experiment_context_from_steps(
    steps: Optional[List[ExperimentStep]],
) -> Tuple[Optional[str], Optional[str], Optional[List[Dict[str, Any]]]]:
    if not steps:
        return None, None, None

    return (
        getattr(steps[0], "_experiment_name", None),
        getattr(steps[0], "_experiment_description", None),
        _serialize_experiment_steps(steps),
    )


def _parse_inline_steps(step_args: Optional[List[str]]) -> List[ExperimentStep]:
    """Parse repeatable --step options into ExperimentStep objects."""
    if not step_args:
        return []

    steps: List[ExperimentStep] = []
    for idx, raw_arg in enumerate(step_args, start=1):
        raw = raw_arg.strip()
        if not raw:
            continue

        if raw.startswith("{"):
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(f"--step #{idx} must be a JSON object")
            steps.append(_make_experiment_step(payload, default_name=f"step_{idx}"))
            continue

        payload: Dict[str, Any] = {}
        if ":" in raw:
            name_part, fields_part = raw.split(":", 1)
            name_value = name_part.strip()
            if name_value:
                payload["name"] = name_value
            fields = [t.strip() for t in fields_part.split(",") if t.strip()]
        else:
            fields = [t.strip() for t in raw.split(",") if t.strip()]

        for token in fields:
            if "=" in token:
                key, value = token.split("=", 1)
                payload[key.strip().replace("-", "_")] = value.strip()
            else:
                key = token.strip().replace("-", "_")
                if key == "perf":
                    payload["perf"] = True
                elif "name" not in payload:
                    payload["name"] = token
                else:
                    raise ValueError(
                        f"Invalid --step token '{token}' in '{raw_arg}' "
                        "(expected key=value or bare 'perf')"
                    )

        steps.append(_make_experiment_step(payload, default_name=f"step_{idx}"))

    if not steps:
        raise ValueError("No valid --step values provided")
    return steps


def _parse_nodes_spec(spec: str) -> List[int]:
    """Parse node spec with support for both thread-style and SLURM-style ranges."""
    try:
        return parse_threads(spec)
    except ValueError:
        return parse_node_spec(spec)


def _format_sweep_display(values: List[int]) -> str:
    """Format a small numeric sweep for concise CLI display."""
    if len(values) == 1:
        return str(values[0])
    if len(values) <= 5:
        return ", ".join(str(value) for value in values)
    return f"{values[0]}-{values[-1]} ({len(values)} values)"

_STEP_RESOLVER = StepResolver(
    configs_dir=CONFIGS_DIR,
    profiles_dir=PROFILES_DIR,
    parse_threads=parse_threads,
    parse_nodes_spec=_parse_nodes_spec,
    parse_inline_steps=_parse_inline_steps,
    load_experiment=_load_experiment,
)


def _run_step(
    runner: BenchmarkRunner,
    bench_list: List[str],
    size: str,
    timeout: int,
    arts_config: Optional[Path],
    threads_list: Optional[List[int]],
    node_counts: Optional[List[int]],
    launcher: Optional[str],
    omp_threads: Optional[int],
    weak_scaling: bool,
    base_size: Optional[int],
    runs: int,
    compile_args: Optional[str],
    perf: bool,
    perf_interval: float,
    run_timestamp: str,
    cflags: Optional[str],
    quiet: bool,
    variant: Optional[str] = None,
) -> List[BenchmarkResult]:
    """Execute one resolved step using existing run dispatch rules."""
    has_thread_sweep = bool(threads_list and len(threads_list) > 1)
    has_node_sweep = bool(node_counts and len(node_counts) > 1)
    has_sweep = has_thread_sweep or has_node_sweep

    threads_override: Optional[int] = None
    nodes_override: Optional[int] = None
    if not has_thread_sweep and threads_list and len(threads_list) == 1:
        threads_override = int(threads_list[0])
    if not has_node_sweep and node_counts and len(node_counts) == 1:
        nodes_override = int(node_counts[0])

    if len(bench_list) == 1 and has_sweep:
        effective_threads = threads_list if threads_list else [None]
        effective_nodes = node_counts if node_counts else [None]
        return runner.run_with_thread_sweep(
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
            compile_args=compile_args,
            perf_enabled=perf,
            perf_interval=perf_interval,
            variant=variant,
        )

    if len(bench_list) > 1 and has_sweep:
        if weak_scaling:
            raise ValueError(
                "`--weak-scaling` is only supported for single-benchmark sweeps."
            )

        effective_threads = threads_list if threads_list else [None]
        effective_nodes = node_counts if node_counts else [None]
        total_configs = len(effective_threads) * len(effective_nodes)
        results: List[BenchmarkResult] = []
        config_idx = 0

        for nodes_value in effective_nodes:
            for threads_value in effective_threads:
                config_idx += 1
                if not quiet:
                    threads_label = threads_value if threads_value is not None else "cfg"
                    nodes_label = nodes_value if nodes_value is not None else "cfg"
                    console.print(
                        f"[{Colors.HIGHLIGHT}]Sweep config {config_idx}/{total_configs}:[/{Colors.HIGHLIGHT}] "
                        f"threads={threads_label}, nodes={nodes_label}"
                    )

                config_results = runner.run_all(
                    bench_list,
                    size=size,
                    timeout=timeout,
                    arts_config=arts_config,
                    threads_override=threads_value,
                    nodes_override=nodes_value,
                    launcher_override=launcher,
                    omp_threads_override=omp_threads,
                    compile_args=compile_args,
                    perf_enabled=perf,
                    perf_interval=perf_interval,
                    runs=runs,
                    run_timestamp=run_timestamp,
                    cflags=cflags or "",
                    variant=variant,
                )
                results.extend(config_results)
        return results

    return runner.run_all(
        bench_list,
        size=size,
        timeout=timeout,
        arts_config=arts_config,
        threads_override=threads_override,
        nodes_override=nodes_override,
        launcher_override=launcher,
        omp_threads_override=omp_threads,
        compile_args=compile_args,
        perf_enabled=perf,
        perf_interval=perf_interval,
        runs=runs,
        run_timestamp=run_timestamp,
        cflags=cflags or "",
        variant=variant,
    )


def _run_step_slurm(
    bench_list: List[str],
    size: str,
    node_counts: List[int],
    runs: int,
    partition: Optional[str],
    timeout: int,
    time_limit: Optional[str],
    arts_config: Optional[Path],
    threads_list: Optional[List[int]],
    results_dir: Path,
    verbose: bool,
    cflags: Optional[str],
    compile_args: Optional[str],
    exclude_nodes: Optional[str],
    perf: bool,
    perf_interval: float,
    artifact_manager: Optional[ArtifactManager] = None,
    step_name: Optional[str] = None,
    max_jobs: int = 0,
    report_steps: Optional[List[ExperimentStep]] = None,
) -> None:
    """Execute one resolved step through SLURM batch mode."""
    if not node_counts:
        raise ValueError("SLURM mode requires at least one node count")

    nodes_arg = ",".join(str(n) for n in node_counts)
    thread_values = threads_list if threads_list else [None]

    for slurm_threads in thread_values:
        _execute_slurm_batch(
            benchmarks=bench_list,
            nodes=nodes_arg,
            size=size,
            timeout=timeout,
            runs=runs,
            partition=partition,
            time_limit=time_limit,
            account=None,
            arts_config=arts_config,
            threads=int(slurm_threads) if slurm_threads is not None else None,
            output_dir=results_dir,
            suite=None,
            dry_run=False,
            no_build=False,
            verbose=verbose,
            cflags=cflags,
            compile_args=compile_args,
            gdb=False,
            profile=None,
            perf=perf,
            perf_interval=perf_interval,
            exclude_nodes=exclude_nodes,
            exclude=None,
            max_jobs=max_jobs,
            artifact_manager=artifact_manager,
            step_name=step_name,
            report_steps=report_steps,
        )


def _rebuild_arts_for_step(step_config: ResolvedStepConfig) -> None:
    """Rebuild ARTS when a resolved step requests instrumentation changes."""
    runtime_missing = not arts_runtime_is_installed()
    if not step_config.should_rebuild_arts and not runtime_missing:
        return
    if runtime_missing:
        print_warning("ARTS runtime is not installed; forcing rebuild before benchmark step")
    _rebuild_arts(
        console,
        debug=step_config.debug,
        profile=step_config.profile_path,
    )


def _run_local_resolved_step(
    *,
    step_config: ResolvedStepConfig,
    request: LocalStepExecutionRequest,
) -> List[BenchmarkResult]:
    """Adapter from step orchestration to the existing local step runner."""
    return _run_step(
        runner=request.runner,
        bench_list=step_config.bench_list,
        size=step_config.size,
        timeout=step_config.timeout,
        arts_config=step_config.arts_config,
        threads_list=step_config.threads_list,
        node_counts=step_config.node_counts,
        launcher=step_config.launcher,
        omp_threads=request.omp_threads,
        weak_scaling=request.weak_scaling,
        base_size=request.base_size,
        runs=step_config.runs,
        compile_args=step_config.compile_args,
        perf=step_config.perf,
        perf_interval=step_config.perf_interval,
        run_timestamp=request.run_timestamp,
        cflags=step_config.cflags,
        quiet=request.quiet,
        variant=request.variant,
    )


def _run_slurm_resolved_step(
    *,
    step_config: ResolvedStepConfig,
    request: SlurmStepExecutionRequest,
    report_steps: Optional[List[ExperimentStep]],
) -> None:
    """Adapter from step orchestration to the existing SLURM step runner."""
    _run_step_slurm(
        bench_list=step_config.bench_list,
        size=step_config.size,
        node_counts=step_config.node_counts or [],
        runs=step_config.runs,
        partition=request.partition,
        timeout=step_config.timeout,
        time_limit=request.time_limit,
        arts_config=step_config.arts_config,
        threads_list=step_config.threads_list,
        results_dir=request.results_dir,
        verbose=request.verbose,
        cflags=step_config.cflags,
        compile_args=step_config.compile_args,
        exclude_nodes=step_config.exclude_nodes,
        perf=step_config.perf,
        perf_interval=step_config.perf_interval,
        artifact_manager=request.artifact_manager,
        step_name=step_config.name,
        max_jobs=request.max_jobs,
        report_steps=report_steps,
    )


_STEP_EXECUTOR = StepExecutionOrchestrator(
    resolver=_STEP_RESOLVER,
    rebuild_step=_rebuild_arts_for_step,
    run_local_step=_run_local_resolved_step,
    run_slurm_step=_run_slurm_resolved_step,
    print_step=print_step,
)


def _is_option_from_cli(
    ctx: typer.Context,
    parameter_name: str,
    long_opt: str,
    short_opt: Optional[str] = None,
) -> bool:
    """Return whether an option was explicitly provided on the command line."""
    get_source = getattr(ctx, "get_parameter_source", None)
    if callable(get_source):
        try:
            source = get_source(parameter_name)
            if source is not None and str(source).lower().endswith("commandline"):
                return True
        except Exception:
            pass

    for token in sys.argv[1:]:
        if token == long_opt or token.startswith(f"{long_opt}="):
            return True
        if short_opt:
            if token == short_opt:
                return True
            if token.startswith(short_opt) and len(token) > len(short_opt) and not token.startswith("--"):
                return True
    return False


@app.command()
def run(
    ctx: typer.Context,
    benchmarks: Optional[List[str]] = typer.Argument(
        None, help="Specific benchmarks to run"),
    size: str = typer.Option(DEFAULT_SIZE, "--size",
                             "-s", help=SIZE_HELP),
    timeout: int = typer.Option(
        DEFAULT_TIMEOUT, "--timeout", "-t", help="Execution timeout in seconds"),
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
    compile_args: Optional[str] = typer.Option(
        None, "--compile-args", help="Extra carts compile args (e.g., '--partition-fallback=fine')"),
    debug_level: int = typer.Option(
        0, "--debug", "-d", help="Debug level: 0=off, 1=commands, 2=verbose console output"),
    profile: Optional[Path] = typer.Option(
        None, "--profile",
        help="Custom counter profile file. Triggers ARTS rebuild with this configuration."),
    runs: int = typer.Option(
        1, "--runs", "-r", help="Number of times to run each benchmark for statistical significance"),
    perf: bool = typer.Option(
        False, "--perf", help="Enable perf stat profiling for cache metrics"),
    perf_interval: float = typer.Option(
        0.1, "--perf-interval", help="Perf stat sampling interval in seconds"),
    results_dir: Optional[Path] = typer.Option(
        None, "--results-dir", help="Base directory for experiment output (default: carts-benchmarks/results/)"),
    experiment: Optional[str] = typer.Option(
        None, "--experiment", "-x",
        help="Experiment definition name/path from configs/experiments/*.json"),
    step: Optional[List[str]] = typer.Option(
        None, "--step",
        help="Inline step: 'name:key=value,...' (repeatable). Example: production:runs=5,perf"),
    slurm: bool = typer.Option(
        False, "--slurm", help="Submit as SLURM batch jobs instead of local execution"),
    partition: Optional[str] = typer.Option(
        None, "--partition", "-p", help="SLURM partition (only with --slurm)"),
    time_limit: Optional[str] = typer.Option(
        None, "--time-limit",
        help="SLURM wall time per job (defaults to benchmark timeout + 30s; only with --slurm)"),
    exclude_nodes: Optional[str] = typer.Option(
        None, "--exclude-nodes", "-X",
        help="SLURM nodes to exclude (comma-separated, e.g. j006,j007)"),
    openmp: bool = typer.Option(
        False, "--openmp", help="Run OpenMP only (skip ARTS build and run)"),
    arts: bool = typer.Option(
        False, "--arts", help="Run ARTS only (skip OpenMP build and run)"),
    exclude: Optional[List[str]] = typer.Option(
        None, "--exclude", "-e",
        help="Benchmarks to exclude (substring match, repeatable)"),
    max_jobs: int = typer.Option(
        0, "--max-jobs", "-J",
        help="Max concurrent SLURM jobs (PENDING+RUNNING). "
             "0 = unlimited (submit all at once). "
             "When set, new jobs are submitted as earlier ones finish. "
             "Only with --slurm"),
):
    """Run benchmarks with verification and timing."""
    try:
        size = parse_size(size, "--size")
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(2)

    if openmp and arts:
        print_error("Cannot use --openmp and --arts together.")
        raise typer.Exit(2)
    variant: Optional[str] = VARIANT_OPENMP if openmp else (VARIANT_ARTS if arts else None)

    clean = not no_clean
    size_from_cli = _is_option_from_cli(ctx, "size", "--size", "-s")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve results_dir — default to carts-benchmarks/results/
    if results_dir is None:
        results_dir = BENCHMARKS_DIR / "results"
    results_dir = Path(results_dir).resolve()

    # Create ArtifactManager — every run gets a self-contained timestamped directory
    am = ArtifactManager(results_dir, run_timestamp)

    # Create runner with artifact manager
    runner = BenchmarkRunner(
        console, verbose=verbose, quiet=quiet, trace=trace, clean=clean,
        debug=debug_level, artifact_manager=am,
    )

    # Parse thread/node specification (base CLI config)
    try:
        base_threads_list = parse_threads(threads) if threads else None
        base_node_counts = _parse_nodes_spec(nodes) if nodes else None
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    # Discover or use provided benchmarks
    requested_benchmarks, dropped_blank = normalize_requested_benchmarks(
        benchmarks)
    if dropped_blank and not quiet:
        print_warning(
            "Ignoring blank benchmark arguments. "
            "If you copied a multiline command, ensure each '\\' is immediately before a newline.")

    if requested_benchmarks:
        invalid = find_invalid_benchmarks(runner, requested_benchmarks)
        if invalid:
            invalid_list = ", ".join(invalid)
            print_error(f"Unknown benchmark(s): {invalid_list}")
            console.print(f"[{Colors.DEBUG}]Use `carts benchmarks list` to see valid names.[/{Colors.DEBUG}]")
            raise typer.Exit(2)
        bench_list = requested_benchmarks
    else:
        bench_list = runner.discover_benchmarks(suite)

    # Apply --exclude filter
    if exclude:
        before = len(bench_list)
        bench_list = [b for b in bench_list
                      if not any(ex in b for ex in exclude)]
        excluded = before - len(bench_list)
        if excluded:
            console.print(f"[{Colors.DEBUG}]  ({excluded} benchmarks excluded via --exclude)[/{Colors.DEBUG}]")

    if not bench_list:
        print_warning("No benchmarks found.")
        raise typer.Exit(1)

    # Resolve experiment steps
    if experiment and step:
        print_error("Use either --experiment or --step, not both.")
        raise typer.Exit(2)

    try:
        steps, explicit_step_mode = _STEP_RESOLVER.load_steps(
            experiment=experiment,
            step_args=step,
            size=size,
            timeout=timeout,
            runs=runs,
            perf=perf,
            perf_interval=perf_interval,
            threads=threads,
            nodes=nodes,
            cflags=cflags,
            compile_args=compile_args,
            exclude_nodes=exclude_nodes,
            arts_config=arts_config,
            profile=profile,
            launcher=launcher,
        )
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    _STEP_RESOLVER.apply_cli_profile_override(
        steps,
        explicit_step_mode=explicit_step_mode,
        profile=profile,
        quiet=quiet,
        print_warning=print_warning,
    )
    try:
        _STEP_RESOLVER.validate_step_paths(steps)
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)

    try:
        _STEP_RESOLVER.validate_step_name_collisions(steps)
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(2)

    effective_size_label = _STEP_RESOLVER.resolve_effective_size_label(
        steps,
        size,
        size_from_cli,
    )

    # Reject simultaneous thread + node sweeps early (per-step checks also exist).
    has_thread_sweep = bool(base_threads_list and len(base_threads_list) > 1)
    has_node_sweep = bool(base_node_counts and len(base_node_counts) > 1)
    if has_thread_sweep and has_node_sweep:
        print_error("Cannot sweep both threads and nodes simultaneously.")
        raise typer.Exit(2)

    step_defaults = StepCliDefaults(
        size=size,
        timeout=timeout,
        threads_spec=threads,
        nodes_spec=nodes,
        runs=runs,
        perf=perf,
        perf_interval=perf_interval,
        cflags=cflags,
        compile_args=compile_args,
        exclude_nodes=exclude_nodes,
        arts_config=arts_config,
        launcher=launcher,
        explicit_step_mode=explicit_step_mode,
        size_from_cli=size_from_cli,
    )

    if slurm:
        if weak_scaling:
            print_error("--weak-scaling is not supported with --slurm.")
            raise typer.Exit(2)

        try:
            _STEP_EXECUTOR.execute_slurm_steps(
                steps=steps,
                bench_list=bench_list,
                defaults=step_defaults,
                request=SlurmStepExecutionRequest(
                    partition=partition,
                    time_limit=time_limit,
                    results_dir=results_dir,
                    verbose=verbose,
                    quiet=quiet,
                    artifact_manager=am,
                    max_jobs=max_jobs,
                ),
            )
        except ValueError as e:
            console.print(f"\n[{Colors.ERROR}]Error:[/{Colors.ERROR}] {e}")
            raise typer.Exit(1)
        return

    if exclude_nodes:
        print_warning("Ignoring --exclude-nodes because --slurm is not enabled.")

    # Print header
    if not quiet:
        config_items = [f"size={effective_size_label}", f"timeout={timeout}s", f"clean={clean}"]
        if variant:
            config_items.append(f"variant={variant}")
        if base_threads_list:
            config_items.append(f"threads={threads}")
        if launcher is not None:
            config_items.append(f"launcher={launcher}")
        if base_node_counts is not None:
            config_items.append(f"nodes={nodes}")
        if runs > 1:
            config_items.append(f"runs={runs}")
        if cflags:
            config_items.append(f"cflags={cflags}")
        if compile_args:
            config_items.append(f"compile-args={compile_args}")
        if debug_level > 0:
            config_items.append(f"debug={debug_level}")
        if profile:
            config_items.append(f"profile={profile.name}")
        if perf:
            config_items.append("perf=on")
        if experiment:
            config_items.append(f"experiment={experiment}")
        if step:
            config_items.append(f"steps={len(step)}")
        subtitle = f"Config: {', '.join(config_items)}"
        print_header("CARTS Benchmark Runner", subtitle)

        # Show effective ARTS configuration
        if bench_list:
            if len(steps) > 1:
                console.print("ARTS Config (experiment): varies by step")
                console.print("  Path: resolved from each step definition")
            else:
                display_step = _STEP_RESOLVER.resolve_step_config(
                    steps[0],
                    1,
                    bench_list,
                    step_defaults,
                )

                if display_step.arts_config is not None:
                    effective_config = display_step.arts_config
                    config_source = f"step '{display_step.name}'"
                elif arts_config:
                    effective_config = arts_config
                    config_source = "custom"
                else:
                    effective_config = DEFAULT_ARTS_CONFIG
                    config_source = "default"

                cfg = parse_arts_cfg(effective_config)
                arts_threads: str = str(int(cfg.get(KEY_WORKER_THREADS, "1")))
                arts_nodes: str = str(int(cfg.get(KEY_NODE_COUNT, "1")))
                arts_launcher = cfg.get(KEY_LAUNCHER, "ssh")

                if display_step.threads_list:
                    arts_threads = _format_sweep_display(display_step.threads_list)
                if display_step.node_counts:
                    arts_nodes = format_node_counts_display(display_step.node_counts)
                if display_step.launcher:
                    arts_launcher = display_step.launcher

                items = [f"threads={arts_threads}", f"nodes={arts_nodes}", f"launcher={arts_launcher}"]
                console.print(f"ARTS Config ({config_source}): {', '.join(items)}")
                console.print(f"  Path: {effective_config}")

        console.print(f"Benchmarks: {len(bench_list)}\n")

    # Run benchmarks
    start_time = time.time()

    # Export metadata: differentiate fixed config vs sweep.
    fixed_threads_meta: Optional[int] = None
    fixed_nodes_meta: Optional[int] = None
    thread_sweep_meta: Optional[List[int]] = base_threads_list if has_thread_sweep else None
    node_sweep_meta: Optional[List[int]] = base_node_counts if has_node_sweep else None

    if not has_thread_sweep and base_threads_list and len(base_threads_list) == 1:
        fixed_threads_meta = int(base_threads_list[0])
    if not has_node_sweep and base_node_counts and len(base_node_counts) == 1:
        fixed_nodes_meta = int(base_node_counts[0])

    try:
        results = _STEP_EXECUTOR.execute_local_steps(
            steps=steps,
            bench_list=bench_list,
            defaults=step_defaults,
            request=LocalStepExecutionRequest(
                runner=runner,
                omp_threads=omp_threads,
                weak_scaling=weak_scaling,
                base_size=base_size,
                run_timestamp=run_timestamp,
                clean=clean,
                quiet=quiet,
                artifact_manager=am,
                variant=variant,
            ),
        )
    except ValueError as e:
        console.print(f"\n[{Colors.ERROR}]Error:[/{Colors.ERROR}] {e}")
        raise typer.Exit(1)

    outlier_counts = annotate_startup_outliers(results, write_artifacts=True)
    total_duration = time.time() - start_time

    # Display results
    if not quiet:
        # Table was already shown via Live display, just show the summary panel
        console.print()
        console.print(create_summary_panel(results, total_duration))

    # Write results.json into the experiment directory
    experiment_name, experiment_description, experiment_steps = _experiment_context_from_steps(steps)
    export_json(
        results,
        am.results_json_path,
        effective_size_label,
        total_duration,
        thread_sweep_meta,
        node_sweep_meta,
        cflags,
        launcher,
        weak_scaling,
        base_size,
        runs,
        ".",  # artifacts are in same directory
        fixed_threads=fixed_threads_meta,
        fixed_nodes=fixed_nodes_meta,
        omp_threads_override=omp_threads,
        arts_config_override=str(arts_config) if arts_config else None,
        experiment_name=experiment_name,
        experiment_description=experiment_description,
        experiment_steps=experiment_steps,
        startup_outlier_counts=outlier_counts,
    )

    report_path: Optional[Path] = None
    try:
        report_steps = steps if (explicit_step_mode or len(steps) > 1) else None
        report_path = generate_report(
            results,
            am.experiment_dir,
            quiet=quiet,
            steps=report_steps,
        )
    except Exception as e:
        if not quiet:
            print_warning(f"Failed to generate report.xlsx: {e}")

    if not quiet and report_path:
        print_info(f"Report: {report_path.name}")
    elif not quiet:
        print_warning("Report not generated (openpyxl may be unavailable in this environment).")

    # Write manifest.json
    command_str = "carts benchmarks " + " ".join(sys.argv[1:])
    am.write_manifest(results, command_str, total_duration)

    # Show single results path
    if not quiet:
        console.print()
        print_success(f"Results: {am.experiment_dir}")

    # Exit with error if any failures
    failed = sum(1 for r in results if benchmark_result_failed(r))
    if failed > 0:
        raise typer.Exit(1)


def _resolve_results_json_path(path: Path) -> Path:
    """Resolve a results.json path from either a file or results directory."""
    candidate = path.resolve()
    if candidate.is_file():
        return candidate

    if candidate.is_dir():
        direct = candidate / RESULTS_FILENAME
        if direct.is_file():
            return direct
        nested = sorted(
            candidate.glob("*/results.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if nested:
            return nested[0]

    raise ValueError(f"Could not resolve results.json from: {path}")


def _coerce(value: Any, type_fn: type = float, default: Any = None) -> Any:
    """Coerce *value* to *type_fn* (e.g. ``float`` or ``int``).

    Returns *default* when *value* is ``None`` or cannot be converted.
    """
    if value is None:
        return default
    try:
        return type_fn(value)
    except (TypeError, ValueError):
        return default


def _result_status_is_pass(result: Dict[str, Any], key: str) -> bool:
    status = str(result.get(key, {}).get("status", "")).strip().lower()
    return status == Status.PASS.value


def _extract_e2e_sec(result: Dict[str, Any], variant: str) -> Optional[float]:
    timing = result.get("timing", {})
    timing_value = _coerce(timing.get(f"{variant}_e2e_sec"))
    if timing_value is not None:
        return timing_value

    run_data = result.get(f"run_{variant}", {})
    e2e_timings = run_data.get("e2e_timings")
    if isinstance(e2e_timings, dict):
        values = [_coerce(v) for v in e2e_timings.values()]
        numeric_values = [v for v in values if v is not None]
        if numeric_values:
            return float(sum(numeric_values))
    return None


def _extract_startup_outlier(result: Dict[str, Any], variant: str) -> bool:
    run_data = result.get(f"run_{variant}", {})
    detail = run_data.get("startup_outlier")
    return bool(isinstance(detail, dict) and detail.get(KEY_IS_OUTLIER))


def _summarize_result_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "speedup_median_filtered": None,
            "arts_e2e_median_filtered": None,
            "omp_e2e_median_filtered": None,
            "arts_outlier_count": 0,
            "omp_outlier_count": 0,
        }

    rows_sorted = sorted(rows, key=lambda r: _coerce(r.get("run_number"), int) or 0)

    core = _compute_robust_summary(
        items=rows_sorted,
        get_arts_e2e=lambda row: _extract_e2e_sec(row, VARIANT_ARTS),
        get_omp_e2e=lambda row: _extract_e2e_sec(row, VARIANT_OMP),
        get_speedup=lambda row: _coerce(row.get(KEY_TIMING, {}).get(KEY_SPEEDUP)),
        get_is_outlier=lambda idx, row: (
            _extract_startup_outlier(row, VARIANT_ARTS),
            _extract_startup_outlier(row, VARIANT_OMP),
        ),
    )

    return {
        "speedup_median_filtered": core["speedup_value"],
        "arts_e2e_median_filtered": core["arts_value"],
        "omp_e2e_median_filtered": core["omp_value"],
        "arts_outlier_count": len(core["arts_outlier_indices"]),
        "omp_outlier_count": len(core["omp_outlier_indices"]),
    }


def _evaluate_perf_gate_entry(
    all_rows: List[Dict[str, Any]],
    entry: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    name = entry.get("name")
    if not name:
        raise ValueError("Each perf-gate benchmark entry must include 'name'")

    size = entry.get("size", defaults.get("size"))
    threads = _coerce(entry.get("threads", defaults.get("threads")), int)
    nodes = _coerce(entry.get("nodes", defaults.get("nodes")), int)
    run_phase = str(entry.get("run_phase", defaults.get("run_phase", "")) or "")

    required = bool(entry.get("required", defaults.get("required", True)))
    require_success = bool(
        entry.get("require_success", defaults.get("require_success", True))
    )
    require_correct = bool(
        entry.get("require_correct", defaults.get("require_correct", True))
    )

    max_startup_outliers: Dict[str, Any] = {}
    defaults_outliers = defaults.get("max_startup_outliers")
    entry_outliers = entry.get("max_startup_outliers")
    if isinstance(defaults_outliers, dict):
        max_startup_outliers.update(defaults_outliers)
    if isinstance(entry_outliers, dict):
        max_startup_outliers.update(entry_outliers)

    min_speedup = _coerce(entry.get("min_speedup"))
    baseline_speedup = _coerce(entry.get("baseline_speedup"))
    tolerance_pct = _coerce(entry.get("tolerance_pct"))
    if min_speedup is None and baseline_speedup is not None:
        tolerance = tolerance_pct if tolerance_pct is not None else 0.10
        min_speedup = baseline_speedup * (1.0 - tolerance)

    max_arts_e2e_sec = _coerce(entry.get("max_arts_e2e_sec"))

    rows = []
    for row in all_rows:
        if row.get("name") != name:
            continue
        if size and row.get("size") != size:
            continue
        cfg = row.get("config", {})
        if threads is not None and _coerce(cfg.get("arts_threads"), int) != threads:
            continue
        if nodes is not None and _coerce(cfg.get("arts_nodes"), int) != nodes:
            continue
        if (row.get("run_phase") or "") != run_phase:
            continue
        rows.append(row)

    summary = _summarize_result_rows(rows)
    reasons: List[str] = []

    if not rows:
        reasons.append("no matching results")
    else:
        all_success = all(
            _result_status_is_pass(row, "build_arts")
            and _result_status_is_pass(row, "build_omp")
            and _result_status_is_pass(row, "run_arts")
            and _result_status_is_pass(row, "run_omp")
            for row in rows
        )
        all_correct = all(
            bool(row.get("verification", {}).get("correct")) for row in rows
        )
        speedup = summary["speedup_median_filtered"]
        arts_e2e = summary["arts_e2e_median_filtered"]

        if require_success and not all_success:
            reasons.append("build/run failure")
        if require_correct and not all_correct:
            reasons.append("correctness mismatch")
        if min_speedup is not None and (speedup is None or speedup < min_speedup):
            reasons.append(f"speedup {speedup if speedup is not None else 'n/a'} < {min_speedup:.3f}")
        if max_arts_e2e_sec is not None and (arts_e2e is None or arts_e2e > max_arts_e2e_sec):
            reasons.append(
                f"ARTS e2e {arts_e2e if arts_e2e is not None else 'n/a'} > {max_arts_e2e_sec:.3f}s"
            )

        max_arts_outliers = _coerce(max_startup_outliers.get("arts"), int)
        max_omp_outliers = _coerce(max_startup_outliers.get("omp"), int)
        if (
            max_arts_outliers is not None
            and summary["arts_outlier_count"] > max_arts_outliers
        ):
            reasons.append(
                f"ARTS startup outliers {summary['arts_outlier_count']} > {max_arts_outliers}"
            )
        if (
            max_omp_outliers is not None
            and summary["omp_outlier_count"] > max_omp_outliers
        ):
            reasons.append(
                f"OpenMP startup outliers {summary['omp_outlier_count']} > {max_omp_outliers}"
            )

    return {
        "id": f"{name}|{size}|{threads}|{nodes}|{run_phase}",
        "name": name,
        "size": size,
        "threads": threads,
        "nodes": nodes,
        "run_phase": run_phase,
        "required": required,
        "min_speedup": min_speedup,
        "summary": summary,
        "pass": len(reasons) == 0,
        "reasons": reasons,
    }


def _perf_gate_entry_id(entry: Dict[str, Any], defaults: Dict[str, Any]) -> str:
    """Build a stable identifier for a perf-gate policy entry."""
    name = entry.get("name", "")
    size = entry.get("size", defaults.get("size"))
    threads = _coerce(entry.get("threads", defaults.get("threads")), int)
    nodes = _coerce(entry.get("nodes", defaults.get("nodes")), int)
    run_phase = str(entry.get("run_phase", defaults.get("run_phase", "")) or "")
    return f"{name}|{size}|{threads}|{nodes}|{run_phase}"


@app.command(name="perf-gate")
def perf_gate(
    results: Path = typer.Argument(
        ...,
        help="Path to a results.json file or an experiment directory.",
    ),
    policy: Path = typer.Option(
        DEFAULT_PERF_GATE_POLICY,
        "--policy",
        help="Perf-gate policy JSON.",
    ),
    strict_advisory: bool = typer.Option(
        False,
        "--strict-advisory",
        help="Treat advisory benchmark failures as gate failures.",
    ),
) -> None:
    """Evaluate benchmark results against a threshold policy.

    Flake-policy fields (max_attempts, min_passes_required) are accepted
    in the policy JSON for documentation purposes but retry logic should
    be handled by CI.  This command evaluates a single results file and
    reports pass/fail.
    """
    try:
        policy_path = policy.resolve()
        with open(policy_path, "r") as handle:
            policy_doc = json.load(handle)
    except FileNotFoundError:
        print_error(f"Perf-gate policy not found: {policy}")
        raise typer.Exit(2)
    except json.JSONDecodeError as exc:
        print_error(f"Invalid JSON in perf-gate policy: {policy} ({exc})")
        raise typer.Exit(2)

    entries = policy_doc.get("benchmarks")
    if not isinstance(entries, list) or not entries:
        print_error("Perf-gate policy must define a non-empty 'benchmarks' list")
        raise typer.Exit(2)

    defaults = policy_doc.get("defaults", {})

    # Note: flake_policy.max_attempts / min_passes_required are kept in the
    # policy schema for documentation but retries should be driven by CI.
    flake_policy = policy_doc.get("flake_policy", {})
    if _coerce(flake_policy.get("max_attempts"), int, 1) > 1:
        print_warning(
            "flake_policy.max_attempts > 1 is ignored; CI should handle retries."
        )

    try:
        resolved_results = _resolve_results_json_path(results)
        with open(resolved_results, "r") as handle:
            doc = json.load(handle)
    except FileNotFoundError:
        print_error(f"Results file not found: {results}")
        raise typer.Exit(2)
    except json.JSONDecodeError as exc:
        print_error(f"Invalid JSON in results file: {results} ({exc})")
        raise typer.Exit(2)
    except ValueError as exc:
        print_error(str(exc))
        raise typer.Exit(2)

    rows = doc.get("results")
    if not isinstance(rows, list):
        print_error(f"Invalid results payload (missing list 'results'): {resolved_results}")
        raise typer.Exit(2)

    evaluations = [
        _evaluate_perf_gate_entry(rows, entry, defaults) for entry in entries
    ]
    eval_by_id = {item["id"]: item for item in evaluations}

    print_header(
        "Benchmark Perf Gate",
        f"Policy: {policy_path}\nResults: {resolved_results}",
    )

    table = Table(box=box.ROUNDED, show_header=True, header_style=Colors.HIGHLIGHT)
    table.add_column("Benchmark", style=Colors.INFO)
    table.add_column("Req", justify="center")
    table.add_column("Speedup", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Outliers", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Notes")

    required_failures = 0
    advisory_failures = 0

    for entry in entries:
        eval_id = _perf_gate_entry_id(entry, defaults)
        result = eval_by_id[eval_id]

        speedup = result["summary"]["speedup_median_filtered"]
        min_speedup = result["min_speedup"]
        arts_outliers = result["summary"]["arts_outlier_count"]
        omp_outliers = result["summary"]["omp_outlier_count"]

        if result["pass"]:
            status_text = f"[{Colors.SUCCESS}]{Symbols.PASS} PASS[/{Colors.SUCCESS}]"
            notes = ""
        else:
            if result["required"]:
                required_failures += 1
                status_text = f"[{Colors.ERROR}]{Symbols.FAIL} FAIL[/{Colors.ERROR}]"
            else:
                advisory_failures += 1
                status_text = f"[{Colors.WARNING}]{Symbols.WARNING} advisory[/{Colors.WARNING}]"
            notes = "; ".join(result["reasons"])

        table.add_row(
            result["name"],
            "yes" if result["required"] else "no",
            f"{speedup:.3f}x" if speedup is not None else "-",
            f"{min_speedup:.3f}x" if min_speedup is not None else "-",
            f"A:{arts_outliers} O:{omp_outliers}",
            status_text,
            notes,
        )

    console.print(table)

    if required_failures > 0:
        print_error(f"Perf gate failed: {required_failures} required benchmark target(s) failed.")
        raise typer.Exit(1)
    if strict_advisory and advisory_failures > 0:
        print_error(
            f"Perf gate failed in strict advisory mode: {advisory_failures} advisory target(s) failed."
        )
        raise typer.Exit(1)

    if advisory_failures > 0:
        print_warning(f"Perf gate passed with {advisory_failures} advisory target failure(s).")
    print_success("Perf gate passed.")


@app.command()
def build(
    benchmarks: Optional[List[str]] = typer.Argument(
        None, help="Specific benchmarks to build"),
    size: str = typer.Option(DEFAULT_SIZE, "--size",
                             "-s", help=SIZE_HELP),
    openmp: bool = typer.Option(
        False, "--openmp", help="Build OpenMP version only"),
    arts: bool = typer.Option(False, "--arts", help="Build ARTS version only"),
    suite: Optional[str] = typer.Option(
        None, "--suite", help="Filter by suite"),
    arts_config: Optional[Path] = typer.Option(
        None, "--arts-config", help="Custom arts.cfg file"),
):
    """Build benchmarks without running."""
    try:
        size = parse_size(size, "--size")
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(2)

    runner = BenchmarkRunner(console)

    # Discover or use provided benchmarks
    requested_benchmarks, dropped_blank = normalize_requested_benchmarks(
        benchmarks)
    if dropped_blank:
        print_warning(
            "Ignoring blank benchmark arguments. "
            "If you copied a multiline command, ensure each '\\' is immediately before a newline.")

    if requested_benchmarks:
        invalid = find_invalid_benchmarks(runner, requested_benchmarks)
        if invalid:
            invalid_list = ", ".join(invalid)
            print_error(f"Unknown benchmark(s): {invalid_list}")
            console.print(f"[{Colors.DEBUG}]Use `carts benchmarks list` to see valid names.[/{Colors.DEBUG}]")
            raise typer.Exit(2)
        bench_list = requested_benchmarks
    else:
        bench_list = runner.discover_benchmarks(suite)

    if not bench_list:
        print_warning("No benchmarks found.")
        raise typer.Exit(1)

    # Determine variants to build
    variants = []
    if openmp:
        variants.append(VARIANT_OPENMP)
    elif arts:
        variants.append(VARIANT_ARTS)
    else:
        variants = [VARIANT_ARTS, VARIANT_OPENMP]

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
                    task, description=f"[{Colors.INFO}]{bench}[/{Colors.INFO}] ({variant})")
                build_output_dir = (
                    runner.benchmarks_dir
                    / "build"
                    / bench.replace("/", "_")
                    / f"size_{size}"
                    / f"variant_{variant}"
                )
                result = runner.build_benchmark(
                    bench, size, variant, arts_config,
                    build_output_dir=build_output_dir,
                )
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
    else:
        requested_benchmarks, dropped_blank = normalize_requested_benchmarks(
            benchmarks)
        if dropped_blank:
            print_warning(
                "Ignoring blank benchmark arguments. "
                "If you copied a multiline command, ensure each '\\' is immediately before a newline.")

        if requested_benchmarks:
            invalid = find_invalid_benchmarks(runner, requested_benchmarks)
            if invalid:
                invalid_list = ", ".join(invalid)
                print_error(f"Unknown benchmark(s): {invalid_list}")
                console.print(
                    f"[{Colors.DEBUG}]Use `carts benchmarks list` to see valid names.[/{Colors.DEBUG}]")
                raise typer.Exit(2)
            bench_list = requested_benchmarks
        else:
            bench_list = runner.discover_benchmarks(suite)

    if not bench_list:
        print_warning("No benchmarks found.")
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


def _execute_slurm_batch(
    benchmarks: Optional[List[str]] = typer.Argument(
        None, help="Specific benchmarks to run (default: all)"),
    nodes: str = typer.Option(
        ..., "--nodes", "-n",
        help="Node counts: single (4), range (1-15), or list (1,2,4,8,16)"),
    size: str = typer.Option(
        DEFAULT_SIZE, "--size", "-s",
        help=SIZE_HELP),
    timeout: int = typer.Option(
        DEFAULT_TIMEOUT, "--timeout",
        help="Execution timeout in seconds inside each SLURM job"),
    runs: int = typer.Option(
        1, "--runs", "-r", help="Number of runs per benchmark"),
    partition: Optional[str] = typer.Option(
        None, "--partition", "-p",
        help="SLURM partition (uses cluster default if not specified)"),
    time_limit: Optional[str] = typer.Option(
        None, "--time-limit", "--time",
        help="SLURM wall time per job (defaults to benchmark timeout + 30s)"),
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
    compile_args: Optional[str] = typer.Option(
        None, "--compile-args", help="Extra carts compile args passed to `carts compile`"),
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
    max_jobs: int = typer.Option(
        0, "--max-jobs", "-J",
        help="Max concurrent SLURM jobs (PENDING+RUNNING). "
             "0 = unlimited (submit all at once). "
             "When set, new jobs are submitted as earlier ones finish"),
    artifact_manager: Optional[ArtifactManager] = None,
    step_name: Optional[str] = None,
    report_steps: Optional[List[ExperimentStep]] = None,
):
    """Submit benchmarks as SLURM batch jobs.

    This command:
    1. Builds all benchmarks locally (reusable across jobs)
    2. Generates sbatch scripts for each (benchmark x node_count x run)
    3. Submits jobs to the SLURM queue
    4. Monitors job progress until all jobs finish
    5. Collects and aggregates results

    By default every job is submitted at once and a polling monitor waits for
    completion.  Use ``--max-jobs N`` to cap how many jobs are active
    (PENDING + RUNNING) simultaneously — new jobs are submitted as earlier ones
    finish, so the SLURM queue never has more than N of your jobs at a time.
    When throttling is active, submission and monitoring happen in a single
    step (Phase 3) and the separate monitoring phase (Phase 4) is skipped.

    Example:
        carts benchmarks run --slurm --nodes=1-15 --runs 10
        carts benchmarks run --slurm --nodes=1-15 --runs 10 --max-jobs 20
        carts benchmarks run polybench/gemm --slurm --nodes=4 --time-limit 00:30:00
        carts benchmarks run --slurm --nodes=1,2,4,8,16 --partition compute

    Key Features:
    - Submit all jobs at once, or throttle with --max-jobs N
    - Exclusive node allocation for resource isolation
    - Each job has isolated counter directory (no data collision)
    - Sweep across multiple node counts with --nodes=1-15
    """
    size = parse_size(size, "--size")
    runner = BenchmarkRunner(console, verbose, False, False, False, 0)
    require_slurm_commands(dry_run)
    resolved_time_limit = resolve_slurm_time_limit(timeout, time_limit)

    # Parse node counts from --nodes parameter
    node_counts = parse_node_spec(nodes)
    if not dry_run:
        validate_requested_node_counts(node_counts, partition)

    # Determine explicit arts config override (if provided).
    explicit_arts_config = arts_config.resolve() if arts_config else None
    if explicit_arts_config is not None and not explicit_arts_config.exists():
        raise ValueError(f"arts.cfg not found: {explicit_arts_config}")

    # Get threads from explicit config or default config.
    if threads is None:
        thread_source_cfg = explicit_arts_config or DEFAULT_ARTS_CONFIG.resolve()
        threads = get_arts_cfg_int(thread_source_cfg, KEY_WORKER_THREADS) or 8

    nodes_display = format_node_counts_display(node_counts)

    config_display = (
        str(explicit_arts_config)
        if explicit_arts_config is not None
        else "benchmark-specific defaults (benchmark/suite/local.cfg)"
    )
    subtitle_parts = [
        f"Config: {config_display}",
        f"Nodes: {nodes_display}, Threads: {threads}",
        f"Runs per benchmark: {runs}, Size: {size}",
        f"Timeout: {timeout}s (wall {resolved_time_limit})",
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
    requested_benchmarks, dropped_blank = normalize_requested_benchmarks(
        benchmarks)
    if dropped_blank:
        print_warning(
            "Ignoring blank benchmark arguments. "
            "If you copied a multiline command, ensure each '\\' is immediately before a newline.")

    if requested_benchmarks:
        invalid = find_invalid_benchmarks(runner, requested_benchmarks)
        if invalid:
            invalid_list = ", ".join(invalid)
            print_error(f"Unknown benchmark(s): {invalid_list}")
            console.print(f"[{Colors.DEBUG}]Use `carts benchmarks list` to see valid names.[/{Colors.DEBUG}]")
            raise typer.Exit(2)
        bench_list = requested_benchmarks
    else:
        bench_list = runner.discover_benchmarks(suite)

    # Apply --exclude filter for benchmarks
    if exclude:
        before = len(bench_list)
        bench_list = [b for b in bench_list
                      if not any(ex in b for ex in exclude)]
        excluded_count = before - len(bench_list)
        if excluded_count:
            print_info(f"{excluded_count} benchmarks excluded via --exclude")

    if not bench_list:
        print_warning("No benchmarks to run.")
        return

    multinode_disabled = find_multinode_disabled_benchmarks(runner, bench_list)
    total_jobs = count_total_slurm_jobs(
        bench_list,
        node_counts,
        runs,
        multinode_disabled,
    )

    print_info(
        f"Found {len(bench_list)} benchmarks, {len(node_counts)} node counts, {total_jobs} total jobs"
    )
    if multinode_disabled and max(node_counts) > 1:
        print_info(f"{len(multinode_disabled)} benchmarks disabled for multi-node")
    request = SlurmBatchRequest(
        bench_list=bench_list,
        node_counts=node_counts,
        size=size,
        runs=runs,
        timeout=timeout,
        partition=partition,
        time_limit=resolved_time_limit,
        account=account,
        explicit_arts_config=explicit_arts_config,
        threads=threads,
        output_dir=output_dir,
        max_jobs=max_jobs,
        dry_run=dry_run,
        no_build=no_build,
        verbose=verbose,
        cflags=cflags,
        compile_args=compile_args,
        gdb=gdb,
        profile=profile,
        perf=perf,
        perf_interval=perf_interval,
        exclude_nodes=exclude_nodes,
        artifact_manager=artifact_manager,
        step_name=step_name,
        report_steps=report_steps,
        command_str="carts benchmarks " + " ".join(sys.argv[1:]),
    )
    deps = SlurmExecutorDependencies(
        resolve_effective_arts_config=_resolve_effective_arts_config,
        parse_time_limit_seconds=parse_slurm_time_limit_seconds,
        get_carts_dir=get_carts_dir,
        get_benchmarks_dir=lambda: BENCHMARKS_DIR,
        step_name_to_token=_STEP_RESOLVER.step_name_to_token,
    )
    SlurmBatchExecutor(runner, request, deps).execute()


if __name__ == "__main__":
    # Enable debug logging when --verbose/-v is in args
    if "--verbose" in sys.argv or "-v" in sys.argv:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    app()
