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

logger = logging.getLogger(__name__)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
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
from scripts.arts_config import parse_arts_cfg as _shared_parse_arts_cfg

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
)

# Shared constants and parsing
from benchmark_common import (
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
    INIT_TIME_PATTERN,
    parse_checksum,
    parse_kernel_timings,
    parse_e2e_timings,
    parse_init_timings,
    parse_perf_csv as _shared_parse_perf_csv,
)

# ============================================================================
# Constants (local-only — shared constants imported from benchmark_common)
# ============================================================================

# Data models (enums + dataclasses)
from benchmark_models import (
    Status, Phase,
    BuildResult, WorkerTiming, ParallelTaskTiming, PerfCacheMetrics,
    RunResult, TimingResult, VerificationResult, ReferenceChecksum, ExperimentStep,
    Artifacts, BenchmarkConfig, BenchmarkResult,
)
from benchmark_execution import (
    BenchmarkExecutionContext,
    BenchmarkProcessRequest,
    BenchmarkProcessRunner,
    BenchmarkRunFiles,
)
from benchmark_orchestration import (
    LocalStepExecutionRequest,
    ResolvedStepConfig,
    SlurmStepExecutionRequest,
    StepCliDefaults,
    StepExecutionOrchestrator,
    StepResolver,
)
from benchmark_pipeline import (
    ConfigExecutionExecutor,
    ConfigExecutionPlan,
    ExecutionHooks,
)

# Artifact management
from benchmark_artifacts import ArtifactManager
from benchmark_report import generate_report

# Reproducibility metadata
from benchmark_metadata import (
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


def get_benchmarks_dir() -> Path:
    """Get the benchmarks directory."""
    return BENCHMARKS_DIR


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
        BENCHMARKS_DIR / "arts.cfg",
        DEFAULT_ARTS_CONFIG,
    ]:
        if candidate.exists():
            return candidate.resolve()

    return DEFAULT_ARTS_CONFIG.resolve()


def _validate_thread_network_topology(
    arts_cfg_path: Optional[Path],
    threads: int,
    node_count: int,
    benchmark_name: str,
) -> None:
    """Validate that at least one worker thread remains after network threads."""
    if node_count <= 1:
        return

    outgoing = get_arts_cfg_int(arts_cfg_path, "outgoing")
    incoming = get_arts_cfg_int(arts_cfg_path, "incoming")
    sender_threads = outgoing if outgoing is not None else 1
    receiver_threads = incoming if incoming is not None else 1
    min_threads = sender_threads + receiver_threads + 1

    if threads < min_threads:
        cfg_display = str(arts_cfg_path) if arts_cfg_path else "<default>"
        raise ValueError(
            f"Invalid thread topology for '{benchmark_name}' (nodes={node_count}): "
            f"threads={threads}, outgoing={sender_threads}, incoming={receiver_threads}. "
            f"Need threads >= {min_threads} so at least one worker thread remains "
            f"(threads > outgoing + incoming). "
            f"Adjust step threads or update outgoing/incoming in {cfg_display}."
        )


def parse_arts_cfg(path: Optional[Path]) -> Dict[str, str]:
    """Parse an ARTS config file (arts.cfg) into a key/value dict.

    Delegates to shared implementation in scripts.arts_config.
    """
    return _shared_parse_arts_cfg(path)


_EMBEDDED_ARTS_CFG_KEYS = (
    "threads",
    "incoming",
    "outgoing",
    "nodeCount",
    "launcher",
    "protocol",
    "port",
    "counterFolder",
)


def _read_text_lossy(path: Path) -> str:
    """Read a text or binary artifact into a string for inspection."""
    try:
        return path.read_text(errors="replace")
    except Exception:
        try:
            return path.read_bytes().decode("utf-8", errors="replace")
        except Exception:
            return ""


def _extract_embedded_arts_cfg(artifacts_dir: Path) -> Dict[str, str]:
    """Extract embedded ARTS config values from LLVM IR or executable output."""
    candidates = sorted(artifacts_dir.glob("*-arts.ll"))
    if not candidates:
        candidates = sorted(artifacts_dir.glob("*_arts"))

    for candidate in candidates:
        text = _read_text_lossy(candidate)
        if not text:
            continue

        embedded: Dict[str, str] = {}
        for key in _EMBEDDED_ARTS_CFG_KEYS:
            match = re.search(
                rf"{re.escape(key)}=(.*?)(?:\\0A|\x00|\r|\n|\")",
                text,
            )
            if match:
                embedded[key] = match.group(1)
        if embedded:
            return embedded

    return {}


def _validate_embedded_arts_cfg(
    artifacts_dir: Path,
    expected_cfg: Optional[Path],
) -> Optional[str]:
    """Return an error string if the built artifact embeds the wrong config."""
    if expected_cfg is None or not expected_cfg.exists():
        return None

    expected = parse_arts_cfg(expected_cfg)
    if not expected:
        return f"Failed to parse expected arts.cfg: {expected_cfg}"

    embedded = _extract_embedded_arts_cfg(artifacts_dir)
    if not embedded:
        return (
            "Failed to inspect generated ARTS artifact for embedded config "
            f"in {artifacts_dir}"
        )

    mismatches: List[str] = []
    for key in _EMBEDDED_ARTS_CFG_KEYS:
        expected_value = expected.get(key)
        if expected_value is None:
            continue
        embedded_value = embedded.get(key)
        if embedded_value != expected_value:
            mismatches.append(
                f"{key}: expected '{expected_value}', embedded '{embedded_value}'"
            )

    if not mismatches:
        return None

    return (
        "Generated ARTS artifact embeds a different config than the compile-time "
        f"arts.cfg. cfg={expected_cfg}, artifacts={artifacts_dir}. "
        + "; ".join(mismatches)
    )


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
            variant="openmp",
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
                "CARTS_BENCHMARKS_REPORT_INIT": "1",
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
        variant: str = "arts",
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
        if variant != "openmp" and effective_arts_config is None:
            # Keep build behavior independent of current working directory.
            effective_arts_config = _resolve_effective_arts_config(bench_path)
        if variant != "openmp" and effective_arts_config is not None:
            effective_arts_config = effective_arts_config.resolve()
            if build_output_dir is not None:
                # Keep the exact compile-time config alongside the build artifacts.
                local_cfg = output_root / "arts.cfg"
                if effective_arts_config != local_cfg.resolve():
                    shutil.copy2(effective_arts_config, local_cfg)
                effective_arts_config = local_cfg

        if variant == "openmp":
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
        if effective_arts_config and variant != "openmp":
            cmd.append(f"ARTS_CFG={effective_arts_config.resolve()}")
        if compile_args and variant != "openmp":
            escaped_args = compile_args.replace("\\", "\\\\").replace(" ", "\\ ")
            cmd.append(f"COMPILE_ARGS={escaped_args}")

        # Debug output level 1: show commands
        if self.debug >= 1:
            env_prefix = " ".join(f"{k}={v}" for k, v in env_overrides.items())
            if env_prefix:
                self.console.print(
                    f"[dim]$ cd {bench_path} && {env_prefix} {' '.join(cmd)}[/]"
                )
            else:
                self.console.print(f"[dim]$ cd {bench_path} && {' '.join(cmd)}[/]")

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
                self.console.print(f"[dim]  Log: {log_file}[/]")

            if result.returncode == 0:
                expected_exe = arts_output_path if variant != "openmp" else omp_output_path
                executable = None
                if expected_exe.is_file() and os.access(expected_exe, os.X_OK):
                    executable = str(expected_exe)
                else:
                    executable = self._find_executable(bench_path, variant)
                if variant != "openmp":
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
            init_timings=self.extract_init_timings(outcome.stdout),
            parallel_task_timing=self.extract_parallel_task_timings(outcome.stdout),
            perf_metrics=perf_metrics,
            perf_csv_path=perf_csv_path,
        )

    def _create_common_env(self) -> Dict[str, str]:
        """Return environment overrides shared by local benchmark runs."""
        env: Dict[str, str] = {}
        if "CARTS_BENCHMARKS_REPORT_INIT" not in os.environ:
            env["CARTS_BENCHMARKS_REPORT_INIT"] = "1"
        return env

    def _create_execution_plan(
        self,
        *,
        execution: BenchmarkExecutionContext,
        timeout: int,
        run_numbers: Tuple[int, ...],
        verify: bool,
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
    ) -> ConfigExecutionPlan:
        """Create the shared execution plan for one resolved benchmark config."""
        return ConfigExecutionPlan(
            execution=execution,
            timeout=timeout,
            run_numbers=run_numbers,
            verify=verify,
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
                _validate_thread_network_topology(
                    effective_config, threads, desired_nodes, name
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
                    verify=True,
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
            counter_dir: Optional path to override the embedded counterFolder config value
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
        report_speedup: bool = True,
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

        if not report_speedup:
            speedup = 0.0
            note = "Speedup hidden for distributed runs (unfair comparison)"
            speedup_basis = "n/a"
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
        compile_args: Optional[str] = None,
        phase_callback: Optional[Callable[[Phase], None]] = None,
        partial_results: Optional[Dict[str, Any]] = None,
        perf_enabled: bool = False,
        perf_interval: float = 0.1,
        run_number: int = 1,
        run_timestamp: str = "",
        perf_dir: Optional[Path] = None,
        cflags: str = "",
    ) -> BenchmarkResult:
        """Run complete pipeline for a single benchmark.

        Args:
            phase_callback: Optional callback invoked when phase changes.
                           Used by run_all to update live display.
            partial_results: Optional dict to store partial results as phases complete.
        """
        bench_path = self.benchmarks_dir / name

        # Determine effective config template.
        effective_config = _resolve_effective_arts_config(bench_path, arts_config)

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
        _validate_thread_network_topology(
            effective_config, desired_threads, desired_nodes, name
        )
        am = self.artifact_manager  # shorthand (may be None)

        # Skip benchmarks disabled for multi-node when running with nodeCount > 1
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
        if run_counter_dir is not None:
            need_generated = True

        effective_arts_cfg: Path
        if need_generated:
            effective_arts_cfg = generate_arts_config(
                effective_config,
                desired_threads,
                run_counter_dir,
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
            verify=verify,
            compile_args=compile_args,
            perf_enabled=perf_enabled,
            perf_interval=perf_interval,
            counter_dir=counter_dir,
            perf_dir=perf_dir,
            run_timestamp=run_timestamp,
            sweep_log_names=False,
            report_speedup=(desired_nodes == 1),
            env_overrides=self._create_common_env(),
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
        verify: bool = True,
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
                        compile_args,
                        perf_enabled=perf_enabled,
                        perf_interval=perf_interval,
                        run_number=run_num,
                        run_timestamp=run_timestamp,
                        perf_dir=perf_dir,
                        cflags=cflags,
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
                            compile_args,
                            phase_callback,
                            current_partial[0],
                            perf_enabled=perf_enabled,
                            perf_interval=perf_interval,
                            run_number=run_num,
                            run_timestamp=run_timestamp,
                            perf_dir=perf_dir,
                            cflags=cflags,
                        )
                    except Exception as e:
                        # Log error and continue to next benchmark
                        self.console.print(
                            f"[red]Error running {bench}:[/] {e}")
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
                        self.console.print(
                            f"[red]Error running {bench}:[/] {e}")
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
                        self.console.print(
                            f"[red]Error running {bench}:[/] {e}")
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


# NOTE: SVG/report generation code was removed.
# Benchmark reports are generated automatically into each results directory.



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

        if p.exists():
            return str(p.resolve())

        search_dirs: List[Path] = []
        if field == "arts_config":
            search_dirs = [CONFIGS_DIR]
        elif field == "profile":
            search_dirs = [PROFILES_DIR]

        for search_dir in search_dirs:
            candidate = (search_dir / p).resolve()
            if candidate.exists():
                return str(candidate)

        raise ValueError(f"Cannot resolve {field} '{raw}' - file not found")

    unknown = sorted(set(normalized.keys()) - KNOWN_STEP_KEYS)
    if unknown:
        console.print(
            f"[yellow]Warning:[/] Unknown step keys: {', '.join(unknown)}"
        )

    step = ExperimentStep(
        name=step_name,
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

    if isinstance(payload, dict):
        if "setup_commands" in payload:
            raise ValueError(
                "Experiment field `setup_commands` is no longer supported."
            )
        known_top_keys = {"name", "description", "steps"} | KNOWN_STEP_KEYS
        unknown_top = sorted(set(payload.keys()) - known_top_keys)
        if unknown_top:
            console.print(
                f"[yellow]Warning:[/] Unknown experiment keys: {', '.join(unknown_top)}"
            )
        raw_steps = payload.get("steps")
        defaults = {k: v for k, v in payload.items() if k in KNOWN_STEP_KEYS}
    elif isinstance(payload, list):
        raw_steps = payload
        defaults = {}
    else:
        raise ValueError(f"Invalid experiment format in {exp_path}")

    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"Experiment '{experiment}' has no steps")

    steps: List[ExperimentStep] = []
    for idx, item in enumerate(raw_steps, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Step {idx} in {exp_path} must be an object")
        merged = {**defaults, **item}
        steps.append(
            _make_experiment_step(
                merged,
                default_name=f"step_{idx}",
                base_dir=exp_path.parent,
            )
        )
    return steps


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
    verify: bool,
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
                        f"[bold]Sweep config {config_idx}/{total_configs}:[/] "
                        f"threads={threads_label}, nodes={nodes_label}"
                    )

                config_results = runner.run_all(
                    bench_list,
                    size=size,
                    timeout=timeout,
                    verify=verify,
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
                )
                results.extend(config_results)
        return results

    return runner.run_all(
        bench_list,
        size=size,
        timeout=timeout,
        verify=verify,
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
    )


def _run_step_slurm(
    bench_list: List[str],
    size: str,
    node_counts: List[int],
    runs: int,
    verify: bool,
    partition: Optional[str],
    time_limit: str,
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
            runs=runs,
            verify=verify,
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
    if not step_config.should_rebuild_arts:
        return
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
        verify=request.verify,
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
        verify=request.verify,
        partition=request.partition,
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
    time_limit: str = typer.Option(
        "01:00:00", "--time-limit", help="SLURM time limit per job (only with --slurm)"),
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
             "When set, new jobs are submitted as earlier ones finish. "
             "Only with --slurm"),
):
    """Run benchmarks with verification and timing."""
    try:
        size = parse_size(size, "--size")
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(2)

    verify = not no_verify
    clean = not no_clean
    size_from_cli = _is_option_from_cli(ctx, "size", "--size", "-s")

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
            console.print("[dim]Use `carts benchmarks list` to see valid names.[/]")
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
            console.print(f"[dim]  ({excluded} benchmarks excluded via --exclude)[/]")

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
                    verify=verify,
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
            console.print(f"\n[red]Error:[/] {e}")
            raise typer.Exit(1)
        return

    if exclude_nodes:
        print_warning("Ignoring --exclude-nodes because --slurm is not enabled.")

    # Print header
    if not quiet:
        config_items = [f"size={effective_size_label}", f"timeout={timeout}s",
                        f"verify={verify}", f"clean={clean}"]
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
            if arts_config:
                effective_config = arts_config
                config_source = "custom"
            else:
                effective_config = DEFAULT_ARTS_CONFIG
                config_source = "default"

            cfg = parse_arts_cfg(effective_config)
            arts_threads = int(cfg.get("threads", "1"))
            arts_nodes = int(cfg.get("nodeCount", "1"))
            arts_launcher = cfg.get("launcher", "ssh")

            # Apply CLI overrides for display
            if base_threads_list and len(base_threads_list) == 1:
                arts_threads = int(base_threads_list[0])
            if base_node_counts and len(base_node_counts) == 1:
                arts_nodes = int(base_node_counts[0])
            if launcher:
                arts_launcher = launcher

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
                verify=verify,
                omp_threads=omp_threads,
                weak_scaling=weak_scaling,
                base_size=base_size,
                run_timestamp=run_timestamp,
                clean=clean,
                quiet=quiet,
                artifact_manager=am,
            ),
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
    failed = sum(1 for r in results if r.run_arts.status in (
        Status.FAIL, Status.CRASH))
    if failed > 0:
        raise typer.Exit(1)


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
            console.print("[dim]Use `carts benchmarks list` to see valid names.[/]")
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
                    "[dim]Use `carts benchmarks list` to see valid names.[/]")
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
    runs: int = typer.Option(
        1, "--runs", "-r", help="Number of runs per benchmark"),
    verify: bool = True,
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

    # Parse node counts from --nodes parameter
    node_counts = parse_node_spec(nodes)

    # Determine explicit arts config override (if provided).
    explicit_arts_config = arts_config.resolve() if arts_config else None
    if explicit_arts_config is not None and not explicit_arts_config.exists():
        raise ValueError(f"arts.cfg not found: {explicit_arts_config}")

    # Get threads from explicit config or default config.
    if threads is None:
        thread_source_cfg = explicit_arts_config or DEFAULT_ARTS_CONFIG.resolve()
        threads = get_arts_cfg_int(thread_source_cfg, "threads") or 8

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
            console.print("[dim]Use `carts benchmarks list` to see valid names.[/]")
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
        verify=verify,
        partition=partition,
        time_limit=time_limit,
        account=account,
        explicit_arts_config=explicit_arts_config,
        threads=threads,
        output_dir=output_dir,
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
        validate_thread_network_topology=_validate_thread_network_topology,
        parse_time_limit_seconds=parse_slurm_time_limit_seconds,
        get_carts_dir=get_carts_dir,
        get_benchmarks_dir=get_benchmarks_dir,
        step_name_to_token=_STEP_RESOLVER.step_name_to_token,
    )
    SlurmBatchExecutor(runner, request, deps).execute()


if __name__ == "__main__":
    # Enable debug logging when --verbose/-v is in args
    if "--verbose" in sys.argv or "-v" in sys.argv:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    app()
