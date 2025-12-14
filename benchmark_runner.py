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
from typing import Any, Dict, List, Optional, Tuple

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

CHECKSUM_PATTERNS = [
    r"checksum[:\s]*=?\s*([0-9.eE+-]+)",
    r"result[:\s]*=?\s*([0-9.eE+-]+)",
    r"sum[:\s]*=?\s*([0-9.eE+-]+)",
    r"total[:\s]*=?\s*([0-9.eE+-]+)",
    r"RMS error[:\s]*\(?\s*([0-9.eE+-]+)",
    r"^([0-9.eE+-]+)\s*$",
]

SKIP_DIRS = {"common", "include", "src", "utilities", ".git", ".svn", ".hg", "build", "logs"}


def filter_benchmark_output(output: str) -> str:
    """Extract only CARTS benchmark output lines (kernel timing, parallel/task timing, checksum).

    Filters out verbose ARTS runtime debug logs and keeps only benchmark-relevant output.
    """
    if not output:
        return ""
    lines = []
    for line in output.splitlines():
        # Keep lines that match benchmark output patterns
        if (line.startswith("kernel.") or
            line.startswith("parallel.") or
            line.startswith("task.") or
            line.startswith("checksum:") or
            line.startswith("tmp_checksum:") or
            "checksum:" in line.lower()):
            lines.append(line)
    return "\n".join(lines)


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
    parallel_timings: Dict[str, List[WorkerTiming]] = field(default_factory=dict)
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
        parallel = {t.worker_id: t.time_sec for t in self.parallel_timings.get(parallel_name, [])}
        task = {t.worker_id: t.time_sec for t in self.task_timings.get(task_name, [])}

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
    parallel_task_timing: Optional[ParallelTaskTiming] = None


@dataclass
class TimingResult:
    """Timing comparison between ARTS and OpenMP."""
    arts_time_sec: float
    omp_time_sec: float
    speedup: float  # omp_time / arts_time (>1 = ARTS faster)
    note: str


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
    benchmark_dir: str
    build_dir: Optional[str] = None
    executable_arts: Optional[str] = None
    executable_omp: Optional[str] = None
    mlir_sequential: Optional[str] = None
    mlir_parallel: Optional[str] = None
    carts_metadata: Optional[str] = None
    counters_dir: Optional[str] = None
    counter_files: List[str] = field(default_factory=list)
    logs_dir: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Complete result for a single benchmark."""
    name: str
    suite: str
    size: str
    build_arts: BuildResult
    build_omp: BuildResult
    run_arts: RunResult
    run_omp: RunResult
    timing: TimingResult
    verification: VerificationResult
    artifacts: Artifacts
    timestamp: str
    total_duration_sec: float
    thread_count: Optional[int] = None
    node_count: Optional[int] = None
    launcher: Optional[str] = None


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


def generate_arts_config(
    base_path: Optional[Path],
    threads: int,
    counter_dir: Optional[Path] = None,
    launcher: str = "ssh",
    node_count: int = 1,
) -> Path:
    """Generate temporary arts.cfg with specific configuration.

    The generated config must include all required ARTS keys:
    - launcher: Required (ARTS dereferences without null check in Config.c:621)
    - threads: Worker thread count per node
    - nodeCount: Number of nodes (1 for single-node)
    - masterNode: Primary node for coordination

    Note: For Slurm, nodeCount in config is IGNORED - ARTS reads SLURM_NNODES
    from environment (set by srun). The launcher controls HOW we invoke the
    executable, not just what's in the config.
    """
    if base_path and base_path.exists():
        content = base_path.read_text()
    else:
        # Default config with ALL required keys
        # ARTS requires launcher to be set (Config.c:621 dereferences without null check)
        # NOTE: launcher=local with ports=0 causes segfaults on shutdown
        content = f"""[ARTS]
threads=1
nodeCount=1
launcher={launcher}
masterNode=localhost
tMT=0
outgoing=1
incoming=1
ports=1
"""

    # Update threads
    if re.search(r'^threads=', content, re.MULTILINE):
        content = re.sub(r'^threads=\d+', f'threads={threads}', content, flags=re.MULTILINE)
    else:
        content = content.replace('[ARTS]', f'[ARTS]\nthreads={threads}')

    # Update nodeCount
    if re.search(r'^nodeCount=', content, re.MULTILINE):
        content = re.sub(r'^nodeCount=\d+', f'nodeCount={node_count}', content, flags=re.MULTILINE)
    else:
        content = content.replace('[ARTS]', f'[ARTS]\nnodeCount={node_count}')

    # Update launcher
    if re.search(r'^launcher=', content, re.MULTILINE):
        content = re.sub(r'^launcher=\w+', f'launcher={launcher}', content, flags=re.MULTILINE)
    else:
        content = content.replace('[ARTS]', f'[ARTS]\nlauncher={launcher}')

    # Add counter settings if requested
    if counter_dir:
        content += f"\ncounterFolder={counter_dir}\ncounterStartPoint=1\n"

    # Write to temp file
    temp_path = Path(tempfile.mkdtemp()) / f"arts_{threads}t_{node_count}n.cfg"
    temp_path.write_text(content)
    return temp_path


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
    ):
        self.console = console
        self.verbose = verbose
        self.quiet = quiet
        self.trace = trace
        self.clean = clean
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
            has_source = any(bench_dir.glob("*.c")) or any(bench_dir.glob("*.cpp"))
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

    def build_benchmark(
        self,
        name: str,
        size: str,
        variant: str = "arts",
        arts_config: Optional[Path] = None,
        cflags: str = "",
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
            "mini": "mini",
            "standard": "standard",
        }

        # Build command using make
        if variant == "openmp":
            # Build only OpenMP variant - pass size via CFLAGS, not make target
            # This ensures OMP builds are independent of ARTS builds
            size_flags = {
                "small": "-DSMALL_DATASET",
                "mini": "-DMINI_DATASET",
                "medium": "-DSTANDARD_DATASET",
                "standard": "-DSTANDARD_DATASET",
                "large": "-DLARGE_DATASET",
            }
            all_cflags = size_flags.get(size, "")
            if cflags:
                all_cflags = f"{all_cflags} {cflags}".strip()
            cmd = ["make", "openmp"]
            if all_cflags:
                cmd.append(f"CFLAGS={all_cflags}")
        else:
            # Build ARTS variant (full pipeline)
            if size in size_targets:
                # Size target builds both all and openmp
                cmd = ["make", size]
                if cflags:
                    cmd.append(f"CFLAGS={cflags}")
            else:
                cmd = ["make", "all"]
                if cflags:
                    cmd.append(f"CFLAGS={cflags}")

        # Add ARTS config override if provided
        if arts_config and variant != "openmp":
            cmd.append(f"ARTS_CFG={arts_config}")

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
        launcher: str = "ssh",
        node_count: int = 1,
    ) -> List[BenchmarkResult]:
        """Run benchmark with multiple thread configurations.

        Args:
            omp_threads: If specified, use this fixed thread count for OpenMP
                        instead of matching ARTS thread count. Useful for
                        comparing ARTS scaling against OpenMP at its optimal.
            launcher: ARTS launcher type (ssh, slurm, lsf, local). For Slurm,
                     this controls how the executable is invoked (via srun).
            node_count: Number of nodes for distributed execution.
        """
        results = []
        for threads in threads_list:
            # Generate arts.cfg with thread count, launcher, and node count
            arts_cfg = generate_arts_config(
                base_config, threads, counter_dir, launcher, node_count
            )

            # Set OMP_NUM_THREADS for OpenMP variant
            # Use omp_threads if specified, otherwise match ARTS threads
            actual_omp_threads = omp_threads if omp_threads else threads
            env = {"OMP_NUM_THREADS": str(actual_omp_threads)}

            # Build both variants
            build_arts = self.build_benchmark(name, size, "arts", arts_cfg, cflags)
            build_omp = self.build_benchmark(name, size, "openmp", None, cflags)

            # Run ARTS version - MUST pass artsConfig env var so runtime uses same config as compile
            # ARTS reads config from: 1) artsConfig env var, 2) ./arts.cfg in CWD
            # Without this, runtime may use wrong config (thread count mismatch)
            if build_arts.status == Status.PASS and build_arts.executable:
                arts_env = {"artsConfig": str(arts_cfg)}
                run_arts = self.run_benchmark(
                    build_arts.executable,
                    timeout,
                    env=arts_env,
                    launcher=launcher,
                    node_count=node_count,
                    threads=threads,
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
                run_omp = self.run_benchmark(build_omp.executable, timeout, env=env)
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
            verification = self.verify_correctness(run_arts, run_omp)

            # Collect artifacts
            bench_path = self.benchmarks_dir / name
            artifacts = self.collect_artifacts(bench_path)

            # Store with thread count and node configuration
            result = BenchmarkResult(
                name=name,
                suite=name.split("/")[0] if "/" in name else "",
                size=size,
                build_arts=build_arts,
                build_omp=build_omp,
                run_arts=run_arts,
                run_omp=run_omp,
                timing=timing,
                verification=verification,
                artifacts=artifacts,
                timestamp=datetime.now().isoformat(),
                total_duration_sec=0.0,  # Will be calculated later
                thread_count=threads,
                node_count=node_count,
                launcher=launcher,
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
    ) -> RunResult:
        """Execute a benchmark and capture output.

        Args:
            executable: Path to the benchmark executable.
            timeout: Maximum execution time in seconds.
            env: Environment variables to set.
            launcher: ARTS launcher type. For 'slurm', wraps executable in srun.
            node_count: Number of nodes for distributed execution (slurm only).
            threads: Number of threads per node (for srun --cpus-per-task).
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

            # Determine status based on exit code
            if result.returncode == 0:
                status = Status.PASS
            elif result.returncode in (139, 134, 136):  # SEGV, ABRT, FPE
                status = Status.CRASH
            else:
                status = Status.FAIL

            checksum = self.extract_checksum(result.stdout)
            kernel_timings = self.extract_kernel_timings(result.stdout)
            parallel_task_timing = self.extract_parallel_task_timings(result.stdout)

            return RunResult(
                status=status,
                duration_sec=duration,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                checksum=checksum,
                kernel_timings=kernel_timings,
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
        pattern = r"kernel\.(\w+):\s*([0-9.]+)s"
        for match in re.finditer(pattern, output):
            kernel_name = match.group(1)
            kernel_time = float(match.group(2))
            timings[kernel_name] = kernel_time
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
            result.parallel_timings[name].append(WorkerTiming(worker_id, time_sec))
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
        """Calculate speedup: omp_time / arts_time."""
        if arts_result.status != Status.PASS or omp_result.status != Status.PASS:
            return TimingResult(
                arts_time_sec=arts_result.duration_sec,
                omp_time_sec=omp_result.duration_sec,
                speedup=0.0,
                note="Cannot calculate: one or both runs failed",
            )

        if arts_result.duration_sec == 0:
            speedup = 0.0
            note = "ARTS time is zero"
        else:
            speedup = omp_result.duration_sec / arts_result.duration_sec
            if speedup > 1:
                note = f"ARTS is {speedup:.2f}x faster"
            elif speedup < 1:
                note = f"OpenMP is {1/speedup:.2f}x faster"
            else:
                note = "Same performance"

        return TimingResult(
            arts_time_sec=arts_result.duration_sec,
            omp_time_sec=omp_result.duration_sec,
            speedup=speedup,
            note=note,
        )

    def collect_artifacts(self, bench_path: Path) -> Artifacts:
        """Collect all artifact paths for a benchmark."""
        build_dir = bench_path / "build"
        counters_dir = bench_path / "counters"
        logs_dir = bench_path / "logs"

        artifacts = Artifacts(benchmark_dir=str(bench_path))

        if build_dir.exists():
            artifacts.build_dir = str(build_dir)

        if logs_dir.exists():
            artifacts.logs_dir = str(logs_dir)

        # Find executables
        for exe in bench_path.glob("*_arts"):
            if exe.is_file() and os.access(exe, os.X_OK):
                artifacts.executable_arts = str(exe)
                break

        for exe in bench_path.glob("*_omp"):
            if exe.is_file() and os.access(exe, os.X_OK):
                artifacts.executable_omp = str(exe)
                break

        # Find MLIR files in build directory
        if build_dir.exists():
            for mlir in build_dir.glob("*_seq.mlir"):
                artifacts.mlir_sequential = str(mlir)
                break

            for mlir in build_dir.glob("*.mlir"):
                if "_seq" not in mlir.name:
                    artifacts.mlir_parallel = str(mlir)
                    break

            # Find CARTS metadata JSON
            for meta in build_dir.glob("*.carts-metadata.json"):
                artifacts.carts_metadata = str(meta)
                break

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
    ) -> BenchmarkResult:
        """Run complete pipeline for a single benchmark."""
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        bench_path = self.benchmarks_dir / name
        suite = name.split("/")[0] if "/" in name else ""

        # Clean before building to avoid stale artifacts
        if self.clean:
            self.clean_benchmark(name)

        # Build ARTS version
        build_arts = self.build_benchmark(name, size, "arts", arts_config)

        # Build OpenMP version
        build_omp = self.build_benchmark(name, size, "openmp", arts_config)

        # Run ARTS version - pass artsConfig env var so runtime uses same config as compile
        if build_arts.status == Status.PASS and build_arts.executable:
            arts_env = {"artsConfig": str(arts_config)} if arts_config else None
            run_arts = self.run_benchmark(build_arts.executable, timeout, env=arts_env)
        else:
            run_arts = RunResult(
                status=Status.SKIP,
                duration_sec=0.0,
                exit_code=-1,
                stdout="",
                stderr="Build failed",
            )

        # Run OpenMP version
        if build_omp.status == Status.PASS and build_omp.executable:
            run_omp = self.run_benchmark(build_omp.executable, timeout)
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

            self.console.print(f"\n[bold cyan]═══ CARTS Output ({name}) ═══[/]")
            self.console.print(arts_output or "[dim](no benchmark output)[/]")

            self.console.print(f"\n[bold green]═══ OMP Output ({name}) ═══[/]")
            self.console.print(omp_output or "[dim](no benchmark output)[/]")
            self.console.print()

        # Calculate timing
        timing = self.calculate_timing(run_arts, run_omp)

        # Verify correctness
        if verify:
            verification = self.verify_correctness(run_arts, run_omp)
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

        total_duration = time.time() - start_time

        return BenchmarkResult(
            name=name,
            suite=suite,
            size=size,
            build_arts=build_arts,
            build_omp=build_omp,
            run_arts=run_arts,
            run_omp=run_omp,
            timing=timing,
            verification=verification,
            artifacts=artifacts,
            timestamp=timestamp,
            total_duration_sec=total_duration,
        )

    def run_all(
        self,
        benchmarks: List[str],
        size: str = DEFAULT_SIZE,
        timeout: int = DEFAULT_TIMEOUT,
        parallel: int = 1,
        verify: bool = True,
        arts_config: Optional[Path] = None,
    ) -> List[BenchmarkResult]:
        """Run benchmark suite with optional parallelization."""
        if parallel > 1:
            return self._run_parallel(
                benchmarks, size, timeout, parallel, verify, arts_config
            )

        results_dict: Dict[str, BenchmarkResult] = {}
        results_list: List[BenchmarkResult] = []
        start_time = time.time()

        if self.quiet:
            # Quiet mode - no live display
            for bench in benchmarks:
                result = self.run_single(bench, size, timeout, verify, arts_config)
                results_list.append(result)
            self.results = results_list
            return results_list

        # Live display mode - show table that updates as benchmarks complete
        with Live(
            create_live_display(benchmarks, results_dict, None, 0),
            console=self.console,
            refresh_per_second=4,
        ) as live:
            for bench in benchmarks:
                # Update display to show current benchmark as in-progress
                elapsed = time.time() - start_time
                live.update(create_live_display(benchmarks, results_dict, bench, elapsed))

                # Run benchmark
                result = self.run_single(bench, size, timeout, verify, arts_config)

                # Update results and refresh display
                results_dict[bench] = result
                results_list.append(result)
                elapsed = time.time() - start_time
                live.update(create_live_display(benchmarks, results_dict, None, elapsed))

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
        in_progress: set = set(benchmarks)  # All benchmarks start as in-progress

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
                        self.clean,
                    ): bench
                    for bench in benchmarks
                }

                for future in as_completed(futures):
                    bench = futures[future]
                    try:
                        result = future.result()
                        results_list.append(result)
                    except Exception as e:
                        results_list.append(self._make_error_result(bench, size, str(e)))

            self.results = results_list
            return results_list

        # Live display mode - show table that updates as benchmarks complete
        with Live(
            create_live_display(benchmarks, results_dict, f"[parallel={n_workers}]", 0),
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
                        self.clean,
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
                        error_result = self._make_error_result(bench, size, str(e))
                        results_dict[bench] = error_result
                        results_list.append(error_result)

                    in_progress.remove(bench)
                    elapsed = time.time() - start_time

                    # Show one of the remaining in-progress benchmarks
                    current_in_progress = next(iter(in_progress), None)
                    live.update(create_live_display(benchmarks, results_dict, current_in_progress, elapsed))

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

        return BenchmarkResult(
            name=name,
            suite=suite,
            size=size,
            build_arts=failed_build,
            build_omp=failed_build,
            run_arts=failed_run,
            run_omp=failed_run,
            timing=TimingResult(0.0, 0.0, 0.0, "Error"),
            verification=VerificationResult(False, None, None, 0.0, error),
            artifacts=Artifacts(benchmark_dir=str(bench_path)),
            timestamp=datetime.now().isoformat(),
            total_duration_sec=0.0,
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
    clean: bool = True,
) -> BenchmarkResult:
    """Worker function for parallel benchmark execution."""
    runner = BenchmarkRunner(Console(force_terminal=False), quiet=True, clean=clean)
    runner.benchmarks_dir = Path(benchmarks_dir)
    return runner.run_single(
        name,
        size,
        timeout,
        verify,
        Path(arts_config) if arts_config else None,
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


def create_results_table(results: List[BenchmarkResult]) -> Table:
    """Create a rich table from benchmark results."""
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")

    table.add_column("Benchmark", style="cyan", no_wrap=True)
    table.add_column("Build", justify="center")
    table.add_column("ARTS Kernel", justify="right")
    table.add_column("OMP Kernel", justify="right")
    table.add_column("Correct", justify="center")
    table.add_column("Speedup", justify="right")

    has_fallback = False
    has_multi_kernel = False
    for r in results:
        # Build status (combined)
        if r.build_arts.status == Status.PASS and r.build_omp.status == Status.PASS:
            build = f"[green]\u2713[/] {r.build_arts.duration_sec + r.build_omp.duration_sec:.1f}s"
        else:
            build = f"[red]\u2717[/] {r.build_arts.status.value}/{r.build_omp.status.value}"

        # Get kernel times with formatted display
        arts_kernel, arts_kernel_str = format_kernel_time(r.run_arts)

        # Track if any benchmark has multiple kernels
        if r.run_arts.kernel_timings and len(r.run_arts.kernel_timings) > 1:
            has_multi_kernel = True
        omp_kernel, omp_kernel_str = format_kernel_time(r.run_omp)

        # Run status with kernel time
        if r.run_arts.status == Status.PASS:
            if arts_kernel is not None:
                run_arts = f"{status_symbol(r.run_arts.status)} {arts_kernel_str}"
            else:
                run_arts = f"{status_symbol(r.run_arts.status)} {r.run_arts.duration_sec:.2f}s*"
                has_fallback = True
        else:
            run_arts = f"{status_symbol(r.run_arts.status)} {r.run_arts.status.value}"

        if r.run_omp.status == Status.PASS:
            if omp_kernel is not None:
                run_omp = f"{status_symbol(r.run_omp.status)} {omp_kernel_str}"
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

        # Speedup based on kernel time if available
        if arts_kernel is not None and omp_kernel is not None and arts_kernel > 0:
            kernel_speedup = omp_kernel / arts_kernel
            if kernel_speedup >= 1.0:
                speedup = f"[green]{kernel_speedup:.2f}x[/]"
            elif kernel_speedup >= 0.8:
                speedup = f"[yellow]{kernel_speedup:.2f}x[/]"
            else:
                speedup = f"[red]{kernel_speedup:.2f}x[/]"
        elif r.timing.speedup > 0:
            # Fallback to total time speedup
            if r.timing.speedup >= 1:
                speedup = f"[green]{r.timing.speedup:.2f}x[/]*"
            else:
                speedup = f"[yellow]{r.timing.speedup:.2f}x[/]*"
            has_fallback = True
        else:
            speedup = "[dim]-[/]"

        table.add_row(
            r.name,
            build,
            run_arts,
            run_omp,
            correct,
            speedup,
        )

    # Build caption based on what notations are used
    captions = []
    if has_multi_kernel:
        captions.append("[N] = sum of N kernels")
    if has_fallback:
        captions.append("* = total time (kernel timing unavailable)")
    if captions:
        table.caption = "[dim]" + "  |  ".join(captions) + "[/]"

    return table


def create_summary_panel(results: List[BenchmarkResult], duration: float) -> Panel:
    """Create a summary panel."""
    passed = sum(1 for r in results if r.run_arts.status == Status.PASS and r.verification.correct)
    failed = sum(1 for r in results if r.run_arts.status in (Status.FAIL, Status.CRASH) or
                 (r.run_arts.status == Status.PASS and not r.verification.correct))
    skipped = sum(1 for r in results if r.run_arts.status == Status.SKIP)

    # Calculate geometric mean speedup based on kernel time
    import math
    speedups = []
    for r in results:
        arts_kernel = get_kernel_time(r.run_arts)
        omp_kernel = get_kernel_time(r.run_omp)
        if arts_kernel is not None and omp_kernel is not None and arts_kernel > 0:
            speedups.append(omp_kernel / arts_kernel)
        elif r.timing.speedup > 0:
            speedups.append(r.timing.speedup)

    if speedups:
        geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        speedup_text = f"Geometric mean speedup: [cyan]{geomean:.2f}x[/]"
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


def create_live_table(
    benchmarks: List[str],
    results: Dict[str, BenchmarkResult],
    in_progress: Optional[str] = None,
) -> Table:
    """Create a live-updating table showing benchmark progress."""
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")

    table.add_column("Benchmark", style="cyan", no_wrap=True)
    table.add_column("Build", justify="center")
    table.add_column("ARTS Kernel", justify="right")
    table.add_column("OMP Kernel", justify="right")
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

            # Get kernel times with formatted display (includes count for multiple kernels)
            arts_kernel, arts_kernel_str = format_kernel_time(r.run_arts)
            omp_kernel, omp_kernel_str = format_kernel_time(r.run_omp)

            # Run status with kernel time
            if r.run_arts.status == Status.PASS:
                if arts_kernel is not None:
                    run_arts = f"{status_symbol(r.run_arts.status)} {arts_kernel_str}"
                else:
                    run_arts = f"{status_symbol(r.run_arts.status)} {r.run_arts.duration_sec:.2f}s*"
                    has_fallback = True
            else:
                run_arts = f"{status_symbol(r.run_arts.status)} {r.run_arts.status.value}"

            if r.run_omp.status == Status.PASS:
                if omp_kernel is not None:
                    run_omp = f"{status_symbol(r.run_omp.status)} {omp_kernel_str}"
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

            # Speedup based on kernel time if available
            if arts_kernel is not None and omp_kernel is not None and arts_kernel > 0:
                kernel_speedup = omp_kernel / arts_kernel
                if kernel_speedup >= 1.0:
                    speedup = f"[green]{kernel_speedup:.2f}x[/]"
                elif kernel_speedup >= 0.8:
                    speedup = f"[yellow]{kernel_speedup:.2f}x[/]"
                else:
                    speedup = f"[red]{kernel_speedup:.2f}x[/]"
            elif r.timing.speedup > 0:
                # Fallback to total time speedup
                if r.timing.speedup >= 1:
                    speedup = f"[green]{r.timing.speedup:.2f}x[/]*"
                else:
                    speedup = f"[yellow]{r.timing.speedup:.2f}x[/]*"
                has_fallback = True
            else:
                speedup = "[dim]-[/]"

            # Track if any benchmark has multiple kernels
            if r.run_arts.kernel_timings and len(r.run_arts.kernel_timings) > 1:
                has_multi_kernel = True

            table.add_row(bench, build, run_arts, run_omp, correct, speedup)

        elif bench == in_progress:
            # Currently running - show spinner indicator
            table.add_row(
                f"[bold]{bench}[/]",
                "[yellow]\u23f3...[/]",
                "[dim]-[/]",
                "[dim]-[/]",
                "[dim]-[/]",
                "[dim]-[/]",
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
            )

    # Build caption based on what notations are used
    captions = []
    if has_multi_kernel:
        captions.append("[N] = sum of N kernels")
    if has_fallback:
        captions.append("* = total time (kernel timing unavailable)")
    if captions:
        table.caption = "[dim]" + "  |  ".join(captions) + "[/]"

    return table


def create_live_summary(
    results: Dict[str, BenchmarkResult],
    total: int,
    elapsed: float,
) -> Text:
    """Create a one-line summary for live display."""
    passed = sum(1 for r in results.values() if r.run_arts.status == Status.PASS and r.verification.correct)
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
) -> Group:
    """Create the complete live display (table + summary)."""
    table = create_live_table(benchmarks, results, in_progress)
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


def _serialize_parallel_task_timing(timing: Optional[ParallelTaskTiming]) -> Optional[Dict]:
    """Serialize ParallelTaskTiming to JSON-compatible dict."""
    if timing is None:
        return None

    return {
        "parallel_timings": {
            name: [{"worker_id": t.worker_id, "time_sec": t.time_sec} for t in timings]
            for name, timings in timing.parallel_timings.items()
        },
        "task_timings": {
            name: [{"worker_id": t.worker_id, "time_sec": t.time_sec} for t in timings]
            for name, timings in timing.task_timings.items()
        },
    }


def export_json(
    results: List[BenchmarkResult],
    output_path: Path,
    size: str,
    total_duration: float,
    threads_list: Optional[List[int]] = None,
    cflags: Optional[str] = None,
) -> None:
    """Export results to JSON file."""
    carts_dir = get_carts_dir()

    # Collect metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "carts_version": f"git:{get_git_hash(carts_dir) or 'unknown'}",
        "hostname": platform.node(),
        "platform": platform.system().lower(),
        "python_version": platform.python_version(),
        "size": size,
        "total_duration_seconds": total_duration,
    }

    # Add thread sweep metadata if applicable
    if threads_list:
        metadata["thread_sweep"] = threads_list
    if cflags:
        metadata["cflags"] = cflags

    # Calculate summary
    passed = sum(1 for r in results if r.run_arts.status == Status.PASS and r.verification.correct)
    failed = sum(1 for r in results if r.run_arts.status in (Status.FAIL, Status.CRASH))
    skipped = sum(1 for r in results if r.run_arts.status == Status.SKIP)
    total = len(results)

    speedups = [r.timing.speedup for r in results if r.timing.speedup > 0]
    if speedups:
        import math
        avg_speedup = sum(speedups) / len(speedups)
        geomean_speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    else:
        avg_speedup = 0.0
        geomean_speedup = 0.0

    summary = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "pass_rate": passed / total if total > 0 else 0.0,
        "avg_speedup": avg_speedup,
        "geometric_mean_speedup": geomean_speedup,
    }

    # Convert results to dict
    def result_to_dict(r: BenchmarkResult) -> Dict[str, Any]:
        result_dict = {
            "name": r.name,
            "suite": r.suite,
            "size": r.size,
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
                "parallel_task_timing": _serialize_parallel_task_timing(r.run_arts.parallel_task_timing),
            },
            "run_omp": {
                "status": r.run_omp.status.value,
                "duration_sec": r.run_omp.duration_sec,
                "exit_code": r.run_omp.exit_code,
                "checksum": r.run_omp.checksum,
                "kernel_timings": r.run_omp.kernel_timings,
                "parallel_task_timing": _serialize_parallel_task_timing(r.run_omp.parallel_task_timing),
            },
            "timing": {
                "arts_time_sec": r.timing.arts_time_sec,
                "omp_time_sec": r.timing.omp_time_sec,
                "speedup": r.timing.speedup,
            },
            "verification": {
                "correct": r.verification.correct,
                "tolerance": r.verification.tolerance_used,
                "note": r.verification.note,
            },
            "artifacts": asdict(r.artifacts),
            "timestamp": r.timestamp,
        }

        # Add scaling configuration if present (for scaling experiments)
        if hasattr(r, 'thread_count') and r.thread_count is not None:
            result_dict["thread_count"] = r.thread_count
        if hasattr(r, 'node_count') and r.node_count is not None:
            result_dict["node_count"] = r.node_count
        if hasattr(r, 'launcher') and r.launcher is not None:
            result_dict["launcher"] = r.launcher

        return result_dict

    # Collect failures
    failures = []
    for r in results:
        if r.run_arts.status in (Status.FAIL, Status.CRASH, Status.TIMEOUT):
            failures.append({
                "name": r.name,
                "phase": "run_arts",
                "error": r.run_arts.status.value,
                "exit_code": r.run_arts.exit_code,
                "stderr": r.run_arts.stderr[:500] if r.run_arts.stderr else "",
                "artifacts": {
                    "benchmark_dir": r.artifacts.benchmark_dir,
                    "logs_dir": r.artifacts.logs_dir,
                },
            })
        elif r.build_arts.status == Status.FAIL:
            failures.append({
                "name": r.name,
                "phase": "build_arts",
                "error": "build_failed",
                "output": r.build_arts.output[:500],
                "artifacts": {
                    "benchmark_dir": r.artifacts.benchmark_dir,
                    "logs_dir": r.artifacts.logs_dir,
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
    suite: Optional[str] = typer.Option(None, "--suite", "-s", help="Filter by suite"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, plain"),
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

        console.print(f"\n[bold]Available CARTS Benchmarks[/] ({len(benchmarks)} total)\n")

        for suite_name in sorted(suites.keys()):
            if suite_name:
                console.print(f"[cyan]{suite_name}:[/]")
            for bench in sorted(suites[suite_name]):
                console.print(f"  {bench}")
            console.print()


@app.command()
def run(
    benchmarks: Optional[List[str]] = typer.Argument(None, help="Specific benchmarks to run"),
    size: str = typer.Option(DEFAULT_SIZE, "--size", "-s", help="Dataset size: small, medium, large"),
    timeout: int = typer.Option(DEFAULT_TIMEOUT, "--timeout", "-t", help="Execution timeout in seconds"),
    parallel: int = typer.Option(1, "--parallel", "-p", help="Number of parallel workers"),
    no_verify: bool = typer.Option(False, "--no-verify", help="Disable correctness verification"),
    no_clean: bool = typer.Option(False, "--no-clean", help="Skip cleaning before build (faster, but may use stale artifacts)"),
    arts_config: Optional[Path] = typer.Option(None, "--arts-config", help="Custom arts.cfg file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Export results to JSON file"),
    suite: Optional[str] = typer.Option(None, "--suite", help="Filter by suite"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output (CI mode)"),
    trace: bool = typer.Option(False, "--trace", help="Show benchmark output (kernel timing and checksum)"),
    threads: Optional[str] = typer.Option(None, "--threads", help="Thread counts: '1,2,4,8' or '1:16:2' for thread sweep"),
    omp_threads: Optional[int] = typer.Option(None, "--omp-threads", help="OpenMP thread count (default: same as ARTS threads)"),
    launcher: str = typer.Option("ssh", "--launcher", "-l", help="ARTS launcher: ssh (default), slurm, lsf, local"),
    node_count: int = typer.Option(1, "--node-count", "-n", help="Number of nodes for distributed execution"),
    cflags: Optional[str] = typer.Option(None, "--cflags", help="Additional CFLAGS: '-DNI=500 -DNJ=500'"),
    collect_counters: bool = typer.Option(False, "--collect-counters", help="Enable ARTS counter collection"),
    counter_dir: Optional[Path] = typer.Option(None, "--counter-dir", help="Counter output directory"),
):
    """Run benchmarks with verification and timing."""
    verify = not no_verify
    clean = not no_clean
    runner = BenchmarkRunner(console, verbose=verbose, quiet=quiet, trace=trace, clean=clean)

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

    # Setup counter directory
    if collect_counters and not counter_dir:
        counter_dir = Path("./counters")
    if counter_dir:
        counter_dir.mkdir(parents=True, exist_ok=True)

    # Print header
    if not quiet:
        console.print(f"\n[bold]CARTS Benchmark Runner v{VERSION}[/]")
        console.print("\u2501" * 30)
        config_items = [f"size={size}", f"timeout={timeout}s", f"parallel={parallel}", f"verify={verify}", f"clean={clean}"]
        if threads_list:
            config_items.append(f"threads={threads}")
        if cflags:
            config_items.append(f"cflags={cflags}")
        if collect_counters:
            config_items.append("counters=on")
        console.print(f"Config: {', '.join(config_items)}")
        console.print(f"Benchmarks: {len(bench_list)}\n")

    # Run benchmarks
    start_time = time.time()

    if threads_list and len(bench_list) == 1:
        # Thread sweep mode for single benchmark
        results = runner.run_with_thread_sweep(
            bench_list[0],
            size,
            threads_list,
            arts_config,
            cflags or "",
            counter_dir,
            timeout,
            omp_threads,
            launcher,
            node_count,
        )
    else:
        # Standard mode
        results = runner.run_all(
            bench_list,
            size=size,
            timeout=timeout,
            parallel=parallel,
            verify=verify,
            arts_config=arts_config,
        )

    total_duration = time.time() - start_time

    # Display results
    if not quiet:
        # Table was already shown via Live display, just show the summary panel
        console.print()
        console.print(create_summary_panel(results, total_duration))

    # Export if requested
    if output:
        export_json(results, output, size, total_duration, threads_list, cflags)
        console.print(f"\n[dim]Results exported to: {output}[/]")

    # Exit with error if any failures
    failed = sum(1 for r in results if r.run_arts.status in (Status.FAIL, Status.CRASH))
    if failed > 0:
        raise typer.Exit(1)


@app.command()
def build(
    benchmarks: Optional[List[str]] = typer.Argument(None, help="Specific benchmarks to build"),
    size: str = typer.Option(DEFAULT_SIZE, "--size", "-s", help="Dataset size: small, medium, large"),
    openmp: bool = typer.Option(False, "--openmp", help="Build OpenMP version only"),
    arts: bool = typer.Option(False, "--arts", help="Build ARTS version only"),
    suite: Optional[str] = typer.Option(None, "--suite", help="Filter by suite"),
    arts_config: Optional[Path] = typer.Option(None, "--arts-config", help="Custom arts.cfg file"),
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

    console.print(f"\n[bold]Building {len(bench_list)} benchmarks[/] (size={size})\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Building...", total=len(bench_list) * len(variants))

        for bench in bench_list:
            for variant in variants:
                progress.update(task, description=f"[cyan]{bench}[/] ({variant})")
                result = runner.build_benchmark(bench, size, variant, arts_config)
                status = status_symbol(result.status)
                if result.status != Status.PASS:
                    console.print(f"  {status} {bench} ({variant}): {result.status.value}")
                progress.advance(task)

    console.print("\n[bold green]Build complete![/]")


@app.command()
def clean(
    benchmarks: Optional[List[str]] = typer.Argument(None, help="Specific benchmarks to clean"),
    all: bool = typer.Option(False, "--all", "-a", help="Clean all benchmarks"),
    suite: Optional[str] = typer.Option(None, "--suite", help="Filter by suite"),
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
    input: Optional[Path] = typer.Option(None, "--input", "-i", help="Input JSON file"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, csv"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
):
    """View or export benchmark results."""
    if not input:
        # Look for most recent results file
        results_files = sorted(Path(".").glob("benchmark_results_*.json"), reverse=True)
        if results_files:
            input = results_files[0]
        else:
            console.print("[yellow]No results file found. Run benchmarks first.[/]")
            raise typer.Exit(1)

    with open(input) as f:
        data = json.load(f)

    if format == "json":
        if output:
            with open(output, "w") as f:
                json.dump(data, f, indent=2)
        else:
            console.print_json(data=data)
    elif format == "csv":
        import csv
        rows = []
        for r in data["results"]:
            rows.append({
                "name": r["name"],
                "suite": r["suite"],
                "size": r["size"],
                "build_arts": r["build_arts"]["status"],
                "build_omp": r["build_omp"]["status"],
                "run_arts": r["run_arts"]["status"],
                "run_omp": r["run_omp"]["status"],
                "arts_time": r["timing"]["arts_time_sec"],
                "omp_time": r["timing"]["omp_time_sec"],
                "speedup": r["timing"]["speedup"],
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




@app.command()
def analyze(
    input: Optional[Path] = typer.Option(None, "--input", "-i", help="Input JSON file with benchmark results"),
    benchmark: Optional[str] = typer.Option(None, "--benchmark", "-b", help="Specific benchmark to analyze"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed per-worker timings"),
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
        results_files = sorted(Path(".").glob("benchmark_results_*.json"), reverse=True)
        if results_files:
            input = results_files[0]
        else:
            console.print("[yellow]No results file found. Run benchmarks first with --output.[/]")
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
                    console.print(f"  task.{task_name}: mean={mean:.6f}s (n={len(times)} workers)")
                    if verbose:
                        for t in timings:
                            console.print(f"    worker {t['worker_id']}: {t['time_sec']:.6f}s")

            for par_name, timings in arts_pt.get("parallel_timings", {}).items():
                times = [t["time_sec"] for t in timings]
                if times:
                    mean = sum(times) / len(times)
                    console.print(f"  parallel.{par_name}: mean={mean:.6f}s (n={len(times)} workers)")

        # Analyze OpenMP timings
        if omp_pt:
            console.print("\n[bold]OpenMP Timings:[/]")
            for task_name, timings in omp_pt.get("task_timings", {}).items():
                times = [t["time_sec"] for t in timings]
                if times:
                    mean = sum(times) / len(times)
                    console.print(f"  task.{task_name}: mean={mean:.6f}s (n={len(times)} workers)")
                    if verbose:
                        for t in timings:
                            console.print(f"    worker {t['worker_id']}: {t['time_sec']:.6f}s")

            for par_name, timings in omp_pt.get("parallel_timings", {}).items():
                times = [t["time_sec"] for t in timings]
                if times:
                    mean = sum(times) / len(times)
                    console.print(f"  parallel.{par_name}: mean={mean:.6f}s (n={len(times)} workers)")

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
                        console.print(f"    η = {eta:.3f} (CARTS={arts_mean:.6f}s / OMP={omp_mean:.6f}s)")
                        console.print(f"    {interpretation}")

            # Compute overhead if we have matching parallel/task names
            arts_parallel = arts_pt.get("parallel_timings", {})
            omp_parallel = omp_pt.get("parallel_timings", {})

            # Try to find matching parallel and task names for overhead calculation
            for par_name in set(arts_parallel.keys()) & set(omp_parallel.keys()):
                # Find a task that might correspond (e.g., "gemm" -> "gemm:kernel")
                matching_tasks = [t for t in arts_tasks.keys() if t.startswith(par_name)]
                if matching_tasks:
                    task_name = matching_tasks[0]
                    if task_name in arts_tasks and task_name in omp_tasks:
                        arts_par_times = [t["time_sec"] for t in arts_parallel[par_name]]
                        arts_task_times = [t["time_sec"] for t in arts_tasks[task_name]]
                        omp_par_times = [t["time_sec"] for t in omp_parallel[par_name]]
                        omp_task_times = [t["time_sec"] for t in omp_tasks[task_name]]

                        if arts_par_times and arts_task_times and omp_par_times and omp_task_times:
                            arts_overhead = sum(arts_par_times) / len(arts_par_times) - sum(arts_task_times) / len(arts_task_times)
                            omp_overhead = sum(omp_par_times) / len(omp_par_times) - sum(omp_task_times) / len(omp_task_times)

                            console.print(f"\n  Overhead Analysis (parallel.{par_name} - task.{task_name}):")
                            console.print(f"    CARTS overhead: {arts_overhead:.6f}s")
                            console.print(f"    OMP overhead:   {omp_overhead:.6f}s")

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
            console.print("[green]✓ Delayed optimizations have minimal impact.[/]")
            console.print("  LLVM -O3 successfully recovers most optimizations.")
        elif geomean_eta <= 1.2:
            console.print("[yellow]⚠ Minor performance impact from delayed optimizations.[/]")
            console.print("  Some optimization opportunities are lost at MLIR level.")
        else:
            console.print("[red]✗ Significant performance degradation detected.[/]")
            console.print("  Delayed MLIR optimizations hurt sequential kernel performance.")
            console.print("  Consider investigating optimization ordering in CARTS pipeline.")
    else:
        console.print("\n[yellow]No parallel/task timing data found in results.[/]")
        console.print("Make sure benchmarks use CARTS_PARALLEL_TIMER_* and CARTS_TASK_TIMER_* macros.")


if __name__ == "__main__":
    app()
