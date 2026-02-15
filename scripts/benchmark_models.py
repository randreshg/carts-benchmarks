"""
Benchmark data models — enums and dataclasses used across the benchmark runner.

These types are shared by benchmark_runner, benchmark_artifacts, and benchmark_metadata.
They depend only on the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from carts_styles import Status


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
