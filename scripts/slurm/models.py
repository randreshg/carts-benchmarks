"""Shared SLURM data models for benchmark submission and result processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


# SLURM job state constants
SLURM_STATE_PENDING = "PENDING"
SLURM_STATE_RUNNING = "RUNNING"
SLURM_STATE_COMPLETED = "COMPLETED"
SLURM_STATE_FAILED = "FAILED"
SLURM_STATE_CANCELLED = "CANCELLED"
SLURM_STATE_TIMEOUT = "TIMEOUT"
SLURM_STATE_NODE_FAIL = "NODE_FAIL"
SLURM_STATE_OUT_OF_MEMORY = "OUT_OF_MEMORY"
SLURM_STATE_UNKNOWN = "UNKNOWN"
SLURM_STATE_DRY_RUN = "DRY_RUN"
SLURM_STATE_SUBMIT_FAILED = "SUBMIT_FAILED"

TERMINAL_JOB_STATES = {
    SLURM_STATE_COMPLETED,
    SLURM_STATE_FAILED,
    SLURM_STATE_CANCELLED,
    SLURM_STATE_TIMEOUT,
    SLURM_STATE_NODE_FAIL,
    SLURM_STATE_OUT_OF_MEMORY,
}


@dataclass
class SlurmJobConfig:
    """Configuration for a single SLURM job."""

    benchmark_name: str
    run_number: int
    node_count: int
    time_limit: str
    partition: Optional[str]
    account: Optional[str]
    executable_arts: Path
    executable_omp: Optional[Path]
    arts_config_path: Path
    python_executable: Path
    run_dir: Path
    size: str
    threads: int
    timeout_seconds: int
    port: Optional[str] = None
    gdb: bool = False
    perf: bool = False
    perf_interval: float = 0.1
    exclude_nodes: Optional[str] = None
    job_label: Optional[str] = None


@dataclass
class SlurmJobStatus:
    """Status of a submitted SLURM job."""

    job_id: str
    benchmark_name: str
    run_number: int
    node_count: int
    state: str
    run_dir: Optional[Path] = None
    exit_code: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    elapsed: Optional[str] = None
    node_list: Optional[str] = None


@dataclass
class SlurmBatchResult:
    """Result of a batch submission experiment."""

    experiment_id: str
    experiment_dir: Path
    job_manifest_path: Path
    jobs_submitted: int
    jobs_completed: int
    jobs_failed: int
    jobs_timeout: int
    total_duration_sec: float
    job_statuses: Dict[str, SlurmJobStatus] = field(default_factory=dict)


@dataclass
class SubmissionFailure:
    """Metadata for an sbatch submission failure."""

    benchmark_name: str
    run_number: int
    node_count: int
    run_dir: Path
    script_path: Path
    error: str
