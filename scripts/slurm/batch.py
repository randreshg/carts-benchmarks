#!/usr/bin/env python3
"""
SLURM Batch Job Submission for CARTS Benchmarks

This module provides functionality to submit all benchmark jobs (benchmarks x runs)
to SLURM queue at once, leveraging SLURM's job scheduler for parallel execution.

Key Features:
- Submit ~240 jobs (24 benchmarks x 10 runs) in one command
- Exclusive node allocation for resource isolation
- Node-based allocation (not thread-based)
- Counter directory isolation to prevent data collision
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from carts_styles import (
    console as _shared_console,
    Colors, Symbols,
    print_header, print_footer, print_step, print_success, print_error,
    print_warning, print_info,
)

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from benchmark_common import PERF_CACHE_EVENTS


# ============================================================================
# Data Classes
# ============================================================================

TERMINAL_JOB_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
}


@dataclass
class SlurmJobConfig:
    """Configuration for a single SLURM job."""
    benchmark_name: str
    run_number: int
    node_count: int
    time_limit: str  # "HH:MM:SS" format
    partition: Optional[str]
    account: Optional[str]
    executable_arts: Path
    executable_omp: Optional[Path]
    arts_config_path: Path
    run_dir: Path
    size: str
    threads: int  # For OpenMP comparison (single-node only)
    port: Optional[str] = None  # Per-job port override (e.g., "10001" or "[10001-10002]")
    gdb: bool = False  # Wrap executable with gdb for backtrace on crash
    perf: bool = False  # Enable perf stat profiling for cache metrics
    perf_interval: float = 0.1  # Perf sampling interval in seconds
    exclude_nodes: Optional[str] = None  # SLURM nodes to exclude (e.g. "j006,j007")
    job_label: Optional[str] = None  # Optional phase/step label to disambiguate job names


@dataclass
class SlurmJobStatus:
    """Status of a submitted SLURM job."""
    job_id: str
    benchmark_name: str
    run_number: int
    node_count: int  # Node count for directory path
    state: str  # PENDING, RUNNING, COMPLETED, FAILED, TIMEOUT, CANCELLED
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


# ============================================================================
# Sbatch Script Generation
# ============================================================================

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={node_count}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={threads}
#SBATCH --exclusive
#SBATCH --time={time_limit}
{partition_line}
{account_line}
{exclude_line}
#SBATCH --output={run_dir}/slurm.out
#SBATCH --error={run_dir}/slurm.err

# CARTS Benchmark SLURM Job
# Benchmark: {benchmark_name}
# Run: {run_number}
# Generated: {timestamp}

set -uo pipefail

# Create per-run directory and subdirectories
RUN_DIR="{run_dir}"
mkdir -p "$RUN_DIR"
COUNTER_DIR="$RUN_DIR/counters"
mkdir -p "$COUNTER_DIR"

{perf_dir_section}

# Generate per-run arts.cfg with correct counterFolder
# (base arts.cfg has placeholder, we override counterFolder for this run)
sed -e "s|^counterFolder=.*|counterFolder=$COUNTER_DIR|" "{arts_config_path}" > "{runtime_arts_cfg}"
export artsConfig="{runtime_arts_cfg}"
export counterFolder="$COUNTER_DIR"
export CARTS_BENCHMARKS_REPORT_INIT=1

ARTS_EXIT=125
ARTS_DURATION=0
OMP_EXIT=-1
OMP_DURATION=0
RESULT_GENERATOR_EXIT=0

echo "=========================================="
echo "CARTS Benchmark: {benchmark_name}"
echo "Run: {run_number}"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Node Count: $SLURM_NNODES"
echo "Counter Dir: $COUNTER_DIR"
echo "Start: $(date -Iseconds)"
echo "=========================================="

# Run ARTS benchmark
echo ""
echo "[ARTS] Running benchmark..."
ARTS_START=$(date +%s)
{srun_command}
ARTS_EXIT=$?
ARTS_END=$(date +%s)
ARTS_DURATION=$((ARTS_END - ARTS_START))
echo "[ARTS] Exit code: $ARTS_EXIT"
echo "[ARTS] Duration: $ARTS_DURATION seconds"

# Run OpenMP benchmark (single-node only - skip for multi-node)
OMP_EXIT=-1
OMP_DURATION=0
{omp_section}

echo ""
echo "=========================================="
echo "End: $(date -Iseconds)"
echo "=========================================="

# Generate result JSON
python3 "{slurm_job_result_script}" \\
    --benchmark "{benchmark_name}" \\
    --run-number {run_number} \\
    --size "{size}" \\
    --arts-exit $ARTS_EXIT \\
    --arts-duration $ARTS_DURATION \\
    --omp-exit $OMP_EXIT \\
    --omp-duration $OMP_DURATION \\
    --counter-dir "$COUNTER_DIR" \\
    --slurm-job-id "$SLURM_JOB_ID" \\
    --slurm-nodelist "$SLURM_JOB_NODELIST" \\
    --output "{result_json}"
RESULT_GENERATOR_EXIT=$?
if [ $RESULT_GENERATOR_EXIT -ne 0 ]; then
    echo "Warning: job_result.py failed with exit code $RESULT_GENERATOR_EXIT"
fi

exit $ARTS_EXIT
"""

OMP_SECTION_TEMPLATE = """
if [ {node_count} -eq 1 ] && [ -x "{executable_omp}" ]; then
    echo ""
    echo "[OpenMP] Running benchmark..."
    export OMP_NUM_THREADS={threads}
    export OMP_WAIT_POLICY=ACTIVE
    OMP_START=$(date +%s)
    {omp_run_command}
    OMP_EXIT=$?
    OMP_END=$(date +%s)
    OMP_DURATION=$((OMP_END - OMP_START))
    echo "[OpenMP] Exit code: $OMP_EXIT"
    echo "[OpenMP] Duration: $OMP_DURATION seconds"
else
    echo "[OpenMP] Skipped (multi-node or executable not found)"
fi
"""


def _sanitize_job_token(value: str) -> str:
    """Sanitize free-form text for SLURM job names."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def generate_sbatch_script(
    config: SlurmJobConfig,
    script_path: Path,
    slurm_job_result_script: Path,
) -> None:
    """Generate an sbatch script for a single benchmark run.

    Directory structure:
        build/{benchmark}/nodes_{N}/              <- shared, reusable
        ├── arts.cfg                              <- compile-time config
        ├── {name}_arts, {name}_omp               <- executables

        results/{timestamp}/{step}/{benchmark}/{T}t_{N}n/      <- experiment-specific
        └── run_1/                                <- per-run directory
            ├── result.json
            ├── arts.cfg                          <- runtime config
            ├── counters/
            ├── perf/                             <- if --perf
            ├── slurm.out
            └── slurm.err

    Args:
        config: Job configuration
        script_path: Path to write the sbatch script
        slurm_job_result_script: Path to the slurm/job_result.py script
    """
    # Build partition and account lines (only if specified)
    partition_line = f"#SBATCH --partition={config.partition}" if config.partition else ""
    account_line = f"#SBATCH --account={config.account}" if config.account else ""
    exclude_line = f"#SBATCH --exclude={config.exclude_nodes}" if config.exclude_nodes else ""

    # CRITICAL: Use absolute paths - jobs may run from different working directories
    run_dir = config.run_dir.resolve()
    counter_dir = run_dir / "counters"
    result_json = run_dir / "result.json"
    runtime_arts_cfg = run_dir / "arts.cfg"
    perf_dir = run_dir / "perf" if config.perf else None

    arts_config_abs = config.arts_config_path.resolve() if config.arts_config_path else None
    executable_arts_abs = config.executable_arts.resolve() if config.executable_arts else None
    executable_omp_abs = config.executable_omp.resolve() if config.executable_omp else None
    slurm_job_result_abs = slurm_job_result_script.resolve()

    # Perf directory section for sbatch template
    if config.perf and perf_dir:
        perf_dir_section = (
            f'# Create per-run perf directory\n'
            f'PERF_DIR="$RUN_DIR/perf"\n'
            f'mkdir -p "$PERF_DIR"'
        )
    else:
        perf_dir_section = '# Perf profiling disabled'

    # Build OpenMP section (only for single-node)
    if config.node_count == 1 and executable_omp_abs:
        if config.perf and perf_dir:
            events = ",".join(PERF_CACHE_EVENTS)
            interval_ms = int(config.perf_interval * 1000)
            omp_run_command = (
                f"perf stat -e {events} -I {interval_ms} -x , "
                f"-o {run_dir}/perf/omp.csv "
                f"-- {executable_omp_abs}"
            )
        else:
            omp_run_command = str(executable_omp_abs)

        omp_section = OMP_SECTION_TEMPLATE.format(
            node_count=config.node_count,
            executable_omp=executable_omp_abs,
            threads=config.threads,
            omp_run_command=omp_run_command,
        )
    elif config.node_count > 1:
        omp_section = (
            '# OpenMP skipped (multi-node run - verification uses the stored '
            'matching OMP reference when available)'
        )
    else:
        omp_section = '# OpenMP skipped (executable not specified)'

    # Safe, collision-resistant job name (SLURM limits to 64 chars).
    safe_name = _sanitize_job_token(config.benchmark_name.replace("/", "_")) or "bench"
    safe_label = _sanitize_job_token(config.job_label or "")
    job_suffix = f"_n{config.node_count}_r{config.run_number}"
    prefix_parts = [p for p in (safe_label, safe_name) if p]
    job_prefix = "__".join(prefix_parts) if prefix_parts else "job"
    max_prefix_len = max(1, 64 - len(job_suffix))
    job_name = f"{job_prefix[:max_prefix_len]}{job_suffix}"

    # Build srun command: gdb, perf, or plain (mutually exclusive)
    if config.gdb:
        srun_command = (
            f'srun --exclusive --cpus-per-task={config.threads} --kill-on-bad-exit=1 bash -c '
            f"'gdb --batch -ex run -ex \"thread apply all bt\" -ex quit --args {executable_arts_abs}'"
        )
    elif config.perf and perf_dir:
        events = ",".join(PERF_CACHE_EVENTS)
        interval_ms = int(config.perf_interval * 1000)
        # Single quotes: run_dir/perf is baked as absolute path at generation time,
        # ${SLURM_PROCID} is expanded by the inner bash (set per-task by srun)
        srun_command = (
            f"srun --exclusive --cpus-per-task={config.threads} --kill-on-bad-exit=1 bash -c "
            f"'perf stat -e {events} -I {interval_ms} -x , "
            f"-o {run_dir}/perf/arts_node_${{SLURM_PROCID}}.csv "
            f"-- {executable_arts_abs}'"
        )
    else:
        srun_command = (
            f"srun --exclusive --cpus-per-task={config.threads} "
            f"--kill-on-bad-exit=1 {executable_arts_abs}"
        )

    script_content = SBATCH_TEMPLATE.format(
        job_name=job_name,
        node_count=config.node_count,
        time_limit=config.time_limit,
        partition_line=partition_line,
        account_line=account_line,
        exclude_line=exclude_line,
        run_dir=run_dir,
        benchmark_name=config.benchmark_name,
        run_number=config.run_number,
        timestamp=datetime.now().isoformat(),
        arts_config_path=arts_config_abs,
        runtime_arts_cfg=runtime_arts_cfg,
        perf_dir_section=perf_dir_section,
        result_json=result_json,
        executable_arts=executable_arts_abs,
        srun_command=srun_command,
        omp_section=omp_section,
        slurm_job_result_script=slurm_job_result_abs,
        size=config.size,
        threads=config.threads,
    )

    # Create run directory
    # CRITICAL: run_dir must exist before sbatch submission because
    # Slurm opens --output/--error files before executing the script body.
    # Slurm 21.08+ does not auto-create parent directories for output paths.
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write script
    script_path.write_text(script_content)
    script_path.chmod(0o755)


def generate_arts_config_for_node(
    base_config: Path,
    build_node_dir: Path,
    node_count: int,
    threads: int,
) -> Path:
    """Generate a node-specific arts.cfg for compilation (goes in build/ directory).

    This config is used at compile time to embed nodeCount into the executable.
    counterFolder is set to a placeholder - the sbatch script will override it per-run
    when creating the runtime arts.cfg in each run directory.

    Args:
        base_config: Base arts.cfg to use as template
        build_node_dir: Build directory (build/{benchmark}/nodes_{N}/{T}T/)
        node_count: Number of nodes
        threads: Thread count

    Returns:
        Path to the generated config file (build/{benchmark}/nodes_{N}/{T}T/arts.cfg)
    """
    content = base_config.read_text()

    # CRITICAL: Use absolute paths - jobs run from different working directories
    # counterFolder placeholder - sbatch script sets actual path per run
    counter_dir_placeholder = (build_node_dir / "counters").resolve()

    # Update or add counterFolder (placeholder, will be overridden per-run)
    if re.search(r'^counterFolder\s*=', content, re.MULTILINE):
        content = re.sub(
            r'^counterFolder\s*=.*$',
            f'counterFolder={counter_dir_placeholder}',
            content,
            flags=re.MULTILINE
        )
    else:
        content = content.replace('[ARTS]', f'[ARTS]\ncounterFolder={counter_dir_placeholder}')

    # Update or add counterStartPoint
    if re.search(r'^counterStartPoint\s*=', content, re.MULTILINE):
        content = re.sub(
            r'^counterStartPoint\s*=.*$',
            'counterStartPoint=1',
            content,
            flags=re.MULTILINE
        )
    else:
        content = content.replace('[ARTS]', '[ARTS]\ncounterStartPoint=1')

    # Update nodeCount
    if re.search(r'^nodeCount\s*=', content, re.MULTILINE):
        content = re.sub(
            r'^nodeCount\s*=.*$',
            f'nodeCount={node_count}',
            content,
            flags=re.MULTILINE
        )
    else:
        content = content.replace('[ARTS]', f'[ARTS]\nnodeCount={node_count}')

    # Update threads to match this build combination.
    # (SLURM runtime still exports SLURM_CPUS_PER_TASK, but keeping the config
    # aligned with the combination makes artifacts self-describing.)
    if re.search(r'^threads\s*=', content, re.MULTILINE):
        content = re.sub(
            r'^threads\s*=.*$',
            f'threads={threads}',
            content,
            flags=re.MULTILINE
        )
    else:
        content = content.replace('[ARTS]', f'[ARTS]\nthreads={threads}')

    # Ensure launcher is slurm
    if re.search(r'^launcher\s*=', content, re.MULTILINE):
        content = re.sub(
            r'^launcher\s*=.*$',
            'launcher=slurm',
            content,
            flags=re.MULTILINE
        )
    else:
        content = content.replace('[ARTS]', '[ARTS]\nlauncher=slurm')

    # Clear nodes and masterNode - SLURM launcher ignores these
    # (ARTS reads SLURM_NNODES and SLURM_STEP_NODELIST instead)
    if re.search(r'^nodes\s*=', content, re.MULTILINE):
        content = re.sub(
            r'^nodes\s*=.*$',
            '# nodes= (managed by SLURM)',
            content,
            flags=re.MULTILINE
        )

    if re.search(r'^masterNode\s*=', content, re.MULTILINE):
        content = re.sub(
            r'^masterNode\s*=.*$',
            '# masterNode= (managed by SLURM)',
            content,
            flags=re.MULTILINE
        )

    # Write to build directory (use absolute path)
    config_path = (build_node_dir / "arts.cfg").resolve()
    build_node_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(content)

    return config_path


# ============================================================================
# Job Submission
# ============================================================================


def submit_job(script_path: Path) -> str:
    """Submit an sbatch script and return the job ID.

    Args:
        script_path: Path to the sbatch script

    Returns:
        SLURM job ID

    Raises:
        subprocess.CalledProcessError: If submission fails
    """
    result = subprocess.run(
        ["sbatch", "--parsable", str(script_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    # sbatch --parsable returns: job_id or job_id;cluster_name
    job_id = result.stdout.strip().split(";")[0]
    return job_id


def submit_all_jobs(
    job_configs: List[Tuple[SlurmJobConfig, Path]],
    console: Console,
    dry_run: bool = False,
) -> Dict[str, SlurmJobStatus]:
    """Submit all sbatch scripts and return job statuses.

    Args:
        job_configs: List of (config, script_path) tuples
        console: Rich console for output
        dry_run: If True, don't actually submit (just validate)

    Returns:
        Dict mapping job_id -> SlurmJobStatus
    """
    job_statuses: Dict[str, SlurmJobStatus] = {}

    if dry_run:
        console.print(f"[{Colors.WARNING}]Dry run mode - scripts generated but not submitted[/{Colors.WARNING}]")
        for config, script_path in job_configs:
            # Use fake job ID for dry run
            fake_id = f"DRY_{config.benchmark_name}_{config.run_number}"
            job_statuses[fake_id] = SlurmJobStatus(
                job_id=fake_id,
                benchmark_name=config.benchmark_name,
                run_number=config.run_number,
                node_count=config.node_count,
                state="DRY_RUN",
                run_dir=config.run_dir,
            )
        return job_statuses

    print_step(f"Submitting {len(job_configs)} jobs to SLURM...")

    submitted = 0
    failed = 0

    for config, script_path in job_configs:
        try:
            job_id = submit_job(script_path)
            job_statuses[job_id] = SlurmJobStatus(
                job_id=job_id,
                benchmark_name=config.benchmark_name,
                run_number=config.run_number,
                node_count=config.node_count,
                state="PENDING",
                run_dir=config.run_dir,
            )
            submitted += 1
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to submit {config.benchmark_name} run {config.run_number}: {e.stderr}")
            failed += 1

    print_success(f"Submitted {submitted} jobs")
    if failed > 0:
        print_error(f"{failed} submissions failed")

    return job_statuses


# ============================================================================
# Job Monitoring
# ============================================================================


def poll_jobs(job_ids: List[str]) -> Dict[str, str]:
    """Query squeue for current job states.

    Args:
        job_ids: List of SLURM job IDs

    Returns:
        Dict mapping job_id -> state (PENDING, RUNNING, COMPLETED, etc.)
    """
    if not job_ids:
        return {}

    try:
        result = subprocess.run(
            [
                "squeue",
                "--jobs=" + ",".join(job_ids),
                "--format=%i|%T",
                "--noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return {}

        states = {}
        for line in result.stdout.strip().split("\n"):
            if "|" in line:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    job_id, state = parts[0], parts[1]
                    states[job_id] = state

        # Jobs not in squeue are finished — use scontrol to get real state
        for job_id in job_ids:
            if job_id not in states:
                states[job_id] = _get_scontrol_status(job_id).state

        return states

    except subprocess.TimeoutExpired:
        # Keep the previous in-memory state on transient scheduler query failures.
        return {}


def _query_sacct_statuses(job_ids: List[str]) -> Dict[str, SlurmJobStatus]:
    """Query sacct for final job states, returning any rows it can resolve."""
    statuses: Dict[str, SlurmJobStatus] = {}
    if not job_ids:
        return statuses

    result = subprocess.run(
        [
            "sacct",
            "--jobs=" + ",".join(job_ids),
            "--format=JobID,JobName,State,ExitCode,Elapsed,NodeList,Start,End",
            "--parsable2",
            "--noheader",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return statuses

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 8:
            continue

        job_id = parts[0]
        if "." in job_id:
            continue

        exit_parts = parts[3].split(":")
        exit_code = int(exit_parts[0]) if exit_parts[0].isdigit() else None

        job_name = parts[1]
        node_count = 1
        run_number = 0
        match = re.match(r"^(.+)_n(\d+)_r(\d+)$", job_name)
        if match:
            job_name = match.group(1)
            node_count = int(match.group(2))
            run_number = int(match.group(3))

        statuses[job_id] = SlurmJobStatus(
            job_id=job_id,
            benchmark_name=job_name,
            run_number=run_number,
            node_count=node_count,
            state=parts[2],
            exit_code=exit_code,
            elapsed=parts[4],
            node_list=parts[5],
            start_time=parts[6],
            end_time=parts[7],
        )

    return statuses


def _empty_slurm_status(job_id: str, state: str = "UNKNOWN") -> SlurmJobStatus:
    """Create a placeholder SLURM status when accounting data is unavailable."""
    return SlurmJobStatus(
        job_id=job_id,
        benchmark_name="",
        run_number=0,
        node_count=1,
        state=state,
    )


def _get_scontrol_status(job_id: str) -> SlurmJobStatus:
    """Get the best-effort job status from scontrol."""
    state = "UNKNOWN"
    exit_code = None
    start_time = None
    end_time = None
    elapsed = None
    node_list = None

    try:
        result = subprocess.run(
            ["scontrol", "show", "job", job_id],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return _empty_slurm_status(job_id, state=state)
        for token in result.stdout.split():
            if token.startswith("JobState="):
                state = token.split("=", 1)[1]
            elif token.startswith("ExitCode="):
                exit_parts = token.split("=", 1)[1].split(":")
                exit_code = int(exit_parts[0]) if exit_parts[0].isdigit() else None
            elif token.startswith("StartTime="):
                start_time = token.split("=", 1)[1]
            elif token.startswith("EndTime="):
                end_time = token.split("=", 1)[1]
            elif token.startswith("RunTime="):
                elapsed = token.split("=", 1)[1]
            elif token.startswith("NodeList="):
                node_list = token.split("=", 1)[1]
    except subprocess.TimeoutExpired:
        pass

    status = _empty_slurm_status(job_id, state=state)
    status.exit_code = exit_code
    status.start_time = start_time
    status.end_time = end_time
    status.elapsed = elapsed
    status.node_list = node_list
    return status


def _has_completed_run_artifact(status: SlurmJobStatus) -> bool:
    """Return True when the job produced its per-run result artifact."""
    return bool(status.run_dir and (status.run_dir / "result.json").exists())


def get_final_job_status(job_ids: List[str]) -> Dict[str, SlurmJobStatus]:
    """Get final job status from sacct after completion.

    Args:
        job_ids: List of SLURM job IDs

    Returns:
        Dict mapping job_id -> SlurmJobStatus with final status
    """
    if not job_ids:
        return {}

    statuses: Dict[str, SlurmJobStatus] = {}

    for attempt in range(3):
        remaining = [job_id for job_id in job_ids if job_id not in statuses]
        if not remaining:
            break
        try:
            statuses.update(_query_sacct_statuses(remaining))
        except subprocess.TimeoutExpired:
            pass
        if len(statuses) == len(job_ids):
            break
        if attempt < 2:
            time.sleep(2)

    # Fallback: use scontrol for any jobs sacct didn't cover
    for job_id in job_ids:
        if job_id not in statuses:
            statuses[job_id] = _get_scontrol_status(job_id)

    return statuses


def wait_for_jobs_completion(
    job_statuses: Dict[str, SlurmJobStatus],
    console: Console,
    poll_interval: int = 10,
) -> Dict[str, SlurmJobStatus]:
    """Wait for all jobs to complete, showing live progress.

    Args:
        job_statuses: Dict of job_id -> SlurmJobStatus (from submit)
        console: Rich console for output
        poll_interval: Seconds between squeue polls

    Returns:
        Updated job_statuses with final states
    """
    job_ids = list(job_statuses.keys())

    if not job_ids:
        return job_statuses

    def is_monitor_terminal(status: SlurmJobStatus) -> bool:
        if status.state in TERMINAL_JOB_STATES:
            return True
        # Some SLURM installations briefly lose final accounting before sacct
        # catches up. A completed per-run result artifact is sufficient to stop
        # polling without treating UNKNOWN itself as terminal.
        return status.state == "UNKNOWN" and _has_completed_run_artifact(status)

    def create_status_table() -> Table:
        """Create a status table for live display."""
        table = Table(title="SLURM Job Status", box=None)
        table.add_column("State", style="bold")
        table.add_column("Count", justify="right")

        # Count states
        state_counts: Dict[str, int] = {}
        for status in job_statuses.values():
            state = status.state
            state_counts[state] = state_counts.get(state, 0) + 1

        # Add rows with colors
        state_colors = {
            "PENDING": Colors.SKIP,
            "RUNNING": Colors.RUNNING,
            "COMPLETED": Colors.PASS,
            "FAILED": Colors.FAIL,
            "TIMEOUT": Colors.FAIL,
            "CANCELLED": Colors.SKIP,
            "UNKNOWN": Colors.DIM,
        }

        for state, count in sorted(state_counts.items()):
            color = state_colors.get(state, Colors.DIM)
            table.add_row(f"[{color}]{state}[/{color}]", str(count))

        return table

    console.print(f"\n[bold]Monitoring {len(job_ids)} jobs (poll every {poll_interval}s)...[/]")
    console.print("[dim]Press Ctrl+C to stop monitoring (jobs will continue running)[/]\n")

    try:
        with Live(create_status_table(), console=console, refresh_per_second=1, transient=True) as live:
            while True:
                # Poll current states
                current_states = poll_jobs(job_ids)

                # Update statuses
                for job_id, state in current_states.items():
                    if job_id in job_statuses:
                        job_statuses[job_id].state = state

                # Update display
                live.update(create_status_table())

                # Check if all jobs are done
                all_done = all(
                    is_monitor_terminal(status)
                    for status in job_statuses.values()
                )

                if all_done:
                    break

                time.sleep(poll_interval)

    except KeyboardInterrupt:
        print_warning("Monitoring stopped. Jobs will continue running.")
        print_info("Use 'squeue -u $USER' to check status or 'scancel' to cancel jobs.")

    # Get final status from sacct
    final_statuses = get_final_job_status(job_ids)
    for job_id, final_status in final_statuses.items():
        if job_id in job_statuses:
            # Preserve benchmark_name and run_number from original
            final_status.benchmark_name = job_statuses[job_id].benchmark_name
            final_status.run_number = job_statuses[job_id].run_number
            final_status.node_count = job_statuses[job_id].node_count
            final_status.run_dir = job_statuses[job_id].run_dir
            job_statuses[job_id] = final_status

    return job_statuses


# ============================================================================
# Results Collection
# ============================================================================


def collect_results(
    job_statuses: Dict[str, SlurmJobStatus],
    experiment_dir: Path,
) -> List[Dict[str, Any]]:
    """Collect results from completed SLURM jobs.

    Directory structure (experiment-specific):
        results/{timestamp}/{step}/{benchmark}/{T}t_{N}n/
        └── run_{N}/
            ├── result.json
            ├── counters/
            ├── perf/
            ├── slurm.out
            └── slurm.err

    Args:
        job_statuses: Final job statuses
        experiment_dir: Experiment directory (kept for API compatibility)

    Returns:
        List of result dictionaries (one per job, including failures)
    """
    results = []
    snapshot_cache: Dict[str, Dict[str, Any]] = {}

    def _parse_key_value_tokens(text: str) -> Dict[str, str]:
        parsed: Dict[str, str] = {}
        for token in text.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            parsed[key] = value
        return parsed

    def _run_snapshot_cmd(command: List[str], timeout: int) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {
            "command": " ".join(command),
            "ok": False,
        }
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            snapshot["return_code"] = proc.returncode
            snapshot["stdout"] = (proc.stdout or "").strip()
            snapshot["stderr"] = (proc.stderr or "").strip()
            snapshot["ok"] = proc.returncode == 0
        except FileNotFoundError:
            snapshot["error"] = f"{command[0]} not found"
        except subprocess.TimeoutExpired:
            snapshot["error"] = f"{command[0]} timed out after {timeout}s"
        except Exception as exc:
            snapshot["error"] = str(exc)
        return snapshot

    def _collect_slurm_snapshot(job_id: str) -> Dict[str, Any]:
        if job_id in snapshot_cache:
            return snapshot_cache[job_id]

        snapshot: Dict[str, Any] = {
            "captured_at": datetime.now().isoformat(),
            "job_id": job_id,
        }

        sacct_cmd = [
            "sacct",
            f"--jobs={job_id}",
            "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList,Start,End,Reason",
            "--parsable2",
            "--noheader",
        ]
        sacct = _run_snapshot_cmd(sacct_cmd, timeout=20)
        if sacct.get("ok") and sacct.get("stdout"):
            lines = [line for line in str(sacct["stdout"]).splitlines() if line.strip()]
            primary = None
            for line in lines:
                parts = line.split("|")
                if parts and parts[0] == job_id:
                    primary = line
                    break
            if primary is None:
                primary = lines[0] if lines else ""
            if primary:
                parts = primary.split("|")
                if len(parts) >= 8:
                    exit_parts = parts[2].split(":", 1)
                    sacct["parsed"] = {
                        "job_id_raw": parts[0],
                        "state": parts[1],
                        "exit_code": parts[2],
                        "elapsed": parts[3],
                        "nodelist": parts[4],
                        "start": parts[5],
                        "end": parts[6],
                        "reason": parts[7],
                        "exit_status": int(exit_parts[0]) if exit_parts[0].isdigit() else None,
                        "exit_signal": int(exit_parts[1]) if len(exit_parts) > 1 and exit_parts[1].isdigit() else None,
                    }
            sacct["lines"] = lines[-20:]
        snapshot["sacct"] = sacct

        scontrol_cmd = ["scontrol", "show", "job", job_id]
        scontrol = _run_snapshot_cmd(scontrol_cmd, timeout=20)
        if scontrol.get("ok") and scontrol.get("stdout"):
            parsed = _parse_key_value_tokens(str(scontrol["stdout"]))
            keys = [
                "JobId",
                "JobName",
                "JobState",
                "Reason",
                "ExitCode",
                "RunTime",
                "SubmitTime",
                "StartTime",
                "EndTime",
                "NodeList",
                "BatchHost",
                "NumNodes",
                "NumCPUs",
                "NumTasks",
                "CPUs/Task",
            ]
            scontrol["parsed"] = {k: parsed.get(k) for k in keys if k in parsed}
            scontrol["stdout_tail"] = str(scontrol["stdout"]).splitlines()[-60:]
        snapshot["scontrol"] = scontrol

        snapshot_cache[job_id] = snapshot
        return snapshot

    def _load_run_config(run_dir: Path) -> Dict[str, Any]:
        run_config_file = run_dir / "run_config.json"
        if not run_config_file.exists():
            return {}
        try:
            payload = json.loads(run_config_file.read_text())
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _apply_run_config(result: Dict[str, Any], run_config: Dict[str, Any]) -> None:
        if not run_config:
            return
        result["threads"] = run_config.get("threads")
        result["nodes"] = run_config.get("nodes")
        result["run_phase"] = run_config.get("run_phase")
        if "size" in run_config:
            result["size"] = run_config.get("size")
        if "compile_args" in run_config:
            result["compile_args"] = run_config.get("compile_args")
        if "cflags" in run_config:
            result["cflags"] = run_config.get("cflags")
        if "config" in run_config and isinstance(run_config["config"], dict):
            result["config"] = run_config["config"]
        if "reference" in run_config and isinstance(run_config["reference"], dict):
            verification = result.setdefault("verification", {})
            reference = run_config["reference"]
            if "checksum" in reference:
                verification.setdefault("reference_checksum", reference.get("checksum"))
            if "source" in reference:
                verification.setdefault("reference_source", reference.get("source"))

    def _apply_compile_artifact_paths(result: Dict[str, Any], run_config: Dict[str, Any]) -> None:
        """Attach compile-time artifact locations from run_config when available."""
        if not run_config:
            return

        arts_cfg_source = run_config.get("arts_cfg_source")
        if not arts_cfg_source:
            return

        try:
            arts_cfg_path = Path(str(arts_cfg_source)).resolve()
        except Exception:
            return

        artifacts = result.setdefault("artifacts", {})
        artifacts.setdefault("arts_config", str(arts_cfg_path))
        artifacts.setdefault("build_dir", str(arts_cfg_path.parent))

        benchmark_name = str(run_config.get("benchmark") or "")
        example_name = benchmark_name.split("/")[-1] if benchmark_name else ""
        if example_name:
            arts_exe = arts_cfg_path.parent / f"{example_name}_arts"
            omp_exe = arts_cfg_path.parent / f"{example_name}_omp"
            if arts_exe.exists():
                artifacts.setdefault("executable_arts", str(arts_exe))
            if omp_exe.exists():
                artifacts.setdefault("executable_omp", str(omp_exe))

    def _summarize_log(log_path: Path, tail_lines: int = 40) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "path": str(log_path),
            "exists": log_path.exists(),
        }
        if not log_path.exists():
            return summary
        try:
            text = log_path.read_text(errors="replace")
        except Exception as exc:
            summary["read_error"] = str(exc)
            return summary

        lines = text.splitlines()
        summary["line_count"] = len(lines)
        summary["tail"] = lines[-tail_lines:]
        if log_path.name.endswith(".err"):
            summary["broken_pipe_count"] = len(re.findall(r"Broken pipe", text))
            summary["srun_error_count"] = len(re.findall(r"^srun: error:", text, flags=re.MULTILINE))
            summary["counter_timeout_warnings"] = len(
                re.findall(r"Could not read counter file", text)
            )
            summary["remote_send_hard_timeout_count"] = len(
                re.findall(r"Remote send hard-timeout", text)
            )
            summary["connection_refused_count"] = len(
                re.findall(r"Connection refused", text)
            )
        return summary

    def _slurm_err_summary(diagnostics: Any) -> Dict[str, Any]:
        if not isinstance(diagnostics, dict):
            return {}
        slurm_stderr = diagnostics.get("slurm_stderr")
        if isinstance(slurm_stderr, dict):
            return slurm_stderr
        slurm_err = diagnostics.get("slurm_err")
        if isinstance(slurm_err, dict):
            return slurm_err
        return {}

    def _runtime_warning_reasons(diagnostics: Any) -> List[str]:
        slurm_err = _slurm_err_summary(diagnostics)
        reasons: List[str] = []
        if not slurm_err:
            return reasons

        srun_errors = int(slurm_err.get("srun_error_count") or 0)
        broken_pipes = int(slurm_err.get("broken_pipe_count") or 0)
        counter_timeouts = int(slurm_err.get("counter_timeout_warnings") or 0)
        remote_send_timeouts = int(slurm_err.get("remote_send_hard_timeout_count") or 0)
        connection_refused = int(slurm_err.get("connection_refused_count") or 0)

        if srun_errors > 0:
            reasons.append(f"srun_error_count={srun_errors}")
        if broken_pipes > 0:
            reasons.append(f"broken_pipe_count={broken_pipes}")
        if counter_timeouts > 0:
            reasons.append(f"counter_timeout_warnings={counter_timeouts}")
        if remote_send_timeouts > 0:
            reasons.append(f"remote_send_hard_timeout_count={remote_send_timeouts}")
        if connection_refused > 0:
            reasons.append(f"connection_refused_count={connection_refused}")
        return reasons

    def _build_failure_result(
        status: SlurmJobStatus,
        error: str,
        run_dir: Path,
        run_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        snapshot = _collect_slurm_snapshot(status.job_id)
        parsed = snapshot.get("scontrol", {}).get("parsed", {})
        snapshot_nodelist = parsed.get("NodeList")
        resolved_nodelist = status.node_list or snapshot_nodelist
        result_status = "TIMEOUT" if status.state == "TIMEOUT" else "FAIL"
        failure_error = error
        if status.state == "TIMEOUT" and error.startswith("No result.json found"):
            failure_error = (
                f"SLURM job timed out before result.json was written: {run_dir} "
                "(increase --time-limit or reduce workload)."
            )
        failure: Dict[str, Any] = {
            "benchmark": status.benchmark_name,
            "run_number": status.run_number,
            "status": result_status,
            "slurm": {
                "job_id": status.job_id,
                "state": status.state,
                "exit_code": status.exit_code,
                "elapsed": status.elapsed,
                "nodelist": resolved_nodelist,
            },
            "error": failure_error,
            "_run_dir": str(run_dir),
            "artifacts": {
                "run_dir": str(run_dir),
                "run_config": str(run_dir / "run_config.json"),
                "result_json": str(run_dir / "result.json"),
                "slurm_out": str(run_dir / "slurm.out"),
                "slurm_err": str(run_dir / "slurm.err"),
            },
            "diagnostics": {
                "slurm_out": _summarize_log(run_dir / "slurm.out"),
                "slurm_err": _summarize_log(run_dir / "slurm.err"),
                "slurm_snapshot": snapshot,
            },
        }
        _apply_run_config(failure, run_config)
        _apply_compile_artifact_paths(failure, run_config)
        warning_reasons = _runtime_warning_reasons(failure.get("diagnostics"))
        if warning_reasons:
            failure.setdefault("diagnostics", {})["runtime_warning"] = {
                "has_warning": True,
                "reasons": warning_reasons,
            }
        return failure

    for job_id, status in job_statuses.items():
        if status.state == "DRY_RUN":
            continue

        if not status.run_dir:
            results.append({
                "benchmark": status.benchmark_name,
                "run_number": status.run_number,
                "status": "FAIL",
                "slurm": {
                    "job_id": job_id,
                    "state": status.state,
                    "exit_code": status.exit_code,
                },
                "error": "Missing run_dir in SLURM job status",
                "diagnostics": {
                    "slurm_snapshot": _collect_slurm_snapshot(job_id),
                },
            })
            continue

        result_file = status.run_dir / "result.json"
        run_config = _load_run_config(status.run_dir)

        if result_file.exists():
            try:
                with open(result_file) as f:
                    result = json.load(f)
                # Merge SLURM status info without erasing richer per-run fields
                # from job_result.py (for example, nodelist captured from
                # SLURM_JOB_NODELIST in the batch script environment).
                slurm_info = result.setdefault("slurm", {})
                effective_state = status.state
                if effective_state == "UNKNOWN" and result.get("status") == "PASS":
                    effective_state = "COMPLETED"
                    result.setdefault("diagnostics", {}).setdefault(
                        "slurm_state_inference",
                        {
                            "state": "COMPLETED",
                            "reason": (
                                "Inferred from a successful result.json because "
                                "SLURM accounting did not return a final state."
                            ),
                        },
                    )
                slurm_info["state"] = effective_state
                if status.exit_code is not None:
                    slurm_info["exit_code"] = status.exit_code
                if status.elapsed is not None:
                    slurm_info["elapsed"] = status.elapsed
                if status.node_list:
                    slurm_info["nodelist"] = status.node_list
                result["_run_dir"] = str(status.run_dir)
                _apply_run_config(result, run_config)
                result.setdefault("artifacts", {}).update({
                    "run_dir": str(status.run_dir),
                    "run_config": str(status.run_dir / "run_config.json"),
                    "result_json": str(result_file),
                    "slurm_out": str(status.run_dir / "slurm.out"),
                    "slurm_err": str(status.run_dir / "slurm.err"),
                })
                _apply_compile_artifact_paths(result, run_config)
                warning_reasons = _runtime_warning_reasons(result.get("diagnostics"))
                if warning_reasons:
                    diagnostics = result.setdefault("diagnostics", {})
                    diagnostics["runtime_warning"] = {
                        "has_warning": True,
                        "reasons": warning_reasons,
                    }
                    if str(result.get("status", "")).upper() == "PASS":
                        result["status_detail"] = "WARN"
                if result.get("status") != "PASS":
                    result.setdefault("diagnostics", {}).setdefault(
                        "slurm_snapshot",
                        _collect_slurm_snapshot(status.job_id),
                    )
                elif warning_reasons:
                    result.setdefault("diagnostics", {}).setdefault(
                        "slurm_snapshot",
                        _collect_slurm_snapshot(status.job_id),
                    )
                results.append(result)
            except json.JSONDecodeError:
                results.append(
                    _build_failure_result(
                        status,
                        "Failed to parse result.json",
                        status.run_dir,
                        run_config,
                    )
                )
        else:
            results.append(
                _build_failure_result(
                    status,
                    f"No result.json found in {status.run_dir}",
                    status.run_dir,
                    run_config,
                )
            )

    return results


def write_job_manifest(
    experiment_dir: Path,
    job_statuses: Dict[str, SlurmJobStatus],
    metadata: Dict[str, Any],
) -> Path:
    """Write job manifest JSON with all job information.

    Args:
        experiment_dir: Experiment directory
        job_statuses: All job statuses
        metadata: Additional metadata (timestamp, config, etc.)

    Returns:
        Path to the manifest file
    """
    manifest_metadata = dict(metadata)
    requested_total_jobs = manifest_metadata.get("total_jobs")
    manifest_metadata["total_jobs"] = len(job_statuses)
    if requested_total_jobs is not None and requested_total_jobs != len(job_statuses):
        manifest_metadata["requested_total_jobs"] = requested_total_jobs

    manifest = {
        "metadata": manifest_metadata,
        "jobs": {
            job_id: asdict(status)
            for job_id, status in job_statuses.items()
        },
    }

    manifest_path = experiment_dir / "job_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest_path


def write_aggregated_results(
    experiment_dir: Path,
    results: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Path:
    """Write aggregated results JSON.

    Args:
        experiment_dir: Experiment directory
        results: List of result dictionaries
        metadata: Experiment metadata

    Returns:
        Path to the results file
    """
    output = {
        "metadata": metadata,
        "summary": {
            "total_jobs": len(results),
            "successful": sum(1 for r in results if r.get("status") == "PASS"),
            "failed": sum(1 for r in results if r.get("status") in ("FAIL", "CRASH", "TIMEOUT")),
        },
        "results": results,
    }

    results_path = experiment_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return results_path


def write_slurm_manifest(
    experiment_dir: Path,
    results: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    command: str,
    total_duration: float,
    reproducibility: Dict[str, Any],
) -> Path:
    """Write manifest.json with the same schema as the standard run manifest.

    This gives both run and run --slurm the same entry point for analysis tools.
    """
    passed = sum(1 for r in results if r.get("status") == "PASS")
    failed = sum(1 for r in results if r.get("status") in ("FAIL", "CRASH", "TIMEOUT"))

    manifest = {
        "version": 1,
        "created": datetime.now().isoformat(),
        "command": command,
        "layout": {
            "results_json": "results.json",
            "job_manifest": "job_manifest.json",
        },
        "summary": {
            "total_jobs": len(results),
            "passed": passed,
            "failed": failed,
            "total_duration_sec": round(total_duration, 1),
        },
        "reproducibility": reproducibility,
    }

    manifest_path = experiment_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest_path
