#!/usr/bin/env python3
"""
SLURM Batch Job Submission for CARTS Benchmarks

This module provides functionality to submit all benchmark jobs (benchmarks x runs)
to SLURM queue, leveraging SLURM's job scheduler for parallel execution.

Two submission modes are supported (controlled via the ``max_concurrent``
parameter of :func:`submit_all_jobs`, exposed as ``--max-jobs`` on the CLI):

  **Unlimited (default, max_concurrent=0):**
    All jobs are submitted immediately and the function returns.  A separate
    monitoring step (``wait_for_jobs_completion``) is expected to poll until
    every job reaches a terminal state.

  **Throttled (max_concurrent>0):**
    At most *max_concurrent* jobs are active (PENDING + RUNNING) at any time.
    The function enters a rolling poll loop — as earlier jobs finish, new ones
    are submitted to fill the freed slots.  The function only returns once
    every job has reached a terminal state, so the caller can skip the
    separate monitoring step entirely.

Key Features:
- Submit ~240 jobs (24 benchmarks x 10 runs) in one command
- Optional concurrency cap (``--max-jobs N``) to avoid flooding the SLURM queue
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
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, replace
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
from .models import (
    TERMINAL_JOB_STATES,
    SlurmBatchResult,
    SlurmJobConfig,
    SlurmJobStatus,
    SubmissionFailure,
)
from .results import (
    build_submission_failure_results as _build_submission_failure_results_impl,
    collect_results as _collect_results_impl,
)

_POLL_SPINNER_FRAMES = ("|", "/", "-", "\\")


# ============================================================================
# Data Classes
# ============================================================================


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
    exclude_nodes: Optional[str] = None  # SLURM nodes to exclude (e.g. "j006,j007")
    job_label: Optional[str] = None  # Optional phase/step label to disambiguate job names


def _create_submission_failure(
    config: SlurmJobConfig,
    script_path: Path,
    error: str,
) -> SubmissionFailure:
    """Create a normalized submission-failure record."""
    return SubmissionFailure(
        benchmark_name=config.benchmark_name,
        run_number=config.run_number,
        node_count=config.node_count,
        run_dir=config.run_dir,
        script_path=script_path,
        error=error.strip(),
    )


def _snapshot_job_statuses(
    job_statuses: Dict[str, SlurmJobStatus],
) -> Dict[str, SlurmJobStatus]:
    """Create a detached status snapshot for background polling."""
    return {job_id: replace(status) for job_id, status in job_statuses.items()}


def _start_poll_future(
    executor: ThreadPoolExecutor,
    job_statuses: Dict[str, SlurmJobStatus],
) -> Future[Dict[str, str]]:
    """Start one background scheduler poll against a stable status snapshot."""
    return executor.submit(poll_jobs, _snapshot_job_statuses(job_statuses))


def _apply_polled_states(
    poll_future: Optional[Future[Dict[str, str]]],
    job_statuses: Dict[str, SlurmJobStatus],
) -> bool:
    """Apply background poll results if the future completed."""
    if poll_future is None or not poll_future.done():
        return False
    current_states = poll_future.result()
    for job_id, state in current_states.items():
        if job_id in job_statuses:
            job_statuses[job_id].state = state
    return True


def _format_poll_status_label(
    *,
    in_flight: bool,
    last_poll_started: Optional[datetime],
    last_poll_completed: Optional[datetime],
) -> str:
    """Format the live polling status shown in the job table."""
    if in_flight:
        spinner = _POLL_SPINNER_FRAMES[int(time.time() * 2) % len(_POLL_SPINNER_FRAMES)]
        started = (
            last_poll_started.strftime("%H:%M:%S")
            if last_poll_started is not None
            else "--:--:--"
        )
        return f"{spinner} polling (started {started})"
    if last_poll_completed is not None:
        return f"idle (last {last_poll_completed.strftime('%H:%M:%S')})"
    return "waiting to poll"


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
            interval_ms = 100  # 0.1s sampling interval
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
        interval_ms = 100  # 0.1s sampling interval
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
    max_concurrent: int = 0,
    poll_interval: int = 10,
) -> Tuple[Dict[str, SlurmJobStatus], List[SubmissionFailure]]:
    """Submit all sbatch scripts and return job statuses.

    Two modes of operation:

    * **Unlimited** (``max_concurrent=0``, the default): every job is submitted
      immediately and the function returns right away.  Job states will still
      be PENDING/RUNNING — the caller is responsible for calling
      ``wait_for_jobs_completion()`` afterwards to block until they finish.

    * **Throttled** (``max_concurrent>0``): at most *max_concurrent* jobs are
      kept active (PENDING + RUNNING) at any time.  As jobs complete, new ones
      are submitted to fill the freed slots.  The function blocks until **all**
      jobs have reached a terminal state (COMPLETED, FAILED, etc.), so the
      caller can skip ``wait_for_jobs_completion()`` entirely.

    Args:
        job_configs: List of (config, script_path) tuples.
        console: Rich console for output.
        dry_run: If True, generate fake statuses without submitting.
        max_concurrent: Maximum number of SLURM jobs that may be active at
            once.  0 means no limit (submit everything immediately).
        poll_interval: Seconds between ``squeue`` polls when throttling.
            Ignored when ``max_concurrent=0``.

    Returns:
        Tuple of (job_statuses dict, submission_failures list).
        When ``max_concurrent>0``, every status will already be in a terminal
        state and will include final ``sacct`` metadata.
    """
    job_statuses: Dict[str, SlurmJobStatus] = {}
    submission_failures: List[SubmissionFailure] = []

    if dry_run:
        print_warning("Dry run mode - scripts generated but not submitted")
        for config, script_path in job_configs:
            # Use fake job ID for dry run
            fake_id = f"DRY_{config.benchmark_name}_{config.run_number}"
            status = _create_pending_job_status(fake_id, config)
            status.state = "DRY_RUN"
            job_statuses[fake_id] = status
        return job_statuses, submission_failures

    if max_concurrent > 0:
        return _submit_jobs_throttled(
            job_configs, console, max_concurrent, poll_interval,
        )

    print_step(f"Submitting {len(job_configs)} jobs to SLURM...")

    submitted = 0
    failed = 0

    for config, script_path in job_configs:
        try:
            job_id = submit_job(script_path)
            job_statuses[job_id] = _create_pending_job_status(job_id, config)
            submitted += 1
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to submit {config.benchmark_name} run {config.run_number}: {e.stderr}")
            failed += 1
            submission_failures.append(_create_submission_failure(
                config,
                script_path,
                e.stderr or e.stdout or str(e),
            ))

    print_success(f"Submitted {submitted} jobs")
    if failed > 0:
        print_error(f"{failed} submissions failed")

    return job_statuses, submission_failures


def _submit_jobs_throttled(
    job_configs: List[Tuple[SlurmJobConfig, Path]],
    console: Console,
    max_concurrent: int,
    poll_interval: int,
) -> Tuple[Dict[str, SlurmJobStatus], List[SubmissionFailure]]:
    """Submit jobs with a rolling concurrency limit, blocking until completion.

    Keeps at most *max_concurrent* jobs active (PENDING + RUNNING).  When a
    running job reaches a terminal state the freed slot is immediately filled
    by submitting the next queued job.  A Rich live table shows real-time
    progress (per-state counts, active/queued totals).

    Ctrl+C stops submitting new jobs but does **not** cancel already-submitted
    ones — the user can inspect or cancel them with ``squeue``/``scancel``.

    Before returning, final ``sacct`` metadata is fetched for every submitted
    job so the caller receives the same rich status it would get from
    ``wait_for_jobs_completion()``.
    """
    job_statuses: Dict[str, SlurmJobStatus] = {}
    submission_failures: List[SubmissionFailure] = []
    pending_configs = list(job_configs)  # configs not yet submitted
    total = len(job_configs)

    def _active_count() -> int:
        return sum(1 for s in job_statuses.values() if not _is_effectively_terminal(s))

    def _submit_next() -> bool:
        """Submit the next pending job. Returns True on success, False on failure."""
        if not pending_configs:
            return False
        config, script_path = pending_configs.pop(0)
        try:
            job_id = submit_job(script_path)
            job_statuses[job_id] = _create_pending_job_status(job_id, config)
            return True
        except subprocess.CalledProcessError as e:
            print_error(
                f"Failed to submit {config.benchmark_name} run {config.run_number}: {e.stderr}"
            )
            submission_failures.append(
                _create_submission_failure(
                    config,
                    script_path,
                    e.stderr or e.stdout or str(e),
                )
            )
            return False

    print_step(
        f"Submitting {total} jobs to SLURM (max {max_concurrent} concurrent)..."
    )
    print_info("Press Ctrl+C to stop submission (already-submitted jobs continue)")

    # Initial burst
    for _ in range(min(max_concurrent, len(pending_configs))):
        _submit_next()

    try:
        with ThreadPoolExecutor(max_workers=1) as poll_executor:
            poll_future: Optional[Future[Dict[str, str]]] = None
            last_poll_started: Optional[datetime] = None
            last_poll_completed: Optional[datetime] = None
            next_poll_deadline = 0.0

            with Live(
                _build_job_state_table(
                    job_statuses,
                    title=f"SLURM Throttled Submission ({len(job_statuses)}/{total})",
                    active_count=_active_count(),
                    queued_count=len(pending_configs),
                    failed_submissions=len(submission_failures),
                    poll_status_label="waiting to poll",
                ),
                console=console,
                refresh_per_second=4,
                transient=True,
            ) as live:
                while True:
                    now = time.monotonic()

                    if _apply_polled_states(poll_future, job_statuses):
                        poll_future = None
                        last_poll_completed = datetime.now()

                    if (
                        poll_future is None
                        and job_statuses
                        and now >= next_poll_deadline
                    ):
                        poll_future = _start_poll_future(poll_executor, job_statuses)
                        last_poll_started = datetime.now()
                        next_poll_deadline = now + poll_interval

                    # Submit more if slots available
                    while pending_configs and _active_count() < max_concurrent:
                        _submit_next()

                    live.update(
                        _build_job_state_table(
                            job_statuses,
                            title=(
                                f"SLURM Throttled Submission "
                                f"({len(job_statuses) + len(submission_failures)}/{total})"
                            ),
                            active_count=_active_count(),
                            queued_count=len(pending_configs),
                            failed_submissions=len(submission_failures),
                            poll_status_label=_format_poll_status_label(
                                in_flight=poll_future is not None,
                                last_poll_started=last_poll_started,
                                last_poll_completed=last_poll_completed,
                            ),
                            last_poll_label=(
                                last_poll_completed.strftime("%H:%M:%S")
                                if last_poll_completed is not None
                                else None
                            ),
                        )
                    )

                    # Done when nothing left to submit and all terminal
                    all_done = (
                        not pending_configs
                        and poll_future is None
                        and all(_is_effectively_terminal(s) for s in job_statuses.values())
                    )
                    if all_done:
                        break

                    time.sleep(1)

    except KeyboardInterrupt:
        print_warning("Submission stopped. Already-submitted jobs will continue running.")
        print_info("Use 'squeue -u $USER' to check status or 'scancel' to cancel jobs.")

    submitted = len(job_statuses)
    failed = len(submission_failures)
    completed = sum(1 for s in job_statuses.values() if _is_effectively_terminal(s))
    print_success(f"Submitted {submitted} jobs ({completed} completed, {failed} failed to submit)")

    # Get final sacct metadata
    job_ids = list(job_statuses.keys())
    if job_ids:
        final_statuses = get_final_job_status(job_ids)
        _apply_final_status_metadata(job_statuses, final_statuses)

    return job_statuses, submission_failures


# ============================================================================
# Job Monitoring
# ============================================================================


def poll_jobs(job_statuses: Dict[str, SlurmJobStatus]) -> Dict[str, str]:
    """Query squeue for current job states.

    Args:
        job_statuses: Current in-memory statuses keyed by job id

    Returns:
        Dict mapping job_id -> state (PENDING, RUNNING, COMPLETED, etc.)
    """
    job_ids = list(job_statuses.keys())
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
            return {job_id: status.state for job_id, status in job_statuses.items()}

        states = {}
        for line in result.stdout.strip().split("\n"):
            if "|" in line:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    job_id, state = parts[0].strip(), parts[1].strip()
                    states[job_id] = state

        # Jobs not in squeue are finished — use scontrol to get real state
        for job_id in job_ids:
            if job_id not in states:
                fallback_state = _get_scontrol_status(job_id).state
                if fallback_state == "UNKNOWN":
                    existing_status = job_statuses[job_id]
                    if _has_completed_run_artifact(existing_status):
                        states[job_id] = "UNKNOWN"
                    else:
                        states[job_id] = existing_status.state
                else:
                    states[job_id] = fallback_state

        return states

    except subprocess.TimeoutExpired:
        # Keep the previous in-memory state on transient scheduler query failures.
        return {job_id: status.state for job_id, status in job_statuses.items()}


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


def _is_effectively_terminal(status: SlurmJobStatus) -> bool:
    """Return whether the job can be treated as terminal for orchestration."""
    if status.state in TERMINAL_JOB_STATES:
        return True
    return status.state == "UNKNOWN" and _has_completed_run_artifact(status)


def _apply_final_status_metadata(
    job_statuses: Dict[str, SlurmJobStatus],
    final_statuses: Dict[str, SlurmJobStatus],
) -> None:
    """Overlay scheduler-finalized fields while preserving submission metadata."""
    for job_id, final_status in final_statuses.items():
        original = job_statuses.get(job_id)
        if original is None:
            continue
        final_status.benchmark_name = original.benchmark_name
        final_status.run_number = original.run_number
        final_status.node_count = original.node_count
        final_status.run_dir = original.run_dir
        job_statuses[job_id] = final_status


def _build_job_state_table(
    job_statuses: Dict[str, SlurmJobStatus],
    *,
    title: str,
    active_count: Optional[int] = None,
    queued_count: Optional[int] = None,
    failed_submissions: int = 0,
    last_poll_label: Optional[str] = None,
    poll_status_label: Optional[str] = None,
) -> Table:
    """Create the standard SLURM job-state table used by live displays."""
    table = Table(title=title, box=None)
    table.add_column("State", style="bold")
    table.add_column("Count", justify="right")

    state_counts: Dict[str, int] = {}
    for status in job_statuses.values():
        state_counts[status.state] = state_counts.get(status.state, 0) + 1

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

    if (
        active_count is not None
        or queued_count is not None
        or failed_submissions
        or last_poll_label is not None
        or poll_status_label is not None
    ):
        table.add_row("", "")
        if active_count is not None:
            table.add_row("[bold]Active[/bold]", str(active_count))
        if queued_count is not None:
            table.add_row("[bold]Queued[/bold]", str(queued_count))
        if failed_submissions:
            table.add_row(
                f"[{Colors.FAIL}]Submit failures[/{Colors.FAIL}]",
                str(failed_submissions),
            )
        if poll_status_label is not None:
            table.add_row("[bold]Polling[/bold]", poll_status_label)
        if last_poll_label is not None:
            table.add_row("[bold]Last poll[/bold]", last_poll_label)

    return table


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

    print_step(f"Monitoring {len(job_ids)} jobs (poll every {poll_interval}s)")
    print_info("Press Ctrl+C to stop monitoring (jobs will continue running)")

    try:
        with ThreadPoolExecutor(max_workers=1) as poll_executor:
            poll_future: Optional[Future[Dict[str, str]]] = None
            last_poll_started: Optional[datetime] = None
            last_poll_completed: Optional[datetime] = None
            next_poll_deadline = 0.0

            with Live(
                _build_job_state_table(
                    job_statuses,
                    title="SLURM Job Status",
                    poll_status_label="waiting to poll",
                ),
                console=console,
                refresh_per_second=4,
                transient=True,
            ) as live:
                while True:
                    now = time.monotonic()

                    if _apply_polled_states(poll_future, job_statuses):
                        poll_future = None
                        last_poll_completed = datetime.now()

                    if poll_future is None and now >= next_poll_deadline:
                        poll_future = _start_poll_future(poll_executor, job_statuses)
                        last_poll_started = datetime.now()
                        next_poll_deadline = now + poll_interval

                    live.update(
                        _build_job_state_table(
                            job_statuses,
                            title="SLURM Job Status",
                            poll_status_label=_format_poll_status_label(
                                in_flight=poll_future is not None,
                                last_poll_started=last_poll_started,
                                last_poll_completed=last_poll_completed,
                            ),
                            last_poll_label=(
                                last_poll_completed.strftime("%H:%M:%S")
                                if last_poll_completed is not None
                                else None
                            ),
                        )
                    )

                    all_done = (
                        poll_future is None
                        and all(
                            _is_effectively_terminal(status)
                            for status in job_statuses.values()
                        )
                    )
                    if all_done:
                        break

                    time.sleep(1)

    except KeyboardInterrupt:
        print_warning("Monitoring stopped. Jobs will continue running.")
        print_info("Use 'squeue -u $USER' to check status or 'scancel' to cancel jobs.")

    # Get final status from sacct
    final_statuses = get_final_job_status(job_ids)
    _apply_final_status_metadata(job_statuses, final_statuses)

    return job_statuses


# ============================================================================
# Results Collection
# ============================================================================


def collect_results(
    job_statuses: Dict[str, SlurmJobStatus],
    experiment_dir: Path,
) -> List[Dict[str, Any]]:
    """Collect results from completed SLURM jobs."""
    return _collect_results_impl(job_statuses, experiment_dir)


def build_submission_failure_results(
    failures: List[SubmissionFailure],
) -> List[Dict[str, Any]]:
    """Build synthetic result rows for sbatch submission failures."""
    return _build_submission_failure_results_impl(failures)


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
