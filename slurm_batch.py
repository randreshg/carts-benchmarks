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

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


# ============================================================================
# Constants
# ============================================================================

# Perf cache events for hardware counter profiling
# (duplicated from benchmark_runner.py to avoid circular import)
PERF_CACHE_EVENTS = [
    "cache-references",
    "cache-misses",
    "L1-dcache-loads",
    "L1-dcache-load-misses",
    "L1-icache-loads",
    "L1-icache-load-misses",
    "dTLB-loads",
    "dTLB-load-misses",
    "iTLB-loads",
    "iTLB-load-misses",
]


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
    output_dir: Path
    size: str
    threads: int  # For OpenMP comparison (single-node only)
    port: Optional[str] = None  # Per-job port override (e.g., "10001" or "[10001-10002]")
    gdb: bool = False  # Wrap executable with gdb for backtrace on crash
    perf: bool = False  # Enable perf stat profiling for cache metrics
    perf_interval: float = 0.1  # Perf sampling interval in seconds


@dataclass
class SlurmJobStatus:
    """Status of a submitted SLURM job."""
    job_id: str
    benchmark_name: str
    run_number: int
    node_count: int  # Node count for directory path
    state: str  # PENDING, RUNNING, COMPLETED, FAILED, TIMEOUT, CANCELLED
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
#SBATCH --output={output_dir}/slurm-%j.out
#SBATCH --error={output_dir}/slurm-%j.err

# CARTS Benchmark SLURM Job
# Benchmark: {benchmark_name}
# Run: {run_number}
# Generated: {timestamp}

set -e

# Create per-run counter directory
COUNTER_DIR="{counter_dir}"
mkdir -p "$COUNTER_DIR"

{perf_dir_section}

# Generate per-run arts.cfg with correct counterFolder
# (base arts.cfg has placeholder, we override counterFolder for this run)
sed -e "s|^counterFolder=.*|counterFolder=$COUNTER_DIR|" {port_sed} "{arts_config_path}" > "{runtime_arts_cfg}"
export artsConfig="{runtime_arts_cfg}"
export CARTS_BENCHMARKS_REPORT_INIT=1

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
ARTS_START=$(date +%s.%N)
{srun_command}
ARTS_EXIT=$?
ARTS_END=$(date +%s.%N)
ARTS_DURATION=$(echo "$ARTS_END - $ARTS_START" | bc)
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

exit $ARTS_EXIT
"""

OMP_SECTION_TEMPLATE = """
if [ {node_count} -eq 1 ] && [ -x "{executable_omp}" ]; then
    echo ""
    echo "[OpenMP] Running benchmark..."
    export OMP_NUM_THREADS={threads}
    export OMP_WAIT_POLICY=ACTIVE
    OMP_START=$(date +%s.%N)
    {omp_run_command}
    OMP_EXIT=$?
    OMP_END=$(date +%s.%N)
    OMP_DURATION=$(echo "$OMP_END - $OMP_START" | bc)
    echo "[OpenMP] Exit code: $OMP_EXIT"
    echo "[OpenMP] Duration: $OMP_DURATION seconds"
else
    echo "[OpenMP] Skipped (multi-node or executable not found)"
fi
"""


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

        results/slurm_{timestamp}/jobs/{benchmark}/nodes_{N}/  <- experiment-specific
        ├── job_1.sbatch, job_2.sbatch, ...
        ├── arts_1.cfg, arts_2.cfg, ...           <- runtime config (counterFolder differs)
        ├── counters_1/, counters_2/, ...
        ├── result_1.json, result_2.json, ...
        └── slurm-*.out/err

    Args:
        config: Job configuration
        script_path: Path to write the sbatch script
        slurm_job_result_script: Path to the slurm_job_result.py script
    """
    # Build partition and account lines (only if specified)
    partition_line = f"#SBATCH --partition={config.partition}" if config.partition else ""
    account_line = f"#SBATCH --account={config.account}" if config.account else ""

    # CRITICAL: Use absolute paths - jobs may run from different working directories
    output_dir_abs = config.output_dir.resolve()

    # Per-run paths: counters_{run}, result_{run}.json, arts_{run}.cfg
    counter_dir = output_dir_abs / f"counters_{config.run_number}"
    result_json = output_dir_abs / f"result_{config.run_number}.json"
    runtime_arts_cfg = output_dir_abs / f"arts_{config.run_number}.cfg"
    perf_dir = output_dir_abs / f"perf_{config.run_number}" if config.perf else None

    arts_config_abs = config.arts_config_path.resolve() if config.arts_config_path else None
    executable_arts_abs = config.executable_arts.resolve() if config.executable_arts else None
    executable_omp_abs = config.executable_omp.resolve() if config.executable_omp else None
    slurm_job_result_abs = slurm_job_result_script.resolve()

    # Perf directory section for sbatch template
    if config.perf and perf_dir:
        perf_dir_section = (
            f'# Create per-run perf directory\n'
            f'PERF_DIR="{perf_dir}"\n'
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
                f"-o {perf_dir}/omp.csv "
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
        omp_section = '# OpenMP skipped (multi-node run - not a fair comparison)'
    else:
        omp_section = '# OpenMP skipped (executable not specified)'

    # Safe job name (SLURM limits to 64 chars)
    safe_name = config.benchmark_name.replace("/", "_").replace(" ", "_")
    job_name = f"{safe_name}_n{config.node_count}_r{config.run_number}"[:64]

    # Per-job port override (for environments where jobs share the same host)
    port_sed = f'-e "s|^port=.*|port={config.port}|"' if config.port else ''

    # Build srun command: gdb, perf, or plain (mutually exclusive)
    if config.gdb:
        srun_command = (
            f'srun --exclusive bash -c '
            f"'gdb --batch -ex run -ex \"thread apply all bt\" -ex quit --args {executable_arts_abs}'"
        )
    elif config.perf and perf_dir:
        events = ",".join(PERF_CACHE_EVENTS)
        interval_ms = int(config.perf_interval * 1000)
        # Single quotes: perf_dir is baked as absolute path at generation time,
        # ${SLURM_PROCID} is expanded by the inner bash (set per-task by srun)
        srun_command = (
            f"srun --exclusive bash -c "
            f"'perf stat -e {events} -I {interval_ms} -x , "
            f"-o {perf_dir}/arts_node_${{SLURM_PROCID}}.csv "
            f"-- {executable_arts_abs}'"
        )
    else:
        srun_command = f'srun --exclusive {executable_arts_abs}'

    script_content = SBATCH_TEMPLATE.format(
        job_name=job_name,
        node_count=config.node_count,
        time_limit=config.time_limit,
        partition_line=partition_line,
        account_line=account_line,
        output_dir=output_dir_abs,
        benchmark_name=config.benchmark_name,
        run_number=config.run_number,
        timestamp=datetime.now().isoformat(),
        arts_config_path=arts_config_abs,
        runtime_arts_cfg=runtime_arts_cfg,
        counter_dir=counter_dir,
        perf_dir_section=perf_dir_section,
        result_json=result_json,
        executable_arts=executable_arts_abs,
        srun_command=srun_command,
        omp_section=omp_section,
        slurm_job_result_script=slurm_job_result_abs,
        size=config.size,
        threads=config.threads,
        port_sed=port_sed,
    )

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

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
    when creating the runtime arts.cfg in the jobs/ directory.

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
        console.print("[yellow]Dry run mode - scripts generated but not submitted[/]")
        for config, script_path in job_configs:
            # Use fake job ID for dry run
            fake_id = f"DRY_{config.benchmark_name}_{config.run_number}"
            job_statuses[fake_id] = SlurmJobStatus(
                job_id=fake_id,
                benchmark_name=config.benchmark_name,
                run_number=config.run_number,
                node_count=config.node_count,
                state="DRY_RUN",
            )
        return job_statuses

    console.print(f"[bold]Submitting {len(job_configs)} jobs to SLURM...[/]")

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
            )
            submitted += 1
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to submit {config.benchmark_name} run {config.run_number}: {e.stderr}[/]")
            failed += 1

    console.print(f"[green]Submitted {submitted} jobs[/]", end="")
    if failed > 0:
        console.print(f" [red]({failed} failed)[/]")
    else:
        console.print()

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

        states = {}
        for line in result.stdout.strip().split("\n"):
            if "|" in line:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    job_id, state = parts[0], parts[1]
                    states[job_id] = state

        # Jobs not in squeue are either completed or failed
        for job_id in job_ids:
            if job_id not in states:
                states[job_id] = "COMPLETED"  # Will be verified with sacct

        return states

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        # If squeue fails, assume jobs are still running
        return {job_id: "UNKNOWN" for job_id in job_ids}


def get_final_job_status(job_ids: List[str]) -> Dict[str, SlurmJobStatus]:
    """Get final job status from sacct after completion.

    Args:
        job_ids: List of SLURM job IDs

    Returns:
        Dict mapping job_id -> SlurmJobStatus with final status
    """
    if not job_ids:
        return {}

    try:
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

        statuses = {}
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 8:
                job_id = parts[0]
                # Skip batch/extern steps, only want main job
                if "." in job_id:
                    continue

                # Parse exit code (format: exitcode:signal)
                exit_parts = parts[3].split(":")
                exit_code = int(exit_parts[0]) if exit_parts[0].isdigit() else None

                # Parse job name: {benchmark}_n{node_count}_r{run_number}
                job_name = parts[1]
                node_count = 1
                run_number = 0
                # Extract node_count and run_number from job name
                if "_n" in job_name and "_r" in job_name:
                    try:
                        n_idx = job_name.rfind("_n")
                        r_idx = job_name.rfind("_r")
                        node_count = int(job_name[n_idx+2:r_idx])
                        run_number = int(job_name[r_idx+2:])
                        job_name = job_name[:n_idx]  # Remove _n{}_r{} suffix
                    except (ValueError, IndexError):
                        pass

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

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return {}


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

    terminal_states = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}

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
            "PENDING": "yellow",
            "RUNNING": "blue",
            "COMPLETED": "green",
            "FAILED": "red",
            "TIMEOUT": "red",
            "CANCELLED": "yellow",
        }

        for state, count in sorted(state_counts.items()):
            color = state_colors.get(state, "white")
            table.add_row(f"[{color}]{state}[/]", str(count))

        return table

    console.print(f"\n[bold]Monitoring {len(job_ids)} jobs (poll every {poll_interval}s)...[/]")
    console.print("[dim]Press Ctrl+C to stop monitoring (jobs will continue running)[/]\n")

    try:
        with Live(create_status_table(), console=console, refresh_per_second=1) as live:
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
                    status.state in terminal_states
                    for status in job_statuses.values()
                )

                if all_done:
                    break

                time.sleep(poll_interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped. Jobs will continue running.[/]")
        console.print("[dim]Use 'squeue -u $USER' to check status or 'scancel' to cancel jobs.[/]")

    # Get final status from sacct
    final_statuses = get_final_job_status(job_ids)
    for job_id, final_status in final_statuses.items():
        if job_id in job_statuses:
            # Preserve benchmark_name and run_number from original
            final_status.benchmark_name = job_statuses[job_id].benchmark_name
            final_status.run_number = job_statuses[job_id].run_number
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
        results/slurm_{timestamp}/jobs/{benchmark}/nodes_{N}/
        ├── result_1.json, result_2.json, ...
        ├── counters_1/, counters_2/, ...
        └── slurm-*.out/err

    Args:
        job_statuses: Final job statuses
        experiment_dir: Experiment directory (results/slurm_{timestamp}/)

    Returns:
        List of result dictionaries (one per successful job)
    """
    results = []

    jobs_dir = experiment_dir / "jobs"

    for job_id, status in job_statuses.items():
        if status.state == "DRY_RUN":
            continue

        # Find result_{run_number}.json for this job
        # Path: jobs/{benchmark}/nodes_{node_count}/result_{run_number}.json
        safe_name = status.benchmark_name.replace("/", "_")
        node_dir = jobs_dir / safe_name / f"nodes_{status.node_count}"
        result_file = node_dir / f"result_{status.run_number}.json"

        if result_file.exists():
            try:
                with open(result_file) as f:
                    result = json.load(f)
                result["slurm_job_id"] = job_id
                result["slurm_state"] = status.state
                result["slurm_exit_code"] = status.exit_code
                result["slurm_elapsed"] = status.elapsed
                result["slurm_nodes"] = status.node_list
                results.append(result)
            except json.JSONDecodeError:
                # Result file exists but is invalid
                results.append({
                    "benchmark": status.benchmark_name,
                    "run_number": status.run_number,
                    "slurm_job_id": job_id,
                    "slurm_state": status.state,
                    "error": "Failed to parse result.json",
                })
        else:
            # No result file - job likely failed before writing
            results.append({
                "benchmark": status.benchmark_name,
                "run_number": status.run_number,
                "slurm_job_id": job_id,
                "slurm_state": status.state,
                "slurm_exit_code": status.exit_code,
                "error": f"No result_{status.run_number}.json found in {node_dir}",
            })

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
    manifest = {
        "metadata": metadata,
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
            "successful": sum(1 for r in results if r.get("slurm_state") == "COMPLETED"),
            "failed": sum(1 for r in results if r.get("slurm_state") in ("FAILED", "TIMEOUT", "CANCELLED")),
        },
        "results": results,
    }

    results_path = experiment_dir / "aggregated_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return results_path
