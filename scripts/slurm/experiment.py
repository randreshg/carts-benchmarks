"""High-level SLURM experiment orchestration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from sniff import (
    Colors,
    console,
    print_info,
    print_step,
    print_success,
    print_warning,
)
from scripts import format_summary_line
from carts_styles import print_footer
from scripts.arts_config import KEY_COUNTER_FOLDER

from artifacts import ArtifactManager
from common import (
    ARTS_CFG_FILENAME,
    COUNTERS_DIR_NAME,
    FAIL_STATUSES,
    JOB_MANIFEST_JSON_FILENAME,
    STATUS_PASS,
    VARIANT_ARTS,
    VARIANT_OPENMP,
)
from metadata import get_reproducibility_metadata
from models import BenchmarkConfig, ExperimentStep, ReferenceChecksum, Status
from report import generate_report_from_rows

from . import batch as slurm_batch


@dataclass(frozen=True)
class SlurmBatchRequest:
    """Resolved SLURM batch request for one experiment/step."""

    bench_list: List[str]
    node_counts: List[int]
    size: str
    runs: int
    timeout: int
    partition: Optional[str]
    time_limit: str
    account: Optional[str]
    explicit_arts_config: Optional[Path]
    threads: int
    output_dir: Path
    max_jobs: int
    dry_run: bool
    no_build: bool
    verbose: bool
    cflags: Optional[str]
    compile_args: Optional[str]
    gdb: bool
    profile: Optional[Path]
    perf: bool
    perf_interval: float
    exclude_nodes: Optional[str]
    artifact_manager: Optional[ArtifactManager]
    step_name: Optional[str]
    report_steps: Optional[List[ExperimentStep]]
    command_str: str


@dataclass(frozen=True)
class SlurmExecutorDependencies:
    """Callbacks required by the high-level SLURM executor."""

    resolve_effective_arts_config: Callable[[Path, Optional[Path]], Path]
    parse_time_limit_seconds: Callable[[str], int]
    get_carts_dir: Callable[[], Path]
    get_benchmarks_dir: Callable[[], Path]
    step_name_to_token: Callable[[str], str]


class SlurmExecutionHost(Protocol):
    """BenchmarkRunner surface required for SLURM experiment orchestration."""

    benchmarks_dir: Path
    artifact_manager: Optional[ArtifactManager]

    def get_executable_paths(self, bench_path: Path) -> Tuple[Path, Path]: ...

    def build_benchmark(
        self,
        name: str,
        size: str,
        variant: str = "arts",
        arts_config: Optional[Path] = None,
        cflags: str = "",
        compile_args: Optional[str] = None,
        build_output_dir: Optional[Path] = None,
    ) -> Any: ...

    def ensure_omp_reference(
        self,
        name: str,
        size: str,
        cflags: str,
        omp_threads: int,
        timeout: int,
    ) -> ReferenceChecksum: ...


def load_existing_job_statuses(
    manifest_file: Path,
) -> Dict[str, slurm_batch.SlurmJobStatus]:
    """Load previously stored SLURM job statuses from job_manifest.json."""
    if not manifest_file.exists():
        return {}
    try:
        existing_manifest = json.loads(manifest_file.read_text())
        statuses = {}
        for job_id, status_payload in existing_manifest.get("jobs", {}).items():
            if not isinstance(status_payload, dict):
                continue
            payload = dict(status_payload)
            run_dir_raw = payload.get("run_dir")
            if run_dir_raw:
                payload["run_dir"] = Path(run_dir_raw)
            statuses[job_id] = slurm_batch.SlurmJobStatus(**payload)
        return statuses
    except Exception:
        return {}


def require_slurm_commands(dry_run: bool) -> None:
    """Validate that the required SLURM executables are available."""
    required_slurm_cmds = ["sbatch"]
    if not dry_run:
        required_slurm_cmds.extend(["squeue", "sacct", "scontrol", "sinfo"])
    missing_slurm_cmds = [cmd for cmd in required_slurm_cmds if shutil.which(cmd) is None]
    if missing_slurm_cmds:
        raise ValueError(
            "Missing required SLURM command(s): "
            + ", ".join(missing_slurm_cmds)
            + ". Run this on a SLURM login node or add SLURM tools to PATH."
        )


def _run_slurm_query(args: Sequence[str], label: str) -> str:
    """Run a SLURM query command and normalize failures into ValueError."""
    try:
        result = subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ValueError(f"Failed to run {label}: {exc}") from exc

    if result.returncode != 0:
        raise ValueError(
            f"Failed to query {label}: {(result.stderr or result.stdout or '').strip()}"
        )
    return result.stdout or ""


def _expand_slurm_nodelist(node_spec: str) -> List[str]:
    """Expand a Slurm hostlist expression into concrete node names."""
    output = _run_slurm_query(
        ["scontrol", "show", "hostnames", node_spec],
        f"SLURM node list '{node_spec}'",
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def get_effective_slurm_partition(partition: Optional[str]) -> Optional[str]:
    """Resolve the partition that SLURM will use for this request."""
    if partition:
        return partition

    output = _run_slurm_query(["sinfo", "-h", "-o", "%P"], "SLURM partitions")
    discovered: List[str] = []
    default_partition: Optional[str] = None

    for raw in output.splitlines():
        token = raw.strip()
        if not token:
            continue
        normalized = token.replace("*", "").strip()
        if not normalized:
            continue
        if normalized not in discovered:
            discovered.append(normalized)
        if "*" in token and default_partition is None:
            default_partition = normalized

    if default_partition is not None:
        return default_partition
    if len(discovered) == 1:
        return discovered[0]
    return None


def get_slurm_partition_nodes(partition: str) -> List[str]:
    """Return the concrete nodes visible in the requested Slurm partition."""
    output = _run_slurm_query(
        ["sinfo", "-h", "-p", partition, "-o", "%N"],
        f"SLURM partition '{partition}'",
    )

    nodes: List[str] = []
    seen: set[str] = set()
    for raw in output.splitlines():
        spec = raw.strip()
        if not spec or spec.lower() == "n/a":
            continue
        for node in _expand_slurm_nodelist(spec):
            if node in seen:
                continue
            seen.add(node)
            nodes.append(node)
    return nodes


def validate_requested_node_counts(
    node_counts: Sequence[int],
    partition: Optional[str],
) -> None:
    """Fail fast when a SLURM sweep requests more nodes than are available."""
    if not node_counts:
        return

    effective_partition = get_effective_slurm_partition(partition)
    if effective_partition is None:
        raise ValueError(
            "Cannot validate --nodes because SLURM does not expose a unique default "
            "partition. Pass --partition explicitly."
        )

    available_nodes = get_slurm_partition_nodes(effective_partition)
    if not available_nodes:
        raise ValueError(f"No nodes are available in SLURM partition '{effective_partition}'.")

    requested_max = max(node_counts)
    available_count = len(available_nodes)
    if requested_max > available_count:
        raise ValueError(
            f"Requested --nodes up to {requested_max}, but partition '{effective_partition}' only exposes "
            f"{available_count} node(s): {', '.join(available_nodes)}"
        )


def format_node_counts_display(node_counts: Sequence[int]) -> str:
    """Format a node-count list for human-readable CLI output."""
    if len(node_counts) == 1:
        return str(node_counts[0])
    if len(node_counts) <= 5:
        return ", ".join(str(n) for n in node_counts)
    return f"{node_counts[0]}-{node_counts[-1]} ({len(node_counts)} values)"


def find_multinode_disabled_benchmarks(
    host: SlurmExecutionHost,
    bench_list: Sequence[str],
) -> set[str]:
    """Return benchmarks that explicitly disable multinode execution."""
    disabled: set[str] = set()
    for bench in bench_list:
        bench_path = host.benchmarks_dir / bench
        if (bench_path / ".disable-multinode").exists():
            disabled.add(bench)
    return disabled


def count_total_slurm_jobs(
    bench_list: Sequence[str],
    node_counts: Sequence[int],
    runs: int,
    multinode_disabled: set[str],
) -> int:
    """Count the total jobs that will be generated for a SLURM sweep."""
    total_jobs = 0
    for node_count in node_counts:
        if node_count == 1:
            total_jobs += len(bench_list) * runs
        else:
            total_jobs += (len(bench_list) - len(multinode_disabled)) * runs
    return total_jobs


def merge_result_rows(
    existing_results: Sequence[Dict[str, Any]],
    submission_failure_results: Sequence[Dict[str, Any]],
    current_results: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge result rows while deduplicating by SLURM job id or run dir."""
    merged_results: List[Dict[str, Any]] = []
    seen_result_keys: set[str] = set()
    for result in list(existing_results) + list(submission_failure_results) + list(current_results):
        slurm_data = result.get("slurm", {})
        job_id = str(slurm_data.get("job_id", "")).strip()
        result_key = job_id
        if not result_key:
            result_key = str((result.get("artifacts") or {}).get("run_dir") or "")
        if result_key and result_key in seen_result_keys:
            continue
        if result_key:
            seen_result_keys.add(result_key)
        merged_results.append(result)
    return merged_results


class SlurmBatchExecutor:
    """Execute one SLURM experiment end-to-end."""

    def __init__(
        self,
        host: SlurmExecutionHost,
        request: SlurmBatchRequest,
        deps: SlurmExecutorDependencies,
    ) -> None:
        self.host = host
        self.request = request
        self.deps = deps

    def execute(self) -> None:
        slurm_start_time = time.time()
        am = self._prepare_artifact_manager()
        experiment_dir = am.experiment_dir
        scripts_dir = experiment_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)

        print_info(f"Experiment directory: {experiment_dir}")
        self._rebuild_profile_if_needed()

        build_results, reference_checksums = self._build_benchmarks(am)
        job_configs = self._generate_job_scripts(
            am=am,
            scripts_dir=scripts_dir,
            build_results=build_results,
            reference_checksums=reference_checksums,
        )

        self._submit_and_collect(
            am=am,
            experiment_dir=experiment_dir,
            job_configs=job_configs,
            total_duration=time.time() - slurm_start_time,
            slurm_start_time=slurm_start_time,
        )

    def _prepare_artifact_manager(self) -> ArtifactManager:
        am = self.request.artifact_manager
        if am is None:
            am = ArtifactManager(
                self.request.output_dir,
                datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
        am.set_phase(self.request.step_name or "default")
        self.host.artifact_manager = am
        return am

    def _rebuild_profile_if_needed(self) -> None:
        profile = self.request.profile
        if profile is None:
            return
        if not profile.exists():
            raise ValueError(f"Profile not found: {profile}")
        print_warning(f"Rebuilding ARTS with profile: {profile}")
        result = subprocess.run(
            ["carts", "build", "--arts", f"--profile={profile}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError(f"ARTS rebuild failed:\n{result.stderr}")
        print_success("ARTS rebuild complete")

    def _build_benchmarks(
        self,
        am: ArtifactManager,
    ) -> Tuple[
        Dict[Tuple[str, int], Tuple[Path, Optional[Path], Path]],
        Dict[str, ReferenceChecksum],
    ]:
        num_workers = min(os.cpu_count() or 1, len(self.request.bench_list))
        print_step(f"Building benchmarks per node count ({num_workers} workers)", 1, 5)

        build_results: Dict[Tuple[str, int], Tuple[Path, Optional[Path], Path]] = {}
        reference_checksums: Dict[str, ReferenceChecksum] = {}
        print_lock = threading.Lock()
        multinode_disabled = find_multinode_disabled_benchmarks(
            self.host,
            self.request.bench_list,
        )

        def build_one_bench(
            bench: str,
        ) -> List[Tuple[Tuple[str, int], Tuple[Path, Optional[Path], Path]]]:
            bench_path = self.host.benchmarks_dir / bench
            effective_base_config = self.deps.resolve_effective_arts_config(
                bench_path,
                self.request.explicit_arts_config,
            )
            src_arts, src_omp = self.host.get_executable_paths(bench_path)
            results: List[Tuple[Tuple[str, int], Tuple[Path, Optional[Path], Path]]] = []

            needs_multinode_reference = any(
                node_count > 1 and bench not in multinode_disabled
                for node_count in self.request.node_counts
            )
            if needs_multinode_reference:
                reference_timeout = self.deps.parse_time_limit_seconds(
                    self.request.time_limit
                )
                reference_timeout = min(reference_timeout, self.request.timeout)
                reference = self.host.ensure_omp_reference(
                    bench,
                    self.request.size,
                    self.request.cflags or "",
                    self.request.threads,
                    timeout=reference_timeout,
                )
                if reference.status != Status.PASS or reference.checksum is None:
                    raise RuntimeError(
                        f"Failed to establish OpenMP reference for {bench}: {reference.note}"
                    )
                with print_lock:
                    console.print(
                        f"  {bench} multinode reference checksum... [{Colors.SUCCESS}]OK[/{Colors.SUCCESS}] "
                        f"[{Colors.DEBUG}]({reference.checksum}, {self.request.threads} OMP threads)[/{Colors.DEBUG}]"
                    )
                reference_checksums[bench] = reference

            for node_count in self.request.node_counts:
                if node_count > 1 and bench in multinode_disabled:
                    continue

                bench_config = BenchmarkConfig(
                    arts_threads=self.request.threads,
                    arts_nodes=node_count,
                    omp_threads=self.request.threads,
                    launcher="slurm",
                )
                build_node_dir = am.get_artifacts_dir(bench, bench_config)
                build_node_dir.mkdir(parents=True, exist_ok=True)

                dst_arts = build_node_dir / src_arts.name
                dst_omp = build_node_dir / src_omp.name if node_count == 1 else None
                build_arts_cfg = build_node_dir / ARTS_CFG_FILENAME

                if dst_arts.exists() and build_arts_cfg.exists():
                    with print_lock:
                        console.print(
                            f"  {bench} (nodes={node_count}, threads={self.request.threads})... "
                            f"[{Colors.INFO}]SKIP (exists)[/{Colors.INFO}]"
                        )
                    if node_count == 1 and dst_omp and not dst_omp.exists():
                        self.host.build_benchmark(
                            bench,
                            self.request.size,
                            variant=VARIANT_OPENMP,
                            cflags=self.request.cflags or "",
                            build_output_dir=build_node_dir,
                        )
                    results.append(
                        (
                            (bench, node_count),
                            (
                                dst_arts,
                                dst_omp if dst_omp and dst_omp.exists() else None,
                                build_arts_cfg,
                            ),
                        )
                    )
                    continue

                if self.request.no_build:
                    with print_lock:
                        console.print(
                            f"  {bench} (nodes={node_count}, threads={self.request.threads})... "
                            f"[{Colors.ERROR}]MISSING (--no-build)[/{Colors.ERROR}]"
                        )
                    continue

                build_arts_cfg = slurm_batch.generate_arts_config_for_node(
                    effective_base_config,
                    build_node_dir,
                    node_count,
                    self.request.threads,
                )
                build_arts = self.host.build_benchmark(
                    bench,
                    self.request.size,
                    variant=VARIANT_ARTS,
                    arts_config=build_arts_cfg,
                    cflags=self.request.cflags or "",
                    compile_args=self.request.compile_args,
                    build_output_dir=build_node_dir,
                )
                if build_arts.status != Status.PASS:
                    with print_lock:
                        console.print(
                            f"  {bench} (nodes={node_count}, threads={self.request.threads})... [{Colors.ERROR}]FAILED[/{Colors.ERROR}]"
                        )
                        if self.request.verbose:
                            console.print(f"    {build_arts.output[:200]}...")
                    continue
                if not dst_arts.exists():
                    with print_lock:
                        console.print(
                            f"  {bench} (nodes={node_count}, threads={self.request.threads})... "
                            f"[{Colors.ERROR}]FAILED (missing ARTS executable in artifacts dir)[/{Colors.ERROR}]"
                        )
                    continue

                dst_omp = None
                if node_count == 1:
                    build_omp = self.host.build_benchmark(
                        bench,
                        self.request.size,
                        variant=VARIANT_OPENMP,
                        cflags=self.request.cflags or "",
                        build_output_dir=build_node_dir,
                    )
                    cached_omp = build_node_dir / src_omp.name
                    if build_omp.status == Status.PASS and cached_omp.exists():
                        dst_omp = cached_omp

                with print_lock:
                    console.print(
                        f"  {bench} (nodes={node_count}, threads={self.request.threads})... [{Colors.SUCCESS}]OK[/{Colors.SUCCESS}]"
                    )
                results.append(((bench, node_count), (dst_arts, dst_omp, build_arts_cfg)))
            return results

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(build_one_bench, bench): bench
                for bench in self.request.bench_list
            }
            for future in as_completed(futures):
                for key, value in future.result():
                    build_results[key] = value

        return build_results, reference_checksums

    def _generate_job_scripts(
        self,
        *,
        am: ArtifactManager,
        scripts_dir: Path,
        build_results: Dict[Tuple[str, int], Tuple[Path, Optional[Path], Path]],
        reference_checksums: Dict[str, ReferenceChecksum],
    ) -> List[Tuple[slurm_batch.SlurmJobConfig, Path]]:
        print_step("Generating job scripts", 2, 5)
        slurm_job_result_script = Path(__file__).parent / "job_result.py"
        job_configs: List[Tuple[slurm_batch.SlurmJobConfig, Path]] = []
        step_token = self.deps.step_name_to_token(self.request.step_name or "default")
        seen_run_dirs: set[Path] = set()
        seen_script_paths: set[Path] = set()

        for (bench, node_count), (arts_exe, omp_exe, build_arts_cfg) in build_results.items():
            safe_name = bench.replace("/", "_")
            for run_num in range(1, self.request.runs + 1):
                bench_config = BenchmarkConfig(
                    arts_threads=self.request.threads,
                    arts_nodes=node_count,
                    omp_threads=self.request.threads,
                    launcher="slurm",
                )
                run_dir = am.get_run_dir(bench, bench_config, run_num)
                run_dir_resolved = run_dir.resolve()
                if run_dir_resolved in seen_run_dirs:
                    raise ValueError(
                        f"Collision detected: duplicate run directory '{run_dir_resolved}'. "
                        "Ensure step names and benchmark/config combinations are unique."
                    )
                seen_run_dirs.add(run_dir_resolved)
                am.save_run_config(
                    bench,
                    bench_config,
                    run_num,
                    arts_cfg_path=build_arts_cfg,
                    runtime_arts_overrides={
                        KEY_COUNTER_FOLDER: str((run_dir / COUNTERS_DIR_NAME).resolve()),
                    },
                    size=self.request.size,
                    cflags=self.request.cflags,
                    compile_args=self.request.compile_args,
                    run_phase=self.request.step_name or "default",
                    profile=str(self.request.profile) if self.request.profile else None,
                    perf=self.request.perf,
                    perf_interval=self.request.perf_interval if self.request.perf else None,
                    timeout=self.request.timeout,
                    time_limit=self.request.time_limit,
                    reference_checksum=(
                        reference_checksums.get(bench).checksum
                        if node_count > 1 and bench in reference_checksums
                        else None
                    ),
                    reference_source=(
                        reference_checksums.get(bench).source
                        if node_count > 1 and bench in reference_checksums
                        else None
                    ),
                    reference_threads=(
                        reference_checksums.get(bench).omp_threads
                        if node_count > 1 and bench in reference_checksums
                        else None
                    ),
                )
                am.record_run(
                    bench,
                    bench_config,
                    run_num,
                    has_counters=bool(self.request.profile),
                    has_perf=self.request.perf,
                )

                config = slurm_batch.SlurmJobConfig(
                    benchmark_name=bench,
                    run_number=run_num,
                    node_count=node_count,
                    time_limit=self.request.time_limit,
                    partition=self.request.partition,
                    account=self.request.account,
                    executable_arts=arts_exe,
                    executable_omp=omp_exe,
                    arts_config_path=build_arts_cfg,
                    python_executable=Path(sys.executable).resolve(),
                    run_dir=run_dir,
                    size=self.request.size,
                    threads=self.request.threads,
                    timeout_seconds=self.request.timeout,
                    gdb=self.request.gdb,
                    perf=self.request.perf,
                    perf_interval=self.request.perf_interval,
                    exclude_nodes=self.request.exclude_nodes,
                    job_label=step_token,
                )
                script_path = (
                    scripts_dir
                    / f"{step_token}__{safe_name}_{self.request.threads}t_{node_count}n_run{run_num}.sbatch"
                )
                script_path_resolved = script_path.resolve()
                if script_path_resolved in seen_script_paths:
                    raise ValueError(
                        f"Collision detected: duplicate script path '{script_path_resolved}'. "
                        "Ensure step names and benchmark/config combinations are unique."
                    )
                seen_script_paths.add(script_path_resolved)
                slurm_batch.generate_sbatch_script(
                    config,
                    script_path,
                    slurm_job_result_script,
                )
                job_configs.append((config, script_path))

        print_info(f"Generated {len(job_configs)} job scripts")
        return job_configs

    def _submit_and_collect(
        self,
        *,
        am: ArtifactManager,
        experiment_dir: Path,
        job_configs: List[Tuple[slurm_batch.SlurmJobConfig, Path]],
        total_duration: float,
        slurm_start_time: float,
    ) -> None:
        del total_duration
        print_step("Submitting jobs", 3, 5)
        job_statuses, submission_failures = slurm_batch.submit_all_jobs(
            job_configs,
            console,
            self.request.dry_run,
            max_concurrent=self.request.max_jobs,
        )
        failed_submissions = len(submission_failures)

        metadata = self._build_metadata(
            experiment_dir=experiment_dir,
            job_count=len(job_configs),
            submitted_jobs=len(job_statuses),
            failed_submissions=failed_submissions,
        )
        manifest_file = experiment_dir / JOB_MANIFEST_JSON_FILENAME
        existing_job_statuses = load_existing_job_statuses(manifest_file)
        existing_job_statuses.update(job_statuses)
        manifest_path = slurm_batch.write_job_manifest(
            experiment_dir,
            existing_job_statuses,
            metadata,
        )
        print_info(f"Job manifest: {manifest_path}")

        if self.request.dry_run:
            print_warning("Dry run complete. Scripts generated but not submitted.")
            print_info(
                f"To submit manually, run sbatch on scripts in: {experiment_dir / 'scripts'}"
            )
            return

        current_results: List[Dict[str, Any]] = []
        if job_statuses:
            if self.request.max_jobs > 0:
                print_step("Monitoring skipped (throttled submission already waited)", 4, 5)
            else:
                print_step("Monitoring jobs", 4, 5)
                job_statuses = slurm_batch.wait_for_jobs_completion(
                    job_statuses,
                    console,
                    poll_interval=10,
                )

            existing_job_statuses = load_existing_job_statuses(manifest_file)
            existing_job_statuses.update(job_statuses)
            slurm_batch.write_job_manifest(experiment_dir, existing_job_statuses, metadata)

            print_step("Collecting results", 5, 5)
            current_results = slurm_batch.collect_results(job_statuses, experiment_dir)
        else:
            print_warning("Phase 4/5 skipped: no jobs were submitted successfully.")

        submission_failure_results = slurm_batch.build_submission_failure_results(
            submission_failures
        )
        existing_results = self._load_existing_results(am)
        merged_results = merge_result_rows(
            existing_results,
            submission_failure_results,
            current_results,
        )

        results_path = slurm_batch.write_aggregated_results(
            experiment_dir,
            merged_results,
            metadata,
        )
        report_path: Optional[Path] = None
        try:
            report_path = generate_report_from_rows(
                merged_results,
                experiment_dir,
                quiet=True,
                steps=self.request.report_steps,
            )
        except Exception as exc:
            print_warning(f"Failed to generate report.xlsx: {exc}")

        repro = get_reproducibility_metadata(
            self.deps.get_carts_dir(),
            self.deps.get_benchmarks_dir(),
        )
        slurm_manifest = slurm_batch.write_slurm_manifest(
            experiment_dir,
            merged_results,
            metadata,
            self.request.command_str,
            time.time() - slurm_start_time,
            repro,
        )

        successful = sum(1 for row in merged_results if row.get("status") == STATUS_PASS)
        failed = sum(
            1
            for row in merged_results
            if row.get("status") in FAIL_STATUSES
        )
        summary_content = (
            f"{format_summary_line(successful, failed, len(merged_results) - successful - failed)}\n\n"
            f"Total jobs: {len(merged_results)}\n"
            f"Results: {results_path}\n"
            f"Manifest: {slurm_manifest}"
        )
        if report_path:
            summary_content += f"\nReport: {report_path}"
        style = Colors.SUCCESS if failed == 0 else Colors.ERROR
        print_footer(f"Experiment Complete — {summary_content}", style=style)

        if failed_submissions > 0:
            raise ValueError(
                f"SLURM submission failed for {failed_submissions}/{len(job_configs)} jobs. "
                "Report artifacts were still generated."
            )

    def _build_metadata(
        self,
        *,
        experiment_dir: Path,
        job_count: int,
        submitted_jobs: int,
        failed_submissions: int,
    ) -> Dict[str, Any]:
        experiment_name = None
        experiment_description = None
        experiment_steps = None
        if self.request.report_steps:
            experiment_name = getattr(
                self.request.report_steps[0], "_experiment_name", None
            )
            experiment_description = getattr(
                self.request.report_steps[0], "_experiment_description", None
            )
            experiment_steps = [
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
                for step in self.request.report_steps
            ]
        return {
            "timestamp": experiment_dir.name,
            "experiment_name": experiment_name,
            "experiment_description": experiment_description,
            "experiment_steps": experiment_steps,
            "size": self.request.size,
            "node_counts": self.request.node_counts,
            "threads": self.request.threads,
            "runs_per_benchmark": self.request.runs,
            "total_jobs": job_count,
            "submitted_jobs": submitted_jobs,
            "failed_submissions": failed_submissions,
            "partition": self.request.partition,
            "time_limit": self.request.time_limit,
            "arts_config": (
                str(self.request.explicit_arts_config)
                if self.request.explicit_arts_config is not None
                else "benchmark-specific defaults"
            ),
            "max_jobs": self.request.max_jobs,
            "dry_run": self.request.dry_run,
            "profile": str(self.request.profile) if self.request.profile else None,
            "perf": self.request.perf,
            "perf_interval": self.request.perf_interval if self.request.perf else None,
        }

    def _load_existing_results(
        self,
        am: ArtifactManager,
    ) -> List[Dict[str, Any]]:
        if not am.results_json_path.exists():
            return []
        try:
            existing_payload = json.loads(am.results_json_path.read_text())
            raw_results = existing_payload.get("results", [])
            return raw_results if isinstance(raw_results, list) else []
        except Exception:
            return []
