"""Shared local execution pipeline for benchmark configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Protocol, Tuple

from benchmark_common import filter_benchmark_output, parse_counter_json
from benchmark_execution import BenchmarkExecutionContext, BenchmarkRunFiles
from benchmark_models import (
    Artifacts,
    BenchmarkResult,
    BuildResult,
    Phase,
    RunResult,
    Status,
    VerificationResult,
)


@dataclass(frozen=True)
class ConfigExecutionPlan:
    """Resolved execution plan for one benchmark/config combination."""

    execution: BenchmarkExecutionContext
    timeout: int
    run_numbers: Tuple[int, ...]
    compile_args: Optional[str]
    perf_enabled: bool
    perf_interval: float
    counter_dir: Optional[Path] = None
    perf_dir: Optional[Path] = None
    run_timestamp: str = ""
    sweep_log_names: bool = False
    report_speedup: bool = True
    env_overrides: Dict[str, str] = field(default_factory=dict)
    persisted_env_overrides: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class ConfigBuildOutputs:
    """Build outputs shared across all runs for one config."""

    build_arts: BuildResult
    build_omp: BuildResult


@dataclass
class ExecutionHooks:
    """Optional hooks for live progress reporting during config execution."""

    phase_callback: Optional[Callable[[Phase], None]] = None
    partial_results: Optional[MutableMapping[str, Any]] = None

    def emit_phase(self, phase: Phase) -> None:
        if self.phase_callback:
            self.phase_callback(phase)

    def store_partial(self, key: str, value: Any) -> None:
        if self.partial_results is not None:
            self.partial_results[key] = value


class BenchmarkExecutionHost(Protocol):
    """Minimal BenchmarkRunner surface required by ConfigExecutionExecutor."""

    artifact_manager: Any
    trace: bool
    console: Any

    def build_benchmark(
        self,
        name: str,
        size: str,
        variant: str,
        arts_config: Optional[Path] = None,
        cflags: str = "",
        compile_args: Optional[str] = None,
        build_output_dir: Optional[Path] = None,
    ) -> BuildResult: ...

    def run_benchmark(
        self,
        executable: str,
        timeout: int = ...,
        env: Optional[Dict[str, str]] = ...,
        launcher: str = ...,
        node_count: int = ...,
        threads: int = ...,
        args: Optional[List[str]] = ...,
        log_file: Optional[Path] = ...,
        perf_enabled: bool = ...,
        perf_interval: float = ...,
        perf_output_name: str = ...,
        perf_output_dir: Optional[Path] = ...,
        counter_dir: Optional[Path] = ...,
    ) -> RunResult: ...

    def collect_artifacts(self, bench_path: Path) -> Artifacts: ...

    def calculate_timing(
        self,
        arts_result: RunResult,
        omp_result: RunResult,
        report_speedup: bool = ...,
    ) -> BenchmarkRunFiles: ...

    def verify_correctness(
        self,
        arts_result: RunResult,
        omp_result: RunResult,
        tolerance: float,
    ) -> VerificationResult: ...

    def get_size_params(self, bench_path: Path, size: str) -> Optional[str]: ...

    def _cleanup_port(self) -> None: ...

    def _index_build_artifacts(
        self,
        artifacts_dir: Path,
        arts_cfg_used: Optional[Path] = None,
    ) -> Dict[str, Optional[str]]: ...

    def _create_run_files(
        self,
        *,
        name: str,
        bench_path: Path,
        config: Any,
        desired_threads: int,
        run_number: int,
        runs: int,
        counter_dir: Optional[Path],
        perf_enabled: bool,
        perf_dir: Optional[Path] = None,
        run_timestamp: str = "",
        sweep_log_names: bool = False,
    ) -> Any: ...


class ConfigExecutionExecutor:
    """Execute a prepared benchmark config plan end-to-end."""

    def __init__(self, host: BenchmarkExecutionHost, plan: ConfigExecutionPlan) -> None:
        self.host = host
        self.plan = plan

    def execute(self, hooks: Optional[ExecutionHooks] = None) -> List[BenchmarkResult]:
        hooks = hooks or ExecutionHooks()
        build_outputs = self._build_variants(hooks)
        results: List[BenchmarkResult] = []
        for run_number in self.plan.run_numbers:
            results.append(self._execute_run(run_number, build_outputs, hooks))
        return results

    def _build_variants(self, hooks: ExecutionHooks) -> ConfigBuildOutputs:
        execution = self.plan.execution
        hooks.emit_phase(Phase.BUILD_ARTS)
        build_arts = self.host.build_benchmark(
            execution.name,
            execution.size,
            "arts",
            execution.effective_arts_cfg,
            execution.effective_cflags,
            self.plan.compile_args,
            build_output_dir=execution.build_output_dir,
        )
        hooks.store_partial("build_arts", build_arts)

        hooks.emit_phase(Phase.BUILD_OMP)
        build_omp = self.host.build_benchmark(
            execution.name,
            execution.size,
            "openmp",
            None,
            execution.effective_cflags,
            build_output_dir=execution.build_output_dir,
        )
        hooks.store_partial("build_omp", build_omp)

        return ConfigBuildOutputs(build_arts=build_arts, build_omp=build_omp)

    def _execute_run(
        self,
        run_number: int,
        build_outputs: ConfigBuildOutputs,
        hooks: ExecutionHooks,
    ) -> BenchmarkResult:
        execution = self.plan.execution
        run_files = self.host._create_run_files(
            name=execution.name,
            bench_path=execution.bench_path,
            config=execution.config,
            desired_threads=execution.desired_threads,
            run_number=run_number,
            runs=len(self.plan.run_numbers),
            counter_dir=self.plan.counter_dir,
            perf_enabled=self.plan.perf_enabled,
            perf_dir=self.plan.perf_dir,
            run_timestamp=self.plan.run_timestamp,
            sweep_log_names=self.plan.sweep_log_names,
        )

        self.host._cleanup_port()
        run_arts = self._run_arts(build_outputs.build_arts, execution, run_files, hooks)
        self._attach_counter_timings(run_arts, run_files.counter_dir)
        hooks.store_partial("run_arts", run_arts)

        run_omp = self._run_omp(build_outputs.build_omp, execution, run_files, hooks)
        self._append_perf_csv(run_files.arts_perf_temp, run_files.arts_perf_main, run_number)
        self._append_perf_csv(run_files.omp_perf_temp, run_files.omp_perf_main, run_number)

        self._print_trace_output(execution.name, run_arts, run_omp)

        timing = self.host.calculate_timing(
            run_arts,
            run_omp,
            report_speedup=self.plan.report_speedup,
        )
        verification = self._verify(run_arts, run_omp, execution.verify_tolerance)
        artifacts = self._collect_artifacts(
            execution=execution,
            run_files=run_files,
            run_number=run_number,
            perf_enabled=self.plan.perf_enabled,
        )

        total_duration = (
            build_outputs.build_arts.duration_sec
            + build_outputs.build_omp.duration_sec
            + run_arts.duration_sec
            + run_omp.duration_sec
        )
        return BenchmarkResult(
            name=execution.name,
            suite=execution.suite,
            size=execution.size,
            config=execution.config,
            run_number=run_number,
            build_arts=build_outputs.build_arts,
            build_omp=build_outputs.build_omp,
            run_arts=run_arts,
            run_omp=run_omp,
            timing=timing,
            verification=verification,
            artifacts=artifacts,
            timestamp=datetime.now().isoformat(),
            total_duration_sec=total_duration,
            size_params=self.host.get_size_params(execution.bench_path, execution.size),
        )

    def _run_arts(
        self,
        build_arts: BuildResult,
        execution: BenchmarkExecutionContext,
        run_files: BenchmarkRunFiles,
        hooks: ExecutionHooks,
    ) -> RunResult:
        hooks.emit_phase(Phase.RUN_ARTS)
        if build_arts.status != Status.PASS or not build_arts.executable:
            return RunResult(
                status=Status.SKIP,
                duration_sec=0.0,
                exit_code=-1,
                stdout="",
                stderr="Build failed",
            )
        return self.host.run_benchmark(
            build_arts.executable,
            self.plan.timeout,
            env=dict(self.plan.env_overrides),
            launcher=execution.desired_launcher,
            node_count=execution.desired_nodes,
            threads=execution.desired_threads,
            args=execution.run_args,
            log_file=run_files.arts_log,
            perf_enabled=self.plan.perf_enabled,
            perf_interval=self.plan.perf_interval,
            perf_output_name=run_files.arts_perf_name or "perf_cache_arts.csv",
            perf_output_dir=run_files.perf_output_dir,
            counter_dir=run_files.counter_dir,
        )

    def _run_omp(
        self,
        build_omp: BuildResult,
        execution: BenchmarkExecutionContext,
        run_files: BenchmarkRunFiles,
        hooks: ExecutionHooks,
    ) -> RunResult:
        hooks.emit_phase(Phase.RUN_OMP)
        if build_omp.status != Status.PASS or not build_omp.executable:
            return RunResult(
                status=Status.SKIP,
                duration_sec=0.0,
                exit_code=-1,
                stdout="",
                stderr="Build failed",
            )
        omp_env = dict(self.plan.env_overrides)
        omp_env["OMP_NUM_THREADS"] = str(execution.actual_omp_threads)
        if "OMP_WAIT_POLICY" not in omp_env:
            omp_env["OMP_WAIT_POLICY"] = "ACTIVE"
        return self.host.run_benchmark(
            build_omp.executable,
            self.plan.timeout,
            env=omp_env,
            args=execution.run_args,
            log_file=run_files.omp_log,
            perf_enabled=self.plan.perf_enabled,
            perf_interval=self.plan.perf_interval,
            perf_output_name=run_files.omp_perf_name or "perf_cache_omp.csv",
            perf_output_dir=run_files.perf_output_dir,
        )

    def _verify(
        self,
        run_arts: RunResult,
        run_omp: RunResult,
        tolerance: float,
    ) -> VerificationResult:
        return self.host.verify_correctness(run_arts, run_omp, tolerance=tolerance)

    def _collect_artifacts(
        self,
        *,
        execution: BenchmarkExecutionContext,
        run_files: BenchmarkRunFiles,
        run_number: int,
        perf_enabled: bool,
    ) -> Artifacts:
        artifacts = self.host.collect_artifacts(execution.bench_path)
        am = self.host.artifact_manager
        if am is None:
            if execution.effective_arts_cfg:
                artifacts.arts_config = str(execution.effective_arts_cfg)
            if run_files.arts_log is not None:
                artifacts.arts_log = str(run_files.arts_log)
            if run_files.omp_log is not None:
                artifacts.omp_log = str(run_files.omp_log)
            return artifacts

        run_dir = run_files.run_dir or am.get_run_dir(execution.name, execution.config, run_number)
        artifacts_dir = execution.build_output_dir or am.get_artifacts_dir(execution.name, execution.config)
        artifact_paths = execution.artifact_paths
        if not artifact_paths and artifacts_dir.exists():
            artifact_paths = self.host._index_build_artifacts(
                artifacts_dir,
                arts_cfg_used=execution.effective_arts_cfg,
            )
        for key, attr in [
            ("arts_config", "arts_config"),
            ("carts_metadata", "carts_metadata"),
            ("arts_metadata_mlir", "arts_metadata_mlir"),
            ("executable_arts", "executable_arts"),
            ("executable_omp", "executable_omp"),
        ]:
            if artifact_paths.get(key):
                setattr(artifacts, attr, artifact_paths[key])
        artifacts.build_dir = str(artifacts_dir)
        artifacts.run_dir = str(run_dir)
        artifacts.arts_log = str(run_files.arts_log) if run_files.arts_log else None
        artifacts.omp_log = str(run_files.omp_log) if run_files.omp_log else None

        counter_path = am.get_counter_dir(execution.name, execution.config, run_number)
        counter_files = sorted(counter_path.glob("*.json"))
        has_counters = bool(counter_files)
        if has_counters:
            artifacts.counters_dir = str(counter_path)
            artifacts.counter_files = [str(path) for path in counter_files]

        has_perf = bool(
            perf_enabled
            and run_files.perf_output_dir
            and run_files.perf_output_dir.exists()
            and list(run_files.perf_output_dir.glob("*.csv"))
        )

        am.record_run(
            execution.name,
            execution.config,
            run_number,
            has_counters=has_counters,
            has_perf=has_perf,
        )

        run_cfg_path = (
            Path(artifact_paths["arts_config"])
            if artifact_paths.get("arts_config")
            else execution.effective_arts_cfg
        )
        am.save_run_config(
            execution.name,
            execution.config,
            run_number,
            arts_cfg_path=run_cfg_path,
            env_overrides=(
                self.plan.persisted_env_overrides
                if self.plan.persisted_env_overrides is not None
                else self.plan.env_overrides
            ),
            size=execution.size,
            cflags=execution.effective_cflags or None,
            compile_args=self.plan.compile_args or None,
            perf=perf_enabled,
        )
        return artifacts

    def _attach_counter_timings(
        self,
        run_arts: RunResult,
        counter_dir: Optional[Path],
    ) -> None:
        if counter_dir and counter_dir.exists():
            init_sec, e2e_sec = parse_counter_json(counter_dir)
            run_arts.counter_init_sec = init_sec
            run_arts.counter_e2e_sec = e2e_sec

    def _print_trace_output(
        self,
        benchmark_name: str,
        run_arts: RunResult,
        run_omp: RunResult,
    ) -> None:
        if not self.host.trace:
            return
        arts_combined = (run_arts.stdout or "") + ("\n" + run_arts.stderr if run_arts.stderr else "")
        omp_combined = (run_omp.stdout or "") + ("\n" + run_omp.stderr if run_omp.stderr else "")
        arts_output = filter_benchmark_output(arts_combined)
        omp_output = filter_benchmark_output(omp_combined)

        self.host.console.print(f"\n[bold cyan]═══ CARTS Output ({benchmark_name}) ═══[/]")
        if arts_output:
            self.host.console.print(arts_output)
        elif arts_combined.strip():
            self.host.console.print(arts_combined.strip())
        else:
            self.host.console.print("[dim](no benchmark output)[/]")

        self.host.console.print(f"\n[bold green]═══ OMP Output ({benchmark_name}) ═══[/]")
        if omp_output:
            self.host.console.print(omp_output)
        elif omp_combined.strip():
            self.host.console.print(omp_combined.strip())
        else:
            self.host.console.print("[dim](no benchmark output)[/]")
        self.host.console.print()

    @staticmethod
    def _append_perf_csv(
        temp_perf_file: Optional[Path],
        main_perf_file: Optional[Path],
        run_number: int,
    ) -> None:
        if not temp_perf_file or not main_perf_file or not temp_perf_file.exists():
            return

        is_first_run = not main_perf_file.exists() or main_perf_file.stat().st_size == 0
        with open(temp_perf_file, "r") as temp_f:
            lines = temp_f.readlines()

        data_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                data_lines.append(stripped)

        if not data_lines:
            temp_perf_file.unlink(missing_ok=True)
            return

        mode = "w" if is_first_run else "a"
        with open(main_perf_file, mode) as main_f:
            if is_first_run:
                main_f.write("# Columns: run,timestamp,value,unit,event,...\n")
            for line in data_lines:
                main_f.write(f"{run_number},{line}\n")

        temp_perf_file.unlink(missing_ok=True)
