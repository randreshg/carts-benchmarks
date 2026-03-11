"""Experiment step resolution and execution orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Protocol, Sequence, Tuple

from benchmark_artifacts import ArtifactManager
from benchmark_models import BenchmarkResult, ExperimentStep


@dataclass(frozen=True)
class StepCliDefaults:
    """CLI-provided defaults used to resolve an experiment step."""

    size: str
    timeout: int
    threads_spec: Optional[str]
    nodes_spec: Optional[str]
    runs: int
    perf: bool
    perf_interval: float
    cflags: Optional[str]
    compile_args: Optional[str]
    exclude_nodes: Optional[str]
    arts_config: Optional[Path]
    launcher: Optional[str]
    explicit_step_mode: bool
    size_from_cli: bool


@dataclass(frozen=True)
class ResolvedStepConfig:
    """Fully resolved execution parameters for a single step."""

    name: str
    bench_list: List[str]
    profile_path: Path
    debug: int
    should_rebuild_arts: bool
    threads_list: Optional[List[int]]
    node_counts: Optional[List[int]]
    timeout: int
    runs: int
    perf: bool
    perf_interval: float
    size: str
    cflags: Optional[str]
    compile_args: Optional[str]
    exclude_nodes: Optional[str]
    arts_config: Optional[Path]
    launcher: Optional[str]


@dataclass(frozen=True)
class LocalStepExecutionRequest:
    """Shared context for executing resolved steps locally."""

    runner: object
    omp_threads: Optional[int]
    weak_scaling: bool
    base_size: Optional[int]
    run_timestamp: str
    clean: bool
    quiet: bool
    artifact_manager: ArtifactManager
    variant: Optional[str] = None  # None=both, "arts", "openmp"


@dataclass(frozen=True)
class SlurmStepExecutionRequest:
    """Shared context for executing resolved steps through SLURM."""

    partition: Optional[str]
    time_limit: str
    results_dir: Path
    verbose: bool
    quiet: bool
    artifact_manager: ArtifactManager
    max_jobs: int


class StepRebuildCallback(Protocol):
    """Rebuild ARTS for one resolved step when needed."""

    def __call__(self, step_config: ResolvedStepConfig) -> None: ...


class LocalStepRunner(Protocol):
    """Execute one resolved step locally."""

    def __call__(
        self,
        *,
        step_config: ResolvedStepConfig,
        request: LocalStepExecutionRequest,
    ) -> List[BenchmarkResult]: ...


class SlurmStepRunner(Protocol):
    """Execute one resolved step through the SLURM batch path."""

    def __call__(
        self,
        *,
        step_config: ResolvedStepConfig,
        request: SlurmStepExecutionRequest,
        report_steps: Optional[List[ExperimentStep]],
    ) -> None: ...


class StepResolver:
    """Resolve experiment steps against CLI defaults."""

    def __init__(
        self,
        *,
        configs_dir: Path,
        profiles_dir: Path,
        parse_threads: Callable[[str], List[int]],
        parse_nodes_spec: Callable[[str], List[int]],
        parse_inline_steps: Callable[[Optional[List[str]]], List[ExperimentStep]],
        load_experiment: Callable[[str, Path], List[ExperimentStep]],
    ) -> None:
        self.configs_dir = configs_dir
        self.profiles_dir = profiles_dir
        self.parse_threads = parse_threads
        self.parse_nodes_spec = parse_nodes_spec
        self.parse_inline_steps = parse_inline_steps
        self.load_experiment = load_experiment

    def load_steps(
        self,
        *,
        experiment: Optional[str],
        step_args: Optional[List[str]],
        size: str,
        timeout: int,
        runs: int,
        perf: bool,
        perf_interval: float,
        threads: Optional[str],
        nodes: Optional[str],
        cflags: Optional[str],
        compile_args: Optional[str],
        exclude_nodes: Optional[str],
        arts_config: Optional[Path],
        profile: Optional[Path],
        launcher: Optional[str],
    ) -> Tuple[List[ExperimentStep], bool]:
        """Load explicit steps or synthesize the default implicit step."""
        explicit_step_mode = bool(experiment or step_args)
        if step_args:
            steps = self.parse_inline_steps(step_args)
        elif experiment:
            steps = self.load_experiment(experiment, self.configs_dir)
        else:
            steps = [
                self._create_implicit_step(
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
            ]
        return steps, explicit_step_mode

    @staticmethod
    def apply_cli_profile_override(
        steps: List[ExperimentStep],
        *,
        explicit_step_mode: bool,
        profile: Optional[Path],
        quiet: bool,
        print_warning: Callable[[str], None],
    ) -> None:
        """Apply top-level --profile only when step definitions did not set one."""
        if not (explicit_step_mode and profile):
            return

        if len(steps) == 1 and not getattr(steps[0], "_has_profile", False):
            steps[0].profile = str(profile.resolve())
            setattr(steps[0], "_has_profile", True)
            return

        if not quiet:
            print_warning(
                "Ignoring --profile because explicit steps are provided. "
                "Use step `profile=...` instead."
            )

    @staticmethod
    def validate_step_paths(steps: Sequence[ExperimentStep]) -> None:
        """Validate any filesystem paths referenced by experiment steps."""
        for step in steps:
            if step.arts_config and not Path(step.arts_config).exists():
                raise ValueError(
                    f"Step '{step.name}': arts_config not found: {step.arts_config}"
                )
            if step.profile and not Path(step.profile).exists():
                raise ValueError(
                    f"Step '{step.name}': profile not found: {step.profile}"
                )

    @staticmethod
    def resolve_effective_size_label(
        steps: Sequence[ExperimentStep],
        size: str,
        size_from_cli: bool,
    ) -> str:
        """Resolve the size label shown in headers/export metadata."""
        step_sizes = [
            step.size for step in steps
            if getattr(step, "_has_size", False) and step.size
        ]
        if size_from_cli:
            return size
        if step_sizes:
            unique_step_sizes = sorted(set(step_sizes))
            return unique_step_sizes[0] if len(unique_step_sizes) == 1 else "mixed"
        return size

    def resolve_step_config(
        self,
        step_def: ExperimentStep,
        step_index: int,
        bench_list: Sequence[str],
        defaults: StepCliDefaults,
    ) -> ResolvedStepConfig:
        """Resolve one step against the top-level CLI/default configuration."""
        step_name = self.resolve_step_name(step_def, step_index)
        step_bench_list = self.resolve_step_bench_list(
            step_name,
            list(bench_list),
            step_def.benchmarks,
        )

        profile_path = (
            Path(step_def.profile)
            if step_def.profile
            else self.profiles_dir / "profile-none.cfg"
        )
        should_rebuild_arts = (
            defaults.explicit_step_mode
            or step_def.profile is not None
            or step_def.debug > 0
        )

        step_threads_spec = (
            step_def.threads
            if self._uses_step_override(
                step_def, "_has_threads", defaults.explicit_step_mode
            )
            else defaults.threads_spec
        )
        step_nodes_spec = (
            step_def.nodes
            if self._uses_step_override(
                step_def, "_has_nodes", defaults.explicit_step_mode
            )
            else defaults.nodes_spec
        )
        step_timeout = (
            step_def.timeout
            if (
                self._uses_step_override(
                    step_def, "_has_timeout", defaults.explicit_step_mode
                )
                and step_def.timeout is not None
            )
            else defaults.timeout
        )
        step_runs = (
            step_def.runs
            if self._uses_step_override(
                step_def, "_has_runs", defaults.explicit_step_mode
            )
            else defaults.runs
        )
        step_perf = (
            step_def.perf
            if self._uses_step_override(
                step_def, "_has_perf", defaults.explicit_step_mode
            )
            else defaults.perf
        )
        step_perf_interval = (
            step_def.perf_interval
            if self._uses_step_override(
                step_def, "_has_perf_interval", defaults.explicit_step_mode
            )
            else defaults.perf_interval
        )
        step_size = (
            step_def.size
            if (
                getattr(step_def, "_has_size", False)
                and not defaults.size_from_cli
                and step_def.size
            )
            else defaults.size
        )
        step_cflags = (
            step_def.cflags
            if self._uses_step_override(
                step_def, "_has_cflags", defaults.explicit_step_mode
            )
            else defaults.cflags
        )
        step_compile_args = (
            step_def.compile_args
            if self._uses_step_override(
                step_def, "_has_compile_args", defaults.explicit_step_mode
            )
            else defaults.compile_args
        )
        step_exclude_nodes = (
            step_def.exclude_nodes
            if self._uses_step_override(
                step_def, "_has_exclude_nodes", defaults.explicit_step_mode
            )
            else defaults.exclude_nodes
        )
        step_arts_config = defaults.arts_config
        if (
            self._uses_step_override(
                step_def, "_has_arts_config", defaults.explicit_step_mode
            )
            and step_def.arts_config
        ):
            step_arts_config = Path(step_def.arts_config).resolve()
        step_launcher = (
            step_def.launcher
            if (
                self._uses_step_override(
                    step_def, "_has_launcher", defaults.explicit_step_mode
                )
                and step_def.launcher
            )
            else defaults.launcher
        )

        threads_list = self.parse_threads(step_threads_spec) if step_threads_spec else None
        node_counts = self.parse_nodes_spec(step_nodes_spec) if step_nodes_spec else None
        if threads_list and len(threads_list) > 1 and node_counts and len(node_counts) > 1:
            raise ValueError("Cannot sweep both threads and nodes simultaneously.")

        return ResolvedStepConfig(
            name=step_name,
            bench_list=step_bench_list,
            profile_path=profile_path,
            debug=step_def.debug,
            should_rebuild_arts=should_rebuild_arts,
            threads_list=threads_list,
            node_counts=node_counts,
            timeout=step_timeout,
            runs=step_runs,
            perf=step_perf,
            perf_interval=step_perf_interval,
            size=step_size,
            cflags=step_cflags,
            compile_args=step_compile_args,
            exclude_nodes=step_exclude_nodes,
            arts_config=step_arts_config,
            launcher=step_launcher,
        )

    @staticmethod
    def resolve_step_bench_list(
        step_name: str,
        all_benchmarks: List[str],
        step_benchmarks: Optional[List[str]],
    ) -> List[str]:
        """Resolve benchmark list for a step, validating explicit subsets."""
        if not step_benchmarks:
            return all_benchmarks

        requested = [b.strip() for b in step_benchmarks if b and b.strip()]
        if not requested:
            raise ValueError(f"Step '{step_name}' has an empty `benchmarks` selection")

        unknown = [b for b in requested if b not in all_benchmarks]
        if unknown:
            raise ValueError(
                f"Step '{step_name}' has unknown benchmark(s): {', '.join(unknown)}"
            )

        seen: set[str] = set()
        ordered: List[str] = []
        for bench in requested:
            if bench in seen:
                continue
            seen.add(bench)
            ordered.append(bench)
        return ordered

    @staticmethod
    def resolve_step_name(step: ExperimentStep, idx: int) -> str:
        """Return a canonical, non-empty step name."""
        resolved = (step.name or f"step_{idx}").strip()
        return resolved if resolved else f"step_{idx}"

    @staticmethod
    def step_name_to_token(step_name: str) -> str:
        """Sanitize a step name for filesystem/script-safe identifiers."""
        import re

        return re.sub(r"[^A-Za-z0-9._-]+", "_", step_name).strip("_") or "default"

    @classmethod
    def validate_step_name_collisions(cls, steps: Sequence[ExperimentStep]) -> None:
        """Fail fast on step-name collisions that would overwrite artifacts/scripts."""
        names_to_indices: dict[str, List[int]] = {}
        token_to_names: dict[str, set[str]] = {}

        for idx, step in enumerate(steps, start=1):
            step_name = cls.resolve_step_name(step, idx)
            names_to_indices.setdefault(step_name, []).append(idx)
            token = cls.step_name_to_token(step_name)
            token_to_names.setdefault(token, set()).add(step_name)

        duplicate_names = {
            name: indices for name, indices in names_to_indices.items() if len(indices) > 1
        }
        if duplicate_names:
            detail = "; ".join(
                f"'{name}' at steps {indices}" for name, indices in duplicate_names.items()
            )
            raise ValueError(
                f"Duplicate step names detected ({detail}). "
                "Step names must be unique to avoid artifact collisions."
            )

        token_collisions = {
            token: sorted(names)
            for token, names in token_to_names.items()
            if len(names) > 1
        }
        if token_collisions:
            detail = "; ".join(
                f"token '{token}' from names {names}"
                for token, names in token_collisions.items()
            )
            raise ValueError(
                f"Step-name token collisions detected ({detail}). "
                "Rename steps to produce unique script-safe names."
            )

    @staticmethod
    def _uses_step_override(
        step: ExperimentStep,
        attribute: str,
        explicit_step_mode: bool,
    ) -> bool:
        return getattr(step, attribute, False) or not explicit_step_mode

    @staticmethod
    def _create_implicit_step(
        *,
        size: str,
        timeout: int,
        runs: int,
        perf: bool,
        perf_interval: float,
        threads: Optional[str],
        nodes: Optional[str],
        cflags: Optional[str],
        compile_args: Optional[str],
        exclude_nodes: Optional[str],
        arts_config: Optional[Path],
        profile: Optional[Path],
        launcher: Optional[str],
    ) -> ExperimentStep:
        step = ExperimentStep(
            name="default",
            description=None,
            profile=str(profile.resolve()) if profile else None,
            debug=0,
            runs=runs,
            perf=perf,
            perf_interval=perf_interval,
            size=size,
            threads=threads,
            nodes=nodes,
            timeout=timeout,
            cflags=cflags,
            compile_args=compile_args,
            exclude_nodes=exclude_nodes,
            arts_config=str(arts_config.resolve()) if arts_config else None,
            launcher=launcher,
        )
        explicit_flags = {
            "_has_runs": True,
            "_has_perf": True,
            "_has_perf_interval": True,
            "_has_size": True,
            "_has_threads": threads is not None,
            "_has_nodes": nodes is not None,
            "_has_timeout": True,
            "_has_cflags": cflags is not None,
            "_has_compile_args": compile_args is not None,
            "_has_exclude_nodes": exclude_nodes is not None,
            "_has_arts_config": arts_config is not None,
            "_has_profile": profile is not None,
            "_has_benchmarks": False,
            "_has_launcher": launcher is not None,
        }
        for attribute, value in explicit_flags.items():
            setattr(step, attribute, value)
        return step


class StepExecutionOrchestrator:
    """Execute resolved experiment steps using injected step runners."""

    def __init__(
        self,
        *,
        resolver: StepResolver,
        rebuild_step: StepRebuildCallback,
        run_local_step: LocalStepRunner,
        run_slurm_step: SlurmStepRunner,
        print_step: Callable[[str, int, int], None],
    ) -> None:
        self.resolver = resolver
        self.rebuild_step = rebuild_step
        self.run_local_step = run_local_step
        self.run_slurm_step = run_slurm_step
        self.print_step = print_step

    def execute_local_steps(
        self,
        *,
        steps: Sequence[ExperimentStep],
        bench_list: Sequence[str],
        defaults: StepCliDefaults,
        request: LocalStepExecutionRequest,
    ) -> List[BenchmarkResult]:
        """Execute resolved experiment steps locally."""
        all_results: List[BenchmarkResult] = []
        for idx, step_def in enumerate(steps, start=1):
            step_config = self.resolver.resolve_step_config(
                step_def,
                idx,
                bench_list,
                defaults,
            )
            if not request.quiet and len(steps) > 1:
                self.print_step(step_config.name, idx, len(steps))

            request.artifact_manager.set_phase(step_config.name)
            request.runner.clean = True if defaults.explicit_step_mode else request.clean
            self.rebuild_step(step_config)
            step_results = self.run_local_step(
                step_config=step_config,
                request=request,
            )
            result_phase = (
                step_config.name
                if (defaults.explicit_step_mode or len(steps) > 1)
                else None
            )
            for result in step_results:
                result.run_phase = result_phase
            all_results.extend(step_results)
        return all_results

    def execute_slurm_steps(
        self,
        *,
        steps: Sequence[ExperimentStep],
        bench_list: Sequence[str],
        defaults: StepCliDefaults,
        request: SlurmStepExecutionRequest,
    ) -> None:
        """Execute resolved experiment steps through the SLURM batch path."""
        report_steps = list(steps) if (defaults.explicit_step_mode or len(steps) > 1) else None
        for idx, step_def in enumerate(steps, start=1):
            step_config = self.resolver.resolve_step_config(
                step_def,
                idx,
                bench_list,
                defaults,
            )
            if not request.quiet and len(steps) > 1:
                self.print_step(step_config.name, idx, len(steps))
            self.rebuild_step(step_config)
            if not step_config.node_counts:
                raise ValueError("--slurm requires --nodes (or step nodes override)")
            self.run_slurm_step(
                step_config=step_config,
                request=request,
                report_steps=report_steps,
            )
