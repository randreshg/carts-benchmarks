from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "scripts"
TOOLS_DIR = REPO_ROOT / "tools"

sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark_models import ExperimentStep  # noqa: E402
from benchmark_orchestration import (  # noqa: E402
    LocalStepExecutionRequest,
    SlurmStepExecutionRequest,
    StepCliDefaults,
    StepExecutionOrchestrator,
    StepResolver,
)


def _parse_csv_ints(spec: str) -> list[int]:
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


class _FakeArtifactManager:
    def __init__(self) -> None:
        self.phases: list[str] = []

    def set_phase(self, phase_label: str | None = None) -> None:
        self.phases.append(phase_label or "")


class BenchmarkOrchestrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.configs_dir = self.root / "configs"
        self.profiles_dir = self.root / "profiles"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        (self.profiles_dir / "profile-none.cfg").write_text("# none\n")

        self.resolver = StepResolver(
            configs_dir=self.configs_dir,
            profiles_dir=self.profiles_dir,
            parse_threads=_parse_csv_ints,
            parse_nodes_spec=_parse_csv_ints,
            parse_inline_steps=lambda args: [],
            load_experiment=lambda experiment, configs_dir: [],
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_implicit_step_load_and_resolution(self) -> None:
        arts_cfg = self.root / "arts.cfg"
        arts_cfg.write_text("[ARTS]\nthreads=4\n")

        steps, explicit = self.resolver.load_steps(
            experiment=None,
            step_args=None,
            size="large",
            timeout=30,
            runs=2,
            perf=True,
            perf_interval=0.5,
            threads="4,8",
            nodes="2",
            cflags="-DTEST=1",
            compile_args="--distributed-db",
            exclude_nodes="j001",
            arts_config=arts_cfg,
            profile=None,
            launcher="slurm",
        )

        self.assertFalse(explicit)
        self.assertEqual(len(steps), 1)

        defaults = StepCliDefaults(
            size="large",
            timeout=30,
            threads_spec="4,8",
            nodes_spec="2",
            runs=2,
            perf=True,
            perf_interval=0.5,
            cflags="-DTEST=1",
            compile_args="--distributed-db",
            exclude_nodes="j001",
            arts_config=arts_cfg,
            launcher="slurm",
            explicit_step_mode=explicit,
            size_from_cli=True,
        )

        resolved = self.resolver.resolve_step_config(
            steps[0],
            1,
            ["polybench/gemm"],
            defaults,
        )
        self.assertEqual(resolved.name, "default")
        self.assertEqual(resolved.bench_list, ["polybench/gemm"])
        self.assertEqual(resolved.threads_list, [4, 8])
        self.assertEqual(resolved.node_counts, [2])
        self.assertEqual(resolved.compile_args, "--distributed-db")
        self.assertEqual(resolved.launcher, "slurm")
        self.assertEqual(resolved.arts_config, arts_cfg.resolve())

    def test_local_execution_orchestrator_sets_phase_and_result_phase(self) -> None:
        artifact_manager = _FakeArtifactManager()
        runner = types.SimpleNamespace(clean=False)
        calls: list[str] = []

        def run_local_step(*, step_config, request):
            calls.append(step_config.name)
            self.assertEqual(request.run_timestamp, "ts")
            return [types.SimpleNamespace(run_phase=None)]

        orchestrator = StepExecutionOrchestrator(
            resolver=self.resolver,
            rebuild_step=lambda step_config: calls.append(f"rebuild:{step_config.name}"),
            run_local_step=run_local_step,
            run_slurm_step=lambda **kwargs: None,
            print_step=lambda label, idx, total: calls.append(f"print:{label}:{idx}/{total}"),
        )

        steps = [ExperimentStep(name="alpha"), ExperimentStep(name="beta")]
        defaults = StepCliDefaults(
            size="small",
            timeout=10,
            threads_spec=None,
            nodes_spec=None,
            runs=1,
            perf=False,
            perf_interval=0.1,
            cflags=None,
            compile_args=None,
            exclude_nodes=None,
            arts_config=None,
            launcher=None,
            explicit_step_mode=True,
            size_from_cli=False,
        )

        results = orchestrator.execute_local_steps(
            steps=steps,
            bench_list=["polybench/gemm"],
            defaults=defaults,
            request=LocalStepExecutionRequest(
                runner=runner,
                omp_threads=None,
                weak_scaling=False,
                base_size=None,
                run_timestamp="ts",
                clean=True,
                quiet=False,
                artifact_manager=artifact_manager,
            ),
        )

        self.assertEqual(len(results), 2)
        self.assertEqual([result.run_phase for result in results], ["alpha", "beta"])
        self.assertEqual(artifact_manager.phases, ["alpha", "beta"])
        self.assertTrue(runner.clean)
        self.assertIn("rebuild:alpha", calls)
        self.assertIn("rebuild:beta", calls)

    def test_slurm_execution_orchestrator_passes_report_steps(self) -> None:
        recorded_report_steps: list[list[str]] = []

        def run_slurm_step(*, step_config, request, report_steps):
            self.assertEqual(request.time_limit, "00:10:00")
            self.assertEqual(step_config.node_counts, [8])
            recorded_report_steps.append([step.name for step in report_steps or []])

        orchestrator = StepExecutionOrchestrator(
            resolver=self.resolver,
            rebuild_step=lambda step_config: None,
            run_local_step=lambda **kwargs: [],
            run_slurm_step=run_slurm_step,
            print_step=lambda label, idx, total: None,
        )

        step = ExperimentStep(name="dist", nodes="8")
        setattr(step, "_has_nodes", True)

        defaults = StepCliDefaults(
            size="small",
            timeout=10,
            threads_spec=None,
            nodes_spec=None,
            runs=1,
            perf=False,
            perf_interval=0.1,
            cflags=None,
            compile_args=None,
            exclude_nodes=None,
            arts_config=None,
            launcher=None,
            explicit_step_mode=True,
            size_from_cli=False,
        )

        orchestrator.execute_slurm_steps(
            steps=[step],
            bench_list=["polybench/gemm"],
            defaults=defaults,
            request=SlurmStepExecutionRequest(
                partition="debug",
                time_limit="00:10:00",
                results_dir=self.root / "results",
                verbose=False,
                quiet=False,
                artifact_manager=_FakeArtifactManager(),
                max_jobs=2,
            ),
        )

        self.assertEqual(recorded_report_steps, [["dist"]])


if __name__ == "__main__":
    unittest.main()
