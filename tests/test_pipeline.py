from __future__ import annotations

import json
import sys
import tempfile
import unittest
from io import StringIO
from types import SimpleNamespace
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "scripts"
TOOLS_DIR = REPO_ROOT / "tools"

sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from arts_config import (  # noqa: E402
    KEY_DEFAULT_PORTS,
    KEY_PIN,
    KEY_PORT_COUNT,
    KEY_PROTOCOL,
    KEY_WORKER_THREADS,
    PROTOCOL_RDMA,
    PROTOCOL_TCP,
    compile_args_for_node_count,
    parse_arts_cfg,
    protocol_for_launcher,
)
from artifacts import ArtifactManager  # noqa: E402
from execution import (  # noqa: E402
    BenchmarkExecutionContext,
    BenchmarkProcessRunner,
    BenchmarkRunFiles,
)
from models import (  # noqa: E402
    Artifacts,
    BenchmarkConfig,
    BuildResult,
    RunResult,
    Status,
    TimingResult,
    VerificationResult,
)
from pipeline import ConfigExecutionExecutor, ConfigExecutionPlan, ExecutionHooks  # noqa: E402
from rich.console import Console  # noqa: E402
from runner import BenchmarkRunner, generate_arts_config  # noqa: E402


class _RunOnlyHost:
    def __init__(self) -> None:
        self.last_env: dict[str, str] | None = None

    def run_benchmark(self, *args, **kwargs) -> RunResult:
        self.last_env = dict(kwargs.get("env") or {})
        return RunResult(
            status=Status.PASS,
            duration_sec=0.01,
            exit_code=0,
            stdout="",
            stderr="",
        )


class _ArtifactHost:
    def __init__(self, artifact_manager: ArtifactManager) -> None:
        self.artifact_manager = artifact_manager
        self.trace = False
        self.console = None

    def collect_artifacts(self, bench_path: Path) -> Artifacts:
        return Artifacts(benchmark_dir=str(bench_path))


class _ExecutionOrderHost:
    def __init__(self, root: Path) -> None:
        self.artifact_manager = None
        self.trace = False
        self.console = Console(file=StringIO())
        self.root = root
        self.run_calls: list[str] = []

    def build_benchmark(
        self,
        name: str,
        size: str,
        variant: str,
        arts_config: Path | None = None,
        cflags: str = "",
        compile_args: str | None = None,
        build_output_dir: Path | None = None,
    ) -> BuildResult:
        suffix = "arts" if variant == "arts" else "omp"
        return BuildResult(
            status=Status.PASS,
            duration_sec=0.01,
            output="",
            executable=str(self.root / f"bench_{suffix}"),
        )

    def run_benchmark(self, executable: str, *args, **kwargs) -> RunResult:
        self.run_calls.append("arts" if executable.endswith("_arts") else "omp")
        return RunResult(
            status=Status.PASS,
            duration_sec=0.01,
            exit_code=0,
            stdout="checksum: 1\nkernel.main: 1.0s\n",
            stderr="",
            checksum="1",
            kernel_timings={"main": 1.0},
        )

    def collect_artifacts(self, bench_path: Path) -> Artifacts:
        return Artifacts(benchmark_dir=str(bench_path))

    def calculate_timing(
        self,
        arts_result: RunResult,
        omp_result: RunResult,
        report_speedup: bool = True,
    ) -> TimingResult:
        return TimingResult(
            arts_time_sec=1.0,
            omp_time_sec=1.0,
            speedup=1.0,
            note="Same performance",
            speedup_basis="kernel",
        )

    def verify_correctness(
        self,
        arts_result: RunResult,
        omp_result: RunResult,
        tolerance: float,
    ) -> VerificationResult:
        return VerificationResult(True, arts_result.checksum, omp_result.checksum, tolerance, "")

    def get_size_params(self, bench_path: Path, size: str) -> str | None:
        return None

    def _cleanup_port(self) -> None:
        return None

    def _index_build_artifacts(
        self,
        artifacts_dir: Path,
        arts_cfg_used: Path | None = None,
    ) -> dict[str, str | None]:
        return {}

    def _create_run_files(self, **kwargs) -> BenchmarkRunFiles:
        return BenchmarkRunFiles(run_number=kwargs["run_number"])


class BenchmarkPipelineTest(unittest.TestCase):
    def test_negative_signal_return_codes_are_crashes(self) -> None:
        self.assertEqual(BenchmarkProcessRunner._status_from_exit_code(-11), Status.CRASH)
        self.assertEqual(BenchmarkProcessRunner._status_from_exit_code(-6), Status.CRASH)
        self.assertEqual(BenchmarkProcessRunner._status_from_exit_code(-8), Status.CRASH)

    def test_arts_run_receives_effective_arts_config_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "arts.cfg"
            cfg.write_text("[ARTS]\nlauncher=local\nnode_count=2\n")
            exe = root / "bench_arts"
            exe.write_text("#!/bin/sh\n")

            execution = BenchmarkExecutionContext(
                name="polybench/gemm",
                suite="polybench",
                size="small",
                bench_path=root,
                config=BenchmarkConfig(
                    arts_threads=4,
                    arts_nodes=2,
                    omp_threads=4,
                    launcher="local",
                ),
                effective_arts_cfg=cfg,
                desired_threads=4,
                desired_nodes=2,
                desired_launcher="local",
                actual_omp_threads=4,
                effective_cflags="",
                run_args=[],
                verify_tolerance=0.0,
            )
            plan = ConfigExecutionPlan(
                execution=execution,
                timeout=10,
                run_numbers=(1,),
                compile_args=None,
                perf_enabled=False,
                perf_interval=0.1,
                env_overrides={"CUSTOM": "1"},
            )
            host = _RunOnlyHost()
            executor = ConfigExecutionExecutor(host, plan)

            result = executor._run_arts(
                BuildResult(
                    status=Status.PASS,
                    duration_sec=0.01,
                    output="",
                    executable=str(exe),
                ),
                execution,
                BenchmarkRunFiles(run_number=1),
                ExecutionHooks(),
            )

            self.assertEqual(result.status, Status.PASS)
            self.assertIsNotNone(host.last_env)
            self.assertEqual(host.last_env["CUSTOM"], "1")
            self.assertEqual(host.last_env["ARTS_CONFIG"], str(cfg.resolve()))

    def test_repeated_paired_runs_alternate_launch_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "arts.cfg"
            cfg.write_text("[ARTS]\nlauncher=local\nnode_count=1\n")
            execution = BenchmarkExecutionContext(
                name="polybench/gemm",
                suite="polybench",
                size="small",
                bench_path=root,
                config=BenchmarkConfig(
                    arts_threads=64,
                    arts_nodes=1,
                    omp_threads=64,
                    launcher="local",
                ),
                effective_arts_cfg=cfg,
                desired_threads=64,
                desired_nodes=1,
                desired_launcher="local",
                actual_omp_threads=64,
                effective_cflags="",
                run_args=[],
                verify_tolerance=0.0,
            )
            plan = ConfigExecutionPlan(
                execution=execution,
                timeout=10,
                run_numbers=(1, 2, 3),
                compile_args=None,
                perf_enabled=False,
                perf_interval=0.1,
                env_overrides={},
            )
            host = _ExecutionOrderHost(root)

            results = ConfigExecutionExecutor(host, plan).execute()

            self.assertEqual([r.run_number for r in results], [1, 2, 3])
            self.assertEqual(
                host.run_calls,
                ["arts", "omp", "omp", "arts", "arts", "omp"],
            )

    def test_warmup_runs_execute_but_are_not_returned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "arts.cfg"
            cfg.write_text("[ARTS]\nlauncher=local\nnode_count=1\n")
            execution = BenchmarkExecutionContext(
                name="polybench/gemm",
                suite="polybench",
                size="small",
                bench_path=root,
                config=BenchmarkConfig(
                    arts_threads=64,
                    arts_nodes=1,
                    omp_threads=64,
                    launcher="local",
                ),
                effective_arts_cfg=cfg,
                desired_threads=64,
                desired_nodes=1,
                desired_launcher="local",
                actual_omp_threads=64,
                effective_cflags="",
                run_args=[],
                verify_tolerance=0.0,
            )
            plan = ConfigExecutionPlan(
                execution=execution,
                timeout=10,
                run_numbers=(1, 2, 3, 4),
                compile_args=None,
                perf_enabled=False,
                perf_interval=0.1,
                warmup_runs=2,
                env_overrides={},
            )
            host = _ExecutionOrderHost(root)

            results = ConfigExecutionExecutor(host, plan).execute()

            self.assertEqual([r.run_number for r in results], [3, 4])
            self.assertEqual(len(host.run_calls), 8)

    def test_host_openmp_fallback_arts_run_uses_runtime_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "arts.cfg"
            cfg.write_text(
                f"[ARTS]\n{KEY_WORKER_THREADS}=64\n{KEY_PIN}=1\nlauncher=local\n"
            )
            exe = root / "bench_arts"
            exe.write_text("#!/bin/sh\n")
            (root / "bench-arts.ll").write_text(
                "declare void @carts_benchmarks_mark_host_openmp()\n"
            )
            run_dir = root / "run_1"
            counter_dir = run_dir / "counters"

            execution = BenchmarkExecutionContext(
                name="polybench/jacobi-2d",
                suite="polybench",
                size="large",
                bench_path=root,
                config=BenchmarkConfig(
                    arts_threads=64,
                    arts_nodes=1,
                    omp_threads=64,
                    launcher="local",
                ),
                effective_arts_cfg=cfg,
                desired_threads=64,
                desired_nodes=1,
                desired_launcher="local",
                actual_omp_threads=64,
                effective_cflags="",
                run_args=[],
                verify_tolerance=0.0,
            )
            plan = ConfigExecutionPlan(
                execution=execution,
                timeout=10,
                run_numbers=(1,),
                compile_args=None,
                perf_enabled=False,
                perf_interval=0.1,
                env_overrides={},
            )
            host = _RunOnlyHost()
            executor = ConfigExecutionExecutor(host, plan)

            result = executor._run_arts(
                BuildResult(
                    status=Status.PASS,
                    duration_sec=0.01,
                    output="",
                    executable=str(exe),
                ),
                execution,
                BenchmarkRunFiles(
                    run_number=1,
                    run_dir=run_dir,
                    counter_dir=counter_dir,
                ),
                ExecutionHooks(),
            )

            self.assertEqual(result.status, Status.PASS)
            self.assertIsNotNone(host.last_env)
            self.assertEqual(host.last_env["OMP_WAIT_POLICY"], "ACTIVE")
            self.assertEqual(host.last_env["OMP_NUM_THREADS"], "64")
            self.assertNotIn("KMP_BLOCKTIME", host.last_env)
            self.assertNotIn("KMP_AFFINITY", host.last_env)
            runtime_cfg = Path(host.last_env["ARTS_CONFIG"])
            self.assertEqual(runtime_cfg, (run_dir / "arts.cfg").resolve())
            parsed = parse_arts_cfg(runtime_cfg)
            self.assertEqual(parsed[KEY_WORKER_THREADS], "1")
            self.assertEqual(parsed[KEY_PIN], "0")
            self.assertEqual(parsed["counter_folder"], str(counter_dir))

    def test_host_openmp_fallback_run_config_reports_launched_arts_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "build_arts.cfg"
            cfg.write_text(
                f"[ARTS]\n{KEY_WORKER_THREADS}=64\n{KEY_PIN}=1\nlauncher=local\n"
            )
            exe = root / "bench_arts"
            exe.write_text("#!/bin/sh\n")
            (root / "bench-arts.ll").write_text(
                "declare void @carts_benchmarks_mark_host_openmp()\n"
            )
            am = ArtifactManager(root / "results", "ts")
            bench = root / "bench"
            bench.mkdir()
            config = BenchmarkConfig(
                arts_threads=64,
                arts_nodes=1,
                omp_threads=64,
                launcher="local",
            )
            run_dir = am.get_run_dir("polybench/jacobi-2d", config, 1)

            execution = BenchmarkExecutionContext(
                name="polybench/jacobi-2d",
                suite="polybench",
                size="large",
                bench_path=bench,
                config=config,
                effective_arts_cfg=cfg,
                desired_threads=64,
                desired_nodes=1,
                desired_launcher="local",
                actual_omp_threads=64,
                effective_cflags="",
                run_args=[],
                verify_tolerance=0.0,
                artifact_paths={"arts_config": str(cfg)},
            )
            plan = ConfigExecutionPlan(
                execution=execution,
                timeout=10,
                run_numbers=(1,),
                compile_args=None,
                perf_enabled=False,
                perf_interval=0.1,
                env_overrides={},
            )
            host = _ArtifactHost(am)
            executor = ConfigExecutionExecutor(host, plan)

            artifacts = executor._collect_artifacts(
                execution=execution,
                run_files=BenchmarkRunFiles(run_number=1, run_dir=run_dir),
                run_number=1,
                perf_enabled=False,
                build_arts=BuildResult(
                    status=Status.PASS,
                    duration_sec=0.01,
                    output="",
                    executable=str(exe),
                ),
            )

            persisted_cfg = run_dir / "arts.cfg"
            run_config = json.loads((run_dir / "run_config.json").read_text())
            parsed = parse_arts_cfg(persisted_cfg)
            self.assertEqual(
                run_config["env_overrides"]["ARTS_CONFIG"],
                str(persisted_cfg.resolve()),
            )
            self.assertEqual(artifacts.arts_config, str(persisted_cfg.resolve()))
            self.assertEqual(
                run_config["runtime_arts_overrides"][KEY_WORKER_THREADS], "1"
            )
            self.assertEqual(run_config["runtime_arts_overrides"][KEY_PIN], "0")
            self.assertEqual(parsed[KEY_WORKER_THREADS], "1")
            self.assertEqual(parsed[KEY_PIN], "0")

    def test_local_generated_config_synthesizes_matching_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "arts_slurm_like.cfg"
            template.write_text(
                "\n".join(
                    [
                        "[ARTS]",
                        "worker_threads=64",
                        "launcher=slurm",
                        "node_count=1",
                        "default_ports=34739",
                    ]
                )
                + "\n"
            )

            generated = generate_arts_config(
                template,
                threads=64,
                launcher="local",
                nodes_override=2,
                benchmark_name="unit/local",
            )
            try:
                parsed = parse_arts_cfg(generated)
                self.assertEqual(parsed["launcher"], "local")
                self.assertEqual(parsed["node_count"], "2")
                self.assertEqual(
                    parsed["nodes"],
                    "localhost:34739,localhost:34740",
                )
                self.assertEqual(parsed["master_node"], "localhost:34739")
                self.assertEqual(parsed["port_count"], "1")
                self.assertEqual(parsed["default_ports"], "34739")
            finally:
                generated.unlink(missing_ok=True)

    def test_local_generated_config_uses_tcp_for_multinode_loopback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "arts.cfg"
            template.write_text(
                "\n".join(
                    [
                        "[ARTS]",
                        "worker_threads=1",
                        "launcher=local",
                        "node_count=1",
                        f"{KEY_PROTOCOL}={PROTOCOL_TCP}",
                    ]
                )
                + "\n"
            )

            rdma_cfg = generate_arts_config(
                template,
                threads=4,
                launcher="local",
                nodes_override=2,
                benchmark_name="unit/protocol",
                rdma=True,
            )
            tcp_cfg = generate_arts_config(
                template,
                threads=4,
                launcher="local",
                nodes_override=2,
                benchmark_name="unit/protocol",
                rdma=False,
            )
            single_cfg = generate_arts_config(
                template,
                threads=4,
                launcher="local",
                nodes_override=1,
                benchmark_name="unit/protocol",
                rdma=True,
            )
            try:
                self.assertEqual(rdma_cfg, tcp_cfg)
                self.assertEqual(parse_arts_cfg(rdma_cfg)[KEY_PROTOCOL], PROTOCOL_TCP)
                self.assertEqual(parse_arts_cfg(tcp_cfg)[KEY_PROTOCOL], PROTOCOL_TCP)
                self.assertEqual(parse_arts_cfg(single_cfg)[KEY_PROTOCOL], PROTOCOL_TCP)
            finally:
                rdma_cfg.unlink(missing_ok=True)
                tcp_cfg.unlink(missing_ok=True)
                single_cfg.unlink(missing_ok=True)

    def test_nonlocal_multinode_launcher_keeps_rdma_protocol(self) -> None:
        self.assertEqual(protocol_for_launcher(True, 2, "slurm"), PROTOCOL_RDMA)
        self.assertEqual(protocol_for_launcher(True, 2, "ssh"), PROTOCOL_RDMA)
        self.assertEqual(protocol_for_launcher(False, 2, "slurm"), PROTOCOL_TCP)
        self.assertEqual(protocol_for_launcher(True, 1, "slurm"), PROTOCOL_TCP)

    def test_nonlocal_generated_rdma_config_uses_production_single_port(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "arts.cfg"
            template.write_text(
                "\n".join(
                    [
                        "[ARTS]",
                        "worker_threads=1",
                        "launcher=slurm",
                        "node_count=1",
                        "default_ports=34739",
                        f"{KEY_PROTOCOL}={PROTOCOL_TCP}",
                    ]
                )
                + "\n"
            )

            rdma_cfg = generate_arts_config(
                template,
                threads=4,
                launcher="slurm",
                nodes_override=2,
                benchmark_name="unit/nonlocal-rdma",
                rdma=True,
            )
            tcp_cfg = generate_arts_config(
                template,
                threads=4,
                launcher="slurm",
                nodes_override=2,
                benchmark_name="unit/nonlocal-tcp",
                rdma=False,
            )
            try:
                rdma_values = parse_arts_cfg(rdma_cfg)
                tcp_values = parse_arts_cfg(tcp_cfg)
                self.assertEqual(rdma_values[KEY_PROTOCOL], PROTOCOL_RDMA)
                self.assertEqual(rdma_values[KEY_PORT_COUNT], "1")
                self.assertEqual(rdma_values[KEY_DEFAULT_PORTS], "34739")
                self.assertEqual(tcp_values[KEY_PROTOCOL], PROTOCOL_TCP)
                self.assertEqual(tcp_values[KEY_PORT_COUNT], "2")
                self.assertEqual(tcp_values[KEY_DEFAULT_PORTS], "34739,34740")
            finally:
                rdma_cfg.unlink(missing_ok=True)
                tcp_cfg.unlink(missing_ok=True)

    def test_compile_args_reject_removed_distributed_flag(self) -> None:
        with self.assertRaises(ValueError):
            compile_args_for_node_count("--distributed-db", 1)
        with self.assertRaises(ValueError):
            compile_args_for_node_count("--distributed-db --foo=bar", 2)
        self.assertEqual(
            compile_args_for_node_count("--no-distributed-db --foo=bar", 1),
            "--no-distributed-db --foo=bar",
        )
        self.assertEqual(
            compile_args_for_node_count("--foo=bar", 2),
            "--foo=bar",
        )

    def test_size_build_passes_user_cflags_as_extra_cflags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench = root / "suite" / "unit"
            bench.mkdir(parents=True)
            (bench / "Makefile").write_text("EXAMPLE_NAME := unit\n")
            cfg = root / "arts.cfg"
            cfg.write_text("[ARTS]\nlauncher=local\nnode_count=1\n")
            build_dir = root / "artifacts"
            observed_cmd: list[str] = []

            def fake_run(cmd, **kwargs):
                del kwargs
                observed_cmd[:] = list(cmd)
                for token in cmd:
                    if token.startswith("ARTS_BINARY="):
                        exe = Path(token.split("=", 1)[1])
                        exe.write_text("#!/bin/sh\n")
                        exe.chmod(0o755)
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch("runner.get_carts_dir", return_value=REPO_ROOT):
                runner = BenchmarkRunner(Console(file=StringIO()))
            runner.benchmarks_dir = root
            with mock.patch("runner.subprocess.run", side_effect=fake_run), \
                 mock.patch("runner._validate_embedded_arts_cfg", return_value=None):
                result = runner.build_benchmark(
                    "suite/unit",
                    "extralarge",
                    arts_config=cfg,
                    cflags="-DNREPS=1",
                    compile_args="--no-distributed-db",
                    build_output_dir=build_dir,
                )

            self.assertEqual(result.status, Status.PASS)
            self.assertIn("EXTRA_CFLAGS=-DNREPS=1", observed_cmd)
            self.assertNotIn("CFLAGS=-DNREPS=1", observed_cmd)


if __name__ == "__main__":
    unittest.main()
