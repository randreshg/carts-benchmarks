from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "scripts"
TOOLS_DIR = REPO_ROOT / "tools"

sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from arts_config import parse_arts_cfg  # noqa: E402
from execution import (  # noqa: E402
    BenchmarkExecutionContext,
    BenchmarkProcessRunner,
    BenchmarkRunFiles,
)
from models import BenchmarkConfig, BuildResult, RunResult, Status  # noqa: E402
from pipeline import ConfigExecutionExecutor, ConfigExecutionPlan, ExecutionHooks  # noqa: E402
from runner import generate_arts_config  # noqa: E402


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
            finally:
                generated.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
