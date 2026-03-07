from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "scripts"
TOOLS_DIR = REPO_ROOT / "tools"

sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_runner  # noqa: E402


class _FakeRunner:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def discover_benchmarks(self, suite):
        del suite
        return ["polybench/gemm"]


class _CapturingExecutor:
    last_request = None

    def __init__(self, runner, request, deps) -> None:
        del runner, deps
        type(self).last_request = request

    def execute(self) -> None:
        return None


class SlurmBatchRequestConstructionTest(unittest.TestCase):
    def test_execute_slurm_batch_passes_max_jobs_to_request(self) -> None:
        with mock.patch.object(benchmark_runner, "BenchmarkRunner", _FakeRunner), mock.patch.object(
            benchmark_runner, "SlurmBatchExecutor", _CapturingExecutor
        ), mock.patch.object(benchmark_runner, "require_slurm_commands", lambda dry_run: None), mock.patch.object(
            benchmark_runner, "find_invalid_benchmarks", lambda runner, requested: []
        ), mock.patch.object(
            benchmark_runner, "find_multinode_disabled_benchmarks", lambda runner, bench_list: set()
        ), mock.patch.object(
            benchmark_runner, "print_header", lambda *args, **kwargs: None
        ), mock.patch.object(
            benchmark_runner, "print_info", lambda *args, **kwargs: None
        ), mock.patch.object(
            benchmark_runner, "print_warning", lambda *args, **kwargs: None
        ):
            benchmark_runner._execute_slurm_batch(
                benchmarks=["polybench/gemm"],
                nodes="1",
                size="small",
                runs=1,
                partition="debug",
                time_limit="00:10:00",
                account=None,
                arts_config=None,
                threads=4,
                output_dir=Path("./results"),
                suite=None,
                dry_run=True,
                no_build=False,
                verbose=False,
                cflags=None,
                compile_args=None,
                gdb=False,
                profile=None,
                perf=False,
                perf_interval=0.1,
                exclude_nodes=None,
                exclude=None,
                max_jobs=8,
                artifact_manager=None,
                step_name=None,
                report_steps=None,
            )

        self.assertIsNotNone(_CapturingExecutor.last_request)
        self.assertEqual(_CapturingExecutor.last_request.max_jobs, 8)


if __name__ == "__main__":
    unittest.main()
