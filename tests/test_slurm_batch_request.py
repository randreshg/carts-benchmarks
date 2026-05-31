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

import runner  # noqa: E402


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
    def test_slurm_exclude_with_launcher_appends_submit_host(self) -> None:
        exclude_nodes = runner._slurm_exclude_with_launcher(
            "d16u43,h08u31",
            nodelist=None,
            launcher_host="a08u25.cluster",
        )
        self.assertEqual(exclude_nodes, "d16u43,h08u31,a08u25")

    def test_slurm_exclude_with_launcher_preserves_explicit_nodelist(self) -> None:
        exclude_nodes = runner._slurm_exclude_with_launcher(
            "d16u43",
            nodelist="a08u25",
            launcher_host="a08u25",
        )
        self.assertEqual(exclude_nodes, "d16u43")

    def test_run_step_slurm_excludes_launcher_host(self) -> None:
        calls = []

        def capture_batch(**kwargs):
            calls.append(kwargs)

        with mock.patch.object(runner, "_execute_slurm_batch", capture_batch), \
                mock.patch.object(runner.platform, "node", lambda: "a08u25"):
            runner._run_step_slurm(
                bench_list=["polybench/gemm"],
                size="small",
                node_counts=[2],
                runs=1,
                partition="debug",
                timeout=30,
                time_limit="00:10:00",
                arts_config=None,
                threads_list=[64],
                results_dir=Path("./results"),
                verbose=False,
                cflags=None,
                compile_args=None,
                exclude_nodes="d16u43,h08u31",
                nodelist=None,
                perf=False,
                perf_interval=0.1,
                dry_run=True,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["exclude_nodes"], "d16u43,h08u31,a08u25")

    def test_execute_slurm_batch_passes_max_jobs_to_request(self) -> None:
        profile = Path("configs/profiles/profile-comm.cfg")
        with mock.patch.object(runner, "BenchmarkRunner", _FakeRunner), mock.patch.object(
            runner, "SlurmBatchExecutor", _CapturingExecutor
        ), mock.patch.object(runner, "require_slurm_commands", lambda *args, **kwargs: None), mock.patch.object(
            runner, "find_invalid_benchmarks", lambda runner, requested: []
        ), mock.patch.object(
            runner, "find_multinode_disabled_benchmarks", lambda runner, bench_list: set()
        ), mock.patch.object(
            runner, "print_header", lambda *args, **kwargs: None
        ), mock.patch.object(
            runner, "print_info", lambda *args, **kwargs: None
        ), mock.patch.object(
            runner, "print_warning", lambda *args, **kwargs: None
        ):
            runner._execute_slurm_batch(
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
                profile=profile,
                perf=False,
                perf_interval=0.1,
                cpu_pinning="default",
                exclude_nodes=None,
                nodelist="b05u[01,07]",
                exclude=None,
                max_jobs=8,
                artifact_manager=None,
                step_name=None,
                report_steps=None,
                rdma=True,
                variant="arts",
            )

        self.assertIsNotNone(_CapturingExecutor.last_request)
        self.assertEqual(_CapturingExecutor.last_request.max_jobs, 8)
        self.assertEqual(_CapturingExecutor.last_request.nodelist, "b05u[01,07]")
        self.assertTrue(_CapturingExecutor.last_request.rdma)
        self.assertEqual(_CapturingExecutor.last_request.profile, profile)
        self.assertEqual(_CapturingExecutor.last_request.variant, "arts")

    def test_run_step_slurm_passes_profile_to_batch(self) -> None:
        profile = Path("configs/profiles/profile-comm.cfg")
        calls = []

        def capture_batch(**kwargs):
            calls.append(kwargs)

        with mock.patch.object(runner, "_execute_slurm_batch", capture_batch):
            runner._run_step_slurm(
                bench_list=["polybench/gemm"],
                size="small",
                node_counts=[2],
                runs=1,
                partition="debug",
                timeout=30,
                time_limit="00:10:00",
                arts_config=None,
                threads_list=[64],
                results_dir=Path("./results"),
                verbose=False,
                cflags=None,
                compile_args=None,
                exclude_nodes=None,
                nodelist=None,
                perf=False,
                perf_interval=0.1,
                profile=profile,
                rdma=True,
                variant="arts",
                dry_run=True,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["profile"], profile)
        self.assertTrue(calls[0]["rdma"])
        self.assertTrue(calls[0]["dry_run"])

    def test_run_slurm_resolved_step_uses_requested_profile(self) -> None:
        profile = Path("configs/profiles/profile-comm.cfg")
        captured = {}

        def capture_step(**kwargs):
            captured.update(kwargs)

        step_config = runner.ResolvedStepConfig(
            name="profiled",
            bench_list=["polybench/gemm"],
            profile_path=profile,
            requested_profile_path=profile,
            rdma=True,
            debug=0,
            should_rebuild_arts=True,
            threads_list=[64],
            node_counts=[2],
            timeout=30,
            runs=1,
            perf=False,
            perf_interval=0.1,
            size="small",
            cflags=None,
            compile_args=None,
            exclude_nodes=None,
            nodelist=None,
            arts_config=None,
            launcher="slurm",
        )

        request = runner.SlurmStepExecutionRequest(
            partition="debug",
            time_limit="00:10:00",
            results_dir=Path("./results"),
            verbose=False,
            quiet=False,
            artifact_manager=mock.Mock(),
            max_jobs=1,
            variant="arts",
        )

        with mock.patch.object(runner, "_run_step_slurm", capture_step):
            runner._run_slurm_resolved_step(
                step_config=step_config,
                request=request,
                report_steps=[],
            )

        self.assertEqual(captured["profile"], profile)
        self.assertTrue(captured["rdma"])

    def test_rebuild_arts_for_step_rebuilds_transport_mismatch(self) -> None:
        step_config = runner.ResolvedStepConfig(
            name="transport",
            bench_list=["polybench/gemm"],
            profile_path=Path("configs/profiles/profile-none.cfg"),
            requested_profile_path=None,
            rdma=True,
            debug=0,
            should_rebuild_arts=False,
            threads_list=[64],
            node_counts=[2],
            timeout=30,
            runs=1,
            perf=False,
            perf_interval=0.1,
            size="small",
            cflags=None,
            compile_args=None,
            exclude_nodes=None,
            nodelist=None,
            arts_config=None,
            launcher="slurm",
        )

        with mock.patch.object(runner, "arts_runtime_is_installed", lambda: True), \
                mock.patch.object(runner, "arts_runtime_uses_rdma", lambda: False), \
                mock.patch.object(runner, "_rebuild_arts") as rebuild, \
                mock.patch.object(runner, "print_warning") as print_warning:
            runner._rebuild_arts_for_step(step_config)

        rebuild.assert_called_once()
        self.assertTrue(rebuild.call_args.kwargs["rdma"])
        self.assertIn(
            "ARTS runtime transport is tcp; rebuilding for requested rdma",
            print_warning.call_args.args[0],
        )

    def test_rebuild_arts_for_step_treats_single_node_as_tcp(self) -> None:
        step_config = runner.ResolvedStepConfig(
            name="single",
            bench_list=["polybench/gemm"],
            profile_path=Path("configs/profiles/profile-none.cfg"),
            requested_profile_path=None,
            rdma=True,
            debug=0,
            should_rebuild_arts=False,
            threads_list=[64],
            node_counts=[1],
            timeout=30,
            runs=1,
            perf=False,
            perf_interval=0.1,
            size="small",
            cflags=None,
            compile_args=None,
            exclude_nodes=None,
            nodelist=None,
            arts_config=None,
            launcher=None,
        )

        with mock.patch.object(runner, "arts_runtime_is_installed", lambda: True), \
                mock.patch.object(runner, "arts_runtime_uses_rdma", lambda: False), \
                mock.patch.object(runner, "_rebuild_arts") as rebuild:
            runner._rebuild_arts_for_step(step_config)

        rebuild.assert_not_called()


if __name__ == "__main__":
    unittest.main()
