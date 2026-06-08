from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import types
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "scripts"
TOOLS_DIR = REPO_ROOT / "tools"

sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from arts_config import (  # noqa: E402
    KEY_PIN,
    KEY_PROTOCOL,
    KEY_WORKER_THREADS,
    PROTOCOL_TCP,
    parse_arts_cfg,
)
from artifacts import ArtifactManager  # noqa: E402
from models import BenchmarkConfig, BuildResult, ReferenceChecksum, Status  # noqa: E402
from slurm.experiment import (  # noqa: E402
    ARTS_RUNTIME_MODE_HOST_OPENMP,
    ARTS_RUNTIME_MODE_TASK,
    SlurmBatchExecutor,
    SlurmBatchRequest,
    SlurmExecutorDependencies,
    compute_prebuild_worker_count,
    count_total_slurm_jobs,
    find_multinode_disabled_benchmarks,
    format_node_counts_display,
    infer_arts_runtime_mode,
    load_existing_job_statuses,
    merge_result_rows,
    require_slurm_commands,
    validate_requested_node_counts,
)


@contextmanager
def isolated_carts_env(carts_root: Path):
    """Run fake-checkout tests without inherited Dekk CARTS path overrides."""
    with mock.patch.dict(
        os.environ,
        {
            "CARTS_DIR": str(carts_root),
            "CARTS_CONFIG": str(carts_root / "carts.config"),
        },
    ):
        for key in ("CARTS_HOME", "CARTS_BUILD_ROOT", "CARTS_INSTALL_ROOT"):
            os.environ.pop(key, None)
        yield


class _FakeHost:
    def __init__(self, benchmarks_dir: Path, runtime_ir_text: str = "") -> None:
        self.benchmarks_dir = benchmarks_dir
        self.artifact_manager = None
        self.reference_calls = 0
        self.build_calls = []
        self.runtime_ir_text = runtime_ir_text

    def get_executable_paths(self, bench_path: Path) -> tuple[Path, Path]:
        return bench_path / "bench_arts", bench_path / "bench_omp"

    def build_benchmark(
        self,
        name: str,
        size: str,
        variant: str = "arts",
        arts_config: Path | None = None,
        cflags: str = "",
        compile_args: str | None = None,
        build_output_dir: Path | None = None,
    ) -> BuildResult:
        del size, cflags
        self.build_calls.append(
            {
                "name": name,
                "variant": variant,
                "arts_config": arts_config,
                "compile_args": compile_args,
            }
        )
        assert build_output_dir is not None
        executable = build_output_dir / ("bench_omp" if variant == "openmp" else "bench_arts")
        executable.write_text("#!/bin/sh\n")
        if variant == "arts" and self.runtime_ir_text:
            (build_output_dir / "bench-arts.ll").write_text(self.runtime_ir_text)
        return BuildResult(Status.PASS, 0.0, "", str(executable))

    def ensure_omp_reference(
        self,
        name: str,
        size: str,
        cflags: str,
        omp_threads: int,
        timeout: int,
    ) -> ReferenceChecksum:
        del name, size, cflags, timeout
        self.reference_calls += 1
        return ReferenceChecksum(
            status=Status.PASS,
            checksum="1.0",
            omp_threads=omp_threads,
            note="ok",
            source="fake",
        )


class SlurmExperimentHelpersTest(unittest.TestCase):
    def test_require_slurm_commands_reports_failed_submit_probe(self) -> None:
        failed_submit = types.SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=(
                "Batch job submission failed: I/O error writing "
                "script/environment to file"
            ),
        )

        with mock.patch(
            "slurm.experiment.shutil.which", return_value="/usr/bin/tool"
        ), mock.patch(
            "slurm.experiment.subprocess.run", return_value=failed_submit
        ):
            with self.assertRaisesRegex(ValueError, "sbatch cannot create"):
                require_slurm_commands(dry_run=False)

    def test_require_slurm_commands_can_skip_submit_probe(self) -> None:
        with mock.patch(
            "slurm.experiment.shutil.which", return_value="/usr/bin/tool"
        ), mock.patch(
            "slurm.experiment.subprocess.run"
        ) as run, mock.patch.dict(
            os.environ, {"CARTS_SLURM_SUBMIT_PROBE": "0"}
        ):
            require_slurm_commands(dry_run=False)

        run.assert_not_called()

    def test_require_slurm_commands_probe_uses_requested_partition_and_account(self) -> None:
        submitted = types.SimpleNamespace(
            returncode=0,
            stdout="12345\n",
            stderr="",
        )
        cancelled = types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with mock.patch(
            "slurm.experiment.shutil.which", return_value="/usr/bin/tool"
        ), mock.patch(
            "slurm.experiment.subprocess.run",
            side_effect=[submitted, cancelled],
        ) as run:
            require_slurm_commands(
                dry_run=False,
                partition="mi300x",
                account="acct",
            )

        submit_cmd = run.call_args_list[0].args[0]
        self.assertIn("--partition=mi300x", submit_cmd)
        self.assertIn("--account=acct", submit_cmd)

    def test_validate_requested_node_counts_skips_partition_query_failure_by_default(self) -> None:
        with mock.patch(
            "slurm.experiment._run_slurm_query",
            side_effect=ValueError("slurm_load_partitions timed out"),
        ), mock.patch("slurm.experiment.print_warning") as warning:
            validate_requested_node_counts([2], partition=None)

        warning.assert_called_once()
        self.assertIn("Skipping SLURM node-count validation", warning.call_args[0][0])

    def test_validate_requested_node_counts_strict_partition_query_failure_raises(self) -> None:
        with mock.patch(
            "slurm.experiment._run_slurm_query",
            side_effect=ValueError("slurm_load_partitions timed out"),
        ), mock.patch.dict(
            os.environ, {"CARTS_SLURM_NODE_COUNT_VALIDATE_STRICT": "1"}
        ):
            with self.assertRaisesRegex(ValueError, "slurm_load_partitions timed out"):
                validate_requested_node_counts([2], partition=None)

    def test_streaming_executor_runs_slurm_preflight_before_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_root = root / "benchmarks"
            (bench_root / "suite" / "a").mkdir(parents=True)
            base_cfg = root / "arts.cfg"
            base_cfg.write_text(f"[ARTS]\nworker_threads=1\n{KEY_PROTOCOL}={PROTOCOL_TCP}\n")
            carts_root = root / "carts"
            deps = SlurmExecutorDependencies(
                resolve_effective_arts_config=lambda bench_path, explicit: base_cfg,
                parse_time_limit_seconds=lambda spec: 60,
                get_carts_dir=lambda: carts_root,
                get_benchmarks_dir=lambda: bench_root,
                step_name_to_token=lambda step: step or "default",
            )
            am = ArtifactManager(root / "results", "ts")
            request = SlurmBatchRequest(
                bench_list=["suite/a"],
                node_counts=[1],
                size="small",
                runs=1,
                timeout=30,
                partition=None,
                time_limit="00:01:00",
                account=None,
                explicit_arts_config=base_cfg,
                threads=1,
                output_dir=root / "results",
                max_jobs=1,
                dry_run=False,
                no_build=False,
                verbose=False,
                cflags=None,
                compile_args=None,
                gdb=False,
                profile=None,
                perf=False,
                perf_interval=0.1,
                cpu_pinning="default",
                exclude_nodes=None,
                nodelist=None,
                rdma=False,
                artifact_manager=am,
                step_name="default",
                report_steps=None,
                command_str="test",
            )

            with mock.patch(
                "slurm.experiment.require_slurm_commands",
                side_effect=ValueError("probe failed"),
            ) as preflight:
                with self.assertRaisesRegex(ValueError, "probe failed"):
                    SlurmBatchExecutor(_FakeHost(bench_root), request, deps).execute()

            preflight.assert_called_once_with(
                False,
                partition=None,
                account=None,
            )
            self.assertFalse((am.experiment_dir / "scripts").exists())

    def test_format_node_counts_display(self) -> None:
        self.assertEqual(format_node_counts_display([4]), "4")
        self.assertEqual(format_node_counts_display([1, 2, 4]), "1, 2, 4")
        self.assertEqual(
            format_node_counts_display([1, 2, 4, 8, 16, 32]),
            "1-32 (6 values)",
        )

    def test_count_total_slurm_jobs(self) -> None:
        total = count_total_slurm_jobs(
            ["a", "b", "c"],
            [1, 2, 4],
            runs=2,
            multinode_disabled={"c"},
        )
        self.assertEqual(total, 14)

    def test_compute_prebuild_worker_count_uses_host_cpus(self) -> None:
        self.assertEqual(
            compute_prebuild_worker_count(
                host_cpus=128,
                requested_threads=64,
                benchmark_count=23,
            ),
            23,
        )
        self.assertEqual(
            compute_prebuild_worker_count(
                host_cpus=16,
                requested_threads=64,
                benchmark_count=23,
            ),
            16,
        )
        self.assertEqual(
            compute_prebuild_worker_count(
                host_cpus=128,
                requested_threads=1,
                benchmark_count=23,
            ),
            23,
        )

    def test_find_multinode_disabled_benchmarks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "suite" / "a").mkdir(parents=True)
            (root / "suite" / "b").mkdir(parents=True)
            (root / "suite" / "b" / ".disable-multinode").write_text("")

            disabled = find_multinode_disabled_benchmarks(
                _FakeHost(root),
                ["suite/a", "suite/b"],
            )
            self.assertEqual(disabled, {"suite/b"})

    def test_infer_arts_runtime_mode_from_generated_ir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / "gemm_arts"
            exe.write_text("#!/bin/sh\n")
            (root / "gemm-arts.ll").write_text(
                "declare i64 @arts_initialize_and_start_epoch(i64, i32)\n"
            )

            mode, source = infer_arts_runtime_mode(exe)

            self.assertEqual(mode, ARTS_RUNTIME_MODE_TASK)
            self.assertTrue(source.endswith("gemm-arts.ll"))

    def test_infer_arts_runtime_mode_marks_host_openmp_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / "jacobi-for_arts"
            exe.write_text("#!/bin/sh\n")
            (root / "jacobi-for-arts.ll").write_text(
                "declare void @carts_benchmarks_mark_host_openmp()\n"
            )

            mode, _ = infer_arts_runtime_mode(exe)

            self.assertEqual(mode, ARTS_RUNTIME_MODE_HOST_OPENMP)

    def test_host_openmp_fallback_uses_runtime_isolated_arts_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_root = root / "benchmarks"
            (bench_root / "suite" / "a").mkdir(parents=True)
            base_cfg = root / "arts.cfg"
            base_cfg.write_text(
                f"[ARTS]\n{KEY_WORKER_THREADS}=64\n{KEY_PIN}=1\n{KEY_PROTOCOL}={PROTOCOL_TCP}\n"
            )
            carts_root = root / "carts"
            runtime_lib_dir = carts_root / ".install" / "arts" / "lib"
            runtime_lib_dir.mkdir(parents=True)
            (runtime_lib_dir / "libarts.so.2").write_text("fake arts runtime\n")
            (carts_root / ".install" / "carts" / "lib").mkdir(parents=True)
            (carts_root / ".install" / "llvm" / "lib").mkdir(parents=True)
            host = _FakeHost(
                bench_root,
                runtime_ir_text="declare void @carts_benchmarks_mark_host_openmp()\n",
            )
            am = ArtifactManager(root / "results", "ts")
            deps = SlurmExecutorDependencies(
                resolve_effective_arts_config=lambda bench_path, explicit: base_cfg,
                parse_time_limit_seconds=lambda spec: 60,
                get_carts_dir=lambda: carts_root,
                get_benchmarks_dir=lambda: bench_root,
                step_name_to_token=lambda step: step,
            )
            request = SlurmBatchRequest(
                bench_list=["suite/a"],
                node_counts=[1],
                size="large",
                runs=1,
                timeout=30,
                partition=None,
                time_limit="00:01:00",
                account=None,
                explicit_arts_config=base_cfg,
                threads=64,
                output_dir=root / "results",
                max_jobs=1,
                dry_run=True,
                no_build=False,
                verbose=False,
                cflags=None,
                compile_args=None,
                gdb=False,
                profile=None,
                perf=False,
                perf_interval=0.1,
                cpu_pinning="default",
                exclude_nodes=None,
                nodelist=None,
                rdma=False,
                artifact_manager=am,
                step_name="single",
                report_steps=None,
                command_str="test",
            )

            SlurmBatchExecutor(host, request, deps).execute()

            run_root = am.experiment_dir / "single" / "suite" / "a" / "64t_1n"
            run_config = json.loads((run_root / "run_1" / "run_config.json").read_text())
            run_cfg = parse_arts_cfg(run_root / "run_1" / "arts.cfg")
            build_cfg = parse_arts_cfg(run_root / "artifacts" / "arts.cfg")
            script = (
                am.experiment_dir
                / "scripts"
                / "single__suite_a_64t_1n_run1.sbatch"
            ).read_text()

            self.assertEqual(run_config["arts_runtime_mode"], ARTS_RUNTIME_MODE_HOST_OPENMP)
            self.assertEqual(run_cfg[KEY_WORKER_THREADS], "1")
            self.assertEqual(run_cfg[KEY_PIN], "0")
            self.assertEqual(build_cfg[KEY_WORKER_THREADS], "64")
            self.assertEqual(build_cfg[KEY_PIN], "1")
            self.assertEqual(run_config["env_overrides"]["OMP_WAIT_POLICY"], "ACTIVE")
            self.assertNotIn("KMP_BLOCKTIME", run_config["env_overrides"])
            self.assertNotIn("KMP_AFFINITY", run_config["env_overrides"])
            self.assertEqual(
                run_config["env_overrides"]["ARTS_CONFIG"],
                str((run_root / "run_1" / "arts.cfg").resolve()),
            )
            self.assertEqual(
                run_config["runtime_arts_overrides"][KEY_WORKER_THREADS],
                "1",
            )
            self.assertEqual(run_config["runtime_arts_overrides"][KEY_PIN], "0")
            self.assertIn("arts.runtime-template.cfg", script)
            self.assertIn(
                f'export ARTS_CONFIG="{(run_root / "run_1" / "arts.cfg").resolve()}"',
                script,
            )
            self.assertIn("OMP_WAIT_POLICY=ACTIVE", script)
            self.assertNotIn("KMP_BLOCKTIME", script)
            self.assertNotIn("KMP_AFFINITY", script)
            self.assertIn("--cpus-per-task=68", script)

    def test_host_openmp_fallback_skips_multinode_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_root = root / "benchmarks"
            (bench_root / "suite" / "a").mkdir(parents=True)
            base_cfg = root / "arts.cfg"
            base_cfg.write_text(
                f"[ARTS]\n{KEY_WORKER_THREADS}=64\n{KEY_PIN}=1\n{KEY_PROTOCOL}={PROTOCOL_TCP}\n"
            )
            carts_root = root / "carts"
            for rel in (
                ".install/arts/lib/libarts.so.2",
                ".install/carts/lib/libcartstest.so",
                ".install/llvm/lib/libLLVM.so",
            ):
                path = carts_root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fake\n")
            host = _FakeHost(
                bench_root,
                runtime_ir_text="declare void @carts_benchmarks_mark_host_openmp()\n",
            )
            am = ArtifactManager(root / "results", "ts")
            deps = SlurmExecutorDependencies(
                resolve_effective_arts_config=lambda bench_path, explicit: base_cfg,
                parse_time_limit_seconds=lambda spec: 60,
                get_carts_dir=lambda: carts_root,
                get_benchmarks_dir=lambda: bench_root,
                step_name_to_token=lambda step: step,
            )
            request = SlurmBatchRequest(
                bench_list=["suite/a"],
                node_counts=[1, 2, 4],
                size="large",
                runs=1,
                timeout=30,
                partition=None,
                time_limit="00:01:00",
                account=None,
                explicit_arts_config=base_cfg,
                threads=64,
                output_dir=root / "results",
                max_jobs=1,
                dry_run=True,
                no_build=False,
                verbose=False,
                cflags=None,
                compile_args=None,
                gdb=False,
                profile=None,
                perf=False,
                perf_interval=0.1,
                cpu_pinning="default",
                exclude_nodes=None,
                nodelist=None,
                rdma=False,
                artifact_manager=am,
                step_name="scale",
                report_steps=None,
                command_str="test",
            )

            SlurmBatchExecutor(host, request, deps).execute()

            scripts = sorted((am.experiment_dir / "scripts").glob("*.sbatch"))
            self.assertEqual(
                [script.name for script in scripts],
                ["scale__suite_a_64t_1n_run1.sbatch"],
            )
            self.assertTrue(
                (
                    am.experiment_dir
                    / "scale"
                    / "suite"
                    / "a"
                    / "64t_1n"
                    / "run_1"
                    / "run_config.json"
                ).exists()
            )
            self.assertFalse(
                (
                    am.experiment_dir
                    / "scale"
                    / "suite"
                    / "a"
                    / "64t_2n"
                    / "run_1"
                    / "run_config.json"
                ).exists()
            )
            arts_builds = [
                call for call in host.build_calls if call["variant"] == "arts"
            ]
            self.assertEqual(len(arts_builds), 1)

    def test_merge_result_rows_prefers_first_seen_key(self) -> None:
        merged = merge_result_rows(
            existing_results=[
                {"slurm": {"job_id": "10"}, "status": "PASS"},
                {"artifacts": {"run_dir": "/tmp/run-1"}, "status": "FAIL"},
            ],
            submission_failure_results=[
                {"slurm": {"job_id": "10"}, "status": "FAIL"},
                {"artifacts": {"run_dir": "/tmp/run-2"}, "status": "FAIL"},
            ],
            current_results=[
                {"slurm": {"job_id": "11"}, "status": "PASS"},
                {"artifacts": {"run_dir": "/tmp/run-1"}, "status": "PASS"},
            ],
        )
        self.assertEqual(
            [row.get("status") for row in merged],
            ["PASS", "FAIL", "FAIL", "PASS"],
        )

    def test_load_existing_job_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "job_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "jobs": {
                            "123": {
                                "job_id": "123",
                                "benchmark_name": "polybench/gemm",
                                "run_number": 1,
                                "node_count": 8,
                                "state": "COMPLETED",
                                "run_dir": str(Path(tmp) / "run_1"),
                            }
                        }
                    }
                )
            )
            statuses = load_existing_job_statuses(manifest)
            self.assertEqual(list(statuses.keys()), ["123"])
            self.assertEqual(statuses["123"].state, "COMPLETED")
            self.assertEqual(statuses["123"].run_dir, Path(tmp) / "run_1")

    def test_artifact_manifest_paths_include_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            am = ArtifactManager(Path(tmp), "ts")
            am.set_phase("scale")
            config = BenchmarkConfig(
                arts_threads=64,
                arts_nodes=64,
                omp_threads=64,
                launcher="slurm",
            )

            am.record_run("polybench/gemm", config, 1, has_counters=True)

            manifest = am._manifest_benchmarks["polybench/gemm"]["configs"]
            self.assertIn("scale/64t_64n", manifest)
            self.assertEqual(
                manifest["scale/64t_64n"]["artifacts"],
                "scale/polybench/gemm/64t_64n/artifacts",
            )
            self.assertEqual(
                manifest["scale/64t_64n"]["runs"]["1"]["path"],
                "scale/polybench/gemm/64t_64n/run_1",
            )

    def test_streaming_dry_run_manifest_accumulates_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_root = root / "benchmarks"
            bench = bench_root / "suite" / "a"
            bench.mkdir(parents=True)
            base_cfg = root / "arts.cfg"
            base_cfg.write_text(f"[ARTS]\nworker_threads=1\n{KEY_PROTOCOL}={PROTOCOL_TCP}\n")
            carts_root = root / "carts"
            runtime_lib_dir = carts_root / ".install" / "arts" / "lib"
            runtime_lib_dir.mkdir(parents=True)
            (runtime_lib_dir / "libarts.so.2").write_text("fake arts runtime\n")
            carts_lib_dir = carts_root / ".install" / "carts" / "lib"
            carts_lib_dir.mkdir(parents=True)
            llvm_lib_dir = carts_root / ".install" / "llvm" / "lib"
            llvm_lib_dir.mkdir(parents=True)
            cache = carts_root / "build" / "arts" / "CMakeCache.txt"
            cache.parent.mkdir(parents=True)
            cache.write_text("ARTS_USE_RDMA:BOOL=OFF\n")
            am = ArtifactManager(root / "results", "ts")
            deps = SlurmExecutorDependencies(
                resolve_effective_arts_config=lambda bench_path, explicit: base_cfg,
                parse_time_limit_seconds=lambda spec: 60,
                get_carts_dir=lambda: carts_root,
                get_benchmarks_dir=lambda: bench_root,
                step_name_to_token=lambda step: step,
            )

            with isolated_carts_env(carts_root):
                for step_name in ("alpha", "beta"):
                    request = SlurmBatchRequest(
                        bench_list=["suite/a"],
                        node_counts=[1],
                        size="small",
                        runs=1,
                        timeout=30,
                        partition=None,
                        time_limit="00:01:00",
                        account=None,
                        explicit_arts_config=base_cfg,
                        threads=1,
                        output_dir=root / "results",
                        max_jobs=1,
                        dry_run=True,
                        no_build=False,
                        verbose=False,
                        cflags=None,
                        compile_args=None,
                        gdb=False,
                        profile=None,
                        perf=False,
                        perf_interval=0.1,
                        cpu_pinning="default",
                        exclude_nodes=None,
                        nodelist=None,
                        rdma=False,
                        artifact_manager=am,
                        step_name=step_name,
                        report_steps=None,
                        command_str="test",
                    )
                    SlurmBatchExecutor(_FakeHost(bench_root), request, deps).execute()

            manifest = json.loads((am.experiment_dir / "job_manifest.json").read_text())
            self.assertEqual(manifest["metadata"]["total_jobs"], 2)
            self.assertEqual(manifest["metadata"]["previous_step_jobs"], 1)
            self.assertEqual(manifest["metadata"]["current_step"], "beta")
            self.assertEqual(len(manifest["jobs"]), 2)
            run_dirs = {payload["run_dir"] for payload in manifest["jobs"].values()}
            self.assertEqual(
                run_dirs,
                {
                    str(am.experiment_dir / "alpha" / "suite" / "a" / "1t_1n" / "run_1"),
                    str(am.experiment_dir / "beta" / "suite" / "a" / "1t_1n" / "run_1"),
                },
            )
            alpha_run_config = json.loads(
                (
                    am.experiment_dir
                    / "alpha"
                    / "suite"
                    / "a"
                    / "1t_1n"
                    / "run_1"
                    / "run_config.json"
                ).read_text()
            )
            runtime_dirs = [
                Path(path) for path in alpha_run_config["runtime_library_dirs"]
            ]
            self.assertIn(runtime_lib_dir.resolve(), runtime_dirs)
            self.assertEqual(alpha_run_config["arts_transport"], PROTOCOL_TCP)

    def test_single_node_slurm_dry_run_uses_tcp_and_rejects_removed_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_root = root / "benchmarks"
            (bench_root / "suite" / "a").mkdir(parents=True)
            base_cfg = root / "arts.cfg"
            base_cfg.write_text(f"[ARTS]\nworker_threads=1\n{KEY_PROTOCOL}={PROTOCOL_TCP}\n")
            carts_root = root / "carts"
            (carts_root / ".install" / "arts" / "lib").mkdir(parents=True)
            (carts_root / ".install" / "arts" / "lib" / "libarts.so.2").write_text(
                "fake arts runtime\n"
            )
            (carts_root / ".install" / "carts" / "lib").mkdir(parents=True)
            (carts_root / ".install" / "llvm" / "lib").mkdir(parents=True)
            host = _FakeHost(bench_root)
            am = ArtifactManager(root / "results", "ts")
            deps = SlurmExecutorDependencies(
                resolve_effective_arts_config=lambda bench_path, explicit: base_cfg,
                parse_time_limit_seconds=lambda spec: 60,
                get_carts_dir=lambda: carts_root,
                get_benchmarks_dir=lambda: bench_root,
                step_name_to_token=lambda step: step,
            )
            request = SlurmBatchRequest(
                bench_list=["suite/a"],
                node_counts=[1],
                size="small",
                runs=1,
                timeout=30,
                partition=None,
                time_limit="00:01:00",
                account=None,
                explicit_arts_config=base_cfg,
                threads=1,
                output_dir=root / "results",
                max_jobs=1,
                dry_run=True,
                no_build=False,
                verbose=False,
                cflags=None,
                compile_args="--distributed-db",
                gdb=False,
                profile=None,
                perf=False,
                perf_interval=0.1,
                cpu_pinning="default",
                exclude_nodes=None,
                nodelist=None,
                rdma=True,
                artifact_manager=am,
                step_name="single",
                report_steps=None,
                command_str="test",
            )

            with self.assertRaises(ValueError):
                SlurmBatchExecutor(host, request, deps).execute()

    def test_distributed_default_dry_run_is_node_specific_without_benchmark_special_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_root = root / "benchmarks"
            (bench_root / "suite" / "writer-reader").mkdir(parents=True)
            base_cfg = root / "arts.cfg"
            base_cfg.write_text(f"[ARTS]\nworker_threads=1\n{KEY_PROTOCOL}={PROTOCOL_TCP}\n")
            carts_root = root / "carts"
            (carts_root / ".install" / "arts" / "lib").mkdir(parents=True)
            (carts_root / ".install" / "arts" / "lib" / "libarts.so.2").write_text(
                "fake arts runtime\n"
            )
            (carts_root / ".install" / "carts" / "lib").mkdir(parents=True)
            (carts_root / ".install" / "llvm" / "lib").mkdir(parents=True)
            host = _FakeHost(bench_root)
            am = ArtifactManager(root / "results", "ts")
            deps = SlurmExecutorDependencies(
                resolve_effective_arts_config=lambda bench_path, explicit: base_cfg,
                parse_time_limit_seconds=lambda spec: 60,
                get_carts_dir=lambda: carts_root,
                get_benchmarks_dir=lambda: bench_root,
                step_name_to_token=lambda step: step,
            )
            request = SlurmBatchRequest(
                bench_list=["suite/writer-reader"],
                node_counts=[1, 2],
                size="small",
                runs=1,
                timeout=30,
                partition=None,
                time_limit="00:01:00",
                account=None,
                explicit_arts_config=base_cfg,
                threads=1,
                output_dir=root / "results",
                max_jobs=1,
                dry_run=True,
                no_build=False,
                verbose=False,
                cflags=None,
                compile_args=None,
                gdb=False,
                profile=None,
                perf=False,
                perf_interval=0.1,
                cpu_pinning="default",
                exclude_nodes=None,
                nodelist=None,
                rdma=False,
                artifact_manager=am,
                step_name="distributed-writer-reader",
                report_steps=None,
                command_str="test",
            )

            SlurmBatchExecutor(host, request, deps).execute()

            single_config = json.loads(
                (
                    am.experiment_dir
                    / "distributed-writer-reader"
                    / "suite"
                    / "writer-reader"
                    / "1t_1n"
                    / "run_1"
                    / "run_config.json"
                ).read_text()
            )
            multinode_config = json.loads(
                (
                    am.experiment_dir
                    / "distributed-writer-reader"
                    / "suite"
                    / "writer-reader"
                    / "1t_2n"
                    / "run_1"
                    / "run_config.json"
                ).read_text()
            )
            script = (
                am.experiment_dir
                / "scripts"
                / "distributed-writer-reader__suite_writer-reader_1t_2n_run1.sbatch"
            ).read_text()

            self.assertNotIn("compile_args", single_config)
            self.assertNotIn("compile_args", multinode_config)
            self.assertEqual(multinode_config["nodes"], 2)
            self.assertEqual(multinode_config["arts_transport"], PROTOCOL_TCP)
            self.assertNotIn("reference", multinode_config)
            self.assertIn("--arts-only", script)
            self.assertIn("# OpenMP skipped (multi-node ARTS-only run)", script)

            arts_build_args = [
                call["compile_args"]
                for call in host.build_calls
                if call["variant"] == "arts"
            ]
            self.assertEqual(arts_build_args, [None, None])

    def test_slurm_build_cache_tracks_effective_compile_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_root = root / "benchmarks"
            (bench_root / "suite" / "a").mkdir(parents=True)
            base_cfg = root / "arts.cfg"
            base_cfg.write_text(f"[ARTS]\nworker_threads=1\n{KEY_PROTOCOL}={PROTOCOL_TCP}\n")
            carts_root = root / "carts"
            host = _FakeHost(bench_root)
            am = ArtifactManager(root / "results", "ts")
            deps = SlurmExecutorDependencies(
                resolve_effective_arts_config=lambda bench_path, explicit: base_cfg,
                parse_time_limit_seconds=lambda spec: 60,
                get_carts_dir=lambda: carts_root,
                get_benchmarks_dir=lambda: bench_root,
                step_name_to_token=lambda step: step,
            )
            request = SlurmBatchRequest(
                bench_list=["suite/a"],
                node_counts=[2],
                size="small",
                runs=1,
                timeout=30,
                partition=None,
                time_limit="00:01:00",
                account=None,
                explicit_arts_config=base_cfg,
                threads=1,
                output_dir=root / "results",
                max_jobs=1,
                dry_run=True,
                no_build=False,
                verbose=False,
                cflags=None,
                compile_args="--no-distributed-db",
                gdb=False,
                profile=None,
                perf=False,
                perf_interval=0.1,
                cpu_pinning="default",
                exclude_nodes=None,
                nodelist=None,
                rdma=True,
                artifact_manager=am,
                step_name="scale",
                report_steps=None,
                command_str="test",
            )
            executor = SlurmBatchExecutor(host, request, deps)

            executor._build_one_bench(
                am=am,
                bench="suite/a",
                multinode_disabled=set(),
                print_lock=threading.Lock(),
            )
            executor_no_args = SlurmBatchExecutor(
                host,
                replace(request, compile_args=None),
                deps,
            )
            executor_no_args._build_one_bench(
                am=am,
                bench="suite/a",
                multinode_disabled=set(),
                print_lock=threading.Lock(),
            )
            executor_no_args._build_one_bench(
                am=am,
                bench="suite/a",
                multinode_disabled=set(),
                print_lock=threading.Lock(),
            )

            arts_builds = [
                call for call in host.build_calls if call["variant"] == "arts"
            ]
            self.assertEqual(
                [call["compile_args"] for call in arts_builds],
                ["--no-distributed-db", None],
            )

    def test_multinode_dry_run_is_arts_only_without_openmp_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_root = root / "benchmarks"
            bench = bench_root / "suite" / "a"
            bench.mkdir(parents=True)
            base_cfg = root / "arts.cfg"
            base_cfg.write_text(f"[ARTS]\nworker_threads=1\n{KEY_PROTOCOL}={PROTOCOL_TCP}\n")
            carts_root = root / "carts"
            runtime_lib_dir = carts_root / ".install" / "arts" / "lib"
            runtime_lib_dir.mkdir(parents=True)
            (runtime_lib_dir / "libarts.so.2").write_text("fake arts runtime\n")
            (carts_root / ".install" / "carts" / "lib").mkdir(parents=True)
            (carts_root / ".install" / "llvm" / "lib").mkdir(parents=True)
            host = _FakeHost(bench_root)
            am = ArtifactManager(root / "results", "ts")
            deps = SlurmExecutorDependencies(
                resolve_effective_arts_config=lambda bench_path, explicit: base_cfg,
                parse_time_limit_seconds=lambda spec: 60,
                get_carts_dir=lambda: carts_root,
                get_benchmarks_dir=lambda: bench_root,
                step_name_to_token=lambda step: step,
            )
            request = SlurmBatchRequest(
                bench_list=["suite/a"],
                node_counts=[2],
                size="small",
                runs=1,
                timeout=30,
                partition=None,
                time_limit="00:01:00",
                account=None,
                explicit_arts_config=base_cfg,
                threads=1,
                output_dir=root / "results",
                max_jobs=1,
                dry_run=True,
                no_build=False,
                verbose=False,
                cflags=None,
                compile_args=None,
                gdb=False,
                profile=None,
                perf=False,
                perf_interval=0.1,
                cpu_pinning="default",
                exclude_nodes=None,
                nodelist=None,
                rdma=False,
                artifact_manager=am,
                step_name="scale",
                report_steps=None,
                command_str="test",
            )

            SlurmBatchExecutor(host, request, deps).execute()

            self.assertEqual(host.reference_calls, 0)
            run_config = json.loads(
                (
                    am.experiment_dir
                    / "scale"
                    / "suite"
                    / "a"
                    / "1t_2n"
                    / "run_1"
                    / "run_config.json"
                ).read_text()
            )
            self.assertNotIn("reference", run_config)
            self.assertEqual(run_config["arts_transport"], PROTOCOL_TCP)
            script = (
                am.experiment_dir
                / "scripts"
                / "scale__suite_a_1t_2n_run1.sbatch"
            ).read_text()
            self.assertIn("--arts-only", script)
            self.assertIn("# OpenMP skipped (multi-node ARTS-only run)", script)
            self.assertNotIn("[OpenMP] Running benchmark", script)

    def test_multinode_arts_job_uses_stored_openmp_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_root = root / "benchmarks"
            bench = bench_root / "suite" / "a"
            bench.mkdir(parents=True)
            base_cfg = root / "arts.cfg"
            base_cfg.write_text(f"[ARTS]\nworker_threads=1\n{KEY_PROTOCOL}={PROTOCOL_TCP}\n")
            carts_root = root / "carts"
            host = _FakeHost(bench_root)
            am = ArtifactManager(root / "results", "ts")
            am.set_phase("scale")
            deps = SlurmExecutorDependencies(
                resolve_effective_arts_config=lambda bench_path, explicit: base_cfg,
                parse_time_limit_seconds=lambda spec: 60,
                get_carts_dir=lambda: carts_root,
                get_benchmarks_dir=lambda: bench_root,
                step_name_to_token=lambda step: step,
            )
            request = SlurmBatchRequest(
                bench_list=["suite/a"],
                node_counts=[2],
                size="small",
                runs=1,
                timeout=30,
                partition=None,
                time_limit="00:01:00",
                account=None,
                explicit_arts_config=base_cfg,
                threads=64,
                output_dir=root / "results",
                max_jobs=1,
                dry_run=False,
                no_build=False,
                verbose=False,
                cflags=None,
                compile_args=None,
                gdb=False,
                profile=None,
                perf=False,
                perf_interval=0.1,
                cpu_pinning="default",
                exclude_nodes=None,
                nodelist=None,
                rdma=False,
                artifact_manager=am,
                step_name="scale",
                report_steps=None,
                command_str="test",
            )
            executor = SlurmBatchExecutor(host, request, deps)
            build_records = executor._build_one_bench(
                am=am,
                bench="suite/a",
                multinode_disabled=set(),
                print_lock=threading.Lock(),
            )
            scripts_dir = am.experiment_dir / "scripts"
            scripts_dir.mkdir(parents=True)

            job_configs = executor._generate_job_scripts(
                am=am,
                scripts_dir=scripts_dir,
                build_results={key: value for key, value in build_records},
                runtime_library_dirs=[],
                emit_header=False,
            )

            self.assertEqual(host.reference_calls, 1)
            self.assertEqual(len(job_configs), 1)
            self.assertTrue(job_configs[0][0].requires_reference_verification)
            run_config = json.loads(
                (
                    am.experiment_dir
                    / "scale"
                    / "suite"
                    / "a"
                    / "64t_2n"
                    / "run_1"
                    / "run_config.json"
                ).read_text()
            )
            self.assertEqual(run_config["reference"]["checksum"], "1.0")
            self.assertEqual(run_config["reference"]["omp_threads"], 64)
            script = (
                am.experiment_dir
                / "scripts"
                / "scale__suite_a_64t_2n_run1.sbatch"
            ).read_text()
            self.assertNotIn("--arts-only", script)
            self.assertIn("# OpenMP skipped (multi-node ARTS-only run)", script)


if __name__ == "__main__":
    unittest.main()
