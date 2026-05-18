from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "scripts"
TOOLS_DIR = REPO_ROOT / "tools"

sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from arts_config import KEY_PROTOCOL, PROTOCOL_TCP  # noqa: E402
from artifacts import ArtifactManager  # noqa: E402
from models import BenchmarkConfig, BuildResult, ReferenceChecksum, Status  # noqa: E402
from slurm.experiment import (  # noqa: E402
    SlurmBatchExecutor,
    SlurmBatchRequest,
    SlurmExecutorDependencies,
    compute_prebuild_worker_count,
    count_total_slurm_jobs,
    find_multinode_disabled_benchmarks,
    format_node_counts_display,
    load_existing_job_statuses,
    merge_result_rows,
)


class _FakeHost:
    def __init__(self, benchmarks_dir: Path) -> None:
        self.benchmarks_dir = benchmarks_dir
        self.artifact_manager = None

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
        del name, size, cflags, compile_args, arts_config
        assert build_output_dir is not None
        executable = build_output_dir / ("bench_omp" if variant == "openmp" else "bench_arts")
        executable.write_text("#!/bin/sh\n")
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
        return ReferenceChecksum(
            status=Status.PASS,
            checksum="1.0",
            omp_threads=omp_threads,
            note="ok",
            source="fake",
        )


class SlurmExperimentHelpersTest(unittest.TestCase):
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

    def test_compute_prebuild_worker_count_caps_by_requested_threads(self) -> None:
        self.assertEqual(
            compute_prebuild_worker_count(
                host_cpus=128,
                requested_threads=64,
                benchmark_count=23,
            ),
            2,
        )
        self.assertEqual(
            compute_prebuild_worker_count(
                host_cpus=16,
                requested_threads=64,
                benchmark_count=23,
            ),
            1,
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
            cache = carts_root / "external" / "arts" / "build" / "CMakeCache.txt"
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
            snapshot_dir = Path(alpha_run_config["arts_runtime_lib_dir"])
            self.assertTrue((snapshot_dir / "libarts.so.2").exists())


if __name__ == "__main__":
    unittest.main()
