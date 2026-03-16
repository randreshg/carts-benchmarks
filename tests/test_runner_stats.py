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

from models import (  # noqa: E402
    Artifacts,
    BenchmarkConfig,
    BenchmarkResult,
    BuildResult,
    RunResult,
    Status,
    TimingResult,
    VerificationResult,
)
from runner import (  # noqa: E402
    annotate_startup_outliers,
    detect_startup_outliers,
    summarize_runs_robust,
)


def _make_result(
    run_number: int,
    *,
    arts_e2e: float,
    omp_e2e: float,
    arts_startup: float,
    omp_startup: float,
    speedup: float,
    run_dir: Path | None = None,
) -> BenchmarkResult:
    config = BenchmarkConfig(arts_threads=16, arts_nodes=1, omp_threads=16, launcher="local")
    build = BuildResult(status=Status.PASS, duration_sec=0.1, output="")
    run_arts = RunResult(
        status=Status.PASS,
        duration_sec=arts_e2e + arts_startup,
        exit_code=0,
        stdout="",
        stderr="",
        e2e_timings={"main": arts_e2e},
        startup_timings={"startup": arts_startup},
    )
    run_omp = RunResult(
        status=Status.PASS,
        duration_sec=omp_e2e + omp_startup,
        exit_code=0,
        stdout="",
        stderr="",
        e2e_timings={"main": omp_e2e},
        startup_timings={"startup": omp_startup},
    )
    timing = TimingResult(
        arts_time_sec=arts_e2e,
        omp_time_sec=omp_e2e,
        speedup=speedup,
        note="ok",
        arts_e2e_sec=arts_e2e,
        omp_e2e_sec=omp_e2e,
        arts_startup_sec=arts_startup,
        omp_startup_sec=omp_startup,
        speedup_basis="e2e",
        arts_total_sec=arts_e2e + arts_startup,
        omp_total_sec=omp_e2e + omp_startup,
    )
    verification = VerificationResult(
        correct=True,
        arts_checksum="1.0",
        omp_checksum="1.0",
        tolerance_used=0.0,
        note="Checksums match",
    )
    artifacts = Artifacts(
        benchmark_dir="polybench/gemm",
        run_dir=str(run_dir) if run_dir is not None else None,
    )
    return BenchmarkResult(
        name="polybench/gemm",
        suite="polybench",
        size="large",
        config=config,
        run_number=run_number,
        build_arts=build,
        build_omp=build,
        run_arts=run_arts,
        run_omp=run_omp,
        timing=timing,
        verification=verification,
        artifacts=artifacts,
        timestamp="2026-03-16T00:00:00",
        total_duration_sec=arts_e2e + omp_e2e,
    )


class BenchmarkRunnerStatsTest(unittest.TestCase):
    def test_detect_startup_outliers_flags_single_spike(self) -> None:
        analysis = detect_startup_outliers([0.11, 0.12, 0.11, 1.5, 0.10])
        self.assertEqual(analysis["outliers"], [False, False, False, True, False])
        self.assertIsNotNone(analysis["threshold"])

    def test_summarize_runs_robust_filters_outlier_from_median(self) -> None:
        runs = [
            _make_result(i + 1, arts_e2e=v, omp_e2e=v + 2.0, arts_startup=s, omp_startup=0.1, speedup=1.2)
            for i, (v, s) in enumerate(
                [
                    (10.0, 0.10),
                    (10.1, 0.11),
                    (9.9, 0.12),
                    (25.0, 2.50),  # startup and e2e spike
                    (10.0, 0.09),
                ]
            )
        ]
        summary = summarize_runs_robust(runs)
        self.assertAlmostEqual(summary["arts_e2e_sec"], 10.0, places=2)
        self.assertEqual(summary["arts_raw_count"], 5)
        self.assertEqual(summary["arts_filtered_count"], 4)

    def test_annotate_startup_outliers_writes_diagnostics_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results: list[BenchmarkResult] = []
            for i, (arts_e2e, arts_startup) in enumerate(
                [(10.0, 0.10), (10.1, 0.11), (10.0, 0.10), (30.0, 3.20), (9.9, 0.09)],
                start=1,
            ):
                run_dir = root / f"run_{i}"
                run_dir.mkdir(parents=True, exist_ok=True)
                results.append(
                    _make_result(
                        i,
                        arts_e2e=arts_e2e,
                        omp_e2e=12.0,
                        arts_startup=arts_startup,
                        omp_startup=0.10,
                        speedup=1.2,
                        run_dir=run_dir,
                    )
                )
                results[-1].run_arts.startup_diagnostics = {
                    "stdout_preview": ["line"] * 25,
                    "process_snapshot_pre": {"stdout": ["x"] * 120, "stderr": []},
                }
                results[-1].run_omp.startup_diagnostics = {
                    "stdout_preview": ["line"] * 25,
                    "process_snapshot_pre": {"stdout": ["x"] * 120, "stderr": []},
                }

            counts = annotate_startup_outliers(results, write_artifacts=True)
            self.assertEqual(counts["arts_outliers"], 1)
            self.assertEqual(counts["omp_outliers"], 0)
            self.assertTrue(
                (root / "run_4" / "startup_outlier_diagnostics.json").exists()
            )
            self.assertFalse(
                (root / "run_1" / "startup_outlier_diagnostics.json").exists()
            )
            self.assertEqual(results[0].run_arts.startup_diagnostics, {})
            self.assertIn("process_snapshot_pre", results[3].run_arts.startup_diagnostics)
            self.assertEqual(
                len(results[3].run_arts.startup_diagnostics["process_snapshot_pre"]["stdout"]),
                80,
            )


if __name__ == "__main__":
    unittest.main()
