from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import math  # noqa: E402

from carts_report import _declutter_log_labels, generate_paper_figures  # noqa: E402


class PaperFiguresGenerationTest(unittest.TestCase):
    def test_emits_all_dat_files_with_expected_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "run"
            results_dir.mkdir()
            payload = {
                "metadata": {"timestamp": "20260525_000000", "total_jobs": 8},
                "results": [
                    self._gemm_thread_result(1, 10.0, omp=12.0),
                    self._gemm_thread_result(64, 1.5, omp=3.0),
                    self._stencil_thread_result(1, 8.0, omp=8.0),
                    self._stencil_thread_result(64, 6.0, omp=4.0),
                    self._gemm_node_result(1, 10.0),
                    self._gemm_node_result(2, 6.0),
                    self._gemm_node_result(4, 4.0),
                ],
            }
            (results_dir / "results.json").write_text(json.dumps(payload))

            artifact = generate_paper_figures(results_dir)

            self.assertTrue(artifact.readme.exists())
            self.assertTrue(artifact.data_dir.is_dir())
            self.assertTrue(artifact.tabs_dir.is_dir())
            for name in (
                "scaling-trends.dat",
                "scaling-trends-labels.tex",
                "speedup-by-strategy.dat",
                "speedup-by-strategy-labels.tex",
                "latency-comparison.dat",
                "latency-comparison-labels.tex",
                "case-gemm-strong.dat",
                "case-gemm-capacity.dat",
                "case-gemm-decomp.dat",
            ):
                self.assertTrue((artifact.data_dir / name).exists(), name)

            scaling = (artifact.data_dir / "scaling-trends.dat").read_text().splitlines()
            self.assertEqual(
                scaling[0],
                "category category_id benchmark series_id threads x speedup",
            )
            self.assertGreater(len([line for line in scaling[1:] if line and not line.startswith("#")]), 0)
            self.assertTrue(any("polybench/gemm" in line for line in scaling))

            strategy = (artifact.data_dir / "speedup-by-strategy.dat").read_text().splitlines()
            self.assertEqual(strategy[0], "strategy strategy_id benchmark x speedup")
            self.assertGreater(len([line for line in strategy[1:] if line and not line.startswith("#")]), 0)
            self.assertTrue(any("polybench/gemm" in line for line in strategy))

            latency = (artifact.data_dir / "latency-comparison.dat").read_text().splitlines()
            self.assertEqual(latency[0], "benchmark carts_time omp_time ratio")
            self.assertGreater(len([line for line in latency[1:] if line and not line.startswith("#")]), 0)
            latency_labels = (artifact.data_dir / "latency-comparison-labels.tex").read_text()
            self.assertIn(r"\CartsLatencySymbolicXCoords", latency_labels)
            self.assertIn(r"\CartsLatencyCategoryLabels", latency_labels)
            self.assertIn(r"\CartsLatencyFirstCoord", latency_labels)
            self.assertIn(r"\CartsLatencyLastCoord", latency_labels)
            self.assertIn("polybench/gemm", latency_labels)

            gemm_strong = (artifact.data_dir / "case-gemm-strong.dat").read_text().splitlines()
            self.assertEqual(gemm_strong[0], "nodes speedup")
            self.assertGreater(len([line for line in gemm_strong[1:] if line and not line.startswith("#")]), 0)

            scaling_labels = (artifact.data_dir / "scaling-trends-labels.tex").read_text()
            self.assertIn(r"\CartsScalingTrendPlotsDenseLA", scaling_labels)
            self.assertIn(r"\CartsScalingTrendLabelsDenseLA", scaling_labels)
            self.assertIn("gemm", scaling_labels)

            strategy_labels = (artifact.data_dir / "speedup-by-strategy-labels.tex").read_text()
            self.assertIn(r"\CartsSpeedupStrategyAnnotations", strategy_labels)
            self.assertIn(r"\CartsSpeedupStrategyLabels", strategy_labels)

            capacity = (artifact.data_dir / "case-gemm-capacity.dat").read_text().splitlines()
            self.assertEqual(capacity[0], "nodes problem_size speedup")
            self.assertTrue(any(line.startswith("#") for line in capacity[1:]))

            decomp = (artifact.data_dir / "case-gemm-decomp.dat").read_text().splitlines()
            self.assertEqual(decomp[0], "nodes startup_s comm_s compute_s")
            self.assertTrue(any(line.startswith("#") for line in decomp[1:]))

            scaling_table = (artifact.tabs_dir / "scaling-results-body.tex").read_text()
            self.assertIn(r"\multicolumn{2}{l}{\textit{Geometric Mean}}", scaling_table)
            self.assertIn("gemm", scaling_table)

    def test_default_output_dir_uses_paper_figures_subdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "run"
            results_dir.mkdir()
            (results_dir / "results.json").write_text(json.dumps({
                "metadata": {"timestamp": "x"},
                "results": [self._gemm_thread_result(64, 1.0, omp=2.0)],
            }))

            artifact = generate_paper_figures(results_dir)
            self.assertEqual(artifact.output_dir.name, "paper-figures")
            self.assertTrue((artifact.output_dir / "README.md").exists())

    @staticmethod
    def _gemm_thread_result(threads: int, e2e: float, *, omp: float | None = None) -> dict[str, object]:
        return _make_result(
            benchmark="polybench/gemm",
            phase="thread-sweep-large-cap120",
            threads=threads,
            nodes=1,
            e2e=e2e,
            omp=omp,
        )

    @staticmethod
    def _stencil_thread_result(threads: int, e2e: float, *, omp: float | None = None) -> dict[str, object]:
        return _make_result(
            benchmark="polybench/jacobi2d",
            phase="thread-sweep-large-cap120",
            threads=threads,
            nodes=1,
            e2e=e2e,
            omp=omp,
        )

    @staticmethod
    def _gemm_node_result(nodes: int, e2e: float) -> dict[str, object]:
        return _make_result(
            benchmark="polybench/gemm",
            phase="node-sweep-extralarge-baseline-cap120",
            threads=64,
            nodes=nodes,
            e2e=e2e,
            omp=None,
        )


def _make_result(
    *,
    benchmark: str,
    phase: str,
    threads: int,
    nodes: int,
    e2e: float,
    omp: float | None,
) -> dict[str, object]:
    if omp is None:
        omp_payload: dict[str, object] = {"skipped": True, "exit_code": -1}
    else:
        omp_payload = {
            "skipped": False,
            "exit_code": 0,
            "e2e_timings": {benchmark: omp},
            "kernel_timings": {benchmark: omp - 0.1},
        }
    return {
        "benchmark": benchmark,
        "run_phase": phase,
        "run_number": 1,
        "size": "large",
        "threads": threads,
        "nodes": nodes,
        "status": "PASS",
        "verification": {"note": "Checksums match"},
        "arts": {
            "exit_code": 0,
            "duration_sec": e2e,
            "e2e_timings": {benchmark: e2e},
            "kernel_timings": {benchmark: max(0.1, e2e - 0.2)},
            "startup_timings": {benchmark: 0.2},
        },
        "omp": omp_payload,
        "slurm": {"state": "COMPLETED", "job_id": "1"},
        "artifacts": {},
        "diagnostics": {},
    }


class DeclutterLogLabelsTest(unittest.TestCase):
    GAP = 0.13

    def test_single_and_empty_unchanged(self) -> None:
        self.assertEqual(_declutter_log_labels([]), [])
        self.assertEqual(_declutter_log_labels([2.5]), [2.5])

    def test_already_separated_values_are_left_alone(self) -> None:
        values = [10.0, 5.0, 2.0, 1.0]
        out = _declutter_log_labels(values, min_gap_dex=self.GAP)
        for original, adjusted in zip(values, out):
            self.assertAlmostEqual(original, adjusted, places=9)

    def test_colliding_values_are_spread_to_the_minimum_gap(self) -> None:
        values = [2.31, 2.03, 1.78, 5.46, 3.21]
        out = _declutter_log_labels(values, min_gap_dex=self.GAP)
        self.assertEqual(
            [i for i, _ in sorted(enumerate(values), key=lambda p: p[1])],
            [i for i, _ in sorted(enumerate(out), key=lambda p: p[1])],
        )
        logs = sorted(math.log10(v) for v in out)
        for lo, hi in zip(logs, logs[1:]):
            self.assertGreaterEqual(hi - lo + 1e-9, self.GAP)

    def test_nonpositive_values_pass_through_in_place(self) -> None:
        out = _declutter_log_labels([2.0, 0.0, -1.0, 2.02], min_gap_dex=self.GAP)
        self.assertEqual(out[1], 0.0)
        self.assertEqual(out[2], -1.0)
        self.assertGreater(abs(math.log10(out[3]) - math.log10(out[0])), 0.0)


if __name__ == "__main__":
    unittest.main()
