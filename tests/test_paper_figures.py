from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from carts_report import generate_paper_figures  # noqa: E402


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
            for name in (
                "scaling-trends.dat",
                "speedup-by-strategy.dat",
                "latency-comparison.dat",
                "case-gemm-strong.dat",
                "case-gemm-capacity.dat",
                "case-gemm-decomp.dat",
            ):
                self.assertTrue((artifact.data_dir / name).exists(), name)

            scaling = (artifact.data_dir / "scaling-trends.dat").read_text().splitlines()
            self.assertEqual(scaling[0], "category threads geomean_speedup")
            self.assertGreater(len([line for line in scaling[1:] if line and not line.startswith("#")]), 0)
            self.assertTrue(any(line.startswith("Dense_LA ") or line.startswith("Dense LA") for line in scaling))

            strategy = (artifact.data_dir / "speedup-by-strategy.dat").read_text().splitlines()
            self.assertEqual(strategy[0], "strategy benchmark speedup")
            self.assertGreater(len([line for line in strategy[1:] if line and not line.startswith("#")]), 0)
            self.assertTrue(any("polybench/gemm" in line for line in strategy))

            latency = (artifact.data_dir / "latency-comparison.dat").read_text().splitlines()
            self.assertEqual(latency[0], "benchmark carts_time omp_time ratio")
            self.assertGreater(len([line for line in latency[1:] if line and not line.startswith("#")]), 0)

            gemm_strong = (artifact.data_dir / "case-gemm-strong.dat").read_text().splitlines()
            self.assertEqual(gemm_strong[0], "nodes speedup")
            self.assertGreater(len([line for line in gemm_strong[1:] if line and not line.startswith("#")]), 0)

            capacity = (artifact.data_dir / "case-gemm-capacity.dat").read_text().splitlines()
            self.assertEqual(capacity[0], "nodes problem_size speedup")
            self.assertTrue(any(line.startswith("#") for line in capacity[1:]))

            decomp = (artifact.data_dir / "case-gemm-decomp.dat").read_text().splitlines()
            self.assertEqual(decomp[0], "nodes startup_s comm_s compute_s")
            self.assertTrue(any(line.startswith("#") for line in decomp[1:]))

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


if __name__ == "__main__":
    unittest.main()
