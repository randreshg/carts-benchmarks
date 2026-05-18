from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from dashboard import generate_dashboard  # noqa: E402


class DashboardGenerationTest(unittest.TestCase):
    def test_generate_dashboard_writes_static_app_and_data_extracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = root / "run"
            results_dir.mkdir()
            counter_dir = results_dir / "node-sweep" / "polybench" / "gemm" / "64t_2n" / "run_1" / "counters"
            counter_dir.mkdir(parents=True)
            (counter_dir / "cluster.json").write_text(json.dumps({
                "counters": {
                    "BYTES_REMOTE_SENT": {"value": 128},
                    "BYTES_REMOTE_RECEIVED": {"value": 128},
                    "NUM_REMOTE_SEND": {"value": 2},
                    "NUM_REMOTE_RECEIVE": {"value": 2},
                    "NUM_REMOTE_CONNECT_ATTEMPT": {"value": 2},
                    "NUM_REMOTE_CONNECT_SUCCESS": {"value": 2},
                    "BYTES_MEMORY_FOOTPRINT": {"value": 4096},
                    "TIME_INIT": {"value_ms": 3.0},
                    "TIME_TOTAL": {"value_ms": 1200.0},
                }
            }))
            payload = {
                "metadata": {"timestamp": "20260518_000000", "total_jobs": 4},
                "results": [
                    self._result("thread-sweep-large-cap120", 1, 1, 10.0, omp=12.0),
                    self._result("thread-sweep-large-cap120", 2, 1, 6.0, omp=7.0),
                    self._result("node-sweep-extralarge-baseline-cap120", 64, 1, 8.0),
                    self._result(
                        "node-sweep-extralarge-baseline-cap120",
                        64,
                        2,
                        5.0,
                        counter_dir=counter_dir,
                    ),
                ],
            }
            (results_dir / "results.json").write_text(json.dumps(payload))

            artifact = generate_dashboard(results_dir)

            self.assertTrue(artifact.index_html.exists())
            self.assertTrue(artifact.data_js.exists())
            self.assertTrue((artifact.output_dir / "assets" / "app.js").exists())
            self.assertTrue((artifact.output_dir / "data" / "results_flat.csv").exists())
            self.assertTrue((artifact.output_dir / "data" / "family_examples.csv").exists())
            self.assertTrue((artifact.output_dir / "data" / "communication_summary.csv").exists())
            data_js = artifact.data_js.read_text()
            self.assertIn("CARTS_DASHBOARD_DATA", data_js)
            self.assertIn("remote_bytes_total", data_js)
            self.assertIn("remote_bytes_per_message", data_js)
            self.assertIn("family_examples", data_js)
            self.assertIn("polybench/gemm", data_js)

    def test_generate_dashboard_merges_extra_results_for_openmp_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "primary"
            extra = root / "extra"
            primary.mkdir()
            extra.mkdir()
            (primary / "results.json").write_text(json.dumps({
                "metadata": {"timestamp": "primary"},
                "results": [self._result("thread-sweep-large-cap120", 64, 1, 4.0)],
            }))
            (extra / "results.json").write_text(json.dumps({
                "metadata": {"timestamp": "extra"},
                "results": [self._result("openmp-compare-large-64t", 64, 1, 5.0, omp=7.5)],
            }))

            artifact = generate_dashboard(primary, extra_results=[extra])
            data_js = artifact.data_js.read_text()

            self.assertIn("openmp-compare-large-64t", data_js)
            self.assertIn("carts_vs_openmp", data_js)
            self.assertIn(str(extra / "results.json"), data_js)

    @staticmethod
    def _result(
        phase: str,
        threads: int,
        nodes: int,
        e2e: float,
        *,
        omp: float | None = None,
        counter_dir: Path | None = None,
    ) -> dict[str, object]:
        omp_payload: dict[str, object]
        if omp is None:
            omp_payload = {"skipped": True, "exit_code": -1}
        else:
            omp_payload = {
                "skipped": False,
                "exit_code": 0,
                "e2e_timings": {"gemm": omp},
                "kernel_timings": {"gemm": omp - 0.1},
            }
        artifacts = {"counter_dir": str(counter_dir)} if counter_dir else {}
        return {
            "benchmark": "polybench/gemm",
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
                "e2e_timings": {"gemm": e2e},
                "kernel_timings": {"gemm": e2e - 0.2},
                "startup_timings": {"gemm": 0.2},
            },
            "omp": omp_payload,
            "slurm": {"state": "COMPLETED", "job_id": "1"},
            "artifacts": artifacts,
            "diagnostics": {},
        }


if __name__ == "__main__":
    unittest.main()
