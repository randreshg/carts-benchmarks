from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "scripts"
TOOLS_DIR = REPO_ROOT / "tools"

sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from runner import _evaluate_perf_gate_entry  # noqa: E402


def _result_row(
    run_number: int,
    *,
    speedup: float = 0.52,
    correct: bool = True,
    arts_outlier: bool = False,
    omp_outlier: bool = False,
) -> dict:
    return {
        "name": "specfem3d/velocity",
        "size": "large",
        "run_phase": None,
        "config": {
            "arts_threads": 16,
            "arts_nodes": 1,
        },
        "run_number": run_number,
        "build_arts": {"status": "pass"},
        "build_omp": {"status": "pass"},
        "run_arts": {
            "status": "pass",
            "startup_outlier": {"is_outlier": arts_outlier},
            "e2e_timings": {"kernel": 6.0},
        },
        "run_omp": {
            "status": "pass",
            "startup_outlier": {"is_outlier": omp_outlier},
            "e2e_timings": {"kernel": 4.0},
        },
        "timing": {
            "speedup": speedup,
            "arts_e2e_sec": 6.0,
            "omp_e2e_sec": 4.0,
        },
        "verification": {"correct": correct},
    }


class PerfGatePolicyTest(unittest.TestCase):
    def test_entry_passes_within_threshold_and_outlier_budget(self) -> None:
        entry = {
            "name": "specfem3d/velocity",
            "baseline_speedup": 0.53,
            "tolerance_pct": 0.12,
        }
        defaults = {
            "size": "large",
            "threads": 16,
            "nodes": 1,
            "require_success": True,
            "require_correct": True,
            "max_startup_outliers": {"arts": 1, "omp": 1},
        }
        rows = [
            _result_row(1, speedup=0.52, arts_outlier=False),
            _result_row(2, speedup=0.51, arts_outlier=True),
            _result_row(3, speedup=0.53, arts_outlier=False),
        ]

        evaluation = _evaluate_perf_gate_entry(rows, entry, defaults)
        self.assertTrue(evaluation["pass"])
        self.assertEqual(evaluation["reasons"], [])

    def test_entry_fails_on_correctness_mismatch(self) -> None:
        entry = {"name": "specfem3d/velocity", "min_speedup": 0.40}
        defaults = {
            "size": "large",
            "threads": 16,
            "nodes": 1,
            "require_success": True,
            "require_correct": True,
        }
        rows = [_result_row(1, correct=True), _result_row(2, correct=False)]

        evaluation = _evaluate_perf_gate_entry(rows, entry, defaults)
        self.assertFalse(evaluation["pass"])
        self.assertIn("correctness mismatch", evaluation["reasons"])

    def test_entry_fails_when_startup_outliers_exceed_cap(self) -> None:
        entry = {"name": "specfem3d/velocity", "min_speedup": 0.40}
        defaults = {
            "size": "large",
            "threads": 16,
            "nodes": 1,
            "max_startup_outliers": {"arts": 1, "omp": 1},
        }
        rows = [
            _result_row(1, arts_outlier=True),
            _result_row(2, arts_outlier=True),
            _result_row(3, arts_outlier=False),
        ]

        evaluation = _evaluate_perf_gate_entry(rows, entry, defaults)
        self.assertFalse(evaluation["pass"])
        self.assertTrue(
            any("startup outliers" in reason for reason in evaluation["reasons"])
        )


if __name__ == "__main__":
    unittest.main()
