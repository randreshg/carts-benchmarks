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

from slurm.experiment import (  # noqa: E402
    count_total_slurm_jobs,
    find_multinode_disabled_benchmarks,
    format_node_counts_display,
    load_existing_job_statuses,
    merge_result_rows,
)


class _FakeHost:
    def __init__(self, benchmarks_dir: Path) -> None:
        self.benchmarks_dir = benchmarks_dir


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


if __name__ == "__main__":
    unittest.main()
