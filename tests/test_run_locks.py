from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))

from run_locks import BenchmarkRunLockError, BenchmarkRunLocks  # noqa: E402


class BenchmarkRunLocksTest(unittest.TestCase):
    def test_local_runs_block_same_host_and_results_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = BenchmarkRunLocks(
                results_dir=root / "results",
                run_id="run_a",
                command="local-a",
                mode="local",
                cwd=root,
                node_lock_dir=root / "locks",
            )
            second = BenchmarkRunLocks(
                results_dir=root / "results",
                run_id="run_b",
                command="local-b",
                mode="local",
                cwd=root,
                node_lock_dir=root / "locks",
            )

            first.acquire()
            try:
                with self.assertRaises(BenchmarkRunLockError):
                    second.acquire()
            finally:
                first.release()

    def test_slurm_runs_can_share_login_host_and_results_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = BenchmarkRunLocks(
                results_dir=root / "results",
                run_id="run_a",
                command="slurm-a",
                mode="slurm",
                cwd=root,
                node_lock=False,
                results_lock=False,
                node_lock_dir=root / "locks",
            )
            second = BenchmarkRunLocks(
                results_dir=root / "results",
                run_id="run_b",
                command="slurm-b",
                mode="slurm",
                cwd=root,
                node_lock=False,
                results_lock=False,
                node_lock_dir=root / "locks",
            )

            first.acquire()
            try:
                second.acquire()
            finally:
                second.release()
                first.release()


if __name__ == "__main__":
    unittest.main()
