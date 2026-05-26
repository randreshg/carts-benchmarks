from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))

from artifacts import ArtifactManager  # noqa: E402
from run_locks import (  # noqa: E402
    RESULTS_LOCK_FILENAME,
    BenchmarkRunLockError,
    BenchmarkRunLocks,
)


class BenchmarkRunLockTest(unittest.TestCase):
    def _locks(
        self,
        root: Path,
        *,
        results_name: str = "results",
        run_id: str = "run",
        mode: str = "local",
        node_lock: bool = False,
    ) -> BenchmarkRunLocks:
        return BenchmarkRunLocks(
            results_dir=root / results_name,
            run_id=run_id,
            command="carts benchmarks run",
            mode=mode,
            cwd=root,
            node_lock=node_lock,
            node_lock_dir=root / "locks",
        )

    def test_same_results_directory_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._locks(root, run_id="first")
            first.acquire()
            try:
                second = self._locks(root, run_id="second")
                with self.assertRaisesRegex(BenchmarkRunLockError, "results directory"):
                    second.acquire()
            finally:
                first.release()

            self.assertFalse((root / "results" / RESULTS_LOCK_FILENAME).exists())

    def test_same_node_is_exclusive_for_local_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._locks(
                root,
                results_name="results-a",
                run_id="first",
                node_lock=True,
            )
            first.acquire()
            try:
                second = self._locks(
                    root,
                    results_name="results-b",
                    run_id="second",
                    node_lock=True,
                )
                with self.assertRaisesRegex(BenchmarkRunLockError, "this node"):
                    second.acquire()
            finally:
                first.release()

    def test_same_node_is_exclusive_for_slurm_runs_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._locks(
                root,
                results_name="results-a",
                run_id="first",
                mode="slurm",
                node_lock=True,
            )
            first.acquire()
            try:
                second = self._locks(
                    root,
                    results_name="results-b",
                    run_id="second",
                    mode="slurm",
                    node_lock=True,
                )
                with self.assertRaisesRegex(BenchmarkRunLockError, "this node"):
                    second.acquire()
            finally:
                first.release()

    def test_stale_same_host_results_lock_is_reclaimed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = root / "results"
            results_dir.mkdir()
            lock_path = results_dir / RESULTS_LOCK_FILENAME
            lock_path.write_text(json.dumps({"host": socket.gethostname(), "pid": -1}))

            locks = self._locks(root)
            locks.acquire()
            try:
                metadata = json.loads(lock_path.read_text())
                self.assertEqual(metadata["pid"], os.getpid())
            finally:
                locks.release()

    def test_other_host_results_lock_is_not_reclaimed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = root / "results"
            results_dir.mkdir()
            lock_path = results_dir / RESULTS_LOCK_FILENAME
            lock_path.write_text(json.dumps({"host": f"other-{socket.gethostname()}", "pid": -1}))

            locks = self._locks(root)
            with self.assertRaisesRegex(BenchmarkRunLockError, "results directory"):
                locks.acquire()

            self.assertTrue(lock_path.exists())

    def test_artifact_manager_rejects_existing_experiment_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ArtifactManager(root / "results", "same-run")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                ArtifactManager(root / "results", "same-run")


if __name__ == "__main__":
    unittest.main()
