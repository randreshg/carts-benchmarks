from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ATAX_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "polybench" / "atax"


class MakeTargetTests(unittest.TestCase):
    def _dry_run(self, target: str) -> str:
        result = subprocess.run(
            ["make", "--dry-run", target],
            cwd=ATAX_DIR,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def test_polybench_mini_arts_target_uses_native_dataset_flag(self) -> None:
        output = self._dry_run("mini-arts")
        self.assertIn("-DMINI_DATASET", output)

    def test_polybench_standard_arts_target_uses_native_dataset_flag(self) -> None:
        output = self._dry_run("standard-arts")
        self.assertIn("-DSTANDARD_DATASET", output)


if __name__ == "__main__":
    unittest.main()
