from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "scripts"
TOOLS_DIR = REPO_ROOT / "tools"

sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark_models import Status  # noqa: E402
from benchmark_verification import verify_against_omp, verify_against_reference  # noqa: E402
from slurm.job_result import determine_status  # noqa: E402


class BenchmarkVerificationTest(unittest.TestCase):
    def test_verify_against_omp_marks_direct_mode(self) -> None:
        verification = verify_against_omp(
            Status.PASS,
            "100.0",
            Status.PASS,
            "100.5",
            0.01,
        )
        self.assertTrue(verification.correct)
        self.assertEqual(verification.mode, "direct_omp")

    def test_verify_against_reference_carries_reference_metadata(self) -> None:
        verification = verify_against_reference(
            Status.PASS,
            "42.0",
            "42.0",
            0.01,
            reference_source="/tmp/reference.json",
            reference_omp_threads=64,
        )
        self.assertTrue(verification.correct)
        self.assertEqual(verification.mode, "stored_omp_reference")
        self.assertEqual(verification.reference_source, "/tmp/reference.json")
        self.assertEqual(verification.reference_omp_threads, 64)

    def test_determine_status_fails_when_multinode_reference_is_missing(self) -> None:
        status, verification = determine_status(
            arts_exit=0,
            omp_exit=-1,
            arts_checksum="123.0",
            omp_checksum=None,
            reference_checksum=None,
            reference_source=None,
            reference_omp_threads=None,
            tolerance=0.01,
        )
        self.assertEqual(status, "FAIL")
        self.assertFalse(verification.correct)
        self.assertEqual(verification.mode, "stored_omp_reference")

    def test_determine_status_fails_when_direct_checksum_is_missing(self) -> None:
        status, verification = determine_status(
            arts_exit=0,
            omp_exit=0,
            arts_checksum="123.0",
            omp_checksum=None,
            reference_checksum=None,
            reference_source=None,
            reference_omp_threads=None,
            tolerance=0.01,
        )
        self.assertEqual(status, "FAIL")
        self.assertFalse(verification.correct)
        self.assertEqual(verification.mode, "direct_omp")


if __name__ == "__main__":
    unittest.main()
