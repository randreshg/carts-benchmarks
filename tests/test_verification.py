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

from common import RUN_CONFIG_JSON_FILENAME, SLURM_OUT_FILENAME, parse_checksum  # noqa: E402
from models import Status  # noqa: E402
from verification import verify_against_omp, verify_against_reference  # noqa: E402
from slurm.job_result import determine_status, generate_result  # noqa: E402


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

    def test_determine_status_passes_arts_only_without_reference(self) -> None:
        status, verification = determine_status(
            arts_exit=0,
            omp_exit=-1,
            arts_checksum="123.0",
            omp_checksum=None,
            reference_checksum=None,
            reference_source=None,
            reference_omp_threads=None,
            tolerance=0.01,
            arts_only=True,
        )
        self.assertEqual(status, "PASS")
        self.assertTrue(verification.correct)
        self.assertEqual(verification.mode, "arts_only")

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

    def test_parse_checksum_ignores_slurm_banner_separator_lines(self) -> None:
        output = """
kernel.velocity: 30.000000s
checksum: 1.042702613658e-05
--------------------------------------------
Job 47531 on b06u37,b07u01
--------------------------------------------
"""
        self.assertEqual(parse_checksum(output), "1.042702613658e-05")

    def test_generate_result_ignores_repeated_slurm_banners_after_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / SLURM_OUT_FILENAME).write_text(
                "\n".join(
                    [
                        "[ARTS] Running benchmark...",
                        "kernel.velocity: 30.000000s",
                        "checksum: 1.042702613658e-05",
                        "--------------------------------------------",
                        "Job 47531 on b06u37,b07u01",
                        "--------------------------------------------",
                        "[ARTS] Exit code: 0",
                        "[ARTS] Duration: 30 seconds",
                    ]
                )
            )
            (run_dir / RUN_CONFIG_JSON_FILENAME).write_text(
                """
{
  "reference": {
    "checksum": "1.042702613658e-05",
    "source": "/tmp/reference.json",
    "omp_threads": 64
  }
}
"""
            )

            result = generate_result(
                benchmark="specfem3d/velocity",
                run_number=1,
                size="extralarge",
                arts_exit=0,
                arts_duration=30.0,
                omp_exit=-1,
                omp_duration=0.0,
                counter_dir=None,
                slurm_job_id="47531",
                slurm_nodelist="b06u37,b07u01",
                output_dir=run_dir,
                arts_only=False,
            )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["arts"]["checksum"], "1.042702613658e-05")
        self.assertEqual(
            result["arts"]["kernel_timings"],
            {"velocity": 30.0},
        )


if __name__ == "__main__":
    unittest.main()
