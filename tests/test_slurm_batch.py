from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "scripts"
TOOLS_DIR = REPO_ROOT / "tools"

sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from slurm.batch import poll_jobs  # noqa: E402
from slurm.models import SlurmJobStatus  # noqa: E402


class SlurmBatchPollingTest(unittest.TestCase):
    def test_poll_jobs_strips_squeue_fields(self) -> None:
        job_statuses = {
            "101": SlurmJobStatus(
                job_id="101",
                benchmark_name="a",
                run_number=1,
                node_count=1,
                state="PENDING",
            ),
            "102": SlurmJobStatus(
                job_id="102",
                benchmark_name="b",
                run_number=1,
                node_count=1,
                state="PENDING",
            ),
        }
        result = SimpleNamespace(
            returncode=0,
            stdout="101   | PENDING \n102| RUNNING\n",
        )
        with patch("slurm.batch.subprocess.run", return_value=result):
            states = poll_jobs(job_statuses)
        self.assertEqual(states, {"101": "PENDING", "102": "RUNNING"})

    def test_poll_jobs_preserves_inflight_state_when_scheduler_temporarily_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_1"
            run_dir.mkdir(parents=True)
            job_statuses = {
                "201": SlurmJobStatus(
                    job_id="201",
                    benchmark_name="a",
                    run_number=1,
                    node_count=8,
                    state="PENDING",
                    run_dir=run_dir,
                )
            }
            result = SimpleNamespace(returncode=0, stdout="")
            unknown = SlurmJobStatus(
                job_id="201",
                benchmark_name="a",
                run_number=1,
                node_count=8,
                state="UNKNOWN",
            )
            with patch("slurm.batch.subprocess.run", return_value=result):
                with patch("slurm.batch._get_scontrol_status", return_value=unknown):
                    states = poll_jobs(job_statuses)
            self.assertEqual(states["201"], "PENDING")

    def test_poll_jobs_marks_unknown_terminal_when_result_artifact_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_1"
            run_dir.mkdir(parents=True)
            (run_dir / "result.json").write_text("{}")
            job_statuses = {
                "301": SlurmJobStatus(
                    job_id="301",
                    benchmark_name="a",
                    run_number=1,
                    node_count=8,
                    state="RUNNING",
                    run_dir=run_dir,
                )
            }
            result = SimpleNamespace(returncode=0, stdout="")
            unknown = SlurmJobStatus(
                job_id="301",
                benchmark_name="a",
                run_number=1,
                node_count=8,
                state="UNKNOWN",
            )
            with patch("slurm.batch.subprocess.run", return_value=result):
                with patch("slurm.batch._get_scontrol_status", return_value=unknown):
                    states = poll_jobs(job_statuses)
            self.assertEqual(states["301"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
