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

from slurm.batch import generate_sbatch_script, poll_jobs  # noqa: E402
from slurm.models import SlurmJobConfig, SlurmJobStatus  # noqa: E402


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

    def test_generate_sbatch_script_uses_configured_python_executable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_1"
            script_path = root / "job.sbatch"
            job_result_script = root / "job_result.py"
            arts_cfg = root / "arts.cfg"
            executable_arts = root / "gemm_arts"
            executable_omp = root / "gemm_omp"
            python_executable = root / ".venv" / "bin" / "python"

            for path in (
                job_result_script,
                arts_cfg,
                executable_arts,
                executable_omp,
                python_executable,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("#!/bin/sh\n")

            config = SlurmJobConfig(
                benchmark_name="polybench/gemm",
                run_number=1,
                node_count=2,
                time_limit="00:05:00",
                partition=None,
                account=None,
                executable_arts=executable_arts,
                executable_omp=executable_omp,
                arts_config_path=arts_cfg,
                python_executable=python_executable,
                run_dir=run_dir,
                size="small",
                threads=4,
            )

            generate_sbatch_script(config, script_path, job_result_script)

            content = script_path.read_text()
            self.assertIn(f'"{python_executable.resolve()}" "{job_result_script.resolve()}"', content)
            self.assertNotIn('python3 "', content)


if __name__ == "__main__":
    unittest.main()
