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

from arts_config import KEY_PROTOCOL, PROTOCOL_RDMA, PROTOCOL_TCP, parse_arts_cfg  # noqa: E402
from slurm.batch import generate_arts_config_for_node, generate_sbatch_script, poll_jobs  # noqa: E402
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
                timeout_seconds=60,
            )

            generate_sbatch_script(config, script_path, job_result_script)

            content = script_path.read_text()
            self.assertIn(f'"{python_executable.resolve()}" "{job_result_script.resolve()}"', content)
            self.assertIn("srun --exclusive -N2 --ntasks=2 --ntasks-per-node=1", content)
            self.assertIn("--cpus-per-task=6 --cpu-bind=none", content)
            self.assertNotIn('python3 "', content)

    def test_generate_sbatch_script_can_skip_openmp_for_arts_only_runs(self) -> None:
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
                node_count=1,
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
                timeout_seconds=60,
                run_openmp=False,
            )

            generate_sbatch_script(config, script_path, job_result_script)

            content = script_path.read_text()
            self.assertIn("# OpenMP skipped (executable not specified)", content)
            self.assertIn("--arts-only", content)
            self.assertNotIn("--arts-only \\\n\n", content)
            self.assertNotIn("[OpenMP] Running benchmark", content)

    def test_generate_arts_config_for_node_sets_protocol_from_rdma_flag_and_unpins_multinode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "arts.cfg"
            template.write_text(
                "\n".join(
                    [
                        "[ARTS]",
                        "launcher=slurm",
                        "node_count=1",
                        "worker_threads=8",
                        f"{KEY_PROTOCOL}={PROTOCOL_TCP}",
                        "pin=1",
                    ]
                )
                + "\n"
            )

            tcp_cfg = generate_arts_config_for_node(
                template,
                root / "tcp-build",
                node_count=2,
                threads=16,
                rdma=False,
            )
            rdma_cfg = generate_arts_config_for_node(
                template,
                root / "rdma-build",
                node_count=2,
                threads=16,
                rdma=True,
            )
            rdma_8_cfg = generate_arts_config_for_node(
                template,
                root / "rdma-8-build",
                node_count=8,
                threads=64,
                rdma=True,
            )
            rdma_32_cfg = generate_arts_config_for_node(
                template,
                root / "rdma-32-build",
                node_count=32,
                threads=64,
                rdma=True,
            )
            rdma_64_cfg = generate_arts_config_for_node(
                template,
                root / "rdma-64-build",
                node_count=64,
                threads=64,
                rdma=True,
            )

            tcp_values = parse_arts_cfg(tcp_cfg)
            rdma_values = parse_arts_cfg(rdma_cfg)
            rdma_8_values = parse_arts_cfg(rdma_8_cfg)
            rdma_32_values = parse_arts_cfg(rdma_32_cfg)
            rdma_64_values = parse_arts_cfg(rdma_64_cfg)

            self.assertEqual(tcp_values["protocol"], PROTOCOL_TCP)
            self.assertEqual(rdma_values["protocol"], PROTOCOL_RDMA)
            self.assertEqual(rdma_8_values["protocol"], PROTOCOL_RDMA)
            self.assertEqual(rdma_32_values["protocol"], PROTOCOL_RDMA)
            self.assertEqual(rdma_64_values["protocol"], PROTOCOL_RDMA)
            self.assertEqual(tcp_values["pin"], "0")
            self.assertEqual(rdma_values["pin"], "0")
            self.assertEqual(rdma_8_values["pin"], "0")
            self.assertEqual(rdma_32_values["pin"], "0")
            self.assertEqual(rdma_64_values["pin"], "0")
            self.assertEqual(tcp_values["sender_threads"], "1")
            self.assertEqual(tcp_values["receiver_threads"], "1")
            self.assertEqual(rdma_values["sender_threads"], "1")
            self.assertEqual(rdma_values["receiver_threads"], "1")
            self.assertEqual(rdma_8_values["sender_threads"], "2")
            self.assertEqual(rdma_8_values["receiver_threads"], "2")
            self.assertEqual(rdma_32_values["sender_threads"], "2")
            self.assertEqual(rdma_32_values["receiver_threads"], "2")
            self.assertEqual(rdma_64_values["sender_threads"], "2")
            self.assertEqual(rdma_64_values["receiver_threads"], "2")
            self.assertEqual(tcp_values["counter_capture_interval"], "10")
            self.assertEqual(rdma_values["counter_capture_interval"], "10")
            self.assertEqual(rdma_8_values["counter_capture_interval"], "10")
            self.assertEqual(rdma_32_values["counter_capture_interval"], "10")
            self.assertEqual(rdma_64_values["counter_capture_interval"], "10")

    def test_generate_sbatch_script_runs_one_unbound_task_per_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_1"
            script_path = root / "job.sbatch"
            job_result_script = root / "job_result.py"
            arts_cfg = root / "arts.cfg"
            executable_arts = root / "gemm_arts"
            executable_omp = root / "gemm_omp"
            python_executable = root / ".venv" / "bin" / "python"

            for path in (job_result_script, executable_arts, executable_omp, python_executable):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("#!/bin/sh\n")
            arts_cfg.parent.mkdir(parents=True, exist_ok=True)
            arts_cfg.write_text(f"[ARTS]\n{KEY_PROTOCOL}={PROTOCOL_RDMA}\n")

            config = SlurmJobConfig(
                benchmark_name="polybench/gemm",
                run_number=1,
                node_count=4,
                time_limit="00:05:00",
                partition=None,
                account=None,
                executable_arts=executable_arts,
                executable_omp=executable_omp,
                arts_config_path=arts_cfg,
                python_executable=python_executable,
                run_dir=run_dir,
                size="small",
                threads=6,
                timeout_seconds=60,
            )

            generate_sbatch_script(config, script_path, job_result_script)

            content = script_path.read_text()
            self.assertIn("#SBATCH --nodes=4", content)
            self.assertIn("#SBATCH --ntasks-per-node=1", content)
            self.assertIn("#SBATCH --cpus-per-task=8", content)
            self.assertIn(
                "srun --exclusive -N4 --ntasks=4 --ntasks-per-node=1 "
                "--cpus-per-task=8 --cpu-bind=none --kill-on-bad-exit=1",
                content,
            )
            self.assertNotIn("ARTS RDMA Env: CONNECT_HELPER=", content)
            self.assertIn(
                'export ARTS_RDMA_CLOSE_AFTER_SEND="${ARTS_RDMA_CLOSE_AFTER_SEND:-1}"',
                content,
            )
            self.assertIn(
                'export ARTS_RDMA_CLOSE_AFTER_SEND_EVERY="${ARTS_RDMA_CLOSE_AFTER_SEND_EVERY:-1}"',
                content,
            )
            self.assertIn(
                'export ARTS_RDMA_CLOSE_WORKERS="${ARTS_RDMA_CLOSE_WORKERS:-4}"',
                content,
            )
            self.assertIn(
                'export ARTS_CONNECT_STEADY_BETWEEN_US="${ARTS_CONNECT_STEADY_BETWEEN_US:-1000}"',
                content,
            )
            self.assertIn(
                'export ARTS_RDMA_EAGER_CONNECT="${ARTS_RDMA_EAGER_CONNECT:-0}"',
                content,
            )
            self.assertIn(
                'export ARTS_LAZY_ACCEPT_DRAIN_LIMIT="${ARTS_LAZY_ACCEPT_DRAIN_LIMIT:-0}"',
                content,
            )
            self.assertIn(
                'export ARTS_LISTEN_BACKLOG="${ARTS_LISTEN_BACKLOG:-0}"',
                content,
            )
            self.assertIn(
                'export ARTS_TRACE_RDMA_SUMMARY="${ARTS_TRACE_RDMA_SUMMARY:-0}"',
                content,
            )
            self.assertNotIn("--ntasks-per-node=4", content)

    def test_generate_sbatch_script_preloads_runtime_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_1"
            script_path = root / "job.sbatch"
            job_result_script = root / "job_result.py"
            arts_cfg = root / "arts.cfg"
            executable_arts = root / "gemm_arts"
            python_executable = root / ".venv" / "bin" / "python"
            runtime_lib_dir = root / "artifacts" / "runtime" / "arts" / "lib"

            for path in (
                job_result_script,
                arts_cfg,
                executable_arts,
                python_executable,
                runtime_lib_dir / "libarts.so.2",
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
                executable_omp=None,
                arts_config_path=arts_cfg,
                python_executable=python_executable,
                run_dir=run_dir,
                size="small",
                threads=64,
                timeout_seconds=180,
                arts_runtime_lib_dir=runtime_lib_dir,
            )

            generate_sbatch_script(config, script_path, job_result_script)

            content = script_path.read_text()
            self.assertIn(f'ARTS_RUNTIME_LIB_DIR="{runtime_lib_dir.resolve()}"', content)
            self.assertIn("export ARTS_RUNTIME_PRELOAD", content)
            self.assertIn("ARTS Runtime Preload: $ARTS_RUNTIME_PRELOAD", content)
            self.assertIn(
                'env LD_PRELOAD="${ARTS_RUNTIME_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}"',
                content,
            )
            self.assertIn(str(executable_arts.resolve()), content)

    def test_generate_sbatch_script_honors_requested_nodelist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_1"
            script_path = root / "job.sbatch"
            job_result_script = root / "job_result.py"
            arts_cfg = root / "arts.cfg"
            executable_arts = root / "gemm_arts"
            python_executable = root / ".venv" / "bin" / "python"

            for path in (job_result_script, arts_cfg, executable_arts, python_executable):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("#!/bin/sh\n")

            config = SlurmJobConfig(
                benchmark_name="polybench/gemm",
                run_number=1,
                node_count=8,
                time_limit="00:05:00",
                partition="mi300x",
                account=None,
                executable_arts=executable_arts,
                executable_omp=None,
                arts_config_path=arts_cfg,
                python_executable=python_executable,
                run_dir=run_dir,
                size="small",
                threads=64,
                timeout_seconds=180,
                nodelist="b05u[01,07,13,19,25,31,37,43]",
            )

            generate_sbatch_script(config, script_path, job_result_script)

            content = script_path.read_text()
            self.assertIn("#SBATCH --nodes=8", content)
            self.assertIn("#SBATCH --nodelist=b05u[01,07,13,19,25,31,37,43]", content)


if __name__ == "__main__":
    unittest.main()
