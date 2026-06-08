from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))

from serial_1_to_2_validation import (  # noqa: E402
    Phase,
    PhaseResult,
    check_scaling,
    phase_command,
    summarize_phase,
)


class SerialOneToTwoValidationTest(unittest.TestCase):
    def test_phase_command_uses_serial_slurm_and_wall_cap(self) -> None:
        phase = Phase(
            name="two-node-distributed-db",
            size="extralarge",
            threads="64",
            nodes="2",
            timeout=270,
            runs=1,
            rdma=True,
            compile_args=None,
        )

        command = phase_command(
            dekk="dekk",
            benchmark="polybench/gemm",
            phase=phase,
            partition="mi300x_cpx",
            results_dir=Path("results"),
            dry_run=False,
            nodelist=None,
            exclude_nodes=None,
            cpu_pinning=None,
            time_limit="00:05:00",
            quiet=True,
        )

        self.assertIn("--max-jobs", command)
        self.assertEqual(command[command.index("--max-jobs") + 1], "1")
        self.assertEqual(command[command.index("--threads") + 1], "64")
        self.assertEqual(command[command.index("--nodes") + 1], "2")
        self.assertEqual(command[command.index("--timeout") + 1], "270")
        self.assertEqual(command[command.index("--time-limit") + 1], "00:05:00")
        self.assertNotIn("--compile-args", command)
        self.assertIn("--quiet", command)

    def test_summarize_phase_fails_on_runtime_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            (result_dir / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "benchmark": "polybench/gemm",
                                "status": "PASS",
                                "arts": {"e2e_timings": {"main": 12.5}},
                                "diagnostics": {
                                    "runtime_warning": {
                                        "has_warning": True,
                                        "reasons": ["remote_send_hard_timeout"],
                                    }
                                },
                            }
                        ]
                    }
                )
            )
            phase = Phase(
                name="two-node-baseline",
                size="extralarge",
                threads="64",
                nodes="2",
                timeout=270,
                runs=1,
                rdma=True,
                compile_args=None,
            )

            result = summarize_phase(
                "polybench/gemm",
                phase,
                result_dir,
                fail_on_runtime_warning=True,
            )

        self.assertEqual(result.status, "runtime-warning")
        self.assertEqual(result.metric_time_sec, 12.5)
        self.assertTrue(result.runtime_warning)

    def test_scaling_check_requires_two_node_distributed_phase(self) -> None:
        rows = [
            PhaseResult(
                benchmark="polybench/gemm",
                phase="single-node-reference",
                status="pass",
                metric_time_sec=100.0,
                result_dir=Path("ref"),
                results_json=None,
            ),
            PhaseResult(
                benchmark="polybench/gemm",
                phase="two-node-distributed-db",
                status="pass",
                metric_time_sec=80.0,
                result_dir=Path("ddb"),
                results_json=None,
            ),
        ]

        self.assertIsNotNone(check_scaling("polybench/gemm", rows, 1.5))
        rows[1].metric_time_sec = 40.0
        self.assertIsNone(check_scaling("polybench/gemm", rows, 1.2))


if __name__ == "__main__":
    unittest.main()
