from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "configs" / "experiments"


class ExperimentConfigTest(unittest.TestCase):
    def test_single_node_scaling_uses_large_with_90s_timeout(self) -> None:
        payload = json.loads((CONFIG_DIR / "scale-single-node-large.json").read_text())
        self.assertEqual(payload["name"], "scale-single-node-large")
        for step in payload["steps"]:
            self.assertEqual(step["size"], "large")
            self.assertEqual(step["timeout"], 90)
            self.assertEqual(step["nodes"], "1")

    def test_multinode_overhead_steps_are_counter_only_by_default(self) -> None:
        payload = json.loads((CONFIG_DIR / "scale-multinode-1-to-64.json").read_text())
        self.assertEqual(payload["name"], "scale-multinode-1-to-64")
        for step in payload["steps"]:
            self.assertTrue(step["rdma"])
            self.assertEqual(step["size"], "large")
            self.assertEqual(step["timeout"], 90)
        overhead_steps = [
            step for step in payload["steps"] if step["name"].startswith("overhead-")
        ]
        self.assertEqual(len(overhead_steps), 2)
        for step in overhead_steps:
            self.assertEqual(step["profile"], "profile-overhead.cfg")
            self.assertFalse(step.get("perf", False))
            self.assertNotIn("perf_interval", step)

    def test_multinode_64n_rdma_runs_all_steps_at_64_nodes(self) -> None:
        payload = json.loads((CONFIG_DIR / "scale-multinode-64n-rdma.json").read_text())
        self.assertEqual(payload["name"], "scale-multinode-64n-rdma")
        self.assertEqual(len(payload["steps"]), 4)
        for step in payload["steps"]:
            self.assertEqual(step["size"], "large")
            self.assertEqual(step["threads"], "64")
            self.assertEqual(step["nodes"], "64")
            self.assertEqual(step["timeout"], 90)
            self.assertTrue(step["rdma"])
            self.assertFalse(step.get("perf", False))


if __name__ == "__main__":
    unittest.main()
