from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "configs" / "experiments"

NO_PERF_SCALABILITY_CONFIGS = [
    "all-benchmarks-full-large-extralarge.json",
    "all-benchmarks-multinode-extralarge.json",
    "all-benchmarks-single-node-large.json",
    "gemm-full-extralarge.json",
    "gemm-multinode-extralarge.json",
    "gemm-validation.json",
    "scale-multinode-1-to-64.json",
    "scale-multinode-64n-rdma.json",
    "scale-single-node-large.json",
    "single-node-all-benchmarks-scaling.json",
]

MULTINODE_EXTRALARGE_CONFIGS = [
    "all-benchmarks-full-large-extralarge.json",
    "all-benchmarks-multinode-extralarge.json",
    "gemm-full-extralarge.json",
    "gemm-multinode-extralarge.json",
    "gemm-validation.json",
    "scale-multinode-1-to-64.json",
    "scale-multinode-64n-rdma.json",
]

FULL_NODE_SWEEP = "1,2,4,8,16,32,64"


class ExperimentConfigTest(unittest.TestCase):
    def test_single_node_all_benchmarks_scaling_experiment_is_self_contained(self) -> None:
        payload = json.loads(
            (CONFIG_DIR / "single-node-all-benchmarks-scaling.json").read_text()
        )
        self.assertEqual(payload["name"], "single-node-all-benchmarks-scaling")
        self.assertEqual(
            [step["name"] for step in payload["steps"]],
            [
                "medium-thread-sweep",
                "large-64-competitive",
                "large-64-runtime-diagnostics",
            ],
        )

        for step in payload["steps"]:
            with self.subTest(step=step["name"]):
                self.assertNotIn("benchmarks", step)
                self.assertEqual(step["nodes"], "1")
                self.assertEqual(step["launcher"], "local")
                self.assertEqual(step["arts_config"], "local.cfg")
                self.assertFalse(step["rdma"])
                self.assertFalse(step.get("perf", False))
                self.assertNotIn("perf_interval", step)
                self.assertIn("64", {item.strip() for item in step["threads"].split(",")})

        sweep = payload["steps"][0]
        self.assertEqual(sweep["size"], "medium")
        self.assertEqual(sweep["threads"], "1,2,4,8,16,32,64")
        self.assertEqual(sweep["runs"], 1)
        self.assertEqual(sweep["timeout"], 600)

        competitive = payload["steps"][1]
        self.assertEqual(competitive["size"], "large")
        self.assertEqual(competitive["threads"], "64")
        self.assertEqual(competitive["runs"], 3)
        self.assertEqual(competitive["timeout"], 300)

        diagnostics = payload["steps"][2]
        self.assertEqual(diagnostics["size"], "large")
        self.assertEqual(diagnostics["threads"], "64")
        self.assertEqual(diagnostics["runs"], 1)
        self.assertEqual(diagnostics["profile"], "profile-thread-edt.cfg")
        self.assertEqual(diagnostics["timeout"], 300)

    def test_single_node_scaling_uses_large_with_90s_timeout(self) -> None:
        payload = json.loads((CONFIG_DIR / "scale-single-node-large.json").read_text())
        self.assertEqual(payload["name"], "scale-single-node-large")
        for step in payload["steps"]:
            self.assertEqual(step["size"], "large")
            self.assertEqual(step["timeout"], 90)
            self.assertEqual(step["nodes"], "1")
            self.assertFalse(step.get("perf", False))
            self.assertNotIn("perf_interval", step)

    def test_multinode_overhead_steps_are_counter_only_by_default(self) -> None:
        payload = json.loads((CONFIG_DIR / "scale-multinode-1-to-64.json").read_text())
        self.assertEqual(payload["name"], "scale-multinode-1-to-64")
        for step in payload["steps"]:
            self.assertTrue(step["rdma"])
            self.assertEqual(step["size"], "extralarge")
            self.assertEqual(step["timeout"], 1200)
            self.assertFalse(step.get("perf", False))
            self.assertNotIn("perf_interval", step)
        overhead_steps = [
            step for step in payload["steps"] if step["name"].startswith("overhead-")
        ]
        self.assertEqual(len(overhead_steps), 2)
        for step in overhead_steps:
            self.assertEqual(step["profile"], "profile-overhead.cfg")

    def test_multinode_64n_rdma_runs_all_steps_at_64_nodes(self) -> None:
        payload = json.loads((CONFIG_DIR / "scale-multinode-64n-rdma.json").read_text())
        self.assertEqual(payload["name"], "scale-multinode-64n-rdma")
        self.assertEqual(len(payload["steps"]), 4)
        for step in payload["steps"]:
            self.assertEqual(step["size"], "extralarge")
            self.assertEqual(step["threads"], "64")
            self.assertEqual(step["nodes"], "64")
            self.assertEqual(step["timeout"], 1200)
            self.assertTrue(step["rdma"])
            self.assertFalse(step.get("perf", False))
            self.assertNotIn("perf_interval", step)

    def test_scalability_experiments_do_not_require_perf(self) -> None:
        for config_name in NO_PERF_SCALABILITY_CONFIGS:
            with self.subTest(config=config_name):
                payload = json.loads((CONFIG_DIR / config_name).read_text())
                for step in payload["steps"]:
                    self.assertFalse(step.get("perf", False))
                    self.assertNotIn("perf_interval", step)
                    self.assertFalse(step["name"].startswith("perf"))

    def test_multinode_scalability_experiments_use_extralarge_through_64_nodes(self) -> None:
        for config_name in MULTINODE_EXTRALARGE_CONFIGS:
            with self.subTest(config=config_name):
                payload = json.loads((CONFIG_DIR / config_name).read_text())
                multinode_steps = [
                    step for step in payload["steps"] if step.get("nodes") != "1"
                ]
                self.assertTrue(multinode_steps)
                for step in multinode_steps:
                    nodes = {item.strip() for item in step["nodes"].split(",")}
                    self.assertEqual(step["size"], "extralarge")
                    self.assertIn("64", nodes)

    def test_node_sweep_steps_cover_every_requested_node_count(self) -> None:
        for config_name in MULTINODE_EXTRALARGE_CONFIGS:
            if config_name == "scale-multinode-64n-rdma.json":
                continue
            with self.subTest(config=config_name):
                payload = json.loads((CONFIG_DIR / config_name).read_text())
                node_sweep_steps = [
                    step
                    for step in payload["steps"]
                    if step["name"].startswith("node-sweep")
                    or step["name"] in {"multinode-baseline", "multinode-distributed-db"}
                ]
                self.assertTrue(node_sweep_steps)
                for step in node_sweep_steps:
                    self.assertEqual(step["nodes"], FULL_NODE_SWEEP)


if __name__ == "__main__":
    unittest.main()
