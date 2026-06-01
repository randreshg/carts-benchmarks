from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "configs" / "experiments"
PERF_GATE_DIR = REPO_ROOT / "external" / "carts-benchmarks" / "configs" / "perf-gates"
BENCHMARK_ROOT = REPO_ROOT / "external" / "carts-benchmarks"
SKIP_DIRS = {"common", "include", "src", "utilities", ".git", ".svn", ".hg", "build", "logs"}

NO_PERF_SCALABILITY_CONFIGS = [
    "all-enabled-1-to-2-64t-validation.json",
    "all-benchmarks-full-large-extralarge.json",
    "all-benchmarks-multinode-extralarge.json",
    "all-enabled-megalarge.json",
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
    "scale-multinode-1-to-64.json",
    "scale-multinode-64n-rdma.json",
]

FULL_NODE_SWEEP = "1,2,4,8,16,32,64"

CGO_EXPERIMENTS = [
    "cgo-artifact-ci.json",
    "cgo-tires.json",
    "cgo-single-node.json",
    "cgo-multinode-gemm.json",
    "cgo-multinode-blockesd.json",
    "cgo-multinode-sim.json",
]

CGO_PERF_GATES = [
    "cgo-tires-gate.json",
    "cgo-single-node-gate.json",
    "cgo-multinode-gemm-gate.json",
    "cgo-multinode-blockesd-gate.json",
    "cgo-multinode-sim-gate.json",
]


def _discover_enabled_benchmarks() -> set[str]:
    benchmarks: set[str] = set()
    for makefile in BENCHMARK_ROOT.rglob("Makefile"):
        bench_dir = makefile.parent
        rel_path = bench_dir.relative_to(BENCHMARK_ROOT)
        if any(part in SKIP_DIRS for part in rel_path.parts):
            continue
        if not any(bench_dir.glob("*.c")) and not any(bench_dir.glob("*.cpp")):
            continue
        if (bench_dir / ".disabled").exists():
            continue
        benchmarks.add(str(rel_path))
    return benchmarks


def _explicit_experiment_benchmarks(payload: dict) -> set[str]:
    benchmarks: set[str] = set()
    for step in payload.get("steps", []):
        benchmarks.update(step.get("benchmarks", []))
    return benchmarks


def _explicit_perf_gate_benchmarks(payload: dict) -> set[str]:
    benchmarks: set[str] = set()
    for entry in payload.get("benchmarks", []):
        if "name" in entry:
            benchmarks.add(entry["name"])
    return benchmarks


def _node_count_list(step: dict) -> list[int]:
    return [int(item.strip()) for item in step["nodes"].split(",")]


class ExperimentConfigTest(unittest.TestCase):
    def test_explicit_experiment_benchmarks_are_enabled(self) -> None:
        enabled = _discover_enabled_benchmarks()
        for config_path in sorted(CONFIG_DIR.glob("*.json")):
            with self.subTest(config=config_path.name):
                payload = json.loads(config_path.read_text())
                unknown = _explicit_experiment_benchmarks(payload) - enabled
                self.assertEqual(unknown, set())

    def test_perf_gate_benchmarks_are_enabled(self) -> None:
        enabled = _discover_enabled_benchmarks()
        for config_path in sorted(PERF_GATE_DIR.glob("*.json")):
            with self.subTest(config=config_path.name):
                payload = json.loads(config_path.read_text())
                unknown = _explicit_perf_gate_benchmarks(payload) - enabled
                self.assertEqual(unknown, set())

    def test_all_enabled_megalarge_experiment_covers_enabled_benchmarks(self) -> None:
        payload = json.loads((CONFIG_DIR / "all-enabled-megalarge.json").read_text())
        self.assertEqual(payload["name"], "all-enabled-megalarge")
        self.assertEqual(
            [step["name"] for step in payload["steps"]],
            [
                "single-node-reference",
                "multinode-baseline",
                "multinode-distributed-db",
            ],
        )
        enabled = _discover_enabled_benchmarks()
        for step in payload["steps"]:
            with self.subTest(step=step["name"]):
                self.assertEqual(step["size"], "megalarge")
                self.assertEqual(step["threads"], "64")
                self.assertEqual(step["launcher"], "slurm")
                self.assertTrue(step["rdma"])
                self.assertEqual(step["runs"], 1)
                self.assertEqual(step.get("warmup_runs", 0), 0)
                self.assertFalse(step.get("perf", False))
                self.assertEqual(step["profile"], "profile-none.cfg")
                self.assertEqual(set(step["benchmarks"]), enabled)
        self.assertEqual(payload["steps"][0]["nodes"], "1")
        self.assertEqual(payload["steps"][1]["nodes"], "1,2,4,8")
        self.assertEqual(payload["steps"][2]["nodes"], "1,2,4,8")
        self.assertEqual(payload["steps"][2]["compile_args"], "--distributed-db")

    def test_all_enabled_megalarge_dry_run_job_shape_is_stable(self) -> None:
        payload = json.loads((CONFIG_DIR / "all-enabled-megalarge.json").read_text())
        enabled = _discover_enabled_benchmarks()

        jobs_by_step = {
            step["name"]: len(step["benchmarks"])
            * len(_node_count_list(step))
            * step["runs"]
            for step in payload["steps"]
        }

        self.assertEqual(
            jobs_by_step,
            {
                "single-node-reference": len(enabled),
                "multinode-baseline": len(enabled) * 4,
                "multinode-distributed-db": len(enabled) * 4,
            },
        )
        self.assertEqual(sum(jobs_by_step.values()), len(enabled) * 9)

        distributed = payload["steps"][2]
        self.assertEqual(distributed["compile_args"], "--distributed-db")
        self.assertIn(1, _node_count_list(distributed))
        self.assertGreater(max(_node_count_list(distributed)), 1)

    def test_all_enabled_1_to_2_validation_is_serial_gate_shape(self) -> None:
        payload = json.loads(
            (CONFIG_DIR / "all-enabled-1-to-2-64t-validation.json").read_text()
        )
        self.assertEqual(payload["name"], "all-enabled-1-to-2-64t-validation")
        self.assertEqual(
            [step["name"] for step in payload["steps"]],
            [
                "single-node-reference",
                "two-node-baseline",
                "two-node-distributed-db",
            ],
        )

        enabled = _discover_enabled_benchmarks()
        for step in payload["steps"]:
            with self.subTest(step=step["name"]):
                self.assertEqual(set(step["benchmarks"]), enabled)
                self.assertEqual(step["size"], "extralarge")
                self.assertEqual(step["threads"], "64")
                self.assertEqual(step["launcher"], "slurm")
                self.assertTrue(step["rdma"])
                self.assertEqual(step["runs"], 1)
                self.assertEqual(step.get("warmup_runs", 0), 0)
                self.assertFalse(step.get("perf", False))
                self.assertEqual(step["profile"], "profile-none.cfg")
                self.assertEqual(step["timeout"], 270)

        self.assertEqual(payload["steps"][0]["nodes"], "1")
        self.assertEqual(payload["steps"][1]["nodes"], "2")
        self.assertEqual(payload["steps"][2]["nodes"], "2")
        self.assertEqual(payload["steps"][2]["compile_args"], "--distributed-db")

        jobs_by_step = {
            step["name"]: len(step["benchmarks"])
            * len(_node_count_list(step))
            * step["runs"]
            for step in payload["steps"]
        }
        self.assertEqual(
            jobs_by_step,
            {
                "single-node-reference": len(enabled),
                "two-node-baseline": len(enabled),
                "two-node-distributed-db": len(enabled),
            },
        )
        self.assertEqual(sum(jobs_by_step.values()), len(enabled) * 3)

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

    def test_validation_experiment_covers_all_benchmarks_from_one_to_two_nodes(self) -> None:
        payload = json.loads((CONFIG_DIR / "gemm-validation.json").read_text())
        self.assertEqual(payload["name"], "carts-suite-1-to-2-rdma-validation")
        self.assertEqual(
            [step["name"] for step in payload["steps"]],
            [
                "single-node-reference",
                "one-to-two-baseline",
                "one-to-two-distributed-db",
            ],
        )
        for step in payload["steps"]:
            with self.subTest(step=step["name"]):
                self.assertNotIn("benchmarks", step)
                self.assertEqual(step["size"], "extralarge")
                self.assertEqual(step["threads"], "64")
                self.assertTrue(step["rdma"])
                self.assertFalse(step.get("perf", False))
                self.assertEqual(step["runs"], 1)
                self.assertEqual(step["timeout"], 900)
        self.assertEqual(payload["steps"][0]["nodes"], "1")
        self.assertEqual(payload["steps"][1]["nodes"], "1,2")
        self.assertEqual(payload["steps"][2]["nodes"], "1,2")
        self.assertEqual(payload["steps"][2]["compile_args"], "--distributed-db")

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

    def test_cgo_experiments_and_gates_are_present(self) -> None:
        for config_name in CGO_EXPERIMENTS:
            with self.subTest(experiment=config_name):
                payload = json.loads((CONFIG_DIR / config_name).read_text())
                self.assertTrue(payload["name"].startswith("cgo-"))
                self.assertTrue(payload["steps"])
                for step in payload["steps"]:
                    self.assertIn("runs", step)
                    self.assertGreater(step["runs"], step.get("warmup_runs", 0))

        for gate_name in CGO_PERF_GATES:
            with self.subTest(gate=gate_name):
                payload = json.loads((PERF_GATE_DIR / gate_name).read_text())
                self.assertTrue(payload["name"].startswith("cgo-"))
                self.assertTrue(payload["benchmarks"])

    def test_cgo_capacity_steps_use_problem_size_override(self) -> None:
        payload = json.loads((CONFIG_DIR / "cgo-multinode-gemm.json").read_text())
        capacity_steps = [
            step for step in payload["steps"] if step["name"].startswith("capacity-")
        ]
        self.assertEqual(len(capacity_steps), 5)
        self.assertEqual(
            [step["problem_size_n"] for step in capacity_steps],
            [8192, 16384, 32768, 65536, 131072],
        )

    def test_cgo_artifact_ci_is_lightweight_local_smoke(self) -> None:
        payload = json.loads((CONFIG_DIR / "cgo-artifact-ci.json").read_text())
        self.assertEqual(payload["name"], "cgo-artifact-ci")
        self.assertEqual(len(payload["steps"]), 1)
        step = payload["steps"][0]
        self.assertEqual(step["name"], "local-smoke")
        self.assertEqual(step["benchmarks"], ["polybench/gemm", "polybench/atax"])
        self.assertEqual(step["size"], "small")
        self.assertEqual(step["threads"], "2")
        self.assertEqual(step["nodes"], "1")
        self.assertEqual(step["launcher"], "local")
        self.assertEqual(step["arts_config"], "local.cfg")
        self.assertFalse(step["rdma"])
        self.assertEqual(step["runs"], 1)
        self.assertEqual(step.get("warmup_runs", 0), 0)
        self.assertFalse(step.get("perf", False))


if __name__ == "__main__":
    unittest.main()
