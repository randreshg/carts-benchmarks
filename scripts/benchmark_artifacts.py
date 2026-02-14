"""
Benchmark artifact manager — manages experiment directories and build artifacts.

Self-contained class for organizing benchmark outputs into a canonical directory
layout with manifests. Used at boundaries: passed into BenchmarkRunner, called
post-run from CLI.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from benchmark_models import BenchmarkConfig, BenchmarkResult, Status
from benchmark_metadata import get_reproducibility_metadata


def _get_carts_dir() -> Path:
    """Get the CARTS root directory (local helper to avoid circular imports)."""
    script_dir = Path(__file__).parent.resolve()
    carts_dir = script_dir.parent.parent.parent
    if not (carts_dir / "tools" / "carts").exists():
        env_dir = os.environ.get("CARTS_DIR")
        if env_dir:
            carts_dir = Path(env_dir)
    return carts_dir


def _get_benchmarks_dir() -> Path:
    """Get the benchmarks directory (local helper to avoid circular imports)."""
    return Path(__file__).parent.parent.resolve()


class ArtifactManager:
    """Manages a self-contained artifact directory for a benchmark experiment.

    Every execution produces a single timestamped directory with all artifacts,
    logs, and a manifest.  Both ``run`` and ``slurm-run`` share the same
    canonical layout:

        {base_results_dir}/{timestamp}/
          manifest.json
          results.json
          {benchmark_name}/
            {threads}t_{nodes}n/
              artifacts/          # build outputs (shared across runs)
              run_1/              # per-run outputs
                arts.log
                omp.log
                counters/         # if counters enabled
                perf/             # if perf enabled
              run_2/
    """

    def __init__(self, base_results_dir: Path, timestamp: str):
        self.experiment_dir = base_results_dir / timestamp
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.results_json_path = self.experiment_dir / "results.json"
        self.manifest_path = self.experiment_dir / "manifest.json"
        self._manifest_benchmarks: Dict[str, Dict] = {}

    # -- directory getters (create on first access) --------------------------

    def get_config_dir(self, benchmark_name: str, config: BenchmarkConfig) -> Path:
        config_label = f"{config.arts_threads}t_{config.arts_nodes}n"
        d = self.experiment_dir / benchmark_name / config_label
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_artifacts_dir(self, benchmark_name: str, config: BenchmarkConfig) -> Path:
        d = self.get_config_dir(benchmark_name, config) / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_run_dir(self, benchmark_name: str, config: BenchmarkConfig, run_number: int) -> Path:
        d = self.get_config_dir(benchmark_name, config) / f"run_{run_number}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_counter_dir(self, benchmark_name: str, config: BenchmarkConfig, run_number: int) -> Path:
        d = self.get_run_dir(benchmark_name, config, run_number) / "counters"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_perf_dir(self, benchmark_name: str, config: BenchmarkConfig, run_number: int) -> Path:
        d = self.get_run_dir(benchmark_name, config, run_number) / "perf"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- artifact operations -------------------------------------------------

    def copy_build_artifacts(
        self,
        bench_path: Path,
        benchmark_name: str,
        config: BenchmarkConfig,
        arts_cfg_used: Optional[Path] = None,
    ) -> Dict[str, Optional[str]]:
        """Copy build artifacts into the experiment's ``artifacts/`` directory."""
        artifacts_dir = self.get_artifacts_dir(benchmark_name, config)
        paths: Dict[str, Optional[str]] = {}

        # arts.cfg (build INPUT)
        cfg_src = (
            arts_cfg_used if arts_cfg_used and arts_cfg_used.exists()
            else (bench_path / "arts.cfg")
        )
        if cfg_src.exists():
            dest = artifacts_dir / "arts.cfg"
            shutil.copy2(cfg_src, dest)
            paths["arts_config"] = str(dest)

        # .carts-metadata.json (compiler metadata)
        metadata = bench_path / ".carts-metadata.json"
        if metadata.exists():
            dest = artifacts_dir / ".carts-metadata.json"
            shutil.copy2(metadata, dest)
            paths["carts_metadata"] = str(dest)

        # *_arts_metadata.mlir
        for mlir in bench_path.glob("*_arts_metadata.mlir"):
            dest = artifacts_dir / mlir.name
            shutil.copy2(mlir, dest)
            paths["arts_metadata_mlir"] = str(dest)

        # Other MLIR files
        for mlir in bench_path.glob("*.mlir"):
            if "_metadata" not in mlir.name:
                shutil.copy2(mlir, artifacts_dir / mlir.name)

        # LLVM IR
        for ll in bench_path.glob("*-arts.ll"):
            shutil.copy2(ll, artifacts_dir / ll.name)

        # Executables
        for exe in bench_path.glob("*_arts"):
            if exe.is_file():
                dest = artifacts_dir / exe.name
                shutil.copy2(exe, dest)
                paths["executable_arts"] = str(dest)
        for exe in list(bench_path.glob("*_omp")) + list(
            (bench_path / "build").glob("*_omp")
        ):
            if exe.is_file():
                dest = artifacts_dir / exe.name
                shutil.copy2(exe, dest)
                paths["executable_omp"] = str(dest)

        # Build logs
        logs_dir = bench_path / "logs"
        if logs_dir.exists():
            for log in ["build_arts.log", "build_openmp.log"]:
                log_file = logs_dir / log
                if log_file.exists():
                    shutil.copy2(log_file, artifacts_dir / log)

        return paths

    def record_run(
        self,
        benchmark_name: str,
        config: BenchmarkConfig,
        run_number: int,
        has_counters: bool = False,
        has_perf: bool = False,
    ):
        """Track a completed run for the manifest."""
        config_label = f"{config.arts_threads}t_{config.arts_nodes}n"
        if benchmark_name not in self._manifest_benchmarks:
            self._manifest_benchmarks[benchmark_name] = {"configs": {}}
        configs = self._manifest_benchmarks[benchmark_name]["configs"]
        if config_label not in configs:
            configs[config_label] = {
                "artifacts": str(Path(benchmark_name) / config_label / "artifacts"),
                "runs": {},
            }
        configs[config_label]["runs"][str(run_number)] = {
            "path": str(Path(benchmark_name) / config_label / f"run_{run_number}"),
            "has_counters": has_counters,
            "has_perf": has_perf,
        }

    def write_manifest(
        self,
        results: List[BenchmarkResult],
        command: str,
        total_duration: float,
    ) -> Path:
        """Write ``manifest.json`` — a structure index and quick summary."""
        import math
        from collections import Counter as _Counter

        passed = sum(
            1 for r in results
            if r.run_arts.status == Status.PASS and r.verification.correct
        )
        failed = sum(
            1 for r in results
            if r.run_arts.status in (Status.FAIL, Status.CRASH)
        )
        total_benchmarks = len(set(r.name for r in results))
        total_configs = len(
            set((r.name, r.config.arts_threads, r.config.arts_nodes) for r in results)
        )
        config_counts = _Counter(
            (r.name, r.config.arts_threads, r.config.arts_nodes) for r in results
        )
        runs_per_config = max(config_counts.values()) if config_counts else 1
        speedups = [r.timing.speedup for r in results if r.timing.speedup > 0]
        geomean = (
            math.exp(sum(math.log(s) for s in speedups) / len(speedups))
            if speedups else 0.0
        )

        carts_dir = _get_carts_dir()
        benchmarks_dir = _get_benchmarks_dir()
        repro = get_reproducibility_metadata(carts_dir, benchmarks_dir)

        manifest = {
            "version": 1,
            "created": datetime.now().isoformat(),
            "command": command,
            "layout": {
                "results_json": "results.json",
                "benchmarks": self._manifest_benchmarks,
            },
            "summary": {
                "total_benchmarks": total_benchmarks,
                "total_configs": total_configs,
                "runs_per_config": runs_per_config,
                "passed": passed,
                "failed": failed,
                "geometric_mean_speedup": round(geomean, 4),
                "total_duration_sec": round(total_duration, 1),
            },
            "reproducibility": repro,
        }

        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        return self.manifest_path
