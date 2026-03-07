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
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    logs, and a manifest.  Both ``run`` and ``run --slurm`` share the same
    canonical layout:

        {base_results_dir}/{timestamp}/
          manifest.json
          results.json
          {benchmark_name}/
            {threads}t_{nodes}n/
              artifacts/          # build outputs + generated config
                arts.cfg          # generated runtime config
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
        self._phase_label: Optional[str] = None

    def set_phase(self, phase_label: Optional[str] = None) -> None:
        self._phase_label = phase_label

    # -- directory getters (create on first access) --------------------------

    def get_config_dir(self, benchmark_name: str, config: BenchmarkConfig) -> Path:
        config_label = f"{config.arts_threads}t_{config.arts_nodes}n"
        base = self.experiment_dir / self._phase_label if self._phase_label else self.experiment_dir
        d = base / benchmark_name / config_label
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

    def _reference_key_token(
        self,
        size: str,
        omp_threads: int,
        cflags: Optional[str],
    ) -> str:
        cflags_token = hashlib.sha1((cflags or "").encode()).hexdigest()[:8]
        return f"size_{size}/{omp_threads}t_{cflags_token}"

    def get_reference_root(
        self,
        benchmark_name: str,
        size: str,
        omp_threads: int,
        cflags: Optional[str] = None,
    ) -> Path:
        d = (
            self.experiment_dir
            / "references"
            / Path(benchmark_name)
            / self._reference_key_token(size, omp_threads, cflags)
        )
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_reference_artifacts_dir(
        self,
        benchmark_name: str,
        size: str,
        omp_threads: int,
        cflags: Optional[str] = None,
    ) -> Path:
        d = self.get_reference_root(benchmark_name, size, omp_threads, cflags) / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_reference_run_dir(
        self,
        benchmark_name: str,
        size: str,
        omp_threads: int,
        cflags: Optional[str] = None,
    ) -> Path:
        d = self.get_reference_root(benchmark_name, size, omp_threads, cflags) / "run_1"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load_reference_result(
        self,
        benchmark_name: str,
        size: str,
        omp_threads: int,
        cflags: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        ref_file = self.get_reference_root(
            benchmark_name, size, omp_threads, cflags
        ) / "reference.json"
        if not ref_file.exists():
            return None
        try:
            payload = json.loads(ref_file.read_text())
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def save_reference_result(
        self,
        benchmark_name: str,
        size: str,
        omp_threads: int,
        cflags: Optional[str],
        payload: Dict[str, Any],
    ) -> Path:
        ref_dir = self.get_reference_root(benchmark_name, size, omp_threads, cflags)
        ref_file = ref_dir / "reference.json"
        with open(ref_file, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return ref_file

    def save_run_config(
        self,
        benchmark_name: str,
        config: BenchmarkConfig,
        run_number: int,
        arts_cfg_path: Optional[Path] = None,
        *,
        command: Optional[str] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        size: Optional[str] = None,
        cflags: Optional[str] = None,
        compile_args: Optional[str] = None,
        run_phase: Optional[str] = None,
        profile: Optional[str] = None,
        perf: Optional[bool] = None,
        perf_interval: Optional[float] = None,
        reference_checksum: Optional[str] = None,
        reference_source: Optional[str] = None,
        reference_threads: Optional[int] = None,
    ) -> Path:
        """Save the effective arts.cfg and a run_config.json into the run directory.

        This makes each run directory fully self-contained with all configuration
        needed to reproduce the run.
        """
        run_dir = self.get_run_dir(benchmark_name, config, run_number)

        # Write run_config.json with full execution context
        run_config: Dict[str, object] = {
            "benchmark": benchmark_name,
            "run_number": run_number,
            "threads": config.arts_threads,
            "nodes": config.arts_nodes,
            "config": {
                "arts_threads": config.arts_threads,
                "arts_nodes": config.arts_nodes,
                "omp_threads": config.omp_threads,
                "launcher": config.launcher,
            },
        }
        if size is not None:
            run_config["size"] = size
        if cflags:
            run_config["cflags"] = cflags
        if compile_args:
            run_config["compile_args"] = compile_args
        if run_phase:
            run_config["run_phase"] = run_phase
        if profile:
            run_config["profile"] = profile
        if perf is not None:
            run_config["perf"] = perf
        if perf_interval is not None:
            run_config["perf_interval"] = perf_interval
        if command:
            run_config["command"] = command
        if env_overrides:
            run_config["env_overrides"] = env_overrides
        if arts_cfg_path:
            run_config["arts_cfg_source"] = str(arts_cfg_path)
        if (
            reference_checksum is not None
            or reference_source is not None
            or reference_threads is not None
        ):
            reference: Dict[str, object] = {}
            if reference_checksum is not None:
                reference["checksum"] = reference_checksum
            if reference_source is not None:
                reference["source"] = reference_source
            if reference_threads is not None:
                reference["omp_threads"] = reference_threads
            run_config["reference"] = reference
        run_config["timestamp"] = datetime.now().isoformat()

        config_path = run_dir / "run_config.json"
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=2, default=str)

        return config_path

    # -- artifact operations -------------------------------------------------

    def copy_build_artifacts(
        self,
        bench_path: Path,
        benchmark_name: str,
        config: BenchmarkConfig,
        build_source_dir: Optional[Path] = None,
    ) -> Dict[str, Optional[str]]:
        """Copy build artifacts into the experiment's ``artifacts/`` directory."""
        artifacts_dir = self.get_artifacts_dir(benchmark_name, config)
        paths: Dict[str, Optional[str]] = {}
        source_dir = (
            build_source_dir
            if build_source_dir is not None and build_source_dir.exists()
            else bench_path
        )

        # .carts-metadata.json (compiler metadata)
        metadata = source_dir / ".carts-metadata.json"
        if metadata.exists():
            dest = artifacts_dir / ".carts-metadata.json"
            shutil.copy2(metadata, dest)
            paths["carts_metadata"] = str(dest)

        # *_arts_metadata.mlir
        for mlir in source_dir.glob("*_arts_metadata.mlir"):
            dest = artifacts_dir / mlir.name
            shutil.copy2(mlir, dest)
            paths["arts_metadata_mlir"] = str(dest)

        # Other MLIR files
        for mlir in source_dir.glob("*.mlir"):
            if "_metadata" not in mlir.name:
                shutil.copy2(mlir, artifacts_dir / mlir.name)

        # LLVM IR
        for ll in source_dir.glob("*-arts.ll"):
            shutil.copy2(ll, artifacts_dir / ll.name)

        # Executables
        for exe in source_dir.glob("*_arts"):
            if exe.is_file():
                dest = artifacts_dir / exe.name
                shutil.copy2(exe, dest)
                paths["executable_arts"] = str(dest)
        for exe in list(source_dir.glob("*_omp")) + list(
            (source_dir / "build").glob("*_omp")
        ):
            if exe.is_file():
                dest = artifacts_dir / exe.name
                shutil.copy2(exe, dest)
                paths["executable_omp"] = str(dest)

        # Build logs
        logs_dir = source_dir / "logs"
        if not logs_dir.exists():
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
