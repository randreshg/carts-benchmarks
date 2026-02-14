"""
Benchmark metadata — system info and reproducibility data collection.

Pure I/O functions for capturing git hashes, compiler versions, CPU info,
and other reproducibility metadata. Used by ArtifactManager and CLI export.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark_models import ParallelTaskTiming

logger = logging.getLogger(__name__)


def _get_carts_dir() -> Path:
    """Get the CARTS root directory (local helper to avoid circular imports)."""
    script_dir = Path(__file__).parent.resolve()
    carts_dir = script_dir.parent.parent.parent
    if not (carts_dir / "tools" / "carts").exists():
        env_dir = os.environ.get("CARTS_DIR")
        if env_dir:
            carts_dir = Path(env_dir)
    return carts_dir


def get_git_hash(repo_path: Path) -> Optional[str]:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        logger.debug("Failed to get git hash for %s", repo_path, exc_info=True)
    return None


def get_compiler_version() -> Dict[str, Optional[str]]:
    """Get compiler version information.

    Prioritizes CARTS-installed LLVM/clang over system compilers,
    since CARTS builds LLVM from source.
    """
    compilers = {}
    carts_dir = _get_carts_dir()

    # Try CARTS-installed LLVM clang first (built from source)
    carts_clang = carts_dir / ".install" / "llvm" / "bin" / "clang"
    clang_paths = [str(carts_clang), "clang"]

    for clang_path in clang_paths:
        try:
            result = subprocess.run(
                [clang_path, "--version"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                first_line = result.stdout.split('\n')[0]
                compilers["clang"] = first_line
                # Record which clang was found
                if clang_path != "clang":
                    compilers["clang_path"] = clang_path
                break
        except Exception:
            logger.debug("Failed to get clang version from %s", clang_path, exc_info=True)
            continue

    # Try gcc
    try:
        result = subprocess.run(
            ["gcc", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            first_line = result.stdout.split('\n')[0]
            compilers["gcc"] = first_line
    except Exception:
        logger.debug("Failed to get gcc version", exc_info=True)

    return compilers


def get_cpu_info() -> Dict[str, Any]:
    """Get CPU information for reproducibility."""
    cpu_info = {}

    system = platform.system().lower()

    if system == "darwin":
        # macOS: use sysctl
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                cpu_info["model"] = result.stdout.strip()

            result = subprocess.run(
                ["sysctl", "-n", "hw.ncpu"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                cpu_info["cores"] = int(result.stdout.strip())

            result = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                cpu_info["physical_cores"] = int(result.stdout.strip())
        except Exception:
            logger.debug("Failed to get macOS CPU info", exc_info=True)
    elif system == "linux":
        # Linux: parse /proc/cpuinfo
        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read()
            for line in content.split('\n'):
                if line.startswith("model name"):
                    cpu_info["model"] = line.split(":")[1].strip()
                    break

            result = subprocess.run(["nproc"], capture_output=True, text=True)
            if result.returncode == 0:
                cpu_info["cores"] = int(result.stdout.strip())
        except Exception:
            logger.debug("Failed to get Linux CPU info", exc_info=True)

    return cpu_info


def get_reproducibility_metadata(carts_dir: Path, benchmarks_dir: Path) -> Dict[str, Any]:
    """Collect comprehensive reproducibility metadata.

    This captures all information needed to reproduce benchmark results:
    - Git commit hashes for all repositories
    - Compiler versions
    - CPU and system information
    - Relevant environment variables
    """
    metadata = {}

    # Git hashes
    metadata["git_commits"] = {
        "carts": get_git_hash(carts_dir) or "unknown",
        "carts_benchmarks": get_git_hash(benchmarks_dir) or "unknown",
    }

    # Check for ARTS submodule
    arts_dir = carts_dir / "external" / "arts"
    if arts_dir.exists():
        metadata["git_commits"]["arts"] = get_git_hash(arts_dir) or "unknown"

    # Compiler versions
    metadata["compilers"] = get_compiler_version()

    # CPU info
    metadata["cpu"] = get_cpu_info()

    # System info
    metadata["system"] = {
        "os": platform.system(),
        "os_version": platform.release(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }

    # Relevant environment variables
    env_vars_to_capture = [
        "OMP_NUM_THREADS",
        "OMP_PROC_BIND",
        "OMP_PLACES",
        "CARTS_DIR",
        "ARTS_DIR",
        "CC",
        "CXX",
        "CFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
    ]
    metadata["environment"] = {
        var: os.environ.get(var) for var in env_vars_to_capture if os.environ.get(var)
    }

    return metadata


def _serialize_parallel_task_timing(timing: Optional[ParallelTaskTiming]) -> Optional[Dict]:
    """Serialize ParallelTaskTiming to JSON-compatible dict."""
    if timing is None:
        return None

    return {
        "parallel_timings": {
            name: [{"worker_id": t.worker_id, "time_sec": t.time_sec}
                   for t in timings]
            for name, timings in timing.parallel_timings.items()
        },
        "task_timings": {
            name: [{"worker_id": t.worker_id, "time_sec": t.time_sec}
                   for t in timings]
            for name, timings in timing.task_timings.items()
        },
    }
