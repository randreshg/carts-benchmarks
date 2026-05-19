"""Shared CARTS path resolution for benchmark tooling.

Benchmark scripts run both through ``dekk carts ...`` and as standalone helper
programs inside generated Slurm jobs.  Keep CARTS root, CARTS_HOME, install
roots, build caches, and runtime library paths in one place so subcommands do
not grow their own checkout-local assumptions.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class CartsPaths:
    carts_dir: Path
    carts_home: Path
    build_dir: Path
    install_dir: Path
    carts_build_dir: Path
    arts_build_dir: Path
    llvm_build_dir: Path
    polygeist_build_dir: Path
    carts_install_dir: Path
    arts_install_dir: Path
    llvm_install_dir: Path
    polygeist_install_dir: Path


def get_carts_dir() -> Path:
    """Return the Dekk-provided CARTS checkout root."""
    env_dir = os.environ.get("CARTS_DIR")
    if not env_dir:
        raise RuntimeError("CARTS_DIR is not set. Run through `dekk carts ...`.")

    candidate = Path(env_dir).expanduser().resolve()
    if not (candidate / "tools" / "carts_cli.py").is_file():
        raise RuntimeError(f"CARTS_DIR does not look like a CARTS checkout: {candidate}")
    return candidate


def _load_project_local_config(carts_dir: Path):
    """Load project-owned local_config.py for all build/install paths."""
    tools_dir = carts_dir / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from scripts import local_config

    return local_config


def resolve_carts_home(carts_dir: Optional[Path] = None) -> Path:
    """Resolve CARTS_HOME with the project-owned precedence rules."""
    root = (carts_dir or get_carts_dir()).resolve()
    return _load_project_local_config(root).resolve_carts_home(root)


def get_carts_paths(carts_dir: Optional[Path] = None) -> CartsPaths:
    root = (carts_dir or get_carts_dir()).resolve()
    local_config = _load_project_local_config(root)
    home = local_config.resolve_carts_home(root)
    build_dir = local_config.resolve_build_dir(root)
    install_dir = local_config.resolve_install_dir(root)
    return CartsPaths(
        carts_dir=root,
        carts_home=home,
        build_dir=build_dir,
        install_dir=install_dir,
        carts_build_dir=local_config.resolve_subproject_build_dir(root, "carts"),
        arts_build_dir=local_config.resolve_subproject_build_dir(root, "arts"),
        llvm_build_dir=local_config.resolve_subproject_build_dir(root, "llvm-project"),
        polygeist_build_dir=local_config.resolve_subproject_build_dir(root, "polygeist"),
        carts_install_dir=local_config.resolve_subproject_install_dir(root, "carts"),
        arts_install_dir=local_config.resolve_subproject_install_dir(root, "arts"),
        llvm_install_dir=local_config.resolve_subproject_install_dir(root, "llvm"),
        polygeist_install_dir=local_config.resolve_subproject_install_dir(root, "polygeist"),
    )


def active_install_dir(carts_dir: Optional[Path] = None) -> Path:
    return get_carts_paths(carts_dir).install_dir


def active_arts_cmake_cache(carts_dir: Optional[Path] = None) -> Path:
    paths = get_carts_paths(carts_dir)
    return paths.arts_build_dir / "CMakeCache.txt"


def managed_runtime_library_dirs(carts_dir: Optional[Path] = None) -> List[Path]:
    """Return Dekk/CARTS-managed shared-library dirs needed by binaries."""
    root = (carts_dir or get_carts_dir()).resolve()
    return _load_project_local_config(root).managed_runtime_library_dirs(root)


def managed_runtime_library_env(carts_dir: Optional[Path] = None) -> str:
    return os.pathsep.join(str(path) for path in managed_runtime_library_dirs(carts_dir))
