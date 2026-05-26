"""Runtime-mode detection and launch overrides for CARTS ARTS artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from arts_config import KEY_PIN, KEY_WORKER_THREADS

ARTS_RUNTIME_MODE_TASK = "arts_task_runtime"
ARTS_RUNTIME_MODE_HOST_OPENMP = "host_openmp_fallback"
ARTS_RUNTIME_MODE_HOST_SERIAL = "host_serial_fallback"
ARTS_RUNTIME_MODE_UNKNOWN = "unknown"
ARTS_RUNTIME_MODE_SOURCE_MISSING = "llvm_ir_missing"
ARTS_RUNTIME_MODE_SOURCE_AMBIGUOUS = "llvm_ir_ambiguous"

ARTS_EPOCH_SYMBOL = "arts_initialize_and_start_epoch"
HOST_OPENMP_MARKER_SYMBOL = "carts_benchmarks_mark_host_openmp"
OMP_DIALECT_TOKEN = "omp."

HOST_OPENMP_RUNTIME_ARTS_OVERRIDES = {
    KEY_WORKER_THREADS: "1",
    KEY_PIN: "0",
}
HOST_OPENMP_RUNTIME_ENV_OVERRIDES = {
    "OMP_WAIT_POLICY": "ACTIVE",
}

HOST_FALLBACK_RUNTIME_MODES = frozenset(
    {
        ARTS_RUNTIME_MODE_HOST_OPENMP,
        ARTS_RUNTIME_MODE_HOST_SERIAL,
    }
)


def infer_arts_runtime_mode(
    executable_arts: Optional[Union[str, Path]],
) -> Tuple[str, str]:
    """Infer whether a CARTS artifact enters the ARTS task runtime.

    Some benchmark sources intentionally keep unsupported OpenMP regions on the
    host.  They still produce a ``*_arts`` executable for apples-to-apples
    compile plumbing, but those binaries use libomp for parallelism.  Persisting
    this mode keeps reports honest and lets launchers avoid an idle ARTS worker
    pool contending with OpenMP threads.
    """
    if not executable_arts:
        return ARTS_RUNTIME_MODE_UNKNOWN, ARTS_RUNTIME_MODE_SOURCE_MISSING

    executable = Path(executable_arts)
    build_dir = executable.parent
    ir_files = sorted(build_dir.glob("*-arts.ll"))
    if not ir_files:
        return ARTS_RUNTIME_MODE_UNKNOWN, ARTS_RUNTIME_MODE_SOURCE_MISSING
    if len(ir_files) > 1:
        executable_stem = executable.name
        if executable_stem.endswith("_arts"):
            candidate = build_dir / f"{executable_stem[:-5]}-arts.ll"
            if candidate.exists():
                ir_files = [candidate]
            else:
                return ARTS_RUNTIME_MODE_UNKNOWN, ARTS_RUNTIME_MODE_SOURCE_AMBIGUOUS

    ir_path = ir_files[0]
    try:
        text = ir_path.read_text(errors="replace")
    except OSError:
        return ARTS_RUNTIME_MODE_UNKNOWN, str(ir_path)

    if ARTS_EPOCH_SYMBOL in text:
        return ARTS_RUNTIME_MODE_TASK, str(ir_path)
    if HOST_OPENMP_MARKER_SYMBOL in text or OMP_DIALECT_TOKEN in text:
        return ARTS_RUNTIME_MODE_HOST_OPENMP, str(ir_path)
    return ARTS_RUNTIME_MODE_HOST_SERIAL, str(ir_path)


def runtime_overrides_for_arts_mode(
    arts_runtime_mode: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return ARTS config and environment overrides for a runtime mode."""
    if arts_runtime_mode != ARTS_RUNTIME_MODE_HOST_OPENMP:
        return {}, {}
    return (
        dict(HOST_OPENMP_RUNTIME_ARTS_OVERRIDES),
        dict(HOST_OPENMP_RUNTIME_ENV_OVERRIDES),
    )


def is_host_fallback_runtime_mode(arts_runtime_mode: str) -> bool:
    """Return true when the artifact does not enter distributed ARTS tasks."""
    return arts_runtime_mode in HOST_FALLBACK_RUNTIME_MODES


def apply_arts_cfg_overrides(content: str, overrides: Dict[str, str]) -> str:
    """Apply key=value overrides to an arts.cfg payload."""
    updated = content
    for key, value in overrides.items():
        replacement = f"{key}={value}"
        pattern = rf"^{re.escape(key)}\s*=.*$"
        if re.search(pattern, updated, re.MULTILINE):
            updated = re.sub(pattern, replacement, updated, flags=re.MULTILINE)
        elif "[ARTS]" in updated:
            updated = updated.replace("[ARTS]", f"[ARTS]\n{replacement}", 1)
        else:
            suffix = "" if updated.endswith("\n") else "\n"
            updated = f"{updated}{suffix}{replacement}\n"
    return updated


def write_runtime_arts_config(
    source_path: Path,
    destination_path: Path,
    overrides: Dict[str, str],
) -> Path:
    """Write the effective runtime arts.cfg used by a benchmark launch."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        apply_arts_cfg_overrides(source_path.read_text(), overrides)
    )
    return destination_path
