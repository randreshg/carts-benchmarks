"""ARTS configuration file parser, key definitions, and manipulation utilities.

Key names mirror the authoritative ``config_entries[]`` table in
``external/arts/libs/src/core/system/config.c``.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Threading
KEY_WORKER_THREADS = "worker_threads"
KEY_STACK_SIZE = "stack_size"

# Pinning
KEY_PIN = "pin"

# Scheduling
KEY_SCHEDULER = "scheduler"
KEY_DEQUE_TYPE = "deque_type"
KEY_WORKER_INIT_DEQUE_SIZE = "worker_init_deque_size"
KEY_ROUTE_TABLE_SIZE = "route_table_size"
KEY_AUTO_SHUTDOWN = "auto_shutdown"
KEY_MIN_ITERATIONS_PER_WORKER = "min_iterations_per_worker"
KEY_MIN_DISTRIBUTED_TILE_BYTES = "min_distributed_tile_bytes"

# GPU
KEY_GPU = "gpu"
KEY_GPU_LOCALITY = "gpu_locality"
KEY_GPU_FIT = "gpu_fit"
KEY_GPU_LC_SYNC = "gpu_lc_sync"
KEY_GPU_MAX_EDTS = "gpu_max_edts"
KEY_GPU_MAX_MEMORY = "gpu_max_memory"
KEY_GPU_P2P = "gpu_p2p"
KEY_GPU_ROUTE_TABLE_SIZE = "gpu_route_table_size"
KEY_FREE_DB_AFTER_GPU_RUN = "free_db_after_gpu_run"
KEY_RUN_GPU_GC_IDLE = "run_gpu_gc_idle"
KEY_RUN_GPU_GC_PRE_EDT = "run_gpu_gc_pre_edt"
KEY_DELETE_ZEROS_GPU_GC = "delete_zeros_gpu_gc"
KEY_GPU_BUFF_ON = "gpu_buff_on"

# Networking
KEY_SENDER_THREADS = "sender_threads"
KEY_RECEIVER_THREADS = "receiver_threads"
KEY_PORT_COUNT = "port_count"
KEY_MASTER_NODE = "master_node"
KEY_DEFAULT_PORTS = "default_ports"
KEY_NET_INTERFACE = "net_interface"

# Launcher / nodes
KEY_LAUNCHER = "launcher"
KEY_NODE_COUNT = "node_count"
KEY_NODES = "nodes"

# Debug
KEY_KILL_MODE = "kill_mode"
KEY_CORE_DUMP = "core_dump"

# Counters
KEY_COUNTER_FOLDER = "counter_folder"
KEY_COUNTER_CAPTURE_INTERVAL = "counter_capture_interval"

# Protocol
KEY_PROTOCOL = "protocol"
PROTOCOL_TCP = "tcp"
PROTOCOL_RDMA = "rdma"
PROTOCOL_ROCE = "roce"
PROTOCOL_AUTO = "auto"
SUPPORTED_PROTOCOLS = frozenset(
    (PROTOCOL_TCP, PROTOCOL_RDMA, PROTOCOL_ROCE, PROTOCOL_AUTO)
)


def protocol_for_rdma(enabled: bool) -> str:
    """Return the benchmark runtime protocol for an RDMA toggle."""
    return PROTOCOL_RDMA if enabled else PROTOCOL_TCP


def rdma_for_node_count(enabled: bool, node_count: int) -> bool:
    """Return whether a run should request RDMA transport for this node count."""
    return enabled and node_count > 1


def protocol_for_node_count(enabled: bool, node_count: int) -> str:
    """Return the transport protocol to write into arts.cfg for a node count."""
    return protocol_for_rdma(rdma_for_node_count(enabled, node_count))


def rdma_for_launcher(enabled: bool, node_count: int, launcher: Optional[str]) -> bool:
    """Return whether a launcher/node-count pair should request RDMA transport."""
    if str(launcher or "").strip().lower() == "local":
        return False
    return rdma_for_node_count(enabled, node_count)


def protocol_for_launcher(enabled: bool, node_count: int, launcher: Optional[str]) -> str:
    """Return the transport protocol for a launcher/node-count pair."""
    return protocol_for_rdma(rdma_for_launcher(enabled, node_count, launcher))


def compile_args_for_node_count(
    compile_args: Optional[str],
    node_count: int,
) -> Optional[str]:
    """Remove multinode-only compiler flags for single-node benchmark builds.

    Set CARTS_KEEP_DISTRIBUTED_DB_1N=1 to disable stripping --distributed-db at
    1n. Needed for apples-to-apples 1n vs 2n scaling comparisons where both
    runs must take the same compile path.
    """
    if not compile_args:
        return compile_args
    if node_count > 1:
        return compile_args
    if os.environ.get("CARTS_KEEP_DISTRIBUTED_DB_1N", "").strip() in {"1", "true", "yes"}:
        return compile_args

    try:
        tokens = shlex.split(compile_args)
    except ValueError:
        tokens = compile_args.split()
    filtered = [token for token in tokens if token != "--distributed-db"]
    if not filtered:
        return None
    return shlex.join(filtered)

EMBEDDED_KEYS: List[str] = [
    KEY_WORKER_THREADS,
    KEY_RECEIVER_THREADS,
    KEY_SENDER_THREADS,
    KEY_NODE_COUNT,
    KEY_LAUNCHER,
    KEY_PROTOCOL,
    KEY_PORT_COUNT,
    KEY_PIN,
    KEY_DEFAULT_PORTS,
]


def parse_arts_cfg(path: Optional[Path]) -> Dict[str, str]:
    """Parse an ARTS config file into a key/value dict."""
    if not path or not path.exists():
        return {}

    values: Dict[str, str] = {}
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.split("#", 1)[0].split(";", 1)[0].strip()
            if key:
                values[key] = value
    except Exception:
        return {}
    return values


def get_cfg_int(path: Optional[Path], key: str) -> Optional[int]:
    """Return an arts.cfg value as int, or None if missing/unparseable."""
    vals = parse_arts_cfg(path)
    if key not in vals:
        return None
    try:
        return int(vals[key])
    except Exception:
        logger.debug("Failed to convert arts.cfg key '%s' to int", key, exc_info=True)
        return None


def get_cfg_str(path: Optional[Path], key: str) -> Optional[str]:
    """Return an arts.cfg value as str, or None if missing/empty."""
    value = parse_arts_cfg(path).get(key)
    return value if value else None


def get_cfg_nodes(path: Optional[Path]) -> List[str]:
    """Parse the ``nodes=`` field into a list of hostnames."""
    cfg = parse_arts_cfg(path)
    nodes_str = cfg.get(KEY_NODES, "localhost")
    return [node.strip() for node in nodes_str.split(",") if node.strip()]


def upsert_cfg_value(content: str, key: str, value: object) -> str:
    """Replace or insert one top-level ARTS config assignment."""
    line = f"{key}={value}"
    pattern = rf"^{re.escape(key)}\s*=.*$"
    if re.search(pattern, content, re.MULTILINE):
        return re.sub(pattern, line, content, flags=re.MULTILINE)
    return content.replace("[ARTS]", f"[ARTS]\n{line}", 1)


def comment_cfg_key(content: str, key: str, reason: str) -> str:
    """Comment out a key in arts.cfg content."""
    pattern = rf"^{re.escape(key)}\s*=.*$"
    if re.search(pattern, content, re.MULTILINE):
        return re.sub(pattern, f"# {key}= ({reason})", content, flags=re.MULTILINE)
    return content


def extract_embedded_cfg(artifacts_dir: Path) -> Dict[str, str]:
    """Extract embedded ARTS config values from LLVM IR or executable output."""
    candidates = sorted(artifacts_dir.glob("*-arts.ll"))
    if not candidates:
        candidates = sorted(artifacts_dir.glob("*_arts"))

    for candidate in candidates:
        try:
            text = candidate.read_text(errors="replace")
        except Exception:
            try:
                text = candidate.read_bytes().decode("utf-8", errors="replace")
            except Exception:
                continue

        embedded: Dict[str, str] = {}
        for key in EMBEDDED_KEYS:
            match = re.search(
                rf"{re.escape(key)}=(.*?)(?:\\0A|\x00|\r|\n|\")",
                text,
            )
            if match:
                embedded[key] = match.group(1)
        if embedded:
            return embedded

    return {}


def validate_embedded_cfg(
    artifacts_dir: Path,
    expected_cfg: Optional[Path],
) -> Optional[str]:
    """Return an error string if the built artifact embeds the wrong config."""
    if expected_cfg is None or not expected_cfg.exists():
        return None

    expected = parse_arts_cfg(expected_cfg)
    if not expected:
        return f"Failed to parse expected arts.cfg: {expected_cfg}"

    embedded = extract_embedded_cfg(artifacts_dir)
    if not embedded:
        return (
            "Failed to inspect generated ARTS artifact for embedded config "
            f"in {artifacts_dir}"
        )

    mismatches: List[str] = []
    for key in EMBEDDED_KEYS:
        expected_value = expected.get(key)
        if expected_value is None:
            continue
        embedded_value = embedded.get(key)
        if embedded_value != expected_value:
            mismatches.append(
                f"{key}: expected '{expected_value}', embedded '{embedded_value}'"
            )

    if not mismatches:
        return None

    return (
        "Generated ARTS artifact embeds a different config than the compile-time "
        f"arts.cfg. cfg={expected_cfg}, artifacts={artifacts_dir}. "
        + "; ".join(mismatches)
    )
