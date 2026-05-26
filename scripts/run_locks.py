"""Lock files for benchmark run orchestration."""

from __future__ import annotations

import atexit
import json
import os
import socket
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


LOCK_DIR_ENV = "CARTS_BENCHMARK_LOCK_DIR"
RESULTS_LOCK_FILENAME = ".carts-benchmark-run.lock"


class BenchmarkRunLockError(RuntimeError):
    """Raised when an active benchmark run already owns a required lock."""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hostname() -> str:
    return socket.gethostname() or "unknown-host"


def _sanitize_token(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return sanitized or "unknown-host"


def _default_node_lock_dir() -> Path:
    configured = os.environ.get(LOCK_DIR_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(tempfile.gettempdir()) / f"carts-benchmark-locks-{os.getuid()}"


def _read_lock_metadata(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _pid_is_alive(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _format_owner(metadata: Dict[str, Any]) -> str:
    if not metadata:
        return "unknown owner"
    parts = []
    pid = metadata.get("pid")
    host = metadata.get("host")
    created_at = metadata.get("created_at")
    cwd = metadata.get("cwd")
    results_dir = metadata.get("results_dir")
    command = metadata.get("command")
    if pid is not None:
        parts.append(f"pid={pid}")
    if host:
        parts.append(f"host={host}")
    if created_at:
        parts.append(f"started={created_at}")
    if cwd:
        parts.append(f"cwd={cwd}")
    if results_dir:
        parts.append(f"results_dir={results_dir}")
    if command:
        parts.append(f"command={command}")
    return ", ".join(parts) if parts else "unknown owner"


class _AtomicRunLock:
    def __init__(self, *, path: Path, scope: str, metadata: Dict[str, Any]) -> None:
        self.path = path
        self.scope = scope
        self.token = uuid.uuid4().hex
        self.metadata = dict(metadata)
        self.metadata.update(
            {
                "scope": scope,
                "pid": os.getpid(),
                "host": _hostname(),
                "created_at": _now_utc(),
                "token": self.token,
            }
        )
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
            except FileExistsError:
                metadata = _read_lock_metadata(self.path)
                if self._remove_if_stale(metadata):
                    continue
                owner = _format_owner(metadata)
                raise BenchmarkRunLockError(
                    f"Another CARTS benchmark run is active for {self.scope}. "
                    f"Lock: {self.path}. Owner: {owner}"
                )

            with os.fdopen(fd, "w") as f:
                json.dump(self.metadata, f, indent=2, sort_keys=True)
                f.write("\n")
            self.acquired = True
            return

    def _remove_if_stale(self, metadata: Dict[str, Any]) -> bool:
        if metadata.get("host") != _hostname():
            return False
        if _pid_is_alive(metadata.get("pid")):
            return False
        try:
            self.path.unlink()
        except FileNotFoundError:
            return True
        return True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            metadata = _read_lock_metadata(self.path)
            if metadata.get("token") == self.token:
                self.path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self.acquired = False


@dataclass
class BenchmarkRunLocks:
    """Owns all lock files needed for one benchmark run."""

    results_dir: Path
    run_id: str
    command: str
    mode: str
    cwd: Path
    node_lock: bool = True
    node_lock_dir: Optional[Path] = None
    _locks: List[_AtomicRunLock] = field(default_factory=list, init=False)
    _atexit_registered: bool = field(default=False, init=False)

    @property
    def experiment_dir(self) -> Path:
        return self.results_dir / self.run_id

    def acquire(self) -> None:
        metadata = {
            "mode": self.mode,
            "cwd": str(self.cwd),
            "command": self.command,
            "results_dir": str(self.results_dir),
            "experiment_dir": str(self.experiment_dir),
        }
        requested: List[_AtomicRunLock] = []
        if self.node_lock:
            lock_root = self.node_lock_dir or _default_node_lock_dir()
            requested.append(
                _AtomicRunLock(
                    path=lock_root / f"benchmark-run-{_sanitize_token(_hostname())}.lock",
                    scope="this node",
                    metadata=metadata,
                )
            )
        requested.append(
            _AtomicRunLock(
                path=self.results_dir / RESULTS_LOCK_FILENAME,
                scope="this results directory",
                metadata=metadata,
            )
        )

        acquired: List[_AtomicRunLock] = []
        try:
            for lock in requested:
                lock.acquire()
                acquired.append(lock)
        except Exception:
            for lock in reversed(acquired):
                lock.release()
            raise

        self._locks = requested
        if not self._atexit_registered:
            atexit.register(self.release)
            self._atexit_registered = True

    def release(self) -> None:
        for lock in reversed(self._locks):
            lock.release()
