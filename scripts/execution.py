"""Shared contracts for local benchmark execution."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import PERF_CACHE_EVENTS
from models import BenchmarkConfig, Status

from rich.console import Console
from sniff import Colors
from scripts.arts_config import KEY_COUNTER_FOLDER


@dataclass(frozen=True)
class BenchmarkExecutionContext:
    """Resolved benchmark configuration shared across builds and runs."""

    name: str
    suite: str
    size: str
    bench_path: Path
    config: BenchmarkConfig
    effective_arts_cfg: Path
    desired_threads: int
    desired_nodes: int
    desired_launcher: str
    actual_omp_threads: int
    effective_cflags: str
    run_args: List[str]
    verify_tolerance: float
    build_output_dir: Optional[Path] = None
    artifact_paths: Dict[str, Optional[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkRunFiles:
    """Filesystem locations used for one benchmark run."""

    run_number: int
    run_dir: Optional[Path] = None
    arts_log: Optional[Path] = None
    omp_log: Optional[Path] = None
    counter_dir: Optional[Path] = None
    perf_output_dir: Optional[Path] = None
    arts_perf_name: Optional[str] = None
    omp_perf_name: Optional[str] = None
    arts_perf_main: Optional[Path] = None
    arts_perf_temp: Optional[Path] = None
    omp_perf_main: Optional[Path] = None
    omp_perf_temp: Optional[Path] = None


@dataclass(frozen=True)
class BenchmarkProcessRequest:
    """Execution request for one process launch."""

    executable: str
    timeout: int
    env: Dict[str, str] = field(default_factory=dict)
    launcher: str = "ssh"
    node_count: int = 1
    threads: int = 1
    args: List[str] = field(default_factory=list)
    log_file: Optional[Path] = None
    perf_enabled: bool = False
    perf_interval: float = 0.1
    perf_output_name: str = "perf_cache.csv"
    perf_output_dir: Optional[Path] = None
    counter_dir: Optional[Path] = None
    capture_diagnostics: bool = False


@dataclass(frozen=True)
class ProcessExecutionOutcome:
    """Raw process outcome before benchmark-specific parsing."""

    status: Status
    duration_sec: float
    exit_code: int
    stdout: str
    stderr: str
    perf_output: Optional[Path] = None
    startup_diagnostics: Dict[str, Any] = field(default_factory=dict)


class BenchmarkProcessRunner:
    """Execute a benchmark process and capture raw output/perf artifacts."""

    def __init__(self, console: Console, *, verbose: bool, debug: int) -> None:
        self.console = console
        self.verbose = verbose
        self.debug = debug

    def execute(self, request: BenchmarkProcessRequest) -> ProcessExecutionOutcome:
        """Run the requested executable and return the raw process outcome."""
        if not request.executable or not os.path.exists(request.executable):
            return ProcessExecutionOutcome(
                status=Status.SKIP,
                duration_sec=0.0,
                exit_code=-1,
                stdout="",
                stderr="Executable not found",
            )

        run_env = os.environ.copy()
        if request.env:
            run_env.update(request.env)
        if request.counter_dir:
            run_env[KEY_COUNTER_FOLDER] = str(request.counter_dir)

        cmd = self._build_command(request)
        perf_output: Optional[Path] = None
        if request.perf_enabled:
            perf_output = self._resolve_perf_output(request)
            cmd = self._wrap_with_perf(cmd, request, perf_output)

        self._print_debug_command(cmd, request.env)
        if request.capture_diagnostics:
            startup_diagnostics = self._capture_startup_diagnostics(request, cmd, run_env)
        else:
            startup_diagnostics: Dict[str, Any] = {}

        start = time.time()
        proc: Optional[subprocess.Popen[str]] = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path(request.executable).parent,
                env=run_env,
                start_new_session=True,
            )
            stdout_text, stderr_text = proc.communicate(timeout=request.timeout)
            duration = time.time() - start
            exit_code = proc.returncode if proc.returncode is not None else -1
            if request.capture_diagnostics:
                startup_diagnostics.update(
                    self._finalize_startup_diagnostics(
                        exit_code=exit_code,
                        duration=duration,
                        stdout_text=stdout_text,
                        stderr_text=stderr_text,
                    )
                )

            self._write_run_log(
                request,
                cmd,
                duration=duration,
                exit_code=exit_code,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
            )

            return ProcessExecutionOutcome(
                status=self._status_from_exit_code(exit_code),
                duration_sec=duration,
                exit_code=exit_code,
                stdout=stdout_text,
                stderr=stderr_text,
                perf_output=perf_output if perf_output and perf_output.exists() else None,
                startup_diagnostics=startup_diagnostics,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.time() - start
            stdout_text = self._to_text(exc.stdout)
            stderr_text = self._to_text(exc.stderr)
            timeout_note = f"Execution timed out after {request.timeout} seconds"

            if proc is not None:
                self._terminate_timed_out_process(proc)
                try:
                    collected_stdout, collected_stderr = proc.communicate(timeout=2)
                    if collected_stdout:
                        stdout_text = self._to_text(collected_stdout)
                    if collected_stderr:
                        stderr_text = self._to_text(collected_stderr)
                except subprocess.TimeoutExpired:
                    pass
                except Exception:
                    pass

            if request.capture_diagnostics:
                startup_diagnostics.update(
                    self._finalize_startup_diagnostics(
                        exit_code=124,
                        duration=duration,
                        stdout_text=stdout_text,
                        stderr_text=f"{stderr_text}\n{timeout_note}".strip(),
                        note=timeout_note,
                    )
                )
            self._write_run_log(
                request,
                cmd,
                duration=duration,
                exit_code=124,
                stdout_text=stdout_text,
                stderr_text=f"{stderr_text}\n{timeout_note}".strip(),
                timed_out=True,
                note=timeout_note,
            )

            return ProcessExecutionOutcome(
                status=Status.TIMEOUT,
                duration_sec=duration,
                exit_code=124,
                stdout=stdout_text,
                stderr=f"{stderr_text}\n{timeout_note}".strip(),
                perf_output=perf_output if perf_output and perf_output.exists() else None,
                startup_diagnostics=startup_diagnostics,
            )
        except Exception as exc:
            duration = time.time() - start
            if request.capture_diagnostics:
                startup_diagnostics.update(
                    self._finalize_startup_diagnostics(
                        exit_code=-1,
                        duration=duration,
                        stdout_text="",
                        stderr_text=str(exc),
                        note="Runner exception",
                    )
                )
            self._write_run_log(
                request,
                cmd,
                duration=duration,
                exit_code=-1,
                stdout_text="",
                stderr_text=str(exc),
                note="Runner exception",
            )
            return ProcessExecutionOutcome(
                status=Status.FAIL,
                duration_sec=duration,
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                perf_output=perf_output if perf_output and perf_output.exists() else None,
                startup_diagnostics=startup_diagnostics,
            )

    def _build_command(self, request: BenchmarkProcessRequest) -> List[str]:
        if request.launcher == "slurm" and request.node_count > 1:
            cmd = [
                "srun",
                f"-N{request.node_count}",
                "--ntasks-per-node=1",
                f"--cpus-per-task={request.threads}",
                request.executable,
            ]
        else:
            cmd = [request.executable]
        if request.args:
            cmd.extend(request.args)
        return cmd

    def _resolve_perf_output(self, request: BenchmarkProcessRequest) -> Optional[Path]:
        if not self._perf_available():
            return None
        output_dir = request.perf_output_dir if request.perf_output_dir else Path(request.executable).parent
        return output_dir / request.perf_output_name

    def _wrap_with_perf(
        self,
        cmd: List[str],
        request: BenchmarkProcessRequest,
        perf_output: Optional[Path],
    ) -> List[str]:
        if perf_output is None:
            return cmd
        events = ",".join(PERF_CACHE_EVENTS)
        interval_ms = int(request.perf_interval * 1000)
        return [
            "perf",
            "stat",
            "-e",
            events,
            "-I",
            str(interval_ms),
            "-x",
            ",",
            "-o",
            str(perf_output),
            "--",
            *cmd,
        ]

    def _perf_available(self) -> bool:
        try:
            result = subprocess.run(
                ["perf", "stat", "-e", "cycles", "--", "/bin/true"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 or "counted" in result.stderr.lower():
                return True
            if self.verbose or self.debug >= 1:
                self.console.print(
                    f"[{Colors.WARNING}]Warning: perf not available (permission denied or not installed). "
                    f"Running without perf profiling.[/{Colors.WARNING}]"
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            if self.verbose or self.debug >= 1:
                self.console.print(
                    f"[{Colors.WARNING}]Warning: perf not found. Running without perf profiling.[/{Colors.WARNING}]"
                )
        return False

    def _print_debug_command(self, cmd: List[str], env: Dict[str, str]) -> None:
        if self.debug < 1:
            return
        env_str = " ".join(f"{k}={v}" for k, v in env.items())
        if env_str:
            self.console.print(f"[{Colors.DEBUG}]$ {env_str} {' '.join(cmd)}[/{Colors.DEBUG}]")
        else:
            self.console.print(f"[{Colors.DEBUG}]$ {' '.join(cmd)}[/{Colors.DEBUG}]")

    def _write_run_log(
        self,
        request: BenchmarkProcessRequest,
        cmd: List[str],
        *,
        duration: float,
        exit_code: int,
        stdout_text: str,
        stderr_text: str,
        timed_out: bool = False,
        note: Optional[str] = None,
    ) -> None:
        if not request.log_file:
            if self.debug >= 2:
                lines = (stdout_text + stderr_text).strip().split("\n")
                if len(lines) > 10:
                    self.console.print(
                        f"[{Colors.DEBUG}]  ({len(lines)} lines of output, use log_file to capture)[/{Colors.DEBUG}]"
                    )
                elif lines and lines[0]:
                    for line in lines[:10]:
                        self.console.print(f"[{Colors.DEBUG}]  {line}[/{Colors.DEBUG}]")
            return

        request.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(request.log_file, "w") as handle:
            handle.write(f"# Command: {' '.join(cmd)}\n")
            handle.write(f"# Duration: {duration:.3f}s\n")
            handle.write(f"# Exit code: {exit_code}\n")
            if timed_out:
                handle.write("# Timed out: true\n")
            if note:
                handle.write(f"# Note: {note}\n")
            handle.write("\n")
            if stdout_text:
                handle.write("=== STDOUT ===\n")
                handle.write(stdout_text)
                if not stdout_text.endswith("\n"):
                    handle.write("\n")
            if stderr_text:
                handle.write("=== STDERR ===\n")
                handle.write(stderr_text)
                if not stderr_text.endswith("\n"):
                    handle.write("\n")

        if self.debug >= 2:
            self.console.print(f"[{Colors.DEBUG}]  Log: {request.log_file}[/{Colors.DEBUG}]")

    def _terminate_timed_out_process(self, proc: subprocess.Popen[str]) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        except Exception:
            pass

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            except Exception:
                pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass

    @staticmethod
    def _status_from_exit_code(exit_code: int) -> Status:
        if exit_code == 0:
            return Status.PASS
        if exit_code in (139, 134, 136):
            return Status.CRASH
        return Status.FAIL

    @staticmethod
    def _to_text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return str(value)

    def _capture_startup_diagnostics(
        self,
        request: BenchmarkProcessRequest,
        cmd: List[str],
        env: Dict[str, str],
    ) -> Dict[str, Any]:
        """Capture launch-time context used for startup outlier triage."""
        return {
            "captured_at": datetime.now().isoformat(),
            "cwd": str(Path(request.executable).parent),
            "launcher": request.launcher,
            "node_count": request.node_count,
            "threads": request.threads,
            "timeout_sec": request.timeout,
            "command": cmd,
            "env": self._redact_env(env),
            "network_snapshot_pre": self._snapshot_command(["ss", "-lntup"], timeout=3),
            "process_snapshot_pre": self._snapshot_command(
                ["ps", "-eo", "pid,ppid,pcpu,pmem,etime,comm"], timeout=3
            ),
        }

    def _finalize_startup_diagnostics(
        self,
        *,
        exit_code: int,
        duration: float,
        stdout_text: str,
        stderr_text: str,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Attach completion-side snapshots and compact output excerpts."""
        payload: Dict[str, Any] = {
            "completed_at": datetime.now().isoformat(),
            "exit_code": exit_code,
            "duration_sec": duration,
            "stdout_preview": self._preview_lines(stdout_text),
            "stderr_preview": self._preview_lines(stderr_text),
            "network_snapshot_post": self._snapshot_command(["ss", "-lntup"], timeout=3),
            "process_snapshot_post": self._snapshot_command(
                ["ps", "-eo", "pid,ppid,pcpu,pmem,etime,comm"], timeout=3
            ),
        }
        if note:
            payload["note"] = note
        return payload

    @staticmethod
    def _redact_env(env: Dict[str, str]) -> Dict[str, str]:
        """Keep only runtime-relevant env vars."""
        kept: Dict[str, str] = {}
        allow_prefixes = ("OMP_", "ARTS_", "SLURM_", "counter_")
        allow_exact = {
            "PATH",
            "LD_LIBRARY_PATH",
            "PWD",
            "HOME",
            "USER",
        }

        for key, value in env.items():
            if key in allow_exact or key.startswith(allow_prefixes):
                kept[key] = str(value)
        return kept

    @staticmethod
    def _preview_lines(text: str, max_lines: int = 20) -> List[str]:
        lines = [line for line in text.splitlines() if line.strip()]
        return lines[:max_lines]

    @staticmethod
    def _snapshot_command(command: List[str], timeout: int) -> Dict[str, Any]:
        """Run a quick system snapshot command for diagnostics."""
        snapshot: Dict[str, Any] = {
            "command": command,
            "ok": False,
        }
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            snapshot["return_code"] = proc.returncode
            snapshot["ok"] = proc.returncode == 0
            stdout_lines = proc.stdout.splitlines()
            stderr_lines = proc.stderr.splitlines()
            snapshot["stdout"] = stdout_lines[:120]
            snapshot["stderr"] = stderr_lines[:40]
        except FileNotFoundError:
            snapshot["error"] = f"{command[0]} not found"
        except subprocess.TimeoutExpired:
            snapshot["error"] = f"{command[0]} timed out after {timeout}s"
        except Exception as exc:
            snapshot["error"] = str(exc)
        return snapshot
