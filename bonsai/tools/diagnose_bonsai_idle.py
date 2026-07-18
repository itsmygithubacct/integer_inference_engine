#!/usr/bin/env python3
"""Measure native Bonsai REPL idle CPU and scheduler interference on Linux.

With no ``--pid`` this tool measures a small independent CPU command, starts
the integer 27B launcher on a pseudo-terminal, waits for ``bonsai>``, gives
OpenMP five seconds to settle, samples process CPU for five seconds, and runs
the independent command again while the REPL is idle.  The launched REPL is
cleanly closed after the diagnostic.  ``--pid`` only samples an existing
process and never signals it.
"""
from __future__ import annotations

import argparse
import json
import os
import pty
import select
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORMAT = "trinote-bonsai35-idle-diagnostic/1"
THREAD_KEYS = (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS",
    "OMP_DYNAMIC", "OMP_WAIT_POLICY", "OMP_PLACES", "OMP_PROC_BIND", "OMP_MAX_ACTIVE_LEVELS",
    "GOMP_SPINCOUNT", "KMP_BLOCKTIME",
)


def read_process_cpu_seconds(pid: int) -> float:
    text = (Path("/proc") / str(pid) / "stat").read_text()
    # The command name in parentheses may contain spaces, so split only after
    # its final ')'.  utime/stime are fields 14/15, indices 11/12 in the tail.
    tail = text[text.rfind(")") + 2:].split()
    ticks = int(tail[11]) + int(tail[12])
    return ticks / os.sysconf("SC_CLK_TCK")


def sample_idle_cpu(pid: int, seconds: float) -> dict[str, float]:
    cpu0 = read_process_cpu_seconds(pid)
    wall0 = time.monotonic()
    time.sleep(seconds)
    wall = time.monotonic() - wall0
    cpu = read_process_cpu_seconds(pid) - cpu0
    return {
        "sample_seconds": wall, "cpu_seconds": cpu,
        "cpu_percent_of_one_core": 0.0 if wall <= 0 else cpu / wall * 100.0,
    }


def process_thread_snapshot(pid: int) -> dict[str, Any]:
    root = Path("/proc") / str(pid)
    tasks = list((root / "task").glob("[0-9]*"))
    states: dict[str, int] = {}
    names: dict[str, int] = {}
    for task in tasks:
        try:
            status = (task / "status").read_text(errors="replace")
        except OSError:
            continue
        name = state = "unknown"
        for line in status.splitlines():
            if line.startswith("Name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("State:"):
                state = line.split(":", 1)[1].strip()
        names[name] = names.get(name, 0) + 1
        states[state] = states.get(state, 0) + 1
    voluntary = involuntary = None
    try:
        for line in (root / "status").read_text().splitlines():
            if line.startswith("voluntary_ctxt_switches:"):
                voluntary = int(line.split(":", 1)[1])
            elif line.startswith("nonvoluntary_ctxt_switches:"):
                involuntary = int(line.split(":", 1)[1])
    except OSError:
        pass
    return {
        "thread_count": len(tasks), "thread_names": names, "thread_states": states,
        "voluntary_context_switches": voluntary, "involuntary_context_switches": involuntary,
    }


def process_thread_environment(pid: int) -> dict[str, str | None]:
    try:
        raw = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
        env = {}
        for item in raw:
            if b"=" in item:
                key, value = item.split(b"=", 1)
                env[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    except OSError:
        env = {}
    return {key: env.get(key) for key in THREAD_KEYS}


def competitor_once(iterations: int) -> float:
    code = (
        "import hashlib,time;"
        "t=time.monotonic();"
        f"hashlib.pbkdf2_hmac('sha256',b'bonsai-idle-probe',b'trinote', {iterations});"
        "print(time.monotonic()-t)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=120, check=True)
    return float(proc.stdout.strip())


def competitor_sample(iterations: int, repetitions: int) -> dict[str, Any]:
    values = [competitor_once(iterations) for _ in range(repetitions)]
    return {"seconds": values, "median_s": statistics.median(values)}


def _await_prompt(fd: int, child_pid: int, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    captured = bytearray()
    while time.monotonic() < deadline:
        waited, status = os.waitpid(child_pid, os.WNOHANG)
        if waited:
            raise RuntimeError(f"REPL exited before prompt (wait status {status}): {captured[-4000:].decode(errors='replace')}")
        ready, _, _ = select.select([fd], [], [], min(0.25, max(0.0, deadline - time.monotonic())))
        if not ready:
            continue
        try:
            block = os.read(fd, 65536)
        except OSError:
            block = b""
        if not block:
            continue
        captured.extend(block)
        if b"bonsai>" in captured:
            return captured.decode("utf-8", "replace")
    raise TimeoutError(f"timed out waiting for bonsai> after {timeout}s; output={captured[-4000:].decode(errors='replace')}")


def _launch_repl(launcher: str, prompt_timeout: float, exercise_prompt: str) -> tuple[int, int, str]:
    pid, fd = pty.fork()
    if pid == 0:
        os.execv(launcher, [launcher, "repl", "-n", "1"])
    output = _await_prompt(fd, pid, prompt_timeout)
    if exercise_prompt:
        os.write(fd, exercise_prompt.encode("utf-8") + b"\n")
        output += _await_prompt(fd, pid, prompt_timeout)
    return pid, fd, output


def _close_repl(pid: int, fd: int) -> None:
    try:
        os.write(fd, b"exit\n")
    except OSError:
        pass
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            waited, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return
        if waited:
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGTERM)
        os.waitpid(pid, 0)
    except (ProcessLookupError, ChildProcessError):
        pass


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Validate Bonsai-27B native REPL idle CPU/thread hygiene",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--pid", type=int, default=0, help="sample an existing PID; do not launch or signal it")
    parser.add_argument("--launcher", default=str(root / "bonsai-integer-27b-cli"))
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--sample-seconds", type=float, default=5.0)
    parser.add_argument("--prompt-timeout", type=float, default=300.0)
    parser.add_argument("--exercise-prompt", default="Hi",
                        help="run this one-token-output turn before the idle sample; empty disables it")
    parser.add_argument("--max-cpu-percent", type=float, default=1.0,
                        help="maximum percent of one CPU core while idle")
    parser.add_argument("--competitor-iterations", type=int, default=2_000_000)
    parser.add_argument("--competitor-repetitions", type=int, default=3)
    parser.add_argument("--max-competitor-slowdown", type=float, default=1.10)
    parser.add_argument("--output", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup_seconds < 0 or args.sample_seconds <= 0:
        raise SystemExit("warmup must be non-negative and sample duration must be positive")
    if args.pid:
        pid, fd, startup, baseline = args.pid, None, None, None
    else:
        baseline = competitor_sample(args.competitor_iterations, args.competitor_repetitions)
        launcher = str(Path(args.launcher).expanduser().resolve())
        pid, fd, startup = _launch_repl(launcher, args.prompt_timeout, args.exercise_prompt)
    try:
        time.sleep(args.warmup_seconds)
        before = process_thread_snapshot(pid)
        idle = sample_idle_cpu(pid, args.sample_seconds)
        after = process_thread_snapshot(pid)
        competing = None if args.pid else competitor_sample(
            args.competitor_iterations, args.competitor_repetitions
        )
        slowdown = None if baseline is None else competing["median_s"] / baseline["median_s"]
        cpu_ok = idle["cpu_percent_of_one_core"] < args.max_cpu_percent
        competitor_ok = slowdown is None or slowdown <= args.max_competitor_slowdown
        payload = {
            "format": FORMAT, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pid": pid, "launched_by_diagnostic": not bool(args.pid),
            "limits": {"max_cpu_percent_of_one_core": args.max_cpu_percent,
                       "max_competitor_slowdown": args.max_competitor_slowdown},
            "idle": idle, "threads_before": before, "threads_after": after,
            "thread_environment": process_thread_environment(pid),
            "competitor_baseline": baseline, "competitor_while_idle": competing,
            "competitor_slowdown": slowdown,
            "acceptance": {"idle_cpu": cpu_ok, "competitor": competitor_ok,
                           "passed": cpu_ok and competitor_ok},
            "startup_tail": startup[-2000:] if startup else None,
        }
        if args.output:
            path = Path(args.output).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload["acceptance"]["passed"] else 1
    finally:
        if fd is not None:
            _close_repl(pid, fd)


if __name__ == "__main__":
    raise SystemExit(main())
