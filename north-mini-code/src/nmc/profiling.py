"""Low-overhead, opt-in profiling for the North Mini Code inference engine.

The CUDA runtime exposes device-boundary counters separately.  This module
accounts for the Python orchestration phases and keeps cold model/expert
discovery separate from warm steady-state work.  Profiling is disabled by
default so receipt-producing paths do not acquire timers or mutate telemetry.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from threading import RLock
from time import perf_counter_ns


class InferenceProfiler:
    """Thread-safe call/time accumulator with explicit cold and warm buckets."""

    SCHEMA_VERSION = 1

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._lock = RLock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._bucket = "cold"
            self._phases = {"cold": {}, "warm": {}}

    def mark_warm(self) -> None:
        with self._lock:
            self._bucket = "warm"

    @property
    def bucket(self) -> str:
        with self._lock:
            return self._bucket

    def record(self, name: str, wall_ns: int = 0, *, calls: int = 1,
               bucket: str | None = None, errors: int = 0) -> None:
        if not self.enabled:
            return
        if not name or calls < 0 or wall_ns < 0 or errors < 0:
            raise ValueError("profile records require a name and non-negative counters")
        with self._lock:
            target = bucket or self._bucket
            if target not in self._phases:
                raise ValueError(f"unknown profile bucket {target!r}")
            row = self._phases[target].setdefault(name, {"calls": 0, "wall_ns": 0, "errors": 0})
            row["calls"] += int(calls)
            row["wall_ns"] += int(wall_ns)
            row["errors"] += int(errors)

    @contextmanager
    def phase(self, name: str, *, bucket: str | None = None):
        """Measure a phase, recording exceptional exits without swallowing them."""
        if not self.enabled:
            yield
            return
        started = perf_counter_ns()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            self.record(name, perf_counter_ns() - started, bucket=bucket, errors=int(failed))

    def snapshot(self, native: dict | None = None) -> dict:
        with self._lock:
            phases = deepcopy(self._phases)
            current = self._bucket
        result = {
            "schema_version": self.SCHEMA_VERSION,
            "enabled": self.enabled,
            "current_bucket": current,
            "phases": phases,
        }
        if native is not None:
            result["native_cuda"] = deepcopy(native)
        return result
