"""One-shot thread-environment bootstrap for Python CLI entry points.

CLI modules necessarily import their dependency graph before ``main()`` can
parse flags.  A real invocation with an explicit/measured thread budget is
therefore re-executed once with that budget already in the process environment,
so NumPy/BLAS and subsequently loaded OpenMP runtimes observe the same policy.
List-based ``main([...])`` calls used by tests and embedders stay in-process.
"""
from __future__ import annotations

import os
import sys


THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TRINOTE_ORACLE_Q1_THREADS",
)
_BOOTSTRAP_MARKER = "TRINOTE_CLI_THREAD_BOOTSTRAP"


def maybe_reexec_with_threads(
    count: int,
    *,
    real_argv: bool,
    module_name: str,
) -> None:
    """Re-exec a real CLI once so runtime imports see ``count`` immediately."""
    threads = int(count)
    if not real_argv or threads <= 0:
        return
    value = str(threads)
    marker = f"{module_name}:{value}"
    if os.environ.get(_BOOTSTRAP_MARKER) == marker:
        drift = [name for name in THREAD_ENV if os.environ.get(name) != value]
        if os.environ.get("OMP_DYNAMIC") != "FALSE":
            drift.append("OMP_DYNAMIC")
        if drift:
            raise RuntimeError(
                "thread environment changed after CLI bootstrap: " + ", ".join(drift)
            )
        return

    environment = os.environ.copy()
    for name in THREAD_ENV:
        environment[name] = value
    environment["OMP_DYNAMIC"] = "FALSE"
    environment.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    environment[_BOOTSTRAP_MARKER] = marker
    command = [sys.executable, "-m", str(module_name), *sys.argv[1:]]
    os.execve(sys.executable, command, environment)
    raise RuntimeError("thread-policy re-exec unexpectedly returned")
