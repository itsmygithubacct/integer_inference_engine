"""Atomic, machine-readable evidence for production and verification runs.

The record deliberately contains operational metadata and digests, never prompt
text, signing-key paths, or other command-line secrets.  It is written after
each update so an interrupted multi-minute verification still leaves useful
evidence, and finalized with an explicit status and cleanup result.
"""
from __future__ import annotations

import json
import math
import os
import platform
import re
import resource
import time
from pathlib import Path
from typing import Any


SCHEMA = "receipt-run/v1"
_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:/[A-Za-z0-9_.~+@%=-]+)+"
    r"|(?<![A-Za-z0-9_.-])[A-Za-z]:\\(?:[^\\\s:]+\\)*[^\\\s:]+"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_WIF_RE = re.compile(r"(?<![A-Za-z0-9])[5KLc][1-9A-HJ-NP-Za-km-z]{50,51}(?![A-Za-z0-9])")
_API_KEY_RE = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}(?![A-Za-z0-9])")
_ASSIGNMENT_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z_][A-Za-z0-9_.-]{0,127})[\"']?\s*[:=]\s*"
)
_SENSITIVE_FIELDS = {
    "prompt", "prompttext", "systemprompt", "systemprompttext",
    "inputtext", "outputtext", "responsetext", "completiontext",
    "message", "messages", "content", "messagecontent",
    "apikey", "authorization", "password", "passwd", "passphrase",
    "secret", "clientsecret", "accesstoken", "refreshtoken",
    "sessiontoken", "authtoken", "bearertoken", "cookie", "setcookie",
    "modelkey", "counterpartykey", "signingkey", "privatekey",
    "modelkeypath", "counterpartykeypath", "signingkeypath", "privatekeypath",
}
_FREE_TEXT_SENSITIVE_MARKERS = (
    "password", "passwd", "passphrase", "secret", "credential", "apikey",
    "token", "authorization", "bearer", "cookie", "privatekey", "signingkey",
    "modelkey", "counterpartykey", "accesskey", "sshkey", "githubpat", "jwt", "wif",
    "mnemonic", "seedphrase", "recoveryphrase", "dsn", "databaseurl",
    "connectionstring", "prompt", "messages", "content", "inputtext", "outputtext",
    "responsetext", "completiontext",
)


class ReceiptRunEvidence:
    """Build and atomically persist one ``receipt-run/v1`` record."""

    def __init__(self, path: str | Path, *, operation: str, options: dict[str, Any] | None = None):
        self.path = Path(path)
        self._started_wall = time.time()
        self._started_mono = time.monotonic()
        self._started_cpu = time.process_time()
        self.record: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "running",
            "operation": str(operation),
            "startedUnixSeconds": self._started_wall,
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "logicalCpuCount": os.cpu_count(),
            },
            "options": _json_values(dict(options or {})),
            "resources": {
                "pid": os.getpid(),
                "threads": _thread_environment(),
            },
            "model": {},
            "engine": {},
            "tokens": {},
            "phases": [],
            "cleanup": {},
        }
        self.flush()

    def update(self, section: str, **values: Any) -> None:
        target = self.record.setdefault(str(section), {})
        if not isinstance(target, dict):
            raise TypeError(f"evidence section {section!r} is not an object")
        target.update(_json_values(values))
        self.flush()

    def add_phase(self, name: str, seconds: float, *, status: str = "ok", **details: Any) -> None:
        phase = {
            "name": str(name),
            "seconds": max(0.0, float(seconds)),
            "status": str(status),
        }
        phase.update(_json_values(details))
        self.record["phases"].append(phase)
        self.flush()

    def finish(self, status: str, *, exit_code: int, error: str | None = None) -> None:
        self.record["status"] = str(status)
        self.record["exitCode"] = int(exit_code)
        self.record["finishedUnixSeconds"] = time.time()
        self.record["totalSeconds"] = max(0.0, time.monotonic() - self._started_mono)
        self.record["resources"]["processCpuSeconds"] = max(0.0, time.process_time() - self._started_cpu)
        self.record["resources"]["maxRssKiB"] = int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
        if error:
            self.record["error"] = _safe_text(str(error))
        self.flush()

    def flush(self) -> None:
        """Durably replace the target without exposing a partially-written JSON file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        nonce = f"{os.getpid()}.{time.monotonic_ns()}"
        tmp = self.path.with_name(f".{self.path.name}.{nonce}.tmp")
        payload = json.dumps(
            self.record, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        ) + "\n"
        try:
            descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self.path)
            # Persist the directory entry when the filesystem supports it.
            try:
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
            except OSError:
                return
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def _thread_environment() -> dict[str, int | str]:
    values: dict[str, int | str] = {}
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TRINOTE_ORACLE_Q1_THREADS",
    ):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            values[name] = int(value)
        except ValueError:
            values[name] = value
    return values


def _json_values(values: dict[str, Any]) -> dict[str, Any]:
    """Normalize values while removing secrets and host-local absolute paths."""
    return {str(key): _json_value(value, field=str(key)) for key, value in values.items()}


def _json_value(value: Any, *, field: str = "") -> Any:
    normalized = re.sub(r"[^a-z0-9]", "", field.lower())
    if _is_sensitive_field(normalized):
        return "[redacted]"
    if isinstance(value, Path):
        return value.name or "[redacted-path]"
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "[non-finite]"
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, field=field) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item, field=str(key)) for key, item in value.items()}
    return _safe_text(str(value))


def _safe_text(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    if any(marker in normalized for marker in _FREE_TEXT_SENSITIVE_MARKERS):
        return "[redacted]"
    value = _BEARER_RE.sub("Bearer [redacted]", value)
    value = _WIF_RE.sub("[redacted-key]", value)
    value = _API_KEY_RE.sub("sk-[redacted]", value)
    # Free-text diagnostics are not a parseable credential format. Once a
    # sensitive assignment marker is seen, conservatively discard the rest of
    # the string. This covers malformed/escaped/multiline quotes without
    # guessing where attacker-controlled secret material ends.
    for match in _ASSIGNMENT_RE.finditer(value):
        normalized = re.sub(r"[^a-z0-9]", "", match.group(1).lower())
        if _is_sensitive_field(normalized):
            value = value[:match.end()] + "[redacted]"
            break
    return _ABSOLUTE_PATH_RE.sub("[redacted-path]", value)


def _is_sensitive_field(normalized: str) -> bool:
    return (
        normalized in _SENSITIVE_FIELDS
        or "password" in normalized
        or "passphrase" in normalized
        or "secret" in normalized
        or "credential" in normalized
        or normalized.endswith("apikey")
        or normalized.endswith("token")
        or normalized.endswith("pat")
        or normalized.endswith("accesskey")
        or normalized.endswith("accesskeyid")
        or normalized.endswith("privatekey")
        or normalized in {"pwd", "dsn", "databaseurl", "connectionstring"}
    )
