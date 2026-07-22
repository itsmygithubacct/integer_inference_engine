#!/usr/bin/env python3
"""Isolated real-model promotion gate for the retained NMC layer executor.

This driver compares the established resident-attention decode path with the
explicit ``Engine.resident_decode_token`` path on the same GGUF.  It performs
one unmeasured reference pass to load exactly the route-lazy weights used by
the prompt, then measures a warmed established pass and the retained-layer
pass.  The retained pass must reproduce every hidden row, full vocabulary
logit row, and greedy token byte-for-byte.

The first retained transition is classified as cold because it configures the
request bank and registers the dense RMSNorm/router inputs.  Later transitions
are warm and must add no CUDA allocations or request-workspace bytes.  Warm
transition plus output-head throughput is compared against the established
path.  A background ``nvidia-smi`` sampler records combined device memory,
including weights, K/V, and all native workspaces.

The report is always published atomically as mode 0600 JSON.  This is an
acceptance tool, not a production switch: it never changes ``Engine.generate``.

Example:
    PYTHONPATH=src NMC_BACKEND=cuda-resident python tools/gate_resident_layers.py \
      MODEL.gguf --tokenizer TOKENIZER_DIR --new-tokens 4 \
      --output resident-layer-gate.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from nmc import cohere2 as c2
from nmc import qk_cuda
from nmc.engine import Engine, FA


SCHEMA_VERSION = 1
DEFAULT_PROMPT = "The capital of France is"


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _array_evidence(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value, dtype="<i8"))
    return {
        "shape": list(array.shape),
        "dtype": "int64-le",
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _array_comparison(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    left = np.asarray(expected)
    right = np.asarray(actual)
    result = {
        "exact": False,
        "expected": _array_evidence(left),
        "actual": _array_evidence(right),
        "mismatch_count": None,
        "first_mismatch": None,
    }
    if left.shape != right.shape:
        return result
    mismatches = np.flatnonzero(left.reshape(-1) != right.reshape(-1))
    result["mismatch_count"] = int(mismatches.size)
    result["exact"] = not mismatches.size
    if mismatches.size:
        flat_index = int(mismatches[0])
        result["first_mismatch"] = {
            "flat_index": flat_index,
            "expected": int(left.reshape(-1)[flat_index]),
            "actual": int(right.reshape(-1)[flat_index]),
        }
    return result


def _counter_delta(before: dict[str, int] | None,
                   after: dict[str, int] | None) -> dict[str, int] | None:
    if before is None or after is None:
        return None
    if set(before) != set(after):
        raise RuntimeError("native CUDA telemetry keys changed during the gate")
    delta = {key: int(after[key]) - int(before[key]) for key in sorted(after)}
    if any(value < 0 for value in delta.values()):
        raise RuntimeError("native CUDA telemetry counter moved backwards")
    return delta


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Durably replace *path* with strict JSON, never a partial report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=True) as stream:
            fd = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _run_text(command: list[str], *, cwd: Path | None = None,
              timeout: float = 30.0) -> str:
    completed = subprocess.run(
        command, cwd=cwd, check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, timeout=timeout,
    )
    return completed.stdout


def _source_identity(root: Path) -> dict[str, Any]:
    try:
        commit = _run_text(["git", "rev-parse", "HEAD"], cwd=root).strip()
        status = _run_text(
            ["git", "status", "--short", "--untracked-files=all", "--", "."], cwd=root,
        )
        paths = [line[3:] for line in status.splitlines() if len(line) >= 4]
        patch = _run_text(["git", "diff", "--binary", "HEAD", "--", "."], cwd=root)
        untracked_text = _run_text(
            ["git", "ls-files", "--others", "--exclude-standard", "--", "."], cwd=root,
        )
        tree_digest = hashlib.sha256()
        tree_digest.update(commit.encode("ascii"))
        tree_digest.update(patch.encode("utf-8"))
        untracked = []
        for relative in sorted(line for line in untracked_text.splitlines() if line):
            candidate = root / relative
            if not candidate.is_file():
                continue
            digest = _sha256_file(candidate)
            untracked.append({"path": relative, "sha256": digest})
            tree_digest.update(relative.encode("utf-8"))
            tree_digest.update(bytes.fromhex(digest))
        result = {
            "git_commit": commit,
            "dirty": bool(paths),
            "dirty_paths": paths,
            "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
            "tracked_patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(),
            "untracked_files": untracked,
            "source_evidence_sha256": tree_digest.hexdigest(),
        }
    except (OSError, subprocess.SubprocessError):
        result = {
            "git_commit": None, "dirty": None, "dirty_paths": [], "status_sha256": None,
            "tracked_patch_sha256": None, "untracked_files": [], "source_evidence_sha256": None,
        }
    snapshot_name = os.environ.get("TRINOTE_SOURCE_SNAPSHOT")
    if snapshot_name:
        snapshot_path = Path(snapshot_name)
        raw = snapshot_path.read_bytes()
        snapshot = json.loads(raw)
        if snapshot.get("format") != "trinote-integer-speed-source-snapshot/v1":
            raise RuntimeError("deployed source snapshot has the wrong format")
        if result["git_commit"] is not None and snapshot.get("baseCommit") != result["git_commit"]:
            raise RuntimeError("deployed source snapshot base does not match checkout HEAD")
        result["snapshot"] = {
            "format": snapshot["format"],
            "base_commit": snapshot["baseCommit"],
            "snapshot_digest": snapshot["snapshotDigest"],
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "entry_count": snapshot["entryCount"],
            "content_bytes": snapshot["contentBytes"],
        }
    return result


def _parse_gpu_rows(text: str) -> list[dict[str, Any]]:
    rows = []
    for fields in csv.reader(line for line in text.splitlines() if line.strip()):
        if len(fields) != 6:
            raise RuntimeError(f"unexpected nvidia-smi GPU row with {len(fields)} fields")
        index, uuid, name, used, total, driver = (field.strip() for field in fields)
        rows.append({
            "index": int(index), "uuid": uuid, "name": name,
            "memory_used_mib": int(used), "memory_total_mib": int(total),
            "driver_version": driver,
        })
    return rows


def _parse_compute_rows(text: str) -> list[dict[str, Any]]:
    rows = []
    for fields in csv.reader(line for line in text.splitlines() if line.strip()):
        if len(fields) != 3:
            raise RuntimeError(f"unexpected nvidia-smi compute row with {len(fields)} fields")
        uuid, pid, used = (field.strip() for field in fields)
        rows.append({"uuid": uuid, "pid": int(pid), "memory_used_mib": int(used)})
    return rows


class NvidiaSmiSampler:
    """Low-rate combined device-memory sampler for an isolated gate GPU."""

    GPU_QUERY = (
        "index,uuid,name,memory.used,memory.total,driver_version"
    )
    COMPUTE_QUERY = "gpu_uuid,pid,used_gpu_memory"

    def __init__(self, gpu_index: int | None, interval_ms: int = 100,
                 runner: Callable[..., str] = _run_text):
        if interval_ms < 20:
            raise ValueError("GPU sampling interval must be at least 20 ms")
        self.interval_s = interval_ms / 1000.0
        self._runner = runner
        initial = self._gpu_rows()
        if not initial:
            raise RuntimeError("nvidia-smi reported no GPUs")
        if gpu_index is None:
            if len(initial) != 1:
                raise RuntimeError("multiple GPUs are visible; select one with --gpu-index")
            selected = initial[0]
        else:
            matches = [row for row in initial if row["index"] == int(gpu_index)]
            if len(matches) != 1:
                raise RuntimeError(f"nvidia-smi did not report GPU index {gpu_index}")
            selected = matches[0]
        self.device = selected
        self.initial_compute_apps = [
            row for row in self._compute_rows() if row["uuid"] == selected["uuid"]
        ]
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _gpu_rows(self) -> list[dict[str, Any]]:
        text = self._runner([
            "nvidia-smi", f"--query-gpu={self.GPU_QUERY}",
            "--format=csv,noheader,nounits",
        ])
        return _parse_gpu_rows(text)

    def _compute_rows(self) -> list[dict[str, Any]]:
        text = self._runner([
            "nvidia-smi", f"--query-compute-apps={self.COMPUTE_QUERY}",
            "--format=csv,noheader,nounits",
        ])
        return _parse_compute_rows(text)

    def _sample_once(self) -> None:
        matches = [row for row in self._gpu_rows() if row["uuid"] == self.device["uuid"]]
        if len(matches) != 1:
            raise RuntimeError("selected GPU disappeared from nvidia-smi")
        row = matches[0]
        self.samples.append({
            "monotonic_ns": time.monotonic_ns(),
            "memory_used_mib": row["memory_used_mib"],
        })

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample_once()
            except Exception as exc:  # evidence records sampler failure; the gate fails closed later
                self.errors.append(f"{type(exc).__name__}: {exc}")
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU sampler already started")
        self._sample_once()
        self._thread = threading.Thread(target=self._loop, name="nmc-gpu-memory-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=max(2.0, self.interval_s * 4))
            self._thread = None
        try:
            self._sample_once()
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")

    def phase_peak(self, started_ns: int, ended_ns: int) -> int | None:
        values = [
            sample["memory_used_mib"] for sample in self.samples
            if started_ns <= sample["monotonic_ns"] <= ended_ns
        ]
        return max(values) if values else None

    def report(self) -> dict[str, Any]:
        return {
            "device": dict(self.device),
            "initial_compute_apps": list(self.initial_compute_apps),
            "sample_count": len(self.samples),
            "sample_interval_ms": int(round(self.interval_s * 1000)),
            "peak_memory_used_mib": max(
                (sample["memory_used_mib"] for sample in self.samples), default=None,
            ),
            "sampler_errors": list(self.errors),
        }


@dataclass
class StepState:
    hidden: np.ndarray
    logits: np.ndarray
    token: int
    logits_ns: int


@dataclass
class Trial:
    name: str
    states: list[StepState]
    transitions: list[dict[str, Any]]
    profile: dict[str, Any]
    generated: list[int]
    started_ns: int
    ended_ns: int


def _native_snapshot() -> dict[str, int] | None:
    return qk_cuda.profile_snapshot()


def _prepare_request(engine: Engine, ids: list[int], n_new: int):
    rope_rows = engine._require_context(len(ids) + max(n_new - 1, 0))
    cos, sin = engine._rope(rope_rows)
    cache = c2.KVCache(engine.NL, max_length=rope_rows)
    hidden = engine._embed(ids)
    try:
        for layer in range(engine.NL):
            hidden = engine._block(hidden, layer, cache, cos, sin)
    finally:
        qk_cuda.release_moe_workspace()
    device_cache = qk_cuda.ResidentAttentionCache(
        engine.NL, rope_rows, engine.cfg.d_model, engine.cfg.n_heads, engine.cfg.n_kv,
        engine.cfg.head_dim, FA, cos, sin,
    )
    try:
        for layer in range(engine.NL):
            device_cache.import_layer(layer, cache.k[layer], cache.v[layer])
    except Exception:
        device_cache.close()
        raise
    engine._mark_profile_warm()
    return hidden[-1:].copy(), device_cache, cos, sin


def _run_trial(engine: Engine, ids: list[int], n_new: int, *, retained: bool,
               name: str) -> Trial:
    engine.reset_profile()
    started_ns = time.monotonic_ns()
    hidden, cache, cos, sin = _prepare_request(engine, ids, n_new)
    states: list[StepState] = []
    transitions: list[dict[str, Any]] = []
    generated: list[int] = []
    try:
        for step in range(n_new):
            logits_started = time.perf_counter_ns()
            logits = engine._klin(
                "token_embd.weight", engine._norm(hidden, "output_norm.weight")
            )[0]
            logits_ns = time.perf_counter_ns() - logits_started
            token = int(np.asarray(logits).argmax())
            states.append(StepState(hidden.copy(), logits.copy(), token, logits_ns))
            generated.append(token)
            if step + 1 == n_new:
                break
            before = _native_snapshot()
            workspace_before = cache.workspace_bytes()
            transition_started = time.perf_counter_ns()
            next_hidden = engine._embed([token])
            if retained:
                next_hidden = engine.resident_decode_token(next_hidden, cache, cos, sin)
            else:
                for layer in range(engine.NL):
                    next_hidden = engine._block(next_hidden, layer, cache, cos, sin)
            transition_ns = time.perf_counter_ns() - transition_started
            workspace_after = cache.workspace_bytes()
            after = _native_snapshot()
            transitions.append({
                "input_step": step,
                "bucket": "cold" if retained and step == 0 else "warm",
                "wall_ns": transition_ns,
                "native_cuda": _counter_delta(before, after),
                "request_workspace_bytes_before": workspace_before,
                "request_workspace_bytes_after": workspace_after,
            })
            hidden = next_hidden
        ended_ns = time.monotonic_ns()
        return Trial(
            name=name, states=states, transitions=transitions,
            profile=engine.profile_snapshot(), generated=generated,
            started_ns=started_ns, ended_ns=ended_ns,
        )
    finally:
        cache.close()


def _trial_evidence(trial: Trial, phase_peak_mib: int | None) -> dict[str, Any]:
    warm_latencies = [
        transition["wall_ns"] + trial.states[index + 1].logits_ns
        for index, transition in enumerate(trial.transitions) if index >= 1
    ]
    warm_seconds = sum(warm_latencies) / 1e9
    return {
        "name": trial.name,
        "generated_tokens": list(trial.generated),
        "states": [
            {
                "step": index,
                "token": state.token,
                "hidden": _array_evidence(state.hidden),
                "logits": _array_evidence(state.logits),
                "logits_wall_ns": state.logits_ns,
            }
            for index, state in enumerate(trial.states)
        ],
        "transitions": trial.transitions,
        "warm_measured_tokens": len(warm_latencies),
        "warm_wall_ns": sum(warm_latencies),
        "warm_tokens_per_second": (len(warm_latencies) / warm_seconds if warm_seconds else None),
        "phase_peak_memory_used_mib": phase_peak_mib,
        "profile": trial.profile,
    }


def _parity_evidence(expected: Trial, actual: Trial) -> dict[str, Any]:
    rows = []
    for index, (left, right) in enumerate(zip(expected.states, actual.states)):
        rows.append({
            "step": index,
            "token_exact": left.token == right.token,
            "expected_token": left.token,
            "actual_token": right.token,
            "hidden": _array_comparison(left.hidden, right.hidden),
            "logits": _array_comparison(left.logits, right.logits),
        })
    same_length = len(expected.states) == len(actual.states)
    return {
        "exact": same_length and all(
            row["token_exact"] and row["hidden"]["exact"] and row["logits"]["exact"]
            for row in rows
        ),
        "same_state_count": same_length,
        "expected_state_count": len(expected.states),
        "actual_state_count": len(actual.states),
        "steps": rows,
    }


def evaluate_gate(*, parity: dict[str, Any], baseline: dict[str, Any],
                  candidate: dict[str, Any], gpu: dict[str, Any],
                  max_throughput_regression: float,
                  max_memory_fraction: float) -> dict[str, Any]:
    baseline_rate = baseline.get("warm_tokens_per_second")
    candidate_rate = candidate.get("warm_tokens_per_second")
    throughput_ratio = (
        candidate_rate / baseline_rate
        if baseline_rate is not None and candidate_rate is not None and baseline_rate > 0 else None
    )
    transitions = candidate.get("transitions", [])
    warm_transitions = transitions[1:]
    allocation_stable = bool(warm_transitions) and all(
        transition.get("native_cuda") is not None
        and transition["native_cuda"].get("allocation_calls") == 0
        and transition.get("request_workspace_bytes_before")
            == transition.get("request_workspace_bytes_after")
        for transition in warm_transitions
    )
    peak = gpu.get("peak_memory_used_mib")
    total = gpu.get("device", {}).get("memory_total_mib")
    memory_ratio = peak / total if peak is not None and total else None
    checks = {
        "exact_hidden_logit_token_parity": bool(parity.get("exact")),
        "warm_allocation_stability": allocation_stable,
        "warm_throughput_no_regression": (
            throughput_ratio is not None
            and throughput_ratio >= 1.0 - max_throughput_regression
        ),
        "combined_peak_below_limit": (
            memory_ratio is not None and memory_ratio <= max_memory_fraction
        ),
        "isolated_gpu_at_start": not gpu.get("initial_compute_apps"),
        "gpu_sampling_complete": (
            bool(gpu.get("sample_count")) and not gpu.get("sampler_errors")
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "measurements": {
            "throughput_ratio_candidate_over_established": throughput_ratio,
            "throughput_regression_fraction": (
                1.0 - throughput_ratio if throughput_ratio is not None else None
            ),
            "combined_peak_memory_fraction": memory_ratio,
            "warm_candidate_transition_count": len(warm_transitions),
        },
        "limits": {
            "max_throughput_regression": max_throughput_regression,
            "max_memory_fraction": max_memory_fraction,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="real North Mini Code GGUF")
    parser.add_argument("--tokenizer", type=Path, default=None, help="extracted tokenizer directory")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--new-tokens", type=int, default=4,
                        help="greedy tokens; at least 4 gives two warm retained transitions")
    parser.add_argument("--output", type=Path, required=True, help="atomic JSON evidence destination")
    parser.add_argument("--bundle-dir", type=Path, default=None,
                        help="local signed bundle directory (default: beside --output)")
    parser.add_argument("--model-key", type=Path, default=None,
                        help="optional persistent model signing-key path")
    parser.add_argument("--counterparty-key", type=Path, default=None,
                        help="optional persistent counterparty signing-key path")
    parser.add_argument("--expected-model-sha256", default=None)
    parser.add_argument("--gpu-index", type=int, default=None,
                        help="physical nvidia-smi index; required when multiple GPUs are visible")
    parser.add_argument("--sample-ms", type=int, default=100)
    parser.add_argument("--max-throughput-regression", type=float, default=0.05)
    parser.add_argument("--max-memory-fraction", type=float, default=0.95)
    args = parser.parse_args(argv)
    if args.new_tokens < 4:
        parser.error("--new-tokens must be at least 4 to measure two warm transitions")
    if not 0 <= args.max_throughput_regression < 1:
        parser.error("--max-throughput-regression must be in [0, 1)")
    if not 0 < args.max_memory_fraction <= 1:
        parser.error("--max-memory-fraction must be in (0, 1]")
    if args.expected_model_sha256 is not None:
        expected = args.expected_model_sha256
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            parser.error("--expected-model-sha256 must be 64 lowercase hex characters")
    return args


def _require_runtime(engine: Engine) -> None:
    if engine.bname != "cuda-resident" or not engine.resident:
        raise RuntimeError(f"gate requires cuda-resident, actual backend: {engine.bname}")
    if not qk_cuda.resident_layer_available():
        raise RuntimeError("current CUDA ABI does not expose the retained layer executor")
    if not engine.resident_attention:
        raise RuntimeError("resident attention must be enabled")
    if engine.DENSE != 1 or engine.NL != 49:
        raise RuntimeError(
            f"gate is committed to the real 1 dense + 48 MoE architecture, got "
            f"{engine.DENSE} dense + {engine.NL - engine.DENSE} MoE"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    root = Path(__file__).resolve().parents[1]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "gate": "north-mini-code-resident-layer-real-model",
        "status": "error",
        "started_utc": started_utc,
        "source": _source_identity(root),
        "input": {
            "model_size_bytes": None,
            "model_sha256": None,
            "expected_model_sha256": args.expected_model_sha256,
            "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
            "new_tokens": args.new_tokens,
        },
    }
    sampler = None
    engine = None
    try:
        if not args.model.is_file():
            raise FileNotFoundError(f"model is not a regular file: {args.model}")
        report["input"]["model_size_bytes"] = args.model.stat().st_size
        print("hashing model", file=sys.stderr, flush=True)
        model_hash = _sha256_file(args.model)
        report["input"]["model_sha256"] = model_hash
        if args.expected_model_sha256 is not None and model_hash != args.expected_model_sha256:
            raise RuntimeError("model SHA-256 does not match --expected-model-sha256")

        sampler = NvidiaSmiSampler(args.gpu_index, args.sample_ms)
        if sampler.initial_compute_apps:
            raise RuntimeError("selected GPU is not isolated at gate start")
        sampler.start()
        print("loading resident engine", file=sys.stderr, flush=True)
        engine = Engine(
            args.model, tok_dir=args.tokenizer, backend="cuda-resident", profile=True,
            resident_attention=True, resident_preprocess=False, resident_layer_executor=True,
        )
        _require_runtime(engine)
        ids = engine.encode(args.prompt)
        report["input"]["prompt_tokens"] = len(ids)
        report["model"] = {
            "architecture": engine.g.kv.get("general.architecture"),
            "layers": engine.NL,
            "leading_dense_layers": engine.DENSE,
            "d_model": engine.cfg.d_model,
            "query_width": engine.cfg.n_heads * engine.cfg.head_dim,
            "attention_heads": engine.cfg.n_heads,
            "kv_heads": engine.cfg.n_kv,
            "head_dim": engine.cfg.head_dim,
            "experts": engine.cfg.n_experts,
            "experts_used": engine.cfg.n_used,
            "expert_ffn": engine.cfg.expert_ffn,
            "context_length": engine.context_length,
            "backend": engine.bname,
            "resident_layer_abi": qk_cuda.resident_layer_available(),
        }
        cuda_library = qk_cuda._so_path()
        report["runtime"] = {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "cuda_library_path": str(cuda_library),
            "cuda_library_sha256": _sha256_file(cuda_library),
            "cuda_abi_version": qk_cuda._CUDA_ABI_VERSION,
        }

        print("reference pass (route warm-up)", file=sys.stderr, flush=True)
        # This is intentionally the public, established production entry. The
        # later instrumented established trial must reproduce it exactly, so a
        # gate cannot accidentally validate two copies of only its own helper.
        public_generate = engine.generate(ids, args.new_tokens, stop_eos=False)
        print("measuring established path", file=sys.stderr, flush=True)
        baseline = _run_trial(engine, ids, args.new_tokens, retained=False, name="established")
        print("measuring retained layer path", file=sys.stderr, flush=True)
        candidate = _run_trial(engine, ids, args.new_tokens, retained=True, name="resident_layer")
        sampler.stop()

        gpu = sampler.report()
        baseline_evidence = _trial_evidence(
            baseline, sampler.phase_peak(baseline.started_ns, baseline.ended_ns),
        )
        candidate_evidence = _trial_evidence(
            candidate, sampler.phase_peak(candidate.started_ns, candidate.ended_ns),
        )
        parity = _parity_evidence(baseline, candidate)
        public_generate_exact = (
            list(public_generate) == baseline.generated == candidate.generated
        )
        parity["public_generate_tokens"] = list(public_generate)
        parity["public_generate_exact"] = public_generate_exact
        parity["exact"] = bool(parity["exact"] and public_generate_exact)
        verdict = evaluate_gate(
            parity=parity, baseline=baseline_evidence, candidate=candidate_evidence, gpu=gpu,
            max_throughput_regression=args.max_throughput_regression,
            max_memory_fraction=args.max_memory_fraction,
        )
        # Preserve the expensive hardware evidence even if receipt setup or
        # signing later fails (for example, a deploy omitted NMC_BONSAI_SRC).
        report.update({
            "gpu": gpu,
            "established": baseline_evidence,
            "resident_layer": candidate_evidence,
            "parity": parity,
            "verdict": verdict,
        })

        receipt_evidence: dict[str, Any] = {
            "passed": False,
            "status": "not-run",
            "reason": "exact established/resident/public replay parity is required",
        }
        # Receipt replay is an independent acceptance dimension.  Build it as
        # soon as the signed input/output commitments are known exact; a
        # throughput or memory failure must not be misreported as a signature
        # replay failure that was never attempted.
        if parity["exact"]:
            print("building local signed receipt bundle", file=sys.stderr, flush=True)
            try:
                from nmc import receipts_runtime as rr
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError(
                    "receipt dependencies unavailable; install requirements_receipts.txt and set "
                    "NMC_BONSAI_SRC to the deployed Bonsai src directory"
                ) from exc
            model_hash, artifact_digest = rr.model_hash(engine)
            receipt_fa = engine.cfg.fa
            # model_hash consumes only GGUF/config/RoPE state. Once it is
            # bound, release all resident handles before key access/signing;
            # no private receipt operation should retain a GPU allocation.
            engine.free()
            model_key, counterparty_key = rr.load_keys(args.model_key, args.counterparty_key)
            bundle_dir = args.bundle_dir or args.output.with_name(f"{args.output.stem}-bundle")
            ledger_path = bundle_dir.with_name(f"{bundle_dir.name}-ledger.jsonl")
            receipt_result = rr.build_verify_pack(
                model_hash=model_hash,
                artifact_digest=artifact_digest,
                input_ids=ids,
                output_ids=candidate.generated,
                sampler=rr.SamplerConfig(mode="greedy"),
                fa=receipt_fa,
                model_key=model_key,
                counterparty_key=counterparty_key,
                out_dir=bundle_dir,
                ledger_path=ledger_path,
                enable_chain=False,
                chain_backend=None,
                broadcast_to_log=False,
            )
            receipt = receipt_result["receipt"]
            emission_onchain = receipt_result["emission"].get("onchain") or {}
            manifest = receipt_result["bundle"].get("manifest") or {}
            manifest_files = set((manifest.get("files") or {}).keys())
            expected_local_files = {
                "receipt.json", "preimage.json", "chain-artifact.json", "ledger-head.json",
            }
            signed_output_commit = receipt["outputCommit"]
            replay_output_commit = rr.token_commit(public_generate)
            resident_output_commit = rr.token_commit(candidate.generated)
            signed_input_commit = receipt["inputCommit"]
            replay_input_commit = rr.token_commit(ids)
            receipt_checks = {
                "local_only_no_chain": (
                    emission_onchain.get("status") == "disabled"
                    and manifest.get("kind") == "local"
                    and "onchain.json" not in manifest_files
                ),
                "local_bundle_shape": manifest_files == expected_local_files,
                "offline_receipt_verified": bool(receipt_result["offline_ok"]),
                "offline_bundle_verified": bool(receipt_result["verify_bundle"].get("ok")),
                "established_replay_tokens_exact": list(public_generate) == candidate.generated,
                "signed_input_commit_reproduced": signed_input_commit == replay_input_commit,
                "signed_output_commit_reproduced": (
                    signed_output_commit == replay_output_commit == resident_output_commit
                ),
                "model_hash_bound": receipt.get("modelHash") == model_hash,
            }
            receipt_evidence = {
                "passed": all(receipt_checks.values()),
                "status": "passed" if all(receipt_checks.values()) else "failed",
                "checks": receipt_checks,
                "bundle_path": receipt_result["bundle"]["path"],
                "bundle_hash": receipt_result["bundle"]["bundleHash"],
                "bundle_kind": manifest["kind"],
                "bundle_files": sorted(manifest_files),
                "ledger_path": str(ledger_path),
                "ledger_index": receipt_result["ledger_entry"].get("index"),
                "receipt_hash": receipt["receiptHash"],
                "model_hash": model_hash,
                "artifact_digest": artifact_digest,
                "input_commit": signed_input_commit,
                "output_commit": signed_output_commit,
                "model_public_key": receipt.get("sigModelPubKey"),
                "counterparty_public_key": receipt.get("sigCounterpartyPubKey"),
                "publish_status": emission_onchain.get("status"),
                "resident_gpu_released_before_signing": True,
            }
        verdict["checks"]["local_signed_receipt_replay"] = receipt_evidence["passed"]
        verdict["passed"] = all(verdict["checks"].values())
        report.update({
            "receipt": receipt_evidence,
            "verdict": verdict,
            "status": "passed" if verdict["passed"] else "failed",
        })
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if sampler is not None:
            sampler.stop()
            report["gpu"] = sampler.report()
    finally:
        if engine is not None:
            try:
                engine.free()
            except Exception as exc:
                report.setdefault("cleanup_errors", []).append(f"{type(exc).__name__}: {exc}")
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_write_json(args.output, report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
