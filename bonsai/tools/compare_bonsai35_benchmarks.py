#!/usr/bin/env python3
"""Fail-closed comparison of controlled Bonsai-27B CPU benchmark records.

The utility is deliberately read-only with respect to benchmark inputs.  It
does not load a model or inference library.  A successful comparison binds the
two raw JSON files, verifies their provenance and repeated commitments, then
computes like-for-like median speedups.  The comparison itself is written
atomically so publication tooling never observes a partial acceptance record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable


FORMAT = "trinote-bonsai35-benchmark-comparison/1"
BENCHMARK_FORMAT = "trinote-bonsai35-benchmark/1"
MINIMUM_REPETITIONS = 5
REQUIRED_SOURCE_FILES = {
    "tools/bench_bonsai35.py",
    "tools/bonsai_q1_kernel.c",
    "src/trinote/infer_int/q1_native.py",
    "src/trinote/infer_int/reference_bonsai35.py",
    "src/trinote/infer_int/reference_bonsai.py",
    "src/trinote/infer_int/sampler.py",
    "src/trinote/infer_int/trace_bonsai35.py",
    "src/trinote/infer_int/artifact_io_bonsai.py",
    "src/trinote/determinism/fixedpoint.py",
}

THREAD_KEYS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_WAIT_POLICY",
    "OMP_PLACES",
    "OMP_PROC_BIND",
    "OMP_MAX_ACTIVE_LEVELS",
    "GOMP_SPINCOUNT",
    "KMP_BLOCKTIME",
)

METRICS = {
    "prefill": "prompt_prefill_s",
    "ttft": "time_to_first_output_token_compute_s",
    "token_3_32": "steady_decode_token_3_32_median_s",
    "token_33_128": "steady_decode_token_33_128_median_s",
}


class ComparisonError(ValueError):
    """A fail-closed comparison rejection with a stable machine code."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details


def _reject(
    code: str, message: str, details: dict[str, Any] | None = None
) -> None:
    raise ComparisonError(code, message, details)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        _reject("input_read", f"cannot read {path}: {exc}")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _reject("input_json", f"invalid JSON in {path}: {exc}")
    if not isinstance(value, dict):
        _reject("input_schema", f"benchmark {path} must contain one JSON object")
    return value, payload


def _iterations(run: dict[str, Any], label: str) -> list[dict[str, Any]]:
    workers = run.get("workers")
    if not isinstance(workers, list) or not workers:
        _reject("run_schema", f"{label} has no benchmark workers")
    result: list[dict[str, Any]] = []
    for worker in workers:
        if not isinstance(worker, dict):
            _reject("run_schema", f"{label} contains a non-object worker")
        rows = worker.get("iterations")
        if not isinstance(rows, list):
            _reject("run_schema", f"{label} worker has no iteration list")
        if not all(isinstance(row, dict) for row in rows):
            _reject("run_schema", f"{label} contains a non-object iteration")
        result.extend(rows)
    return result


def _single_worker_value(
    run: dict[str, Any],
    label: str,
    identity_name: str,
    extract: Callable[[dict[str, Any]], Any],
) -> Any:
    values = []
    for worker in run["workers"]:
        value = extract(worker)
        values.append(value)
    if not values or any(value is None for value in values):
        _reject("identity_invalid", f"{label} lacks {identity_name} identity")
    if len({_canonical(value) for value in values}) != 1:
        _reject("identity_invalid", f"{label} workers disagree on {identity_name} identity")
    return values[0]


def _source_identity(worker: dict[str, Any]) -> dict[str, Any] | None:
    source = worker.get("environment", {}).get("source")
    if not isinstance(source, dict):
        return None
    result = {
        "revision": source.get("revision"),
        "dirty": source.get("dirty"),
        "porcelain_entry_count": source.get("porcelain_entry_count"),
        "porcelain_sha256": source.get("porcelain_sha256"),
        "files": source.get("files"),
    }
    if (
        not isinstance(result["revision"], str)
        or type(result["dirty"]) is not bool
        or type(result["porcelain_entry_count"]) is not int
        or not _is_sha256(result["porcelain_sha256"])
        or not isinstance(result["files"], dict)
        or set(result["files"]) != REQUIRED_SOURCE_FILES
    ):
        return None
    for relative, identity in result["files"].items():
        if (
            not isinstance(relative, str)
            or not isinstance(identity, dict)
            or not _is_sha256(identity.get("sha256"))
            or type(identity.get("size_bytes")) is not int
            or identity["size_bytes"] <= 0
        ):
            return None
    return result


def _kernel_identity(worker: dict[str, Any]) -> dict[str, Any] | None:
    kernel = worker.get("environment", {}).get("kernel")
    if not isinstance(kernel, dict) or kernel.get("present") is not True:
        return None
    result = {
        "sha256": kernel.get("sha256"),
        "elf_build_id": kernel.get("elf_build_id"),
        "compiler_comment": kernel.get("compiler_comment"),
    }
    if not _is_sha256(result["sha256"]):
        return None
    if result["elf_build_id"] is not None and not isinstance(result["elf_build_id"], str):
        return None
    if result["compiler_comment"] is not None and not isinstance(result["compiler_comment"], str):
        return None
    return result


def _artifact_identity(worker: dict[str, Any]) -> dict[str, Any] | None:
    artifact = worker.get("artifact")
    if not isinstance(artifact, dict):
        return None
    result = {
        "sha256": artifact.get("sha256"),
        "size_bytes": artifact.get("size_bytes"),
    }
    if not _is_sha256(result["sha256"]):
        return None
    if type(result["size_bytes"]) is not int or result["size_bytes"] <= 0:
        return None
    return result


def _thread_identity(run: dict[str, Any], label: str) -> dict[str, Any]:
    control = run.get("control")
    if not isinstance(control, dict):
        _reject("identity_invalid", f"{label} lacks benchmark controls")
    environment = control.get("thread_environment")
    if not isinstance(environment, dict):
        _reject("identity_invalid", f"{label} lacks thread environment")
    selected = {key: environment.get(key) for key in THREAD_KEYS}
    if any(not isinstance(value, str) or not value for value in selected.values()):
        _reject("identity_invalid", f"{label} has an incomplete thread environment")
    effective_affinity = control.get("effective_affinity")
    requested_affinity = control.get("requested_affinity")
    if (
        not isinstance(effective_affinity, list)
        or not all(type(cpu) is int for cpu in effective_affinity)
        or not isinstance(requested_affinity, list)
        or not all(type(cpu) is int for cpu in requested_affinity)
    ):
        _reject("identity_invalid", f"{label} has invalid affinity identity")
    return {
        "environment": selected,
        "core_mode": control.get("core_mode"),
        "requested_affinity": requested_affinity,
        "effective_affinity": effective_affinity,
    }


def _workload_identity(
    run: dict[str, Any], label: str, iterations: list[dict[str, Any]]
) -> dict[str, Any]:
    configuration = run.get("configuration")
    control = run.get("control")
    if not isinstance(configuration, dict) or not isinstance(control, dict):
        _reject("identity_invalid", f"{label} lacks workload configuration")
    input_values = [iteration.get("input_ids") for iteration in iterations]
    if (
        not input_values
        or any(
            not isinstance(value, list)
            or not value
            or not all(type(token) is int for token in value)
            for value in input_values
        )
        or len({_canonical(value) for value in input_values}) != 1
    ):
        _reject("identity_invalid", f"{label} repetitions disagree on input token IDs")
    workload_keys = (
        "raw_ids", "prompt", "chat", "max_new", "sampler", "seed", "ignore_eos"
    )
    return {
        "mode": run.get("mode"),
        "condition": run.get("condition"),
        "configuration": {key: configuration.get(key) for key in workload_keys},
        "input_ids": input_values[0],
        "prefill_q1_chunk": control.get("prefill_q1_chunk"),
        "repetitions": control.get("repetitions"),
    }


def _commitment_record(
    run: dict[str, Any], label: str, iterations: list[dict[str, Any]]
) -> dict[str, Any]:
    aggregate = run.get("aggregate")
    if not isinstance(aggregate, dict):
        _reject("run_schema", f"{label} lacks an aggregate record")
    gate = aggregate.get("commitment_consistency_gate")
    if not isinstance(gate, dict):
        _reject("commitment_consistency", f"{label} lacks the commitment gate")

    records: list[tuple[str, str, int]] = []
    for iteration in iterations:
        commitments = iteration.get("commitments")
        output_count = iteration.get("output_token_count")
        if not isinstance(commitments, dict):
            _reject("commitment_consistency", f"{label} iteration lacks commitments")
        output_hash = commitments.get("output_ids_sha256")
        cache_hash = commitments.get("cache_state_sha256")
        layer_hash = commitments.get("layer_trace_sha256")
        if (
            not _is_sha256(output_hash)
            or not _is_sha256(cache_hash)
            or type(output_count) is not int
            or output_count < 0
        ):
            _reject("commitment_consistency", f"{label} iteration has invalid commitments")
        if layer_hash is not None and layer_hash != cache_hash:
            _reject("commitment_consistency", f"{label} cache commitment aliases disagree")
        output_ids = iteration.get("output_ids")
        if isinstance(output_ids, list) and len(output_ids) != output_count:
            _reject("commitment_consistency", f"{label} output count disagrees with output IDs")
        records.append((output_hash, cache_hash, output_count))

    output_hashes = sorted({record[0] for record in records})
    cache_hashes = sorted({record[1] for record in records})
    output_counts = sorted({record[2] for record in records})
    recomputed_pass = (
        len(records) == len(iterations)
        and len(output_hashes) == 1
        and len(cache_hashes) == 1
        and len(output_counts) == 1
    )
    gate_matches = (
        gate.get("applicable") is True
        and gate.get("eligible") is True
        and gate.get("passed") is True
        and gate.get("iteration_count") == len(iterations)
        and gate.get("complete_record_count") == len(iterations)
        and gate.get("output_ids_sha256_values") == output_hashes
        and gate.get("cache_state_sha256_values") == cache_hashes
        and gate.get("output_token_count_values") == output_counts
        and gate.get("output_ids_sha256_identical") is True
        and gate.get("cache_state_sha256_identical") is True
        and gate.get("output_token_count_identical") is True
    )
    if not recomputed_pass or not gate_matches:
        _reject("commitment_consistency", f"{label} repetition commitments are inconsistent")
    return {
        "iteration_count": len(iterations),
        "output_ids_sha256": output_hashes[0],
        "cache_state_sha256": cache_hashes[0],
        "output_token_count": output_counts[0],
    }


def _validate_run(run: dict[str, Any], expected_producer: str, label: str) -> dict[str, Any]:
    if run.get("format") != BENCHMARK_FORMAT or run.get("mode") != "model":
        _reject("run_schema", f"{label} is not a Bonsai-27B model benchmark")
    configuration = run.get("configuration")
    if not isinstance(configuration, dict) or configuration.get("producer") != expected_producer:
        _reject("producer_role", f"{label} must use producer {expected_producer}")
    if run.get("accepted") is not True:
        _reject("acceptance", f"{label} benchmark was rejected by its controller")
    control = run.get("control")
    if not isinstance(control, dict):
        _reject("run_schema", f"{label} lacks benchmark controls")
    if control.get("busy_override") is not False or run.get("rejection_reasons"):
        _reject("acceptance", f"{label} used a busy override or recorded contamination")
    repetitions = control.get("repetitions")
    if type(repetitions) is not int or repetitions < MINIMUM_REPETITIONS:
        _reject("repetitions", f"{label} has fewer than {MINIMUM_REPETITIONS} repetitions")

    iterations = _iterations(run, label)
    aggregate = run.get("aggregate")
    if (
        not isinstance(aggregate, dict)
        or aggregate.get("iteration_count") != len(iterations)
        or len(iterations) != repetitions
    ):
        _reject("repetitions", f"{label} repetition accounting is inconsistent")
    variation = aggregate.get("variation_exit_gate")
    if not isinstance(variation, dict):
        _reject("variation", f"{label} lacks a variation gate")
    sample_counts = variation.get("sample_counts")
    if (
        variation.get("eligible") is not True
        or variation.get("passed") is not True
        or not isinstance(sample_counts, dict)
        or not sample_counts
        or any(
            type(count) is not int or count < MINIMUM_REPETITIONS
            for count in sample_counts.values()
        )
    ):
        _reject("variation", f"{label} variation gate is ineligible or failed")

    workers = run["workers"]
    if any(worker.get("producer") != expected_producer for worker in workers):
        _reject("producer_role", f"{label} worker producer identity is inconsistent")
    artifact = _single_worker_value(run, label, "artifact", _artifact_identity)
    source = _single_worker_value(run, label, "source", _source_identity)
    kernel = _single_worker_value(run, label, "kernel", _kernel_identity)
    thread = _thread_identity(run, label)
    workload = _workload_identity(run, label, iterations)
    commitments = _commitment_record(run, label, iterations)
    return {
        "artifact": artifact,
        "source": source,
        "kernel": kernel,
        "thread": thread,
        "workload": workload,
        "commitments": commitments,
    }


def _positive_median(summary: Any) -> float | None:
    if not isinstance(summary, dict):
        return None
    value = summary.get("median")
    if type(value) not in (int, float):
        return None
    value = float(value)
    return value if math.isfinite(value) and value > 0 else None


def _metric_comparison(
    legacy: dict[str, Any], native: dict[str, Any], metric_key: str
) -> dict[str, Any]:
    legacy_summary = legacy.get("aggregate", {}).get("metrics", {}).get(metric_key)
    native_summary = native.get("aggregate", {}).get("metrics", {}).get(metric_key)
    legacy_median = _positive_median(legacy_summary)
    native_median = _positive_median(native_summary)
    available = legacy_median is not None and native_median is not None
    return {
        "metric_key": metric_key,
        "available": available,
        "legacy_summary": legacy_summary,
        "native_summary": native_summary,
        "speedup": legacy_median / native_median if available else None,
    }


def _threshold_record(
    metrics: dict[str, dict[str, Any]], names: tuple[str, ...], minimum: float | None
) -> dict[str, Any]:
    if minimum is None:
        return {"required": False, "minimum_speedup": None, "passed": None}
    if not math.isfinite(minimum) or minimum <= 0:
        _reject("threshold_configuration", "speedup thresholds must be finite and positive")
    unavailable = [name for name in names if not metrics[name]["available"]]
    if unavailable:
        _reject("metric_missing", f"required metrics are unavailable: {', '.join(unavailable)}")
    values = {name: metrics[name]["speedup"] for name in names}
    passed = all(value >= minimum for value in values.values())
    return {
        "required": True,
        "minimum_speedup": minimum,
        "observed_speedups": values,
        "passed": passed,
    }


def compare_runs(
    legacy: dict[str, Any],
    native: dict[str, Any],
    *,
    minimum_steady_speedup: float | None = 4.0,
    minimum_prompt_speedup: float | None = None,
) -> dict[str, Any]:
    """Validate and compare two already-loaded benchmark JSON objects."""

    legacy_identity = _validate_run(legacy, "legacy-native", "legacy")
    native_identity = _validate_run(native, "native", "native")
    for name in ("artifact", "source", "kernel", "thread", "workload"):
        if legacy_identity[name] != native_identity[name]:
            _reject(f"{name}_identity_mismatch", f"legacy/native {name} identity differs")
    if legacy_identity["commitments"] != native_identity["commitments"]:
        _reject("cross_commitments", "legacy/native output, cache, or output count differs")

    metrics = {
        name: _metric_comparison(legacy, native, key)
        for name, key in METRICS.items()
    }
    steady = _threshold_record(
        metrics, ("token_3_32", "token_33_128"), minimum_steady_speedup
    )
    prompt = _threshold_record(
        metrics, ("prefill", "ttft"), minimum_prompt_speedup
    )
    threshold_details = {
        "metrics": metrics,
        "thresholds": {"steady": steady, "prompt": prompt},
    }
    if steady["required"] and not steady["passed"]:
        _reject(
            "steady_threshold",
            "one or more steady-decode speedups missed the threshold",
            threshold_details,
        )
    if prompt["required"] and not prompt["passed"]:
        _reject(
            "prompt_threshold",
            "prefill or TTFT speedup missed the threshold",
            threshold_details,
        )

    return {
        "format": FORMAT,
        "status": "pass",
        "roles": {"legacy": "legacy-native", "native": "native"},
        "identity": {
            key: legacy_identity[key]
            for key in ("artifact", "source", "kernel", "thread", "workload")
        },
        "commitments": {
            "legacy": legacy_identity["commitments"],
            "native": native_identity["commitments"],
            "cross_producer_equal": True,
        },
        "metrics": metrics,
        "thresholds": {"steady": steady, "prompt": prompt},
        "hardware_counters": {
            "legacy": legacy.get("hardware_counters"),
            "native": native.get("hardware_counters"),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed comparison of matched Bonsai-27B CPU benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--legacy", required=True, help="legacy-native benchmark JSON")
    parser.add_argument("--native", required=True, help="resident native benchmark JSON")
    parser.add_argument("--json-out", required=True, help="atomic comparison JSON output")
    parser.add_argument("--min-steady-speedup", type=float, default=4.0)
    parser.add_argument(
        "--skip-steady", action="store_true",
        help="do not require steady windows (use only for a prompt-only comparison)",
    )
    parser.add_argument(
        "--require-prompt-speedup", action="store_true",
        help="require both prefill and compute-TTFT speedups",
    )
    parser.add_argument("--min-prompt-speedup", type=float, default=3.0)
    return parser


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (_canonical(value) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    legacy_path = Path(args.legacy).expanduser()
    native_path = Path(args.native).expanduser()
    output_path = Path(args.json_out).expanduser()
    if output_path.resolve() in {legacy_path.resolve(), native_path.resolve()}:
        report = {
            "format": FORMAT,
            "status": "fail",
            "error": {
                "code": "output_conflict",
                "message": "comparison output must not replace an input benchmark",
            },
        }
        print(_canonical(report))
        return 2

    inputs: dict[str, Any] = {}
    legacy: dict[str, Any] | None = None
    native: dict[str, Any] | None = None
    try:
        legacy, legacy_bytes = _load_json(legacy_path)
        inputs["legacy"] = {
            "path": str(legacy_path.resolve()),
            "sha256": _sha256_bytes(legacy_bytes),
            "size_bytes": len(legacy_bytes),
        }
        native, native_bytes = _load_json(native_path)
        inputs["native"] = {
            "path": str(native_path.resolve()),
            "sha256": _sha256_bytes(native_bytes),
            "size_bytes": len(native_bytes),
        }
        if args.skip_steady and not args.require_prompt_speedup:
            _reject(
                "threshold_configuration",
                "--skip-steady requires --require-prompt-speedup",
            )
        report = compare_runs(
            legacy,
            native,
            minimum_steady_speedup=(None if args.skip_steady else args.min_steady_speedup),
            minimum_prompt_speedup=(
                args.min_prompt_speedup if args.require_prompt_speedup else None
            ),
        )
        report["inputs"] = inputs
        exit_code = 0
    except ComparisonError as exc:
        report = {
            "format": FORMAT,
            "status": "fail",
            "inputs": inputs,
            "error": {"code": exc.code, "message": str(exc)},
            "threshold_request": {
                "minimum_steady_speedup": (
                    None if args.skip_steady else args.min_steady_speedup
                ),
                "minimum_prompt_speedup": (
                    args.min_prompt_speedup if args.require_prompt_speedup else None
                ),
            },
        }
        if exc.details is not None:
            report["details"] = exc.details
        if legacy is not None or native is not None:
            report["hardware_counters"] = {
                "legacy": legacy.get("hardware_counters") if legacy else None,
                "native": native.get("hardware_counters") if native else None,
            }
        exit_code = 2

    tool_path = Path(__file__).resolve()
    tool_bytes = tool_path.read_bytes()
    report["tool"] = {
        "path": str(tool_path),
        "sha256": _sha256_bytes(tool_bytes),
        "size_bytes": len(tool_bytes),
    }
    payload = _json_bytes(report)
    _atomic_write(output_path, payload)
    print(payload.decode("utf-8"), end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
