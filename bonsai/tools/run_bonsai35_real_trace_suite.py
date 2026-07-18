#!/usr/bin/env python3
"""Write or verify the opt-in Bonsai-27B real-model trace suite.

The 4 GiB artifact is deliberately external to the source checkout.  This
runner consumes a small, reviewable input manifest and an equally portable
directory of JSON trace commitments.  ``write`` is intended to mint canonical
NumPy-oracle expectations; ``verify`` may replay either the oracle or a native
producer against those same bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


INPUT_FORMAT = "trinote-bonsai35-real-trace-inputs/1"
SUITE_FORMAT = "trinote-bonsai35-real-trace-suite/1"
REPORT_FORMAT = "trinote-bonsai35-real-trace-report/1"
DEFAULT_INPUTS = (
    Path.home() / ".local" / "trinote" / "results"
    / "bonsai35-real-trace-suite-v1.inputs.json"
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CASE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def report_exit_code(report: dict[str, Any]) -> int:
    """A machine-readable resident mismatch is still a failing gate."""

    return 1 if report.get("status") == "fail" else 0


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def default_expected_dir(inputs_path: Path) -> Path:
    suffix = ".inputs.json"
    name = inputs_path.name
    stem = name[: -len(suffix)] if name.endswith(suffix) else inputs_path.stem
    return inputs_path.with_name(f"{stem}.expected")


def _ids(case_name: str, entry: Any) -> list[int]:
    if not isinstance(entry, dict):
        raise ValueError(f"input {case_name!r} must be an object")
    values = entry.get("ids")
    if (
        not isinstance(values, list)
        or not values
        or any(type(token) is not int or token < 0 for token in values)
    ):
        raise ValueError(f"input {case_name!r}.ids must be non-empty non-negative integers")
    return values


def load_suite_inputs(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read real-trace input manifest {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != INPUT_FORMAT:
        raise ValueError(f"real-trace input manifest must use {INPUT_FORMAT!r}")
    digest = data.get("artifactSha256")
    if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        raise ValueError("artifactSha256 must be a lowercase SHA-256 digest")
    inputs = data.get("inputs")
    required = {"rawHi", "chatHi", "prompt32Unicode", "prompt128"}
    if not isinstance(inputs, dict) or not required.issubset(inputs):
        raise ValueError(f"inputs must contain {sorted(required)}")
    for name, entry in inputs.items():
        if not isinstance(name, str) or not _CASE.fullmatch(name):
            raise ValueError(f"unsafe trace case name {name!r}")
        _ids(name, entry)
    if _ids("rawHi", inputs["rawHi"]) != [12675]:
        raise ValueError("rawHi must contain the release token ID 12675")
    if inputs["rawHi"].get("expectedNextGreedyToken") != 11:
        raise ValueError("rawHi must commit expected next greedy token 11")
    if len(_ids("prompt32Unicode", inputs["prompt32Unicode"])) != 32:
        raise ValueError("prompt32Unicode must contain exactly 32 token IDs")
    if len(_ids("prompt128", inputs["prompt128"])) < 128:
        raise ValueError("prompt128 must contain at least 128 token IDs")

    traces = data.get("traces")
    if not isinstance(traces, dict):
        raise ValueError("traces must be an object")
    prefill = traces.get("prefill")
    if not isinstance(prefill, list) or set(prefill) != required or len(prefill) != 4:
        raise ValueError("prefill traces must name each required input exactly once")
    cached = traces.get("cachedGreedy")
    if not isinstance(cached, dict):
        raise ValueError("cachedGreedy must be an object")
    if cached.get("input") not in inputs or cached.get("newTokens") != 32:
        raise ValueError("cachedGreedy must generate 32 tokens from a named input")
    return data


def trace_jobs(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    for case_name in inputs["traces"]["prefill"]:
        jobs.append({
            "key": f"prefill/{case_name}",
            "kind": "prefill",
            "case": case_name,
            "tokens": _ids(case_name, inputs["inputs"][case_name]),
            "file": f"prefill-{case_name}.json",
        })
    cached = inputs["traces"]["cachedGreedy"]
    case_name = cached["input"]
    n_new = int(cached["newTokens"])
    jobs.append({
        "key": f"cached-greedy/{case_name}/{n_new}",
        "kind": "cached-greedy",
        "case": case_name,
        "tokens": _ids(case_name, inputs["inputs"][case_name]),
        "newTokens": n_new,
        "file": f"cached-greedy-{case_name}-{n_new}.json",
    })
    return jobs


def _trace_job(artifact: dict, job: dict[str, Any], *, native: bool) -> dict[str, Any]:
    from trinote.infer_int.trace_bonsai35 import trace_cached_greedy, trace_prefill

    if job["kind"] == "prefill":
        return trace_prefill(
            artifact, job["tokens"], native=native, include_intermediates=True
        )
    return trace_cached_greedy(
        artifact,
        job["tokens"],
        int(job["newTokens"]),
        native=native,
        # Keep every cached-token layer/cache checkpoint inspectable rather
        # than retaining only the aggregate SHA used by the small smoke test.
        include_layer_traces=True,
    )


def _load_artifact(path: Path, expected_digest: str) -> tuple[dict, str]:
    from trinote.infer_int.artifact_io_bonsai import load_artifact_bonsai

    artifact, info = load_artifact_bonsai(path)
    if str(artifact.get("config", {}).get("architecture")) != "qwen35":
        raise ValueError("real-trace suite requires a Qwen3.5 artifact")
    actual = str(info["digest"])
    if actual != expected_digest:
        raise ValueError(
            f"artifact digest {actual} does not match input manifest {expected_digest}"
        )
    return artifact, actual


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def verify_resident_cached_trace(
    artifact: dict,
    expected: dict[str, Any],
    executor,
) -> dict[str, Any]:
    """Verify the resident model ABI at the cached-32 release boundary.

    The ordinary ``native`` trace producer deliberately runs the inspectable
    Python graph with native primitives.  It therefore cannot prove that the
    fused one-call resident executor preserves the complete model cache.  This
    check consumes the oracle-minted cached trace and compares all 32
    pre-consume logits/token choices, the final layer-63 residual, and every
    final recurrent state/conv or attention K/V tensor digest.
    """

    from trinote.infer_int.trace_bonsai35 import tensor_digest

    if expected.get("format") != "trinote-bonsai35-trace/1":
        raise ValueError("resident verification expected trace has the wrong format")
    input_ids = expected.get("inputIds")
    steps = expected.get("steps")
    expected_ids = expected.get("outputIds")
    layers = artifact.get("layers")
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError("resident verification needs non-empty expected inputIds")
    if not isinstance(steps, list) or len(steps) != 32:
        raise ValueError("resident release verification requires exactly 32 cached steps")
    if not isinstance(expected_ids, list) or len(expected_ids) != len(steps):
        raise ValueError("resident expected outputIds/steps are inconsistent")
    if not isinstance(layers, list) or not layers:
        raise ValueError("resident verification artifact has no layers")
    final_layer_rows = steps[-1].get("layers")
    if not isinstance(final_layer_rows, list) or len(final_layer_rows) != len(layers):
        raise ValueError("resident verification requires full final cached layer traces")

    stats_before = executor.stats()
    logits = executor.prefill_logits(input_ids)
    generated_ids: list[int] = []
    logits_records: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        expected_logits = step.get("logits")
        actual_logits = tensor_digest(logits)
        logits_records.append({
            "step": index,
            "expected": expected_logits,
            "actual": actual_logits,
            "equal": actual_logits == expected_logits,
        })
        token = int(logits[0].argmax())
        generated_ids.append(token)
        # Consume every oracle-trace token, including token 32.  The first 31
        # decodes also supply the next pre-consume logits; the last needs only
        # the hidden/state update required by the committed final cache.
        if index + 1 < len(steps):
            logits = executor.decode_logits(token)
        else:
            executor.decode(token)

    final_residual_actual = tensor_digest(executor.export_last_residual())
    final_residual_expected = final_layer_rows[-1].get("output")
    cache_records: list[dict[str, Any]] = []
    caches_equal = True
    for layer_index, (layer, expected_layer) in enumerate(
        zip(layers, final_layer_rows)
    ):
        kind = str(layer.get("kind"))
        if expected_layer.get("layer") != layer_index or expected_layer.get("kind") != kind:
            raise ValueError(f"resident expected layer metadata drift at layer {layer_index}")
        names = ("state", "conv") if kind == "recurrent" else ("k", "v") if kind == "attention" else ()
        if not names:
            raise ValueError(f"resident artifact has unknown layer kind {kind!r}")
        expected_cache = expected_layer.get("cache")
        if not isinstance(expected_cache, dict):
            raise ValueError(f"resident expected cache is missing at layer {layer_index}")
        record: dict[str, Any] = {"layer": layer_index, "kind": kind}
        for name in names:
            actual_digest = tensor_digest(executor.export_cache_tensor(layer_index, name))
            expected_digest = expected_cache.get(name)
            equal = actual_digest == expected_digest
            record[name] = {
                "actual": actual_digest,
                "expected": expected_digest,
                "equal": equal,
            }
            caches_equal = caches_equal and equal
        cache_records.append(record)

    stats_after = executor.stats()
    counter_names = ("prefill_calls", "decode_calls", "team_entries")
    counter_delta = {
        name: int(stats_after.get(name, 0)) - int(stats_before.get(name, 0))
        for name in counter_names
    }
    expected_position = len(input_ids) + len(steps)
    acceptance = {
        "generated_ids_equal": generated_ids == [int(value) for value in expected_ids],
        "all_preconsume_logits_equal": all(row["equal"] for row in logits_records),
        "final_layer63_residual_equal": final_residual_actual == final_residual_expected,
        "all_final_cache_tensors_equal": caches_equal,
        "position_exact": int(executor.position()) == expected_position,
        "one_prefill_and_one_decode_team_per_consumed_step": counter_delta == {
            "prefill_calls": 1,
            "decode_calls": len(steps),
            "team_entries": 1 + len(steps),
        },
    }
    return {
        "status": "pass" if all(acceptance.values()) else "fail",
        "acceptance": acceptance,
        "inputIds": [int(value) for value in input_ids],
        "expectedOutputIds": [int(value) for value in expected_ids],
        "generatedOutputIds": generated_ids,
        "preconsumeLogits": logits_records,
        "finalLayer63Residual": {
            "expected": final_residual_expected,
            "actual": final_residual_actual,
            "equal": final_residual_actual == final_residual_expected,
        },
        "finalCaches": cache_records,
        "expectedPosition": expected_position,
        "actualPosition": int(executor.position()),
        "residentCountersBefore": stats_before,
        "residentCountersAfter": stats_after,
        "residentCounterDelta": counter_delta,
    }


def verify_resident_cached_suite(
    *,
    artifact_path: Path,
    inputs_path: Path,
    expected_dir: Path,
) -> dict[str, Any]:
    """Verify the real release resident executor against oracle cached-32."""

    from trinote.infer_int.q1_native import Bonsai35NativeExecutor

    inputs = load_suite_inputs(inputs_path)
    artifact, artifact_digest = _load_artifact(
        artifact_path, inputs["artifactSha256"]
    )
    manifest_path = expected_dir / "suite-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read expected suite manifest {manifest_path}: {exc}") from exc
    if manifest.get("format") != SUITE_FORMAT:
        raise ValueError(f"expected suite must use {SUITE_FORMAT!r}")
    if manifest.get("expectedProducer") != "oracle":
        raise ValueError("resident verification requires NumPy-oracle-minted expectations")
    if manifest.get("artifactSha256") != artifact_digest:
        raise ValueError("expected suite is bound to a different artifact")
    if manifest.get("inputManifestSha256") != sha256_file(inputs_path):
        raise ValueError("expected suite is bound to different trace inputs")
    records = manifest.get("traces")
    jobs = trace_jobs(inputs)
    if not isinstance(records, dict) or set(records) != {job["key"] for job in jobs}:
        raise ValueError("expected suite trace set is incomplete or stale")
    # Validate every suite record before selecting cached-32, so a resident
    # PASS cannot be attached to an otherwise incomplete/mixed suite.
    for job in jobs:
        record = records.get(job["key"])
        path = expected_dir / job["file"]
        if not isinstance(record, dict) or record.get("file") != job["file"]:
            raise ValueError(f"expected suite is missing {job['key']}")
        if sha256_file(path) != record.get("sha256"):
            raise ValueError(f"expected trace digest mismatch for {path}")
    cached_job = next(job for job in jobs if job["kind"] == "cached-greedy")
    cached_record = records[cached_job["key"]]
    expected_path = expected_dir / cached_job["file"]
    expected = json.loads(expected_path.read_text())
    if expected.get("artifactSha256") != artifact_digest:
        raise ValueError("cached expected trace is not bound to the loaded artifact")

    executor = Bonsai35NativeExecutor(artifact)
    try:
        resident = verify_resident_cached_trace(artifact, expected, executor)
    finally:
        executor.close()
    return {
        "format": REPORT_FORMAT,
        "mode": "verify",
        "producer": "resident",
        "artifactSha256": artifact_digest,
        "expectedProducer": manifest["expectedProducer"],
        "expectedTrace": str(expected_path),
        "expectedTraceSha256": cached_record["sha256"],
        "verified": [cached_job["key"]],
        **resident,
    }


def write_expected_suite(
    *,
    artifact_path: Path,
    inputs_path: Path,
    expected_dir: Path,
    producer: str,
    force: bool,
) -> dict[str, Any]:
    if producer != "oracle":
        raise ValueError(
            "canonical expected traces must be minted by the NumPy oracle; "
            "use --mode verify --producer native to validate an optimized producer"
        )
    inputs = load_suite_inputs(inputs_path)
    artifact, artifact_digest = _load_artifact(
        artifact_path, inputs["artifactSha256"]
    )
    native = producer == "native"
    if native:
        from trinote.infer_int.q1_native import q1_set_isa

        q1_set_isa(os.environ.get("TRINOTE_Q1_ISA", "auto"))
    expected_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = expected_dir / "suite-manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"expected suite already exists at {manifest_path}; pass --force to replace it"
        )

    records: dict[str, Any] = {}
    for job in trace_jobs(inputs):
        trace = _trace_job(artifact, job, native=native)
        trace["artifactSha256"] = artifact_digest
        payload = canonical_bytes(trace)
        target = expected_dir / job["file"]
        if target.exists() and not force:
            raise FileExistsError(f"trace already exists at {target}; pass --force")
        _atomic_write(target, payload)
        records[job["key"]] = {
            "file": job["file"],
            "sha256": sha256_bytes(payload),
        }
    manifest = {
        "format": SUITE_FORMAT,
        "artifactSha256": artifact_digest,
        "inputManifestSha256": sha256_file(inputs_path),
        "expectedProducer": producer,
        "traces": records,
    }
    _atomic_write(manifest_path, canonical_bytes(manifest))
    return manifest


def verify_expected_suite(
    *,
    artifact_path: Path,
    inputs_path: Path,
    expected_dir: Path,
    producer: str,
) -> dict[str, Any]:
    from trinote.infer_int.trace_bonsai35 import assert_trace_equal

    inputs = load_suite_inputs(inputs_path)
    artifact, artifact_digest = _load_artifact(
        artifact_path, inputs["artifactSha256"]
    )
    manifest_path = expected_dir / "suite-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read expected suite manifest {manifest_path}: {exc}") from exc
    if manifest.get("format") != SUITE_FORMAT:
        raise ValueError(f"expected suite must use {SUITE_FORMAT!r}")
    if manifest.get("artifactSha256") != artifact_digest:
        raise ValueError("expected suite is bound to a different artifact")
    if manifest.get("inputManifestSha256") != sha256_file(inputs_path):
        raise ValueError("expected suite is bound to different trace inputs")
    expected_records = manifest.get("traces")
    if not isinstance(expected_records, dict):
        raise ValueError("expected suite manifest has no trace records")

    native = producer == "native"
    selected_isa = "oracle"
    if native:
        from trinote.infer_int.q1_native import q1_selected_isa, q1_set_isa

        q1_set_isa(os.environ.get("TRINOTE_Q1_ISA", "auto"))
        selected_isa = q1_selected_isa()
    verified = []
    for job in trace_jobs(inputs):
        record = expected_records.get(job["key"])
        if not isinstance(record, dict) or record.get("file") != job["file"]:
            raise ValueError(f"expected suite is missing {job['key']}")
        expected_path = expected_dir / job["file"]
        if sha256_file(expected_path) != record.get("sha256"):
            raise ValueError(f"expected trace digest mismatch for {expected_path}")
        expected = json.loads(expected_path.read_text())
        actual = _trace_job(artifact, job, native=native)
        actual["artifactSha256"] = artifact_digest
        assert_trace_equal(actual, expected)
        verified.append(job["key"])
    if set(expected_records) != set(verified):
        raise ValueError("expected suite contains unrecognized or stale trace records")
    return {
        "format": REPORT_FORMAT,
        "mode": "verify",
        "producer": producer,
        "selectedIsa": selected_isa,
        "artifactSha256": artifact_digest,
        "expectedProducer": manifest.get("expectedProducer"),
        "verified": verified,
    }


def build_parser() -> argparse.ArgumentParser:
    from trinote.notary_paths import notary_home

    default_artifact = (
        Path(notary_home()) / "models" / "Bonsai-27B-Q1_0-int-qwen35.safetensors"
    )
    parser = argparse.ArgumentParser(
        description="Write or verify canonical Bonsai-27B real-model traces"
    )
    parser.add_argument("--mode", choices=("write", "verify", "plan"), default="verify")
    parser.add_argument(
        "--producer", choices=("oracle", "native", "resident"), default="oracle"
    )
    parser.add_argument("--artifact", type=Path, default=default_artifact)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument(
        "--expected-dir",
        type=Path,
        default=None,
        help="defaults to <inputs basename>.expected beside the input manifest",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--json-out", type=Path,
        help="atomically write the exact canonical stdout report to this path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs_path = args.inputs.expanduser().resolve()
    expected_dir = (
        args.expected_dir.expanduser().resolve()
        if args.expected_dir is not None
        else default_expected_dir(inputs_path)
    )
    if args.mode == "plan":
        inputs = load_suite_inputs(inputs_path)
        report = {
            "format": REPORT_FORMAT,
            "mode": "plan",
            "artifactSha256": inputs["artifactSha256"],
            "expectedDir": str(expected_dir),
            "jobs": trace_jobs(inputs),
        }
    elif args.mode == "write":
        manifest = write_expected_suite(
            artifact_path=args.artifact.expanduser().resolve(),
            inputs_path=inputs_path,
            expected_dir=expected_dir,
            producer=args.producer,
            force=args.force,
        )
        report = {
            "format": REPORT_FORMAT,
            "mode": "write",
            "producer": args.producer,
            "expectedDir": str(expected_dir),
            "manifest": manifest,
        }
    elif args.producer == "resident":
        report = verify_resident_cached_suite(
            artifact_path=args.artifact.expanduser().resolve(),
            inputs_path=inputs_path,
            expected_dir=expected_dir,
        )
        report["expectedDir"] = str(expected_dir)
    else:
        report = verify_expected_suite(
            artifact_path=args.artifact.expanduser().resolve(),
            inputs_path=inputs_path,
            expected_dir=expected_dir,
            producer=args.producer,
        )
        report["expectedDir"] = str(expected_dir)
    payload = canonical_bytes(report)
    if args.json_out is not None:
        _atomic_write(args.json_out.expanduser().resolve(), payload)
    sys.stdout.buffer.write(payload)
    return report_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
