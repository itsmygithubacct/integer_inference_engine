#!/usr/bin/env python3
"""Benchmark exact Bonsai-27B receipt verification and emit a routing policy.

Each (engine, algorithm, thread-count) cell runs in a fresh subprocess so the
OpenMP/BLAS thread settings take effect before native libraries initialize.
The default workload is the checked-in 19-input/20-output prefix of the release
golden turn; pass ``--output-counts`` to measure additional committed lengths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path


BENCH_SCHEMA = "receipt-verifier-benchmark/v1"
POLICY_SCHEMA = "receipt-verifier-policy/v1"
VARIANTS = {
    "oracle-cached": ("oracle", "cached-replay"),
    "oracle-teacher": ("oracle", "teacher-forced"),
    "native-teacher": ("native", "teacher-forced"),
    "native-cached": ("native", "cached-replay"),
}
EXPECTED_RESULT_STRATEGY = {
    "teacher-forced": "resample-full",
    "cached-replay": "resample-cached-replay",
}
DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/bonsai35_19x64_golden.json"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha(values: list[int]) -> str:
    return hashlib.sha256(b"".join(struct.pack("<q", int(value)) for value in values)).hexdigest()


def _load_fixture(path: Path) -> dict:
    fixture = json.loads(path.read_text("utf-8"))
    if fixture.get("schema") != "trinote-bonsai35-golden/v1":
        raise ValueError("unsupported golden fixture schema")
    input_ids = fixture.get("inputIds")
    output_ids = fixture.get("outputIds")
    if not isinstance(input_ids, list) or len(input_ids) != 19:
        raise ValueError("golden fixture must contain exactly 19 input IDs")
    if not isinstance(output_ids, list) or len(output_ids) != 64:
        raise ValueError("golden fixture must contain exactly 64 output IDs")
    commitments = fixture.get("commitments") or {}
    if _ids_sha(input_ids) != commitments.get("inputIdsInt64LeSha256"):
        raise ValueError("golden input IDs do not match their commitment")
    if _ids_sha(output_ids) != commitments.get("outputIdsInt64LeSha256"):
        raise ValueError("golden output IDs do not match their commitment")
    return fixture


def _configure_threads(count: int) -> None:
    value = str(int(count))
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        "TRINOTE_ORACLE_Q1_THREADS",
    ):
        os.environ[name] = value
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")


def _worker(args) -> int:
    _configure_threads(args.worker_threads)
    worker_cpu_started = time.process_time()
    # Deferred imports are load-bearing: native runtimes must see the thread
    # environment before NumPy/OpenMP shared libraries initialize.
    from trinote.infer_int.artifact_io_bonsai import load_artifact_bonsai
    from trinote.infer_int.reference_bonsai import oracle_q1_worker_count
    from trinote.infer_int.reference_bonsai35 import BonsaiQwen35ReferenceModel
    from trinote.infer_int.sampler import resolve_sampler
    from trinote.infer_int.verify import verify_resample

    fixture = _load_fixture(args.fixture)
    expected_artifact = fixture["release"]["artifactSha256"]
    artifact_sha = _sha_file(args.artifact)
    if not args.allow_artifact_digest_mismatch and artifact_sha != expected_artifact:
        raise RuntimeError(
            f"release artifact mismatch: expected {expected_artifact}, got {artifact_sha}"
        )
    engine, strategy = VARIANTS[args.worker_variant]
    load_started = time.monotonic()
    artifact, _info = load_artifact_bonsai(args.artifact)
    model = BonsaiQwen35ReferenceModel(artifact)
    actual_engine = "oracle"
    if engine == "native":
        if not model.enable_native():
            raise RuntimeError("native packed-Q1 verifier was requested but could not be enabled")
        if getattr(model, "_model_executor", None) is None:
            raise RuntimeError(
                "native packed-Q1 verifier was requested but its resident model executor "
                "could not be created"
            )
        actual_engine = "native"
    model.receipt_verify_strategy = strategy
    load_seconds = time.monotonic() - load_started
    sampler = resolve_sampler(
        "bonsai27-rec", seed=0, rep_penalty=0, no_repeat_ngram=4
    )
    input_ids = list(fixture["inputIds"])
    output_ids = list(fixture["outputIds"][: args.worker_output_count])
    expected_result_strategy = EXPECTED_RESULT_STRATEGY[strategy]
    timings = []
    result = None
    for _ in range(args.repeats):
        started = time.monotonic()
        result = verify_resample(model, input_ids, output_ids, sampler_cfg=sampler)
        timings.append(time.monotonic() - started)
        if (
            not result.get("ok")
            or int(result.get("checked", -1)) != len(output_ids)
            or result.get("strategy") != expected_result_strategy
        ):
            break
    replay_verified = bool(
        result
        and result.get("ok")
        and int(result.get("checked", -1)) == len(output_ids)
        and result.get("strategy") == expected_result_strategy
    )
    native_stats = model.native_executor_stats() if actual_engine == "native" else None
    actual_threads = (
        int(native_stats.get("last_team_size", 0))
        if native_stats is not None
        else int(oracle_q1_worker_count())
    )
    threads_matched = actual_threads > 0 and actual_threads == args.worker_threads
    verified = replay_verified and threads_matched
    payload = {
        "variant": args.worker_variant,
        "requestedEngine": engine,
        "actualEngine": actual_engine,
        "requestedStrategy": strategy,
        "actualStrategy": (result or {}).get("strategy"),
        "strategyMatched": bool(
            result and result.get("strategy") == expected_result_strategy
        ),
        # ``threads`` remains the policy-selection field, but now records the
        # effective count rather than merely echoing the request.
        "threads": actual_threads,
        "requestedThreads": args.worker_threads,
        "actualThreads": actual_threads,
        "threadsMatched": threads_matched,
        "inputTokens": len(input_ids),
        "outputTokens": len(output_ids),
        "artifactSha256": artifact_sha,
        "artifactLoadedSha256": artifact_sha,
        "loadSeconds": load_seconds,
        "verifySeconds": timings,
        "medianVerifySeconds": statistics.median(timings),
        "verified": verified,
        "checked": (result or {}).get("checked"),
        "processCpuSeconds": time.process_time() - worker_cpu_started,
        "maxRssKiB": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["verified"] else 1


def _positive_csv(value: str, *, maximum: int | None = None) -> list[int]:
    parsed = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    if maximum is not None and any(item > maximum for item in parsed):
        raise argparse.ArgumentTypeError(f"values must be <= {maximum}")
    return parsed


def _write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _policy(results: list[dict], *, artifact_sha: str, engine_filter: str | None = None) -> dict:
    successful = [
        item for item in results
        if item.get("verified")
        and item.get("threadsMatched") is True
        and type(item.get("requestedThreads")) is int
        and type(item.get("actualThreads")) is int
        and item["requestedThreads"] == item["actualThreads"] == item.get("threads")
    ]
    if engine_filter is not None:
        successful = [item for item in successful if item.get("requestedEngine") == engine_filter]
    if not successful:
        raise RuntimeError("no requested verifier benchmark cell passed exact replay")
    output_counts = sorted({int(item["outputTokens"]) for item in successful})
    # Pick one process-wide thread count first. OpenMP runtimes cannot be
    # safely retuned between bundles after model/native-library load.
    thread_scores = []
    for threads in sorted({int(item["threads"]) for item in successful}):
        cells = [item for item in successful if int(item["threads"]) == threads]
        if {int(item["outputTokens"]) for item in cells} != set(output_counts):
            continue
        score = sum(
            min(float(item["medianVerifySeconds"]) for item in cells
                if int(item["outputTokens"]) == count)
            for count in output_counts
        )
        thread_scores.append((score, threads))
    if not thread_scores:
        raise RuntimeError("no single thread count passed every measured output length")
    _score, selected_threads = min(thread_scores)
    successful = [item for item in successful if int(item["threads"]) == selected_threads]
    rules = []
    winners = []
    for count in output_counts:
        candidates = [item for item in successful if int(item["outputTokens"]) == count]
        winner = min(candidates, key=lambda item: float(item["medianVerifySeconds"]))
        winners.append(winner)
        rules.append({
            "minInputTokens": int(winner["inputTokens"]),
            "maxInputTokens": int(winner["inputTokens"]),
            "minOutputTokens": count,
            "maxOutputTokens": count,
            "engine": winner["requestedEngine"],
            "strategy": winner["requestedStrategy"],
            "measuredThreads": int(winner["threads"]),
            "medianVerifySeconds": float(winner["medianVerifySeconds"]),
        })
    default_winner = winners[-1]
    evidence_digest = hashlib.sha256(
        json.dumps(results, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    policy = {
        "schema": POLICY_SCHEMA,
        "artifactSha256": artifact_sha,
        "evidenceSha256": evidence_digest,
        "threads": selected_threads,
        "requireMeasuredPoint": True,
        "measuredPoints": [
            {"inputTokens": int(item["inputTokens"]), "outputTokens": int(item["outputTokens"])}
            for item in winners
        ],
        "rules": rules,
        "default": {
            "engine": default_winner["requestedEngine"],
            "strategy": default_winner["requestedStrategy"],
        },
    }
    if engine_filter is not None:
        policy["engineConstraint"] = engine_filter
    return policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--threads", type=lambda value: _positive_csv(value), default=[1, 2, 4, 8, 16])
    parser.add_argument("--output-counts", type=lambda value: _positive_csv(value, maximum=64), default=[20])
    parser.add_argument("--variants", choices=sorted(VARIANTS), nargs="+", default=sorted(VARIANTS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cell-timeout", type=int, default=1800,
                        help="maximum seconds for one fresh-process matrix cell (default: 1800)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy-out", type=Path, default=None)
    parser.add_argument("--oracle-policy-out", type=Path, default=None,
                        help="also write an oracle-only policy suitable for fresh-oracle receipt issuance")
    parser.add_argument("--allow-artifact-digest-mismatch", action="store_true",
                        help="development-only: benchmark a non-release artifact while retaining exact token gates")
    parser.add_argument("--worker-variant", choices=sorted(VARIANTS), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-threads", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-count", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.repeats <= 0:
        parser.error("--repeats must be > 0")
    if args.cell_timeout <= 0:
        parser.error("--cell-timeout must be > 0")
    if args.worker_variant:
        return _worker(args)

    fixture = _load_fixture(args.fixture)
    artifact_sha = _sha_file(args.artifact)
    results = []
    started = time.time()
    for output_count in args.output_counts:
        for thread_count in args.threads:
            for variant in args.variants:
                command = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--artifact", str(args.artifact),
                    "--fixture", str(args.fixture),
                    "--out", str(args.out),
                    "--repeats", str(args.repeats),
                    "--worker-variant", variant,
                    "--worker-threads", str(thread_count),
                    "--worker-output-count", str(output_count),
                ]
                if args.allow_artifact_digest_mismatch:
                    command.append("--allow-artifact-digest-mismatch")
                try:
                    completed = subprocess.run(
                        command, text=True, capture_output=True, timeout=args.cell_timeout
                    )
                except subprocess.TimeoutExpired as exc:
                    cell = {
                        "variant": variant,
                        "threads": None,
                        "requestedThreads": thread_count,
                        "actualThreads": None,
                        "threadsMatched": False,
                        "inputTokens": len(fixture["inputIds"]),
                        "outputTokens": output_count,
                        "verified": False,
                        "returnCode": None,
                        "error": f"worker exceeded {args.cell_timeout}s timeout",
                    }
                    if exc.stderr:
                        cell["stderr"] = str(exc.stderr)
                    results.append(cell)
                    print(json.dumps(cell, sort_keys=True), flush=True)
                    continue
                if completed.stdout.strip():
                    try:
                        cell = json.loads(completed.stdout.strip().splitlines()[-1])
                    except json.JSONDecodeError:
                        cell = {"verified": False, "error": completed.stdout.strip()}
                else:
                    cell = {"verified": False, "error": completed.stderr.strip() or "worker failed"}
                cell["returnCode"] = completed.returncode
                if completed.stderr.strip():
                    cell["stderr"] = completed.stderr.strip()
                results.append(cell)
                print(json.dumps(cell, sort_keys=True), flush=True)

    report = {
        "schema": BENCH_SCHEMA,
        "status": "pass" if results and all(item.get("verified") for item in results) else "fail",
        "startedUnixSeconds": started,
        "finishedUnixSeconds": time.time(),
        "fixture": str(args.fixture),
        "artifactSha256": artifact_sha,
        "expectedArtifactSha256": fixture["release"]["artifactSha256"],
        "results": results,
    }
    _write_atomic(args.out, report)
    matrix_passed = report["status"] == "pass"
    if args.policy_out and matrix_passed:
        _write_atomic(args.policy_out, _policy(results, artifact_sha=artifact_sha))
    if args.oracle_policy_out and matrix_passed:
        _write_atomic(
            args.oracle_policy_out,
            _policy(results, artifact_sha=artifact_sha, engine_filter="oracle"),
        )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
