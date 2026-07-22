#!/usr/bin/env python3
"""Fail-closed resident-CUDA verification of the Bonsai-27B 19x64 golden turn."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
DEFAULT_FIXTURE = ROOT / "tests/fixtures/bonsai35_19x64_golden.json"
SCHEMA = "trinote-bonsai35-gpu-golden/v1"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha(values: list[int]) -> str:
    raw = b"".join(struct.pack("<q", int(value)) for value in values)
    return hashlib.sha256(raw).hexdigest()


def atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)

    from trinote.infer_int.artifact_io_bonsai import load_artifact_bonsai
    from trinote.infer_int.gguf_tokenizer_v2 import load_gguf_tokens, token_bytes
    from trinote.infer_int.gpu_bonsai35 import Bonsai35GpuExecutor
    from trinote.infer_int.sampler import resolve_sampler

    fixture = json.loads(args.fixture.read_text("utf-8"))
    if fixture.get("schema") != "trinote-bonsai35-golden/v1":
        raise SystemExit("unsupported golden fixture schema")
    input_ids = [int(value) for value in fixture.get("inputIds", [])]
    expected_ids = [int(value) for value in fixture.get("outputIds", [])]
    if len(input_ids) != 19 or len(expected_ids) != 64:
        raise SystemExit("golden fixture must contain exactly 19 input and 64 output IDs")
    expected = fixture["commitments"]
    release = fixture["release"]
    artifact_sha = sha_file(args.artifact)
    gguf_sha = sha_file(args.gguf)
    preflight = {
        "inputCommitment": ids_sha(input_ids) == expected["inputIdsInt64LeSha256"],
        "outputCommitment": ids_sha(expected_ids) == expected["outputIdsInt64LeSha256"],
        "artifactDigest": artifact_sha == release["artifactSha256"],
        "ggufDigest": gguf_sha == release["ggufSha256"],
    }
    if not all(preflight.values()):
        result = {"schema": SCHEMA, "status": "fail", "stage": "preflight", "gates": preflight}
        atomic_write(args.json_out, result)
        print(json.dumps(result, sort_keys=True))
        return 2

    load_started = time.monotonic()
    artifact, _info = load_artifact_bonsai(args.artifact)
    executor, report = Bonsai35GpuExecutor.try_create_reported(artifact)
    if executor is None:
        result = {
            "schema": SCHEMA,
            "status": "fail",
            "stage": "gpu-residency",
            "memoryProof": report.as_dict(),
        }
        atomic_write(args.json_out, result)
        print(json.dumps(result, sort_keys=True))
        return 3
    loaded = time.monotonic()
    output_ids: list[int] = []
    complete = False
    stats: dict = {}
    timing: dict = {}
    graph: dict = {}
    execution_error: str | None = None
    cleanup_error: str | None = None
    generated = loaded
    try:
        sampler = resolve_sampler(
            "bonsai27-rec", seed=0, rep_penalty=0, no_repeat_ngram=4
        )
        output_ids, complete = executor.generate_device(
            input_ids, len(expected_ids), sampler, eos=None,
        )
        generated = time.monotonic()
        stats = executor.stats()
        timing = executor.timing_stats()
        graph = executor.graph_metadata()
    except Exception as exc:
        generated = time.monotonic()
        execution_error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            executor.close()
        except Exception as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
    cleanup = {
        "gpuClosed": bool(getattr(executor, "closed", False)),
    }
    if cleanup_error is not None:
        cleanup["error"] = cleanup_error
    if execution_error is not None or cleanup_error is not None:
        errors = [value for value in (execution_error, cleanup_error) if value is not None]
        result = {
            "schema": SCHEMA,
            "status": "fail",
            "stage": "gpu-execution" if execution_error is not None else "gpu-cleanup",
            "error": "; ".join(errors),
            "model": {
                "artifactSha256": artifact_sha,
                "artifactLoadedSha256": artifact_sha,
                "ggufSha256": gguf_sha,
            },
            "request": {"inputTokens": len(input_ids), "outputTokens": len(expected_ids)},
            "output": {
                "tokensProduced": len(output_ids),
                "idsSha256": ids_sha(output_ids),
            },
            "engine": {
                "memoryProof": report.as_dict(),
                "stats": stats,
                "timing": timing,
                "graph": graph,
            },
            "gates": {**preflight, "gpuComplete": False},
            "timingSeconds": {
                "loadAndResidency": loaded - load_started,
                "prefillAndDecode": generated - loaded,
            },
            "cleanup": cleanup,
        }
        atomic_write(args.json_out, result)
        print(json.dumps(result, sort_keys=True))
        return 4

    vocabulary = load_gguf_tokens(args.gguf)
    visible = b"".join(token_bytes(token, vocabulary) for token in output_ids)
    consumed_positions = len(input_ids) + len(expected_ids) - 1
    gates = {
        **preflight,
        "gpuComplete": bool(complete),
        "outputIdsExact": output_ids == expected_ids,
        "outputCommitmentExact": ids_sha(output_ids) == expected["outputIdsInt64LeSha256"],
        "visibleBytesExact": hashlib.sha256(visible).hexdigest() == expected["visibleBytesSha256"],
        "oneGraphPerConsumedPosition": (
            int(stats.get("graph_launches", -1))
            == int(stats.get("position", -2))
            == consumed_positions
        ),
        "contextHealthy": not bool(stats.get("poisoned", True)),
        "deviceLogitsInputMode": stats.get("input_mode") == "token_id_device_logits",
        "singleNativePrefill": (
            int(stats.get("prefill_calls", -1)) == 1
            and int(stats.get("prefill_tokens", -1)) == len(input_ids)
        ),
        "deviceOnlyConsumedPositions": (
            int(stats.get("device_only_decode_submissions", -1))
            == consumed_positions
        ),
        "deviceSamplerCallsExact": (
            int(stats.get("device_sampler_prepare_calls", -1)) == len(expected_ids)
            and int(timing.get("device_sampling_calls", -1)) == len(expected_ids)
        ),
        "zeroFullLogitsD2H": (
            int(timing.get("prefill_final_logits_d2h_bytes", -1)) == 0
            and int(timing.get("decode_full_logits_d2h_bytes", -1)) == 0
        ),
        "samplerTrafficBelowOneLogitsRow": (
            0 <= int(stats.get("device_sampler_host_bytes", -1))
            < int(artifact["config"]["vocab"]) * 8
        ),
    }
    result = {
        "schema": SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "model": {
            "artifactSha256": artifact_sha,
            "artifactLoadedSha256": artifact_sha,
            "ggufSha256": gguf_sha,
        },
        "request": {"inputTokens": len(input_ids), "outputTokens": len(expected_ids)},
        "output": {
            "idsSha256": ids_sha(output_ids),
            "visibleBytesSha256": hashlib.sha256(visible).hexdigest(),
        },
        "engine": {
            "memoryProof": report.as_dict(),
            "stats": stats,
            "timing": timing,
            "graph": graph,
        },
        "gates": gates,
        "timingSeconds": {
            "loadAndResidency": loaded - load_started,
            "prefillAndDecode": generated - loaded,
        },
        "cleanup": cleanup,
    }
    atomic_write(args.json_out, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
