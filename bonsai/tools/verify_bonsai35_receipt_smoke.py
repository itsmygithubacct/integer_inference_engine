#!/usr/bin/env python3
"""Verify a saved Bonsai-27B receipt and a deliberate output corruption.

This tool is intentionally read-only with respect to receipt state: it does
not issue, sign, ledger, bundle, or broadcast anything.  It loads one raw
``{receipt, preimage}`` JSON saved by ``run_bonsai_cli --save-bundle``, binds
the release artifact and accepted identity, constructs a fresh pure-NumPy
Qwen3.5 oracle, verifies the real receipt, then mutates one saved output token
and requires the corrupted copy to fail both commitments and re-execution.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trinote.infer_int.artifact_io_bonsai import load_artifact_bonsai  # noqa: E402
from trinote.infer_int.bonsai_runtime import validate_bonsai35_receipt_identity  # noqa: E402
from trinote.infer_int.reference_bonsai35 import BonsaiQwen35ReferenceModel  # noqa: E402
from trinote.receipts.verify import verify_receipt  # noqa: E402


FORMAT = "trinote-bonsai35-receipt-smoke-verification/1"
RELEASE_ARTIFACT_SHA256 = "7eab414ceff3fff1489053d415d0c6adb1e646e552d091cc1a898d0456adf3fb"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
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
        temporary.unlink(missing_ok=True)


def corrupted_output_copy(bundle: dict[str, Any], vocab: int) -> tuple[dict[str, Any], dict[str, int]]:
    corrupted = copy.deepcopy(bundle)
    try:
        output_ids = corrupted["preimage"]["outputIds"]
    except (KeyError, TypeError) as exc:
        raise ValueError("saved receipt has no preimage.outputIds") from exc
    if not isinstance(output_ids, list) or not output_ids:
        raise ValueError("saved receipt has no output token to corrupt")
    original = output_ids[0]
    if type(original) is not int or original < 0 or original >= int(vocab):
        raise ValueError("saved receipt first output token is outside the release vocabulary")
    changed = (original + 1) % int(vocab)
    output_ids[0] = changed
    return corrupted, {"index": 0, "original": original, "corrupted": changed}


def evaluate_receipt_smoke(bundle: dict[str, Any], *, oracle, model_digest: str) -> dict[str, Any]:
    """Return complete correct/corrupt verification and hard acceptance gates."""

    if getattr(oracle, "_native", False) or getattr(oracle, "_model_executor", None) is not None:
        raise ValueError("receipt smoke verifier must be a pure CPU oracle")
    if str(oracle.cfg.get("architecture", "")) != "qwen35":
        raise ValueError("receipt smoke verifier must use Qwen3.5")
    correct = verify_receipt(bundle, model=oracle, model_digest=model_digest)
    corrupted, mutation = corrupted_output_copy(bundle, int(oracle.cfg["vocab"]))
    corrupt = verify_receipt(corrupted, model=oracle, model_digest=model_digest)
    serialized_bundle = json.dumps(bundle, sort_keys=True).lower()
    cache_markers = ("prompt-cache", "prompt_cache", "promptcache")
    acceptance = {
        "fresh_oracle_is_pure": True,
        "correct_receipt_ok": correct.get("ok") is True,
        "correct_structural_ok": correct.get("structuralOk") is True,
        "correct_signature_ok": correct.get("signatureOk") is True,
        "correct_reexecution_ok": correct.get("reexecOk") is True,
        "correct_artifact_binding_ok": correct.get("artifactBoundOk") is True,
        "committed_sampler_present": correct.get("committedSamplerPresent") is True,
        "preimage_sampler_matches_commitment": correct.get("preimageSamplerMatch") is True,
        "corrupted_output_rejected": corrupt.get("ok") is False,
        "corrupted_output_commitment_rejected": corrupt.get("commitMatch") is False,
        "corrupted_output_reexecution_rejected": corrupt.get("reexecOk") is False,
        "runtime_prompt_cache_not_serialized": not any(
            marker in serialized_bundle for marker in cache_markers
        ),
    }
    return {
        "status": "pass" if all(acceptance.values()) else "fail",
        "verificationMode": "fresh-canonical-numpy-oracle",
        "receiptHash": bundle.get("receipt", {}).get("receiptHash"),
        "inputIds": bundle.get("preimage", {}).get("inputIds"),
        "outputIds": bundle.get("preimage", {}).get("outputIds"),
        "sampler": bundle.get("receipt", {}).get("trace", {}).get("sampler"),
        "mutation": mutation,
        "acceptance": acceptance,
        "correctVerification": correct,
        "corruptedVerification": corrupt,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--bundle", type=Path, required=True,
                        help="raw receipt-<hash>.json emitted by --save-bundle")
    parser.add_argument("--artifact", type=Path, required=True,
                        help="release Bonsai-27B integer artifact")
    parser.add_argument("--identity", type=Path, required=True,
                        help="accepted sibling-gate-bound Bonsai-27B identity")
    parser.add_argument("--json-out", type=Path,
                        help="atomically write the exact canonical stdout report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bundle_path = args.bundle.expanduser().resolve()
    artifact_path = args.artifact.expanduser().resolve()
    identity_path = args.identity.expanduser().resolve()
    bundle_bytes = bundle_path.read_bytes()
    bundle = json.loads(bundle_bytes)
    artifact, info = load_artifact_bonsai(artifact_path)
    digest = str(info["digest"])
    if digest != RELEASE_ARTIFACT_SHA256:
        raise ValueError(
            f"receipt smoke requires release artifact {RELEASE_ARTIFACT_SHA256}, got {digest}"
        )
    validate_bonsai35_receipt_identity(identity_path, digest)
    oracle = BonsaiQwen35ReferenceModel(artifact)
    result = evaluate_receipt_smoke(bundle, oracle=oracle, model_digest=digest)
    report = {
        "format": FORMAT,
        "artifact": {"path": str(artifact_path), "sha256": digest},
        "identity": {"path": str(identity_path), "sha256": sha256_file(identity_path)},
        "bundle": {"path": str(bundle_path), "sha256": hashlib.sha256(bundle_bytes).hexdigest()},
        **result,
    }
    payload = canonical_json_bytes(report)
    if args.json_out is not None:
        atomic_write(args.json_out, payload)
    sys.stdout.buffer.write(payload)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
