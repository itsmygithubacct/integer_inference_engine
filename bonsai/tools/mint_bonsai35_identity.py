#!/usr/bin/env python3
"""Mint the Bonsai-27B receipt identity from a completed release gate.

This tool is deliberately narrow: release artifact identities, GGUF identities,
Prism runtime identities, and fidelity policy are compiled in.  It will not mint
an identity from a smoke result, an oracle-produced result, or a weakened gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trinote.infer_int.bonsai_runtime import (  # noqa: E402
    BONSAI35_GATE_METRIC,
    BONSAI35_IDENTITY_ENGINE,
    BONSAI35_IDENTITY_FORMAT,
    BONSAI35_LABEL,
    BONSAI35_PRISM_RUNTIME_RELEASE,
    BONSAI35_RELEASE_ARTIFACT_SHA256,
    BONSAI35_RELEASE_GGUF_SHA256,
    BONSAI35_WEIGHT_PROVENANCE,
    validate_bonsai35_receipt_identity,
)


RELEASE_ARTIFACT_SHA256 = BONSAI35_RELEASE_ARTIFACT_SHA256
RELEASE_GGUF_SHA256 = BONSAI35_RELEASE_GGUF_SHA256
GATE_METRIC = BONSAI35_GATE_METRIC
IDENTITY_ENGINE = BONSAI35_IDENTITY_ENGINE
IDENTITY_FORMAT = BONSAI35_IDENTITY_FORMAT
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"quality gate {field} must be an integer >= {minimum}")
    return value


def _ratio(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"quality gate {field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"quality gate {field} must be finite and in [0, 1]")
    return number


def validate_release_gate(gate: object) -> dict[str, Any]:
    """Validate all evidence needed to mint the single pinned 27B identity."""

    if not isinstance(gate, dict):
        raise ValueError("quality gate must be a JSON object")
    exact = {
        "architecture": "qwen35",
        "artifactSha256": RELEASE_ARTIFACT_SHA256,
        "ggufSha256": RELEASE_GGUF_SHA256,
        "metric": GATE_METRIC,
        "producer": "native",
        "verdict": "PASS",
    }
    for field, expected in exact.items():
        if gate.get(field) != expected:
            raise ValueError(f"quality gate {field} is not the pinned release value")
    if gate.get("generatedOnly") is not False:
        raise ValueError("quality gate must compare native logits, not generated tokens only")
    if gate.get("top1Pass") is not True or gate.get("targetPass") is not True:
        raise ValueError("quality gate must carry both exact true pass flags")

    prism = gate.get("prism")
    if not isinstance(prism, dict):
        raise ValueError("quality gate lacks Prism evidence")
    if tuple(prism.get("runtimeRelease") or ()) != BONSAI35_PRISM_RUNTIME_RELEASE:
        raise ValueError("quality gate Prism runtime tuple is not the pinned release")
    teacher_hash = prism.get("teacherHarnessSha256")
    if not isinstance(teacher_hash, str) or _HEX64.fullmatch(teacher_hash) is None:
        raise ValueError("quality gate Prism teacher harness digest is malformed")

    count = _integer(gate.get("count"), field="count", minimum=10)
    cases = gate.get("cases")
    if not isinstance(cases, list) or len(cases) < 5:
        raise ValueError("quality gate must contain at least five cases")
    case_total = 0
    case_top1_matches = 0
    case_target_matches = 0
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"quality gate case {index} must be an object")
        current_count = _integer(
            case.get("count"), field=f"cases[{index}].count", minimum=1
        )
        current_top1 = _integer(
            case.get("top1Matches"), field=f"cases[{index}].top1Matches"
        )
        current_target = _integer(
            case.get("targetMatches"), field=f"cases[{index}].targetMatches"
        )
        if current_top1 > current_count or current_target > current_count:
            raise ValueError(f"quality gate case {index} matches exceed count")
        current_top1_ratio = _ratio(
            case.get("top1Agreement"), field=f"cases[{index}].top1Agreement"
        )
        current_target_ratio = _ratio(
            case.get("targetAgreement"), field=f"cases[{index}].targetAgreement"
        )
        if (not math.isclose(
                current_top1 / current_count,
                current_top1_ratio,
                rel_tol=0.0,
                abs_tol=1e-15,
            ) or not math.isclose(
                current_target / current_count,
                current_target_ratio,
                rel_tol=0.0,
                abs_tol=1e-15,
            )):
            raise ValueError(f"quality gate case {index} ratios are inconsistent")
        case_total += current_count
        case_top1_matches += current_top1
        case_target_matches += current_target
    if case_total != count:
        raise ValueError("quality gate case counts do not sum to count")

    value = _ratio(gate.get("value"), field="value")
    threshold = _ratio(gate.get("threshold"), field="threshold")
    target_value = _ratio(gate.get("targetAgreement"), field="targetAgreement")
    target_threshold = _ratio(gate.get("targetThreshold"), field="targetThreshold")
    if threshold < 0.80 or target_threshold < 0.50:
        raise ValueError("quality gate weakens the 0.80/0.50 release thresholds")
    if value < threshold or target_value < target_threshold:
        raise ValueError("quality gate agreement is below its declared threshold")

    # A release gate must retain the integer evidence behind its redundant
    # ratios.  Requiring both makes NaN/rounding/display-field substitution
    # fail closed in the minter and again in the shared receipt validator.
    for matches_field, ratio_field, ratio in (
        ("top1Matches", "value", value),
        ("targetMatches", "targetAgreement", target_value),
    ):
        matches = _integer(gate.get(matches_field), field=matches_field)
        if matches > count or not math.isclose(matches / count, ratio, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"quality gate {matches_field} is inconsistent with {ratio_field}")
    if (case_top1_matches != gate["top1Matches"]
            or case_target_matches != gate["targetMatches"]):
        raise ValueError("quality gate case match counts do not sum to aggregate matches")
    return gate


def build_identity(gate: dict[str, Any], *, gate_file: str, gate_hash: str) -> dict[str, Any]:
    """Build the canonical identity.  No charter/Ricardian claim is invented."""

    if Path(gate_file).name != gate_file or gate_file in {"", ".", ".."}:
        raise ValueError("quality-gate reference must be a sibling basename")
    if _HEX64.fullmatch(gate_hash) is None:
        raise ValueError("quality-gate digest is malformed")
    prism = gate["prism"]
    cases = gate["cases"]
    return {
        "format": IDENTITY_FORMAT,
        "inferenceEngine": IDENTITY_ENGINE,
        "modelHash": RELEASE_ARTIFACT_SHA256,
        "name": BONSAI35_LABEL,
        "qualityGate": {
            "caseCount": len(cases),
            "count": gate["count"],
            "gateFile": gate_file,
            "gateHash": gate_hash,
            "metric": GATE_METRIC,
            "prismRuntimeRelease": list(BONSAI35_PRISM_RUNTIME_RELEASE),
            "producer": "native",
            "targetAgreement": gate["targetAgreement"],
            "targetPass": True,
            "targetThreshold": gate["targetThreshold"],
            "teacherHarnessSha256": prism["teacherHarnessSha256"],
            "threshold": gate["threshold"],
            "top1Pass": True,
            "value": gate["value"],
            "verdict": "PASS",
        },
        "weightProvenance": dict(BONSAI35_WEIGHT_PROVENANCE),
    }


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(directory, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def mint_identity(gate_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Validate *gate_path* and atomically create a sibling identity.

    The final path is created with a hard link from a fully written, fsynced
    same-directory temporary file.  On Linux this is atomic and cannot clobber
    an existing identity.  The shared receipt validator checks both names.
    """

    gate = Path(gate_path).expanduser().absolute()
    output = Path(output_path).expanduser().absolute()
    if gate.name in {"", ".", ".."} or Path(gate.name).name != gate.name:
        raise ValueError("quality-gate path must end in a basename")
    if gate.parent.resolve() != output.parent.resolve():
        raise ValueError("quality gate and identity must be sibling files")
    if gate == output:
        raise ValueError("quality gate and identity paths must be distinct")
    if gate.is_symlink() or not gate.is_file():
        raise ValueError("quality gate must be a regular, non-symlink file")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace existing identity: {output}")

    gate_bytes = gate.read_bytes()
    try:
        parsed = json.loads(gate_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"quality gate is not valid UTF-8 JSON: {exc}") from exc
    validated = validate_release_gate(parsed)
    gate_hash = hashlib.sha256(gate_bytes).hexdigest()
    identity = build_identity(validated, gate_file=gate.name, gate_hash=gate_hash)
    identity_bytes = _canonical_bytes(identity)

    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(identity_bytes)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())

        validate_bonsai35_receipt_identity(temporary, RELEASE_ARTIFACT_SHA256)
        if hashlib.sha256(gate.read_bytes()).hexdigest() != gate_hash:
            raise ValueError("quality gate changed while the identity was being minted")

        # Atomic, no-clobber publication on the Linux deployment target.
        os.link(temporary, output, follow_symlinks=False)
        published = True
        _fsync_directory(output.parent)
        if output.read_bytes() != identity_bytes:
            raise ValueError("published identity bytes differ from the validated temporary file")
        validate_bonsai35_receipt_identity(output, RELEASE_ARTIFACT_SHA256)
        return identity
    except Exception:
        if published:
            output.unlink(missing_ok=True)
            _fsync_directory(output.parent)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True, help="completed quality-gate JSON")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="new identity JSON (must be beside --gate and must not already exist)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        identity = mint_identity(args.gate, args.out)
    except (OSError, ValueError) as exc:
        print(f"mint_bonsai35_identity: error: {exc}", file=sys.stderr)
        return 1
    print(
        f"minted {args.out} modelHash={identity['modelHash']} "
        f"gateHash={identity['qualityGate']['gateHash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
