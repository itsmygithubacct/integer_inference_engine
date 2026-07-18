from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from trinote.infer_int.bonsai_runtime import validate_bonsai35_receipt_identity


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "mint_bonsai35_identity.py"
SPEC = importlib.util.spec_from_file_location("mint_bonsai35_identity", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
mint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mint)


def _gate() -> dict:
    top1_matches = [2, 2, 2, 2, 1]
    target_matches = [2, 2, 1, 1, 1]
    return {
        "architecture": "qwen35",
        "artifactSha256": mint.RELEASE_ARTIFACT_SHA256,
        "cases": [
            {
                "prompt": f"case-{index}",
                "count": 2,
                "top1Agreement": top1_matches[index] / 2,
                "top1Matches": top1_matches[index],
                "targetAgreement": target_matches[index] / 2,
                "targetMatches": target_matches[index],
            }
            for index in range(5)
        ],
        "count": 10,
        "generatedOnly": False,
        "ggufSha256": mint.RELEASE_GGUF_SHA256,
        "metric": mint.GATE_METRIC,
        "prism": {
            "runtimeRelease": list(mint.BONSAI35_PRISM_RUNTIME_RELEASE),
            "teacherHarnessSha256": "18" * 32,
        },
        "producer": "native",
        "targetAgreement": 0.7,
        "targetMatches": 7,
        "targetPass": True,
        "targetThreshold": 0.5,
        "threshold": 0.8,
        "top1Matches": 9,
        "top1Pass": True,
        "value": 0.9,
        "verdict": "PASS",
    }


def _write_gate(path: Path, gate: dict | None = None) -> None:
    path.write_text(json.dumps(_gate() if gate is None else gate, sort_keys=True) + "\n")


def test_minter_emits_deterministic_valid_identity_without_invented_charter(tmp_path):
    gate_path = tmp_path / "bonsai27.quality-gate.json"
    first = tmp_path / "bonsai27.identity.json"
    second = tmp_path / "bonsai27-copy.identity.json"
    _write_gate(gate_path)

    calls: list[Path] = []
    actual_validator = mint.validate_bonsai35_receipt_identity

    def recording_validator(path, digest):
        calls.append(Path(path))
        return actual_validator(path, digest)

    mint.validate_bonsai35_receipt_identity = recording_validator
    try:
        identity = mint.mint_identity(gate_path, first)
        mint.mint_identity(gate_path, second)
    finally:
        mint.validate_bonsai35_receipt_identity = actual_validator

    assert first.read_bytes() == second.read_bytes()
    assert calls[0].name.startswith(f".{first.name}.") and calls[0].suffix == ".tmp"
    assert calls[1] == first.absolute()
    assert validate_bonsai35_receipt_identity(first, mint.RELEASE_ARTIFACT_SHA256) == identity
    assert identity["name"] == "ATLAS-Notarized-Bonsai-27B"
    assert identity["inferenceEngine"] == "int-ref@bonsai-qwen35"
    assert identity["modelHash"] == mint.RELEASE_ARTIFACT_SHA256
    assert identity["weightProvenance"]["kind"] == "imported-weights"
    assert identity["weightProvenance"]["ggufSha256"] == mint.RELEASE_GGUF_SHA256
    assert identity["qualityGate"]["gateFile"] == gate_path.name
    assert identity["qualityGate"]["gateHash"] == hashlib.sha256(gate_path.read_bytes()).hexdigest()
    serialized = json.dumps(identity).lower()
    assert "ricardian" not in serialized
    assert "charter" not in serialized


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda g: g.update(artifactSha256="00" * 32), "artifactSha256"),
        (lambda g: g.update(ggufSha256="00" * 32), "ggufSha256"),
        (lambda g: g.update(metric="generated-token-agreement"), "metric"),
        (lambda g: g.update(producer="oracle"), "producer"),
        (lambda g: g.update(generatedOnly=True), "native logits"),
        (lambda g: g["prism"].update(runtimeRelease=["unpinned"]), "runtime tuple"),
        (lambda g: g.update(count=9), "count"),
        (lambda g: g.update(cases=g["cases"][:4]), "five cases"),
        (lambda g: g.update(threshold=0.79), "weakens"),
        (lambda g: g.update(targetThreshold=0.49), "weakens"),
        (lambda g: g.update(top1Pass=False), "pass flags"),
        (lambda g: g.update(verdict="FAIL"), "verdict"),
        (lambda g: g["cases"][0].update(count=1), "exceed count"),
    ],
)
def test_minter_fails_closed_for_ineligible_gate(tmp_path, mutation, message):
    gate = copy.deepcopy(_gate())
    mutation(gate)
    gate_path = tmp_path / "gate.json"
    output = tmp_path / "identity.json"
    _write_gate(gate_path, gate)

    with pytest.raises(ValueError, match=message):
        mint.mint_identity(gate_path, output)
    assert not output.exists()


def test_minter_requires_sibling_gate_and_never_replaces_identity(tmp_path):
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    gate_path = gate_dir / "gate.json"
    output = tmp_path / "identity.json"
    _write_gate(gate_path)
    with pytest.raises(ValueError, match="sibling"):
        mint.mint_identity(gate_path, output)

    sibling_gate = tmp_path / "gate.json"
    _write_gate(sibling_gate)
    output.write_text("user-owned\n")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        mint.mint_identity(sibling_gate, output)
    assert output.read_text() == "user-owned\n"


def test_failed_temporary_validation_leaves_no_published_identity(tmp_path, monkeypatch):
    gate_path = tmp_path / "gate.json"
    output = tmp_path / "identity.json"
    _write_gate(gate_path)

    def reject(_path, _digest):
        raise ValueError("validator rejected temporary identity")

    monkeypatch.setattr(mint, "validate_bonsai35_receipt_identity", reject)
    with pytest.raises(ValueError, match="temporary identity"):
        mint.mint_identity(gate_path, output)
    assert not output.exists()
    assert list(tmp_path.glob(".identity.json.*.tmp")) == []
