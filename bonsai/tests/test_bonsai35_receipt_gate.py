from __future__ import annotations

import copy
import hashlib
import json

import pytest

from trinote.infer_int.artifact_io_bonsai import (
    _LOADED_ARTIFACT_SHA256,
    _LOADED_CONFIG_SHA256,
    _config_sha256,
    load_artifact_bonsai,
    save_artifact_bonsai,
)
from trinote.infer_int.bonsai_runtime import (
    BONSAI35_GATE_METRIC,
    BONSAI35_IDENTITY_ENGINE,
    BONSAI35_IDENTITY_FORMAT,
    BONSAI35_LABEL,
    BONSAI35_PRISM_RUNTIME_RELEASE,
    BONSAI35_RELEASE_ARTIFACT_SHA256,
    BONSAI35_WEIGHT_PROVENANCE,
    emit_and_verify_bonsai_receipt,
    validate_bonsai35_receipt_identity,
)
from trinote.infer_int.prompt_cache_bonsai35 import (
    build_prompt_state,
    generate_from_prompt_state,
)
from trinote.infer_int.reference_bonsai35 import (
    BonsaiQwen35ReferenceModel,
    random_bonsai35_artifact,
)
from trinote.infer_int.sampler import SamplerConfig
from trinote.receipts import build_receipt, keygen, verify_receipt


def _keys():
    return (
        keygen(label="bonsai35", secret_hex="31" * 32),
        keygen(label="counterparty", secret_hex="32" * 32),
    )


def _write_gated_identity(
    tmp_path, digest: str = BONSAI35_RELEASE_ARTIFACT_SHA256
):
    cases = [
        {
            "prompt": str(index),
            "count": 2,
            "top1Agreement": 1.0,
            "top1Matches": 2,
            "targetAgreement": 1.0,
            "targetMatches": 2,
        }
        for index in range(5)
    ]
    gate = {
        "architecture": "qwen35",
        "artifactSha256": digest,
        "cases": cases,
        "count": 10,
        "generatedOnly": False,
        "ggufSha256": BONSAI35_WEIGHT_PROVENANCE["ggufSha256"],
        "metric": BONSAI35_GATE_METRIC,
        "prism": {
            "runtimeRelease": list(BONSAI35_PRISM_RUNTIME_RELEASE),
            "teacherHarnessSha256": "18" * 32,
        },
        "producer": "native",
        "targetAgreement": 1.0,
        "targetMatches": 10,
        "targetThreshold": 0.5,
        "targetPass": True,
        "threshold": 0.8,
        "top1Matches": 10,
        "top1Pass": True,
        "value": 1.0,
        "verdict": "PASS",
    }
    gate_path = tmp_path / "bonsai27.quality-gate.json"
    gate_path.write_text(json.dumps(gate, sort_keys=True) + "\n")
    identity = {
        "format": BONSAI35_IDENTITY_FORMAT,
        "name": BONSAI35_LABEL,
        "inferenceEngine": BONSAI35_IDENTITY_ENGINE,
        "modelHash": digest,
        "qualityGate": {
            "caseCount": len(cases),
            "count": gate["count"],
            "gateFile": gate_path.name,
            "gateHash": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
            "metric": BONSAI35_GATE_METRIC,
            "prismRuntimeRelease": list(BONSAI35_PRISM_RUNTIME_RELEASE),
            "producer": "native",
            "targetAgreement": gate["targetAgreement"],
            "targetPass": True,
            "targetThreshold": gate["targetThreshold"],
            "teacherHarnessSha256": gate["prism"]["teacherHarnessSha256"],
            "threshold": gate["threshold"],
            "top1Pass": True,
            "value": gate["value"],
            "verdict": "PASS",
        },
        "weightProvenance": dict(BONSAI35_WEIGHT_PROVENANCE),
    }
    identity_path = tmp_path / "bonsai27.identity.json"
    identity_path.write_text(json.dumps(identity, sort_keys=True) + "\n")
    return identity_path, gate_path


def _model_pair(seed: int, digest: str = BONSAI35_RELEASE_ARTIFACT_SHA256):
    producer_artifact = random_bonsai35_artifact(seed=seed)
    verifier_artifact = copy.deepcopy(producer_artifact)
    for artifact in (producer_artifact, verifier_artifact):
        artifact[_LOADED_ARTIFACT_SHA256] = digest
        artifact[_LOADED_CONFIG_SHA256] = _config_sha256(artifact["config"])
    return (
        BonsaiQwen35ReferenceModel(producer_artifact),
        BonsaiQwen35ReferenceModel(verifier_artifact),
    )


def test_qwen35_receipt_identity_requires_hash_bound_gate_evidence(tmp_path):
    digest = BONSAI35_RELEASE_ARTIFACT_SHA256
    identity_path, gate_path = _write_gated_identity(tmp_path, digest)
    assert validate_bonsai35_receipt_identity(identity_path, digest)["modelHash"] == digest

    gate_path.write_text(gate_path.read_text() + " ")
    with pytest.raises(ValueError, match="file digest"):
        validate_bonsai35_receipt_identity(identity_path, digest)


def test_qwen35_receipt_identity_rejects_weakened_gate_thresholds(tmp_path):
    digest = BONSAI35_RELEASE_ARTIFACT_SHA256
    identity_path, gate_path = _write_gated_identity(tmp_path, digest)
    gate = json.loads(gate_path.read_text())
    gate["threshold"] = 0.0
    gate_path.write_text(json.dumps(gate, sort_keys=True) + "\n")
    identity = json.loads(identity_path.read_text())
    identity["qualityGate"]["gateHash"] = hashlib.sha256(
        gate_path.read_bytes()
    ).hexdigest()
    identity_path.write_text(json.dumps(identity, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="weakened"):
        validate_bonsai35_receipt_identity(identity_path, digest)


def test_qwen35_receipt_identity_rejects_a_bare_user_supplied_model_hash(tmp_path):
    path = tmp_path / "not-minted.identity.json"
    path.write_text(json.dumps({"modelHash": BONSAI35_RELEASE_ARTIFACT_SHA256}))
    with pytest.raises(ValueError, match="format"):
        validate_bonsai35_receipt_identity(path, BONSAI35_RELEASE_ARTIFACT_SHA256)


def test_qwen35_prompt_cache_is_runtime_only_and_fresh_oracle_reexecutes(tmp_path):
    producer, verifier = _model_pair(54)
    digest = BONSAI35_RELEASE_ARTIFACT_SHA256
    prompt = [2, 8, 5]
    state = build_prompt_state(producer, prompt, digest)
    output_ids = generate_from_prompt_state(
        producer,
        state,
        3,
        lambda row, _position, _history: int(row.argmax()),
        keep_reusable=False,
    )
    identity_path, _gate_path = _write_gated_identity(tmp_path, digest)
    bundle, verification, emission = emit_and_verify_bonsai_receipt(
        producer,
        input_ids=prompt,
        output_ids=output_ids,
        model_digest=digest,
        sampler=SamplerConfig(mode="greedy"),
        verifier_model=verifier,
        verifier_mode="fresh-oracle",
        identity_path=identity_path,
        ledger_path=tmp_path / "ledger.jsonl",
        broadcast_to_log=False,
    )
    assert verification["ok"] and verification["reexecOk"]
    assert verification["verificationMode"] == "fresh-oracle"
    assert emission
    assert bundle["receipt"]["modelHash"] == digest
    assert "prompt-cache" not in json.dumps(bundle).lower()


def test_qwen35_receipt_reexecutes_and_uses_distinct_label():
    model = BonsaiQwen35ReferenceModel(random_bonsai35_artifact(seed=51))
    input_ids = [2, 5, 7]
    output_ids = model.generate_greedy_tokens_cached(input_ids, 3)
    digest = "35" * 32
    mk, ck = _keys()
    bundle = build_receipt(
        model_hash=digest,
        input_ids=input_ids,
        output_ids=output_ids,
        sampler=SamplerConfig(mode="greedy"),
        model_key=mk,
        counterparty_key=ck,
        model_label="ATLAS-Notarized-Bonsai-27B",
        artifact_digest=digest,
    )
    verified = verify_receipt(
        bundle,
        model=model,
        model_digest=digest,
        model_key=mk,
        counterparty_key=ck,
    )
    assert verified["ok"] and verified["reexecOk"]
    assert bundle["preimage"]["modelLabel"] == "ATLAS-Notarized-Bonsai-27B"


def test_qwen35_cryptographically_valid_wrong_output_fails_reexecution():
    model = BonsaiQwen35ReferenceModel(random_bonsai35_artifact(seed=52))
    input_ids = [1, 4, 8]
    correct = model.generate_greedy_tokens_cached(input_ids, 2)
    wrong = correct.copy()
    wrong[1] = (wrong[1] + 1) % int(model.cfg["vocab"])
    digest = "36" * 32
    mk, ck = _keys()
    forged = build_receipt(
        model_hash=digest,
        input_ids=input_ids,
        output_ids=wrong,
        sampler=SamplerConfig(mode="greedy"),
        model_key=mk,
        counterparty_key=ck,
        model_label="ATLAS-Notarized-Bonsai-27B",
        artifact_digest=digest,
    )
    verified = verify_receipt(
        forged,
        model=model,
        model_digest=digest,
        model_key=mk,
        counterparty_key=ck,
    )
    assert verified["structuralOk"]
    assert verified["signatureOk"]
    assert not verified["reexecOk"]
    assert not verified["ok"]


def test_qwen35_runtime_identity_binding_fails_closed(tmp_path):
    artifact_path = tmp_path / "qwen35.safetensors"
    save_artifact_bonsai(
        random_bonsai35_artifact(seed=53),
        artifact_path,
        provenance={"kind": "test"},
    )
    artifact, info = load_artifact_bonsai(artifact_path)
    model = BonsaiQwen35ReferenceModel(artifact)
    verifier = BonsaiQwen35ReferenceModel(artifact)
    input_ids = [3, 6]
    output_ids = model.generate_greedy_tokens_cached(input_ids, 1)
    identity, _gate_path = _write_gated_identity(tmp_path, "00" * 32)
    with pytest.raises(ValueError, match="modelHash"):
        emit_and_verify_bonsai_receipt(
            model,
            input_ids=input_ids,
            output_ids=output_ids,
            model_digest=info["digest"],
            sampler=SamplerConfig(mode="greedy"),
            verifier_model=verifier,
            verifier_mode="fresh-oracle",
            identity_path=identity,
            ledger_path=tmp_path / "ledger.jsonl",
            broadcast_to_log=False,
        )


def test_qwen35_shared_receipt_api_cannot_bypass_identity_or_fresh_oracle(tmp_path):
    producer, verifier = _model_pair(55)
    prompt = [1, 2]
    output_ids = producer.generate_greedy_tokens_cached(prompt, 1)
    digest = BONSAI35_RELEASE_ARTIFACT_SHA256
    identity, _gate_path = _write_gated_identity(tmp_path, digest)

    with pytest.raises(ValueError, match="fresh-oracle"):
        emit_and_verify_bonsai_receipt(
            producer,
            input_ids=prompt,
            output_ids=output_ids,
            model_digest=digest,
            sampler=SamplerConfig(mode="greedy"),
            identity_path=identity,
            ledger_path=tmp_path / "ledger.jsonl",
            broadcast_to_log=False,
        )
    with pytest.raises(ValueError, match="explicit 27B identity"):
        emit_and_verify_bonsai_receipt(
            producer,
            input_ids=prompt,
            output_ids=output_ids,
            model_digest=digest,
            sampler=SamplerConfig(mode="greedy"),
            verifier_model=verifier,
            verifier_mode="fresh-oracle",
            ledger_path=tmp_path / "ledger.jsonl",
            broadcast_to_log=False,
        )
    with pytest.raises(ValueError, match="distinct fresh oracle"):
        emit_and_verify_bonsai_receipt(
            producer,
            input_ids=prompt,
            output_ids=output_ids,
            model_digest=digest,
            sampler=SamplerConfig(mode="greedy"),
            verifier_model=producer,
            verifier_mode="fresh-oracle",
            identity_path=identity,
            ledger_path=tmp_path / "ledger.jsonl",
            broadcast_to_log=False,
        )


def _receipt_call(tmp_path, producer, verifier, identity_path):
    return emit_and_verify_bonsai_receipt(
        producer,
        input_ids=[1, 2],
        output_ids=[3],
        model_digest=BONSAI35_RELEASE_ARTIFACT_SHA256,
        sampler=SamplerConfig(mode="greedy"),
        verifier_model=verifier,
        verifier_mode="fresh-oracle",
        identity_path=identity_path,
        ledger_path=tmp_path / "ledger.jsonl",
        broadcast_to_log=False,
    )


def test_qwen35_fresh_oracle_rejects_duck_typed_fake(tmp_path):
    producer, canonical_verifier = _model_pair(56)
    identity, _gate = _write_gated_identity(tmp_path)

    class DuckOracle:
        pass

    fake = DuckOracle()
    fake.artifact = canonical_verifier.artifact
    fake.cfg = canonical_verifier.cfg
    fake._native = False
    fake._native_runtime = None
    fake._model_executor = None
    with pytest.raises(ValueError, match="exact canonical Qwen3.5 model class"):
        _receipt_call(tmp_path, producer, fake, identity)


def test_qwen35_release_digest_cannot_be_disguised_as_another_architecture(tmp_path):
    canonical_producer, verifier = _model_pair(56)
    identity, _gate = _write_gated_identity(tmp_path)

    class DuckProducer:
        pass

    fake = DuckProducer()
    fake.artifact = canonical_producer.artifact
    fake.cfg = dict(canonical_producer.cfg, architecture="qwen3")
    with pytest.raises(ValueError, match="producer must be the canonical Qwen3.5 model class"):
        _receipt_call(tmp_path, fake, verifier, identity)


def test_qwen35_fresh_oracle_rejects_subclass_override(tmp_path):
    producer, canonical_verifier = _model_pair(57)
    identity, _gate = _write_gated_identity(tmp_path)

    class OverridingOracle(BonsaiQwen35ReferenceModel):
        def generate_greedy_tokens_cached(self, *_args, **_kwargs):
            return [3]

    verifier = OverridingOracle(canonical_verifier.artifact)
    with pytest.raises(ValueError, match="exact canonical Qwen3.5 model class"):
        _receipt_call(tmp_path, producer, verifier, identity)


@pytest.mark.parametrize("field", ["_native", "_native_runtime", "_model_executor"])
def test_qwen35_fresh_oracle_rejects_every_native_runtime_handle(tmp_path, field):
    producer, verifier = _model_pair(58)
    identity, _gate = _write_gated_identity(tmp_path)
    setattr(verifier, field, True if field == "_native" else object())
    with pytest.raises(ValueError, match="no native runtime"):
        _receipt_call(tmp_path, producer, verifier, identity)


def test_qwen35_fresh_oracle_rejects_loaded_digest_or_config_drift(tmp_path):
    producer, verifier = _model_pair(59)
    identity, _gate = _write_gated_identity(tmp_path)
    verifier.artifact[_LOADED_ARTIFACT_SHA256] = "00" * 32
    with pytest.raises(ValueError, match="committed artifact digest"):
        _receipt_call(tmp_path, producer, verifier, identity)

    verifier.artifact[_LOADED_ARTIFACT_SHA256] = BONSAI35_RELEASE_ARTIFACT_SHA256
    verifier.cfg["vocab"] += 1
    with pytest.raises(ValueError, match="configs do not match"):
        _receipt_call(tmp_path, producer, verifier, identity)


def test_qwen35_fresh_oracle_requires_separately_loaded_artifact(tmp_path):
    producer, _verifier = _model_pair(60)
    verifier = BonsaiQwen35ReferenceModel(producer.artifact)
    identity, _gate = _write_gated_identity(tmp_path)
    with pytest.raises(ValueError, match="separately loaded artifact"):
        _receipt_call(tmp_path, producer, verifier, identity)


def test_bonsai_loader_records_actual_digest_and_config_for_receipts(tmp_path):
    path = tmp_path / "model.safetensors"
    save_artifact_bonsai(random_bonsai35_artifact(seed=61), path)
    artifact, info = load_artifact_bonsai(path)
    assert _LOADED_ARTIFACT_SHA256 not in artifact
    digest = info["digest"]
    assert artifact[_LOADED_ARTIFACT_SHA256] == digest
    assert artifact[_LOADED_CONFIG_SHA256] == _config_sha256(artifact["config"])


@pytest.mark.parametrize(
    "field",
    ["source", "quant", "importer", "ggufSha256"],
)
def test_qwen35_identity_rejects_nonrelease_weight_provenance(tmp_path, field):
    identity_path, _gate = _write_gated_identity(tmp_path)
    identity = json.loads(identity_path.read_text())
    identity["weightProvenance"][field] = "wrong"
    identity_path.write_text(json.dumps(identity, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match=f"weightProvenance.{field}"):
        validate_bonsai35_receipt_identity(
            identity_path, BONSAI35_RELEASE_ARTIFACT_SHA256
        )


def test_qwen35_identity_rejects_nonfinite_gate_and_stale_summary(tmp_path):
    identity_path, gate_path = _write_gated_identity(tmp_path)
    gate = json.loads(gate_path.read_text())
    gate["value"] = float("nan")
    gate_path.write_text(json.dumps(gate, sort_keys=True) + "\n")
    identity = json.loads(identity_path.read_text())
    identity["qualityGate"]["value"] = float("nan")
    identity["qualityGate"]["gateHash"] = hashlib.sha256(
        gate_path.read_bytes()
    ).hexdigest()
    identity_path.write_text(json.dumps(identity, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="finite"):
        validate_bonsai35_receipt_identity(
            identity_path, BONSAI35_RELEASE_ARTIFACT_SHA256
        )

    identity_path, _gate = _write_gated_identity(tmp_path)
    identity = json.loads(identity_path.read_text())
    identity["qualityGate"]["count"] = 9
    identity_path.write_text(json.dumps(identity, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="qualityGate.count disagrees"):
        validate_bonsai35_receipt_identity(
            identity_path, BONSAI35_RELEASE_ARTIFACT_SHA256
        )
