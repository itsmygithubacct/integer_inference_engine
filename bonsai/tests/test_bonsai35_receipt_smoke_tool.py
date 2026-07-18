from __future__ import annotations

import importlib.util
from pathlib import Path

from trinote.infer_int.reference_bonsai35 import (
    BonsaiQwen35ReferenceModel,
    random_bonsai35_artifact,
)
from trinote.infer_int.sampler import SamplerConfig
from trinote.receipts import build_receipt, keygen


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "verify_bonsai35_receipt_smoke.py"


def _module():
    spec = importlib.util.spec_from_file_location("verify_bonsai35_receipt_smoke", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_receipt_smoke_tool_accepts_correct_and_rejects_corrupted_output():
    module = _module()
    oracle = BonsaiQwen35ReferenceModel(random_bonsai35_artifact(seed=812))
    input_ids = [2, 5, 7]
    output_ids = oracle.generate_greedy_tokens_cached(input_ids, 2)
    digest = "81" * 32
    model_key = keygen(label="bonsai35", secret_hex="41" * 32)
    counterparty_key = keygen(label="counterparty", secret_hex="42" * 32)
    bundle = build_receipt(
        model_hash=digest,
        input_ids=input_ids,
        output_ids=output_ids,
        sampler=SamplerConfig(mode="greedy"),
        model_key=model_key,
        counterparty_key=counterparty_key,
        model_label="ATLAS-Notarized-Bonsai-27B",
        artifact_digest=digest,
    )
    # HMAC verification needs the external secrets; use the normal EC path in
    # the smoke report by rebuilding with generated EC keys is unnecessary for
    # the corruption logic.  Supply a small wrapper that pins the HMAC keys by
    # monkeypatching the tool's verifier call.
    real_verify = module.verify_receipt

    def verify_with_keys(value, **kwargs):
        return real_verify(
            value, **kwargs, model_key=model_key, counterparty_key=counterparty_key
        )

    module.verify_receipt = verify_with_keys
    report = module.evaluate_receipt_smoke(bundle, oracle=oracle, model_digest=digest)
    assert report["status"] == "pass"
    assert all(report["acceptance"].values())
    assert report["mutation"]["original"] != report["mutation"]["corrupted"]
    assert not report["corruptedVerification"]["commitMatch"]
    assert not report["corruptedVerification"]["reexecOk"]


def test_receipt_smoke_atomic_json_output(tmp_path):
    module = _module()
    target = tmp_path / "nested" / "report.json"
    payload = module.canonical_json_bytes({"status": "pass", "café": True})
    module.atomic_write(target, payload)
    assert target.read_bytes() == payload
    assert list(target.parent.iterdir()) == [target]
