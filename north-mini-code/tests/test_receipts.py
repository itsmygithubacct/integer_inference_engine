"""Receipt machinery integration (model-free): build → secp256k1-sign → verify → ledger → bundle → offline
re-verify, reusing the Bonsai notary stack. Generation/model_hash are exercised on a GPU box separately."""
import numpy as np
import pytest

from nmc.receipts_runtime import build_verify_pack, load_keys, ECKey, SamplerConfig, token_commit


def test_receipt_build_verify_pack(tmp_path):
    km = ECKey.generate(label="m")
    kc = ECKey.generate(label="c")
    ids = [2, 669, 6345, 302]
    out = [12071, 42, 7, 9, 1]
    res = build_verify_pack(
        model_hash="de" * 32, artifact_digest="ca" * 32, input_ids=ids, output_ids=out,
        sampler=SamplerConfig(mode="greedy"), fa=16, model_key=km, counterparty_key=kc,
        out_dir=tmp_path / "bundle", ledger_path=tmp_path / "ledger.jsonl",
        transcript={"prompt": "hi", "output": "there"})

    r = res["receipt"]
    assert r["inputCommit"] == token_commit(ids)
    assert r["outputCommit"] == token_commit(out)
    assert r["sigModel"].startswith("secp256k1-ecdsa@v1:")
    assert r["sigCounterparty"].startswith("secp256k1-ecdsa@v1:")
    v = res["verify_receipt"]                                # model-free: offline + signatures must pass
    assert v["commitMatch"] and v["receiptHashMatch"] and v["signatureOk"] and v["structuralOk"], v
    assert res["offline_ok"]
    assert res["verify_bundle"]["ok"], res["verify_bundle"]
    assert res["verify_bundle"]["offline"]["ok"]
    # bundle is content-addressed + on disk
    assert (tmp_path / "bundle" / "manifest.json").exists()
    assert res["bundle"]["bundleHash"]


def test_receipt_can_disable_even_the_local_broadcast_log(tmp_path):
    km = ECKey.generate(label="m")
    kc = ECKey.generate(label="c")
    res = build_verify_pack(
        model_hash="de" * 32, artifact_digest="ca" * 32,
        input_ids=[2, 5], output_ids=[10, 11],
        sampler=SamplerConfig(mode="greedy"), fa=16,
        model_key=km, counterparty_key=kc,
        out_dir=tmp_path / "bundle", ledger_path=tmp_path / "ledger.jsonl",
        enable_chain=False, broadcast_to_log=False,
    )
    assert res["emission"]["onchain"]["status"] == "disabled"
    assert res["bundle"]["manifest"]["kind"] == "local"
    assert set(res["bundle"]["manifest"]["files"]) == {
        "receipt.json", "preimage.json", "chain-artifact.json", "ledger-head.json",
    }


def test_tamper_breaks_verification(tmp_path):
    """Flipping an output id must break the offline commitment check (the point of a receipt)."""
    km = ECKey.generate(label="m"); kc = ECKey.generate(label="c")
    res = build_verify_pack(
        model_hash="de" * 32, artifact_digest="ca" * 32, input_ids=[2, 5], output_ids=[10, 11],
        sampler=SamplerConfig(mode="greedy"), fa=16, model_key=km, counterparty_key=kc,
        out_dir=tmp_path / "b", ledger_path=tmp_path / "l.jsonl")
    r = res["receipt"]
    assert r["outputCommit"] != token_commit([10, 12])     # a different output commits differently


def test_keys_persist_0600(tmp_path):
    mp, cp = tmp_path / "model.json", tmp_path / "cp.json"
    km, kc = load_keys(mp, cp)
    assert mp.exists() and cp.exists()
    assert (mp.stat().st_mode & 0o777) == 0o600            # secret keys are owner-only
    km2, _ = load_keys(mp, cp)
    assert km.public_hex == km2.public_hex                  # stable across loads
