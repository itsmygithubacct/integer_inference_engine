from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from trinote.cli import receipt_bundle_cli as bundle_cli
from trinote.cli.run_bonsai_cli import _qwen3_chat_prompt, _validate_args
from trinote.receipts.signing_ec import ECKey


def test_qwen35_non_thinking_chat_prefix_is_hard_closed():
    kv = {
        "general.architecture": "qwen35",
        "tokenizer.chat_template": (
            "<|im_start|><|im_end|> add_generation_prompt <think>\\n"
        ),
    }
    rendered = _qwen3_chat_prompt("how many r's are in strawberry?", kv, thinking=False)
    assert rendered == (
        "<|im_start|>user\nhow many r's are in strawberry?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_no_think_requires_chat(capsys):
    args = SimpleNamespace(
        no_think=True, chat=False, sampler="greedy", temp=1.0, top_k=1,
        top_p=1.0, min_p=0.0, no_repeat_ngram=0, receipt=False,
        engine="native", onchain=False, json=False, ctx_size=None,
        n_gpu_layers=None, fast_required=False, fast=False,
        prompt_cache=False, verify_mode="fast-local",
    )
    assert _validate_args(args) == 2
    assert "--no-think requires --chat" in capsys.readouterr().err


def test_stateful_emission_record_becomes_stateful_bundle_inputs():
    record = {
        "actionTxid": "aa" * 32,
        "receiptHashOnChain": "bb" * 32,
        "txCount": "2",
        "lockTime": "123",
        "amount": "1000",
        "actionHash": "cc" * 32,
        "provenanceHash": "dd" * 32,
        "receiptVout": 1,
        "rawTx": "01000000",
        "sizeBytes": 4,
        "identity": {
            "ricardianHash": "ee" * 32,
            "genesisTxid": "ff" * 32,
            "agentPubKey": "02" + "11" * 32,
            "counterpartyPubKey": "03" + "22" * 32,
        },
    }
    onchain, identity = bundle_cli._stateful_from_record(record, "main")
    assert onchain["kind"] == "stateful"
    assert onchain["actionTxid"] == record["actionTxid"]
    assert onchain["action"]["txCount"] == 2
    assert onchain["action"]["actionHash"] == record["actionHash"]
    assert onchain["rawTx"] == record["rawTx"]
    assert identity["genesisTxid"] == record["identity"]["genesisTxid"]


def test_bundle_loader_selects_qwen35_model(monkeypatch):
    import trinote.infer_int.artifact_io_bonsai as artifact_io
    import trinote.infer_int.reference_bonsai35 as reference_bonsai35

    class FakeQwen35:
        def __init__(self, artifact):
            self.artifact = artifact

    artifact = {"config": {"architecture": "qwen35"}}
    monkeypatch.setattr(artifact_io, "load_artifact_bonsai", lambda _: (artifact, {"digest": "ab" * 32}))
    monkeypatch.setattr(reference_bonsai35, "BonsaiQwen35ReferenceModel", FakeQwen35)
    model, digest, engine = bundle_cli._load_model("fake.safetensors", fast=False)
    assert isinstance(model, FakeQwen35)
    assert digest == "ab" * 32
    assert engine == "oracle"


def test_chain_c_private_hex_keyfile_can_sign_receipts():
    expected = ECKey.from_secret_hex("11" * 32)
    loaded = ECKey.from_json({
        "private_key_hex": "11" * 32,
        "public_key_hex": expected.public_hex,
    })
    assert loaded.public_hex == expected.public_hex
    with pytest.raises(ValueError, match="claimed public key"):
        ECKey.from_json({"private_key_hex": "11" * 32, "public_key_hex": "02" + "00" * 32})


def test_wallet_compressed_wif_keyfile_can_sign_receipts():
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    expected = ECKey.from_secret_hex("22" * 32)
    payload = b"\x80" + bytes.fromhex("22" * 32) + b"\x01"
    raw = payload + hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    leading = len(raw) - len(raw.lstrip(b"\x00"))
    value = int.from_bytes(raw, "big")
    chars = ""
    while value:
        value, digit = divmod(value, 58)
        chars = alphabet[digit] + chars
    wif = "1" * leading + chars
    loaded = ECKey.from_json({"wif": wif, "publicKeyHex": expected.public_hex})
    assert loaded.public_hex == expected.public_hex
