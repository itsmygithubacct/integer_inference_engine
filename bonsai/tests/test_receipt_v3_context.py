"""v3 binds a receipt to the request that asked for it — and leaves v2 untouched.

The vectors in `fixtures/receipt-v3.vectors.json` were produced by an independent
implementation of the same specification. Checking the engine against them is the
point: if these two ever disagree about a byte, a signature made by one cannot be
verified by the other, and the digests involved are anchored on-chain.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from trinote.receipts.canonical import canonical_bytes
from trinote.receipts.receipt import SCHEMA, SCHEMA_V3, build_receipt, receipt_hash
from trinote.receipts.signing import LocalKey

VECTORS = json.loads((pathlib.Path(__file__).parent / "fixtures" / "receipt-v3.vectors.json").read_text())
MESSAGES = {case["name"]: case for case in VECTORS["messages"]}

CONTEXT = "c0" * 32


def _keys():
    return LocalKey(b"model-secret", "key:model-a"), LocalKey(b"cp-secret", "key:cp-b")


def _build(**over):
    model_key, cp_key = _keys()
    args = dict(model_hash="e5" * 32, input_ids=[1, 2, 3], output_ids=[4, 5],
                sampler={"mode": "greedy"}, model_key=model_key, counterparty_key=cp_key)
    args.update(over)
    return build_receipt(**args)["receipt"]


def test_v2_is_unchanged_by_the_existence_of_v3():
    receipt = _build()
    assert receipt["schema"] == SCHEMA
    assert "contextCommit" not in receipt
    # the hash still recomputes over the body — the property every anchored v2
    # commitment depends on
    assert receipt_hash(receipt) == receipt["receiptHash"]


def test_v3_commits_the_context_in_the_body():
    receipt = _build(schema_version="v3", context_commit=CONTEXT)
    assert receipt["schema"] == SCHEMA_V3
    assert receipt["contextCommit"] == CONTEXT
    assert receipt_hash(receipt) == receipt["receiptHash"]


def test_v3_receipt_hash_differs_from_v2_for_the_same_execution():
    """Same model, same tokens, same trace — a different request context must not
    produce the same receipt."""
    assert _build()["receiptHash"] != _build(schema_version="v3", context_commit=CONTEXT)["receiptHash"]


def test_a_different_context_changes_the_receipt():
    a = _build(schema_version="v3", context_commit=CONTEXT)
    b = _build(schema_version="v3", context_commit="c1" * 32)
    assert a["receiptHash"] != b["receiptHash"]
    assert a["sigModel"] != b["sigModel"]
    assert a["sigCounterparty"] != b["sigCounterparty"]


@pytest.mark.parametrize("bad", [None, "", "C0" * 32, "c0" * 31, 42])
def test_v3_without_a_valid_binding_is_refused(bad):
    """A v3 receipt without a valid context commitment would claim a freshness it
    does not have, so it fails loud rather than emitting an unbound v3."""
    with pytest.raises(ValueError):
        _build(schema_version="v3", context_commit=bad)


def test_context_commit_is_rejected_for_v2():
    with pytest.raises(ValueError):
        _build(schema_version="v2", context_commit=CONTEXT)


def test_unknown_schema_version_fails_loud():
    with pytest.raises(ValueError):
        _build(schema_version="v4")


@pytest.mark.parametrize("name", ["v2-model", "v2-counterparty", "v3-model", "v3-counterparty"])
def test_signed_message_bytes_match_the_independent_vectors(name):
    """The signed messages are rebuilt exactly as build_receipt assembles them, and
    compared with bytes an independent implementation produced from the same inputs."""
    case = MESSAGES[name]
    i = case["inputs"]
    entry = {"modelHash": i["model_hash"], "inputCommit": i["input_commit"],
             "outputCommit": i["output_commit"]}
    if case["entry"] == "model":
        entry["traceCommit"] = i["trace_commit"]
    if i.get("context_commit") is not None:
        entry["contextCommit"] = i["context_commit"]
    assert canonical_bytes(entry).decode() == case["canonicalText"]
