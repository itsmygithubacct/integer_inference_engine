"""v3 binds a receipt to the request that asked for it — and leaves v2 untouched.

The vectors in `fixtures/receipt-v3.vectors.json` were produced by an independent
implementation of the same specification. Checking the engine against them is the
point: if these two ever disagree about a byte, a signature made by one cannot be
verified by the other, and the digests involved are anchored on-chain.

Two things have to hold for that check to mean anything, so both are tested here:

* the vectors' bytes are the bytes this engine's canonical encoder produces, and
* `build_receipt` actually signs that shape.

`_signed_entry` is the single definition of the shape, used by both halves. Checking
the vectors against a shape that `build_receipt` never uses would pass while every
real signature diverged, which is the failure this file exists to make impossible.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from trinote.receipts.canonical import canonical_bytes
from trinote.receipts.receipt import SCHEMA, SCHEMA_V3, build_receipt, receipt_hash
from trinote.receipts.signing import LocalKey

VECTORS = json.loads((pathlib.Path(__file__).parent / "fixtures" / "receipt-v3.vectors.json").read_text())
MESSAGES = {case["name"]: case for case in VECTORS["messages"]}
RECEIPTS = {case["name"]: case for case in VECTORS["receipts"]}

CONTEXT = "c0" * 32


def _signed_entry(*, model_hash, input_commit, output_commit,
                  trace_commit=None, context_commit=None) -> dict:
    """The exact object `build_receipt` signs.

    The model entry carries the trace commitment; the counterparty entry does not,
    because the counterparty co-signs only what it observed. `contextCommit` is
    present for v3 and absent for v2, which is what keeps v2's bytes unmoved.
    """
    entry = {"modelHash": model_hash, "inputCommit": input_commit, "outputCommit": output_commit}
    if trace_commit is not None:
        entry["traceCommit"] = trace_commit
    if context_commit is not None:
        entry["contextCommit"] = context_commit
    return entry


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


# ── what build_receipt actually signs ─────────────────────────────────────────
# Without these, the vector checks below compare an independent implementation
# against a shape assembled in this file, and a rename inside build_receipt would
# leave every vector passing while no signature verified against anything.

@pytest.mark.parametrize("schema_version,context", [("v2", None), ("v3", CONTEXT)])
def test_build_receipt_signs_the_documented_entry_shape(schema_version, context):
    model_key, cp_key = _keys()
    over = {"schema_version": schema_version}
    if context is not None:
        over["context_commit"] = context
    receipt = _build(**over)

    common = dict(model_hash=receipt["modelHash"], input_commit=receipt["inputCommit"],
                  output_commit=receipt["outputCommit"], context_commit=receipt.get("contextCommit"))
    model_entry = _signed_entry(trace_commit=receipt["trace"]["traceCommit"], **common)
    cp_entry = _signed_entry(**common)

    assert receipt["sigModel"] == model_key.sign(canonical_bytes(model_entry))
    assert receipt["sigCounterparty"] == cp_key.sign(canonical_bytes(cp_entry))


def test_the_counterparty_signature_covers_the_context():
    """A counterparty signature that omitted the context would be transferable
    between requests — the failure v3 exists to prevent."""
    model_key, cp_key = _keys()
    receipt = _build(schema_version="v3", context_commit=CONTEXT)
    unbound = _signed_entry(model_hash=receipt["modelHash"], input_commit=receipt["inputCommit"],
                            output_commit=receipt["outputCommit"])
    assert receipt["sigCounterparty"] != cp_key.sign(canonical_bytes(unbound))


# ── the independent vectors ───────────────────────────────────────────────────
# Parametrised over the fixture itself: a vector that is added but never asserted
# would be dead data claiming a coverage it does not have.

@pytest.mark.parametrize("name", sorted(MESSAGES))
def test_signed_message_bytes_match_the_independent_vectors(name):
    case = MESSAGES[name]
    i = case["inputs"]
    entry = _signed_entry(model_hash=i["model_hash"], input_commit=i["input_commit"],
                          output_commit=i["output_commit"], trace_commit=i.get("trace_commit"),
                          context_commit=i.get("context_commit"))
    encoded = canonical_bytes(entry)
    assert encoded.decode() == case["canonicalText"]
    # the digest is what actually gets signed; pin it too, so a change in the
    # encoder cannot be absorbed by an equally-changed expectation
    assert hashlib.sha256(encoded).hexdigest() == case["signingDigest"]


@pytest.mark.parametrize("name", sorted(RECEIPTS))
def test_receipt_hashes_match_the_independent_vectors(name):
    case = RECEIPTS[name]
    assert receipt_hash(case["body"]) == case["receiptHash"]


def test_the_anchored_v2_receipt_hash_is_pinned():
    """The one that cannot be allowed to move: a v2 receiptHash is anchored on-chain,
    so it is asserted against a literal rather than against a recomputation of itself.
    A test that only checks a receipt against its own hash passes just as happily
    when both sides move together."""
    assert receipt_hash(RECEIPTS["v2-body"]["body"]) == \
        "034c423b293b1d2fae6a81984c1010721db006aba1ac1c6d4b4ea5b243da3202"


def test_a_v3_receipt_hash_ignores_a_receipt_hash_already_in_the_body():
    with_field = RECEIPTS["v3-body-hash-excludes-itself"]["body"]
    without_field = RECEIPTS["v3-body"]["body"]
    assert receipt_hash(with_field) == receipt_hash(without_field)
