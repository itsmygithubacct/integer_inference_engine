"""The counterparty signature comes from a key this host does not hold.

The property under test is not "a signature is produced" — it is that the producer
cannot obtain a signature over anything except a counterparty message, cannot be
fooled by a service that answers with the wrong key, and cannot silently fall back to
signing both entries itself.
"""
from __future__ import annotations

import json

import pytest

from trinote.infer_int.bonsai_runtime import (
    COUNTERPARTY_COMMAND_ENV,
    COUNTERPARTY_LOCAL_OK_ENV,
    COUNTERPARTY_PUBKEY_ENV,
    resolve_signing_keys,
)
from trinote.receipts.receipt import build_receipt
from trinote.receipts.remote_counterparty import (
    CounterpartyNotConfigured,
    RemoteCounterpartySigner,
    SigningRefused,
    canonical_bytes,
    key_id_for,
)
from trinote.receipts.signing_ec import ec_keygen
from trinote.receipts.verify import verify_receipt

H = lambda b: b * 64                                    # noqa: E731


def _service(key, *, echo_key=None):
    """A counterparty that rebuilds the message from named fields and signs that."""

    def transport(request: bytes) -> bytes:
        req = json.loads(request)
        if req.get("schema") != "trinote.counterparty-signing-request/v1":
            return canonical_bytes({"ok": False, "code": "bad-schema"})
        msg = req["message"]
        rebuilt = {k: msg[k] for k in sorted(msg)}      # rebuild, never sign what arrived
        signer = echo_key or key
        return canonical_bytes({"ok": True, "signature": signer.sign(canonical_bytes(rebuilt)),
                                "keyId": signer.key_id, "publicKey": signer.public_hex})

    return transport


# ------------------------------------------------------------------ the happy path

def test_a_receipt_is_signed_by_a_key_the_producer_never_holds():
    model, counterparty = ec_keygen(label="m"), ec_keygen(label="c")
    remote = RemoteCounterpartySigner(public_hex=counterparty.public_hex,
                                      transport=_service(counterparty))

    bundle = build_receipt(model_hash=H("a"), input_ids=[1, 2], output_ids=[3],
                           sampler={"mode": "greedy"}, model_key=model, counterparty_key=remote,
                           schema_version="v3", context_commit=H("c"))

    receipt = bundle["receipt"] if "receipt" in bundle else bundle
    assert receipt["sigCounterpartyKeyId"] == counterparty.key_id
    # the producer's own key material is not what vouched
    assert receipt["sigCounterpartyKeyId"] != model.key_id


def test_the_receipt_verifies_from_the_public_key_alone():
    model, counterparty = ec_keygen(label="m"), ec_keygen(label="c")
    remote = RemoteCounterpartySigner(public_hex=counterparty.public_hex,
                                      transport=_service(counterparty))
    bundle = build_receipt(model_hash=H("a"), input_ids=[1, 2], output_ids=[3],
                           sampler={"mode": "greedy"}, model_key=model, counterparty_key=remote,
                           schema_version="v3", context_commit=H("c"))
    receipt = bundle["receipt"] if "receipt" in bundle else bundle
    preimage = {"schema": "trinote.receipt-preimage/v3", "receiptHash": receipt["receiptHash"],
                "modelHash": receipt["modelHash"], "modelLabel": "", "artifactDigest": None,
                "inputIds": [1, 2], "outputIds": [3],
                "sampler": receipt["trace"]["sampler"], "trace": receipt["trace"]}

    res = verify_receipt({"receipt": receipt, "preimage": preimage},
                         model_pubkey=model.public_hex,
                         counterparty_pubkey=counterparty.public_hex)
    assert res["sigModelOk"] is True
    assert res["sigCounterpartyOk"] is True


def test_key_id_is_derived_locally_not_learned_from_the_service():
    counterparty = ec_keygen(label="c")
    assert key_id_for(counterparty.public_hex) == counterparty.key_id


# ------------------------------------------------------------- refusing to be an oracle

@pytest.mark.parametrize("payload, code", [
    (b'"not an object"', "bad-payload"),
    (b'{"please":"sign this"}', "not-a-counterparty-message"),
    # a real counterparty message plus one extra field is still not a counterparty message
    (canonical_bytes({"modelHash": H("a"), "inputCommit": H("b"), "outputCommit": H("c"),
                      "contextCommit": H("d"), "extra": H("e")}), "not-a-counterparty-message"),
    (canonical_bytes({"modelHash": "short", "inputCommit": H("b"), "outputCommit": H("c")}), "bad-hex"),
])
def test_the_producer_cannot_get_arbitrary_bytes_signed(payload, code):
    counterparty = ec_keygen(label="c")
    calls = []

    def counting(request):
        calls.append(request)
        return _service(counterparty)(request)

    remote = RemoteCounterpartySigner(public_hex=counterparty.public_hex, transport=counting)
    with pytest.raises(SigningRefused) as exc:
        remote.sign(payload)
    assert exc.value.code == code
    assert calls == [], "a non-message must not reach the counterparty at all"


def test_a_non_canonical_payload_is_refused_rather_than_signed():
    counterparty = ec_keygen(label="c")
    remote = RemoteCounterpartySigner(public_hex=counterparty.public_hex,
                                      transport=_service(counterparty))
    # same fields, spaces between them: parses identically, encodes differently
    spaced = json.dumps({"modelHash": H("a"), "inputCommit": H("b"), "outputCommit": H("c")},
                        sort_keys=True).encode()
    with pytest.raises(SigningRefused) as exc:
        remote.sign(spaced)
    assert exc.value.code == "non-canonical-payload"


# ------------------------------------------------------------------------- the pin

def test_a_stranger_who_signs_correctly_is_still_refused():
    counterparty, stranger = ec_keygen(label="c"), ec_keygen(label="s")
    # the service answers with a perfectly valid signature — from the wrong key
    remote = RemoteCounterpartySigner(public_hex=counterparty.public_hex,
                                      transport=_service(counterparty, echo_key=stranger))
    with pytest.raises(SigningRefused) as exc:
        remote.sign(canonical_bytes({"modelHash": H("a"), "inputCommit": H("b"),
                                     "outputCommit": H("c")}))
    assert exc.value.code == "unpinned-counterparty"


def test_a_refusal_from_the_service_is_propagated_not_swallowed():
    counterparty = ec_keygen(label="c")
    remote = RemoteCounterpartySigner(
        public_hex=counterparty.public_hex,
        transport=lambda _: canonical_bytes({"ok": False, "code": "unbound-request"}))
    with pytest.raises(SigningRefused) as exc:
        remote.sign(canonical_bytes({"modelHash": H("a"), "inputCommit": H("b"),
                                     "outputCommit": H("c")}))
    assert exc.value.code == "unbound-request"


def test_a_refusal_that_exits_non_zero_is_still_read_as_a_refusal():
    """Found by running it: the service exits non-zero when it refuses, and an earlier
    version judged the exit code before reading stdout — so "this counterparty does not
    co-sign unbound receipts" arrived as "transport-failed", pointing the operator at
    the network instead of at the policy."""
    counterparty = ec_keygen(label="c")
    remote = RemoteCounterpartySigner(public_hex=counterparty.public_hex,
                                      command=["/bin/sh", "-c",
                                               "cat >/dev/null; "
                                               "printf '%s' '{\"code\":\"unbound-request\",\"ok\":false}'; "
                                               "exit 1"])
    with pytest.raises(SigningRefused) as exc:
        remote.sign(canonical_bytes({"modelHash": H("a"), "inputCommit": H("b"),
                                     "outputCommit": H("c")}))
    assert exc.value.code == "unbound-request"


def test_a_transport_that_really_broke_is_reported_as_transport_failure():
    counterparty = ec_keygen(label="c")
    remote = RemoteCounterpartySigner(
        public_hex=counterparty.public_hex,
        command=["/bin/sh", "-c", "cat >/dev/null; echo 'ssh: connect refused' >&2; exit 255"])
    with pytest.raises(SigningRefused) as exc:
        remote.sign(canonical_bytes({"modelHash": H("a"), "inputCommit": H("b"),
                                     "outputCommit": H("c")}))
    assert exc.value.code == "transport-failed"
    assert "connect refused" in exc.value.detail


@pytest.mark.parametrize("pin", ["", "not-hex", "ab" * 32, "ab" * 34])
def test_the_pin_itself_is_validated_at_construction(pin):
    with pytest.raises(SigningRefused):
        RemoteCounterpartySigner(public_hex=pin, transport=lambda _: b"{}")


# ------------------------------------------------------- what the default now refuses

def test_co_resident_signing_is_refused_unless_it_is_asked_for(tmp_path, monkeypatch):
    for var in (COUNTERPARTY_COMMAND_ENV, COUNTERPARTY_PUBKEY_ENV, COUNTERPARTY_LOCAL_OK_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TRINOTE_DEMO_KEYS_OK", "")
    monkeypatch.delenv("TRINOTE_DEMO_KEYS_OK", raising=False)
    # _demo_keys_requested() is true under pytest, so pin it false for this one check
    monkeypatch.setattr("trinote.infer_int.bonsai_runtime._demo_keys_requested", lambda: False)

    with pytest.raises(CounterpartyNotConfigured) as exc:
        resolve_signing_keys(str(tmp_path / "model.json"))
    assert exc.value.code == "co-resident-counterparty"


def test_the_waiver_is_one_flag_and_it_works(tmp_path, monkeypatch):
    monkeypatch.setattr("trinote.infer_int.bonsai_runtime._demo_keys_requested", lambda: False)
    model, counterparty = resolve_signing_keys(str(tmp_path / "m.json"),
                                               allow_local_counterparty=True)
    assert counterparty.public_hex != model.public_hex


def test_an_explicitly_named_local_key_still_stands(tmp_path, monkeypatch):
    monkeypatch.setattr("trinote.infer_int.bonsai_runtime._demo_keys_requested", lambda: False)
    model, counterparty = resolve_signing_keys(str(tmp_path / "m.json"),
                                               str(tmp_path / "c.json"))
    assert counterparty.public_hex != model.public_hex


def test_a_remote_counterparty_without_a_pin_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv(COUNTERPARTY_PUBKEY_ENV, raising=False)
    with pytest.raises(CounterpartyNotConfigured) as exc:
        resolve_signing_keys(str(tmp_path / "m.json"), counterparty_command="ssh notary sign")
    assert exc.value.code == "unpinned-counterparty"


def test_the_environment_configures_it_too(tmp_path, monkeypatch):
    counterparty = ec_keygen(label="c")
    monkeypatch.setenv(COUNTERPARTY_COMMAND_ENV, "ssh notary trinote-counterparty-sign")
    monkeypatch.setenv(COUNTERPARTY_PUBKEY_ENV, counterparty.public_hex)
    _, signer = resolve_signing_keys(str(tmp_path / "m.json"))
    assert isinstance(signer, RemoteCounterpartySigner)
    assert signer.command == ["ssh", "notary", "trinote-counterparty-sign"]
    assert signer.key_id == counterparty.key_id
