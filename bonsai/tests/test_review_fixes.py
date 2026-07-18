"""Regression tests for the 2026-06-30 deep-review remediation (offline, no network/weights).

Each test pins a specific finding from the review so the fix cannot silently regress. Finding refs
match FIXLOG.md / the review's ranked findings.
"""
from __future__ import annotations

import pytest

from trinote.bundle import pack_bundle, BundleError
from trinote.bundle import chain_read
from trinote.bundle import verify as V


# --- #1: on-chain verification is network-authoritative + network allowlist ----------------------

def test_chain_read_validate_network_rejects_unknown():
    for good in ("main", "test", "stn"):
        assert chain_read._validate_network(good) == good
    for bad in ("evil", "main/tx/x", "", "MAIN", "main; rm -rf", "mainnet"):
        with pytest.raises(chain_read.ChainReadError):
            chain_read._validate_network(bad)


def test_verify_onchain_network_is_caller_authoritative(monkeypatch):
    """A bundle declaring network='test' must NOT redirect a --network main audit; the caller's
    network is used for the fetch and the declared mismatch is a FAILING check (finding #1)."""
    seen = {}

    def fake_standalone(txid, network):
        seen["net"] = network
        return {"found": False, "reason": "stub (not fetched in test)"}

    monkeypatch.setattr(V.chain_read, "read_standalone_anchor", fake_standalone)
    loaded = {"obj": {
        "onchain.json": {"kind": "standalone", "network": "test", "txid": "ab" * 32},
        "receipt.json": {"modelHash": "00" * 32, "receiptHash": "11" * 32},
        "identity.json": {},
    }}
    res = V._verify_onchain(loaded, "main")

    assert seen["net"] == "main", "fetch must use the caller's network, not the bundle's"
    net_checks = [c for c in res["checks"] if c["check"] == "onchain.network"]
    assert net_checks and net_checks[0]["ok"] is False, "declared!=caller network must fail closed"
    assert res["ok"] is False


def test_verify_onchain_tag_pinned_to_protocol_constant(monkeypatch):
    """The standalone tag check is pinned to 'trinote/r1', not a bundle-supplied onchain.tag."""
    def fake_standalone(txid, network):
        return {"found": True, "modelHash": "00" * 32, "receiptHash": "11" * 32, "tag": "attacker/r1"}

    monkeypatch.setattr(V.chain_read, "read_standalone_anchor", fake_standalone)
    loaded = {"obj": {
        "onchain.json": {"kind": "standalone", "txid": "ab" * 32, "tag": "attacker/r1"},
        "receipt.json": {"modelHash": "00" * 32, "receiptHash": "11" * 32},
        "identity.json": {},
    }}
    res = V._verify_onchain(loaded, "main")
    tag_checks = [c for c in res["checks"] if c["check"] == "onchain.tag"]
    assert tag_checks and tag_checks[0]["ok"] is False, "a non-trinote/r1 anchor tag must not verify"


# --- #4: stateful provenance walk is mandatory ----------------------------------------------------

def test_pack_stateful_requires_genesis(tmp_path):
    """pack_bundle must refuse a stateful bundle without identity.genesisTxid (finding #4)."""
    bundle = {"receipt": {"receiptHash": "11" * 32}, "preimage": {"modelLabel": "x"}}
    with pytest.raises(BundleError, match="genesisTxid"):
        pack_bundle(bundle=bundle, onchain={"kind": "stateful"},
                    identity={"ricardianHash": "22" * 32}, out_dir=str(tmp_path / "b"))


def test_verify_onchain_missing_genesis_fails_closed(monkeypatch):
    """A stateful bundle whose identity omits genesisTxid must fail the chainToGenesis check
    instead of silently skipping it (finding #4)."""
    def fake_stateful(txid, network):
        return {"found": True, "receiptHash": "11" * 32}

    monkeypatch.setattr(V.chain_read, "read_stateful_anchor", fake_stateful)
    loaded = {"obj": {
        "onchain.json": {"kind": "stateful", "actionTxid": "cd" * 32, "receiptHashOnChain": "11" * 32},
        "receipt.json": {"modelHash": "00" * 32, "receiptHash": "11" * 32},
        "identity.json": {"ricardianHash": "22" * 32},  # no genesisTxid
    }}
    res = V._verify_onchain(loaded, "main")
    g = [c for c in res["checks"] if c["check"] == "onchain.chainToGenesis"]
    assert g and g[0]["ok"] is False, "missing genesisTxid must fail the provenance check"
    assert res["ok"] is False


# --- #5: identity binding fails closed on a supplied-but-missing identity file -------------------

def test_identity_model_hash_none_is_binding_off():
    from trinote.infer_int.bonsai_runtime import identity_model_hash
    assert identity_model_hash(None) is None


def test_identity_model_hash_missing_path_fails_closed(tmp_path):
    from trinote.infer_int.bonsai_runtime import identity_model_hash
    with pytest.raises(FileNotFoundError):
        identity_model_hash(tmp_path / "not-minted-yet.identity.json")


def test_identity_model_hash_reads_modelhash(tmp_path):
    from trinote.infer_int.bonsai_runtime import identity_model_hash
    p = tmp_path / "id.json"
    p.write_text('{"modelHash": "abcd1234"}')
    assert identity_model_hash(p) == "abcd1234"


# --- #10: verify_receipt is uniformly fail-closed on adversarial bundles -------------------------

def test_verify_receipt_failclosed_on_malformed_bundle():
    """A bundle that passes initial extraction but lacks later required fields must return
    ok:False, not raise an uncaught KeyError (which would DoS a ledger-sweep verifier)."""
    from trinote.receipts.verify import verify_receipt
    bundle = {"receipt": {"modelHash": "00" * 32}, "preimage": {"inputIds": [1], "outputIds": [2]}}
    res = verify_receipt(bundle)            # must not raise
    assert isinstance(res, dict) and res.get("ok") is False


def test_verify_receipt_failclosed_on_non_dict():
    from trinote.receipts.verify import verify_receipt
    res = verify_receipt("not a bundle")    # must not raise
    assert isinstance(res, dict) and res.get("ok") is False


# --- review-3 HIGH: empty inputIds must not false-accept via negative-index wrap ------------------

def test_row_predicting_output_refuses_empty_prompt_first_token():
    """The predicting row for output[0] with empty inputIds is index -1; indexing full[-1] would wrap
    to the LAST prefill row (a false-accept vector). The helper must fail loud instead."""
    import numpy as np
    from trinote.infer_int.verify import _row_predicting_output
    full = np.arange(30, dtype=np.int64).reshape(3, 10)   # 3 prefill rows
    # output[0] with no input has no predicting row -> raise, do NOT return full[-1]
    with pytest.raises(ValueError):
        _row_predicting_output(model=None, input_ids=[], output_ids=[7], i=0, eff=64, full=full)
    # sanity: a legitimate prefix still resolves (input=[a], output[0] predicted by row 0)
    row = _row_predicting_output(model=None, input_ids=[1], output_ids=[7], i=0, eff=64, full=full)
    assert np.array_equal(row, full[0])


def test_verify_receipt_rejects_empty_inputids_bundle():
    """A crafted bundle with empty inputIds and a non-empty output must be rejected at the boundary,
    never reaching re-execution (would otherwise be the forged-receipt vector)."""
    from trinote.receipts.verify import verify_receipt
    bundle = {"receipt": {"modelHash": "00" * 32,
                          "inputCommit": "x", "outputCommit": "y", "receiptHash": "z"},
              "preimage": {"inputIds": [], "outputIds": [4, 5]}}
    res = verify_receipt(bundle)             # must not raise
    assert isinstance(res, dict) and res.get("ok") is False and res.get("fullyVerified") is False


# --- review-3 MEDIUM: importer must fail loud on non-finite / out-of-range float scales ------------

def test_import_rint_to_fixed_rejects_non_finite():
    import numpy as np
    from trinote.infer_int.import_bonsai_gguf import _rint_to_fixed_i64
    # finite values round correctly
    out = _rint_to_fixed_i64(np.array([0.5, -0.5, 1.0], dtype=np.float64), 16, "test")
    assert out.dtype == np.int64 and out[2] == (1 << 16)
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            _rint_to_fixed_i64(np.array([1.0, bad], dtype=np.float64), 16, "test")
    # a scale too large to hold in int64 at this frac is rejected, not silently wrapped
    with pytest.raises(ValueError):
        _rint_to_fixed_i64(np.array([1e30], dtype=np.float64), 40, "test")


def test_import_rejects_out_of_range_frac():
    import pytest
    from trinote.infer_int.import_bonsai_gguf import import_bonsai_gguf_to_artifact
    for bad in (0, 30, -1, 64):
        with pytest.raises(ValueError):
            import_bonsai_gguf_to_artifact("/nonexistent.gguf", frac=bad)   # frac checked before file open


# --- review-3 MEDIUM: bonsai DP4A L=4 envelope must equal the exact balanced-base-256 capacity ------

def test_dp4a_l4_envelope_matches_balanced_capacity():
    """The L=4 range guard (host + device) must clamp to the EXACT max magnitude 4 balanced base-256 digits
    hold — 127*(256^4-1)/255 up, -128*(256^4-1)/255 down — not a looser 2^31 bound, else it would admit
    values the greedy decomposition can't reconstruct (a silent GPU wrap)."""
    from trinote.infer_int.gpu_native import _L4_LO, _L4_HI
    cap = (256 ** 4 - 1) // 255
    assert _L4_HI == 127 * cap == 2139062143
    assert _L4_LO == -128 * cap == -2155905152


# --- review-2 #2: offline rawTx-binding must fail closed on a truncated rawTx (not crash) ---------

def test_parse_tx_raises_chainreaderror_and_verify_catches_it():
    from trinote.bundle import chain_read
    with pytest.raises(chain_read.ChainReadError):
        chain_read.parse_tx("01000000")     # declares inputs but is truncated
    # ChainReadError is a RuntimeError, NOT in (ValueError,KeyError,IndexError,TypeError), so the
    # offline-binding except in verify.py MUST list it explicitly or a malformed bundle crashes.
    assert issubclass(chain_read.ChainReadError, RuntimeError)
    assert not issubclass(chain_read.ChainReadError, (ValueError, KeyError, IndexError, TypeError))
    import inspect
    from trinote.bundle import verify as V
    assert "chain_read.ChainReadError" in inspect.getsource(V._verify_offline)


# --- review-2 #5/#12: identity binding fails closed on a JSON-null / empty / non-string modelHash --

def test_identity_model_hash_null_fails_closed(tmp_path):
    from trinote.infer_int.bonsai_runtime import identity_model_hash
    p = tmp_path / "id.json"
    for bad in ('{"modelHash": null}', '{"modelHash": ""}', '{"modelHash": 123}', '{}'):
        p.write_text(bad)
        with pytest.raises(ValueError):
            identity_model_hash(p)
