"""Tests for the receipt-bundle pack/verify tooling (trinote.bundle) — offline, no network, no weights."""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from trinote.bundle import pack_bundle, verify_bundle, load_bundle, agent_action_receipt_hash, BundleError
from trinote.bundle import chain_read
from trinote.receipts import build_receipt, keygen, log_transaction, read_tx_log, tx_record
from trinote.hashing.sha import sha256_hex, txid_of, double_sha256
from trinote.infer_int.sampler import SamplerConfig


def _sampler():
    return SamplerConfig(mode="greedy", temperature=1.0, top_k=0, top_p=1.0, seed=0,
                         rep_penalty=0, no_repeat_ngram=0)


def _receipt_bundle():
    mk = keygen(label="model", secret_hex="11" * 32)
    ck = keygen(label="counterparty", secret_hex="22" * 32)
    return build_receipt(
        model_hash=sha256_hex("bonsai-model"),
        input_ids=[1, 2, 3, 4],
        output_ids=[5, 6, 7],
        sampler=_sampler(),
        model_key=mk,
        counterparty_key=ck,
        model_label="ATLAS-Notarized-Bonsai-8B",
        artifact_digest=sha256_hex("bonsai-model"),
    )


# --------------------------------------------------------------------------------------------------
# Standalone bundle
# --------------------------------------------------------------------------------------------------
def test_pack_verify_standalone(tmp_path):
    rb = _receipt_bundle()
    receipt = rb["receipt"]
    onchain = {"kind": "standalone", "network": "main", "tag": "trinote/r1",
               "txid": "ab" * 32, "modelHash": receipt["modelHash"], "receiptHash": receipt["receiptHash"]}
    res = pack_bundle(bundle=rb, onchain=onchain, out_dir=tmp_path / "b1")
    assert res["bundleHash"]
    out = verify_bundle(tmp_path / "b1")
    assert out["ok"], [c for c in out["offline"]["checks"] if not c["ok"]]
    assert out["kind"] == "standalone"
    assert out["bundleHash"] == res["bundleHash"]


def test_tamper_receipt_file_detected(tmp_path):
    rb = _receipt_bundle()
    receipt = rb["receipt"]
    onchain = {"kind": "standalone", "network": "main", "tag": "trinote/r1",
               "txid": "ab" * 32, "modelHash": receipt["modelHash"], "receiptHash": receipt["receiptHash"]}
    pack_bundle(bundle=rb, onchain=onchain, out_dir=tmp_path / "b2")
    # Flip a byte in receipt.json without touching the manifest digest.
    rp = tmp_path / "b2" / "receipt.json"
    data = json.loads(rp.read_text())
    data["modelHash"] = "00" * 32
    rp.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")))
    out = verify_bundle(tmp_path / "b2")
    assert not out["ok"]
    names = {c["check"] for c in out["offline"]["checks"] if not c["ok"]}
    assert "file:receipt.json" in names


def test_verify_bundle_failclosed_on_noninteger_seed(tmp_path):
    """review-3 MEDIUM: a crafted receipt whose trace.sampler.seed is not an integer makes chain_artifact()
    raise inside _verify_offline. verify_bundle must catch it and fail closed, not crash the verifier."""
    rb = _receipt_bundle()
    receipt = rb["receipt"]
    onchain = {"kind": "standalone", "network": "main", "tag": "trinote/r1",
               "txid": "ab" * 32, "modelHash": receipt["modelHash"], "receiptHash": receipt["receiptHash"]}
    pack_bundle(bundle=rb, onchain=onchain, out_dir=tmp_path / "bseed")
    rp = tmp_path / "bseed" / "receipt.json"
    data = json.loads(rp.read_text())
    data.setdefault("trace", {}).setdefault("sampler", {})["seed"] = "not-an-integer"
    rp.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")))
    out = verify_bundle(tmp_path / "bseed")                  # must NOT raise
    assert isinstance(out, dict) and out["ok"] is False
    assert any(not c["ok"] for c in out["offline"]["checks"])


def test_nonobject_manifest_raises_typed_bundle_error(tmp_path):
    """A syntactically valid JSON manifest still has to be an object at the container boundary."""
    root = tmp_path / "nonobject-manifest"
    root.mkdir()
    (root / "manifest.json").write_text("[]\n")
    with pytest.raises(BundleError, match="manifest.json must contain a JSON object"):
        verify_bundle(root)


def test_tamper_manifest_bundlehash_detected(tmp_path):
    rb = _receipt_bundle()
    receipt = rb["receipt"]
    onchain = {"kind": "standalone", "network": "main", "tag": "trinote/r1",
               "txid": "ab" * 32, "modelHash": receipt["modelHash"], "receiptHash": receipt["receiptHash"]}
    pack_bundle(bundle=rb, onchain=onchain, out_dir=tmp_path / "b3")
    mp = tmp_path / "b3" / "manifest.json"
    m = json.loads(mp.read_text())
    m["bundleHash"] = "ff" * 32
    mp.write_text(json.dumps(m, sort_keys=True, separators=(",", ":")))
    out = verify_bundle(tmp_path / "b3")
    assert not out["ok"]
    names = {c["check"] for c in out["offline"]["checks"] if not c["ok"]}
    assert "bundleHash" in names


def test_tar_roundtrip(tmp_path):
    rb = _receipt_bundle()
    receipt = rb["receipt"]
    onchain = {"kind": "standalone", "network": "main", "tag": "trinote/r1",
               "txid": "ab" * 32, "modelHash": receipt["modelHash"], "receiptHash": receipt["receiptHash"]}
    res = pack_bundle(bundle=rb, onchain=onchain, out_dir=tmp_path / "b4.tar.gz", as_tar=True)
    assert Path(res["path"]).is_file()
    loaded = load_bundle(res["path"])
    assert loaded["manifest"]["bundleHash"] == res["bundleHash"]
    out = verify_bundle(res["path"])
    assert out["ok"], [c for c in out["offline"]["checks"] if not c["ok"]]


# --------------------------------------------------------------------------------------------------
# Stateful (AgentTea) bundle
# --------------------------------------------------------------------------------------------------
def _stateful_inputs(receipt):
    ricardian = sha256_hex("charter")
    agent_pk = "02" + "11" * 32          # 33-byte compressed pubkey
    cpty_pk = "03" + "22" * 32
    amount, tx_count, lock_time = 1000, 0, 1_718_000_000
    action_hash = receipt["receiptHash"]
    provenance_hash = receipt["modelHash"]
    onchain_hash = agent_action_receipt_hash(
        ricardian_hash=ricardian, agent_pubkey=agent_pk, counterparty_pubkey=cpty_pk,
        amount=amount, action_hash=action_hash, provenance_hash=provenance_hash,
        tx_count=tx_count, lock_time=lock_time)
    identity = {"ricardianHash": ricardian, "genesisTxid": "cd" * 32,
                "agentPubKey": agent_pk, "counterpartyPubKey": cpty_pk}
    onchain = {"kind": "stateful", "network": "main", "actionTxid": "ef" * 32,
               "receiptHashOnChain": onchain_hash,
               "action": {"amount": amount, "txCount": tx_count, "lockTime": lock_time,
                          "actionHash": action_hash, "provenanceHash": provenance_hash}}
    return identity, onchain


def test_pack_verify_stateful(tmp_path):
    rb = _receipt_bundle()
    identity, onchain = _stateful_inputs(rb["receipt"])
    res = pack_bundle(bundle=rb, onchain=onchain, out_dir=tmp_path / "s1", identity=identity)
    out = verify_bundle(tmp_path / "s1")
    assert out["ok"], [c for c in out["offline"]["checks"] if not c["ok"]]
    assert out["kind"] == "stateful"


def test_stateful_binding_mismatch_detected(tmp_path):
    rb = _receipt_bundle()
    identity, onchain = _stateful_inputs(rb["receipt"])
    # Break the actionHash<->receiptHash binding.
    onchain["action"]["actionHash"] = "00" * 32
    onchain["receiptHashOnChain"] = agent_action_receipt_hash(
        ricardian_hash=identity["ricardianHash"], agent_pubkey=identity["agentPubKey"],
        counterparty_pubkey=identity["counterpartyPubKey"], amount=onchain["action"]["amount"],
        action_hash="00" * 32, provenance_hash=onchain["action"]["provenanceHash"],
        tx_count=onchain["action"]["txCount"], lock_time=onchain["action"]["lockTime"])
    pack_bundle(bundle=rb, onchain=onchain, out_dir=tmp_path / "s2", identity=identity)
    out = verify_bundle(tmp_path / "s2")
    assert not out["ok"]
    names = {c["check"] for c in out["offline"]["checks"] if not c["ok"]}
    assert "stateful.actionHash==receiptHash" in names


def test_stateful_requires_identity(tmp_path):
    rb = _receipt_bundle()
    _, onchain = _stateful_inputs(rb["receipt"])
    with pytest.raises(Exception):
        pack_bundle(bundle=rb, onchain=onchain, out_dir=tmp_path / "s3")  # no identity


# --------------------------------------------------------------------------------------------------
# Exact byte layout of the AgentTea action receipt hash (locks the contract-matching encoding)
# --------------------------------------------------------------------------------------------------
def test_agent_action_receipt_hash_byte_layout():
    ricardian = "aa" * 32
    agent_pk = "02" + "bb" * 32
    cpty_pk = "03" + "cc" * 32
    amount, tx_count, lock_time = 1234, 7, 1_700_000_000
    action_hash = "dd" * 32
    provenance_hash = "ee" * 32
    preimage = (
        bytes.fromhex(ricardian) + bytes.fromhex(agent_pk) + bytes.fromhex(cpty_pk)
        + amount.to_bytes(8, "little") + bytes.fromhex(action_hash) + bytes.fromhex(provenance_hash)
        + tx_count.to_bytes(8, "little") + lock_time.to_bytes(4, "little")
    )
    expected = hashlib.sha256(preimage).hexdigest()
    got = agent_action_receipt_hash(
        ricardian_hash=ricardian, agent_pubkey=agent_pk, counterparty_pubkey=cpty_pk,
        amount=amount, action_hash=action_hash, provenance_hash=provenance_hash,
        tx_count=tx_count, lock_time=lock_time)
    assert got == expected
    # GOLDEN VECTOR: this exact hex was produced by the scrypt-ts contract encoding
    # (int2ByteString + sha256) for the same inputs — verified in chain/ (see CONTRIBUTING / commit notes).
    # Guards against any silent drift between the Python verifier and the on-chain AgentTea byte layout.
    assert got == "cdc6a3e1b4bfd4ac931e25d31aa0309938d10900807cd403f74222ed2a00a33d"


def test_agent_action_receipt_hash_field_validation():
    with pytest.raises(ValueError):
        agent_action_receipt_hash(ricardian_hash="aa", agent_pubkey="02" + "bb" * 32,
                                  counterparty_pubkey="03" + "cc" * 32, amount=1, action_hash="dd" * 32,
                                  provenance_hash="ee" * 32, tx_count=0, lock_time=1)  # short ricardian


# --------------------------------------------------------------------------------------------------
# OP_RETURN parsing (no network)
# --------------------------------------------------------------------------------------------------
def _build_raw_tx(op_return_script_hex: str) -> str:
    """A minimal raw BSV tx with a single OP_RETURN output (and a dummy input)."""
    script = bytes.fromhex(op_return_script_hex)
    out = (0).to_bytes(8, "little") + bytes([len(script)]) + script
    raw = (
        bytes.fromhex("01000000")                         # version
        + bytes([1])                                      # vin count
        + bytes(32) + bytes.fromhex("ffffffff")           # prevout (zeros:ffffffff)
        + bytes([0]) + bytes.fromhex("ffffffff")          # empty scriptSig + sequence
        + bytes([1])                                      # vout count
        + out
        + bytes.fromhex("00000000")                       # locktime
    )
    return raw.hex()


def test_op_return_parse_standalone():
    tag = b"trinote/r1"
    mh = bytes.fromhex("12" * 32)
    rh = bytes.fromhex("34" * 32)
    script = (b"\x00\x6a" + bytes([len(tag)]) + tag + b"\x20" + mh + b"\x20" + rh).hex()
    raw = _build_raw_tx(script)
    tx = chain_read.parse_tx(raw)
    assert len(tx["outputs"]) == 1
    items = chain_read.op_return_items(tx["outputs"][0]["scriptHex"])
    assert items == [tag.hex(), mh.hex(), rh.hex()]
    hit = chain_read.find_op_return(tx["outputs"])
    assert hit[0] == 0 and hit[1][0] == tag.hex()


def test_op_return_parse_stateful():
    rh = bytes.fromhex("56" * 32)
    script = (b"\x00\x6a" + b"\x20" + rh).hex()
    raw = _build_raw_tx(script)
    tx = chain_read.parse_tx(raw)
    items = chain_read.op_return_items(tx["outputs"][0]["scriptHex"])
    assert items == [rh.hex()]


def test_parse_tx_inputs():
    rh = bytes.fromhex("56" * 32)
    script = (b"\x00\x6a" + b"\x20" + rh).hex()
    raw = _build_raw_tx(script)
    tx = chain_read.parse_tx(raw)
    assert tx["inputs"][0]["prevTxid"] == "00" * 32
    assert tx["inputs"][0]["vout"] == 0xffffffff


# --------------------------------------------------------------------------------------------------
# Stateful tamper detection, tar round-trip, and the ON-CHAIN verification layer (monkeypatched WoC)
# --------------------------------------------------------------------------------------------------
def test_tamper_stateful_identity_detected(tmp_path):
    rb = _receipt_bundle()
    identity, onchain = _stateful_inputs(rb["receipt"])
    pack_bundle(bundle=rb, onchain=onchain, out_dir=tmp_path / "st", identity=identity)
    ip = tmp_path / "st" / "identity.json"
    d = json.loads(ip.read_text()); d["ricardianHash"] = "00" * 32
    ip.write_text(json.dumps(d, sort_keys=True, separators=(",", ":")))
    out = verify_bundle(tmp_path / "st")
    assert not out["ok"]
    assert "file:identity.json" in {c["check"] for c in out["offline"]["checks"] if not c["ok"]}


def test_tar_roundtrip_stateful(tmp_path):
    rb = _receipt_bundle()
    identity, onchain = _stateful_inputs(rb["receipt"])
    res = pack_bundle(bundle=rb, onchain=onchain, out_dir=tmp_path / "s.tar.gz", identity=identity, as_tar=True)
    out = verify_bundle(res["path"])
    assert out["ok"] and out["kind"] == "stateful"


def _varint(n: int) -> bytes:
    if n < 0xfd:
        return bytes([n])
    if n <= 0xffff:
        return b"\xfd" + n.to_bytes(2, "little")
    if n <= 0xffffffff:
        return b"\xfe" + n.to_bytes(4, "little")
    return b"\xff" + n.to_bytes(8, "little")


def _raw_tx(inputs, outputs) -> str:
    """inputs: [(prev_txid_display_hex, vout)]; outputs: [(sats, script_hex)]."""
    b = bytes.fromhex("01000000") + _varint(len(inputs))
    for txid, vout in inputs:
        b += bytes.fromhex(txid)[::-1] + vout.to_bytes(4, "little") + _varint(0) + bytes.fromhex("ffffffff")
    b += _varint(len(outputs))
    for sats, script in outputs:
        sc = bytes.fromhex(script)
        b += sats.to_bytes(8, "little") + _varint(len(sc)) + sc
    return (b + bytes.fromhex("00000000")).hex()


def test_onchain_stateful_anchor_and_chain_walk(tmp_path, monkeypatch):
    rb = _receipt_bundle()
    identity, onchain = _stateful_inputs(rb["receipt"])
    genesis = identity["genesisTxid"]
    action_txid = onchain["actionTxid"]
    p2pkh = "76a914" + "00" * 20 + "88ac"                       # output[0]: recreated identity (not OP_RETURN)
    op_return = "006a20" + onchain["receiptHashOnChain"]        # output[1]: the stateful Third Entry
    action_raw = _raw_tx([(genesis, 0)], [(1, p2pkh), (0, op_return)])

    def fake_fetch(txid, network="main", timeout=20.0):
        if txid == action_txid:
            return action_raw
        raise chain_read.ChainReadError(f"unexpected fetch {txid}")

    monkeypatch.setattr(chain_read, "fetch_raw_tx", fake_fetch)
    pack_bundle(bundle=rb, onchain=onchain, out_dir=tmp_path / "so", identity=identity)
    res = verify_bundle(tmp_path / "so", onchain=True)
    assert res["onchain"]["ok"], [c for c in res["onchain"]["checks"] if not c["ok"]]
    assert res["ok"]
    names = {c["check"] for c in res["onchain"]["checks"]}
    assert {"onchain.found", "onchain.actionReceiptHash", "onchain.chainToGenesis"} <= names


def test_onchain_stateful_detects_wrong_chain(tmp_path, monkeypatch):
    rb = _receipt_bundle()
    identity, onchain = _stateful_inputs(rb["receipt"])
    action_txid = onchain["actionTxid"]
    op_return = "006a20" + onchain["receiptHashOnChain"]
    # input[0] points at an UNRELATED tx, not the identity's genesis → chain walk must fail.
    action_raw = _raw_tx([("99" * 32, 0)], [(1, "76a914" + "00" * 20 + "88ac"), (0, op_return)])
    # The unrelated parent has no inputs (coinbase-like), so the walk terminates cleanly (never reaches genesis).
    unrelated_raw = _raw_tx([], [(1, "76a914" + "11" * 20 + "88ac")])

    def fake_fetch(txid, network="main", timeout=20.0):
        if txid == action_txid:
            return action_raw
        if txid == "99" * 32:
            return unrelated_raw
        raise chain_read.ChainReadError(f"unexpected fetch {txid}")

    monkeypatch.setattr(chain_read, "fetch_raw_tx", fake_fetch)
    pack_bundle(bundle=rb, onchain=onchain, out_dir=tmp_path / "sw", identity=identity)
    res = verify_bundle(tmp_path / "sw", onchain=True)
    assert not res["ok"]
    assert "onchain.chainToGenesis" in {c["check"] for c in res["onchain"]["checks"] if not c["ok"]}


# --------------------------------------------------------------------------------------------------
# Robustness: corrupt archive + out-dir guard
# --------------------------------------------------------------------------------------------------
def test_corrupt_tar_raises_bundle_error(tmp_path):
    p = tmp_path / "bad.tar.gz"
    p.write_bytes(b"this is not a tar archive")
    with pytest.raises(BundleError):
        verify_bundle(p)


def test_pack_rejects_file_out_dir(tmp_path):
    rb = _receipt_bundle()
    receipt = rb["receipt"]
    onchain = {"kind": "standalone", "network": "main", "tag": "trinote/r1", "txid": "ab" * 32,
               "modelHash": receipt["modelHash"], "receiptHash": receipt["receiptHash"]}
    f = tmp_path / "afile"; f.write_text("x")
    with pytest.raises(BundleError):
        pack_bundle(bundle=rb, onchain=onchain, out_dir=f)


# --------------------------------------------------------------------------------------------------
# rawTx in the bundle: offline txid == hash256(rawTx) integrity check
# --------------------------------------------------------------------------------------------------
def _standalone_op_return(model_hash: str, receipt_hash: str) -> str:
    tag = b"trinote/r1"
    return (b"\x00\x6a" + bytes([len(tag)]) + tag + b"\x20" + bytes.fromhex(model_hash)
            + b"\x20" + bytes.fromhex(receipt_hash)).hex()


def test_verify_standalone_with_rawtx(tmp_path):
    rb = _receipt_bundle()
    receipt = rb["receipt"]
    script = _standalone_op_return(receipt["modelHash"], receipt["receiptHash"])
    raw = _raw_tx([("11" * 32, 0)], [(0, script)])
    onchain = {"kind": "standalone", "network": "main", "tag": "trinote/r1", "txid": txid_of(raw),
               "modelHash": receipt["modelHash"], "receiptHash": receipt["receiptHash"], "rawTx": raw}
    pack_bundle(bundle=rb, onchain=onchain, out_dir=tmp_path / "rt")
    out = verify_bundle(tmp_path / "rt")
    assert out["ok"], [c for c in out["offline"]["checks"] if not c["ok"]]
    assert "onchain.txidMatchesRawTx" in {c["check"] for c in out["offline"]["checks"]}


def test_verify_detects_rawtx_txid_mismatch(tmp_path):
    rb = _receipt_bundle()
    receipt = rb["receipt"]
    raw = _raw_tx([("11" * 32, 0)], [(0, _standalone_op_return(receipt["modelHash"], receipt["receiptHash"]))])
    onchain = {"kind": "standalone", "network": "main", "tag": "trinote/r1", "txid": "00" * 32,  # wrong txid
               "modelHash": receipt["modelHash"], "receiptHash": receipt["receiptHash"], "rawTx": raw}
    pack_bundle(bundle=rb, onchain=onchain, out_dir=tmp_path / "rtbad")
    out = verify_bundle(tmp_path / "rtbad")
    assert not out["ok"]
    assert "onchain.txidMatchesRawTx" in {c["check"] for c in out["offline"]["checks"] if not c["ok"]}


# --------------------------------------------------------------------------------------------------
# Off-chain transaction log (receipts/txlog.py)
# --------------------------------------------------------------------------------------------------
def test_txid_of_matches_double_sha256():
    raw = _raw_tx([("11" * 32, 0)], [(0, "006a0464656d6f")])
    assert txid_of(raw) == double_sha256(bytes.fromhex(raw))[::-1].hex()


def test_txlog_records_and_self_verifies(tmp_path):
    raw = _raw_tx([("11" * 32, 0)], [(0, "006a0464656d6f")])
    txid = txid_of(raw)
    rec = log_transaction(tmp_path / "tx.log", {"txid": txid, "rawTx": raw, "status": "broadcast",
                                                "network": "main", "fee": 200, "sizeBytes": len(raw) // 2},
                          kind="standalone")
    assert rec is not None and rec["txidVerified"] is True
    rows = read_tx_log(tmp_path / "tx.log")
    assert len(rows) == 1 and rows[0]["kind"] == "standalone" and rows[0]["txid"] == txid


def test_txlog_flags_txid_mismatch():
    raw = _raw_tx([("11" * 32, 0)], [(0, "006a0464656d6f")])
    rec = tx_record({"txid": "00" * 32, "rawTx": raw}, kind="standalone")
    assert rec["txidVerified"] is False


def test_log_transaction_skips_synthetic_local_log(tmp_path):
    # The LogBroadcastBackend's synthetic 'log:' id is not a real tx — nothing to record.
    assert log_transaction(tmp_path / "tx.log", {"txid": "log:abc", "status": "logged"}, kind="standalone") is None
    assert read_tx_log(tmp_path / "tx.log") == []


# --------------------------------------------------------------------------------------------------
# R1/R12/R13: bounded ingestion of UNTRUSTED bundle archives (verify is the third-party-audit front door)
# --------------------------------------------------------------------------------------------------
def _standalone_bundle_dir(tmp_path, name="b"):
    rb = _receipt_bundle()
    receipt = rb["receipt"]
    onchain = {"kind": "standalone", "network": "main", "tag": "trinote/r1", "txid": "ab" * 32,
               "modelHash": receipt["modelHash"], "receiptHash": receipt["receiptHash"]}
    pack_bundle(bundle=rb, onchain=onchain, out_dir=tmp_path / name)
    return tmp_path / name


def _tar_gz_with(members: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_bundle_rejects_decompression_bomb_member(tmp_path):
    # 9 MiB member that compresses to ~KBs — must be refused BEFORE full expansion (R1).
    big = b"{}" + b" " * (9 * 1024 * 1024)
    p = tmp_path / "bomb.tar.gz"
    p.write_bytes(_tar_gz_with({"bundle/manifest.json": big}))
    assert len(p.read_bytes()) < 100 * 1024            # the archive itself is tiny
    with pytest.raises(BundleError):
        load_bundle(p)


def test_bundle_rejects_member_flood(tmp_path):
    members = {f"bundle/f{i}.json": b"{}" for i in range(200)}
    members["bundle/manifest.json"] = b"{}"
    p = tmp_path / "flood.tar.gz"
    p.write_bytes(_tar_gz_with(members))
    with pytest.raises(BundleError):
        load_bundle(p)


def test_bundle_rejects_malformed_json_member(tmp_path):
    p = tmp_path / "bad.tar.gz"
    p.write_bytes(_tar_gz_with({"bundle/manifest.json": b"{not json"}))
    with pytest.raises(BundleError):       # guarded parse → BundleError, not a raw JSONDecodeError (R13)
        load_bundle(p)


def test_bundle_rejects_json_recursion_bomb(tmp_path):
    bomb = b"[" * 100_000 + b"]" * 100_000   # < 8 MiB, passes size cap, blows the json recursion limit
    p = tmp_path / "rec.tar.gz"
    p.write_bytes(_tar_gz_with({"bundle/manifest.json": bomb}))
    with pytest.raises(BundleError):       # RecursionError caught and re-raised as BundleError (R13)
        load_bundle(p)


def test_legitimate_bundle_still_loads_under_caps(tmp_path):
    # The caps must not regress a normal bundle.
    d = _standalone_bundle_dir(tmp_path, "ok")
    out = verify_bundle(d)
    assert out["ok"], [c for c in out["offline"]["checks"] if not c["ok"]]


# --------------------------------------------------------------------------------------------------
# #10: the chain artifact's sampler mode/seed/schema must match the receipt it claims to mark
# --------------------------------------------------------------------------------------------------
def test_chain_artifact_seed_tamper_detected(tmp_path):
    from trinote.receipts.canonical import canonical_bytes, commit
    d = _standalone_bundle_dir(tmp_path, "ca")
    # A motivated attacker tampers the chain artifact's seed AND re-commits its digest + bundleHash, so the
    # file-digest and bundleHash checks still pass — only the cross-check against the receipt catches it.
    cap = d / "chain-artifact.json"
    art = json.loads(cap.read_text()); art["seed"] = 999_999
    cap.write_bytes(canonical_bytes(art))
    mp = d / "manifest.json"; m = json.loads(mp.read_text())
    m["files"]["chain-artifact.json"] = commit(art)
    m["bundleHash"] = commit({k: v for k, v in m.items() if k != "bundleHash"})
    mp.write_bytes(canonical_bytes(m))
    out = verify_bundle(d)
    assert not out["ok"]
    names = {c["check"] for c in out["offline"]["checks"] if not c["ok"]}
    assert "chainArtifact.seed" in names
    assert "file:chain-artifact.json" not in names   # digest was re-committed, so ONLY the cross-check fires


# --------------------------------------------------------------------------------------------------
# #9 / #11: the re-exec layer must fold in artifactBoundOk (no manifest.modelHash fallback) and signatures
# --------------------------------------------------------------------------------------------------
def test_reexec_requires_artifact_binding_no_manifest_fallback(tmp_path, monkeypatch):
    import trinote.receipts.verify as rv
    d = _standalone_bundle_dir(tmp_path, "rx")
    captured = {}

    def fake_verify_receipt(bundle, **kw):
        captured.update(kw)
        bound = kw.get("model_digest") is not None
        return {"structuralOk": True, "reexecOk": True, "artifactBoundOk": bound,
                "artifactBindingOk": True, "modelHashMatch": bound, "signatureOk": None,
                "reexec": {"ok": True, "strategy": "greedy", "checked": 3}, "ok": bound}

    monkeypatch.setattr(rv, "verify_receipt", fake_verify_receipt)
    # No real digest of the loaded weights → binding not provable → layer fails CLOSED (was tautologically ok).
    out = verify_bundle(d, reexec=True, model=object())
    assert out["reexec"]["ok"] is False and not out["ok"]
    assert captured.get("model_digest") is None        # must NOT fall back to manifest.modelHash
    # With a real digest of the loaded weights → binding holds → passes.
    out2 = verify_bundle(d, reexec=True, model=object(), model_digest="deadbeef")
    assert out2["reexec"]["ok"] is True and out2["ok"]


def test_reexec_layer_fails_on_invalid_signature(tmp_path, monkeypatch):
    import trinote.receipts.verify as rv
    d = _standalone_bundle_dir(tmp_path, "sig")

    def fake_verify_receipt(bundle, **kw):
        return {"structuralOk": True, "reexecOk": True, "artifactBoundOk": True,
                "artifactBindingOk": True, "modelHashMatch": True, "signatureOk": False,
                "reexec": {"ok": True, "strategy": "greedy", "checked": 3}, "ok": False}

    monkeypatch.setattr(rv, "verify_receipt", fake_verify_receipt)
    out = verify_bundle(d, reexec=True, model=object(), model_digest="x")
    assert out["reexec"]["ok"] is False        # a bad signature now fails the layer (was ignored)


@pytest.mark.parametrize(
    ("signature_fields", "expected_ok"),
    [
        ({}, False),
        ({
            "sigModelOk": True,
            "sigModelAuthenticated": True,
        }, False),
        ({
            "sigModelOk": True,
            "sigCounterpartyOk": True,
            "sigModelAuthenticated": True,
            "sigCounterpartyAuthenticated": True,
        }, True),
    ],
)
def test_reexec_signature_pinning_requires_both_authenticated_signatures(
    tmp_path, monkeypatch, signature_fields, expected_ok
):
    import trinote.receipts.verify as rv

    d = _standalone_bundle_dir(tmp_path, "strict-signatures")

    def fake_verify_receipt(bundle, **kw):
        return {
            "structuralOk": True,
            "reexecOk": True,
            "artifactBoundOk": True,
            "artifactBindingOk": True,
            "modelHashMatch": True,
            "signatureOk": (
                True if signature_fields.get("sigModelOk") is True
                and signature_fields.get("sigCounterpartyOk") is True else None
            ),
            "reexec": {"ok": True, "strategy": "greedy", "checked": 3},
            "ok": True,
            **signature_fields,
        }

    monkeypatch.setattr(rv, "verify_receipt", fake_verify_receipt)
    out = verify_bundle(
        d,
        reexec=True,
        model=object(),
        model_digest="x",
        model_pubkey="02" + "11" * 32,
        counterparty_pubkey="03" + "22" * 32,
    )
    assert out["reexec"]["ok"] is expected_ok
    assert out["reexec"]["signaturePinned"] is expected_ok
    assert out["ok"] is expected_ok


def test_receipt_verifier_treats_missing_required_ec_signatures_as_invalid():
    from trinote.receipts.verify import verify_receipt

    bundle = _receipt_bundle()
    for key in (
        "sigModel", "sigModelPubKey", "sigModelKeyId",
        "sigCounterparty", "sigCounterpartyPubKey", "sigCounterpartyKeyId",
    ):
        bundle["receipt"].pop(key, None)
    result = verify_receipt(
        bundle,
        model_pubkey="02" + "11" * 32,
        counterparty_pubkey="03" + "22" * 32,
    )
    assert result["sigModelOk"] is False
    assert result["sigCounterpartyOk"] is False
    assert result["signatureOk"] is False


@pytest.mark.parametrize("bad_pin", ["", "02" + "AA" * 32, "04" + "11" * 32, "02ff"])
def test_receipt_verifier_rejects_noncanonical_identity_pins(bad_pin):
    from trinote.receipts.verify import verify_receipt

    result = verify_receipt(_receipt_bundle(), model_pubkey=bad_pin)
    assert result["ok"] is False
    assert "canonical lowercase compressed" in result["error"]


def test_reexec_reports_one_requested_pin_without_claiming_both(tmp_path, monkeypatch):
    import trinote.receipts.verify as rv

    d = _standalone_bundle_dir(tmp_path, "one-pin")

    def fake_verify_receipt(bundle, **kw):
        return {
            "structuralOk": True,
            "reexecOk": True,
            "artifactBoundOk": True,
            "artifactBindingOk": True,
            "modelHashMatch": True,
            "signatureOk": True,
            "sigModelOk": True,
            "sigModelAuthenticated": True,
            "reexec": {"ok": True, "strategy": "greedy", "checked": 3},
            "ok": True,
        }

    monkeypatch.setattr(rv, "verify_receipt", fake_verify_receipt)
    out = verify_bundle(
        d,
        reexec=True,
        model=object(),
        model_digest="x",
        model_pubkey="02" + "11" * 32,
    )
    rx = out["reexec"]
    assert out["ok"] is True
    assert rx["requestedSignaturePinsAuthenticated"] is True
    assert rx["signaturePinned"] is False
    assert next(c for c in rx["checks"] if c["check"] == "signaturePins")["ok"] is True


@pytest.mark.parametrize(
    ("key_field", "source_field"),
    [
        ("model_pubkey", "model_pin_source"),
        ("counterparty_pubkey", "counterparty_pin_source"),
    ],
)
def test_reexec_helper_rejects_pin_source_without_a_pin(key_field, source_field):
    from trinote.bundle.verify import _verify_reexec

    arguments = {key_field: None, source_field: "caller"}
    with pytest.raises(ValueError, match="pin and pin source"):
        _verify_reexec({}, object(), "ab" * 32, **arguments)


def test_reexec_stateful_defaults_pinned_identity(tmp_path, monkeypatch):
    import trinote.receipts.verify as rv
    rb = _receipt_bundle()
    identity, onchain = _stateful_inputs(rb["receipt"])
    pack_bundle(bundle=rb, onchain=onchain, out_dir=tmp_path / "sp", identity=identity)
    captured = {}

    def fake_verify_receipt(bundle, **kw):
        captured.update(kw)
        return {"structuralOk": True, "reexecOk": True, "artifactBoundOk": True,
                "artifactBindingOk": True, "modelHashMatch": True, "signatureOk": True,
                "sigModelOk": True, "sigCounterpartyOk": True,
                "sigModelAuthenticated": True, "sigCounterpartyAuthenticated": True,
                "reexec": {"ok": True, "strategy": "greedy", "checked": 3}, "ok": True}

    monkeypatch.setattr(rv, "verify_receipt", fake_verify_receipt)
    result = verify_bundle(tmp_path / "sp", reexec=True, model=object(), model_digest="x")
    assert captured.get("model_pubkey") == identity["agentPubKey"]
    assert captured.get("counterparty_pubkey") == identity["counterpartyPubKey"]
    reexec = result["reexec"]
    assert reexec["modelSignaturePinSource"] == "bundle-identity"
    assert reexec["counterpartySignaturePinSource"] == "bundle-identity"
    assert reexec["modelSignaturePinRequested"] is False
    assert reexec["counterpartySignaturePinRequested"] is False
    assert reexec["requestedSignaturePinsAuthenticated"] is None
    assert reexec["effectiveSignaturePinsAuthenticated"] is True
    assert reexec["signaturePinned"] is False


# --------------------------------------------------------------------------------------------------
# R15 / #12 / R16: ledger truncation detection, EC canonical-encoding enforcement, bundle ASCII guard.
# --------------------------------------------------------------------------------------------------
def test_ledger_verify_chain_detects_truncation_with_expected_head(tmp_path):
    """R15: a tail-truncated ledger is internally consistent (passes the plain walk), so verify_chain now
    accepts an expected_head/expected_count from a verifier who pinned it out-of-band, catching the drop."""
    from trinote.receipts import LocalLedger
    led = LocalLedger(tmp_path / "ledger.jsonl")
    for i in range(3):
        led.record({"receiptHash": f"{i:064x}", "modelHash": "ab" * 32}, ts="t")
    full = led.verify_chain()
    assert full["ok"] and full["count"] == 3
    head, count = full["head"], full["count"]
    # drop the last line — still internally consistent
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    (tmp_path / "ledger.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    assert led.verify_chain()["ok"] is True                      # plain walk cannot tell
    bad = led.verify_chain(expected_head=head, expected_count=count)
    assert bad["ok"] is False and "reason" in bad               # pinned head/count catches it


def test_verify_ec_rejects_noncanonical_encodings():
    """#12: verify_ec enforces the canonical wire form — a same-key UNCOMPRESSED re-encoding and a high-S
    malleated (but mathematically valid) signature are both rejected; the canonical signature still verifies."""
    import hashlib
    from ecdsa import SECP256k1, VerifyingKey
    from trinote.receipts import ec_keygen
    from trinote.receipts.signing_ec import verify_ec
    mk = ec_keygen(secret_hex="ab" * 32)
    payload = b"canonical-encoding-test"
    sig = mk.sign(payload)
    assert verify_ec(payload, sig) is True
    scheme, pub_hex, sig_hex = sig.split(":")
    # same key, uncompressed (65-byte) encoding → rejected (must be 33-byte compressed)
    vk = VerifyingKey.from_string(bytes.fromhex(pub_hex), curve=SECP256k1, hashfunc=hashlib.sha256)
    uncompressed = vk.to_string("uncompressed").hex()
    assert verify_ec(payload, f"{scheme}:{uncompressed}:{sig_hex}") is False
    # high-S malleation: (r, n-s) is a valid ECDSA sig but violates low-S policy → rejected
    raw = bytes.fromhex(sig_hex)
    r_int = int.from_bytes(raw[:32], "big")
    s_int = int.from_bytes(raw[32:], "big")
    high = (r_int.to_bytes(32, "big") + (SECP256k1.order - s_int).to_bytes(32, "big")).hex()
    assert verify_ec(payload, f"{scheme}:{pub_hex}:{high}") is False


def test_pack_bundle_rejects_non_ascii_label(tmp_path):
    """R16: bundleHash commits free-text modelLabel under ensure_ascii=False, so a non-ASCII label would make
    the digest implementation-dependent. Reject it; an ASCII label still packs."""
    rb = _receipt_bundle()
    receipt = rb["receipt"]
    onchain = {"kind": "standalone", "network": "main", "tag": "trinote/r1", "txid": "ab" * 32,
               "modelHash": receipt["modelHash"], "receiptHash": receipt["receiptHash"]}
    with pytest.raises(BundleError):
        pack_bundle(bundle=rb, onchain=onchain, out_dir=tmp_path / "nonascii", model_label="Bonsaï-café-モデル")
    res = pack_bundle(bundle=rb, onchain=onchain, out_dir=tmp_path / "ascii", model_label="ATLAS-Bonsai-8B")
    assert res["bundleHash"]


# --------------------------------------------------------------------------------------------------
# Local (BSV-off) bundle: kind="local" (no on-chain descriptor), human-readable transcript, verifiable
# offline by commitments + bundleHash; reproducible by re-execution (replay test lives in the smoke suite).
# --------------------------------------------------------------------------------------------------
def test_pack_verify_local_bundle_offline(tmp_path):
    rb = _receipt_bundle()
    transcript = {"prompt": "how many letter r's are there in \"strawberry\"",
                  "output": "Answer: 3", "modelLabel": "ATLAS-Notarized-Bonsai-8B",
                  "sampler": "greedy", "seed": 0}
    res = pack_bundle(bundle=rb, onchain=None, out_dir=tmp_path / "loc", transcript=transcript)
    assert res["manifest"]["kind"] == "local"
    assert "onchain.json" not in res["manifest"]["files"]
    assert "transcript.json" in res["manifest"]["files"] and "transcript.md" in res["manifest"]["files"]
    out = verify_bundle(tmp_path / "loc")
    assert out["ok"], [c for c in out["offline"]["checks"] if not c["ok"]]
    assert out["kind"] == "local" and "onchain" not in out          # no on-chain layer for a local bundle
    # the plaintext is actually in the human-readable transcript
    md = (tmp_path / "loc" / "transcript.md").read_text()
    assert "strawberry" in md and "Answer: 3" in md
    # tampering the transcript is caught by the committed file digest (bundleHash covers transcript.md)
    (tmp_path / "loc" / "transcript.md").write_text("tampered")
    bad = verify_bundle(tmp_path / "loc")
    assert not bad["ok"]
    assert "file:transcript.md" in {c["check"] for c in bad["offline"]["checks"] if not c["ok"]}


def test_local_bundle_fails_when_onchain_verification_is_required(tmp_path):
    rb = _receipt_bundle()
    pack_bundle(bundle=rb, onchain=None, out_dir=tmp_path / "local-onchain-required")
    out = verify_bundle(tmp_path / "local-onchain-required", onchain=True)
    assert out["ok"] is False
    assert out["onchain"]["ok"] is False
    assert out["onchain"]["skipped"] is True
    assert out["onchain"]["checks"] == [{
        "check": "onchain.localBundle",
        "ok": False,
        "detail": "on-chain verification was requested, but this local bundle has no BSV third entry",
    }]


def test_local_bundle_tar_roundtrip_includes_transcript(tmp_path):
    rb = _receipt_bundle()
    transcript = {"prompt": "p", "output": "o", "modelLabel": "m"}
    res = pack_bundle(bundle=rb, onchain=None, out_dir=tmp_path / "loc.tar.gz",
                      transcript=transcript, as_tar=True)
    loaded = load_bundle(res["path"])
    assert loaded["manifest"]["kind"] == "local"
    assert "transcript.md" in loaded["raw"] and "transcript.json" in loaded["obj"]   # .md in raw, .json parsed
    out = verify_bundle(res["path"])
    assert out["ok"], [c for c in out["offline"]["checks"] if not c["ok"]]


def test_signing_keys_load_or_generate(tmp_path, monkeypatch):
    """Real secp256k1 receipt signing keys: generated on first use under $BONSAI_NOTARY_HOME/keys (chmod 0600),
    idempotent on reload, and distinct model/counterparty identities."""
    import os
    import stat
    monkeypatch.setenv("BONSAI_NOTARY_HOME", str(tmp_path))
    from trinote.infer_int.bonsai_runtime import load_or_generate_signing_keys
    from trinote.notary_paths import model_key_default, counterparty_key_default
    mk, ck = load_or_generate_signing_keys()
    assert mk.public_hex and ck.public_hex and mk.public_hex != ck.public_hex
    assert os.path.exists(model_key_default()) and os.path.exists(counterparty_key_default())
    assert stat.S_IMODE(os.stat(model_key_default()).st_mode) == 0o600
    mk2, ck2 = load_or_generate_signing_keys()              # idempotent: reload, do not regenerate
    assert mk2.secret_hex == mk.secret_hex and ck2.secret_hex == ck.secret_hex


def test_signing_key_is_created_private_without_postwrite_chmod(tmp_path, monkeypatch):
    """The initial file mode must be safe even if a pathname chmod is unavailable."""
    import os
    import stat
    from trinote.receipts.signing_ec import ECKey

    def chmod_unavailable(*_args, **_kwargs):
        raise OSError("simulated chmod failure")

    monkeypatch.setattr(os, "chmod", chmod_unavailable)
    previous = os.umask(0)
    try:
        path = tmp_path / "issuer.key.json"
        ECKey.from_secret_hex("33" * 32, label="issuer").save(path)
    finally:
        os.umask(previous)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_concurrent_first_use_returns_one_persisted_identity(tmp_path, monkeypatch):
    """Concurrent installers must all return the one key that wins the atomic publication."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from trinote.receipts.signing_ec import ECKey

    original_generate = ECKey.generate.__func__
    both_generated = Barrier(2)

    def synchronized_generate(cls, *, label="", secret_hex=None):
        key = original_generate(cls, label=label, secret_hex=secret_hex)
        both_generated.wait(timeout=5)
        return key

    monkeypatch.setattr(ECKey, "generate", classmethod(synchronized_generate))
    path = tmp_path / "shared.key.json"
    with ThreadPoolExecutor(max_workers=2) as pool:
        keys = list(pool.map(lambda _: ECKey.load_or_generate(path, label="shared"), range(2)))

    persisted = ECKey.from_json(json.loads(path.read_text()))
    assert keys[0].secret_hex == keys[1].secret_hex == persisted.secret_hex
    assert not list(tmp_path.glob(".shared.key.json.*.tmp"))


def test_receipt_ec_signature_third_party_verifiable():
    """A receipt signed with real EC keys is verifiable from the committed PUBLIC key alone (no shared secret)
    — the authenticity property the demo HMAC keys lacked. (Keys passed explicitly to bypass the pytest demo path.)"""
    from trinote.receipts import ec_keygen, build_receipt
    from trinote.receipts.signing_ec import SCHEME_EC, verify_ec
    from trinote.receipts.canonical import canonical_bytes
    mk = ec_keygen(label="model"); ck = ec_keygen(label="cp")
    bundle = build_receipt(model_hash="ab" * 32, input_ids=[1, 2, 3], output_ids=[4, 5],
                           sampler=_sampler(), model_key=mk, counterparty_key=ck,
                           model_label="t", artifact_digest="ab" * 32, fp_frac_bits=16)
    r = bundle["receipt"]
    assert r["sigModel"].startswith(SCHEME_EC) and r["sigCounterparty"].startswith(SCHEME_EC)
    assert r["sigModelPubKey"] == mk.public_hex          # the committed PUBLIC key, no secret in the receipt
    # An external verifier reconstructs the signed message from the receipt and checks it with the public key
    # alone (message forms mirror receipt.py): model signs {modelHash,input,output,traceCommit}.
    model_msg = canonical_bytes({"modelHash": r["modelHash"], "inputCommit": r["inputCommit"],
                                 "outputCommit": r["outputCommit"], "traceCommit": r["trace"]["traceCommit"]})
    cp_msg = canonical_bytes({"modelHash": r["modelHash"], "inputCommit": r["inputCommit"],
                              "outputCommit": r["outputCommit"]})
    assert verify_ec(model_msg, r["sigModel"], expected_pubkey_hex=mk.public_hex)
    assert verify_ec(cp_msg, r["sigCounterparty"], expected_pubkey_hex=ck.public_hex)
    assert not verify_ec(model_msg, r["sigModel"], expected_pubkey_hex=ck.public_hex)   # wrong key must fail
