"""Verify a receipt bundle — three independent, separately-reported layers.

  1. OFFLINE  (always)      stdlib only, no network, no weights: recompute every file digest, the bundleHash,
                            the receiptHash, the input/output/trace commitments, the chain artifact, and (for
                            a stateful bundle) the AgentTea action hash from its committed fields.
  2. ON-CHAIN (--onchain)   fetch the third-entry tx from WhatsOnChain, parse the OP_RETURN, and bind it to
                            the receipt; for a stateful action also walk input[0] back to the genesis tx.
  3. RE-EXEC  (--reexec)    load the model and re-run the bit-exact reference engine (receipts.verify_receipt).

A bundle is OK iff every requested layer passes. Each layer is reported even when a later one is skipped, so a
consumer can choose how much trust they need: offline alone proves internal consistency; on-chain proves the
mark is published + immutable; re-exec proves the model actually produced the output.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

from ..receipts.canonical import canonical_bytes, commit, token_commit
from ..receipts.receipt import receipt_hash
from ..receipts.emit import chain_artifact
from ..hashing.sha import txid_of
from .pack import BUNDLE_SCHEMA, BundleError
from .stateful import agent_action_receipt_hash
from . import chain_read


# Resource caps on UNTRUSTED bundle ingestion. `verify`/`inspect` are the third-party-audit entry points,
# so the adversary is exactly the producer who hands you a bundle. A legitimate bundle is a handful of small
# JSON files; these caps sit orders of magnitude above real use while defusing (R1) decompression bombs — we
# read at most _MAX_MEMBER_BYTES+1 from each member's *decompressing* stream, so a high-ratio archive never
# fully expands; (R12) member-count floods; and (R13) json/recursion bombs (guarded parse below).
_MAX_MEMBERS = 64
_MAX_MEMBER_BYTES = 8 * 1024 * 1024
_MAX_TOTAL_BYTES = 16 * 1024 * 1024


def _read_capped(fileobj, name: str, budget: list) -> bytes:
    """Read at most the per-member cap from a (possibly decompressing) stream; debit the shared total budget.

    Reading cap+1 bytes from a lazily-decompressing tar member means a bomb only ever expands by the cap,
    not to its full declared size — the materialization, not the header, is what's bounded.
    """
    data = fileobj.read(_MAX_MEMBER_BYTES + 1) if fileobj is not None else b""
    if len(data) > _MAX_MEMBER_BYTES:
        raise BundleError(f"bundle member {name!r} exceeds {_MAX_MEMBER_BYTES} bytes "
                          "(refusing to expand — possible decompression bomb)")
    budget[0] -= len(data)
    if budget[0] < 0:
        raise BundleError(f"bundle total decompressed size exceeds {_MAX_TOTAL_BYTES} bytes")
    return data


def load_bundle(path: str | Path) -> dict:
    """Load a bundle dir or .tar.gz into {root, manifest, raw:{name:bytes}, obj:{name:dict}}.

    Verification covers exactly the files declared in `manifest.files` (whose digests the `bundleHash`
    commits). Undeclared files cannot affect the `bundleHash` and are ignored; for a stateful bundle,
    `identity.json` IS declared, so tampering with it is caught. Ingestion is bounded (size/count/parse) —
    see the `_MAX_*` caps — because this runs on attacker-supplied archives BEFORE any commitment check.
    """
    p = Path(path)
    raw: dict[str, bytes] = {}
    budget = [_MAX_TOTAL_BYTES]
    if p.is_dir():
        root = str(p)
        for child in sorted(p.iterdir()):
            if not child.is_file():               # ingest ALL declared files (manifest covers .md transcripts too)
                continue
            if len(raw) >= _MAX_MEMBERS:
                raise BundleError(f"bundle has more than {_MAX_MEMBERS} members")
            if child.stat().st_size > _MAX_MEMBER_BYTES:
                raise BundleError(f"bundle member {child.name!r} exceeds {_MAX_MEMBER_BYTES} bytes")
            with child.open("rb") as fh:
                raw[child.name] = _read_capped(fh, child.name, budget)
    elif p.is_file() and (p.name.endswith(".tar.gz") or p.name.endswith(".tgz") or p.suffix == ".tar"):
        root = str(p)
        try:
            with tarfile.open(p, "r:*") as tar:
                for m in tar:                     # stream members lazily — never materialize getmembers()
                    if not m.isfile():
                        continue
                    name = Path(m.name).name      # ignore any path components (traversal-safe; we never write to disk)
                    if name in raw:
                        raise BundleError(f"duplicate bundle member {name!r} in archive {p}")
                    if len(raw) >= _MAX_MEMBERS:
                        raise BundleError(f"bundle archive has more than {_MAX_MEMBERS} members")
                    raw[name] = _read_capped(tar.extractfile(m), name, budget)
        except tarfile.TarError as exc:
            raise BundleError(f"corrupt bundle archive {p}: {exc}") from exc
    else:
        raise FileNotFoundError(f"not a bundle dir or .tar.gz: {path}")
    if "manifest.json" not in raw:
        raise FileNotFoundError(f"bundle has no manifest.json: {path}")
    # Parse only the JSON members into `obj`; non-JSON declared files (e.g. transcript.md) stay in `raw` so
    # their bundleHash digest is still verified, but they are never json.loads'd.
    try:
        obj = {name: json.loads(b.decode("utf-8")) for name, b in raw.items() if name.endswith(".json")}
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise BundleError(f"bundle member is not valid JSON: {exc}") from exc
    return {"root": root, "manifest": obj["manifest.json"], "raw": raw, "obj": obj}


def _check(checks: list, name: str, ok: bool, detail: str = "") -> bool:
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


def _verify_offline(loaded: dict) -> dict:
    checks: list[dict] = []
    manifest = loaded["manifest"]
    raw, obj = loaded["raw"], loaded["obj"]

    _check(checks, "manifest.schema", manifest.get("schema") == BUNDLE_SCHEMA,
           f"{manifest.get('schema')!r}")

    # bundleHash commits the manifest (schema/kind/metadata + every file digest).
    body = {k: v for k, v in manifest.items() if k != "bundleHash"}
    _check(checks, "bundleHash", commit(body) == manifest.get("bundleHash"), manifest.get("bundleHash", ""))

    # Every declared file is present and its on-disk bytes hash to the recorded digest.
    files = manifest.get("files", {})
    for name, want in files.items():
        present = name in raw
        got = commit_bytes(raw[name]) if present else None
        _check(checks, f"file:{name}", present and got == want,
               "missing" if not present else ("ok" if got == want else f"{got} != {want}"))

    receipt = obj.get("receipt.json")
    preimage = obj.get("preimage.json")
    if not isinstance(receipt, dict) or not isinstance(preimage, dict):
        _check(checks, "receipt+preimage", False, "missing receipt.json or preimage.json")
        return {"ok": False, "checks": checks}

    # receiptHash recomputed over the receipt body (everything except receiptHash itself).
    rh = receipt.get("receiptHash")
    _check(checks, "receiptHash", receipt_hash(receipt) == rh, rh or "")
    _check(checks, "manifest.receiptHash", manifest.get("receiptHash") == rh, manifest.get("receiptHash", ""))
    _check(checks, "manifest.modelHash", manifest.get("modelHash") == receipt.get("modelHash"),
           manifest.get("modelHash", ""))

    # Commitments recomputed from the off-chain preimage ids.
    _check(checks, "inputCommit", token_commit(preimage.get("inputIds", [])) == receipt.get("inputCommit"),
           receipt.get("inputCommit", ""))
    _check(checks, "outputCommit", token_commit(preimage.get("outputIds", [])) == receipt.get("outputCommit"),
           receipt.get("outputCommit", ""))
    trace = receipt.get("trace") or {}
    if "traceCommit" in trace:
        tbody = {k: v for k, v in trace.items() if k != "traceCommit"}
        _check(checks, "traceCommit", commit(tbody) == trace.get("traceCommit"), trace.get("traceCommit", ""))

    # Chain artifact must match the receipt it claims to mark — in FULL. Beyond modelHash/receiptHash/tag,
    # the v2 artifact also carries the reproducibility nonce (samplerMode + seed) and its schema; checking
    # only the first three let a bundle publish a chain artifact whose declared draw mode/seed disagreed with
    # the receipt's authoritative committed seed and still pass offline verify (R-finding #10).
    art = obj.get("chain-artifact.json") or {}
    want_art = chain_artifact(receipt)
    _check(checks, "chainArtifact.modelHash", art.get("modelHash") == receipt.get("modelHash"),
           art.get("modelHash", ""))
    _check(checks, "chainArtifact.receiptHash", art.get("receiptHash") == rh, art.get("receiptHash", ""))
    _check(checks, "chainArtifact.tag", art.get("tag") == want_art["tag"], art.get("tag", ""))
    _check(checks, "chainArtifact.schema", art.get("schema") == want_art["schema"],
           f"{art.get('schema')!r} != {want_art['schema']!r}")
    _check(checks, "chainArtifact.samplerMode", art.get("samplerMode") == want_art["samplerMode"],
           f"{art.get('samplerMode')!r} != {want_art['samplerMode']!r}")
    _check(checks, "chainArtifact.seed", art.get("seed") == want_art["seed"],
           f"{art.get('seed')!r} != {want_art['seed']!r}")

    # A LOCAL bundle (kind="local") has no on-chain descriptor: its third entry is the local hash-linked
    # ledger, and trust rests on the offline commitments above + bit-exact re-execution. There is simply no
    # onchain.json to bind, so skip the on-chain offline checks entirely.
    if manifest.get("kind") == "local":
        _check(checks, "kind.local", "onchain.json" not in obj,
               "local bundle must not carry an onchain.json")
    else:
        onchain = obj.get("onchain.json") or {}
        kind = onchain.get("kind")
        _check(checks, "onchain.kind", kind in ("standalone", "stateful"), str(kind))

        if kind == "standalone":
            _check(checks, "standalone.modelHash", onchain.get("modelHash") == receipt.get("modelHash"),
                   onchain.get("modelHash", ""))
            _check(checks, "standalone.receiptHash", onchain.get("receiptHash") == rh, onchain.get("receiptHash", ""))
        elif kind == "stateful":
            _verify_stateful_offline(checks, receipt, onchain, obj.get("identity.json") or {})

        # If the bundle carries the raw transaction, confirm it hashes to the claimed txid (offline, no network).
        raw = onchain.get("rawTx")
        if isinstance(raw, str) and raw:
            claimed = onchain.get("txid") if kind == "standalone" else onchain.get("actionTxid")
            try:
                match = txid_of(raw) == claimed
            except ValueError:
                match = False
            _check(checks, "onchain.txidMatchesRawTx", match, f"hash256(rawTx) vs {claimed}")
            # A matching txid is NOT enough: parse the rawTx and confirm its OP_RETURN actually
            # commits THIS receipt. Otherwise an offline-only auditor would accept a bundle whose
            # embedded rawTx legitimately hashes to `claimed` but commits an unrelated receipt
            # (review finding #14). parse_tx/find_op_return run with no network.
            try:
                parsed = chain_read.parse_tx(raw)
                hit = chain_read.find_op_return(parsed["outputs"])
            except (chain_read.ChainReadError, ValueError, KeyError, IndexError, TypeError):
                # parse_tx raises ChainReadError (a RuntimeError subclass) on a truncated rawTx —
                # include it so a malformed bundle records a FAILED check instead of crashing the
                # verifier with an uncaught traceback (fail-closed, review-2 finding #2).
                hit = None
            if hit is None:
                _check(checks, "onchain.rawTxBindsReceipt", False, "no parseable OP_RETURN in rawTx")
            else:
                _vout, items = hit
                data = [it for it in items if it]
                if kind == "standalone":
                    binds = (len(items) == 3
                             and items[1] == receipt.get("modelHash")
                             and items[2] == rh)
                    _check(checks, "onchain.rawTxBindsReceipt", binds, f"rawTx OP_RETURN items={items}")
                elif kind == "stateful":
                    binds = (len(data) == 1 and data[0] == onchain.get("receiptHashOnChain"))
                    _check(checks, "onchain.rawTxBindsReceipt", binds,
                           f"rawTx OP_RETURN data={data} vs {onchain.get('receiptHashOnChain')}")

    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "checks": checks}


def _verify_stateful_offline(checks: list, receipt: dict, onchain: dict, identity: dict) -> None:
    """Recompute the AgentTea action hash from the bundle's committed fields and bind it to the receipt."""
    action = onchain.get("action") or {}
    rid = identity.get("ricardianHash")
    agent_pk = identity.get("agentPubKey")
    cpty_pk = identity.get("counterpartyPubKey")
    action_hash = action.get("actionHash")
    provenance_hash = action.get("provenanceHash")

    # The Bonsai binding: actionHash IS the trinote receiptHash, provenanceHash IS the modelHash.
    _check(checks, "stateful.actionHash==receiptHash", action_hash == receipt.get("receiptHash"),
           f"{action_hash} vs {receipt.get('receiptHash')}")
    _check(checks, "stateful.provenanceHash==modelHash", provenance_hash == receipt.get("modelHash"),
           f"{provenance_hash} vs {receipt.get('modelHash')}")

    have = all(isinstance(x, str) for x in (rid, agent_pk, cpty_pk, action_hash, provenance_hash))
    have = have and all(isinstance(action.get(k), int) for k in ("amount", "txCount", "lockTime"))
    if not have:
        _check(checks, "stateful.fields", False, "missing identity/action fields for action-hash recompute")
        return
    try:
        recomputed = agent_action_receipt_hash(
            ricardian_hash=rid, agent_pubkey=agent_pk, counterparty_pubkey=cpty_pk,
            amount=action["amount"], action_hash=action_hash, provenance_hash=provenance_hash,
            tx_count=action["txCount"], lock_time=action["lockTime"],
        )
    except ValueError as exc:
        _check(checks, "stateful.actionReceiptHash", False, str(exc))
        return
    _check(checks, "stateful.actionReceiptHash", recomputed == onchain.get("receiptHashOnChain"),
           f"{recomputed} vs {onchain.get('receiptHashOnChain')}")


def _verify_onchain(loaded: dict, network: str) -> dict:
    checks: list[dict] = []
    onchain = loaded["obj"].get("onchain.json") or {}
    receipt = loaded["obj"].get("receipt.json") or {}
    identity = loaded["obj"].get("identity.json") or {}
    kind = onchain.get("kind")
    # Network is CALLER-authoritative. A bundle must not be able to redirect on-chain
    # verification to a different network than the auditor asked for: otherwise a producer
    # publishes a zero-cost testnet OP_RETURN, sets onchain.network='test', and an auditor
    # running --network main reports ok=True — a free forgery of the (costly) mainnet anchor.
    # We always fetch against `network`, and FAIL if the bundle declares something else.
    declared = onchain.get("network")
    if declared is not None and declared != network:
        _check(checks, "onchain.network", False,
               f"bundle declares network {declared!r} but verifying against {network!r}")
    net = network
    try:
        if kind == "standalone":
            txid = onchain.get("txid")
            anchor = chain_read.read_standalone_anchor(txid, net)
            _check(checks, "onchain.found", anchor.get("found"), anchor.get("reason", ""))
            if anchor.get("found"):
                _check(checks, "onchain.modelHash", anchor["modelHash"] == receipt.get("modelHash"),
                       anchor["modelHash"])
                _check(checks, "onchain.receiptHash", anchor["receiptHash"] == receipt.get("receiptHash"),
                       anchor["receiptHash"])
                # Pin the tag to the protocol constant, NOT a bundle-supplied onchain.tag:
                # the on-chain OP_RETURN tag is fixed at 'trinote/r1', and trusting the bundle's
                # claimed tag would let a producer paper over an anchor written under a different tag.
                _check(checks, "onchain.tag", anchor.get("tag") == "trinote/r1",
                       str(anchor.get("tag")))
        elif kind == "stateful":
            txid = onchain.get("actionTxid")
            anchor = chain_read.read_stateful_anchor(txid, net)
            _check(checks, "onchain.found", anchor.get("found"), anchor.get("reason", ""))
            if anchor.get("found"):
                _check(checks, "onchain.actionReceiptHash",
                       anchor["receiptHash"] == onchain.get("receiptHashOnChain"), anchor["receiptHash"])
                genesis = identity.get("genesisTxid")
                if not genesis:
                    # A stateful bundle MUST carry identity.genesisTxid. Without the genesis
                    # walk we only know the action's receiptHash is on SOME tx — not that it
                    # sits on THIS identity's official history. Omitting it would let an
                    # attacker run an executeAction under their own identity with
                    # actionHash=victim's receiptHash and have the bundle verify. Fail closed.
                    _check(checks, "onchain.chainToGenesis", False,
                           "stateful bundle missing identity.genesisTxid (cannot prove identity provenance)")
                else:
                    walk = chain_read.walk_identity_to_genesis(txid, genesis, net)
                    _check(checks, "onchain.chainToGenesis", walk.get("ok"),
                           f"{walk.get('hops')} hops; {walk.get('reason', 'reached genesis')}")
        else:
            _check(checks, "onchain.kind", False, f"unknown kind {kind!r}")
    except chain_read.ChainReadError as exc:
        _check(checks, "onchain.read", False, str(exc))
    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "checks": checks}


def _verify_reexec(loaded: dict, model, model_digest: str | None,
                   model_pubkey: str | None = None, counterparty_pubkey: str | None = None,
                   sample_k: int = 0, sample_seed: int = 0) -> dict:
    """Bit-exact re-execution layer — defers to receipts.verify_receipt (needs the model artifact).

    The layer is OK only if re-execution reproduced the committed tokens AND the artifact actually loaded is
    bound to the committed modelHash (artifactBoundOk) AND no supplied/embedded signature is invalid. Two
    bugs are fixed here (R-finding #9): (a) `model_digest` is used as-is — we do NOT fall back to the
    bundle's own `manifest.modelHash`, which would make `modelHashMatch` tautological and defeat the binding;
    when no real digest of the loaded weights is supplied, the binding is reported absent and the layer fails
    closed. (b) `artifactBoundOk` is folded into `ok` (previously dropped). Signatures are also folded in:
    pass `model_pubkey`/`counterparty_pubkey` to PIN the expected signer identity (#11).
    """
    from ..receipts.verify import verify_receipt
    bundle = {"receipt": loaded["obj"]["receipt.json"], "preimage": loaded["obj"]["preimage.json"]}
    res = verify_receipt(bundle, model=model, model_digest=model_digest,
                         model_pubkey=model_pubkey, counterparty_pubkey=counterparty_pubkey,
                         sample_k=sample_k, sample_seed=sample_seed)
    reexec = res.get("reexec") or {}
    sig_ok = res.get("signatureOk")            # None = no signature present / none checked
    checks = [
        {"check": "structuralOk", "ok": bool(res.get("structuralOk")),
         "detail": "commitments + receiptHash + traceCommit + sampler shape"},
        {"check": "reexecOk", "ok": bool(res.get("reexecOk")),
         "detail": (f"strategy={reexec.get('strategy')} checked={reexec.get('checked')}"
                    + (f" of={reexec.get('of')} (PROBABILISTIC AUDIT — partial coverage)"
                       if reexec.get("sampled") else ""))},
        {"check": "artifactBindingOk", "ok": bool(res.get("artifactBindingOk")),
         "detail": "preimage.artifactDigest == committed modelHash"},
        {"check": "modelHashMatch", "ok": bool(res.get("modelHashMatch")),
         "detail": "digest(loaded weights) == committed modelHash"
                   if model_digest is not None else "no model_digest supplied (binding not provable)"},
    ]
    if sig_ok is not None:
        pinned = model_pubkey is not None or counterparty_pubkey is not None
        checks.append({"check": "signatureOk", "ok": bool(sig_ok),
                       "detail": "pinned to expected identity" if pinned
                       else "valid for the receipt's embedded public key (identity NOT pinned)"})
    ok = (bool(res.get("structuralOk")) and bool(res.get("reexecOk"))
          and bool(res.get("artifactBoundOk")) and sig_ok is not False)
    return {"ok": ok, "checks": checks, "structuralOk": res.get("structuralOk"),
            "reexecOk": res.get("reexecOk"), "artifactBoundOk": res.get("artifactBoundOk"),
            "signatureOk": sig_ok, "signaturePinned": model_pubkey is not None or counterparty_pubkey is not None,
            "strategy": reexec.get("strategy"), "checked": reexec.get("checked"),
            "of": reexec.get("of"), "sampled": bool(reexec.get("sampled")), "raw": res}


def commit_bytes(b: bytes) -> str:
    """sha256 hex of raw bytes (file digest as recorded in the manifest)."""
    import hashlib
    return hashlib.sha256(b).hexdigest()


def _failclosed_layer(label: str, fn):
    """Run a verification layer, converting ANY exception into a failing check (never an escaping raise).
    A crafted bundle must not be able to crash the public verifier — an unexpected exception on adversarial
    input is a verification failure, so this catches broadly by design."""
    try:
        return fn()
    except Exception as e:                                   # noqa: BLE001 — security boundary, fail closed
        return {"ok": False, "checks": [{"check": f"{label}.error", "ok": False,
                                         "detail": f"{type(e).__name__}: {e}"}]}


def verify_bundle(path: str | Path, *, onchain: bool = False, network: str = "main",
                  reexec: bool = False, model=None, model_digest: str | None = None,
                  model_pubkey: str | None = None, counterparty_pubkey: str | None = None,
                  sample_k: int = 0, sample_seed: int = 0) -> dict:
    """Verify a bundle. Returns {ok, kind, bundleHash, offline, onchain?, reexec?}.

    `ok` is the AND of every layer actually run. `offline` always runs. `onchain` runs iff `onchain=True`
    (network). `reexec` runs iff `reexec=True` and a `model` is supplied. For EC-signed receipts, pass
    `model_pubkey`/`counterparty_pubkey` to PIN the expected signer identity; a stateful bundle defaults them
    to the committed `identity.json` (`agentPubKey`/`counterpartyPubKey`) so the action's own identity is
    enforced unless the caller overrides.
    """
    # A malformed CONTAINER (unparseable/corrupt archive, missing manifest) raises a typed BundleError — a
    # deliberate, tested contract the caller catches (test_corrupt_tar_raises_bundle_error). What must NOT
    # escape is a crash while processing well-formed-but-ADVERSARIAL CONTENT (e.g. a non-integer
    # trace.sampler.seed makes chain_artifact() raise inside _verify_offline). So load_bundle propagates, and
    # each content layer below is fail-closed (sibling of receipts.verify_receipt's #10 guard).
    loaded = load_bundle(path)
    manifest = loaded["manifest"]
    result = {"kind": manifest.get("kind"), "bundleHash": manifest.get("bundleHash"),
              "path": loaded["root"]}

    off = _failclosed_layer("offline", lambda: _verify_offline(loaded))
    result["offline"] = off
    layers_ok = [off["ok"]]

    if onchain:
        if manifest.get("kind") == "local":
            result["onchain"] = {"ok": True, "skipped": True,
                                 "checks": [{"check": "onchain.localBundle", "ok": True,
                                             "detail": "local bundle has no on-chain third entry (BSV off)"}]}
        else:
            on = _failclosed_layer("onchain", lambda: _verify_onchain(loaded, network))
            result["onchain"] = on
            layers_ok.append(on["ok"])

    if reexec:
        if model is None:
            result["reexec"] = {"ok": False, "checks": [{"check": "model", "ok": False,
                                "detail": "re-exec requested but no model supplied"}]}
            layers_ok.append(False)
        else:
            def _do_reexec():
                identity = loaded["obj"].get("identity.json") or {}
                mpk = model_pubkey or (identity.get("agentPubKey") if manifest.get("kind") == "stateful" else None)
                cpk = counterparty_pubkey or (identity.get("counterpartyPubKey") if manifest.get("kind") == "stateful" else None)
                return _verify_reexec(loaded, model, model_digest, model_pubkey=mpk, counterparty_pubkey=cpk,
                                      sample_k=sample_k, sample_seed=sample_seed)
            rx = _failclosed_layer("reexec", _do_reexec)
            result["reexec"] = rx
            layers_ok.append(rx["ok"])

    result["ok"] = all(layers_ok)
    return result
