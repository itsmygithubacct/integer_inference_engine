"""trinote-receipt-bundle — package and verify a portable Bonsai receipt bundle.

    trinote-receipt-bundle pack    --receipt-bundle B.json (--txid TXID | --onchain O.json | --from-emission E.json) -o OUT [--tar]
    trinote-receipt-bundle verify  BUNDLE [--onchain] [--reexec --artifact A.safetensors] [--json]
    trinote-receipt-bundle inspect BUNDLE

A bundle is the self-contained artifact a third party needs to audit a notarized inference offline (and,
with --onchain, confirm the third entry is published on BSV). See docs/receipts/RECEIPT-BUNDLE.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ..bundle import pack_bundle, verify_bundle, load_bundle, BundleError


def _load_json(path: str):
    if path == "-":
        return json.load(sys.stdin)
    return json.loads(Path(path).read_text("utf-8"))


def _standalone_from_txid(receipt: dict, txid: str, network: str, tag: str) -> dict:
    return {"kind": "standalone", "network": network, "tag": tag, "txid": txid,
            "modelHash": receipt["modelHash"], "receiptHash": receipt["receiptHash"]}


def _stateful_from_record(record: dict, network: str) -> tuple[dict, dict]:
    """Convert a chain_c/AgentTea action record into bundle inputs.

    Keep this transform in the engine package so ``pack --from-emission`` does
    not need to import the optional ``bsv_third_entry`` composition package.
    The bundle verifier independently recomputes every binding.
    """
    required = (
        "actionTxid", "receiptHashOnChain", "txCount", "lockTime", "amount",
        "actionHash", "provenanceHash", "identity",
    )
    missing = [key for key in required if key not in record]
    ident = record.get("identity")
    identity_required = ("ricardianHash", "genesisTxid", "agentPubKey", "counterpartyPubKey")
    if missing or not isinstance(ident, dict):
        raise ValueError(f"incomplete stateful action record; missing {', '.join(missing) or 'identity'}")
    identity_missing = [key for key in identity_required if key not in ident]
    if identity_missing:
        raise ValueError(
            "incomplete stateful identity; missing " + ", ".join(identity_missing)
        )
    onchain = {
        "kind": "stateful",
        "network": network,
        "actionTxid": record["actionTxid"],
        "receiptVout": record.get("receiptVout", 1),
        "receiptHashOnChain": record["receiptHashOnChain"],
        "action": {
            "amount": int(record["amount"]),
            "txCount": int(record["txCount"]),
            "lockTime": int(record["lockTime"]),
            "actionHash": record["actionHash"],
            "provenanceHash": record["provenanceHash"],
        },
    }
    for key in ("rawTx", "sizeBytes"):
        if record.get(key) is not None:
            onchain[key] = record[key]
    identity = {key: ident[key] for key in identity_required}
    return onchain, identity


def _cmd_pack(args) -> int:
    try:
        rb = _load_json(args.receipt_bundle)
    except (OSError, ValueError) as exc:
        print(f"[bundle] cannot read --receipt-bundle: {exc}", file=sys.stderr)
        return 2
    receipt = rb.get("receipt", {})
    ledger_entry = _load_json(args.ledger_entry) if args.ledger_entry else None
    identity = _load_json(args.identity) if args.identity else None

    sources = [bool(args.onchain), bool(args.from_emission), bool(args.txid)]
    if sum(sources) != 1:
        print("[bundle] pack needs exactly one of --onchain / --from-emission / --txid", file=sys.stderr)
        return 2
    if args.onchain:
        onchain = _load_json(args.onchain)
    elif args.from_emission:
        emission = _load_json(args.from_emission)
        oc = emission.get("onchain", {})
        txid = oc.get("txid") or oc.get("txId")
        if not txid or str(txid).startswith("log:"):
            print(f"[bundle] --from-emission has no real on-chain txid (status={oc.get('status')}); "
                  "the receipt was not broadcast", file=sys.stderr)
            return 2
        record = oc.get("record")
        if isinstance(record, dict):
            try:
                onchain, derived_identity = _stateful_from_record(record, args.network)
            except (TypeError, ValueError) as exc:
                print(f"[bundle] invalid stateful action record: {exc}", file=sys.stderr)
                return 2
            if identity is not None and identity != derived_identity:
                print("[bundle] --identity disagrees with the stateful action record", file=sys.stderr)
                return 2
            identity = derived_identity
        else:
            tag = emission.get("chainArtifact", {}).get("tag", "trinote/r1")
            onchain = _standalone_from_txid(receipt, txid, args.network, tag)
            if oc.get("rawTx"):  # self-contained + re-broadcastable
                onchain["rawTx"] = oc["rawTx"]
                onchain["sizeBytes"] = oc.get("sizeBytes")
    else:
        onchain = _standalone_from_txid(receipt, args.txid, args.network, args.tag)

    try:
        res = pack_bundle(bundle=rb, onchain=onchain, out_dir=args.out, ledger_entry=ledger_entry,
                          identity=identity, model_label=args.model_label, created=args.created,
                          as_tar=args.tar)
    except BundleError as exc:
        print(f"[bundle] pack failed: {exc}", file=sys.stderr)
        return 2
    print(f"[bundle] packed {res['path']}", file=sys.stderr)
    print(f"[bundle] bundleHash {res['bundleHash']}", file=sys.stderr)
    print(json.dumps({"path": res["path"], "bundleHash": res["bundleHash"], "kind": onchain["kind"]}))
    return 0


def _load_model(artifact: str, *, fast: bool = True):
    from ..infer_int.artifact_io_bonsai import load_artifact_bonsai
    from ..infer_int.reference_bonsai import BonsaiReferenceModel
    from ..infer_int.reference_bonsai35 import BonsaiQwen35ReferenceModel
    art, info = load_artifact_bonsai(artifact)
    architecture = str(art.get("config", {}).get("architecture", ""))
    if architecture == "qwen35":
        model = BonsaiQwen35ReferenceModel(art)
    elif architecture == "qwen3":
        model = BonsaiReferenceModel(art)
    else:
        raise ValueError(f"unsupported Bonsai artifact architecture {architecture!r}")
    engine = "oracle"
    if fast:
        # Engage the native packed-Q1 kernel for re-execution. It is BYTE-IDENTICAL to the pure-NumPy oracle
        # (proven in tests/test_bonsai_smoke.py), so the replay verdict is unchanged — but ~100x faster, the
        # difference between a usable verify and a multi-minute oracle replay. Falls back to sign-cache, then
        # the oracle, if the native kernel can't load.
        if model.enable_native():
            engine = "native"
        elif model.enable_fast(check_ram=True, cache_output=True):
            engine = "sign-cache"
    # Telemetry/guard: a silent downgrade to the ~100x-slower pure-NumPy oracle is the main way a "verify" looks
    # broken (multi-minute hangs). Make the engine path VISIBLE on stderr and in the debug log so it's
    # diagnosable; re-exec stays byte-exact regardless of path.
    if engine == "oracle":
        print("[bundle] WARNING: native packed-Q1 kernel AND sign-cache are both unavailable — re-executing on "
              "the pure-NumPy ORACLE (~100x slower, expect minutes/turn). Re-exec is still byte-exact.",
              file=sys.stderr)
    else:
        print(f"[bundle] re-exec engine: {engine}", file=sys.stderr)
    try:
        from .json_mode import debug_log
        debug_log({"ts": time.time(), "schema": "bonsai-debug/v1", "event": "receipt-verify-load",
                   "artifact": str(artifact), "enginePath": engine, "fastRequested": bool(fast)})
    except Exception:                                   # telemetry must never break a verify
        pass
    return model, info["digest"], engine


def _cmd_verify(args) -> int:
    bundles = args.bundle if isinstance(args.bundle, list) else [args.bundle]
    model, model_digest = None, None
    if args.reexec:
        if not args.artifact:
            print("[bundle] --reexec requires --artifact <safetensors>", file=sys.stderr)
            return 2
        # Batch model-load amortization: load + enable_native() ONCE, reuse the read-only model across every
        # bundle (verify_bundle takes model=/model_digest=; KV state is per-call). Turns an N-bundle sweep
        # from N·(load+reexec) into load_once + N·reexec.
        print(f"[bundle] loading model once for {len(bundles)} bundle(s): {args.artifact}", file=sys.stderr)
        model, model_digest, _engine = _load_model(args.artifact, fast=not args.oracle)
        if args.cached_replay_threshold is not None:
            model.receipt_verify_cached_threshold = max(1, int(args.cached_replay_threshold))
            print(f"[bundle] cached-replay threshold = {model.receipt_verify_cached_threshold} output tokens",
                  file=sys.stderr)

    results, rc = [], 0
    for b in bundles:
        try:
            res = verify_bundle(b, onchain=args.onchain, network=args.network,
                                reexec=args.reexec, model=model, model_digest=model_digest,
                                model_pubkey=args.model_pubkey, counterparty_pubkey=args.counterparty_pubkey,
                                sample_k=args.sample_positions, sample_seed=args.sample_seed)
        except (FileNotFoundError, BundleError) as exc:
            print(f"[bundle] verify failed for {b}: {exc}", file=sys.stderr)
            rc = 2
            continue
        results.append(res)
        if not args.json:
            if len(bundles) > 1:
                print(f"\n===== {b} =====")
            _print_human(res)
        if not res["ok"]:
            rc = rc or 1

    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results))
    if len(bundles) > 1:
        n_ok = sum(1 for r in results if r["ok"])
        print(f"\n[bundle] SUMMARY: {n_ok}/{len(bundles)} bundles VERIFIED (model loaded once)", file=sys.stderr)
    return rc


def _print_layer(name: str, layer: dict) -> None:
    mark = "PASS" if layer.get("ok") else "FAIL"
    print(f"  [{mark}] {name}")
    for c in layer.get("checks", []):
        m = "ok " if c["ok"] else "ERR"
        detail = f"  ({c['detail']})" if c.get("detail") and not c["ok"] else ""
        print(f"      {m} {c['check']}{detail}")


def _print_human(res: dict) -> None:
    print(f"bundle : {res['path']}")
    print(f"kind   : {res['kind']}   bundleHash {res['bundleHash']}")
    _print_layer("offline (hashes, commitments, bundleHash)", res["offline"])
    if "onchain" in res:
        _print_layer("on-chain (WhatsOnChain third entry)", res["onchain"])
    if "reexec" in res:
        rx = res["reexec"]
        mark = "PASS" if rx.get("ok") else "FAIL"
        extra = ""
        if "strategy" in rx:
            extra = f"strategy={rx.get('strategy')} checked={rx.get('checked')}"
            if rx.get("sampled"):
                extra += f" of={rx.get('of')} (PROBABILISTIC AUDIT — partial coverage, NOT a full verify)"
        print(f"  [{mark}] re-exec (bit-exact reference engine) {extra}")
        for c in rx.get("checks", []):
            print(f"      {'ok ' if c['ok'] else 'ERR'} {c['check']}  ({c.get('detail','')})")
        if rx.get("signatureOk") is not None and not rx.get("signaturePinned"):
            print("      NOTE signature is valid for the receipt's EMBEDDED key but identity was NOT pinned"
                  " — pass --model-pubkey to bind it to an expected signer")
        elif rx.get("signatureOk") is None:
            print("      NOTE no third-party-verifiable signature checked (HMAC vouch or none present)")
    print(f"\nRESULT : {'VERIFIED' if res['ok'] else 'NOT VERIFIED'}")


def _cmd_inspect(args) -> int:
    try:
        loaded = load_bundle(args.bundle)
    except (FileNotFoundError, BundleError) as exc:
        print(f"[bundle] {exc}", file=sys.stderr)
        return 2
    m = loaded["manifest"]
    print(json.dumps(m, indent=2, sort_keys=True))
    oc = loaded["obj"].get("onchain.json")
    if oc:
        print("\nonchain.json:")
        print(json.dumps(oc, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="trinote-receipt-bundle",
                                 description="Package and verify portable Bonsai receipt bundles")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack", help="package a receipt + its on-chain anchor into a bundle")
    p.add_argument("--receipt-bundle", required=True,
                   help="JSON with {receipt, preimage} (build_receipt output); '-' reads stdin")
    p.add_argument("--onchain", help="explicit onchain.json descriptor (kind=standalone|stateful)")
    p.add_argument("--from-emission", help="emit_receipt() return JSON — derive a standalone descriptor from its txid")
    p.add_argument("--txid", help="standalone: the OP_RETURN third-entry txid (modelHash/receiptHash taken from the receipt)")
    p.add_argument("--tag", default="trinote/r1", help="standalone OP_RETURN protocol tag")
    p.add_argument("--network", default="main", help="BSV network for the on-chain descriptor")
    p.add_argument("--ledger-entry", help="optional local ledger entry JSON to include")
    p.add_argument("--identity", help="stateful: identity.json (ricardianHash, genesisTxid, pubkeys)")
    p.add_argument("--model-label", default="", help="human label for the model")
    p.add_argument("--created", default=None, help="ISO timestamp to record (optional)")
    p.add_argument("-o", "--out", required=True, help="output bundle dir, or .tar.gz path with --tar")
    p.add_argument("--tar", action="store_true", help="emit a .tar.gz instead of a directory")
    p.set_defaults(func=_cmd_pack)

    v = sub.add_parser("verify", help="verify one or more bundles (offline; optional on-chain + re-execution)")
    v.add_argument("bundle", nargs="+",
                   help="bundle directory or .tar.gz (pass several to verify a batch, loading the model once)")
    v.add_argument("--sample-positions", type=int, default=0, metavar="K",
                   help="PROBABILISTIC AUDIT: re-derive only K deterministically-chosen output positions per "
                        "greedy receipt — a fast ledger-wide screen (lower assurance, NOT a full verification)")
    v.add_argument("--sample-seed", type=int, default=0, help="seed for --sample-positions selection (default 0)")
    v.add_argument("--cached-replay-threshold", type=int, default=None, metavar="N",
                   help="override the long-turn KV-cached-replay threshold: output turns >= N tokens replay via "
                        "cached M=1 decode instead of an M=N teacher-forced prefill (M=N is slower on this CPU)")
    v.add_argument("--onchain", action="store_true", help="also confirm the third entry on WhatsOnChain (network)")
    v.add_argument("--network", default="main", help="BSV network for --onchain (default main)")
    v.add_argument("--reexec", action="store_true", help="also re-run the bit-exact reference engine (needs --artifact)")
    v.add_argument("--artifact", help="safetensors artifact for --reexec")
    v.add_argument("--oracle", action="store_true",
                   help="re-execute on the pure-NumPy ORACLE (the source of truth) instead of the byte-identical "
                        "native kernel — paranoid + ~100x slower; default uses the fast native path")
    v.add_argument("--model-pubkey", default=None,
                   help="PIN the expected model signer (compressed secp256k1 hex) for an EC-signed receipt; "
                        "a wrong signer then fails. Stateful bundles default to identity.json's agentPubKey.")
    v.add_argument("--counterparty-pubkey", default=None,
                   help="PIN the expected counterparty signer (compressed secp256k1 hex)")
    v.add_argument("--json", action="store_true", help="emit the full result as JSON")
    v.set_defaults(func=_cmd_verify)

    i = sub.add_parser("inspect", help="print a bundle's manifest and on-chain descriptor")
    i.add_argument("bundle", help="bundle directory or .tar.gz")
    i.set_defaults(func=_cmd_inspect)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
