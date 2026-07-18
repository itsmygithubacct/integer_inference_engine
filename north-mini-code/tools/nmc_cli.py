#!/usr/bin/env python3
"""north-mini-code CLI — deterministic integer inference for cohere2moe (30B MoE), with optional receipts.

Modes: one-shot (default), `repl`, `json`. `--receipts` emits a secp256k1-signed, re-verifiable receipt for the
generation (self-verified by a byte-exact re-execution); its on-chain 3rd-entry artifact (modelHash+receiptHash)
is produced + dry-run-logged by default. `--broadcast` does a REAL BSV broadcast via the bonsai-notary wallet
(builds the tx; `--confirm` to actually send — spends sats). json/one-shot print ONLY the model output unless
`--verbose` (diagnostics go to stderr). The GGUF blob is found via --blob / $NMC_BLOB / the largest ollama blob.
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from nmc.engine import Engine


def find_blob(explicit=None):
    if explicit and Path(explicit).exists():
        return explicit
    b = os.environ.get("NMC_BLOB")
    if b and Path(b).exists():
        return b
    cands = []
    for d in ("/usr/share/ollama/.ollama/models/blobs", str(Path.home() / ".ollama/models/blobs")):
        cands += [c for c in glob.glob(d + "/sha256-*") if Path(c).is_file()]
    if cands:
        return max(cands, key=lambda c: Path(c).stat().st_size)         # the ~18GB GGUF
    sys.exit("no GGUF blob found — set NMC_BLOB=<path>, pass --blob, or `ollama pull north-mini-code-1.0`")


def _vlog(on, *a):
    if on:
        print(*a, file=sys.stderr, flush=True)


def _receipt_ok(receipts, result):
    bundle = result.get("verify_bundle") if isinstance(result, dict) else None
    return (not receipts) or bool(result and result.get("offline_ok") and bundle and bundle.get("ok"))


def _report_receipt_failure():
    print("[nmc] ERROR: receipt failed offline self-verification — bundle is NOT trustworthy", file=sys.stderr)


def _streamer(eng):
    """Returns (on_token, get_text). Decodes the FULL accumulated ids each token and writes the new suffix —
    correct for byte-level BPE where one char can span tokens (a naive per-token decode would mangle them)."""
    acc, prev = [], [""]

    def on_token(tok):
        acc.append(tok)
        cur = eng.decode(acc)
        sys.stdout.write(cur[len(prev[0]):]); sys.stdout.flush()
        prev[0] = cur
    return on_token, (lambda: prev[0])


def run_one(eng, prompt, n, *, verbose, receipts, broadcast, confirm, stream):
    t = time.time()
    on_token, _ = _streamer(eng) if stream else (None, None)
    if receipts:
        from nmc.receipts_runtime import emit_and_verify, STATE_HOME
        out_dir = STATE_HOME / "receipts" / "bundles" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        # verify=False: skip the byte-exact self re-run (2× faster) — the receipt is still signed, bundled, and
        # re-verifiable offline by anyone; the per-run re-execution is the verifier's job, not the producer's.
        out, res = emit_and_verify(eng, prompt, n, out_dir=out_dir, enable_chain=broadcast, confirm=confirm,
                                   verify=False, on_token=on_token)
        if stream:
            print()
        r = res["receipt"]; ok = bool(res["offline_ok"] and res["verify_bundle"]["ok"])
        _vlog(verbose, f"[nmc] backend={eng.bname} {time.time()-t:.1f}s  {len(out)} tok  receiptHash={r['receiptHash'][:16]}… "
                       f"modelHash={r['modelHash'][:16]}… verified(offline)={ok}  3rd-entry={res['emission']['onchain']['status']}"
                       f"  bundle={res['bundle']['path']}")
        return (None if stream else eng.decode(out)), res
    out = eng.generate(eng.encode(prompt), n, on_token=on_token)         # greedy, deterministic
    if stream:
        print()
    _vlog(verbose, f"[nmc] backend={eng.bname} {time.time()-t:.1f}s  {len(out)} tok")
    return (None if stream else eng.decode(out)), None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="north-mini-code-cli", add_help=True)
    ap.add_argument("a", nargs="?", help="mode (repl|json) or, if omitted, the prompt")
    ap.add_argument("b", nargs="?", help="prompt (for json mode)")
    ap.add_argument("-n", "--max-new", type=int, default=1024, help="max new tokens (a CAP; EOS ends earlier)")
    ap.add_argument("--receipts", action="store_true", help="emit + self-verify a receipt")
    ap.add_argument("--broadcast", action="store_true",
                    help="real BSV on-chain broadcast via the bonsai-notary wallet (the chain artifact is logged by "
                         "default without this); builds the tx but does not send unless --confirm")
    ap.add_argument("--confirm", action="store_true", help="actually SEND the broadcast — spends sats (requires --broadcast)")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--blob", default=None)
    args = ap.parse_args(argv)
    if args.confirm and not args.broadcast:
        ap.error("--confirm requires --broadcast")
    if args.broadcast and not args.receipts:
        # broadcast only happens inside the receipts path; without --receipts it would be silently ignored.
        ap.error("--broadcast requires --receipts (the on-chain 3rd entry marks a receipt)")

    mode, prompt = "oneshot", args.a
    if args.a in ("repl", "json"):
        mode, prompt = args.a, args.b
    eng = Engine(find_blob(args.blob))
    _vlog(args.verbose or mode == "repl", f"[nmc] loaded backend={eng.bname} fused={eng.fused} "
          f"{eng.cfg.d_model}d {eng.NL}L {eng.cfg.n_experts}e/top{eng.cfg.n_used}"
          + ("  [receipts ON]" if args.receipts else ""))

    if mode == "repl":
        print("north-mini-code REPL — Ctrl-D to exit", file=sys.stderr)
        while True:
            try:
                line = input("nmc> ")
            except EOFError:
                print(file=sys.stderr); break
            if line.strip():
                _, res = run_one(eng, line, args.max_new, verbose=True, receipts=args.receipts,  # streams to stdout
                                 broadcast=args.broadcast, confirm=args.confirm, stream=True)
                if not _receipt_ok(args.receipts, res):
                    _report_receipt_failure()
                    return 1
        return 0

    if not prompt:
        ap.error("a prompt is required (or use `repl`)")
    stream = (mode != "json")                                     # one-shot streams; json needs the full object
    txt, res = run_one(eng, prompt, args.max_new, verbose=args.verbose, receipts=args.receipts,
                       broadcast=args.broadcast, confirm=args.confirm, stream=stream)
    receipt_ok = _receipt_ok(args.receipts, res)
    if mode == "json":
        obj = {"output": txt}
        if args.verbose:
            obj["backend"] = eng.bname
            if res:
                r = res["receipt"]
                obj["receipt"] = {"receiptHash": r["receiptHash"], "modelHash": r["modelHash"],
                                  "verified": receipt_ok,
                                  "thirdEntry": res["emission"]["onchain"]["status"],
                                  "bundle": res["bundle"]["path"]}
        print(json.dumps(obj))
    # one-shot already streamed the output to stdout
    if not receipt_ok:
        # the receipt failed its own offline/self-verification — surface it loudly and exit nonzero rather
        # than reporting success for a bundle that would not re-verify.
        _report_receipt_failure()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
