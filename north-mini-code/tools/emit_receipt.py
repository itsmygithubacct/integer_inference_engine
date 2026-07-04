#!/usr/bin/env python3
"""Emit + self-verify a verifiable receipt for a north-mini-code generation.

Generates with the deterministic integer engine, re-executes to prove byte-identical output, then builds a
secp256k1-signed, hash-chained, content-addressed receipt bundle (reusing the Bonsai notary stack). The receipt
attests *what the integer engine deterministically produced*, re-verifiable by anyone — NOT float-parity with
llama.cpp (the model label says so). Keys + ledger + bundles live under ~/.local/integer_inference_engine/north-mini-code.

    sudo env PYTHONPATH=src NMC_BONSAI_SRC=<bonsai/src> .venv/bin/python tools/emit_receipt.py <blob> "prompt" [n_new]
"""
import sys
from datetime import datetime, timezone

from nmc.engine import Engine
from nmc.receipts_runtime import emit_and_verify, STATE_HOME

blob = sys.argv[1]
prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
n_new = int(sys.argv[3]) if len(sys.argv) > 3 else 12

eng = Engine(blob)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out_dir = STATE_HOME / "receipts" / "bundles" / stamp
print(f"[receipt] backend={eng.bname} fused={eng.fused}  prompt={prompt!r} n_new={n_new}", flush=True)

out, res = emit_and_verify(eng, prompt, n_new, out_dir=out_dir)
r = res["receipt"]
ok = bool(res["offline_ok"] and res["verify_bundle"]["ok"])
print(f"[receipt] output={eng.decode(out)!r}")
print(f"[receipt] modelHash={r['modelHash'][:24]}…  receiptHash={r['receiptHash'][:24]}…")
print(f"[receipt] signed: model={r['sigModelKeyId']}  counterparty={r['sigCounterpartyKeyId']}")
print(f"[receipt] offline+signature verified: {res['offline_ok']}   bundle (manifest/commits) verified: {res['verify_bundle']['ok']}")
print(f"[receipt] ledger entry #{res['ledger_entry']['index']}  bundle={res['bundle']['path']}")
print("[receipt] VERIFIED" if ok else "[receipt] FAILED")
sys.exit(0 if ok else 1)
