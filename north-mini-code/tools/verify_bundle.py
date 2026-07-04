#!/usr/bin/env python3
"""Offline-verify a receipt bundle — NO model, NO GPU, NO network. Two layers, both from the bundle alone:

  • STRUCTURAL (trinote.bundle.verify): manifest schema + bundleHash, every packed file's digest, receiptHash
    self-consistency, the input/output/trace commitments, and the chain artifact.
  • SIGNATURE  (trinote.receipts.verify): the secp256k1-ecdsa sigs validate over the canonical signed message.

It does NOT re-execute the model — that's `replay_receipt.py`, which reproduces the output byte-for-byte and is
what binds the SIGNED outputCommit to the deterministic computation. Run both for the full claim.

Authenticity note: by default the sigs are checked against the pubkey EMBEDDED in the receipt — that proves the
receipt is self-consistently signed (untampered), NOT who signed it (a forger can self-sign with a fresh key).
To prove a SPECIFIC producer signed it, pin their known key: --model-pubkey / --counterparty-pubkey.

    NMC_BONSAI_SRC=<…/bonsai/src> PYTHONPATH=src python3 tools/verify_bundle.py <bundle_dir> \
        [--model-pubkey HEX] [--counterparty-pubkey HEX]

Exit 0 = offline-valid (structural + signature).
"""
import argparse
import json

from nmc import receipts_runtime as rr
from trinote.receipts.verify import verify_receipt

ap = argparse.ArgumentParser()
ap.add_argument("bundle")
ap.add_argument("--model-pubkey", default=None, help="pin the expected model signer (authenticity, not just integrity)")
ap.add_argument("--counterparty-pubkey", default=None, help="pin the expected counterparty signer")
a = ap.parse_args()
bundle = a.bundle.rstrip("/")

# layer 1 — structural (files, commitments, receiptHash self-consistency, chain artifact)
bv = rr.verify_bundle(bundle, reexec=False)
print("=== structural ===")
for c in bv.get("offline", {}).get("checks", []):
    print(f"  [{'ok ' if c['ok'] else 'FAIL'}] {c['check']}: {c['detail']}")
structural_ok = bool(bv.get("offline", {}).get("ok"))

# layer 2 — secp256k1 signatures over the canonical signed message
rc = json.load(open(f"{bundle}/receipt.json"))
pre = json.load(open(f"{bundle}/preimage.json"))
vr = verify_receipt({"receipt": rc, "preimage": pre},
                    model_pubkey=a.model_pubkey, counterparty_pubkey=a.counterparty_pubkey)
print("\n=== signature / commitments ===")
for k in ("structuralOk", "signatureOk", "commitMatch", "receiptHashMatch",
          "sigModelAuthenticated", "sigCounterpartyAuthenticated"):
    print(f"  [{'ok ' if vr.get(k) else '－ ' if vr.get(k) is None else 'FAIL'}] {k}: {vr.get(k)}")
pinned = bool(a.model_pubkey or a.counterparty_pubkey)

print("\n--- the signed claim ---")
print(f"  modelHash    {rc['modelHash']}")
print(f"  inputCommit  {rc['inputCommit']}")
print(f"  outputCommit {rc['outputCommit']}")
print(f"  receiptHash  {rc['receiptHash']}   (commits the signed pair — key-dependent)")
print(f"  sampler      {rc['trace']['sampler']}")
print(f"  sigModel pub        {rc.get('sigModelPubKey', '')}")
print(f"  sigCounterparty pub {rc.get('sigCounterpartyPubKey', '')}")

offline_ok = structural_ok and bool(vr.get("signatureOk") and vr.get("commitMatch") and vr.get("receiptHashMatch"))
print(f"\nBUNDLE OFFLINE-VALID: {offline_ok}"
      f"  ({'identity PINNED' if pinned else 'integrity only — pin a pubkey to prove WHO signed'})")
print("Next: replay_receipt.py re-executes the model to reproduce the output byte-for-byte (the full claim).")
raise SystemExit(0 if offline_ok else 1)
