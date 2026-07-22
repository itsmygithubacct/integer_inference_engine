#!/usr/bin/env python3
"""Replay a receipt bundle on THIS machine — the cross-machine determinism claim.

Given a bundle produced elsewhere (its authoritative record is preimage.json: inputIds, outputIds, sampler,
modelHash), this:
  1. offline-verifies the producer's bundle (secp256k1 sigs + commitments + chain artifact),
  2. confirms THIS box's GGUF hashes to the same modelHash (same weights + same integer semantics),
  3. re-executes the exact input under the recorded sampler (greedy/seed-0) and asserts the output reproduces
     BYTE-FOR-BYTE, and
  4. ties that re-execution to the producer's SIGNED claim: token_commit(re-exec output) == the outputCommit
     the producer signed.

Note: receiptHash = sha256(receipt body INCLUDING the producer's signatures) — it commits the signed pair, so
it is KEY-dependent. A third-party verifier (no producer private keys) does NOT reproduce receiptHash; it
*verifies* the producer's signatures with the public keys in the bundle (step 1) and reproduces the OUTPUT
COMMITMENT (step 4). That is the complete claim: the producer cryptographically attested an output commitment,
and a different physical machine re-running the deterministic integer engine reproduces it byte-for-byte.

    PYTHONPATH=src NMC_BACKEND=cuda-resident .venv/bin/python tools/replay_receipt.py <bundle_dir> <gguf_blob>
"""
import json
import sys

import numpy as np

from nmc.engine import Engine
from nmc import receipts_runtime as rr

bundle = sys.argv[1].rstrip("/")
blob = sys.argv[2]
pre = json.load(open(f"{bundle}/preimage.json"))
rc0 = json.load(open(f"{bundle}/receipt.json"))
in_ids = [int(x) for x in pre["inputIds"]]
out0 = [int(x) for x in pre["outputIds"]]
print(f"[replay] target: receiptHash={rc0['receiptHash'][:16]}…  outputCommit={rc0['outputCommit'][:16]}…  "
      f"modelHash={pre['modelHash'][:16]}…  in={len(in_ids)} out={len(out0)}", flush=True)

# 1) offline-verify the producer's bundle (no re-exec — sigs + commitments + chain)
try:
    bv = rr.verify_bundle(bundle, reexec=False)
    ok1 = bool(bv.get("ok", bv)) if isinstance(bv, dict) else bool(bv)
    print(f"[replay] (1) offline bundle verify (sigs+commits+chain): {ok1}  {bv if isinstance(bv, dict) else ''}")
except Exception as e:
    ok1 = False
    print(f"[replay] (1) offline bundle verify error: {e}")

# 2) same model? (GGUF tensor data + integer-engine config → modelHash)
eng = Engine(blob)
mh, art = rr.model_hash(eng)
ok2 = mh == pre["modelHash"]
print(f"[replay] (2) modelHash match={ok2}  artifactDigest match={art == pre.get('artifactDigest')}")
assert ok2, f"DIFFERENT MODEL: this box {mh} != bundle {pre['modelHash']}"

# 3) re-execute the exact input under the recorded sampler → byte-exact output?
sampler = rr.SamplerConfig(mode="greedy")
fa = eng.cfg.fa


def pick(row, pos, hist):
    return rr.sample_token(np.asarray(row, np.int64), sampler, position=pos, frac_bits=fa, history_ids=hist)


# The recorded output length is authoritative. Disabling early EOS stopping is
# equivalent for ordinary receipts (EOS, if present, is their final token) and
# also makes fixed-length hardware-gate receipts exactly replayable.
new = [int(x) for x in eng.generate(in_ids, len(out0), pick=pick, stop_eos=False)]
ok3 = new == out0
print(f"[replay] (3) re-executed {len(new)} tokens; BYTE-EXACT output match: {ok3}")
if not ok3:
    for i, (a, b) in enumerate(zip(new, out0)):
        if a != b:
            print(f"           first divergence @ token {i}: got {a} want {b}")
            break
    print(f"           len(new)={len(new)} len(bundle)={len(out0)}")

# 4) tie the re-execution to the producer's SIGNED commitment. receiptHash binds the producer's signatures
#    (commits the signed pair) and is key-dependent — so a verifier reproduces the OUTPUT COMMITMENT, not the
#    receiptHash. Matching the signed outputCommit + valid producer sigs (step 1) = the full attestation.
if eng.resident:
    eng.free()
oc_new = rr.token_commit(new)
ok4 = oc_new == rc0["outputCommit"]
print(f"[replay] (4) re-exec outputCommit == producer's SIGNED outputCommit: {ok4}  ({oc_new[:16]}…)")

passed = ok1 and ok2 and ok3 and ok4
print(f"[replay]   producer sigs valid={ok1}  same model={ok2}  byte-exact output={ok3}  signed-commit match={ok4}")
print(f"[replay] RESULT: {'BYTE-EXACT REPRODUCED on a different machine ✓' if passed else 'REPRODUCTION FAILED ✗'}")
print("[replay] done")
sys.exit(0 if passed else 1)
