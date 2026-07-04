"""Verifiable receipts for the north-mini-code integer engine — reuses the Bonsai notary stack verbatim.

A receipt attests: *the north-mini-code INTEGER engine, on THESE weights, with THIS sampler, deterministically
produced THESE output token ids from THESE input ids* — verified by re-executing the (deterministic) engine and
reproducing the output byte-for-byte. It does NOT claim float-parity with llama.cpp (that's the fidelity eval);
the model label says so explicitly.

Model-agnostic machinery (commitments, secp256k1 signing, hash-chained ledger, content-addressed bundle,
receipt-safe integer samplers) is imported from the proven `trinote.*` stack. The only north-mini-code pieces:
`model_hash` (a digest over the GGUF tensor data + the integer-engine config) and the deterministic generate +
re-exec (the engine itself). State/keys live OUT of tree under ~/.local/integer_inference_engine/north-mini-code.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# The receipt/bundle/signing/ledger/sampler layers are model-agnostic — import them from the Bonsai engine.
_BONSAI_SRC = Path(os.environ.get("NMC_BONSAI_SRC",
                                  Path(__file__).resolve().parents[3] / "bonsai" / "src"))
if str(_BONSAI_SRC) not in sys.path:
    sys.path.insert(0, str(_BONSAI_SRC))

from trinote.receipts.canonical import commit, token_commit          # noqa: E402
from trinote.receipts.receipt import build_receipt                   # noqa: E402
from trinote.receipts.signing_ec import ECKey                        # noqa: E402
from trinote.receipts.ledger import LocalLedger                      # noqa: E402
from trinote.receipts.verify import verify_receipt                   # noqa: E402
from trinote.receipts.emit import emit_receipt                       # noqa: E402
from trinote.receipts.broadcast import WalletThirdEntryBackend       # noqa: E402
from trinote.bundle.pack import pack_bundle                          # noqa: E402
from trinote.bundle.verify import verify_bundle                      # noqa: E402
from trinote.infer_int.sampler import SamplerConfig, sample_token    # noqa: E402

STATE_HOME = Path.home() / ".local/integer_inference_engine/north-mini-code"
MODEL_LABEL = "north-mini-code-1.0 (integer engine)"


def model_hash(eng):
    """(modelHash, artifactDigest) for the engine. artifactDigest = sha256 of the GGUF TENSOR DATA region (pins
    the exact weights); modelHash = commit(artifactDigest + integer-engine config) — pins weights + fixed-point
    semantics (fa/fw/arch/RoPE convention), i.e. exactly which deterministic computation the receipt is for."""
    h = hashlib.sha256()
    with open(eng.g.path, "rb") as f:
        f.seek(eng.g.data_start)
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    artifact = h.hexdigest()
    cfg = eng.cfg
    config = {"arch": "cohere2moe", "engine": "north-mini-code-integer",
              "fa": cfg.fa, "fw": cfg.fw, "d_model": cfg.d_model, "n_layers": eng.NL,
              "leading_dense": eng.DENSE, "n_heads": cfg.n_heads, "n_kv": cfg.n_kv,
              "head_dim": cfg.head_dim, "n_experts": cfg.n_experts, "n_used": cfg.n_used,
              "expert_ffn": cfg.expert_ffn, "vocab": cfg.vocab, "rope": "interleaved-norm", "nope_full": True}
    mh = commit({"artifactDigest": artifact, "config": config})
    return mh, artifact


def load_keys(model_key_path=None, counterparty_key_path=None):
    """Real secp256k1 keys (third-party-verifiable), auto-generated 0600 under the state home if absent."""
    kd = STATE_HOME / "keys"
    km = ECKey.load_or_generate(model_key_path or kd / "model.json", label="north-mini-code model")
    kc = ECKey.load_or_generate(counterparty_key_path or kd / "counterparty.json", label="north-mini-code counterparty")
    return km, kc


def wallet_backend(*, confirm=False, **kw):
    """The real BSV on-chain 3rd-entry backend (bonsai-notary's HD wallet). DRY-RUN unless confirm=True
    (the two-key money interlock: enable_chain=True AND confirm=True). Reused verbatim from the Bonsai stack."""
    return WalletThirdEntryBackend(confirm=confirm, **kw)


def build_verify_pack(*, model_hash, artifact_digest, input_ids, output_ids, sampler, fa,
                      model_key, counterparty_key, out_dir, transcript=None, ledger_path=None,
                      model_label=MODEL_LABEL, enable_chain=False, chain_backend=None):
    """Build the receipt, self-verify (commitments + secp256k1 sigs), record the 3rd entry (local ledger +
    chain artifact — dry-run-logged by default; real BSV broadcast iff enable_chain=True + a confirmed wallet
    backend), pack the bundle, and offline-verify it. Model-free (synthetic ids OK) so it is unit-testable."""
    bundle = build_receipt(model_hash=model_hash, input_ids=list(input_ids), output_ids=list(output_ids),
                           sampler=sampler, model_key=model_key, counterparty_key=counterparty_key,
                           model_label=model_label, artifact_digest=artifact_digest, fp_frac_bits=fa)
    rcpt = bundle["receipt"]
    assert rcpt["inputCommit"] == token_commit(input_ids), "input commitment mismatch"
    assert rcpt["outputCommit"] == token_commit(output_ids), "output commitment mismatch"
    ver = verify_receipt(bundle, model=None, model_pubkey=model_key.public_hex,
                         counterparty_pubkey=counterparty_key.public_hex)
    offline_ok = bool(ver.get("commitMatch") and ver.get("receiptHashMatch")
                      and ver.get("signatureOk") and ver.get("structuralOk"))
    if not offline_ok:
        raise RuntimeError(f"receipt failed offline/signature verification: {ver}")
    rdir = STATE_HOME / "receipts"
    led = LocalLedger(str(ledger_path or rdir / "ledger.jsonl"))
    emission = emit_receipt(rcpt, ledger=led, ts=datetime.now(timezone.utc).isoformat(),
                            enable_chain=enable_chain, chain_backend=chain_backend, broadcast_to_log=True,
                            chain_artifacts_dir=str(rdir / "chain"), broadcast_log=str(rdir / "broadcast.log"))
    onchain = emission["onchain"] if enable_chain else None   # only a REAL send makes the bundle non-local
    info = pack_bundle(bundle=bundle, onchain=onchain, out_dir=str(out_dir),
                       ledger_entry=emission["ledgerEntry"], model_label=model_label, transcript=transcript)
    bv = verify_bundle(info["path"], reexec=False)        # independent offline re-check of the packed bundle
    return {"receipt": rcpt, "verify_receipt": ver, "offline_ok": offline_ok, "emission": emission,
            "bundle": info, "verify_bundle": bv, "ledger_entry": emission["ledgerEntry"]}


def emit_and_verify(eng, prompt, n_new, *, sampler=None, out_dir, model_key=None, counterparty_key=None,
                    enable_chain=False, confirm=False, wallet_kw=None, verify=True, on_token=None):
    """Full path: generate (producer) → optionally re-execute (byte-identical self-verify) → receipt → ledger +
    chain artifact → bundle. enable_chain=True broadcasts the 3rd entry to BSV via the wallet (DRY-RUN unless
    confirm=True — real money). `verify=False` skips the re-execution (2× faster — the receipt is still signed,
    bundled, and offline/re-exec-verifiable BY ANYONE; the per-run self-check is then just deferred to the
    verifier). on_token streams the producer's tokens. Returns (output_ids, result-dict)."""
    sampler = sampler or SamplerConfig(mode="greedy")
    fa = eng.cfg.fa
    ids = eng.encode(prompt)

    def pick(row, pos, hist):
        return sample_token(np.asarray(row, np.int64), sampler, position=pos, frac_bits=fa, history_ids=hist)

    out = eng.generate(ids, n_new, pick=pick, on_token=on_token)    # PRODUCER (streams via on_token)
    if verify:
        if eng.generate(ids, n_new, pick=pick) != out:             # VERIFIER: deterministic re-execution
            raise RuntimeError("non-deterministic generation — receipt would be unverifiable")
    if eng.resident:
        eng.free()
    mh, art = model_hash(eng)
    km, kc = (model_key, counterparty_key) if model_key else load_keys()
    chain_backend = wallet_backend(confirm=confirm, **(wallet_kw or {})) if enable_chain else None
    res = build_verify_pack(model_hash=mh, artifact_digest=art, input_ids=ids, output_ids=out,
                            sampler=sampler, fa=fa, model_key=km, counterparty_key=kc, out_dir=out_dir,
                            transcript={"prompt": prompt, "output": eng.decode(out)},
                            enable_chain=enable_chain, chain_backend=chain_backend)
    return out, res
