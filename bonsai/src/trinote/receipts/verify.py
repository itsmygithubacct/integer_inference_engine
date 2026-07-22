"""Re-verify a local TEA receipt bundle — commitments, receiptHash, signatures, and re-execution.

The trustless core (docs/receipts/RECEIPTS.md): a verifier does not BELIEVE a receipt, it RECOMPUTES
it. Given the off-chain `preimage` (the raw token ids) and the reference model, `verify_receipt`:

  0. binds the committed modelHash to the weights actually re-executed                 (no secret needed)
  1. recomputes inputCommit / outputCommit from the ids   → must equal the receipt   (no secret needed)
  2. recomputes receiptHash from the receipt body         → must equal the committed   (no secret needed)
  3. re-runs the bit-exact forward (infer_int.verify) → output must re-derive  (any receipt-bound sampler)
  4. (optional) re-checks the local-hmac signatures       → needs the shared secret(s)

Steps 0–3 are the re-execution core and need NO key. Step 4 (the 1st/2nd-entry vouches) is only
checkable by a holder of the symmetric secret (receipts/signing.py) — supply the keys to include it.
On-chain attestation/settlement is out of scope here (disabled); this is the LOCAL re-execution check.

WHAT `structuralOk` (steps 0–2 + trace/sampler shape) PROVES: SELF-CONSISTENCY, not AUTHENTICITY. The
commitments, receiptHash, and traceCommit are all recomputed from the receipt's OWN fields, so with no
keys supplied an attacker who controls the bundle can rewrite the trace/sampler, recompute traceCommit
→ receiptHash → the commits, and reach `structuralOk=True` over fabricated content. Body tampering is
caught only by (a) the HMAC signatures (step 4) OR (b) the artifact-binding + bit-exact re-execution
gate. That is why `ok`/`fullyVerified` REQUIRE `reexecOk` and `artifactBoundOk` (and never let a
signature be False) — `structuralOk` alone is necessary but not sufficient. See
docs/receipts/RECEIPTS.md and docs/architecture/DETERMINISM.md.

THE BINDING (step 0) is what makes step 3's "re-derived from the COMMITTED weights" a fact rather than
an assumption: re-execution alone only proves *the supplied model* reproduces the output, not that that
model IS the one named by `modelHash`. Two checks close the gap, both additive (only run when the data
is present, so signature-free / digest-free callers stay backward-compatible):
  * `artifactBindingOk` — the off-chain preimage's `artifactDigest` must equal the committed `modelHash`
    (receipt and preimage name the SAME artifact). Runs whenever the preimage carries an artifactDigest.
  * `modelHashMatch`    — the digest of the artifact actually loaded for re-execution (pass
    `model_digest=load_artifact_v2(path)[1]["digest"]`) must equal the committed `modelHash`.
"""
from __future__ import annotations

import re

from .canonical import canonical_bytes, commit, token_commit
from .receipt import receipt_hash, sampler_to_block
from .signing import LocalKey, verify_signature
from .signing_ec import SCHEME_EC as EC_SCHEME
from ..infer_int.verify import verify_greedy, verify_resample, verify_sampled
from ..infer_int.sampler import RECEIPT_SAFE_MODES, sampler_config_from_block


def _invalid_result(error: str) -> dict:
    return {
        "ok": False,
        "fullyVerified": False,
        "structuralOk": False,
        "reexecOk": False,
        "auditedOk": False,
        "artifactBoundOk": False,
        "signatureOk": None,
        "commitMatch": False,
        "receiptHashMatch": False,
        "traceCommitMatch": False,
        "receiptBound": False,
        "committedSamplerPresent": False,
        "receiptBoundModeOk": False,
        "reexec": None,
        "error": error,
    }


def _token_ids(value, field: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list of token ids")
    out = []
    for i, v in enumerate(value):
        if isinstance(v, bool):
            raise ValueError(f"{field}[{i}] must be an integer token id, not bool")
        try:
            iv = int(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field}[{i}] must be an integer token id") from exc
        if iv < 0:
            raise ValueError(f"{field}[{i}] must be >= 0")
        out.append(iv)
    return out


def _validate_vocab(model, ids: list[int]) -> None:
    if model is None or not ids:
        return
    vocab = int(model.cfg.get("vocab", 0))
    if vocab <= 0:
        return
    bad = next((i for i in ids if i >= vocab), None)
    if bad is not None:
        raise ValueError(f"token id {bad} is outside model vocab size {vocab}")


def _validate_ec_pubkey_pin(value: str | None, field: str) -> str | None:
    """Reject ambiguous identity pins before signature verification.

    A caller-supplied pin is a security boundary, so false-y strings must not
    silently select the receipt's attacker-controlled embedded key.  Pins use
    the same canonical compressed secp256k1 representation emitted by ECKey.
    """
    if value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r"(?:02|03)[0-9a-f]{64}", value) is None:
        raise ValueError(
            f"{field} must be canonical lowercase compressed secp256k1 public-key hex"
        )
    return value


def verify_receipt(bundle: dict, **kwargs) -> dict:
    """Fail-closed wrapper (review finding #10): any unexpected KeyError/TypeError/ValueError/
    IndexError raised by a malformed or ADVERSARIAL bundle returns a result dict with ok:False
    (via _invalid_result) instead of escaping. A public ledger-sweep verifier must never be
    crashable/DoS-able by one crafted bundle. The full verification logic is in
    _verify_receipt_impl; only its initial extraction was previously guarded."""
    try:
        return _verify_receipt_impl(bundle, **kwargs)
    except (KeyError, TypeError, ValueError, IndexError) as e:
        return _invalid_result(str(e))


def _verify_receipt_impl(bundle: dict, *, model=None, model_digest: str | None = None,
                   model_key: LocalKey | None = None,
                   counterparty_key: LocalKey | None = None,
                   model_pubkey: str | None = None,
                   counterparty_pubkey: str | None = None,
                   sample_k: int = 0, sample_seed: int = 0) -> dict:
    """Verify a `build_receipt` bundle. `model` enables re-execution; `model_digest` binds those weights to
    the committed modelHash. For ASYMMETRIC (secp256k1) receipts, signatures verify from the committed public
    key with NO secret — pass `model_pubkey`/`counterparty_pubkey` to PIN the expected signer (identity
    binding). For legacy HMAC receipts, pass the shared `model_key`/`counterparty_key` to check the vouches.

    `sample_k > 0` runs a PROBABILISTIC AUDIT of a GREEDY receipt: only `k` deterministically-chosen output
    positions (Philox-keyed by `sample_seed`) are re-derived instead of all N — a fast, lower-assurance
    screening tier for ledger-wide sweeps. The re-exec result then carries strategy=`greedy-sampled`,
    `checked=k`, `of=N`, and `sampled=True`; it is NOT a full verification (don't read it as such). Ignored
    for seeded sampler modes (which always do the full replay)."""
    try:
        model_pubkey = _validate_ec_pubkey_pin(model_pubkey, "model_pubkey")
        counterparty_pubkey = _validate_ec_pubkey_pin(
            counterparty_pubkey, "counterparty_pubkey"
        )
        if not isinstance(bundle, dict):
            raise ValueError("bundle must be a dict")
        receipt = bundle["receipt"]
        preimage = bundle["preimage"]
        if not isinstance(receipt, dict) or not isinstance(preimage, dict):
            raise ValueError("bundle receipt and preimage must be dicts")
        in_ids = _token_ids(preimage["inputIds"], "preimage.inputIds")
        out_ids = _token_ids(preimage["outputIds"], "preimage.outputIds")
        # A trustless generation receipt MUST commit a non-empty prompt: re-execution derives output[i]
        # from row (len(input)+i-1), so an empty inputIds makes output[0]'s predicting row index -1 —
        # NumPy would wrap that to the last prefill row and a crafted bundle could reach fullyVerified for
        # a token the model never produced. The engine never generates from an empty context, so reject it.
        if out_ids and not in_ids:
            raise ValueError("preimage.inputIds is empty — a generation receipt must commit a non-empty prompt")
        _validate_vocab(model, in_ids + out_ids)
        model_hash = str(receipt["modelHash"])
    except (KeyError, TypeError, ValueError) as e:
        return _invalid_result(str(e))

    commit_ok = (token_commit(in_ids) == receipt["inputCommit"]
                 and token_commit(out_ids) == receipt["outputCommit"])
    hash_ok = receipt_hash(receipt) == receipt["receiptHash"]
    bound = bool(receipt.get("receiptBound", False))
    trace = receipt.get("trace") or {}
    trace_record = {k: v for k, v in trace.items() if k != "traceCommit"}
    trace_commit_ok = bool(trace) and commit(trace_record) == trace.get("traceCommit")
    committed_sampler = trace.get("sampler")
    sampler_present = isinstance(committed_sampler, dict)
    receipt_bound_mode_ok = sampler_present and bound == (
        committed_sampler.get("mode", "greedy") in RECEIPT_SAFE_MODES
    )

    result = {"ok": False, "fullyVerified": False, "commitMatch": commit_ok, "receiptHashMatch": hash_ok,
              "traceCommitMatch": trace_commit_ok, "receiptBound": bound,
              "committedSamplerPresent": sampler_present,
              "receiptBoundModeOk": receipt_bound_mode_ok}

    # The sampler used for re-execution must be the sampler committed in the receipt trace. The preimage
    # repeats it for convenience, but it is off-chain mutable material; treat any drift as a verification
    # failure instead of silently replaying the uncommitted value. Both sides are normalized through the
    # SAME function so a receipt/v2 (integer) and a legacy v1 (float) block each compare self-consistently;
    # `fpFracBits` from the committed block pins the scale (v1 blocks have none → the standard 16).
    _frac = int((committed_sampler or {}).get("fpFracBits", 16) or 16) if sampler_present else 16
    committed_norm = sampler_to_block(committed_sampler, _frac) if sampler_present else None
    preimage_sampler = preimage.get("sampler")
    if preimage_sampler is not None and sampler_present:
        result["preimageSamplerMatch"] = (sampler_to_block(preimage_sampler, _frac) == committed_norm)
    preimage_trace_sampler = (preimage.get("trace") or {}).get("sampler")
    if preimage_trace_sampler is not None and sampler_present:
        result["preimageTraceSamplerMatch"] = (
            sampler_to_block(preimage_trace_sampler, _frac) == committed_norm
        )

    # 0. identity binding — the re-executed/referenced artifact must BE the committed modelHash, not
    #    merely "some model that happens to reproduce these tokens". Both checks are additive.
    art_digest = preimage.get("artifactDigest")
    if art_digest is not None:
        result["artifactBindingOk"] = (art_digest == model_hash)
    if model_digest is not None:
        result["modelHashMatch"] = (model_digest == model_hash)

    # 3. bit-exact re-execution. Every receipt-bound sampler is re-derivable: greedy by argmax, the
    #    seeded modes (temp/top-k/top-p) by replaying the committed seed + absolute position over the
    #    fully-integer draw (docs/architecture/SAMPLER-INTEGER.md). Dispatch on the committed mode.
    if model is not None and bound and sampler_present:
        s = committed_sampler
        # Make the committed fixed-point scale LOAD-BEARING: re-execution shifts by the engine's
        # model.cfg["frac"], so a receipt/v2 block's committed fpFracBits MUST equal it or the committed
        # invTempFp/topPFp would be applied at the wrong scale (silent token drift, or a forged scale
        # slipping past). Fail closed on mismatch instead of replaying against a different distribution.
        committed_frac = s.get("fpFracBits")
        model_frac = int(getattr(model, "cfg", {}).get("frac", 0) or 0)
        if committed_frac is not None and model_frac:
            result["fpFracBitsMatch"] = (int(committed_frac) == model_frac)
        try:
            if committed_frac is not None and model_frac and int(committed_frac) != model_frac:
                result["reexec"] = {"ok": False, "checked": 0, "strategy": "fpfracbits-mismatch",
                                    "error": f"committed fpFracBits {committed_frac} != engine frac {model_frac}"}
            elif s.get("mode", "greedy") == "greedy":
                if sample_k and int(sample_k) > 0:
                    rx = verify_sampled(model, in_ids, out_ids, k=int(sample_k), seed=int(sample_seed),
                                        rep_penalty_fp=int(s.get("repPenalty", 0)),
                                        no_repeat_ngram=int(s.get("noRepeatNgram", 0)))
                    result["reexec"] = {"ok": rx["ok"], "checked": rx["checked"], "of": rx.get("of"),
                                        "strategy": rx["strategy"], "sampled": True}
                else:
                    rx = verify_greedy(model, in_ids, out_ids, rep_penalty_fp=int(s.get("repPenalty", 0)),
                                       no_repeat_ngram=int(s.get("noRepeatNgram", 0)))
                    result["reexec"] = {"ok": rx["ok"], "checked": rx["checked"], "strategy": rx["strategy"]}
            else:
                rx = verify_resample(model, in_ids, out_ids, sampler_cfg=sampler_config_from_block(s))
                result["reexec"] = {"ok": rx["ok"], "checked": rx["checked"], "strategy": rx["strategy"]}
        except (OverflowError, ValueError, IndexError, KeyError, TypeError) as e:
            result["reexec"] = {"ok": False, "checked": 0, "strategy": "reexec-error", "error": str(e)}
    else:
        result["reexec"] = None

    # 4. signatures. ASYMMETRIC (secp256k1-ecdsa) verifies with the PUBLIC key — committed in the receipt
    #    (`sigModelPubKey`) or pinned by the caller (`model_pubkey` = the identity's authorized key) — with
    #    NO secret, so a third party can check it. SYMMETRIC (local-hmac) still needs the shared key.
    #
    #    CRITICAL (#10): "signature valid" != "signer authenticated". When the caller does NOT pin
    #    `model_pubkey`, the EC signature is checked only against the key EMBEDDED IN the receipt, so an
    #    attacker who controls the bundle can self-sign a forged receipt with a fresh key and reach
    #    sigModelOk=True. A True signature is therefore SELF-CONSISTENT, not AUTHENTIC. We surface that
    #    distinction explicitly: `sig*Authenticated` is True only when an EXTERNAL authenticator confirmed the
    #    signer — a pinned pubkey (EC) / supplied shared secret (HMAC) that the signature passed against — OR
    #    (for the model entry) the modelHash re-execution binding is established. `warnings` carries a loud
    #    string whenever an EC signature was accepted unpinned.
    warnings: list[str] = []
    model_sig_pinned = False
    cp_sig_pinned = False
    sig_model = receipt.get("sigModel")
    if sig_model and str(sig_model).startswith(EC_SCHEME + ":"):
        msg = canonical_bytes({"modelHash": receipt["modelHash"], "inputCommit": receipt["inputCommit"],
                               "outputCommit": receipt["outputCommit"],
                               "traceCommit": receipt["trace"]["traceCommit"]})
        result["sigModelPubKey"] = receipt.get("sigModelPubKey")
        expected_model_pubkey = (
            model_pubkey if model_pubkey is not None else receipt.get("sigModelPubKey")
        )
        result["sigModelOk"] = verify_signature(
            msg, sig_model, expected_pubkey=expected_model_pubkey
        )
        model_sig_pinned = model_pubkey is not None
        if not model_sig_pinned and result["sigModelOk"]:
            warnings.append(
                "sigModel is valid only against the pubkey EMBEDDED in the receipt (no pinned model_pubkey "
                "supplied): the signature is self-consistent but the SIGNER IS NOT AUTHENTICATED — a forged "
                "receipt can be self-signed with a fresh key. Pin model_pubkey, or rely on the modelHash "
                "re-execution binding (modelHashMatch), for authenticity.")
    elif model_pubkey is not None:
        # Supplying an expected identity is a requirement, not a hint.  A
        # missing signature (or one using an unsupported scheme) must not be
        # treated like the backward-compatible "no signatures requested"
        # case below.
        result["sigModelOk"] = False
        model_sig_pinned = True
    elif model_key is not None:
        msg = canonical_bytes({"modelHash": receipt["modelHash"], "inputCommit": receipt["inputCommit"],
                               "outputCommit": receipt["outputCommit"],
                               "traceCommit": receipt["trace"]["traceCommit"]})
        result["sigModelOk"] = verify_signature(msg, sig_model, key=model_key)
        model_sig_pinned = True                 # caller supplied the producing secret = an external authenticator
    sig_cp = receipt.get("sigCounterparty")
    if sig_cp and str(sig_cp).startswith(EC_SCHEME + ":"):
        msg = canonical_bytes({"modelHash": receipt["modelHash"], "inputCommit": receipt["inputCommit"],
                               "outputCommit": receipt["outputCommit"]})
        result["sigCounterpartyPubKey"] = receipt.get("sigCounterpartyPubKey")
        expected_counterparty_pubkey = (
            counterparty_pubkey
            if counterparty_pubkey is not None
            else receipt.get("sigCounterpartyPubKey")
        )
        result["sigCounterpartyOk"] = verify_signature(
            msg, sig_cp, expected_pubkey=expected_counterparty_pubkey
        )
        cp_sig_pinned = counterparty_pubkey is not None
        if not cp_sig_pinned and result["sigCounterpartyOk"]:
            warnings.append(
                "sigCounterparty is valid only against the pubkey EMBEDDED in the receipt (no pinned "
                "counterparty_pubkey supplied): the signature is self-consistent but the SIGNER IS NOT "
                "AUTHENTICATED. Pin counterparty_pubkey to bind it to an expected identity.")
    elif counterparty_pubkey is not None:
        result["sigCounterpartyOk"] = False
        cp_sig_pinned = True
    elif counterparty_key is not None:
        msg = canonical_bytes({"modelHash": receipt["modelHash"], "inputCommit": receipt["inputCommit"],
                               "outputCommit": receipt["outputCommit"]})
        result["sigCounterpartyOk"] = verify_signature(msg, sig_cp, key=counterparty_key)
        cp_sig_pinned = True

    structural_checks = [commit_ok, hash_ok, trace_commit_ok, sampler_present, receipt_bound_mode_ok]
    structural_checks += [result[k] for k in ("preimageSamplerMatch", "preimageTraceSamplerMatch") if k in result]
    result["structuralOk"] = all(structural_checks)
    # A SAMPLED re-exec (sample_k>0) re-derives only k of N output positions, so a forged receipt whose
    # tampered tokens fall OUTSIDE the sample would still re-derive cleanly: it is a probabilistic AUDIT,
    # not a full re-execution. Surface its pass as `auditedOk`, but NEVER let it satisfy `reexecOk` — the
    # strong signal every top-level verdict gates on (fullyVerified/ok below; and, downstream, the re-exec
    # layer `ok` + the 'RESULT: VERIFIED' line in bundle/verify.py + cli/receipt_bundle_cli.py, which read
    # reexecOk). The strong verdict therefore REQUIRES full coverage; the default full re-exec (sample_k=0)
    # is unchanged (not sampled → reexecOk == the re-exec pass, exactly as before).
    reexec_passed = bool(result["reexec"] and result["reexec"].get("ok"))
    reexec_sampled = bool(result["reexec"] and result["reexec"].get("sampled"))
    result["auditedOk"] = reexec_passed                  # partial/probabilistic pass (k-of-N when sampled)
    result["reexecOk"] = reexec_passed and not reexec_sampled
    result["artifactBoundOk"] = bool(result.get("artifactBindingOk") and result.get("modelHashMatch"))
    sig_values = [result[k] for k in ("sigModelOk", "sigCounterpartyOk") if k in result]
    result["signatureOk"] = all(sig_values) if sig_values else None

    # AUTHENTICATION (#10) — distinct from "signature valid". The model entry is authenticated either by a
    # pinned/secret-backed signature check OR by the modelHash re-execution binding (which proves the very
    # weights named by modelHash reproduce the output, so the producer's identity claim is grounded in the
    # committed artifact rather than a self-chosen key). The counterparty entry has no re-exec binding, so it
    # is authenticated ONLY by pinning. Self-signed-but-unpinned EC signatures stay sigModelOk=True yet
    # sigModelAuthenticated=False, with a loud `warnings` entry.
    if "sigModelOk" in result:
        result["sigModelAuthenticated"] = bool(
            (model_sig_pinned and result["sigModelOk"])
            # The re-execution binding only AUTHENTICATES if the named weights actually reproduce the
            # output: modelHashMatch (artifact identity) alone is not enough — pair it with reexecOk.
            or (result.get("modelHashMatch") and result.get("reexecOk")))
    if "sigCounterpartyOk" in result:
        result["sigCounterpartyAuthenticated"] = bool(cp_sig_pinned and result["sigCounterpartyOk"])
    if warnings:
        result["warnings"] = warnings

    # `fullyVerified` already CANNOT be True without `artifactBoundOk`, which REQUIRES `modelHashMatch` (the
    # re-execution binding). So the strongest verdict is never reachable on signature self-consistency alone —
    # it always carries an authenticated model entry (pinned match OR modelHash re-exec match). Keep that gate.
    # It also REQUIRES `reexecOk`, which (above) is False for a sampled audit — so a partial-coverage screen
    # can never reach fullyVerified/ok; only a FULL re-execution does (see `auditedOk` for the partial pass).
    # #21 (forward-looking guard): re-execution re-derives only the OUTPUT TOKENS, never the
    # committed MI/interpretability trace (topkFeatures/attributableShards). Today miStatus is
    # 'pending' with empty MI fields so this is inert, but once MI becomes LOAD-BEARING an actor
    # with the local model could strip signatures, FABRICATE an MI trace, recompute the commits, and
    # reach fullyVerified on signature self-consistency alone. So whenever a non-pending / non-empty
    # MI trace is present, REQUIRE an AUTHENTICATED model signature (not merely 'signatureOk is not
    # False') for the strongest verdict. Until MI lands this leaves current behavior unchanged.
    _mi_present = bool(
        (isinstance(trace, dict) and trace.get("miStatus", "pending") != "pending")
        or (trace.get("topkFeatures") if isinstance(trace, dict) else None)
        or (trace.get("attributableShards") if isinstance(trace, dict) else None)
    )
    result["miTraceGateOk"] = (not _mi_present) or bool(result.get("sigModelAuthenticated"))

    result["fullyVerified"] = (
        result["structuralOk"]
        and result["reexecOk"]
        and result["artifactBoundOk"]
        and result["signatureOk"] is not False
        and result["miTraceGateOk"]
    )
    result["ok"] = result["fullyVerified"]
    return result
