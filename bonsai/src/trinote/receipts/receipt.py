"""Build a per-inference triple-entry TEA receipt (`trinote.receipt/v2`) — LOCALLY.

SCHEMA NOTE (v2): the committed sampler block is now FULLY INTEGER — temperature/top-p are committed as
their fixed-point images `invTempFp = round(2^f / T)` and `topPFp = round(top_p · 2^f)` (with `fpFracBits`
pinning the scale), instead of IEEE floats. This removes the last float from the committed preimage, so
`canonical_bytes(receipt)` is byte-reproducible in ANY language (no CPython float-repr dependency). v1
(float `temperature`/`topP`) is still reproducible via `build_receipt(schema_version="v1")` so historical
receipts (e.g. the packaged tensor demo) can be re-derived bit-for-bit.

The `receipt` is the on-chain-committable half: commitments + signatures + receiptHash, with **no raw
text** (the on-chain/off-chain split; raw ids live in the off-chain `preimage` — see
docs/receipts/RECEIPTS.md). Three entries:

  1st  sigModel         model key signs (modelHash, inputCommit, outputCommit, traceCommit)
  2nd  sigCounterparty  caller key co-signs (modelHash, inputCommit, outputCommit)  — only what it saw
  3rd  sigLedger        recorded by receipts/emit.py → the LOCAL ledger; on-chain broadcast is DISABLED.

`build_receipt` returns a BUNDLE `{"receipt": …, "preimage": …}`: the receipt is the committable half,
the preimage is the off-chain record (the raw token ids + sampler + trace) a verifier needs to
re-execute (receipts/verify.py). The MI `trace` is committed but EMPTY by construction — the MI pillar
is P5/pending, so it carries no circuit claims yet (`miStatus="pending"`); the field is wired so a
receipt schema does not change when MI lands.
"""
from __future__ import annotations

from .canonical import canonical_bytes, commit, token_commit
from .signing import LocalKey, sign
from ..hashing.sha import sha256_hex
from ..infer_int.sampler import RECEIPT_SAFE_MODES, inv_temp_fp, top_p_fp, min_p_fp

SCHEMA = "trinote.receipt/v2"
PREIMAGE_SCHEMA = "trinote.receipt-preimage/v2"
SCHEMA_V1 = "trinote.receipt/v1"
PREIMAGE_SCHEMA_V1 = "trinote.receipt-preimage/v1"


def _assert_no_floats_v2(obj, path: str) -> None:
    """Reject IEEE floats anywhere in a committed receipt/v2 substructure. A float would make
    canonical_bytes() depend on CPython float repr and break v2's language-neutral commitment guarantee —
    commit a fixed-point int instead. (bool is an int subclass and is allowed.)"""
    if isinstance(obj, bool):
        return
    if isinstance(obj, float):
        raise ValueError(f"receipt/v2 forbids floats in the committed preimage: {path} = {obj!r}; "
                         f"commit a fixed-point integer instead")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_floats_v2(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_no_floats_v2(v, f"{path}[{i}]")


def _sampler_fields(sampler):
    """(mode, temperature, top_k, top_p, seed, rep_penalty, no_repeat_ngram, inv_committed, topp_committed)
    from a SamplerConfig or a snake/camel dict (used by the non-v2-dict paths)."""
    if isinstance(sampler, dict):
        g = sampler.get
        return (g("mode", "greedy"), float(g("temperature", 1.0)),
                int(g("topK", g("top_k", 0)) or 0), float(g("topP", g("top_p", 1.0))),
                int(g("seed", 0) or 0), int(g("repPenalty", g("rep_penalty", 0)) or 0),
                int(g("noRepeatNgram", g("no_repeat_ngram", 0)) or 0),
                g("invTempFp"), g("topPFp"))
    return (getattr(sampler, "mode", "greedy"), float(getattr(sampler, "temperature", 1.0)),
            int(getattr(sampler, "top_k", 0) or 0), float(getattr(sampler, "top_p", 1.0)),
            int(getattr(sampler, "seed", 0) or 0), int(getattr(sampler, "rep_penalty", 0) or 0),
            int(getattr(sampler, "no_repeat_ngram", 0) or 0),
            getattr(sampler, "inv_temp_fp_committed", None), getattr(sampler, "top_p_fp_committed", None))


def _sampler_min_p(sampler) -> tuple[float, int | None]:
    """(min_p float, committed minPFp int|None) from a SamplerConfig or snake/camel dict."""
    if isinstance(sampler, dict):
        g = sampler.get
        return float(g("minP", g("min_p", 0.0)) or 0.0), g("minPFp")
    return float(getattr(sampler, "min_p", 0.0) or 0.0), getattr(sampler, "min_p_fp_committed", None)


def sampler_to_block(sampler, frac_bits: int = 16) -> dict:
    """Normalize a SamplerConfig / sampler dict to the COMMITTED receipt/v2 sampler block — fully integer
    (no IEEE floats). temperature/top-p are committed as `invTempFp = round(2^f / T)` and
    `topPFp = round(top_p · 2^f)`; `fpFracBits` pins the scale. `minPFp = round(min_p · 2^f)` is added ONLY
    for min-p turns, so every non-min-p block (and its receiptHash) is byte-identical to before. Accepts an
    already-v2 dict (pass-through, preserving its `fpFracBits`), a legacy v1 float dict, or a SamplerConfig
    (using its committed ints if present). The canonical normalizer the verifier uses to compare receipt vs
    preimage."""
    if isinstance(sampler, dict) and "invTempFp" in sampler:        # already v2 — normalize / pass-through
        g = sampler.get
        f = int(g("fpFracBits", frac_bits))
        block = {"mode": g("mode", "greedy"), "fpFracBits": f,
                 "invTempFp": int(g("invTempFp")), "topPFp": int(g("topPFp", 1 << f)),
                 "topK": int(g("topK", 0) or 0), "seed": int(g("seed", 0) or 0),
                 "repPenalty": int(g("repPenalty", 0) or 0), "noRepeatNgram": int(g("noRepeatNgram", 0) or 0)}
        if g("minPFp") is not None:                              # carry min-p only if present (block-compat)
            block["minPFp"] = int(g("minPFp"))
        return block
    mode, temperature, top_k, top_p, seed, rep_penalty, no_repeat_ngram, inv_c, topp_c = _sampler_fields(sampler)
    min_p, min_c = _sampler_min_p(sampler)
    f = int(frac_bits)
    inv_fp = int(inv_c) if inv_c is not None else (inv_temp_fp(temperature, f) if temperature and temperature > 0 else (1 << f))
    topp_fp = int(topp_c) if topp_c is not None else top_p_fp(top_p, f)
    block = {"mode": mode, "fpFracBits": f, "invTempFp": int(inv_fp), "topPFp": int(topp_fp),
             "topK": int(top_k), "seed": int(seed),
             "repPenalty": int(rep_penalty), "noRepeatNgram": int(no_repeat_ngram)}
    # Add minPFp ONLY for min-p turns so existing (non-min-p) receipt blocks + hashes are unchanged.
    if mode == "min_p" or min_c is not None or (min_p and float(min_p) > 0):
        block["minPFp"] = int(min_c) if min_c is not None else min_p_fp(min_p, f)
    return block


def _sampler_to_block_v1(sampler) -> dict:
    """Legacy receipt/v1 sampler block (float `temperature`/`topP`). Retained ONLY so `build_receipt` can
    reproduce a historical v1 receiptHash byte-for-byte (e.g. the packaged tensor demo replay)."""
    mode, temperature, top_k, top_p, seed, rep_penalty, no_repeat_ngram, _i, _t = _sampler_fields(sampler)
    return {"mode": mode, "temperature": float(temperature), "topK": int(top_k), "topP": float(top_p),
            "seed": int(seed), "repPenalty": int(rep_penalty), "noRepeatNgram": int(no_repeat_ngram)}


def build_receipt(*, model_hash: str, input_ids, output_ids, sampler,
                  model_key: LocalKey, counterparty_key: LocalKey,
                  trace: dict | None = None, model_label: str = "",
                  artifact_digest: str | None = None,
                  fp_frac_bits: int = 16, schema_version: str = "v2") -> dict:
    """Build a receipt BUNDLE for one inference turn. Pure: no I/O, no chain. Greedy → receiptBound.

    `schema_version="v2"` (default) commits a fully-integer sampler block (float-free, language-neutral);
    `fp_frac_bits` is the fixed-point scale (must match the engine's `model.cfg["frac"]`). Pass
    `schema_version="v1"` ONLY to reproduce a historical float-block receiptHash byte-for-byte."""
    if schema_version not in ("v1", "v2"):              # fail loud, never silently fall through to v2
        raise ValueError(f"unknown receipt schema_version {schema_version!r} (expected 'v1' or 'v2')")
    input_commit = token_commit(input_ids)
    output_commit = token_commit(output_ids)
    if schema_version == "v1":
        sampler_block = _sampler_to_block_v1(sampler)
        schema, preimage_schema = SCHEMA_V1, PREIMAGE_SCHEMA_V1
    else:
        sampler_block = sampler_to_block(sampler, fp_frac_bits)
        schema, preimage_schema = SCHEMA, PREIMAGE_SCHEMA
    # Receipt-bound iff the sampler is re-executable. Greedy (argmax) and the SEEDED integer samplers
    # (temp/top-k/top-p) are all bit-exactly re-derivable — the seed is committed in the block and the
    # draw is the counter-based integer Lemire draw (sampler.py / docs/architecture/SAMPLER-INTEGER.md).
    receipt_bound = sampler_block["mode"] in RECEIPT_SAFE_MODES

    trace = trace or {}
    trace_record = {
        "topkFeatures": list(trace.get("topkFeatures", [])),         # §4 MI — EMPTY until P5
        "attributableShards": list(trace.get("attributableShards", [])),
        "sampler": sampler_block,
        "miStatus": trace.get("miStatus", "pending"),                # honest: MI attribution not wired
    }
    if schema_version == "v2":
        # The v2 float-free / language-neutral guarantee covers the WHOLE committed preimage, not just the
        # sampler block. The MI fields are empty today (P5/pending), but enforce the invariant now so that
        # when MI lands it must commit fixed-point ints (e.g. actFp=round(act·2^f)) rather than IEEE floats
        # — a raw float here would re-introduce a CPython-repr dependency in traceCommit→receiptHash.
        _assert_no_floats_v2(trace_record["topkFeatures"], "trace.topkFeatures")
        _assert_no_floats_v2(trace_record["attributableShards"], "trace.attributableShards")
    trace_block = dict(trace_record)
    trace_block["traceCommit"] = commit(trace_record)

    # 1st entry — model signs the full claim including the trace commitment. `key.sign` is polymorphic:
    # a LocalKey emits a symmetric HMAC vouch; an ECKey emits a third-party-verifiable secp256k1 signature.
    model_msg = canonical_bytes({"modelHash": model_hash, "inputCommit": input_commit,
                                 "outputCommit": output_commit,
                                 "traceCommit": trace_block["traceCommit"]})
    sig_model = model_key.sign(model_msg)
    # 2nd entry — counterparty co-signs only the input/output it observed (not the internal trace).
    cp_msg = canonical_bytes({"modelHash": model_hash, "inputCommit": input_commit,
                              "outputCommit": output_commit})
    sig_counterparty = counterparty_key.sign(cp_msg)

    body = {
        "schema": schema,
        "modelHash": model_hash,
        "inputCommit": input_commit,
        "outputCommit": output_commit,
        "receiptBound": receipt_bound,
        "trace": trace_block,
        "sigModel": sig_model,
        "sigModelKeyId": model_key.key_id,
        "sigCounterparty": sig_counterparty,
        "sigCounterpartyKeyId": counterparty_key.key_id,
    }
    # For ASYMMETRIC keys, commit the signer's PUBLIC key so any third party can verify with no secret.
    # (Only added for EC keys — HMAC receipts are byte-unchanged, preserving legacy/demo receiptHashes.)
    model_pub = getattr(model_key, "public_hex", None)
    if model_pub:
        body["sigModelPubKey"] = model_pub
    cp_pub = getattr(counterparty_key, "public_hex", None)
    if cp_pub:
        body["sigCounterpartyPubKey"] = cp_pub
    receipt = dict(body)
    receipt["receiptHash"] = sha256_hex(canonical_bytes(body))   # commits the signed pair (3rd-entry payload)

    preimage = {
        "schema": preimage_schema,
        "receiptHash": receipt["receiptHash"],
        "modelHash": model_hash,
        "modelLabel": model_label,
        "artifactDigest": artifact_digest,
        "inputIds": [int(i) for i in input_ids],
        "outputIds": [int(i) for i in output_ids],
        "sampler": sampler_block,
        "trace": trace_record,
    }
    return {"receipt": receipt, "preimage": preimage}


def receipt_hash(receipt: dict) -> str:
    """Recompute a receipt's `receiptHash` from its body (everything except the field itself)."""
    body = {k: v for k, v in receipt.items() if k != "receiptHash"}
    return sha256_hex(canonical_bytes(body))
