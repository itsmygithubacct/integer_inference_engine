"""Receipt verification over the bit-exact reference engine — full, sampled, and challenged.

A greedy receipt (`receipts.build_receipt`) commits `inputIds` and `outputIds`; verification re-derives
the continuation on the canonical integer reference engine (`ReferenceModelV2`) and checks every output
token is the argmax of its predicting row — this is the mechanism that makes the TEA "third entry"
trustless (docs/DESIGN.md §5.4): the counterparty does not believe the receipt, they recompute it.
`receipts.verify.verify_receipt` calls `verify_greedy` here as its re-execution step.

Strategies, cheapest-useful first:
  * `verify_greedy`   — small turns use ONE teacher-forced prefill of input+output. Long Bonsai turns can
    opt into KV-cached replay, which avoids a full all-position Bonsai forward while checking the same
    committed continuation.
  * `verify_resample` — re-derive a SEEDED sampled turn (temp/top-k/top-p) by replaying the committed
    integer sampler at each position. Long opted-in models use the same KV-cached replay strategy.
  * `verify_sampled`  — check only a deterministic random subset of positions (a cheap probabilistic
    audit when re-deriving every position is still too costly at 1B scale).
  * `challenge_position` — INDEPENDENTLY re-derive one disputed position from scratch (prefill the
    exact committed prefix, read the last row), the primitive a dispute resolver uses.

Every sampler mode is now receipt-bound (greedy by argmax, the seeded modes by replaying seed+position),
so all receiptBound turns are verifiable here.
"""
from __future__ import annotations

import numpy as np

from .sampler import logits_digest, apply_rep_penalty, sample_token, SamplerConfig


def _penalized_argmax(model, row, history, rep_penalty_fp, no_repeat_ngram) -> int:
    """argmax of the row AFTER the deterministic repetition penalty (history = tokens before this one).
    Identical transform to loop._run / reference.generate_greedy, so a penalized greedy turn re-derives
    bit-exactly. The committed `logitsDigest` stays the RAW row (the penalty is a selection transform,
    not a model output)."""
    if rep_penalty_fp or no_repeat_ngram:
        row = apply_rep_penalty(row, history, rep_penalty_fp, no_repeat_ngram, int(model.cfg["frac"]))
    return int(row.argmax())


def _eff_ctx(model) -> int:
    """Effective window the engine ACTUALLY uses = min(context_len, committed RoPE-table rows).

    `ReferenceModelV2.forward` asserts T <= cos_fp rows, and the receipt-producing engine
    (`ReferenceModelV2.generate_greedy`) slides exactly the context window — so verification MUST clamp
    identically, or it crashes (a single prefill of a too-long turn) and mismatches how it was produced.
    """
    return min(int(model.cfg["context_len"]), int(model.artifact["cos_fp"].shape[0]))


def teacher_forced_logits(model, ids) -> np.ndarray:
    """All per-position fixed-point logits (T, vocab) in ONE forward pass. Caller ensures T <= eff_ctx."""
    if hasattr(model, "teacher_forced_logits"):
        return model.teacher_forced_logits(list(ids))
    return model.forward(list(ids))


def _cached_replay_threshold(model) -> int | None:
    """Model opt-in for long-turn receipt verification by KV-cached replay.

    Teacher-forced verification is still the best small-turn strategy because it produces per-position raw
    logits and digests in one call. For long Bonsai turns, the full prefill is much slower and memory-heavier
    than replaying the already-validated cached decode path. Models opt in by defining
    `receipt_verify_cached_threshold`.
    """
    threshold = getattr(model, "receipt_verify_cached_threshold", None)
    if threshold is None or not hasattr(model, "generate_cached"):
        return None
    return max(1, int(threshold))


def _full_verification_strategy(model, input_count: int, output_count: int) -> str:
    """Resolve the exact full-verification algorithm for a committed turn.

    ``receipt_verify_strategy`` is an operator-selected, evidence-backed override
    used by the bundle verifier.  It changes only how the same deterministic
    logits/tokens are recomputed; it never weakens full coverage.  ``auto``
    preserves the model's measured cached-replay threshold.
    """
    selected = str(getattr(model, "receipt_verify_strategy", "auto") or "auto")
    if selected not in {"auto", "teacher-forced", "cached-replay"}:
        raise ValueError(f"unknown receipt verification strategy {selected!r}")
    if selected == "teacher-forced":
        return selected
    threshold = _cached_replay_threshold(model)
    if selected == "cached-replay":
        if threshold is None:
            raise ValueError("cached-replay requested but this model has no exact cached generator")
        return selected
    if threshold is not None and int(output_count) >= threshold:
        return "cached-replay"
    return "teacher-forced"


def _cached_replay_positions(input_ids, output_ids, replayed) -> tuple[bool, list[dict]]:
    n_in = len(input_ids)
    positions = []
    ok = len(replayed) == len(output_ids)
    for i, expected in enumerate(output_ids):
        predicted = int(replayed[i]) if i < len(replayed) else None
        match = predicted == int(expected)
        ok = ok and match
        positions.append({"position": n_in + i, "expected": int(expected),
                          "predicted": predicted, "match": match})
    return ok, positions


def _verify_greedy_cached_replay(model, input_ids, output_ids, *,
                                 rep_penalty_fp: int = 0, no_repeat_ngram: int = 0) -> dict:
    frac = int(model.cfg["frac"])

    if not rep_penalty_fp and not no_repeat_ngram and hasattr(model, "generate_greedy_tokens_cached"):
        replayed = model.generate_greedy_tokens_cached(input_ids, len(output_ids), eos=None)
    else:
        def _pick(row, _pos, hist):
            return _penalized_argmax(model, row, hist, rep_penalty_fp, no_repeat_ngram)
        replayed = model.generate_cached(input_ids, len(output_ids), _pick, eos=None)
    ok, positions = _cached_replay_positions(input_ids, output_ids, replayed)
    return {"ok": ok, "checked": len(output_ids), "positions": positions,
            "strategy": "greedy-cached-replay", "logitsDigest": None, "fracBits": frac}


def _verify_resample_cached_replay(model, input_ids, output_ids, *, sampler_cfg: SamplerConfig) -> dict:
    frac = int(model.cfg["frac"])

    def _pick(row, pos, hist):
        return sample_token(row, sampler_cfg, position=pos, frac_bits=frac, history_ids=hist)

    replayed = model.generate_cached(input_ids, len(output_ids), _pick, eos=None)
    ok, positions = _cached_replay_positions(input_ids, output_ids, replayed)
    return {"ok": ok, "checked": len(output_ids), "positions": positions,
            "strategy": "resample-cached-replay", "logitsDigest": None}


def _row_predicting_output(model, input_ids, output_ids, i: int, eff: int, full=None) -> np.ndarray:
    """The logits row that predicts output token `i`, matching the engine's sliding window.

    Causal: token at absolute index `len(input)+i` is predicted by row `len(input)+i-1`. When the whole
    turn fits `eff_ctx`, read it from the single prefill `full`; otherwise forward the SAME clamped
    prefix the engine used (`(input+output[:i])[-eff:]`) and take its last row.
    """
    # Absolute index of the predicting row. When input is empty AND i==0 this is -1, which would silently
    # index `full[-1]` (the LAST prefill row) via NumPy negative-index wrap — a false-accept vector: a
    # bundle with empty inputIds could satisfy `output[0] == argmax(full[-1])` and reach fullyVerified for
    # a token the model never produced. There is no valid predicting row for the first token of an
    # empty-prompt turn (nothing precedes it), so fail loud instead of wrapping.
    predicting = len(input_ids) + i - 1
    if predicting < 0:
        raise ValueError(
            "no predicting row for output[0] with empty inputIds — a generation receipt must commit a "
            "non-empty prompt; refusing to index full[-1] by negative-index wrap")
    if full is not None:
        return full[predicting]
    return model.forward((list(input_ids) + list(output_ids)[:i])[-eff:])[-1]


def verify_greedy(model, input_ids, output_ids, *, rep_penalty_fp: int = 0,
                  no_repeat_ngram: int = 0) -> dict:
    """Full check: every output token must be the (penalized) argmax of its predicting row. One prefill
    when the turn fits the window; else a per-position sliding-window re-derivation matching the engine."""
    input_ids, output_ids = list(input_ids), list(output_ids)
    if not output_ids:
        return {"ok": True, "checked": 0, "positions": [], "strategy": "greedy-full"}
    eff = _eff_ctx(model)
    seq = input_ids + output_ids
    strategy = _full_verification_strategy(model, len(input_ids), len(output_ids))
    if (strategy == "cached-replay" and len(seq) > eff
            and getattr(model, "receipt_verify_strategy", "auto") == "cached-replay"):
        raise ValueError(
            "cached-replay was explicitly selected, but the committed turn exceeds the exact context window"
        )
    if strategy == "cached-replay" and len(seq) <= eff:
        return _verify_greedy_cached_replay(
            model,
            input_ids,
            output_ids,
            rep_penalty_fp=rep_penalty_fp,
            no_repeat_ngram=no_repeat_ngram,
        )
    full = teacher_forced_logits(model, seq) if len(seq) <= eff else None   # single prefill iff it fits
    n_in = len(input_ids)
    positions, ok = [], True
    for i, tok in enumerate(output_ids):
        row = _row_predicting_output(model, input_ids, output_ids, i, eff, full)
        predicted = _penalized_argmax(model, row, input_ids + output_ids[:i], rep_penalty_fp, no_repeat_ngram)
        match = predicted == int(tok)
        ok = ok and match
        positions.append({"position": n_in + i, "expected": int(tok), "predicted": predicted,
                          "match": match, "logitsDigest": logits_digest(row)})   # RAW row digest
    return {"ok": ok, "checked": len(positions), "positions": positions,
            "strategy": "greedy-full" if full is not None else "greedy-sliding"}


def verify_sampled(model, input_ids, output_ids, *, k: int, seed: int = 0,
                   rep_penalty_fp: int = 0, no_repeat_ngram: int = 0) -> dict:
    """Probabilistic audit: re-check `k` deterministically-chosen output positions from one prefill."""
    input_ids, output_ids = list(input_ids), list(output_ids)
    n = len(output_ids)
    if n == 0:
        return {"ok": True, "checked": 0, "positions": [], "strategy": "greedy-sampled"}
    k = max(1, min(k, n))
    # deterministic selection (counter-based, no global RNG state) so an auditor's sample is reproducible
    rng = np.random.Generator(np.random.Philox(key=seed))
    chosen = sorted(int(j) for j in rng.choice(n, size=k, replace=False))
    eff = _eff_ctx(model)
    seq = input_ids + output_ids
    full = teacher_forced_logits(model, seq) if len(seq) <= eff else None
    n_in = len(input_ids)
    positions, ok = [], True
    for i in chosen:
        row = _row_predicting_output(model, input_ids, output_ids, i, eff, full)
        predicted = _penalized_argmax(model, row, input_ids + output_ids[:i], rep_penalty_fp, no_repeat_ngram)
        match = predicted == int(output_ids[i])
        ok = ok and match
        positions.append({"position": n_in + i, "expected": int(output_ids[i]), "predicted": predicted,
                          "match": match, "logitsDigest": logits_digest(row)})   # RAW row digest
    return {"ok": ok, "checked": k, "of": n, "positions": positions, "strategy": "greedy-sampled"}


def challenge_position(model, input_ids, output_ids, i: int, *, rep_penalty_fp: int = 0,
                       no_repeat_ngram: int = 0) -> dict:
    """Independently re-derive output position `i` from scratch: prefill the EXACT committed prefix
    (everything before output[i]) and read the last row. The dispute-resolution primitive."""
    input_ids, output_ids = list(input_ids), list(output_ids)
    if not (0 <= i < len(output_ids)):
        raise IndexError(i)
    prefix = (input_ids + output_ids[:i])[-_eff_ctx(model):]  # tokens before output[i], engine-clamped
    row = model.forward(prefix)[-1]
    # penalty history is the FULL prefix (input + output[:i]), matching the engine (not the clamped window)
    predicted = _penalized_argmax(model, row, input_ids + output_ids[:i], rep_penalty_fp, no_repeat_ngram)
    return {"position": len(input_ids) + i, "expected": int(output_ids[i]), "predicted": predicted,
            "match": predicted == int(output_ids[i]), "logitsDigest": logits_digest(row),
            "strategy": "greedy-challenge"}


def verify_resample(model, input_ids, output_ids, *, sampler_cfg: SamplerConfig) -> dict:
    """Re-derive a SEEDED sampled turn by replaying the committed integer sampler at every position.

    Teacher-forced like `verify_greedy`: one prefill of input+output gives each position's logits row, and
    each output token must equal `sample_token(row, cfg, position, frac, history)` where `position` is the
    ABSOLUTE index (`len(input)+i`) and `history` is the committed prefix (`input + output[:i]`) — exactly
    the (seed, position, distribution) the producer drew from (ReferenceModelV2.generate_cached passes
    `position=len(seq)`). The fully-integer, counter-based draw (sampler.draw_uniform_int) makes this
    bit-exact and machine-independent, so a seeded temp/top-k/top-p turn is as re-executable as greedy.
    """
    input_ids, output_ids = list(input_ids), list(output_ids)
    if not output_ids:
        return {"ok": True, "checked": 0, "positions": [], "strategy": "resample-full"}
    frac = int(model.cfg["frac"])
    eff = _eff_ctx(model)
    seq = input_ids + output_ids
    strategy = _full_verification_strategy(model, len(input_ids), len(output_ids))
    if (strategy == "cached-replay" and len(seq) > eff
            and getattr(model, "receipt_verify_strategy", "auto") == "cached-replay"):
        raise ValueError(
            "cached-replay was explicitly selected, but the committed turn exceeds the exact context window"
        )
    if strategy == "cached-replay" and len(seq) <= eff:
        return _verify_resample_cached_replay(model, input_ids, output_ids, sampler_cfg=sampler_cfg)
    full = teacher_forced_logits(model, seq) if len(seq) <= eff else None   # single prefill iff it fits
    n_in = len(input_ids)
    positions, ok = [], True
    for i, tok in enumerate(output_ids):
        row = _row_predicting_output(model, input_ids, output_ids, i, eff, full)
        predicted = sample_token(row, sampler_cfg, position=n_in + i, frac_bits=frac,
                                 history_ids=input_ids + output_ids[:i])
        match = predicted == int(tok)
        ok = ok and match
        positions.append({"position": n_in + i, "expected": int(tok), "predicted": predicted,
                          "match": match, "logitsDigest": logits_digest(row)})   # RAW row digest
    return {"ok": ok, "checked": len(positions), "positions": positions,
            "strategy": "resample-full" if full is not None else "resample-sliding"}
