"""Sampling over the reference engine's fixed-point logits — ALL modes receipt-bound, fully integer.

Every sampler here is **deterministic, integer-only, and re-executable**, so all of them are RECEIPT_SAFE:

  * GREEDY — integer argmax over a (vocab,) int64 fixed-point logits row. No RNG, no float.
  * TEMP / TOP_K / TOP_P — a *seeded* draw that is also bit-exact across machines. The three things that
    used to make sampling non-reproducible are removed (docs/architecture/SAMPLER-INTEGER.md):
      1. temperature is applied as a COMMITTED fixed-point inverse-temperature
         (`(logit * inv_temp_fp) >> frac`), not a float divide;
      2. the softmax + top-k/top-p truncation are the engine's own integer ops;
      3. the uniform draw is a counter-based **SHA-256** PRNG keyed by (seed, absolute position) reduced
         to `[0, total)` by Lemire's integer multiply-shift — no `rng.random()`, no float, no
         double-conversion, no dependence on a library RNG stream.

A sampled turn is re-derivable from its receipt because the seed is committed and the PRNG is keyed by
(seed, position): a verifier replays the exact same draws (infer_int/verify.py::verify_resample). The only
floats anywhere are the *scalar* conversions `inv_temp_fp()` / `top_p_fp()` — single correctly-rounded
IEEE-754 ops (not reductions, not transcendentals), so they are cross-platform deterministic. They are
invoked PER TOKEN in the sampled modes (`sample_token`/`top_probs` recompute them each call from the config),
but each is one scalar conversion, not a reduction; the per-token hot path is otherwise pure integer.
SHA-256 (already the determinism bedrock of every commitment) is the one external primitive, and it is
bit-identical on every machine and every version.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..determinism.fixedpoint import fixed_point_softmax
from ..hashing.sha import sha256_hex

# Greedy + the seeded integer samplers are ALL re-executable → all receipt-safe
# (docs/architecture/SAMPLER-INTEGER.md).
RECEIPT_SAFE_MODES = frozenset({"greedy", "temp", "top_k", "top_p", "min_p"})

# Named presets usable as a `--sampler` value (expanded by `resolve_sampler`). qwen3-rec is the Qwen3 vendor
# recommendation — our model IS Qwen3-based, and Qwen3 advises AGAINST greedy decoding (top_p nucleus + top_k
# + low temperature). At a fixed seed it is still receipt-bound and byte-exactly reproducible.
SAMPLER_PRESETS = {
    "qwen3-rec": {"mode": "top_p", "temperature": 0.6, "top_k": 20, "top_p": 0.95},
}


@dataclass(frozen=True)
class SamplerConfig:
    mode: str = "greedy"          # 'greedy' | 'temp' | 'top_k' | 'top_p' | 'min_p'  (all receipt-bound)
    temperature: float = 1.0
    top_k: int = 0                # 0 = off
    top_p: float = 1.0            # 1.0 = off
    min_p: float = 0.0            # 0.0 = off; keep tokens with prob >= min_p * max_prob (relative nucleus)
    seed: int = 0
    # --- repetition control (deterministic, receipt-safe; applied BEFORE greedy/softmax) ---
    rep_penalty: int = 0          # CTRL penalty, COMMITTED FIXED-POINT ≈ (θ-1)·2^fpFracBits (e.g. 13107≈θ1.2); 0=off
    no_repeat_ngram: int = 0      # ban tokens completing an n-gram already in history (e.g. 3); 0/1=off
    # --- receipt/v2 committed fixed-point overrides (float-free replay) ---
    # When set (reconstructed from a committed receipt/v2 sampler block), the per-token path uses these
    # EXACT integers verbatim instead of recomputing inv_temp_fp()/top_p_fp() from the float
    # temperature/top_p — so a verifier replays the producer's committed draw with NO float anywhere on
    # the per-token path. Left None for the user-facing API (which takes float temperature/top_p).
    inv_temp_fp_committed: int | None = None
    top_p_fp_committed: int | None = None
    min_p_fp_committed: int | None = None


def is_receipt_safe(cfg: SamplerConfig) -> bool:
    return cfg.mode in RECEIPT_SAFE_MODES


def validate_sampler_config(cfg: SamplerConfig) -> SamplerConfig:
    """Fail closed on sampler settings whose receipt replay semantics would be ambiguous."""
    if cfg.mode not in RECEIPT_SAFE_MODES:
        raise ValueError(f"unsupported sampler mode {cfg.mode!r}")
    if cfg.mode in ("temp", "top_k", "top_p", "min_p") and cfg.temperature <= 0:
        raise ValueError("temperature must be > 0 for sampling modes")
    if cfg.mode == "top_k" and cfg.top_k <= 0:
        raise ValueError("top_k must be > 0 for top_k sampling")
    if cfg.mode == "top_p" and not (0.0 < cfg.top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1] for top_p sampling")
    if cfg.mode == "min_p" and not (0.0 < cfg.min_p <= 1.0):
        raise ValueError("min_p must be in (0, 1] for min_p sampling")
    if cfg.top_k < 0:
        raise ValueError("top_k must be >= 0")
    if not (0.0 <= cfg.min_p <= 1.0):
        raise ValueError("min_p must be in [0, 1]")
    if cfg.rep_penalty < 0:
        raise ValueError("rep_penalty must be >= 0")
    if cfg.no_repeat_ngram < 0:
        raise ValueError("no_repeat_ngram must be >= 0")
    return cfg


DEFAULT_MIN_P = 0.1   # the min-p coefficient used when `--sampler min_p` is selected without an explicit --min-p


def resolve_sampler(name: str, *, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
                    min_p: float = 0.0, seed: int = 0, rep_penalty: int = 0,
                    no_repeat_ngram: int = 0) -> SamplerConfig:
    """Build a SamplerConfig from a CLI `--sampler` value, expanding named presets (e.g. 'qwen3-rec').
    A preset fixes mode/temperature/top_k/top_p; the seed + repetition controls are still taken from args.
    For bare `min_p` with no --min-p given, defaults the coefficient to DEFAULT_MIN_P (so it is the usable
    default seeded sampler out of the box)."""
    if name in SAMPLER_PRESETS:
        return validate_sampler_config(SamplerConfig(seed=seed, rep_penalty=rep_penalty,
                                                     no_repeat_ngram=no_repeat_ngram, **SAMPLER_PRESETS[name]))
    if name == "min_p" and (not min_p or min_p <= 0):
        min_p = DEFAULT_MIN_P
    return validate_sampler_config(SamplerConfig(mode=name, temperature=temperature, top_k=top_k, top_p=top_p,
                                                 min_p=min_p, seed=seed, rep_penalty=rep_penalty,
                                                 no_repeat_ngram=no_repeat_ngram))


def sampler_config_from_block(block: dict) -> SamplerConfig:
    """Reconstruct a SamplerConfig from a committed receipt sampler block (camelCase keys) so a verifier
    re-derives a turn with the EXACT committed settings.

    receipt/v2 blocks carry the fixed-point `invTempFp`/`topPFp` directly (float-free) — these are used
    verbatim, so the verifier's per-token draw has no float at all. Legacy receipt/v1 blocks carry float
    `temperature`/`topP`, reconstructed as before (the IEEE-scalar conversion happens at draw time)."""
    g = block.get
    inv_committed = g("invTempFp")
    if inv_committed is not None:                       # receipt/v2 — committed fixed-point ints
        frac = int(g("fpFracBits", 16) or 16)
        itf = int(inv_committed)
        if itf <= 0:                                    # range-check the committed ints (v1 had this guard
            raise ValueError(f"committed invTempFp must be > 0, got {itf}")   # via float temperature)
        topp_committed = g("topPFp")
        tpf = int(topp_committed) if topp_committed is not None else None
        if tpf is not None and not (0 < tpf <= (1 << frac)):
            raise ValueError(f"committed topPFp must be in (0, 2^{frac}], got {tpf}")
        minp_committed = g("minPFp")
        mpf = int(minp_committed) if minp_committed is not None else None
        if mpf is not None and not (0 < mpf <= (1 << frac)):
            raise ValueError(f"committed minPFp must be in (0, 2^{frac}], got {mpf}")
        return validate_sampler_config(SamplerConfig(
            mode=g("mode", "greedy"),
            top_k=int(g("topK", 0) or 0),
            seed=int(g("seed", 0) or 0),
            rep_penalty=int(g("repPenalty", 0) or 0),
            no_repeat_ngram=int(g("noRepeatNgram", 0) or 0),
            inv_temp_fp_committed=itf,
            top_p_fp_committed=tpf,
            min_p=(mpf / (1 << frac)) if mpf is not None else 0.0,   # float for validate; the fp int drives the draw
            min_p_fp_committed=mpf,
        ))
    return validate_sampler_config(SamplerConfig(       # legacy receipt/v1 — float temperature/topP
        mode=g("mode", "greedy"),
        temperature=float(g("temperature", 1.0)),
        top_k=int(g("topK", 0) or 0),
        top_p=float(g("topP", 1.0)),
        min_p=float(g("minP", 0.0)),
        seed=int(g("seed", 0) or 0),
        rep_penalty=int(g("repPenalty", 0) or 0),
        no_repeat_ngram=int(g("noRepeatNgram", 0) or 0),
    ))


def greedy(logits_row: np.ndarray) -> int:
    """The locked argmax sampler: integer argmax over fixed-point logits (no RNG, no float)."""
    return int(np.asarray(logits_row).argmax())


# A large-but-BOUNDED negative sentinel for n-gram bans: argmax never picks it, and unlike
# iinfo(int64).min it cannot underflow the (max − logit) distance math in fixed_point_softmax (/probs).
_NGRAM_BAN = np.int64(-(1 << 40))


def apply_rep_penalty(logits_row: np.ndarray, history_ids, rep_penalty_fp: int,
                      no_repeat_ngram: int, frac_bits: int) -> np.ndarray:
    """Deterministic integer repetition control over a fixed-point logits row — RECEIPT-SAFE.

    Returns a NEW int64 row (never mutates the caller). Two transforms, both pure integer (no float,
    no RNG → bit-exact across machines, reproducible inside a receipt):
      * CTRL repetition penalty in COMMITTED fixed-point (`rep_penalty_fp` ≈ (θ−1)·2^frac_bits): each
        already-seen token's logit l → l·FP // (FP+p) if l≥0 else l·(FP+p) // FP  (i.e. ÷θ / ×θ), which
        lowers its score. Floor-division is deterministic; updates hit distinct indices so set-iteration
        order is irrelevant to the result.
      * no-repeat-ngram: any token that would complete an `n`-gram already present in `history_ids` is
        set to `_NGRAM_BAN` so argmax/softmax can never choose it.
    """
    row = np.array(logits_row, dtype=np.int64)            # copy — caller's row is untouched
    hist = [int(x) for x in history_ids]
    if not hist:
        return row
    if rep_penalty_fp:
        FP = np.int64(1) << frac_bits
        denom = FP + np.int64(int(rep_penalty_fp))
        for t in {x for x in hist}:                       # unique seen tokens; order-independent
            l = row[t]
            row[t] = (l * FP) // denom if l >= 0 else (l * denom) // FP
    n = int(no_repeat_ngram)
    if n > 1 and len(hist) >= n:
        prefix = tuple(hist[-(n - 1):])
        for i in range(len(hist) - n + 1):                # ban the continuations of any matching prefix
            if tuple(hist[i:i + n - 1]) == prefix:
                row[hist[i + n - 1]] = _NGRAM_BAN
    return row


# ── fixed-point sampler primitives (the receipt-safe, fully-integer draw) ─────────────────────────────

def inv_temp_fp(temperature: float, frac_bits: int) -> int:
    """The COMMITTED fixed-point inverse-temperature `round(2^frac / T)`. A single correctly-rounded IEEE-754
    scalar op (deterministic across platforms); in the sampled modes it is recomputed once per token (one
    scalar conversion, not a reduction), then the per-token path multiplies by it in pure integer.
    `T == 1.0` → exactly `2^frac` (an identity scale, no rounding)."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    v = round((1 << frac_bits) / float(temperature))
    if v >= (1 << 62):
        raise ValueError("temperature too small for fixed-point sampling — use greedy for argmax")
    return int(v)


def top_p_fp(top_p: float, frac_bits: int) -> int:
    """The COMMITTED fixed-point nucleus threshold `round(top_p · 2^frac)` (probs sum to ~2^frac)."""
    return int(round(float(top_p) * (1 << frac_bits)))


def min_p_fp(min_p: float, frac_bits: int) -> int:
    """The COMMITTED fixed-point min-p coefficient `round(min_p · 2^frac)`. The per-token threshold is
    `(min_p_fp · max_prob) >> frac` — a cutoff RELATIVE to the top token's probability (min-p sampling)."""
    return int(round(float(min_p) * (1 << frac_bits)))


def _resolve_min_p_fp(cfg: "SamplerConfig", frac_bits: int) -> int:
    """The fixed-point min-p coefficient the draw uses — committed receipt/v2 int verbatim if present, else
    one correctly-rounded IEEE-754 scalar conversion of the float `min_p`."""
    return int(cfg.min_p_fp_committed) if cfg.min_p_fp_committed is not None else min_p_fp(cfg.min_p, frac_bits)


def _resolve_fp(cfg: "SamplerConfig", frac_bits: int) -> tuple[int, int]:
    """The fixed-point (inv_temp, top_p) the draw uses. If the config carries committed receipt/v2 ints
    (`inv_temp_fp_committed`/`top_p_fp_committed`) they are used VERBATIM — the verifier then replays the
    producer's draw with zero float on the per-token path. Otherwise the float temperature/top_p are
    converted once (a single correctly-rounded IEEE-754 scalar op each, cross-platform deterministic)."""
    itf = cfg.inv_temp_fp_committed if cfg.inv_temp_fp_committed is not None else inv_temp_fp(cfg.temperature, frac_bits)
    tpf = cfg.top_p_fp_committed if cfg.top_p_fp_committed is not None else top_p_fp(cfg.top_p, frac_bits)
    return int(itf), int(tpf)


def _apply_temp_fp(row: np.ndarray, inv_temp_fp_val: int, frac_bits: int) -> np.ndarray:
    """Temperature scaling in pure integer: `(logit * inv_temp_fp) >> frac` (arithmetic shift = floor,
    deterministic for negatives). Fails LOUD on int64 overflow rather than silently wrapping (the worst
    receipt failure — producer and verifier would wrap identically and 'agree' on a wrong distribution)."""
    row = np.asarray(row, dtype=np.int64)
    # Take the peak magnitude from the int64 extremes via Python ints — NOT np.abs(row).max(): np.abs on an
    # int64 array wraps abs(INT64_MIN) back to INT64_MIN (negative), so a most-negative logit would slip past
    # the guard and the (row*inv_temp_fp) multiply could then wrap silently (the worst receipt failure). The
    # companion fixedpoint._assert_no_int64_overflow guards the same way.
    lo, hi = int(row.min(initial=0)), int(row.max(initial=0))
    peak = max(abs(lo), abs(hi)) * int(inv_temp_fp_val)
    if peak >= (1 << 63):
        raise OverflowError("temperature scaling overflows int64; temperature too small for these logits")
    return (row * np.int64(inv_temp_fp_val)) >> frac_bits


_SAMPLER_PRNG_TAG = b"trinote-sampler-draw/v1"
_U64 = 1 << 64
_MASK64 = _U64 - 1


def _prng_word(seed: int, position: int, counter: int) -> int:
    """One uniform 64-bit word from a counter-based SHA-256 stream keyed by (seed, absolute position).

    Pure integer + SHA-256 — bit-identical on every machine AND every library version (SHA-256 is the
    determinism bedrock every commitment already relies on), unlike a float RNG stream. `& _MASK64` takes
    the two's-complement low 64 bits so a negative seed is handled deterministically."""
    msg = (_SAMPLER_PRNG_TAG
           + (seed & _MASK64).to_bytes(8, "big")
           + (position & _MASK64).to_bytes(8, "big")
           + (counter & _MASK64).to_bytes(8, "big"))
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], "big")


def draw_uniform_int(total: int, seed: int, position: int) -> int:
    """An unbiased uniform integer in `[0, total)` via Lemire's multiply-shift over the counter-based word
    stream — no float, no double-conversion. Deterministic in (total, seed, position); the rejection loop
    (essentially never taken for total≈2^16) consumes successive counter words, staying reproducible."""
    n = int(total)
    if n <= 1:
        return 0
    j = 0
    x = _prng_word(seed, position, j); j += 1
    m = x * n
    low = m & _MASK64
    if low < n:
        t = _U64 % n                                   # Lemire rejection threshold = 2^64 mod n
        while low < t:
            x = _prng_word(seed, position, j); j += 1
            m = x * n
            low = m & _MASK64
    return m >> 64


def _probs_fp(logits_row: np.ndarray, inv_temp_fp_val: int, frac_bits: int) -> np.ndarray:
    """Fixed-point softmax probabilities (vocab,), with integer temperature scaling (no float divide)."""
    row = np.asarray(logits_row, dtype=np.int64)
    if inv_temp_fp_val != (1 << frac_bits):                # T != 1 → integer-scale; T == 1 is identity
        row = _apply_temp_fp(row, inv_temp_fp_val, frac_bits)
    return fixed_point_softmax(row[None, :], frac_bits)[0]


def top_probs(logits_row: np.ndarray, k: int, frac_bits: int,
              *, cfg: "SamplerConfig | None" = None, history_ids=None) -> list[tuple[int, int]]:
    """Glass-box surface: top-k (token_id, fixed-point prob) from the engine's own softmax.

    Without `cfg`, the raw temperature=1.0 softmax. With `cfg`, reflects the ACTIVE sampler — the
    deterministic repetition penalty (when `history_ids` is supplied), temp scaling, and top-k/top-p
    truncation — i.e. *exactly the distribution `sample_token` draws from*, so /probs cannot disagree
    with what is sampled (the documented glass-box contract).
    """
    row = logits_row
    if cfg is not None and (cfg.rep_penalty or cfg.no_repeat_ngram) and history_ids is not None:
        row = apply_rep_penalty(row, history_ids, cfg.rep_penalty, cfg.no_repeat_ngram, frac_bits)
    if cfg is None or cfg.mode == "greedy":
        probs = _probs_fp(row, 1 << frac_bits, frac_bits)                  # T = 1 (inv_temp_fp == identity)
    else:
        inv_temp_val, top_p_fp_val = _resolve_fp(cfg, frac_bits)           # committed v2 ints if present
        probs = _probs_fp(row, inv_temp_val, frac_bits)
        if cfg.mode in ("top_k", "top_p"):
            probs = _truncate(probs, cfg.top_k, top_p_fp_val, frac_bits)
        elif cfg.mode == "min_p":
            probs = _truncate_min_p(probs, _resolve_min_p_fp(cfg, frac_bits), frac_bits)
    k = max(1, min(k, probs.shape[0]))
    # Stable top-k by a single stable argsort (NOT argpartition, whose tie/order behaviour varies by
    # numpy build) — same selection rule as `_truncate`, so /probs cannot disagree with sample_token.
    idx = np.argsort(probs, kind="stable")[::-1][:k]            # descending, ties broken by index
    return [(int(i), int(probs[i])) for i in idx]


def _truncate(probs: np.ndarray, top_k: int, top_p_fp_val: int, frac_bits: int) -> np.ndarray:
    """Zero everything outside the top-k / nucleus (top-p), always keeping >= 1 token. Pure integer:
    `top_p_fp_val` is the COMMITTED fixed-point nucleus threshold (probs sum to ~2^frac)."""
    probs = probs.astype(np.int64).copy()
    order = np.argsort(probs, kind="stable")[::-1]             # descending by prob
    keep = np.zeros(probs.shape[0], dtype=bool)
    keep[order[: max(1, top_k)] if top_k and top_k > 0 else order] = True
    if top_p_fp_val < (1 << frac_bits):                        # top_p < 1.0 → nucleus truncation
        csum = np.cumsum(probs[order])
        n = int(np.searchsorted(csum, int(top_p_fp_val), side="left")) + 1   # +1: include the crossing token
        nucleus = np.zeros(probs.shape[0], dtype=bool)
        nucleus[order[: max(1, min(n, probs.shape[0]))]] = True
        keep &= nucleus
    probs[~keep] = 0
    if not probs.any():                                       # never an all-zero distribution
        probs[int(order[0])] = 1
    return probs


def _truncate_min_p(probs: np.ndarray, min_p_fp_val: int, frac_bits: int) -> np.ndarray:
    """min-p truncation: zero every token whose fixed-point prob < `min_p · max_prob` (a cutoff RELATIVE to
    the top token), always keeping the argmax. Pure integer: threshold = (min_p_fp · max_prob) >> frac on
    Python ints (no overflow), so producer and verifier truncate bit-identically."""
    probs = probs.astype(np.int64).copy()
    if min_p_fp_val <= 0:                                     # off → no truncation
        return probs
    amax = int(probs.argmax())
    pmax = int(probs[amax])
    if pmax <= 0:
        return probs
    thresh = (int(min_p_fp_val) * pmax) >> frac_bits          # min_p * max_prob, fixed-point (Python ints)
    probs[probs < np.int64(thresh)] = 0
    if not probs.any():                                       # never an all-zero distribution (defensive)
        probs[amax] = 1
    return probs


def sample_token(logits_row: np.ndarray, cfg: SamplerConfig, position: int, frac_bits: int,
                 history_ids=None) -> int:
    """Pick one token — RECEIPT-BOUND for every mode (greedy + seeded temp/top-k/top-p).

    Greedy is integer argmax. The seeded modes apply the committed fixed-point temperature, the engine's
    integer softmax + truncation, then an integer Lemire draw keyed by (cfg.seed, `position`) — so given
    the same committed prefix, seed, and absolute position a verifier re-derives the SAME token bit-for-bit
    (infer_int/verify.py::verify_resample). The per-token hot path is integer EXCEPT the two correctly-rounded
    IEEE-754 scalar conversions `inv_temp_fp()` / `top_p_fp()` (single conversions, not reductions, not
    transcendentals — cross-platform deterministic). With a nonzero rep_penalty/no_repeat_ngram AND
    `history_ids`, the deterministic repetition penalty is applied to a COPY of the row first.
    `history_ids=None` (default) leaves every existing 4-arg caller byte-identical.

    TIE-BREAK NOTE (deterministic, shared by producer and verifier): greedy uses `np.argmax`, which favours
    the LOWEST index on ties; the top-k/top-p selection (and `top_probs`) sort by a *stable* descending
    argsort that, on equal probabilities, favours the HIGHEST index. The two rules differ but each is fully
    deterministic, and both the producer and the verifier run the SAME path, so receipts replay exactly."""
    cfg = validate_sampler_config(cfg)
    row = logits_row
    if (cfg.rep_penalty or cfg.no_repeat_ngram) and history_ids is not None:
        row = apply_rep_penalty(logits_row, history_ids, cfg.rep_penalty, cfg.no_repeat_ngram, frac_bits)
    if cfg.mode == "greedy":
        return greedy(row)
    inv_temp_val, top_p_fp_val = _resolve_fp(cfg, frac_bits)
    probs = _probs_fp(row, inv_temp_val, frac_bits)
    if cfg.mode in ("top_k", "top_p"):
        probs = _truncate(probs, cfg.top_k, top_p_fp_val, frac_bits)
    elif cfg.mode == "min_p":
        probs = _truncate_min_p(probs, _resolve_min_p_fp(cfg, frac_bits), frac_bits)
    total = int(probs.sum())
    if total <= 0:                                            # degenerate all-zero distribution → argmax
        return int(np.asarray(row).argmax())
    target = draw_uniform_int(total, cfg.seed, position)      # integer, counter-based, no float
    csum = np.cumsum(probs.astype(np.int64))
    return int(np.searchsorted(csum, target, side="right"))   # never lands on a zero-prob token


def logits_digest(logits_row: np.ndarray) -> str:
    """SHA-256 over the canonical LITTLE-ENDIAN int64 bytes of a fixed-point logits row (receipt
    commitment). Pinning `<i8` (not native `int64`) keeps the digest byte-identical across endianness —
    the literal cross-machine guarantee — and is a no-op on the little-endian hardware in use today."""
    return sha256_hex(np.ascontiguousarray(logits_row, dtype="<i8").tobytes())
