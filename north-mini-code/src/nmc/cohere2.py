"""Stage 3 — the cohere2moe DENSE path in deterministic integer fixed-point.

Implements the Cohere **parallel block** (one RMSNorm feeds BOTH attention and FFN; their outputs are summed
into the residual: `x = x + Attn(n(x)) + FFN(n(x))`), GQA **full** attention with **interleaved/NORM** RoPE
(lanes 2i, 2i+1; θ=50000) — NOT NeoX (see `apply_rope_fixed` import below) — the dense SwiGLU FFN, and the
**tied-embedding** LM head, all integer. This is the leading dense block (block 0) and the scaffold the MoE
blocks (Stage 5) slot into.

Reuses the proven Bonsai integer primitives (`fixed_point_rmsnorm/softmax/sigmoid`, NeoX RoPE) and the Stage-2
Q4_K/Q6_K codec for weights. A matching float64 reference defines the architecture; the gate is fidelity
(integer ≈ float) plus determinism (integer ⇒ reproducible / order-free).

Scales: activations at `fa` (default 16); linear-layer weights at `fw` (default 24, from Stage 2 fidelity).
A linear is `(x@Wᵀ) >> fw` so activations stay at `fa` while weights keep their precision (asymmetric scale).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Vendored byte-exact integer primitives (fixed-point RMSNorm/softmax/sigmoid + NeoX RoPE) — see nmc/_bonsai.
# Self-contained: no cross-repo path dependency, so the engine deploys standalone.
from nmc._bonsai.fixedpoint import fixed_point_rmsnorm, fixed_point_softmax, fixed_point_sigmoid
from nmc._bonsai.rope_v2 import build_rope_tables
from nmc._bonsai.rope import apply_rope_fixed   # cohere2 uses NORM/INTERLEAVED RoPE (lanes 2i,2i+1), not NeoX

_I64_MAX = (1 << 63) - 1


def _assert_i64_contraction(amax: int, bmax: int, k: int, where: str) -> None:
    """Fail loud if an int64 contraction max|a|*max|b|*K would wrap. nmc's decode hot path uses int64 numpy
    matmul for speed — byte-identical to a big-int accumulation ONLY within the fixed-point envelope. A silent
    wrap would diverge from the big-int oracle deterministically (the receipt-lethal failure: producer and
    verifier agree on a wrong logit). `amax`/`bmax` are Python-int magnitudes (so int64-min can't wrap the
    guard). Called once per matmul — cheap vs the matmul itself."""
    if amax and bmax and amax * bmax * int(k) > _I64_MAX:
        raise OverflowError(
            f"{where}: int64 contraction max|a|*max|b|*K = {amax * bmax * int(k)} > 2^63-1 — an activation "
            f"left the fixed-point envelope; the int64 matmul would wrap (deterministic-but-wrong for a receipt)")


def _absmax_int(a: np.ndarray) -> int:
    """max|a| as a Python int, avoiding np.abs(int64-min) wrap."""
    return max(abs(int(a.min())), abs(int(a.max()))) if a.size else 0


@dataclass
class Cfg:
    d_model: int
    n_heads: int
    n_kv: int
    head_dim: int
    ffn: int
    vocab: int
    sliding_window: int = 4096
    n_experts: int = 128      # MoE: total experts
    n_used: int = 8           # MoE: top-k routed per token
    expert_ffn: int = 768     # MoE: per-expert FFN dim
    rope_base: float = 50000.0
    eps: int = 1
    fa: int = 16          # activation fixed-point bits
    fw: int = 24          # weight fixed-point bits

    @property
    def rep(self): return self.n_heads // self.n_kv


# north-mini-code-1.0 attention-type pattern (49 layers, read from the GGUF in Stage 1): 0=full, 1=sliding,
# i.e. FULL attention every 4th layer. Period-4 `0,1,1,1` repeating (layers 0,4,…,48 full; the rest SWA).
NORTH_SWA_PATTERN = ([0, 1, 1, 1] * 12) + [0]            # 49 entries


def is_full_layer(idx: int) -> bool:
    """True if layer `idx` uses FULL attention (else sliding-window). Matches NORTH_SWA_PATTERN."""
    return idx % 4 == 0


def window_for_layer(cfg: "Cfg", idx: int) -> "int | None":
    """The attention window for layer `idx`: None (full causal) or `cfg.sliding_window` (SWA)."""
    return None if is_full_layer(idx) else cfg.sliding_window


def _attn_mask(seq: int, window: "int | None") -> np.ndarray:
    """Boolean [seq,seq] where True = DISALLOWED. Full causal masks the future (j>i); sliding-window also
    masks keys older than `window` (j <= i-window), so each query sees the most recent `window` keys incl. self."""
    i = np.arange(seq)[:, None]; j = np.arange(seq)[None, :]
    future = j > i
    if window is None:
        return future
    return future | (j <= i - window)


def to_fixed(x, frac): return np.round(np.asarray(x, np.float64) * (1 << frac)).astype(np.int64)
def from_fixed(x, frac): return np.asarray(x, np.float64) / float(1 << frac)


# ----------------------------------------------------------------------------- integer building blocks -------
def linear(x_int, W_fixed, fw):
    """y = x @ Wᵀ with x at 2**fa, W at 2**fw -> y at 2**fa. Exact big-int accumulate (no silent wrap)."""
    x = np.asarray(x_int, dtype=object)
    W = np.asarray(W_fixed, dtype=object)
    return ((x @ W.T) >> fw).astype(np.int64)


def silu_int(x_int, fa):
    sig = fixed_point_sigmoid(np.asarray(x_int, np.int64), fa)        # sigmoid(x) at 2**fa
    return ((np.asarray(x_int, object) * np.asarray(sig, object)) >> fa).astype(np.int64)


def _rope_int(t_int, cos, sin, fa, start=0):
    """t_int [seq, n, hd] -> RoPE'd (cohere2 INTERLEAVED convention, lanes 2i,2i+1), at absolute positions
    [start, start+seq). Per head via (n, seq, hd) using the cos/sin rows for those positions."""
    seq, n, hd = t_int.shape
    out = apply_rope_fixed(np.transpose(t_int, (1, 0, 2)).astype(np.int64),
                           cos[start:start + seq], sin[start:start + seq], fa)
    return np.transpose(out, (1, 0, 2))


def attention_int(q, k, v, cfg, cos, sin, window=None, rope=True):
    """GQA causal attention, integer. q [seq,H*hd], k/v [seq,Hkv*hd] at 2**fa -> [seq,H*hd] at 2**fa.
    window=None → full causal; window=int → sliding-window. rope=False → NoPE (cohere2 applies no positional
    embedding on full-attention layers; the caller decides via rope=is_swa or dense-prefix)."""
    fa, H, hd = cfg.fa, cfg.n_heads, cfg.head_dim
    seq = q.shape[0]
    q = _rope_int(q.reshape(seq, H, hd), cos, sin, fa) if rope else q.reshape(seq, H, hd)
    k = _rope_int(k.reshape(seq, cfg.n_kv, hd), cos, sin, fa) if rope else k.reshape(seq, cfg.n_kv, hd)
    v = v.reshape(seq, cfg.n_kv, hd)
    inv_sqrt = int(round((1.0 / math.sqrt(hd)) * (1 << fa)))
    neg = -(1 << (fa + 30))                                            # masked -> ~0 weight after softmax
    mask = _attn_mask(seq, window)                                     # True = disallowed
    out = np.empty((seq, H, hd), dtype=np.int64)
    for h in range(H):
        kv = h // cfg.rep
        qh = q[:, h, :].astype(object); kh = k[:, kv, :].astype(object); vh = v[:, kv, :].astype(object)
        scores = ((qh @ kh.T) >> fa)                                  # [seq,seq] at 2**fa
        scores = ((scores * inv_sqrt) >> fa).astype(np.int64)         # / sqrt(hd)
        scores[mask] = neg
        probs = fixed_point_softmax(scores, fa)                       # [seq,seq] at 2**fa
        out[:, h, :] = ((probs.astype(object) @ vh) >> fa).astype(np.int64)
    return out.reshape(seq, H * hd)


def dense_block_int(x_int, W, cfg, cos, sin, window=None):
    """Cohere parallel dense block, integer. x_int [seq, d] at 2**fa -> [seq, d] at 2**fa.
    window: None=full causal, int=sliding-window (per the layer's attention type)."""
    fa, fw = cfg.fa, cfg.fw
    h = fixed_point_rmsnorm(x_int, fa, cfg.eps, gain_q=W["attn_norm"])        # one shared norm
    q = linear(h, W["wq"], fw); k = linear(h, W["wk"], fw); v = linear(h, W["wv"], fw)
    attn = linear(attention_int(q, k, v, cfg, cos, sin, window), W["wo"], fw)
    g = silu_int(linear(h, W["gate"], fw), fa)
    u = linear(h, W["up"], fw)
    gu = ((g.astype(object) * u.astype(object)) >> fa).astype(np.int64)
    ffn = linear(gu, W["down"], fw)
    return (np.asarray(x_int, np.int64) + attn + ffn)                         # parallel residual


# ------------------------------------------------------------------------------------------ MoE (Stage 5) ----
def _topk_lowidx(logits_row, k):
    """Indices of the k largest logits, ties broken by LOWEST index (the deterministic routing rule). Stable
    argsort on the negated logits keeps original (ascending-index) order among equal values → low index wins."""
    return np.argsort(-np.asarray(logits_row, dtype=np.int64), kind="stable")[:k]


def moe_layer_int(h_int, We, cfg):
    """cohere2moe MoE FFN, integer. h_int [seq, d] at 2**fa (already normed) -> [seq, d] at 2**fa.

    router logits = h·Wrouterᵀ (integer); SIGMOID gating; top-`n_used` by logit (low-index tie-break); each
    expert is SwiGLU; combine = Σ sigmoid(logit_e)·Expert_e(h). expert_weights_norm=false, scale=1 (no renorm).
    """
    fa, fw, k = cfg.fa, cfg.fw, cfg.n_used
    seq = h_int.shape[0]
    router_logits = linear(h_int, We["router"], fw)                  # [seq, n_experts] at 2**fa
    out = np.zeros((seq, cfg.d_model), dtype=object)
    for t in range(seq):
        sel = _topk_lowidx(router_logits[t], k)                      # the routed experts (a SET; sum is order-free)
        gates = fixed_point_sigmoid(router_logits[t][sel].astype(np.int64), fa)   # [k] sigmoid weights at 2**fa
        ht = h_int[t:t + 1]                                          # [1, d]
        for j, e in enumerate(sel):
            g = silu_int(linear(ht, We["gate"][e], fw), fa)         # [1, expert_ffn]
            u = linear(ht, We["up"][e], fw)
            gu = ((g.astype(object) * u.astype(object)) >> fa).astype(np.int64)
            eo = linear(gu, We["down"][e], fw)[0].astype(object)    # [d] at 2**fa
            out[t] += (eo * int(gates[j])) >> fa                    # weighted expert output, at 2**fa
    return out.astype(np.int64)


def moe_block_int(x_int, W, We, cfg, cos, sin, window=None):
    """Cohere parallel MoE block (blocks 1..48), integer: x = x + Attn(n(x)) + MoE(n(x)). cohere2 NoPE: RoPE
    only on sliding-window layers (full-attention MoE layers get no positional embedding)."""
    h = fixed_point_rmsnorm(x_int, cfg.fa, cfg.eps, gain_q=W["attn_norm"])
    q = linear(h, W["wq"], cfg.fw); k = linear(h, W["wk"], cfg.fw); v = linear(h, W["wv"], cfg.fw)
    attn = linear(attention_int(q, k, v, cfg, cos, sin, window, rope=window is not None), W["wo"], cfg.fw)
    moe = moe_layer_int(h, We, cfg)
    return (np.asarray(x_int, np.int64) + attn + moe)


def tied_head_int(x_int, W, cfg):
    """Final RMSNorm + tied LM head (logits = embedᵀ·h), integer. Returns logits [seq, vocab] at 2**fa."""
    hn = fixed_point_rmsnorm(x_int, cfg.fa, cfg.eps, gain_q=W["output_norm"])
    return linear(hn, W["token_embd"], cfg.fw)                                # logit_scale = 1


# ================================================ KV-cache decode path ========================================
# A per-layer cache of post-RoPE K and raw V lets us decode token-by-token (O(L) per step) instead of re-running
# the full prefill each step (O(L²)). The cache-aware block UNIFIES prefill (m positions, start=0) and decode
# (m=1, start=cache_len): RoPE at absolute positions, append K/V, attend over the cache within the window.
# Gate: generating with the cache is BYTE-IDENTICAL to re-prefilling the growing sequence from scratch.

class KVCache:
    """Per-layer post-RoPE K and raw V, each [Hkv, L, hd] int64, grown by `append` per step."""
    def __init__(self, n_layers):
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    def length(self, li=0):
        return 0 if self.k[li] is None else self.k[li].shape[1]

    def append(self, li, k_new, v_new):                      # k_new/v_new [Hkv, m, hd]
        self.k[li] = k_new if self.k[li] is None else np.concatenate([self.k[li], k_new], axis=1)
        self.v[li] = v_new if self.v[li] is None else np.concatenate([self.v[li], v_new], axis=1)


def attention_cached(q, ck, cv, start, cfg, window):
    """Integer GQA attention of m queries (at abs positions [start, start+m), already RoPE'd) over the cache
    ck/cv [Hkv, Lc, hd] (Lc = start+m, includes the m just-appended). Causal + sliding-window over absolute
    positions — identical masking to `attention_int` for the prefill case (start=0, m=Lc)."""
    fa, H, hd = cfg.fa, cfg.n_heads, cfg.head_dim
    m, Lc = q.shape[0], ck.shape[1]
    inv_sqrt = int(round((1.0 / math.sqrt(hd)) * (1 << fa)))
    neg = -(1 << (fa + 30))
    out = np.empty((m, H, hd), dtype=np.int64)
    # int64 (not object) is byte-identical here — q·k over head_dim ≲ 2**40 and prob·v ≲ 2**38 both fit int64,
    # so native numpy matmul == big-int but ~100× faster (the decode hot path). Gated by the decode tests.
    # Fail loud if that envelope is actually breached (one cheap vectorized check bounds every per-head matmul,
    # so a silent int64 wrap can never diverge from the big-int oracle unnoticed).
    _assert_i64_contraction(_absmax_int(q), _absmax_int(ck), hd, "attention_cached q·kᵀ")
    # Each softmax row is non-negative and sums to at most 2**fa (integer division floors each entry), so the
    # entire probability·V accumulation is bounded by 2**fa*max|V| — not that value times Lc. Multiplying by
    # Lc again would reject safe long-context rows even though their probability mass is still one.
    _assert_i64_contraction((1 << fa), _absmax_int(cv), 1, "attention_cached probs·v")
    for h in range(H):
        kv = h // cfg.rep
        qh = q[:, h, :]; kh = ck[kv]; vh = cv[kv]                     # all int64
        scores = ((qh @ kh.T) >> fa)                                  # [m, Lc] at 2**fa
        scores = ((scores * inv_sqrt) >> fa)
        for i in range(m):                                            # query i is absolute position start+i
            p = start + i
            scores[i, p + 1:] = neg                                   # causal: no future keys
            if window is not None and p - window >= 0:
                scores[i, :p - window + 1] = neg                      # sliding: drop keys older than `window`
        probs = fixed_point_softmax(scores, fa)
        out[:, h, :] = ((probs @ vh) >> fa)
    return out.reshape(m, H * hd)


def block_cached(x_new, W, We, cfg, cos, sin, cache, li, window):
    """Cohere parallel block (dense if We is None, else MoE) over m NEW positions, using + extending the cache.
    x_new [m, d] at 2**fa -> [m, d] at 2**fa. Works for prefill (m=seq, empty cache) and decode (m=1)."""
    fa, fw = cfg.fa, cfg.fw
    start = cache.length(li)
    m = x_new.shape[0]
    h = fixed_point_rmsnorm(x_new, fa, cfg.eps, gain_q=W["attn_norm"])
    q = linear(h, W["wq"], fw).reshape(m, cfg.n_heads, cfg.head_dim)
    k = linear(h, W["wk"], fw).reshape(m, cfg.n_kv, cfg.head_dim)
    v = linear(h, W["wv"], fw).reshape(m, cfg.n_kv, cfg.head_dim)
    if window is not None or We is None:          # cohere2 NoPE: RoPE only on SWA + dense-prefix (We is None) layers
        q = _rope_int(q, cos, sin, fa, start)
        k = _rope_int(k, cos, sin, fa, start)
    cache.append(li, np.transpose(k, (1, 0, 2)), np.transpose(v, (1, 0, 2)))   # store [Hkv, m, hd]
    attn = linear(attention_cached(q, cache.k[li], cache.v[li], start, cfg, window), W["wo"], fw)
    if We is None:
        gg = silu_int(linear(h, W["gate"], fw), fa); uu = linear(h, W["up"], fw)
        gu = ((gg.astype(object) * uu.astype(object)) >> fa).astype(np.int64)
        ffn = linear(gu, W["down"], fw)
    else:
        ffn = moe_layer_int(h, We, cfg)
    return np.asarray(x_new, np.int64) + attn + ffn


def _run_layers_cached(x_new, layers, cfg, cos, sin, cache):
    """Run all layers over the m new positions, extending the cache. layers[li] = (W, We) (We None for dense)."""
    for li, (W, We) in enumerate(layers):
        x_new = block_cached(x_new, W, We, cfg, cos, sin, cache, li, window_for_layer(cfg, li))
    return x_new


def generate(prompt_ids, embed_fn, layers, head_W, cfg, cos, sin, n_new, pick, *, use_cache=True):
    """Greedy/sampled generation. layers[li]=(W,We); embed_fn(tok)->[d] int64 at fa; head_W for tied_head_int;
    pick(logits_row, pos, hist)->token. use_cache=True: prefill once + KV-cached decode (O(L)/step). False:
    re-prefill the growing sequence each step (O(L²) reference). The two MUST yield identical token lists."""
    nl = len(layers)
    out, seq = [], list(prompt_ids)
    if use_cache:
        cache = KVCache(nl)
        last = _run_layers_cached(np.stack([embed_fn(t) for t in seq]), layers, cfg, cos, sin, cache)[-1:]
        for step in range(n_new):
            tok = int(pick(tied_head_int(last, head_W, cfg)[0], len(seq), seq))
            out.append(tok); seq.append(tok)
            if step + 1 >= n_new:
                break
            last = _run_layers_cached(embed_fn(tok)[None, :], layers, cfg, cos, sin, cache)
    else:
        for step in range(n_new):
            x = np.stack([embed_fn(t) for t in seq])
            x = _run_layers_cached(x, layers, cfg, cos, sin, KVCache(nl))   # fresh cache = from scratch
            tok = int(pick(tied_head_int(x[-1:], head_W, cfg)[0], len(seq), seq))
            out.append(tok); seq.append(tok)
    return out


# ------------------------------------------------------------------------------------- float reference -------
def _rmsnorm_f(x, gain, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * gain


def _rope_f(t, base, start=0):                 # t [seq, n, hd] — cohere2 INTERLEAVED (lanes 2i,2i+1)
    seq, n, hd = t.shape; half = hd // 2
    inv = base ** (-2.0 * np.arange(half) / hd)
    ang = np.outer(np.arange(start, start + seq), inv)        # [seq, half]
    c = np.cos(ang)[:, None, :]; s = np.sin(ang)[:, None, :]
    x0, x1 = t[..., 0::2], t[..., 1::2]
    out = np.empty_like(t)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    return out


def dense_block_float(x, W, cfg, window=None):
    H, Hkv, hd = cfg.n_heads, cfg.n_kv, cfg.head_dim; seq = x.shape[0]
    h = _rmsnorm_f(x, W["attn_norm"])
    q = _rope_f((h @ W["wq"].T).reshape(seq, H, hd), cfg.rope_base)
    k = _rope_f((h @ W["wk"].T).reshape(seq, Hkv, hd), cfg.rope_base)
    v = (h @ W["wv"].T).reshape(seq, Hkv, hd)
    out = np.empty((seq, H, hd))
    mask = np.where(_attn_mask(seq, window), -np.inf, 0.0)
    for hh in range(H):
        kv = hh // cfg.rep
        sc = (q[:, hh] @ k[:, kv].T) / math.sqrt(hd) + mask
        sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
        out[:, hh] = sc @ v[:, kv]
    attn = out.reshape(seq, H * hd) @ W["wo"].T
    g = (lambda z: z / (1 + np.exp(-z)))(h @ W["gate"].T)
    u = h @ W["up"].T
    ffn = (g * u) @ W["down"].T
    return x + attn + ffn


def tied_head_float(x, W):
    hn = _rmsnorm_f(x, W["output_norm"])
    return hn @ W["token_embd"].T


def moe_layer_float(h, We, cfg):
    """Float reference of the MoE FFN: sigmoid gating, top-k by logit, no renorm. h [seq,d] -> [seq,d]."""
    k = cfg.n_used
    router = h @ We["router"].T                       # [seq, n_experts]
    out = np.zeros_like(h)
    for t in range(h.shape[0]):
        logits = router[t]
        sel = np.argsort(-logits, kind="stable")[:k]
        gates = 1.0 / (1.0 + np.exp(-logits[sel]))   # sigmoid
        ht = h[t]
        for j, e in enumerate(sel):
            g = (lambda z: z / (1 + np.exp(-z)))(ht @ We["gate"][e].T)
            u = ht @ We["up"][e].T
            out[t] += gates[j] * ((g * u) @ We["down"][e].T)
    return out


def moe_block_float(x, W, We, cfg, window=None):
    H, Hkv, hd = cfg.n_heads, cfg.n_kv, cfg.head_dim; seq = x.shape[0]
    h = _rmsnorm_f(x, W["attn_norm"])
    q = (h @ W["wq"].T).reshape(seq, H, hd); k = (h @ W["wk"].T).reshape(seq, Hkv, hd)
    if window is not None:                         # cohere2 NoPE: full-attention MoE layers get no RoPE
        q = _rope_f(q, cfg.rope_base); k = _rope_f(k, cfg.rope_base)
    v = (h @ W["wv"].T).reshape(seq, Hkv, hd)
    out = np.empty((seq, H, hd)); mask = np.where(_attn_mask(seq, window), -np.inf, 0.0)
    for hh in range(H):
        kv = hh // cfg.rep
        sc = (q[:, hh] @ k[:, kv].T) / math.sqrt(hd) + mask
        sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
        out[:, hh] = sc @ v[:, kv]
    attn = out.reshape(seq, H * hd) @ W["wo"].T
    return x + attn + moe_layer_float(h, We, cfg)


def random_expert_weights_float(cfg, seed=0):
    """Random MoE expert weights (float): router [n_experts,d], and per-expert gate/up [eff,d], down [d,eff]."""
    r = np.random.default_rng(seed); d, E, eff = cfg.d_model, cfg.n_experts, cfg.expert_ffn
    return {
        "router": r.standard_normal((E, d)) * 0.04,
        "gate": r.standard_normal((E, eff, d)) * 0.04,
        "up": r.standard_normal((E, eff, d)) * 0.04,
        "down": r.standard_normal((E, d, eff)) * 0.04,
    }


def expert_weights_to_fixed(Wef, cfg):
    """Convert float expert weights -> fixed-point at fw (router included — it is integer in the engine)."""
    return {k: to_fixed(v, cfg.fw) for k, v in Wef.items()}


# ------------------------------------------------------------------------------- synthetic weights -----------
def random_weights_float(cfg, seed=0):
    r = np.random.default_rng(seed); d = cfg.d_model
    def lin(o, i, s): return r.standard_normal((o, i)) * s
    return {
        "attn_norm": np.abs(r.standard_normal(d)) + 0.5,
        "wq": lin(cfg.n_heads * cfg.head_dim, d, 0.04), "wk": lin(cfg.n_kv * cfg.head_dim, d, 0.04),
        "wv": lin(cfg.n_kv * cfg.head_dim, d, 0.04), "wo": lin(d, cfg.n_heads * cfg.head_dim, 0.04),
        "gate": lin(cfg.ffn, d, 0.04), "up": lin(cfg.ffn, d, 0.04), "down": lin(d, cfg.ffn, 0.04),
        "output_norm": np.abs(r.standard_normal(d)) + 0.5,
        "token_embd": lin(cfg.vocab, d, 0.04),
    }


def weights_to_fixed(Wf, cfg):
    """Convert float weights -> the integer block's weight dict (norms->fa gains, linears->fw)."""
    lin = {"wq", "wk", "wv", "wo", "gate", "up", "down", "token_embd"}
    return {k: (to_fixed(v, cfg.fw) if k in lin else to_fixed(v, cfg.fa)) for k, v in Wf.items()}
