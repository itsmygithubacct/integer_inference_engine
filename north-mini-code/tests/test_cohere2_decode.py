"""KV-cache decode gate: generating with the incremental KV cache is BYTE-IDENTICAL to re-prefilling the
growing sequence from scratch each step — for a dense+MoE stack with interleaved SWA/full attention. This is
the determinism contract for the decode path (the prerequisite for fidelity eval + receipts)."""
import numpy as np
import pytest

from nmc import cohere2 as c2

# Small dense+MoE stack; sliding_window=3 so SWA actually bites over the prompt+decode length.
CFG = c2.Cfg(d_model=256, n_heads=4, n_kv=2, head_dim=64, ffn=512, vocab=512,
             n_experts=8, n_used=2, expert_ffn=512, sliding_window=3)
NL = 5                                        # layers 0,4 full; 1,2,3 sliding (is_full_layer); block 0 dense
PROMPT = [3, 9, 1, 5, 2, 8]
N_NEW = 12


def _build(seed=0):
    rng = np.random.default_rng(seed)
    layers = []
    for li in range(NL):
        Wq = c2.weights_to_fixed(c2.random_weights_float(CFG, seed + li), CFG)
        We = None if li == 0 else c2.expert_weights_to_fixed(c2.random_expert_weights_float(CFG, seed + 100 + li), CFG)
        layers.append((Wq, We))
    head = {"output_norm": c2.to_fixed(np.abs(rng.standard_normal(CFG.d_model)) + 0.5, CFG.fa),
            "token_embd": c2.to_fixed(rng.standard_normal((CFG.vocab, CFG.d_model)) * 0.04, CFG.fw)}
    emb = c2.to_fixed(rng.standard_normal((CFG.vocab, CFG.d_model)) * 0.04, CFG.fa)
    cos, sin = c2.build_rope_tables(64, CFG.head_dim, base=int(CFG.rope_base), frac_bits=CFG.fa)
    return layers, head, (lambda t: emb[t]), cos, sin


def _greedy(row, pos, hist):
    return int(np.asarray(row).argmax())


def test_decode_matches_prefill_byte_exact():
    layers, head, embed, cos, sin = _build()
    cached = c2.generate(PROMPT, embed, layers, head, CFG, cos, sin, N_NEW, _greedy, use_cache=True)
    fresh = c2.generate(PROMPT, embed, layers, head, CFG, cos, sin, N_NEW, _greedy, use_cache=False)
    assert cached == fresh, (cached, fresh)
    assert len(cached) == N_NEW


def test_cached_prefill_matches_original_blocks():
    """The cache-aware block (prefill, m positions) is byte-identical to the original dense_block_int /
    moe_block_int — so real_generate's prefill == real_forward_int's proven forward."""
    layers, head, embed, cos, sin = _build()
    x0 = np.stack([embed(t) for t in PROMPT])
    cache = c2.KVCache(NL); xc = x0.copy()
    for li, (W, We) in enumerate(layers):
        xc = c2.block_cached(xc, W, We, CFG, cos, sin, cache, li, c2.window_for_layer(CFG, li))
    xo = x0.copy()
    for li, (W, We) in enumerate(layers):
        win = c2.window_for_layer(CFG, li)
        xo = (c2.dense_block_int(xo, W, CFG, cos, sin, win) if We is None
              else c2.moe_block_int(xo, W, We, CFG, cos, sin, win))
    assert np.array_equal(xc, xo)


def test_decode_deterministic():
    layers, head, embed, cos, sin = _build()
    a = c2.generate(PROMPT, embed, layers, head, CFG, cos, sin, N_NEW, _greedy, use_cache=True)
    b = c2.generate(PROMPT, embed, layers, head, CFG, cos, sin, N_NEW, _greedy, use_cache=True)
    assert a == b


def test_cache_grows_and_window_bites():
    """Cache length tracks the sequence, and the sliding window actually changes the output (guards a no-op
    window: a tiny window must diverge from full attention over a long-enough sequence)."""
    layers, head, embed, cos, sin = _build()
    cache = c2.KVCache(NL)
    c2._run_layers_cached(np.stack([embed(t) for t in PROMPT]), layers, CFG, cos, sin, cache)
    assert cache.length(0) == len(PROMPT) and cache.length(1) == len(PROMPT)
    c2._run_layers_cached(embed(PROMPT[0])[None, :], layers, CFG, cos, sin, cache)
    assert cache.length(0) == len(PROMPT) + 1

    wide = c2.Cfg(**{**CFG.__dict__, "sliding_window": 4096})
    g_swa = c2.generate(PROMPT, embed, layers, head, CFG, cos, sin, N_NEW, _greedy, use_cache=True)
    g_full = c2.generate(PROMPT, embed, layers, head, wide, cos, sin, N_NEW, _greedy, use_cache=True)
    assert g_swa != g_full       # window=3 vs ~unbounded must differ over 18 positions
