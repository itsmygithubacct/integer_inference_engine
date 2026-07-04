"""Stage-5 gates: the cohere2moe MoE layer — routing determinism (the novelty) + int≈float fidelity."""
import numpy as np
import pytest

from nmc import cohere2 as c2

# Small MoE config (top-2 of 8 experts) — Q4_K-friendly dims (in_features = 256 / expert_ffn = 512).
CFG = c2.Cfg(d_model=256, n_heads=4, n_kv=2, head_dim=64, ffn=512, vocab=512,
             n_experts=8, n_used=2, expert_ffn=512)


def _io(seed=0, seq=6):
    Wf = c2.random_weights_float(CFG, seed); Wq = c2.weights_to_fixed(Wf, CFG)
    Wef = c2.random_expert_weights_float(CFG, seed + 100); Weq = c2.expert_weights_to_fixed(Wef, CFG)
    x = np.random.default_rng(seed + 1).standard_normal((seq, CFG.d_model)) * 0.6
    cos, sin = c2.build_rope_tables(seq, CFG.head_dim, base=int(CFG.rope_base), frac_bits=CFG.fa)
    return Wf, Wq, Wef, Wef and Weq, x, cos, sin


def _rel(a, b): return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-9)


def test_topk_lowidx_tiebreak():
    """All-equal logits ⇒ the lowest k indices are selected (deterministic tie-break)."""
    assert list(c2._topk_lowidx(np.zeros(8, np.int64), 3)) == [0, 1, 2]
    logits = np.array([5, 9, 9, 1, 9, 0, 0, 0], np.int64)     # three 9s at idx 1,2,4
    sel = sorted(c2._topk_lowidx(logits, 2))
    assert sel == [1, 2]                                       # top-2 of the tied 9s = lowest two indices


@pytest.mark.parametrize("seed", range(8))
def test_routing_agreement_int_vs_float(seed):
    """The integer router selects the SAME experts as the float router (the determinism-critical check)."""
    Wf, Wq, Wef, Weq, x, cos, sin = _io(seed)
    h_f = c2._rmsnorm_f(x, Wf["attn_norm"])
    h_i = c2.fixed_point_rmsnorm(c2.to_fixed(x, CFG.fa), CFG.fa, CFG.eps, gain_q=Wq["attn_norm"])
    rl_f = h_f @ Wef["router"].T
    rl_i = c2.linear(h_i, Weq["router"], CFG.fw)
    for t in range(x.shape[0]):
        sf = set(np.argsort(-rl_f[t], kind="stable")[:CFG.n_used])
        si = set(c2._topk_lowidx(rl_i[t], CFG.n_used))
        assert sf == si, (t, sf, si)


@pytest.mark.parametrize("seed", range(8))
def test_moe_layer_fidelity(seed):
    Wf, Wq, Wef, Weq, x, cos, sin = _io(seed)
    h_f = c2._rmsnorm_f(x, Wf["attn_norm"])
    h_i = c2.fixed_point_rmsnorm(c2.to_fixed(x, CFG.fa), CFG.fa, CFG.eps, gain_q=Wq["attn_norm"])
    out_f = c2.moe_layer_float(h_f, Wef, CFG)
    out_i = c2.from_fixed(c2.moe_layer_int(h_i, Weq, CFG), CFG.fa)
    assert _rel(out_i, out_f) < 5e-3, _rel(out_i, out_f)


@pytest.mark.parametrize("seed", range(6))
def test_moe_block_fidelity_and_argmax(seed):
    """Full parallel MoE block + tied head: fidelity + same predicted tokens as float."""
    Wf, Wq, Wef, Weq, x, cos, sin = _io(seed)
    xf = c2.moe_block_float(x, Wf, Wef, CFG, window=None)
    xi = c2.moe_block_int(c2.to_fixed(x, CFG.fa), Wq, Weq, CFG, cos, sin, window=None)
    assert _rel(c2.from_fixed(xi, CFG.fa), xf) < 5e-3
    lf = c2.tied_head_float(xf, Wf); li = c2.from_fixed(c2.tied_head_int(xi, Wq, CFG), CFG.fa)
    assert np.array_equal(li.argmax(-1), lf.argmax(-1))


def test_moe_deterministic():
    _, Wq, _, Weq, x, cos, sin = _io(3)
    xi = c2.to_fixed(x, CFG.fa)
    a = c2.moe_block_int(xi, Wq, Weq, CFG, cos, sin)
    b = c2.moe_block_int(xi, Wq, Weq, CFG, cos, sin)
    assert np.array_equal(a, b) and a.dtype == np.int64
