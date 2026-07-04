"""Stage-3 gates for the cohere2 DENSE path: integer ≈ float (fidelity) + integer determinism (reproducible)."""
import numpy as np
import pytest

from nmc import cohere2 as c2
from nmc import qk_codec as qk

# Q4_K-friendly dims (in_features multiples of 256 so the codec-integration test packs cleanly).
CFG = c2.Cfg(d_model=256, n_heads=4, n_kv=2, head_dim=64, ffn=512, vocab=512)


def _io(seed=0, seq=6):
    Wf = c2.random_weights_float(CFG, seed)
    Wq = c2.weights_to_fixed(Wf, CFG)
    x = np.random.default_rng(seed + 1).standard_normal((seq, CFG.d_model)) * 0.6
    cos, sin = c2.build_rope_tables(seq, CFG.head_dim, base=int(CFG.rope_base), frac_bits=CFG.fa)
    return Wf, Wq, x, cos, sin


def _rel(a, b): return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-9)


@pytest.mark.parametrize("seed", range(8))
def test_dense_block_fidelity(seed):
    Wf, Wq, x, cos, sin = _io(seed)
    out_f = c2.dense_block_float(x, Wf, CFG)
    out_i = c2.from_fixed(c2.dense_block_int(c2.to_fixed(x, CFG.fa), Wq, CFG, cos, sin), CFG.fa)
    assert _rel(out_i, out_f) < 5e-3, _rel(out_i, out_f)


@pytest.mark.parametrize("seed", range(8))
def test_tied_head_fidelity_and_argmax(seed):
    Wf, Wq, x, cos, sin = _io(seed)
    xf = c2.dense_block_float(x, Wf, CFG)
    xi = c2.dense_block_int(c2.to_fixed(x, CFG.fa), Wq, CFG, cos, sin)
    lf = c2.tied_head_float(xf, Wf)
    li = c2.from_fixed(c2.tied_head_int(xi, Wq, CFG), CFG.fa)
    assert _rel(li, lf) < 1e-2, _rel(li, lf)
    assert np.array_equal(li.argmax(-1), lf.argmax(-1))      # same predicted tokens


def test_dense_block_deterministic():
    """Integer path is reproducible bit-for-bit (the determinism keystone) — run twice, identical int64."""
    _, Wq, x, cos, sin = _io(3)
    xi = c2.to_fixed(x, CFG.fa)
    a = c2.dense_block_int(xi, Wq, CFG, cos, sin)
    b = c2.dense_block_int(xi, Wq, CFG, cos, sin)
    assert np.array_equal(a, b) and a.dtype == np.int64


def test_codec_integration_q4k_linear():
    """End-to-end Stage2+Stage3: a Q4_K-quantized linear, dequantized by the integer codec, matches the float
    dequant of the SAME blocks within fidelity — proving the codec feeds the block correctly."""
    out_f, in_f, fw = 8, 256, CFG.fw            # one Q4_K super-block per row
    blocks = [qk.random_q4k(500 + r) for r in range(out_f)]
    W_int = np.stack([qk.dequant_q4k_int(**b, frac=fw) for b in blocks])           # [out,in] at 2**fw
    W_flt = np.stack([qk.dequant_q4k_float(**b) for b in blocks])
    x = np.random.default_rng(9).standard_normal(in_f) * 0.5
    y_i = c2.from_fixed(c2.linear(c2.to_fixed(x, CFG.fa)[None, :], W_int, fw), CFG.fa)[0]
    y_f = W_flt @ x
    assert _rel(y_i, y_f) < 1e-3, _rel(y_i, y_f)
