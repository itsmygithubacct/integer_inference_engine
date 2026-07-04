"""Kernel parity gate: the C `qk_linear` must be BYTE-IDENTICAL to the numpy integer oracle
(qk_codec.dequant_*_tensor + cohere2.linear), for Q4_K and Q6_K, across shapes/seeds/token-counts."""
import numpy as np
import pytest

from nmc import qk_codec as qk
from nmc import qk_native
from tests.test_qk_codec import _q4k_raw, _q6k_raw

FW = 24
pytestmark = pytest.mark.skipif(not qk_native.available(),
                                reason="kernel not built (run tools/build_nmc_kernel.sh)")


def _lin(x, W, fw):                                                           # numpy big-int reference matmul
    return ((np.asarray(x, dtype=object) @ np.asarray(W, dtype=object).T) >> fw).astype(np.int64)


def _oracle(raw, out_f, n_blocks, qtype, x):
    deq = qk.dequant_q4k_tensor if qtype == qk_native.Q4_K else qk.dequant_q6k_tensor
    W = deq(raw, out_f * n_blocks * 256, FW).reshape(out_f, n_blocks * 256)   # [out, in] at 2**FW
    return _lin(x, W, FW)                                                     # kernel-test is self-contained


@pytest.mark.parametrize("qtype", [qk_native.Q4_K, qk_native.Q6_K])
@pytest.mark.parametrize("out_f,n_blocks,T", [(8, 1, 1), (16, 2, 3), (5, 3, 4), (32, 8, 2), (1, 4, 6)])
def test_kernel_matches_oracle(qtype, out_f, n_blocks, T):
    rng = np.random.default_rng(out_f * 100 + n_blocks * 7 + T + qtype)
    gen, pack = ((qk.random_q4k, _q4k_raw) if qtype == qk_native.Q4_K else (qk.random_q6k, _q6k_raw))
    raw = b"".join(pack(gen(seed)) for seed in range(out_f * n_blocks))    # out_f rows × n_blocks blocks
    x = rng.integers(-(1 << (FW - 6)), 1 << (FW - 6), size=(T, n_blocks * 256), dtype=np.int64)
    got = qk_native.qk_linear(raw, x, out_f, n_blocks, FW, qtype)
    want = _oracle(raw, out_f, n_blocks, qtype, x)
    assert np.array_equal(got, want), (qtype, out_f, n_blocks, T, np.abs(got - want).max())


def test_kernel_negative_floor_matches():
    """Negative accumulators: arithmetic >> floors toward -inf identically in C (__int128) and numpy big-int."""
    raw = b"".join(_q4k_raw(qk.random_q4k(s)) for s in range(4 * 4))      # out_f=4 × n_blocks=4 blocks
    x = -np.abs(np.random.default_rng(1).integers(1, 1 << 14, size=(2, 4 * 256), dtype=np.int64))
    got = qk_native.qk_linear(raw, x, 4, 4, FW, qk_native.Q4_K)
    want = _oracle(raw, 4, 4, qk_native.Q4_K, x)
    assert np.array_equal(got, want)
