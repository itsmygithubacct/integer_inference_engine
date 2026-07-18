"""Long-context parity/race gates for the Qwen3.5 CUDA attention primitive.

The file is intentionally standalone and skips cleanly unless both per-host
native libraries are present.  Cases are constructed and released one at a
time so the 4096-token checks do not multiply host/device cache allocations.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from trinote.determinism.fixedpoint import fixed_point_sigmoid
from trinote.infer_int.gpu_bonsai35 import attention_decode_gpu
from trinote.infer_int.reference_bonsai import _head_rmsnorm
from trinote.infer_int.reference_bonsai35 import (
    _apply_partial_neox_rope,
    random_bonsai35_artifact,
)
from trinote.infer_int.q1_native import attention_decode_native, q1_native_available


TOTAL_LENGTHS = (1, 2, 31, 32, 127, 128, 512, 1024, 4095, 4096)


@pytest.fixture(scope="module")
def long_attention_fixture():
    from trinote.infer_int.gpu_native import _load_lib as load_gpu
    from trinote.infer_int.q1_native import _load_lib as load_cpu

    gpu = load_gpu()
    if gpu is None or not hasattr(gpu, "bonsai35_attention_decode_gpu"):
        pytest.skip("Qwen3.5 CUDA attention ABI unavailable; rebuild the per-host GPU library")
    if not q1_native_available() or not hasattr(load_cpu(), "bonsai_attention_decode_i64"):
        pytest.skip("CPU native attention ABI unavailable for the exact parity oracle")
    artifact = random_bonsai35_artifact(seq_len=max(TOTAL_LENGTHS), seed=3501)
    layer = next(layer for layer in artifact["layers"] if layer["kind"] == "attention")
    return artifact, layer


def _transformed_inputs(
    artifact: dict,
    qg: np.ndarray,
    k_new: np.ndarray,
    *,
    position: int,
    q_gain: np.ndarray,
    k_gain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = artifact["config"]
    frac = int(cfg["frac"])
    eps = int(cfg["rmsEpsilonFp2"])
    n_rot = int(cfg["ropeRotDim"])
    q = _head_rmsnorm(
        qg[:, 0, :][:, None, :], frac, q_gain, native=False, eps=eps
    )[:, 0]
    k = _head_rmsnorm(
        k_new[:, None, :], frac, k_gain, native=False, eps=eps
    )[:, 0]
    cos = artifact["cos_fp"][position]
    sin = artifact["sin_fp"][position]
    q = _apply_partial_neox_rope(
        q[:, None], cos[None], sin[None], frac, n_rot
    )[:, 0]
    k = _apply_partial_neox_rope(
        k[:, None], cos[None], sin[None], frac, n_rot
    )[:, 0]
    return q, k, cos, sin


def _cpu_result(
    artifact: dict,
    qg: np.ndarray,
    k_new: np.ndarray,
    v_new: np.ndarray,
    k_prefix: np.ndarray | None,
    v_prefix: np.ndarray | None,
    q_gain: np.ndarray,
    k_gain: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    prefix = 0 if k_prefix is None else int(k_prefix.shape[1])
    q, k_row, cos, sin = _transformed_inputs(
        artifact,
        qg,
        k_new,
        position=prefix,
        q_gain=q_gain,
        k_gain=k_gain,
    )
    K = k_row[:, None, :] if k_prefix is None else np.concatenate(
        (k_prefix, k_row[:, None, :]), axis=1
    )
    V = v_new[:, None, :] if v_prefix is None else np.concatenate(
        (v_prefix, v_new[:, None, :]), axis=1
    )
    cfg = artifact["config"]
    heads = attention_decode_native(
        q, K, V, int(cfg["frac"]), int(cfg["attentionScaleFp"])
    )
    if heads is None:
        return None, k_row, cos, sin
    gate = fixed_point_sigmoid(qg[:, 1], int(cfg["frac"]))
    return (heads * gate) >> int(cfg["frac"]), k_row, cos, sin


def _gpu_result(
    artifact: dict,
    qg: np.ndarray,
    k_new: np.ndarray,
    v_new: np.ndarray,
    k_prefix: np.ndarray | None,
    v_prefix: np.ndarray | None,
    q_gain: np.ndarray,
    k_gain: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
):
    return attention_decode_gpu(
        qg,
        k_new,
        v_new,
        k_prefix,
        v_prefix,
        q_gain,
        k_gain,
        cos,
        sin,
        artifact,
    )


def _mixed_case(artifact: dict, length: int):
    cfg = artifact["config"]
    h, hkv, hd = int(cfg["n_heads"]), int(cfg["n_heads_kv"]), int(cfg["head_dim"])
    rng = np.random.default_rng(9000 + length)
    qg = rng.integers(-20_000, 20_001, (h, 2, hd), dtype=np.int64)
    k_new = rng.integers(-20_000, 20_001, (hkv, hd), dtype=np.int64)
    v_new = rng.integers(-20_000, 20_001, (hkv, hd), dtype=np.int64)
    if length == 1:
        k_prefix = v_prefix = None
    else:
        k_prefix = rng.integers(
            -20_000, 20_001, (hkv, length - 1, hd), dtype=np.int64
        )
        v_prefix = rng.integers(
            -20_000, 20_001, (hkv, length - 1, hd), dtype=np.int64
        )
    return qg, k_new, v_new, k_prefix, v_prefix


def test_cuda_attention_matches_cpu_at_long_context_boundaries(long_attention_fixture):
    artifact, layer = long_attention_fixture
    q_gain = layer["q_norm_gain_fp"]
    k_gain = layer["k_norm_gain_fp"]
    for length in TOTAL_LENGTHS:
        qg, k_new, v_new, K, V = _mixed_case(artifact, length)
        expected, expected_k, cos, sin = _cpu_result(
            artifact, qg, k_new, v_new, K, V, q_gain, k_gain
        )
        assert expected is not None, f"CPU oracle unexpectedly declined safe length {length}"
        actual = _gpu_result(
            artifact, qg, k_new, v_new, K, V, q_gain, k_gain, cos, sin
        )
        assert actual is not None, f"CUDA attention declined safe length {length}"
        assert np.array_equal(actual[0], expected), f"attention mismatch at total length {length}"
        assert np.array_equal(actual[1], expected_k), f"K-row mismatch at total length {length}"


def test_cuda_attention_tied_scores_and_alternating_values_at_4096(long_attention_fixture):
    artifact, layer = long_attention_fixture
    cfg = artifact["config"]
    h, hkv, hd = int(cfg["n_heads"]), int(cfg["n_heads_kv"]), int(cfg["head_dim"])
    length = 4096
    qg = np.zeros((h, 2, hd), dtype=np.int64)  # every q.K score ties exactly at zero
    k_new = np.arange(hkv * hd, dtype=np.int64).reshape(hkv, hd) - hd // 2
    K = np.zeros((hkv, length - 1, hd), dtype=np.int64)
    signs = np.where(np.arange(length) & 1, -1, 1).astype(np.int64)
    channels = (np.arange(hd, dtype=np.int64) % 17) + 1
    all_v = signs[:, None] * channels[None, :]
    V = np.broadcast_to(all_v[:-1], (hkv, length - 1, hd)).copy()
    v_new = np.broadcast_to(all_v[-1], (hkv, hd)).copy()
    expected, expected_k, cos, sin = _cpu_result(
        artifact,
        qg,
        k_new,
        v_new,
        K,
        V,
        layer["q_norm_gain_fp"],
        layer["k_norm_gain_fp"],
    )
    assert expected is not None
    actual = _gpu_result(
        artifact,
        qg,
        k_new,
        v_new,
        K,
        V,
        layer["q_norm_gain_fp"],
        layer["k_norm_gain_fp"],
        cos,
        sin,
    )
    assert actual is not None
    assert np.array_equal(actual[0], expected)
    assert np.array_equal(actual[1], expected_k)


def test_cuda_attention_fail_loud_qk_and_probability_v_guards(long_attention_fixture):
    artifact, layer = long_attention_fixture
    cfg = artifact["config"]
    h, hkv, hd = int(cfg["n_heads"]), int(cfg["n_heads_kv"]), int(cfg["head_dim"])
    qg = np.zeros((h, 2, hd), dtype=np.int64)
    qg[:, 0, 0] = 1
    k_new = np.zeros((hkv, hd), dtype=np.int64)
    k_new[:, 0] = 1
    v_new = np.zeros((hkv, hd), dtype=np.int64)
    # Pre-gain normalized values are about sqrt(hd)*2^frac.  This gain keeps
    # RMSNorm's own multiply in int64 but makes maxQ*maxK*hd exceed INT64_MAX.
    huge_safe_gain = np.full(hd, 1 << 40, dtype=np.int64)
    expected, _krow, cos, sin = _cpu_result(
        artifact, qg, k_new, v_new, None, None, huge_safe_gain, huge_safe_gain
    )
    assert expected is None, "CPU q.K guard fixture did not leave the exact envelope"
    assert _gpu_result(
        artifact,
        qg,
        k_new,
        v_new,
        None,
        None,
        huge_safe_gain,
        huge_safe_gain,
        cos,
        sin,
    ) is None

    # With q=K=0, L=1 gives probability exactly 2^frac.  A 2^50 V
    # therefore fails only the probability@V contraction bound.
    qg.fill(0)
    k_new.fill(0)
    v_new.fill(1 << 50)
    expected, _krow, cos, sin = _cpu_result(
        artifact,
        qg,
        k_new,
        v_new,
        None,
        None,
        layer["q_norm_gain_fp"],
        layer["k_norm_gain_fp"],
    )
    assert expected is None, "CPU probability.V guard fixture did not leave the exact envelope"
    assert _gpu_result(
        artifact,
        qg,
        k_new,
        v_new,
        None,
        None,
        layer["q_norm_gain_fp"],
        layer["k_norm_gain_fp"],
        cos,
        sin,
    ) is None


def test_cuda_attention_long_boundary_is_repeatable(long_attention_fixture):
    artifact, layer = long_attention_fixture
    for length in (4095, 4096):
        qg, k_new, v_new, K, V = _mixed_case(artifact, length)
        expected, expected_k, cos, sin = _cpu_result(
            artifact,
            qg,
            k_new,
            v_new,
            K,
            V,
            layer["q_norm_gain_fp"],
            layer["k_norm_gain_fp"],
        )
        assert expected is not None
        for repetition in range(4):
            actual = _gpu_result(
                artifact,
                qg,
                k_new,
                v_new,
                K,
                V,
                layer["q_norm_gain_fp"],
                layer["k_norm_gain_fp"],
                cos,
                sin,
            )
            assert actual is not None, (length, repetition)
            assert np.array_equal(actual[0], expected), (length, repetition)
            assert np.array_equal(actual[1], expected_k), (length, repetition)


def test_cuda_attention_release_geometry_matches_cpu_at_4096(long_attention_fixture):
    # One non-parametrized release-shape case catches assumptions hidden by the
    # lightweight H=2/Hkv=1/hd=64 matrix.  K+V are about 64 MiB total and are
    # released when this test returns.
    del long_attention_fixture  # fixture establishes both per-host ABIs
    frac, h, hkv, hd, n_rot, length = 16, 24, 4, 256, 64, 4096
    artifact = {
        "config": {
            "n_heads": h,
            "n_heads_kv": hkv,
            "head_dim": hd,
            "ropeRotDim": n_rot,
            "frac": frac,
            "rmsEpsilonFp2": round(1e-6 * (1 << (2 * frac))),
            "attentionScaleFp": round((1.0 / math.sqrt(hd)) * (1 << frac)),
        }
    }
    cos = np.full(n_rot // 2, 1 << frac, dtype=np.int64)
    sin = np.zeros(n_rot // 2, dtype=np.int64)
    artifact["cos_fp"] = np.broadcast_to(cos, (length, n_rot // 2))
    artifact["sin_fp"] = np.broadcast_to(sin, (length, n_rot // 2))
    gain = np.full(hd, 1 << frac, dtype=np.int64)
    rng = np.random.default_rng(27_4096)
    qg = rng.integers(-20_000, 20_001, (h, 2, hd), dtype=np.int64)
    k_new = rng.integers(-20_000, 20_001, (hkv, hd), dtype=np.int64)
    v_new = rng.integers(-20_000, 20_001, (hkv, hd), dtype=np.int64)
    K = rng.integers(-20_000, 20_001, (hkv, length - 1, hd), dtype=np.int64)
    V = rng.integers(-20_000, 20_001, (hkv, length - 1, hd), dtype=np.int64)

    expected, expected_k, cos_row, sin_row = _cpu_result(
        artifact, qg, k_new, v_new, K, V, gain, gain
    )
    assert expected is not None
    actual = _gpu_result(
        artifact, qg, k_new, v_new, K, V, gain, gain, cos_row, sin_row
    )
    assert actual is not None
    assert np.array_equal(actual[0], expected)
    assert np.array_equal(actual[1], expected_k)
