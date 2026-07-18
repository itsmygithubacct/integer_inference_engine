from __future__ import annotations

import copy

import numpy as np
import pytest

from trinote.determinism.fixedpoint import fixed_point_matmul
from trinote.infer_int import reference_bonsai35 as rb35
from trinote.infer_int.q1_native import (
    attention_decode_native,
    q1_native_stats,
    q1_prepare_apply_many_native,
    q1_selected_isa,
    q1_set_isa,
    q1_weight_group,
)
from trinote.infer_int.reference_bonsai import q1_linear_ref
from trinote.infer_int.reference_bonsai35 import (
    BonsaiQwen35ReferenceModel,
    _Qwen35Cache,
    _apply_partial_neox_rope,
    _full_attention,
    _recurrent_attention,
    random_bonsai35_artifact,
)


@pytest.mark.parametrize("prefer_lut32", [False, True])
def test_qwen35_q1_extreme_scale_mask_shift_and_wrap_parity(prefer_lut32):
    """Cover Q1 masks, scale extrema, negative shifts, the int32 guard, and wrap.

    The last activation row forces the guarded int32 subset builder to reject
    before output and exercises the canonical mod-2^64 Q1 accumulation policy.
    """
    bits = np.empty((6, 1, 16), dtype=np.uint8)
    for row, byte in enumerate((0x00, 0xFF, 0x55, 0xAA, 0x0F, 0xF0)):
        bits[row].fill(byte)
    scales = np.asarray(
        [
            np.iinfo(np.int32).min,
            np.iinfo(np.int32).max,
            -1,
            0,
            1,
            1 << 16,
        ],
        dtype=np.int32,
    ).reshape(6, 1)
    around_shift = np.resize(
        np.asarray(
            [-65537, -65536, -65535, -1, 0, 1, 65535, 65536, 65537],
            dtype=np.int64,
        ),
        128,
    )
    x = np.stack(
        [
            np.zeros(128, dtype=np.int64),
            np.ones(128, dtype=np.int64),
            -np.ones(128, dtype=np.int64),
            around_shift,
            np.full(128, np.iinfo(np.int64).max, dtype=np.int64),
        ]
    )
    group = q1_weight_group(((bits, scales),))
    q1_native_stats(reset=True)
    with np.errstate(over="ignore"):
        expected = q1_linear_ref(x, bits, scales, 16)
        got = q1_prepare_apply_many_native(
            x, group, 16, prefer_lut32=prefer_lut32
        )
    if got is None:
        pytest.skip("fused native Q1 kernel is unavailable")
    assert np.array_equal(got[0], expected)
    stats = q1_native_stats()
    assert stats["u64_calls"] == 1
    if prefer_lut32:
        assert stats["lut32_fallbacks"] == 1
        assert stats["lut32_hits"] == 0
    else:
        assert stats["lut32_fallbacks"] == 0


def test_qwen35_forced_portable_and_avx2_match_at_real_down_projection_width():
    """Exercise the AVX2 gather threshold (17,408 inputs / 136 blocks)."""
    rng = np.random.default_rng(701)
    blocks = 136
    x = rng.integers(-(1 << 14), 1 << 14, size=(2, blocks * 128), dtype=np.int64)
    bits = rng.integers(0, 256, size=(7, blocks, 16), dtype=np.uint8)
    scales = rng.integers(-(1 << 15), 1 << 15, size=(7, blocks), dtype=np.int32)
    group = q1_weight_group(((bits, scales),))
    expected = q1_linear_ref(x, bits, scales, 16)
    original = q1_selected_isa()
    try:
        assert q1_set_isa("portable") == "portable"
        portable = q1_prepare_apply_many_native(
            x, group, 16, prefer_lut32=True
        )
        if portable is None:
            pytest.skip("fused native Q1 kernel is unavailable")
        try:
            assert q1_set_isa("avx2") == "avx2"
        except RuntimeError:
            pytest.skip("host does not support forced AVX2")
        avx2 = q1_prepare_apply_many_native(x, group, 16, prefer_lut32=True)
        assert avx2 is not None
        assert np.array_equal(portable[0], expected)
        assert np.array_equal(avx2[0], expected)
    finally:
        # Do not leak process-global ISA selection into unrelated tests.
        q1_set_isa(original if original in {"portable", "avx2"} else "auto")


def test_qwen35_kv_cache_grows_exactly_at_16_17_32_33():
    cache = _Qwen35Cache(1)
    expected_k = np.empty((2, 0, 3), dtype=np.int64)
    expected_v = np.empty((2, 0, 3), dtype=np.int64)
    cursor = 0
    for add, expected_capacity in ((15, 16), (1, 16), (1, 32), (15, 32), (1, 64)):
        values = np.arange(cursor, cursor + 2 * add * 3, dtype=np.int64).reshape(2, add, 3)
        keys = values - 1000
        vals = values + 1000
        expected_k = np.concatenate((expected_k, keys), axis=1)
        expected_v = np.concatenate((expected_v, vals), axis=1)
        cache.extend_attention(0, keys, vals)
        cursor += values.size
        assert cache.lengths[0] == expected_k.shape[1]
        assert cache.k_buf[0].shape[1] == expected_capacity
        assert cache.v_buf[0].shape[1] == expected_capacity
        assert np.array_equal(cache.k[0], expected_k)
        assert np.array_equal(cache.v[0], expected_v)


def test_qwen35_tied_attention_scores_are_causal_and_cache_reusable():
    artifact = random_bonsai35_artifact(seed=702, seq_len=40)
    li = 3
    layer = copy.deepcopy(artifact["layers"][li])
    # Zero Q/K scales produce exactly tied, all-zero unmasked scores. V and the
    # output projection remain nonzero, so a causal-mask error is observable.
    layer["wqg_scale_fp"].fill(0)
    layer["wk_scale_fp"].fill(0)
    rng = np.random.default_rng(702)
    x = rng.integers(
        -(1 << 15),
        1 << 15,
        size=(3, artifact["config"]["dModel"]),
        dtype=np.int64,
    )

    short_cache = _Qwen35Cache(4)
    short = _full_attention(x[:2], layer, artifact, short_cache, li, 0, native=False)
    full_cache = _Qwen35Cache(4)
    full = _full_attention(x, layer, artifact, full_cache, li, 0, native=False)
    assert np.array_equal(short, full[:2])

    reused_cache = _Qwen35Cache(4)
    prefix = _full_attention(x[:2], layer, artifact, reused_cache, li, 0, native=False)
    suffix = _full_attention(x[2:], layer, artifact, reused_cache, li, 2, native=False)
    assert np.array_equal(prefix, full[:2])
    assert np.array_equal(suffix, full[2:])
    assert np.array_equal(reused_cache.k[li], full_cache.k[li])
    assert np.array_equal(reused_cache.v[li], full_cache.v[li])


def test_qwen35_recurrent_fail_loud_guard_is_transactional():
    artifact = random_bonsai35_artifact(seed=703, seq_len=8)
    layer = copy.deepcopy(artifact["layers"][0])
    frac = int(artifact["config"]["frac"])
    # Positive A makes softplus(dt) * A positive, violating the committed
    # non-positive decay gate. Validation must happen before conv/state commit.
    layer["ssm_a_fp"] = np.full_like(layer["ssm_a_fp"], 1 << frac)
    cache = _Qwen35Cache(4)
    x = np.zeros((1, artifact["config"]["dModel"]), dtype=np.int64)
    with pytest.raises(OverflowError, match="decay gate became positive"):
        _recurrent_attention(x, layer, artifact, cache, 0, native=False)
    assert cache.conv[0] is None
    assert cache.state[0] is None
    assert cache.t == 0


def test_qwen35_attention_fail_loud_restores_logical_kv(monkeypatch):
    artifact = random_bonsai35_artifact(seed=704, seq_len=8)
    li = 3
    layer = artifact["layers"][li]
    x = np.ones((2, artifact["config"]["dModel"]), dtype=np.int64)
    cache = _Qwen35Cache(4)
    _full_attention(x[:1], layer, artifact, cache, li, 0, native=False)
    old_k = cache.k[li].copy()
    old_v = cache.v[li].copy()
    old_length = cache.lengths[li]

    def fail_loud(*_args, **_kwargs):
        raise OverflowError("synthetic attention range guard")

    monkeypatch.setattr(rb35, "fixed_point_matmul", fail_loud)
    with pytest.raises(OverflowError, match="range guard"):
        _full_attention(x[1:], layer, artifact, cache, li, 1, native=False)
    assert cache.lengths[li] == old_length
    assert np.array_equal(cache.k[li], old_k)
    assert np.array_equal(cache.v[li], old_v)
    assert cache.t == 0


def test_qwen35_native_attention_range_guard_falls_back_without_mutation():
    q = np.asarray([[np.iinfo(np.int64).max]], dtype=np.int64)
    k = np.asarray([[[2]]], dtype=np.int64)
    v = np.asarray([[[1]]], dtype=np.int64)
    before = (q.copy(), k.copy(), v.copy())
    got = attention_decode_native(q, k, v, 16, 1)
    if got is not None:
        pytest.skip("native attention kernel unavailable or guard envelope changed")
    assert np.array_equal(q, before[0])
    assert np.array_equal(k, before[1])
    assert np.array_equal(v, before[2])
    with pytest.raises(OverflowError):
        fixed_point_matmul(q, k[0].T, 16)


def test_qwen35_negative_rope_floor_shift_boundaries():
    frac = 4
    heads = np.asarray([[[-33, -32, -31, -17, -16, -15]]], dtype=np.int64)
    cos = np.asarray([[15, -15]], dtype=np.int64)
    sin = np.asarray([[1, -1]], dtype=np.int64)
    got = _apply_partial_neox_rope(heads, cos, sin, frac, 4)
    x = heads[0, 0]
    expected = heads.copy()
    expected[0, 0, 0] = (int(x[0]) * 15 - int(x[2]) * 1) // (1 << frac)
    expected[0, 0, 1] = (int(x[1]) * -15 - int(x[3]) * -1) // (1 << frac)
    expected[0, 0, 2] = (int(x[0]) * 1 + int(x[2]) * 15) // (1 << frac)
    expected[0, 0, 3] = (int(x[1]) * -1 + int(x[3]) * -15) // (1 << frac)
    assert np.array_equal(got, expected)
    assert np.array_equal(got[..., 4:], heads[..., 4:])


def test_qwen35_tied_output_logits_choose_lowest_token_native_and_oracle():
    artifact = random_bonsai35_artifact(seed=705)
    artifact["output_scale_fp"].fill(0)
    oracle = BonsaiQwen35ReferenceModel(artifact)
    expected = oracle.prefill_logits([1, 2, 3])
    assert np.count_nonzero(expected) == 0
    assert oracle.generate_greedy_tokens_cached([1, 2, 3], 2) == [0, 0]

    native = BonsaiQwen35ReferenceModel(artifact)
    if not native.enable_native():
        pytest.skip("native Qwen3.5 runtime unavailable")
    assert np.array_equal(native.prefill_logits([1, 2, 3]), expected)
    assert native.generate_greedy_tokens_cached([1, 2, 3], 2) == [0, 0]
