from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from trinote.infer_int.artifact_io_bonsai import (
    ARTIFACT_FORMAT_BONSAI_QWEN35,
    load_artifact_bonsai,
    save_artifact_bonsai,
)
from trinote.infer_int.reference_bonsai35 import (
    BonsaiQwen35ReferenceModel,
    fixed_point_exp_negative_lut,
    fixed_point_softplus_lut,
    random_bonsai35_artifact,
)


_REAL_EXECUTOR_OPT_IN = os.environ.get("TRINOTE_RUN_BONSAI35_REAL_EXECUTOR", "") == "1"


def _real_executor_artifact():
    from trinote.infer_int.q1_native import Bonsai35NativeExecutor

    path = Path(os.environ.get(
        "BONSAI_INTEGER_27B_ARTIFACT",
        str(Path.home() / ".local/trinote/models/Bonsai-27B-Q1_0-int-qwen35.safetensors"),
    ))
    assert path.is_file(), path
    artifact, _info = load_artifact_bonsai(path)
    return artifact, Bonsai35NativeExecutor(artifact)


def test_qwen35_artifact_roundtrip_preserves_hybrid_schema(tmp_path):
    artifact = random_bonsai35_artifact(seed=9)
    path = tmp_path / "tiny-qwen35.safetensors"
    digest = save_artifact_bonsai(artifact, path, provenance={"kind": "test"})
    loaded, info = load_artifact_bonsai(path)

    assert info["format"] == ARTIFACT_FORMAT_BONSAI_QWEN35
    assert info["digest"] == digest
    assert [layer["kind"] for layer in loaded["layers"]] == [
        "recurrent", "recurrent", "recurrent", "attention"
    ]
    assert loaded["embed_scale_fp"].dtype == np.int32
    before = BonsaiQwen35ReferenceModel(artifact).forward([1, 2, 3])
    after = BonsaiQwen35ReferenceModel(loaded).forward([1, 2, 3])
    assert np.array_equal(before, after)


def test_qwen35_artifact_write_failure_preserves_existing_destination(tmp_path, monkeypatch):
    from trinote.infer_int import artifact_io_bonsai as artifact_io

    artifact = random_bonsai35_artifact(seed=10)
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"known-good-previous-artifact")

    def interrupted_save(_tensors, temp_path, *, metadata):
        assert metadata["trinote"]
        Path(temp_path).write_bytes(b"partial")
        raise RuntimeError("simulated interrupted import")

    monkeypatch.setattr(artifact_io, "save_file", interrupted_save)
    with pytest.raises(RuntimeError, match="interrupted import"):
        artifact_io.save_artifact_bonsai(artifact, path)
    assert path.read_bytes() == b"known-good-previous-artifact"
    assert list(tmp_path.iterdir()) == [path]


def test_artifact_validation_cli_rejects_partial_file(tmp_path, capsys):
    from trinote.cli.validate_bonsai_artifact_cli import main

    good = tmp_path / "good.safetensors"
    save_artifact_bonsai(random_bonsai35_artifact(seed=11), good)
    assert main(["--artifact", str(good), "--architecture", "qwen35"]) == 0
    assert '"ok": true' in capsys.readouterr().out

    partial = tmp_path / "partial.safetensors"
    partial.write_bytes(b"partial")
    assert main(["--artifact", str(partial), "--architecture", "qwen35"]) == 2
    assert "validation failed" in capsys.readouterr().err


def test_qwen35_cached_prefill_matches_full_forward_last_row():
    model = BonsaiQwen35ReferenceModel(random_bonsai35_artifact(seed=4))
    ids = [7, 3, 11, 2]
    assert np.array_equal(model.forward(ids, last_only=True), model.prefill_logits(ids))


def test_qwen35_native_q1_path_matches_integer_oracle_if_available():
    artifact = random_bonsai35_artifact(seed=12)
    oracle = BonsaiQwen35ReferenceModel(artifact)
    expected = oracle.forward([1, 5, 8])
    native = BonsaiQwen35ReferenceModel(artifact)
    if not native.enable_native():
        return
    assert np.array_equal(native.forward([1, 5, 8]), expected)
    assert native.generate_greedy_tokens_cached([1, 5], 3) == oracle.generate_greedy_tokens_cached([1, 5], 3)


@pytest.mark.skipif(
    not _REAL_EXECUTOR_OPT_IN,
    reason="set TRINOTE_RUN_BONSAI35_REAL_EXECUTOR=1 for the external 4 GiB artifact",
)
def test_real_release_executor_one_call_team_output_and_reset(monkeypatch):
    from trinote.infer_int.reference_bonsai import _rmsnorm
    from trinote.infer_int.reference_bonsai35 import _Qwen35Cache
    from trinote.infer_int.trace_bonsai35 import (
        canonical_cache_digest,
        canonical_cache_record,
    )

    artifact, executor = _real_executor_artifact()
    expected = BonsaiQwen35ReferenceModel(artifact)
    # Keep this comparison on the canonical Python graph plus native
    # primitives, not the full resident executor under test.
    monkeypatch.setenv("TRINOTE_BONSAI35_MODEL_EXECUTOR", "0")
    assert expected.enable_native()
    oracle_cache = _Qwen35Cache(len(artifact["layers"]))
    oracle_residual = expected._run_layers([12675], oracle_cache)
    last = _rmsnorm(
        oracle_residual[-1:],
        int(expected.cfg["frac"]),
        artifact["final_norm_gain_fp"],
        native=True,
        eps=int(expected.cfg["rmsEpsilonFp2"]),
    )
    expected_logits = expected._output_linear(last)

    before = executor.stats()
    actual_logits = executor.prefill_logits([12675])
    after = executor.stats()
    assert np.array_equal(actual_logits, expected_logits)
    assert int(actual_logits.argmax(axis=1)[0]) == 11
    assert after["team_entries"] - before["team_entries"] == 1
    assert after["q1_groups"] - before["q1_groups"] == 257  # 64*4 + output
    assert executor.position() == 1

    kinds = [layer["kind"] for layer in artifact["layers"]]

    def oracle_tensor(layer, name):
        return getattr(oracle_cache, name)[layer]

    oracle_commitment = canonical_cache_record(
        kinds,
        position=oracle_cache.t,
        tensor_for=oracle_tensor,
        last_residual=oracle_residual,
    )
    native_commitment = canonical_cache_record(
        kinds,
        position=executor.position(),
        tensor_for=executor.export_cache_tensor,
        last_residual=executor.export_last_residual(),
    )
    assert native_commitment == oracle_commitment
    assert canonical_cache_digest(native_commitment) == canonical_cache_digest(
        oracle_commitment
    )
    executor.reset()
    assert executor.position() == 0
    assert executor.prefill_argmax([12675]) == 11


@pytest.mark.skipif(
    not _REAL_EXECUTOR_OPT_IN,
    reason="set TRINOTE_RUN_BONSAI35_REAL_EXECUTOR=1 for the external 4 GiB artifact",
)
@pytest.mark.parametrize("kind", ["recurrent", "attention"])
def test_real_release_executor_recovers_post_mutation_failure(kind):
    _artifact, executor = _real_executor_artifact()
    first = executor.prefill_argmax([12675])
    before = (executor.position(), executor.cache_fingerprints())
    executor.debug_fail_after_mutation(kind)
    with pytest.raises(RuntimeError, match="cache restored to committed prefix"):
        executor.decode_argmax(first)
    assert (executor.position(), executor.cache_fingerprints()) == before
    # The restored cache remains usable and produces the canonical next ID.
    assert executor.decode_argmax(first) == 353


def test_qwen35_fused_lut32_scale32_dispatch_and_workspace_reuse_if_available():
    import trinote.infer_int.q1_native as qn

    lib = qn._load_lib()
    if lib is None or not hasattr(lib, "bonsai_q1_prepare_apply_multi_i64_lut32_scale32"):
        pytest.skip("fused Q1 prepare/apply-many LUT32+scale32 kernel unavailable")
    artifact = random_bonsai35_artifact(seed=121)
    layer = artifact["layers"][0]
    names = ("wqkv", "wz", "walpha", "wbeta")
    weights = tuple(
        (layer[f"{name}_bits"], layer[f"{name}_scale_fp"])
        for name in names
    )
    group = qn.q1_weight_group(weights)
    assert group.scale32 is True
    rng = np.random.default_rng(121)
    width = group.n_blocks * 128
    x = rng.integers(-(1 << 14), 1 << 14, size=(1, width), dtype=np.int64)
    qn.q1_native_stats(reset=True)
    got1 = qn.q1_prepare_apply_many_native(x, group, 16, prefer_lut32=True)
    ptrs1 = tuple(y.ctypes.data for y in got1)
    got2 = qn.q1_prepare_apply_many_native(x, group, 16, prefer_lut32=True)
    ptrs2 = tuple(y.ctypes.data for y in got2)
    stats = qn.q1_native_stats()
    assert ptrs1 == ptrs2
    assert stats["lut32_hits"] == 2 and stats["u64_calls"] == 0
    for i, name in enumerate(names):
        from trinote.infer_int.reference_bonsai import q1_linear_ref
        expected = q1_linear_ref(
            x, layer[f"{name}_bits"], layer[f"{name}_scale_fp"], 16
        )
        assert np.array_equal(got2[i], expected)


@pytest.mark.parametrize("scale_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("prefer_lut32", [False, True])
def test_qwen35_fused_prepare_apply_all_storage_combinations_if_available(
    scale_dtype, prefer_lut32
):
    import trinote.infer_int.q1_native as qn
    from trinote.infer_int.reference_bonsai import q1_linear_ref

    lib = qn._load_lib()
    needed = "bonsai_q1_prepare_apply_multi_i64"
    if prefer_lut32:
        needed += "_lut32"
    if scale_dtype == np.int32:
        needed += "_scale32"
    if lib is None or not hasattr(lib, needed):
        pytest.skip(f"fused Q1 symbol {needed} unavailable")
    artifact = random_bonsai35_artifact(seed=124)
    layer = artifact["layers"][0]
    bits = layer["w2_bits"]
    scale = np.ascontiguousarray(layer["w2_scale_fp"], dtype=scale_dtype)
    group = qn.q1_weight_group(((bits, scale),))
    rng = np.random.default_rng(124)
    x = rng.integers(
        -(1 << 15), 1 << 15, size=(3, group.n_blocks * 128), dtype=np.int64
    )
    got = qn.q1_prepare_apply_many_native(
        x, group, 16, prefer_lut32=prefer_lut32
    )
    expected = q1_linear_ref(x, bits, scale, 16)
    assert got is not None and np.array_equal(got[0], expected)


def test_qwen35_fused_lut32_guard_retries_uint64_exactly_if_available():
    import trinote.infer_int.q1_native as qn
    from trinote.infer_int.reference_bonsai import q1_linear_ref

    lib = qn._load_lib()
    if lib is None or not hasattr(lib, "bonsai_q1_prepare_apply_multi_i64_lut32_scale32"):
        pytest.skip("fused Q1 prepare/apply-many LUT32+scale32 kernel unavailable")
    artifact = random_bonsai35_artifact(seed=122)
    layer = artifact["layers"][0]
    weights = ((layer["w2_bits"], layer["w2_scale_fp"]),)
    group = qn.q1_weight_group(weights)
    x = np.zeros((1, group.n_blocks * 128), dtype=np.int64)
    x[0, 0] = np.int64(1 << 40)
    qn.q1_native_stats(reset=True)
    got = qn.q1_prepare_apply_many_native(x, group, 16, prefer_lut32=True)
    stats = qn.q1_native_stats()
    expected = q1_linear_ref(x, layer["w2_bits"], layer["w2_scale_fp"], 16)
    assert got is not None and np.array_equal(got[0], expected)
    assert stats["lut32_fallbacks"] == 1
    assert stats["u64_calls"] == 1


def test_qwen35_native_gdn_decode_matches_numpy_state_and_output_if_available():
    import trinote.infer_int.q1_native as qn

    lib = qn._load_lib()
    if lib is None or not hasattr(lib, "bonsai_gdn_decode_i64"):
        pytest.skip("native Qwen3.5 GDN decode kernel unavailable")
    rng = np.random.default_rng(123)
    heads, dim, frac, state_frac = 4, 32, 16, 30
    outer_shift = 2 * frac - state_frac
    inv_sqrt = 5793
    for state_bound in (1 << 22, 1 << 52):
        state = rng.integers(-state_bound, state_bound, size=(heads, dim, dim), dtype=np.int64)
        q = rng.integers(-(1 << 18), 1 << 18, size=(heads, dim), dtype=np.int64)
        k = rng.integers(-(1 << 18), 1 << 18, size=(heads, dim), dtype=np.int64)
        v = rng.integers(-(1 << 20), 1 << 20, size=(heads, dim), dtype=np.int64)
        beta = rng.integers(0, 1 << frac, size=heads, dtype=np.int64)
        decay = rng.integers(0, 1 << frac, size=heads, dtype=np.int64)

        expected_state = state.copy()
        with np.errstate(over="ignore"):
            expected_state[:] = (expected_state * decay[:, None, None]) >> frac
            pred = np.einsum("hij,hi->hj", expected_state, k, optimize=True) >> state_frac
            delta = ((v - pred) * beta[:, None]) >> frac
            expected_state += (k[:, :, None] * delta[:, None, :]) >> outer_shift
            score = np.einsum("hij,hi->hj", expected_state, q, optimize=True) >> frac
            expected_out = (score * inv_sqrt) >> frac

        got_state = state.copy()
        got_out = qn.gdn_decode_native(
            got_state, q, k, v, beta, decay,
            frac, state_frac, outer_shift, inv_sqrt,
        )
        assert got_out is not None and np.array_equal(got_out, expected_out)
        assert np.array_equal(got_state, expected_state)


def test_qwen35_committed_transcendental_luts_are_accurate_and_monotone():
    artifact = random_bonsai35_artifact(seed=0)
    frac = artifact["config"]["frac"]
    fp = 1 << frac
    x = np.asarray([-12.25, -3.125, 0.0, 2.75, 12.0])
    x_fp = np.rint(x * fp).astype(np.int64)
    soft = fixed_point_softplus_lut(x_fp, artifact).astype(np.float64) / fp
    assert np.max(np.abs(soft - np.log1p(np.exp(x)))) < 3 / fp
    assert np.all(np.diff(soft) > 0)

    neg = np.asarray([-30.0, -9.5, -1.25, -0.01, 0.0])
    neg_fp = np.rint(neg * fp).astype(np.int64)
    exp = fixed_point_exp_negative_lut(neg_fp, artifact).astype(np.float64) / fp
    assert np.max(np.abs(exp - np.exp(neg))) < 3 / fp
    assert np.all(np.diff(exp) >= 0)
    assert exp[-1] == 1.0 and math.isclose(soft[2], math.log(2), abs_tol=3 / fp)
