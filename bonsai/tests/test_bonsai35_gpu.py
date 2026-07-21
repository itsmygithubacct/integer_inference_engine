"""Qwen3.5 resident CUDA parity, fallback, and memory gates."""
import numpy as np
import pytest

from trinote.infer_int.gpu_native import gpu_available, gpu_memory_info
from trinote.infer_int.gpu_bonsai35 import (
    Bonsai35GpuExecutor,
    attention_decode_gpu,
    cpu_oracle_trace,
    kv_i32_transaction_gpu,
    kv_i32_roundtrip_gpu,
    prove_bonsai35_gpu_memory,
    qwen35_workspace_components,
    qwen35_embedding_output_identity,
    recurrent_step_gpu,
)
from trinote.infer_int.reference_bonsai35 import (
    _apply_partial_neox_rope,
    _fixed_l2_norm,
    fixed_point_exp_negative_lut,
    fixed_point_softplus_lut,
    random_bonsai35_artifact,
)
from trinote.infer_int.reference_bonsai import _head_rmsnorm, _rmsnorm, fixed_point_silu
from trinote.infer_int.q1_native import attention_decode_native
from trinote.determinism.fixedpoint import fixed_point_sigmoid


def _need_gpu():
    if not gpu_available():
        pytest.skip("Qwen3.5 CUDA library/device unavailable")


def _artifact(seed=0):
    # BMMA output tiles are eight rows; the release value-head count is 48.
    return random_bonsai35_artifact({"ssmTimeStepRank": 8}, seq_len=40, seed=seed)


def test_qwen35_memory_probe_no_gpu_is_clean(monkeypatch):
    import trinote.infer_int.gpu_bonsai35 as g
    monkeypatch.setattr(g, "gpu_available", lambda: False)
    report, handles = g.prove_bonsai35_gpu_memory(_artifact())
    assert not report.available and not report.feasible and handles is None


def test_qwen35_executor_rejects_stale_graph_abi_before_model_upload(monkeypatch):
    import trinote.infer_int.gpu_bonsai35 as g

    class StaleLibrary:
        bonsai35_ctx_create = object()

    monkeypatch.setattr(g, "_load_lib", lambda: StaleLibrary())

    def upload_must_not_run(*args, **kwargs):
        raise AssertionError("stale graph ABI attempted a residency proof")

    monkeypatch.setattr(g, "prove_bonsai35_gpu_memory", upload_must_not_run)
    executor, report = g.Bonsai35GpuExecutor.try_create_reported(_artifact(90))
    assert executor is None
    assert not report.available and not report.feasible
    assert "unavailable or stale" in report.reason


def test_qwen35_memory_probe_rejects_conflict_before_upload(monkeypatch):
    import trinote.infer_int.gpu_bonsai35 as g
    artifact = _artifact(9)
    monkeypatch.setattr(g, "gpu_available", lambda: True)
    monkeypatch.setattr(g, "gpu_memory_info", lambda: {
        "free_bytes": 64 << 20,
        "total_bytes": 8 << 30,
        "used_bytes": (8 << 30) - (64 << 20),
        "resident_weight_bytes": 0,
    })

    def upload_must_not_run(*args, **kwargs):
        raise AssertionError("preflight conflict uploaded a Q1 tensor")

    monkeypatch.setattr(g, "q1_register_weight", upload_must_not_run)
    report, handles = g.prove_bonsai35_gpu_memory(artifact)
    assert report.available and not report.feasible and handles is None
    assert report.reason.startswith("GPU exclusivity conflict:")
    assert report.reason.endswith("no model upload attempted")
    assert report.weight_count == report.static_buffer_count == 0
    assert report.resident_weight_bytes == 0


def test_qwen35_real_shape_memory_components():
    cfg = {
        "architecture": "qwen35", "dModel": 5120, "dFfn": 17408,
        "n_heads": 24, "n_heads_kv": 4, "head_dim": 256,
        "context_len": 4096, "ssmInnerSize": 6144, "ssmStateSize": 128,
        "ssmGroupCount": 16, "ssmTimeStepRank": 48, "ssmConvKernel": 4,
        "vocab": 248320,
    }
    artifact = {"config": cfg, "layers": [
        {"kind": "attention" if (i + 1) % 4 == 0 else "recurrent"}
        for i in range(64)
    ]}
    c = qwen35_workspace_components(artifact, kv_bits=64)
    assert c["recurrent_state_q30"] == 301989888
    assert c["attention_k_cache"] == c["attention_v_cache"] == 536870912
    assert c["attention_guard_maxima"] == 1024
    assert c["token_id_input"] == 8
    assert c["debug_layer_trace"] == 0
    assert sum(c.values()) < 2.0 * (1 << 30)

    traced = qwen35_workspace_components(artifact, kv_bits=64, capture_trace=True)
    assert traced["debug_layer_trace"] == 65 * 5120 * 8
    assert sum(traced.values()) - sum(c.values()) == traced["debug_layer_trace"]

    c32 = qwen35_workspace_components(artifact, kv_bits=32)
    assert c32["attention_k_cache"] == c32["attention_v_cache"] == 268435456
    assert sum(c.values()) - sum(c32.values()) == 512 << 20


def test_qwen35_kv_i32_guard_accepts_endpoints_and_rejects_neighbors():
    _need_gpu()
    safe = np.asarray([-(1 << 31), -1, 0, 1, (1 << 31) - 1], dtype=np.int64)
    actual = kv_i32_roundtrip_gpu(safe)
    assert actual is not None and np.array_equal(actual, safe)
    assert kv_i32_roundtrip_gpu(np.asarray([-(1 << 31) - 1], dtype=np.int64)) is None
    assert kv_i32_roundtrip_gpu(np.asarray([1 << 31], dtype=np.int64)) is None


def test_qwen35_kv_i32_pair_preflight_is_transactional_for_mixed_lanes():
    _need_gpu()
    n = 257  # crosses multiple CUDA blocks; the unsafe lane is not first
    k = np.arange(n, dtype=np.int64) - 100
    v = -k
    initial_k = np.full(n, -123456789, dtype=np.int64)
    initial_v = np.full(n, 987654321, dtype=np.int64)
    safe = kv_i32_transaction_gpu(k, v, initial_k, initial_v)
    assert safe is not None and safe[0]
    assert np.array_equal(safe[1], k) and np.array_equal(safe[2], v)

    k[173] = 1 << 31
    unsafe = kv_i32_transaction_gpu(k, v, initial_k, initial_v)
    assert unsafe is not None and not unsafe[0]
    assert np.array_equal(unsafe[1], initial_k)
    assert np.array_equal(unsafe[2], initial_v)

    k[173] = 0
    v[256] = -(1 << 31) - 1
    unsafe_v = kv_i32_transaction_gpu(k, v, initial_k, initial_v)
    assert unsafe_v is not None and not unsafe_v[0]
    assert np.array_equal(unsafe_v[1], initial_k)
    assert np.array_equal(unsafe_v[2], initial_v)


def test_qwen35_embedding_output_alias_requires_digest_identity():
    art = _artifact(91)
    distinct = qwen35_embedding_output_identity(art)
    assert not distinct["tied"]
    assert distinct["embedding_sha256"] != distinct["output_sha256"]
    tied = dict(art)
    tied["output_bits"] = art["embed_bits"]
    tied["output_scale_fp"] = art["embed_scale_fp"]
    identity = qwen35_embedding_output_identity(tied)
    assert identity["tied"]
    assert identity["embedding_sha256"] == identity["output_sha256"]
    assert identity["embedding_bytes"] == identity["output_bytes"]


def test_qwen35_memory_probe_cleanup_and_insufficient_ceiling():
    _need_gpu()
    artifact = _artifact(1)
    before = gpu_memory_info()["used_bytes"]
    report, handles = prove_bonsai35_gpu_memory(artifact, ceiling_bytes=1)
    assert report.available and not report.feasible and handles is None
    assert report.reason.startswith("GPU exclusivity conflict:")
    assert gpu_memory_info()["used_bytes"] == before == report.post_cleanup_used_bytes


def test_qwen35_recurrent_primitive_matches_cpu_oracle():
    _need_gpu()
    art = _artifact(2); c = art["config"]; f = c["frac"]; sf = c["ssmStateFrac"]
    kh, vh, n = c["ssmGroupCount"], c["ssmTimeStepRank"], c["ssmStateSize"]
    rng = np.random.default_rng(4)
    q = rng.integers(-20000, 20000, (kh, n), dtype=np.int64)
    k = rng.integers(-20000, 20000, (kh, n), dtype=np.int64)
    v = rng.integers(-30000, 30000, (vh, n), dtype=np.int64)
    z = rng.integers(-30000, 30000, (vh, n), dtype=np.int64)
    alpha = rng.integers(-10000, 10000, vh, dtype=np.int64)
    beta = rng.integers(-20000, 20000, vh, dtype=np.int64)
    state = rng.integers(-1000000, 1000000, (vh, n, n), dtype=np.int64)
    layer = art["layers"][0]
    qo, ko = _fixed_l2_norm(q, f), _fixed_l2_norm(k, f)
    mapping = np.arange(vh) % kh; qo, ko = qo[mapping], ko[mapping]
    bo = fixed_point_sigmoid(beta, f)
    soft = fixed_point_softplus_lut(alpha + layer["dt_bias_fp"], art)
    decay = fixed_point_exp_negative_lut((soft * layer["ssm_a_fp"]) >> f, art)
    expected_state = (state * decay[:, None, None]) >> f
    pred = np.einsum("hij,hi->hj", expected_state, ko, optimize=True) >> sf
    delta = ((v - pred) * bo[:, None]) >> f
    expected_state += (ko[:, :, None] * delta[:, None, :]) >> (2 * f - sf)
    out = np.einsum("hij,hi->hj", expected_state, qo, optimize=True) >> f
    out = (out * c["gdnScaleFp"]) >> f
    norm = _rmsnorm(out, f, layer["ssm_norm_gain_fp"], native=False,
                    eps=c["ssmRmsEpsilonFp2"])
    expected = (norm * fixed_point_silu(z, f, native=False)) >> f
    got = recurrent_step_gpu(q, k, v, z, alpha, beta, state,
                             layer["dt_bias_fp"], layer["ssm_a_fp"],
                             layer["ssm_norm_gain_fp"], art)
    assert got is not None
    assert np.array_equal(got[0], expected)
    assert np.array_equal(got[1], expected_state)


@pytest.mark.parametrize("prefix_len", [0, 1, 31])
def test_qwen35_attention_primitive_matches_cpu(prefix_len):
    _need_gpu()
    art = _artifact(3); c = art["config"]; f = c["frac"]
    h, hkv, hd, nr = c["n_heads"], c["n_heads_kv"], c["head_dim"], c["ropeRotDim"]
    layer = next(x for x in art["layers"] if x["kind"] == "attention")
    rng = np.random.default_rng(prefix_len + 8)
    qg = rng.integers(-20000, 20000, (h, 2, hd), dtype=np.int64)
    k = rng.integers(-20000, 20000, (hkv, hd), dtype=np.int64)
    v = rng.integers(-20000, 20000, (hkv, hd), dtype=np.int64)
    K = rng.integers(-20000, 20000, (hkv, prefix_len, hd), dtype=np.int64) if prefix_len else None
    V = rng.integers(-20000, 20000, (hkv, prefix_len, hd), dtype=np.int64) if prefix_len else None
    q = _head_rmsnorm(qg[:, 0, :][:, None, :], f, layer["q_norm_gain_fp"],
                      native=False, eps=c["rmsEpsilonFp2"])[:, 0]
    kn = _head_rmsnorm(k[:, None, :], f, layer["k_norm_gain_fp"],
                       native=False, eps=c["rmsEpsilonFp2"])[:, 0]
    cos, sin = art["cos_fp"][prefix_len], art["sin_fp"][prefix_len]
    q = _apply_partial_neox_rope(q[:, None], cos[None], sin[None], f, nr)[:, 0]
    kn = _apply_partial_neox_rope(kn[:, None], cos[None], sin[None], f, nr)[:, 0]
    KF = kn[:, None] if K is None else np.concatenate((K, kn[:, None]), axis=1)
    VF = v[:, None] if V is None else np.concatenate((V, v[:, None]), axis=1)
    expected = attention_decode_native(q, KF, VF, f, c["attentionScaleFp"])
    expected = (expected * fixed_point_sigmoid(qg[:, 1], f)) >> f
    got = attention_decode_gpu(qg, k, v, K, V, layer["q_norm_gain_fp"],
                               layer["k_norm_gain_fp"], cos, sin, art)
    assert got is not None and np.array_equal(got[0], expected) and np.array_equal(got[1], kn)


def test_qwen35_resident_graph_full_trace_and_poison_reset():
    _need_gpu()
    art = _artifact(5); ids = [2, 3, 4]
    cpu = cpu_oracle_trace(art, ids)
    executor = Bonsai35GpuExecutor.try_create(art, capture_trace=True)
    assert executor is not None
    try:
        logits = executor.prefill(ids)
        gpu = executor.debug_snapshot()
        assert logits is not None and gpu is not None and np.array_equal(logits, cpu["logits"])
        for key in ("trace", "state", "conv", "k", "v"):
            assert np.array_equal(gpu[key], cpu[key]), key
        graph = executor.graph_metadata()
        assert graph["trace_enabled"] is True
        assert graph["trace_copy_nodes_per_launch"] == len(art["layers"]) + 1
        assert graph["graph_nodes"] == sum(
            graph[key] for key in (
                "kernel_nodes", "memcpy_nodes", "memset_nodes", "other_nodes"
            )
        )
        stats = executor.stats()
        assert stats == {
            "graph_launches": len(ids), "position": len(ids),
            "graph_ready": True, "poisoned": False, "input_mode": "token_id",
            "token_input_submissions": len(ids), "embedded_input_submissions": 0,
            "model_input_host_bytes": len(ids) * 8,
        }
        # One context owns one captured graph.  An accidental host-embedding
        # call cannot silently switch the production token-input graph.
        assert executor.decode_embedded(np.zeros(int(art["config"]["dModel"]), dtype=np.int64)) is None
        assert executor.stats()["poisoned"] is False
    finally:
        executor.close()

    # Embedded rows remain a separately captured diagnostic/fault-injection
    # mode.  Poison/reset/replay is tested without allocating a second graph in
    # one context (which would break the 8 GiB memory envelope).
    diagnostic = Bonsai35GpuExecutor.try_create(art)
    assert diagnostic is not None
    try:
        huge = np.full(int(art["config"]["dModel"]), 1 << 40, dtype=np.int64)
        assert diagnostic.decode_embedded(huge) is None
        assert diagnostic.stats()["poisoned"] is True
        assert diagnostic.reset()
        from trinote.infer_int.reference_bonsai import q1_rows_fp
        safe = q1_rows_fp(
            art["embed_bits"], art["embed_scale_fp"],
            np.asarray([2], dtype=np.int64), int(art["config"]["frac"]),
        )[0]
        assert diagnostic.decode_embedded(safe) is not None
        dstats = diagnostic.stats()
        assert dstats["input_mode"] == "embedded_row"
        assert dstats["token_input_submissions"] == 0
        assert dstats["embedded_input_submissions"] == 2
        assert dstats["model_input_host_bytes"] == 2 * int(art["config"]["dModel"]) * 8
    finally:
        diagnostic.close()


def test_qwen35_production_graph_omits_debug_trace_copies():
    _need_gpu()
    art = _artifact(51)

    def run(capture_trace):
        executor = Bonsai35GpuExecutor.try_create(art, capture_trace=capture_trace)
        assert executor is not None
        try:
            with pytest.raises(RuntimeError, match="decode at least one token"):
                executor.graph_metadata()
            logits = []
            for token in (2, 3, 4):
                row = executor.decode_token(token)
                assert row is not None
                logits.append(row.copy())
            graph = executor.graph_metadata()
            if not capture_trace:
                with pytest.raises(RuntimeError, match="capture_trace=True"):
                    executor.debug_snapshot()
                snapshot = executor.state_snapshot()
            else:
                snapshot = executor.debug_snapshot()
            assert snapshot is not None
            return np.stack(logits), snapshot, graph
        finally:
            executor.close()

    production_logits, production_state, production = run(False)
    diagnostic_logits, diagnostic_state, diagnostic = run(True)
    assert np.array_equal(production_logits, diagnostic_logits)
    for key in ("state", "conv", "k", "v"):
        assert np.array_equal(production_state[key], diagnostic_state[key]), key
    expected_copies = len(art["layers"]) + 1
    assert production["trace_enabled"] is False
    assert production["trace_copy_nodes_per_launch"] == 0
    assert production["trace_copy_bytes_per_launch"] == 0
    assert diagnostic["graph_nodes"] - production["graph_nodes"] == expected_copies
    assert diagnostic["memcpy_nodes"] - production["memcpy_nodes"] == expected_copies
    assert diagnostic["kernel_nodes"] == production["kernel_nodes"]
    assert diagnostic["memset_nodes"] == production["memset_nodes"]


def test_qwen35_tied_embedding_output_reuses_one_resident_handle():
    _need_gpu()
    art = _artifact(92)
    art["output_bits"] = art["embed_bits"]
    art["output_scale_fp"] = art["embed_scale_fp"]
    executor, report = Bonsai35GpuExecutor.try_create_reported(art)
    assert executor is not None
    try:
        assert report.embedding_output_tied
        assert report.aliased_weight_count == 1
        assert report.logical_weight_bytes > report.expected_weight_bytes
        assert executor.handles["weights"]["embed"] == executor.handles["weights"]["output"]
        assert executor.decode_token(2) is not None
        assert executor.stats()["model_input_host_bytes"] == 8
    finally:
        executor.close()
