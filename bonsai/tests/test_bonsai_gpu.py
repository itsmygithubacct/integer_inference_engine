"""Parity gate for the per-host opt-in GPU Q1 kernel.

Skips when ``libbonsai_q1_gpu.so`` / a usable GPU is absent, so CI on CPU-only hosts stays green and the
committed build is never blocked by the per-host GPU artifact. When the kernel exists, every test asserts
``np.array_equal(gpu, cpu_oracle)`` — the same byte-exact discipline as the native-kernel tests. This is the
gate that must pass before ``--gpu`` is permitted to emit a receipt (see
``research/bonsai-notary/IMPLEMENT-GPU-MODE.md`` §4).
"""
import numpy as np
import pytest

from trinote.infer_int.gpu_native import gpu_available, q1_apply_gpu, rmsnorm_gpu
from trinote.infer_int.reference_bonsai import q1_linear_ref
from trinote.determinism.fixedpoint import fixed_point_rmsnorm

import contextlib
import os


@contextlib.contextmanager
def _env(key, val):
    old = os.environ.get(key)
    os.environ[key] = val
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old

_FRAC = 16


def _need_gpu():
    if not gpu_available():
        pytest.skip("GPU Q1 kernel (tools/libbonsai_q1_gpu.so) / usable GPU not available")


def _need_rmsnorm():
    _need_gpu()
    from trinote.infer_int.gpu_native import _load_lib
    if not hasattr(_load_lib(), "bonsai_rmsnorm_gpu"):
        pytest.skip("GPU .so has no RMSNorm kernel (rebuild via tools/build_bonsai_q1_gpu.sh)")


def test_gpu_module_importable_and_inert_without_lib():
    """Without the .so the module is inert: gpu_available() is False and q1_apply_gpu returns None (-> CPU)."""
    if gpu_available():
        pytest.skip("GPU lib present; this test covers the absent-lib default")
    x = np.zeros((1, 256), dtype=np.int64)
    bits = np.zeros((4, 2, 16), dtype=np.uint8)
    scale = np.ones((4, 2), dtype=np.int64)
    assert q1_apply_gpu(x, bits, scale, _FRAC) is None


def test_gpu_q1_matches_cpu_oracle_shapes():
    """G1/G2: byte-identical to the int64 oracle across shapes + activation magnitudes (incl. K=12288)."""
    _need_gpu()
    rng = np.random.default_rng(7)
    # (tokens, out_features, n_blocks) — last covers the w2 down-proj contraction K = 96*128 = 12288.
    for (tokens, out_f, n_blocks) in [(1, 64, 2), (4, 128, 4), (8, 256, 8), (2, 64, 96)]:
        K = n_blocks * 128
        for hi in (1 << 12, 1 << 20):                  # test envelope + a saturating case
            x = rng.integers(-hi, hi, (tokens, K), dtype=np.int64)
            bits = rng.integers(0, 256, (out_f, n_blocks, 16), dtype=np.uint8)
            scale = rng.integers(-(1 << 11), 1 << 11, (out_f, n_blocks), dtype=np.int64)
            gpu = q1_apply_gpu(x, bits, scale, _FRAC)
            oracle = q1_linear_ref(x, bits, scale, _FRAC)
            assert gpu is not None, (tokens, out_f, n_blocks, hi)
            assert np.array_equal(gpu, oracle), (tokens, out_f, n_blocks, hi)


def test_gpu_q1_matches_cpu_oracle_at_wrap_boundary():
    """G3: the int64 wrap boundary — proves the GPU used NO wider-than-64-bit accumulator (doc §2.4)."""
    _need_gpu()
    # Mirror the CPU overflow-boundary case: a single large activation, frac=1, mixed-sign per-block scales.
    x = np.zeros((1, 256), dtype=np.int64)
    x[0, 0] = 1 << 62
    bits = np.zeros((3, 2, 16), dtype=np.uint8)
    bits[:, :, 0] = 1                                   # weight bit 0 set -> +1 on x[0,0]
    scale = np.array([[3, 3], [-3, -3], [4, 4]], dtype=np.int64)
    gpu = q1_apply_gpu(x, bits, scale, 1)
    oracle = q1_linear_ref(x, bits, scale, 1)
    # The GPU may legitimately decline (rc!=0 -> None) and let the CPU produce; but if it produces, it must match.
    if gpu is not None:
        assert np.array_equal(gpu, oracle)


def test_gpu_geometry_invariance():
    """G5: identical bytes across batch sizes (stands in for launch-geometry invariance; see doc I5)."""
    _need_gpu()
    rng = np.random.default_rng(11)
    out_f, n_blocks = 128, 4
    K = n_blocks * 128
    bits = rng.integers(0, 256, (out_f, n_blocks, 16), dtype=np.uint8)
    scale = rng.integers(-(1 << 10), 1 << 10, (out_f, n_blocks), dtype=np.int64)
    base = rng.integers(-(1 << 12), 1 << 12, (16, K), dtype=np.int64)
    full = q1_apply_gpu(base, bits, scale, _FRAC)
    assert full is not None
    for m in (1, 4, 8):                                 # each sub-batch must match the corresponding rows
        sub = q1_apply_gpu(base[:m], bits, scale, _FRAC)
        assert sub is not None and np.array_equal(sub, full[:m]), m


def test_gpu_resident_apply_matches_oracle():
    """Weight-residency path: register a weight once, apply to fresh activations — byte-identical to the oracle
    and to the per-call upload path. Skips if the .so predates the residency API."""
    _need_gpu()
    from trinote.infer_int.gpu_native import q1_register_weight, q1_apply_resident, residency_available
    if not residency_available():
        pytest.skip("GPU .so has no residency API (rebuild via tools/build_bonsai_q1_gpu.sh)")
    rng = np.random.default_rng(3)
    for (tokens, out_f, n_blocks) in [(1, 64, 2), (4, 128, 4), (3, 96, 96)]:   # incl. K=12288
        K = n_blocks * 128
        bits = rng.integers(0, 256, (out_f, n_blocks, 16), dtype=np.uint8)
        scale = rng.integers(-(1 << 10), 1 << 10, (out_f, n_blocks), dtype=np.int64)
        h = q1_register_weight(bits, scale)
        assert h is not None, (out_f, n_blocks)
        for _ in range(3):                                  # reuse the resident weight across several applies
            x = rng.integers(-(1 << 12), 1 << 12, (tokens, K), dtype=np.int64)
            res = q1_apply_resident(h, x, out_f, n_blocks, _FRAC)
            assert res is not None and np.array_equal(res, q1_linear_ref(x, bits, scale, _FRAC))


def test_gpu_forward_end_to_end_matches_oracle(monkeypatch):
    """End-to-end: with TRINOTE_GPU=1, forward_fast routes EVERY Q1 apply (all per-layer projections + the
    output head) through the GPU resident path and stays byte-identical to the pure-NumPy oracle. Also asserts
    the GPU path was actually exercised — guarding against a silent CPU fallback masking a broken kernel."""
    _need_gpu()
    from test_bonsai_smoke import _small_bonsai
    import trinote.infer_int.reference_bonsai as rb
    from trinote.infer_int.reference_bonsai import BonsaiReferenceModel

    art = _small_bonsai(seed=1)
    ids = list(range(1, 13))
    oracle = BonsaiReferenceModel(art).forward(ids)        # pure-NumPy reference

    calls = {"n": 0}
    _orig = rb.q1_apply_resident                            # the model uses the resident path under --gpu

    def _counting(*a, **k):
        calls["n"] += 1
        return _orig(*a, **k)

    monkeypatch.setattr(rb, "q1_apply_resident", _counting)
    monkeypatch.setenv("TRINOTE_GPU", "1")
    ref = BonsaiReferenceModel(art)
    assert ref.enable_native(), "native CPU kernel required (the GPU path keeps a CPU fallback)"
    gpu = ref.forward_fast(ids)

    n_layers = len(art["layers"])
    assert calls["n"] == n_layers * 7 + 1, calls["n"]      # 7 projections/layer + 1 output head; GPU path ran
    assert np.array_equal(gpu, oracle)


def test_gpu_rmsnorm_matches_oracle():
    """M2: GPU RMSNorm byte-identical to fixed_point_rmsnorm across decode/prefill/q-norm shapes and the
    committed activation envelope, with and without an integer gain."""
    _need_rmsnorm()
    rng = np.random.default_rng(0)
    cases = [
        (1, 4096, 1 << 14, False, "decode no-gain"),
        (1, 4096, 1 << 14, True, "decode +gain"),
        (8, 4096, 1 << 24, True, "committed-envelope +gain"),
        (512, 4096, 1 << 20, True, "prefill M=512 +gain"),
        (4, 128, 1 << 25, True, "head_dim q/k-norm +gain"),
    ]
    for rows, cols, mag, use_gain, label in cases:
        x = rng.integers(-mag, mag, (rows, cols), dtype=np.int64)
        g = rng.integers(-(1 << 20), 1 << 20, (cols,), dtype=np.int64) if use_gain else None
        gpu = rmsnorm_gpu(x, _FRAC, 1, g)
        ora = fixed_point_rmsnorm(x, _FRAC, 1, g)
        assert gpu is not None and np.array_equal(gpu, ora), label


def test_gpu_rmsnorm_rc4_fallback_lockstep():
    """The GPU must return None (→ CPU fallback) EXACTLY when the CPU needs >128-bit big-ints or the gain
    leaves the int64 envelope — never silently wrap. Lockstep with the oracle's compute/raise behaviour."""
    _need_rmsnorm()
    # (1) sum-of-squares overflows 128 bits → GPU None; the oracle computes it with big-ints.
    big = np.full((1, 4096), (1 << 62), dtype=np.int64)
    assert rmsnorm_gpu(big, _FRAC, 1, None) is None
    assert fixed_point_rmsnorm(big, _FRAC, 1, None).shape == (1, 4096)
    # (2) pathological gain leaves the envelope → GPU None; the oracle RAISES (lockstep refuse).
    x = np.random.default_rng(1).integers(-(1 << 14), 1 << 14, (1, 4096), dtype=np.int64)
    huge = np.full((4096,), (1 << 60), dtype=np.int64)
    assert rmsnorm_gpu(x, _FRAC, 1, huge) is None
    with pytest.raises(OverflowError):
        fixed_point_rmsnorm(x, _FRAC, 1, huge)


def test_gpu_prefill_attention_matches_cpu_native():
    """M3: GPU causal M=N prefill attention byte-identical to the CPU native kernel (itself == NumPy oracle)
    across MHA / GQA / continued-prefill (start>0) / varied widths."""
    _need_gpu()
    import math
    from trinote.infer_int.gpu_native import _load_lib, attention_prefill_gpu
    from trinote.infer_int.q1_native import attention_prefill_native, q1_native_available
    if not hasattr(_load_lib(), "bonsai_attention_prefill_gpu"):
        pytest.skip("GPU .so has no prefill-attention kernel (rebuild via tools/build_bonsai_q1_gpu.sh)")
    if not q1_native_available():
        pytest.skip("CPU native kernel needed as the parity reference")
    rng = np.random.default_rng(0)
    cases = [
        (4, 4, 16, 8, 0, 1 << 8, "MHA start=0"),
        (32, 8, 128, 16, 0, 1 << 6, "GQA 8B-shape start=0"),
        (32, 8, 128, 12, 20, 1 << 6, "GQA start=20 (continued prefill)"),
        (8, 2, 64, 32, 0, 1 << 7, "GQA wide M=32"),
        (2, 1, 8, 4, 0, 1 << 10, "tiny MHA"),
    ]
    for H, Hkv, hd, M, start, mag, label in cases:
        L = start + M
        inv_sqrt_fp = round((1.0 / math.sqrt(hd)) * (1 << _FRAC))
        q = rng.integers(-mag, mag, (H, M, hd), dtype=np.int64)
        k = rng.integers(-mag, mag, (Hkv, L, hd), dtype=np.int64)
        v = rng.integers(-mag, mag, (Hkv, L, hd), dtype=np.int64)
        g = attention_prefill_gpu(q, k, v, start, _FRAC, inv_sqrt_fp)
        c = attention_prefill_native(q, k, v, start, _FRAC, inv_sqrt_fp)
        assert g is not None and c is not None and np.array_equal(g, c), label


def test_gpu_prefill_forward_end_to_end_on_device(monkeypatch):
    """M3: a full prefill forward with TRINOTE_GPU=1 + TRINOTE_GPU_FULL=1 routes Q1 apply + RMSNorm +
    prefill-attention through GPU kernels and stays byte-identical to the NumPy oracle. Asserts all three GPU
    entry points were exercised. (TRINOTE_GPU_FULL is off by default — the per-op rmsnorm/attention dispatch is
    a perf regression vs applies-only until x is device-resident; this test gates byte-exactness of the path.)"""
    _need_gpu()
    from trinote.infer_int.gpu_native import _load_lib
    if not hasattr(_load_lib(), "bonsai_attention_prefill_gpu"):
        pytest.skip("GPU .so missing prefill-attention kernel")
    from test_bonsai_smoke import _small_bonsai
    import trinote.infer_int.reference_bonsai as rb
    from trinote.infer_int.reference_bonsai import BonsaiReferenceModel

    art = _small_bonsai(seed=1)
    ids = list(range(1, 17))                                # T=16 prefill (>1 → prefill attention path)
    oracle = BonsaiReferenceModel(art).forward(ids, last_only=True)

    cnt = {"q1": 0, "rms": 0, "attn": 0}
    oq, orms, oattn = rb.q1_apply_resident, rb.rmsnorm_gpu, rb.attention_prefill_gpu
    monkeypatch.setattr(rb, "q1_apply_resident", lambda *a, **k: (cnt.__setitem__("q1", cnt["q1"] + 1), oq(*a, **k))[1])
    monkeypatch.setattr(rb, "rmsnorm_gpu", lambda *a, **k: (cnt.__setitem__("rms", cnt["rms"] + 1), orms(*a, **k))[1])
    monkeypatch.setattr(rb, "attention_prefill_gpu", lambda *a, **k: (cnt.__setitem__("attn", cnt["attn"] + 1), oattn(*a, **k))[1])
    monkeypatch.setenv("TRINOTE_GPU", "1")
    monkeypatch.setenv("TRINOTE_GPU_FULL", "1")         # also route RMSNorm + prefill-attention to GPU
    ref = BonsaiReferenceModel(art)
    assert ref.enable_native()
    gpu = ref.forward_fast(ids, last_only=True)

    nl = len(art["layers"])
    assert cnt["q1"] == nl * 7 + 1, cnt           # 7 projections/layer + output head
    assert cnt["attn"] == nl, cnt                  # one prefill-attention call per layer
    assert cnt["rms"] >= nl * 2 + 1, cnt           # at least n1+n2 per layer + final (q/k-norm add more)
    assert np.array_equal(gpu, oracle)


def test_gpu_resident_monolith_prefill_matches_oracle():
    """M3 TRUE-RESIDENCY: the on-device prefill monolith (residual never leaves the GPU) returns last-position
    logits byte-identical to forward(..., last_only=True) across T and shapes. The single biggest correctness
    gate for the resident path (transposes, RoPE, SiLU, residual adds, attention, all on-device)."""
    _need_gpu()
    from trinote.infer_int.gpu_native import monolith_available
    if not monolith_available():
        pytest.skip("GPU .so has no prefill monolith (rebuild via tools/build_bonsai_q1_gpu.sh)")
    from test_bonsai_smoke import _small_bonsai
    from trinote.infer_int.reference_bonsai import BonsaiReferenceModel
    for seed, T in [(1, 16), (2, 4), (3, 32), (4, 1), (5, 8)]:
        art = _small_bonsai(seed=seed)
        ids = list(range(1, 1 + T))
        oracle = BonsaiReferenceModel(art).forward(ids, last_only=True)
        ref = BonsaiReferenceModel(art)
        assert ref.enable_native()
        gpu = ref.prefill_logits_gpu_resident(ids)
        assert gpu is not None, (seed, T)
        assert np.array_equal(gpu, oracle), (seed, T)


def test_gpu_dp4a_apply_matches_oracle():
    """DP4A Q1 apply (int8 dp4a hot loop, L=4/8 base-256 limbs) byte-identical to q1_linear_ref — both the
    thread-per-output (L>0) and warp-per-output (L<0) variants, across shapes incl. K=12288 and the L=8 path."""
    _need_gpu()
    from trinote.infer_int.gpu_native import _load_lib, q1_apply_dp4a_gpu
    if not hasattr(_load_lib(), "bonsai_q1_linear_dp4a_gpu"):
        pytest.skip("GPU .so has no DP4A kernel (rebuild via tools/build_bonsai_q1_gpu.sh)")
    rng = np.random.default_rng(0)
    for (tok, out_f, nb, mag) in [(1, 64, 2, 1 << 14), (256, 256, 32, 1 << 24), (8, 128, 96, 1 << 20)]:
        K = nb * 128
        x = rng.integers(-mag, mag, (tok, K), dtype=np.int64)
        bits = rng.integers(0, 256, (out_f, nb, 16), dtype=np.uint8)
        scale = rng.integers(-(1 << 11), 1 << 11, (out_f, nb), dtype=np.int64)
        oracle = q1_linear_ref(x, bits, scale, _FRAC)
        for L in (4, -4, 8):                                  # thread L=4, warp L=4, thread L=8
            g = q1_apply_dp4a_gpu(x, bits, scale, _FRAC, L=L)
            assert g is not None and np.array_equal(g, oracle), (tok, out_f, nb, L)
    # large |x| outside the L=4 balanced range must still be byte-exact via auto L=8
    x_big = rng.integers(-(1 << 40), 1 << 40, (2, 256), dtype=np.int64)
    bits = rng.integers(0, 256, (64, 2, 16), dtype=np.uint8)
    scale = rng.integers(-(1 << 11), 1 << 11, (64, 2), dtype=np.int64)
    g = q1_apply_dp4a_gpu(x_big, bits, scale, _FRAC)          # auto-picks L=8
    assert g is not None and np.array_equal(g, q1_linear_ref(x_big, bits, scale, _FRAC))


def test_gpu_kv_export_seeds_cache_byte_exact():
    """KV-export: the monolith's per-layer K/V (post-RoPE K, raw V) exported to seed the decode cache are
    bit-for-bit equal to what CPU prefill caches (per layer + cache.t)."""
    _need_gpu()
    from trinote.infer_int.gpu_native import monolith_available
    if not monolith_available():
        pytest.skip("GPU .so has no prefill monolith")
    from test_bonsai_smoke import _small_bonsai
    from trinote.infer_int.reference_bonsai import BonsaiReferenceModel, _BonsaiKVCache
    art = _small_bonsai(seed=3)
    prompt = list(range(2, 18))
    nl = len(art["layers"])
    rc = BonsaiReferenceModel(art); assert rc.enable_native()
    ccpu = _BonsaiKVCache(nl); rc._run_layers(prompt, ccpu)        # CPU prefill cache
    rg = BonsaiReferenceModel(art); assert rg.enable_native()
    with _env("TRINOTE_GPU", "1"):
        out = rg._gpu_prefill(prompt, want_kv=True)
    assert out is not None
    _, cgpu = out
    assert cgpu.t == ccpu.t == len(prompt)
    for li in range(nl):
        assert np.array_equal(ccpu.k[li], cgpu.k[li]), li
        assert np.array_equal(ccpu.v[li], cgpu.v[li]), li


def test_gpu_generate_cached_matches_cpu():
    """End-to-end generative decode: generate_cached with GPU resident-prefill (KV-export seeds the cache)
    produces byte-identical tokens to the CPU-prefill path, for greedy AND seeded min_p."""
    _need_gpu()
    from trinote.infer_int.gpu_native import monolith_available
    if not monolith_available():
        pytest.skip("GPU .so has no prefill monolith")
    from test_bonsai_smoke import _small_bonsai
    from trinote.infer_int.reference_bonsai import BonsaiReferenceModel
    from trinote.infer_int.sampler import resolve_sampler, sample_token
    art = _small_bonsai(seed=1)
    prompt, n = list(range(1, 13)), 20
    frac = int(art["config"]["frac"])
    cfg = resolve_sampler("min_p", seed=0)
    picks = {
        "greedy": lambda row, pos, hist: int(np.asarray(row).argmax()),
        "min_p": lambda row, pos, hist: sample_token(row, cfg, position=pos, frac_bits=frac, history_ids=hist),
    }
    for name, pick in picks.items():
        rc = BonsaiReferenceModel(art); assert rc.enable_native()
        cpu = rc.generate_cached(prompt, n, pick)
        rg = BonsaiReferenceModel(art); assert rg.enable_native()
        with _env("TRINOTE_GPU", "1"):
            gpu = rg.generate_cached(prompt, n, pick)
        assert cpu == gpu, name


def test_gpu_generate_batched_matches_sequential():
    """M=B batch serving on GPU: generate_batched (the M=B decode path; Q1 applies run at M=B on the GPU via
    the residency hook) is byte-identical to running each sequence's generate_cached standalone — greedy AND
    seeded min_p, ragged prompt lengths."""
    _need_gpu()
    from trinote.infer_int.gpu_native import monolith_available
    if not monolith_available():
        pytest.skip("GPU .so unavailable")
    from test_bonsai_smoke import _small_bonsai
    from trinote.infer_int.reference_bonsai import BonsaiReferenceModel
    from trinote.infer_int.sampler import resolve_sampler, sample_token
    art = _small_bonsai(seed=5)
    prompts = [list(range(1, 9)), list(range(2, 14)), list(range(3, 7)), list(range(1, 11))]   # ragged
    n = 16
    frac = int(art["config"]["frac"])
    cfg = resolve_sampler("min_p", seed=0)
    pickers = {
        "greedy": lambda row, pos, hist: int(np.asarray(row).argmax()),
        "min_p": lambda row, pos, hist: sample_token(row, cfg, position=pos, frac_bits=frac, history_ids=hist),
    }
    for name, pick in pickers.items():
        with _env("TRINOTE_GPU", "1"):
            rg = BonsaiReferenceModel(art); assert rg.enable_native()
            batched = rg.generate_batched(prompts, n, [pick] * len(prompts))
            seq = [rg.generate_cached(p, n, pick) for p in prompts]
        assert batched == seq, name


def test_gpu_batched_resident_path_exercised(monkeypatch):
    """The fully-resident M=B batched decode (device KV + on-device RMSNorm/RoPE/attention) is byte-identical to
    sequential when opted in via TRINOTE_GPU_RESIDENT_BATCH (default OFF — it regresses on sm_86, see
    _gpu_resident_batch_enabled). Asserts decode_step IS exercised so a silent fallback can't mask a broken path."""
    _need_gpu()
    from trinote.infer_int.gpu_native import batched_decode_available
    if not batched_decode_available():
        pytest.skip("GPU .so has no batched decode context")
    from test_bonsai_smoke import _small_bonsai
    import trinote.infer_int.reference_bonsai as rb
    from trinote.infer_int.reference_bonsai import BonsaiReferenceModel
    art = _small_bonsai(seed=7)
    prompts = [list(range(1, 9)), list(range(2, 16)), list(range(3, 7)), list(range(1, 12))]
    n = 14
    def pick(row, pos, hist):
        return int(np.asarray(row).argmax())
    rc = BonsaiReferenceModel(art); assert rc.enable_native()
    cpu = [rc.generate_cached(p, n, pick) for p in prompts]          # sequential reference
    calls = {"n": 0}
    _orig = rb.decode_step
    monkeypatch.setattr(rb, "decode_step", lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), _orig(*a, **k))[1])
    monkeypatch.setenv("TRINOTE_GPU", "1")
    monkeypatch.setenv("TRINOTE_GPU_RESIDENT_BATCH", "1")           # opt into the resident path (default OFF: it regresses)
    rg = BonsaiReferenceModel(art); assert rg.enable_native()
    gpu = rg.generate_batched(prompts, n, [pick] * len(prompts))
    assert calls["n"] == n - 1, calls["n"]                          # decode_step runs for steps 1..n-1
    assert gpu == cpu
