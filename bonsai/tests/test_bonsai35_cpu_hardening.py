from __future__ import annotations

import ctypes
import gc
import platform
import shutil
import subprocess
import threading
import time
import weakref
from pathlib import Path

import numpy as np
import pytest

import trinote.infer_int.reference_bonsai35 as rb35
from trinote.determinism.fixedpoint import fixed_point_rmsnorm
from trinote.infer_int.artifact_io_bonsai import load_artifact_bonsai
from trinote.infer_int.q1_native import (
    Bonsai35NativeExecutor,
    _B35ExecStats,
    _B35LayerDesc,
    _B35ModelDesc,
    _B35Q1Desc,
    _load_lib,
    _validate_b35_abi,
    _validate_b35_release_artifact,
    _validate_b35_token_ids,
    q1_set_isa,
    rmsnorm_native,
)
from trinote.infer_int.reference_bonsai35 import (
    BonsaiQwen35ReferenceModel,
    _Qwen35Cache,
    random_bonsai35_artifact,
)
from trinote.infer_int.trace_bonsai35 import trace_prefill


_REAL_OPT_IN = __import__("os").environ.get(
    "TRINOTE_RUN_BONSAI35_REAL_EXECUTOR", ""
) == "1"


def _real_artifact():
    path = Path.home() / ".local/trinote/models/Bonsai-27B-Q1_0-int-qwen35.safetensors"
    assert path.is_file(), path
    artifact, _info = load_artifact_bonsai(path)
    return artifact


def _release_config(*, vocab: int = 1, context: int = 1) -> dict:
    return {
        "architecture": "qwen35",
        "nLayers": 64,
        "frac": 16,
        "dModel": 5120,
        "dFfn": 17408,
        "n_heads": 24,
        "n_heads_kv": 4,
        "head_dim": 256,
        "ropeRotDim": 64,
        "ssmStateSize": 128,
        "ssmGroupCount": 16,
        "ssmInnerSize": 6144,
        "ssmTimeStepRank": 48,
        "ssmConvKernel": 4,
        "ssmStateFrac": 30,
        "fullAttentionInterval": 4,
        "context_len": context,
        "vocab": vocab,
        "rmsEpsilonFp2": 1,
        "ssmRmsEpsilonFp2": 1,
        "attentionScaleFp": 4096,
        "gdnScaleFp": 5793,
        "lutStepFp": 1,
        "softplusLutMinFp": -1,
        "softplusLutMaxFp": 1,
        "expNegLutMinFp": -1,
        "expNegLutMaxFp": 1,
    }


def _assert_same_cache_refs(before: tuple, after: tuple) -> None:
    for old_list, new_list in zip(before[:7], after[:7]):
        if old_list and isinstance(old_list[0], (int, np.integer)):
            assert old_list == new_list
        else:
            assert len(old_list) == len(new_list)
            assert all(old is new for old, new in zip(old_list, new_list))
    assert before[7] == after[7]


def test_release_admission_rejects_layer_count_before_descriptor_construction():
    artifact = {"config": _release_config(), "layers": []}
    with pytest.raises(ValueError, match="layer count 0 != nLayers 64"):
        _validate_b35_release_artifact(artifact)


def test_release_admission_rejects_malformed_global_extent():
    layers = [
        {"kind": "attention" if (index + 1) % 4 == 0 else "recurrent"}
        for index in range(64)
    ]
    artifact = {
        "config": _release_config(),
        "layers": layers,
        "embed_bits": np.empty((1, 39, 16), dtype=np.uint8),
        "embed_scale_fp": np.empty((1, 40), dtype=np.int32),
    }
    with pytest.raises(ValueError, match="embed_bits.*expected"):
        _validate_b35_release_artifact(artifact)


def test_abi_version_and_all_ctypes_sizes_must_match():
    expected = (
        ctypes.sizeof(_B35ModelDesc), ctypes.sizeof(_B35LayerDesc),
        ctypes.sizeof(_B35Q1Desc), ctypes.sizeof(_B35ExecStats),
    )

    class Good:
        def bonsai35_model_abi_version(self):
            return 2

        def bonsai35_model_abi_sizeof(self, kind):
            return expected[kind]

    _validate_b35_abi(Good())

    class Bad(Good):
        def bonsai35_model_abi_sizeof(self, kind):
            return expected[kind] + (1 if kind == 1 else 0)

    with pytest.raises(RuntimeError, match="ABI mismatch"):
        _validate_b35_abi(Bad())


@pytest.mark.parametrize("bad", [-1, 9, 1 << 80])
def test_strict_token_ids_reject_out_of_range_and_non_int64(bad):
    with pytest.raises(ValueError):
        _validate_b35_token_ids([bad], 9, where="test")


@pytest.mark.parametrize("bad", [1.0, True, np.int64(1), "1"])
def test_strict_token_ids_require_python_int(bad):
    with pytest.raises(TypeError):
        _validate_b35_token_ids([bad], 9, where="test")


@pytest.mark.parametrize(
    ("method", "args", "error"),
    [
        ("prefill", ([-1],), ValueError),
        ("prefill", ([1 << 80],), ValueError),
        ("prefill", ([np.int64(1)],), TypeError),
        ("decode", (-1,), ValueError),
        ("decode", (1 << 80,), ValueError),
        ("decode", (np.int64(1),), TypeError),
    ],
)
def test_resident_executor_validates_ids_before_entering_native_code(method, args, error):
    class NativeMustNotRun:
        def __getattr__(self, name):
            raise AssertionError(f"invalid token reached native method {name}")

    executor = object.__new__(Bonsai35NativeExecutor)
    executor._lock = threading.RLock()
    executor._handle = ctypes.c_void_p(1)
    executor._lib = NativeMustNotRun()
    executor._history = []
    executor.vocab = 9
    executor.d_model = 1
    with pytest.raises(error):
        getattr(executor, method)(*args)


def test_reference_and_trace_apply_strict_token_validation():
    artifact = random_bonsai35_artifact(seed=903)
    model = BonsaiQwen35ReferenceModel(artifact)
    for bad, error in (([-1], ValueError), ([1 << 80], ValueError), ([1.0], TypeError)):
        with pytest.raises(error):
            model.forward(bad)
        with pytest.raises(error):
            trace_prefill(artifact, bad)


@pytest.mark.parametrize("failure", ["later_ffn", "attention_projection"])
def test_python_cached_graph_rolls_back_every_layer_failure(monkeypatch, failure):
    artifact = random_bonsai35_artifact(seed=907)
    model = BonsaiQwen35ReferenceModel(artifact)
    cache = _Qwen35Cache(len(artifact["layers"]))
    model._run_layers([1, 2], cache)
    before = cache.checkpoint()

    if failure == "later_ffn":
        original = rb35._ffn

        def fail_ffn(x, layer, frac, **kwargs):
            if layer is artifact["layers"][2]:
                raise RuntimeError("injected later FFN failure")
            return original(x, layer, frac, **kwargs)

        monkeypatch.setattr(rb35, "_ffn", fail_ffn)
    else:
        original = rb35._project

        def fail_projection(x, owner, name, frac, **kwargs):
            if owner is artifact["layers"][3] and name == "wo":
                raise RuntimeError("injected attention projection failure")
            return original(x, owner, name, frac, **kwargs)

        monkeypatch.setattr(rb35, "_project", fail_projection)

    with pytest.raises(RuntimeError, match="injected"):
        model._run_layers([3], cache)
    _assert_same_cache_refs(before, cache.checkpoint())


def test_executor_public_abi_is_locked_and_close_cannot_race_an_export():
    locked = (
        "reset", "prefill", "decode", "prefill_logits", "decode_logits",
        "prefill_argmax", "decode_argmax", "force_lut_fallback",
        "debug_fail_after_mutation", "debug_trace_layer", "export_debug_trace",
        "position", "cache_fingerprints", "export_cache_tensor",
        "export_last_residual", "stats",
    )
    assert all(hasattr(getattr(Bonsai35NativeExecutor, name), "__wrapped__") for name in locked)

    entered = threading.Event()
    release = threading.Event()
    freed = threading.Event()

    class FakeLib:
        def bonsai35_model_position(self, _handle):
            entered.set()
            assert release.wait(2)
            return 7

        def bonsai35_model_free(self, _handle):
            freed.set()

    executor = object.__new__(Bonsai35NativeExecutor)
    executor._lock = threading.RLock()
    executor._handle = ctypes.c_void_p(1)
    executor._lib = FakeLib()
    result: list[int] = []
    reader = threading.Thread(target=lambda: result.append(executor.position()))
    reader.start()
    assert entered.wait(1)
    closer = threading.Thread(target=executor.close)
    closer.start()
    time.sleep(0.05)
    assert not freed.is_set()
    release.set()
    reader.join(2)
    closer.join(2)
    assert result == [7] and freed.is_set()
    with pytest.raises(RuntimeError, match="closed"):
        executor.position()


def test_native_rms_refuses_oracle_gain_overflow_and_handles_negative_rows():
    if _load_lib() is None:
        pytest.skip("native kernel unavailable")
    frac = 16
    x = np.array([[-(1 << 40), 1 << 39, -(1 << 38), 7]], dtype=np.int64)
    gain = np.array([1 << 16, 3 << 15, 1 << 14, 5 << 14], dtype=np.int64)
    assert np.array_equal(
        rmsnorm_native(x, frac, gain_q=gain),
        fixed_point_rmsnorm(x, frac, gain_q=gain),
    )
    dangerous = np.full(x.shape[1], np.iinfo(np.int64).max, dtype=np.int64)
    with pytest.raises(OverflowError):
        fixed_point_rmsnorm(x, frac, gain_q=dangerous)
    assert rmsnorm_native(x, frac, gain_q=dangerous) is None


def test_kernel_links_without_openmp_and_exports_abi(tmp_path):
    cc = shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler")
    source = Path(__file__).resolve().parents[1] / "tools/bonsai_q1_kernel.c"
    output = tmp_path / "libbonsai-q1-noomp.so"
    subprocess.run(
        [cc, "-std=c11", "-fPIC", "-shared", "-Werror=implicit-function-declaration",
         str(source), "-o", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    lib = ctypes.CDLL(str(output))
    lib.bonsai35_model_abi_version.restype = ctypes.c_uint64
    assert int(lib.bonsai35_model_abi_version()) == 2


def test_direct32_resident_kernel_boundary_selftest_fresh_build(tmp_path):
    """Gate decode/prefill direct32 offsets, Q16, replay, and adversarial edges."""
    if platform.machine().lower() not in {"x86_64", "amd64", "i386", "i686"}:
        pytest.skip("direct32 is an x86 AVX2 specialization")
    cc = shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler")
    source = Path(__file__).resolve().parents[1] / "tools/bonsai_q1_kernel.c"
    output = tmp_path / "libbonsai-q1-direct32-gate.so"
    subprocess.run(
        [cc, "-std=c11", "-O3", "-fwrapv", "-fno-strict-overflow",
         "-fPIC", "-shared", str(source), "-o", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    lib = ctypes.CDLL(str(output))
    assert hasattr(lib, "bonsai_q1_direct32_boundary_selftest")
    fn = lib.bonsai_q1_direct32_boundary_selftest
    fn.argtypes = []
    fn.restype = ctypes.c_int
    rc = int(fn())
    if int(lib.bonsai_q1_runtime_has_avx2()):
        assert rc == 0
    else:
        assert rc == 5


def test_native_attention_rejects_overflowing_extents_before_pointer_access():
    lib = _load_lib()
    if lib is None:
        pytest.skip("native kernel unavailable")
    one = np.ones(1, dtype=np.int64)
    p = one.ctypes.data
    i64max = np.iinfo(np.int64).max

    # H*hd overflows size_t/byte extents; the one-element sentinels must never
    # be traversed while rejecting this malformed direct-C-ABI request.
    assert lib.bonsai_attention_decode_i64(
        p, p, p, i64max, 1, i64max, 1, i64max, i64max,
        16, 1 << 16, p, p, ctypes.c_size_t(i64max),
    ) == 1

    # Validate start+M without a signed int64 expression.
    assert lib.bonsai_attention_prefill_i64(
        p, p, p, 1, 1, 1, 2, 1, i64max, 16, 1 << 16, p,
    ) == 1

    pointers = (ctypes.c_void_p * 1)(p)
    lengths = np.array([i64max], dtype=np.int64)
    strides = np.array([i64max], dtype=np.int64)
    # lengths[0]*hd exceeds the signed stride domain and is rejected before
    # the max-absolute-value scan touches either cache pointer.
    assert lib.bonsai_attention_decode_batched_i64(
        p, pointers, pointers, lengths.ctypes.data, strides.ctypes.data,
        strides.ctypes.data, 1, 1, 1, 2, 16, 1 << 16, p,
    ) == 1


@pytest.mark.skipif(
    not _REAL_OPT_IN,
    reason="set TRINOTE_RUN_BONSAI35_REAL_EXECUTOR=1 for the external artifact",
)
def test_real_executor_pins_owners_against_caller_replacement():
    artifact = _real_artifact()
    original = artifact["layers"][0]["n1_gain_fp"]
    reference = weakref.ref(original)
    executor = Bonsai35NativeExecutor(artifact)
    assert isinstance(executor._owners, tuple)
    assert any(owner is original for owner in executor._owners)
    artifact["layers"][0]["n1_gain_fp"] = np.zeros_like(original)
    del original
    gc.collect()
    assert reference() is not None
    assert executor.prefill([12675]).shape == (1, 5120)


@pytest.mark.skipif(
    not _REAL_OPT_IN,
    reason="set TRINOTE_RUN_BONSAI35_REAL_EXECUTOR=1 for the external artifact",
)
@pytest.mark.parametrize("gain_name", ["n1_gain_fp", "n2_gain_fp"])
def test_real_resident_and_fused_rms_gain_refusal_is_transactional(gain_name):
    artifact = _real_artifact()
    artifact["layers"] = list(artifact["layers"])
    layer = dict(artifact["layers"][0])
    layer[gain_name] = np.full(5120, np.iinfo(np.int64).max, dtype=np.int64)
    artifact["layers"][0] = layer
    executor = Bonsai35NativeExecutor(artifact)
    with pytest.raises(RuntimeError, match="cache restored"):
        executor.prefill([12675])
    assert executor.position() == 0


@pytest.mark.skipif(
    not _REAL_OPT_IN,
    reason="set TRINOTE_RUN_BONSAI35_REAL_EXECUTOR=1 for the external artifact",
)
def test_real_resident_isa_is_stable_after_process_global_drift(monkeypatch):
    lib = _load_lib()
    if not int(lib.bonsai_q1_runtime_has_avx2()):
        pytest.skip("host has no AVX2")
    artifact = _real_artifact()
    monkeypatch.setenv("TRINOTE_Q1_ISA", "portable")
    portable = Bonsai35NativeExecutor(artifact)
    monkeypatch.setenv("TRINOTE_Q1_ISA", "avx2")
    avx2 = Bonsai35NativeExecutor(artifact)
    # The second constructor changed the standalone process-global selector.
    # Each resident handle must nevertheless keep its admitted ISA and report
    # it truthfully for the actual path it executes.
    assert q1_set_isa("avx2") == "avx2"
    a = portable.prefill([12675])
    assert q1_set_isa("portable") == "portable"
    b = avx2.prefill([12675])
    assert np.array_equal(a, b)
    assert portable.stats()["selected_isa"] == 1
    assert avx2.stats()["selected_isa"] == 2
