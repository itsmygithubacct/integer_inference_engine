"""CPU-only guards for the optional CUDA ctypes API."""

from pathlib import Path
from types import SimpleNamespace
import inspect

import numpy as np
import pytest

from nmc import qk_cuda


class _Callable:
    def __init__(self, result=0, callback=None):
        self.result = result
        self.callback = callback

    def __call__(self, *_args):
        if self.callback is not None:
            self.callback()
        return self.result


class _AbiLib:
    def __init__(self, version):
        for name in qk_cuda._REQUIRED_CUDA_SYMBOLS:
            setattr(self, name, _Callable())
        self.qk_cuda_abi_version = _Callable(version)


def test_cuda_runtime_abi_rejects_stale_or_incomplete_library():
    assert qk_cuda._has_current_abi(_AbiLib(qk_cuda._CUDA_ABI_VERSION))
    assert not qk_cuda._has_current_abi(_AbiLib(qk_cuda._CUDA_ABI_VERSION - 1))
    incomplete = _AbiLib(qk_cuda._CUDA_ABI_VERSION)
    del incomplete.qk_moe_workspace_release
    assert not qk_cuda._has_current_abi(incomplete)


def test_cuda_build_gate_lists_every_required_runtime_symbol():
    root = Path(__file__).resolve().parents[1]
    build = (root / "tools" / "build_nmc_cuda.sh").read_text()
    source = (root / "tools" / "nmc_qk_cuda.cu").read_text()
    assert f"#define NMC_CUDA_ABI_VERSION {qk_cuda._CUDA_ABI_VERSION}" in source
    for symbol in qk_cuda._REQUIRED_CUDA_SYMBOLS:
        assert symbol in build, f"build publication does not gate required symbol {symbol}"
        assert f'extern "C"' in source and symbol in source
    incomplete = _AbiLib(qk_cuda._CUDA_ABI_VERSION)
    del incomplete.qk_apply_resident_grouped
    assert not qk_cuda._has_current_abi(incomplete)


class _FailedRegistrationLib:
    @staticmethod
    def qk_register_weight(_raw, _out_f, _n_blocks, _qtype):
        return -1

    @staticmethod
    def qk_register_i64(_raw, _rows, _cols):
        return -1


def test_registration_failure_raises_without_returning_cacheable_handle(monkeypatch):
    monkeypatch.setattr(qk_cuda, "_lib", lambda: _FailedRegistrationLib())
    with pytest.raises(qk_cuda.CudaRegistrationError, match="available VRAM"):
        qk_cuda.register_weight(bytes(144), out_f=1, n_blocks=1, qtype=qk_cuda.Q4_K)
    with pytest.raises(qk_cuda.CudaRegistrationError, match="available VRAM"):
        qk_cuda.register_i64(np.ones((2, 3), dtype=np.int64))


def test_resident_attention_bank_keeps_hidden_and_query_widths_distinct(monkeypatch):
    calls = []

    class _AttentionLib:
        @staticmethod
        def qk_attention_bank_create(*args):
            assert qk_cuda._CUDA_LOCK._is_owned()
            calls.append(args)
            return 7

        @staticmethod
        def qk_attention_bank_destroy(_handle):
            assert qk_cuda._CUDA_LOCK._is_owned()

    monkeypatch.setattr(qk_cuda, "_lib", lambda: _AttentionLib())
    cos = np.zeros((2, 32), dtype=np.int64)
    sin = np.zeros_like(cos)
    cache = qk_cuda.ResidentAttentionCache(1, 2, 256, 8, 2, 64, 16, cos, sin)
    assert cache.d_model == 256
    assert cache.q_width == 512
    assert calls and calls[0][:7] == (1, 2, 256, 8, 2, 64, 16)
    cache.close()

    with pytest.raises(ValueError, match="dimensions"):
        qk_cuda.ResidentAttentionCache(1, 2, 128, 8, 2, 64, 16, cos, sin)


def test_resident_preprocess_envelope_fallback_and_memory_proof(monkeypatch):
    calls = []

    class _PreprocessLib:
        @staticmethod
        def qk_rmsnorm_router(*_args):
            assert qk_cuda._CUDA_LOCK._is_owned()
            calls.append("native")
            return 3

    monkeypatch.setattr(qk_cuda, "_lib", lambda: _PreprocessLib())
    safe = np.arange(24, dtype=np.int64).reshape(3, 8)
    assert qk_cuda.rmsnorm_router(1, 2, safe, 3, 16, 24) is None
    assert calls == ["native"]

    unsafe = np.zeros((1, 8), dtype=np.int64)
    unsafe[0, :2] = np.iinfo(np.int64).min
    assert qk_cuda.rmsnorm_router(1, 2, unsafe, 3, 16, 24) is None
    assert calls == ["native"]             # rejected before crossing the ctypes boundary

    assert qk_cuda.resident_preprocess_bytes(48, 2048, 128) == 101_449_728
    with pytest.raises(ValueError, match="dimensions"):
        qk_cuda.resident_preprocess_bytes(-1, 2048, 128)


def test_engine_consumes_resident_routes_without_host_rerouting(monkeypatch):
    from nmc.engine import Engine
    from nmc.profiling import InferenceProfiler

    engine = object.__new__(Engine)
    engine.cfg = SimpleNamespace(n_used=2, n_experts=4, d_model=4, expert_ffn=256)
    engine.fused = True
    engine.batch_moe = False
    engine._profile = InferenceProfiler(False)
    engine._router = lambda *_args: (_ for _ in ()).throw(AssertionError("host router was called"))
    selected = []

    def expert_handle(name, expert):
        selected.append((name, int(expert)))
        return len(selected)

    engine._ehandle = expert_handle
    engine._native_call = lambda _operation, fn, *args, **kwargs: fn(*args, **kwargs)
    monkeypatch.setattr(qk_cuda, "moe_ffn", lambda *_args: np.arange(4, dtype=np.int64))
    hidden = np.zeros((1, 4), dtype=np.int64)
    routed = (np.array([[3, 1]], dtype=np.int32), np.array([[11, 7]], dtype=np.int64))
    got = engine._moe(hidden, "blk.1.", 1, routed=routed)

    assert got.tolist() == [[0, 1, 2, 3]]
    assert [expert for _name, expert in selected] == [3, 1, 3, 1, 3, 1]


class _MoeLib:
    @staticmethod
    def qk_moe_ffn(*_args):
        return 0

    @staticmethod
    def qk_moe_ffn_batched(*_args):
        return 0


class _GroupedLib:
    @staticmethod
    def qk_apply_resident_grouped(*_args):
        return 0


def test_moe_bridge_rejects_mismatched_handle_counts(monkeypatch):
    monkeypatch.setattr(qk_cuda, "_lib", lambda: _MoeLib())
    with pytest.raises(ValueError, match="same positive number"):
        qk_cuda.moe_ffn([1, 2], [3], [4, 5], [0] * 256, [1, 1], 256, 256, 16, 24)
    with pytest.raises(ValueError, match=r"m\*k=4"):
        qk_cuda.moe_ffn_batched([1], [2], [3], 2, 2, [0] * 512, [1] * 4, 256, 256, 16, 24)


def test_moe_bridge_holds_process_lock_across_ctypes_call(monkeypatch):
    def require_lock():
        assert qk_cuda._CUDA_LOCK._is_owned()

    lib = _MoeLib()
    lib.qk_moe_ffn = _Callable(0, require_lock)
    monkeypatch.setattr(qk_cuda, "_lib", lambda: lib)
    result = qk_cuda.moe_ffn([1], [2], [3], [0] * 256, [1], 256, 256, 16, 24)
    assert result is not None


def test_grouped_bridge_validates_shapes_and_holds_lock(monkeypatch):
    monkeypatch.setattr(qk_cuda, "_lib", lambda: _GroupedLib())
    with pytest.raises(ValueError, match="matching positive output sizes"):
        qk_cuda.apply_resident_grouped([1, 2], [0] * 256, [4], 24)
    with pytest.raises(ValueError, match="phase_ids"):
        qk_cuda.apply_resident_grouped([1], [0] * 256, [4], 24, [99])

    def require_lock(*_args):
        assert qk_cuda._CUDA_LOCK._is_owned()
        return 0

    lib = _GroupedLib()
    lib.qk_apply_resident_grouped = require_lock
    monkeypatch.setattr(qk_cuda, "_lib", lambda: lib)
    result = qk_cuda.apply_resident_grouped([1], [0] * 256, [4], 24, [qk_cuda.PROFILE_Q])
    assert result is not None and result[0].shape == (1, 4)


def test_batched_dp4a_envelope_code_falls_back_under_same_lock(monkeypatch):
    calls = []

    class _BatchLib(_MoeLib):
        @staticmethod
        def qk_moe_ffn_batched_dp4a(*_args):
            assert qk_cuda._CUDA_LOCK._is_owned()
            calls.append("dp4a")
            return 2

        @staticmethod
        def qk_moe_ffn_batched(*_args):
            assert qk_cuda._CUDA_LOCK._is_owned()
            calls.append("exact")
            return 0

    monkeypatch.setattr(qk_cuda, "_lib", lambda: _BatchLib())
    result = qk_cuda.moe_ffn_batched(
        [1], [2], [3], 1, 1, [0] * 256, [1], 256, 256, 16, 24, dp4a=True,
    )
    assert result is not None and calls == ["dp4a", "exact"]


def test_resident_layer_bridge_exposes_only_cold_prefix_and_retained_marker(monkeypatch):
    calls = []

    class _LayerLib:
        @staticmethod
        def qk_attention_bank_moe_configure(*_args):
            assert qk_cuda._CUDA_LOCK._is_owned()
            calls.append("configure")
            return 0

        @staticmethod
        def qk_attention_bank_moe_bind(*_args):
            assert qk_cuda._CUDA_LOCK._is_owned()
            calls.append("bind")
            return 0

        @staticmethod
        def qk_attention_bank_moe_begin(*args):
            assert qk_cuda._CUDA_LOCK._is_owned()
            calls.append(("begin", args[8], args[9]))
            args[-2]._obj.value = 1 if args[1] == 0 else 0
            args[-1][0] = 3
            if args[1] == 0:
                args[-1][1] = 99       # outside count: must never be exposed
            return 0

        @staticmethod
        def qk_attention_bank_moe_continue(*_args):
            assert qk_cuda._CUDA_LOCK._is_owned()
            calls.append("continue")
            return 0

    cache = object.__new__(qk_cuda.ResidentAttentionCache)
    cache.handle = 7
    cache.n_layers, cache.max_length = 2, 8
    cache.n_heads, cache.n_kv, cache.head_dim, cache.fa = 1, 1, 256, 16
    cache.d_model = 256
    cache._lengths = [0, 0]
    cache._moe_configs = {}
    cache._retained_generation = 0
    cache._pending_moe_layer = None
    cache._closed = False
    monkeypatch.setattr(qk_cuda, "_lib", lambda: _LayerLib())

    cache.configure_moe_layer(0, 4, 2, 256, 256)
    cache.configure_moe_layer(1, 4, 2, 256, 256)
    cache.bind_moe_expert(0, 1, 10, 11, 12)
    hidden = np.zeros((1, 256), dtype=np.int64)
    assert cache.begin_moe_layer(0, 1, 2, 3, 4, 5, 6, hidden, 24, 1, None, False) == (3,)
    retained = cache.continue_moe_layer(0, 24)
    assert isinstance(retained, qk_cuda.ResidentLayerHidden)
    assert cache.begin_moe_layer(1, 1, 2, 3, 4, 5, 6, retained, 24, 1, None, False) == ()
    assert calls[-1][0] == "begin" and calls[-1][1] is None and calls[-1][2] == 1

    with pytest.raises(ValueError, match="stale"):
        stale = qk_cuda.ResidentLayerHidden(cache, retained._generation - 1)
        cache._pending_moe_layer = None
        cache.begin_moe_layer(1, 1, 2, 3, 4, 5, 6, stale, 24, 1, None, False)


def test_resident_layer_bridge_fails_closed_on_native_continuation_status(monkeypatch):
    class _LayerLib:
        @staticmethod
        def qk_attention_bank_moe_continue(*_args):
            assert qk_cuda._CUDA_LOCK._is_owned()
            return 4

    cache = object.__new__(qk_cuda.ResidentAttentionCache)
    cache.handle = 9
    cache.n_layers, cache.max_length, cache.d_model = 1, 8, 256
    cache._moe_configs = {0: (4, 2, 256, 256)}
    cache._pending_moe_layer = 0
    cache._retained_generation = 0
    cache._closed = False
    monkeypatch.setattr(qk_cuda, "_lib", lambda: _LayerLib())
    with pytest.raises(qk_cuda.CudaContextError, match="cold experts remain"):
        cache.continue_moe_layer(0, 24)
    assert cache._pending_moe_layer == 0       # caller may bind and retry the same prepared layer


def _fake_resident_cache(n_layers=49, d_model=256):
    cache = object.__new__(qk_cuda.ResidentAttentionCache)
    cache.handle = 17
    cache.n_layers, cache.max_length = n_layers, 8
    cache.n_heads, cache.n_kv, cache.head_dim, cache.fa = 1, 1, d_model, 16
    cache.d_model = d_model
    cache._lengths = [0] * n_layers
    cache._moe_configs = {}
    cache._retained_generation = 0
    cache._pending_moe_layer = None
    cache._token_executor = None
    cache._closed = False
    return cache


def test_resident_token_executor_orchestrates_48_layers_and_loads_only_cold():
    cache = _fake_resident_cache()
    configured, bound, begins, continued, lookups, loads = [], [], [], [], [], []

    def configure(layer, *_dims):
        configured.append(layer)

    def bind(layer, expert, *handles):
        bound.append((layer, expert, handles))

    def begin(layer, *_args):
        begins.append(layer)
        return (1,)

    def continuation(layer, _fw, *, publish=False):
        continued.append((layer, publish))
        cache._retained_generation += 1
        return (np.full((1, 256), layer, dtype=np.int64) if publish else
                qk_cuda.ResidentLayerHidden(cache, cache._retained_generation))

    cache.configure_moe_layer = configure
    cache.bind_moe_expert = bind
    cache.begin_moe_layer = begin
    cache.continue_moe_layer = continuation

    def lookup(layer, expert):
        lookups.append((layer, expert))
        return (100, 101, 102) if expert == 0 else None

    def load(layer, expert):
        loads.append((layer, expert))
        return 200 + layer, 300 + layer, 400 + layer

    executor = qk_cuda.ResidentMoeTokenExecutor(
        cache, first_layer=1, layer_count=48, n_experts=4, n_used=2,
        d_model=256, expert_ffn=256, fw=24, eps=1,
        lookup_expert=lookup, load_expert=load,
    )
    specs = tuple(qk_cuda.ResidentMoeLayerSpec(layer, 1, 2, 3, 4, 5, 6, None, False)
                  for layer in range(1, 49))
    executor.prepare(specs)
    assert configured == list(range(1, 49))
    assert begins == []                              # preflight does not mutate K/V
    assert lookups == [(layer, expert) for layer in range(1, 49) for expert in range(4)]
    result = executor.run(np.zeros((1, 256), dtype=np.int64), specs)
    assert result.tolist() == [[48] * 256]
    assert begins == list(range(1, 49))
    assert loads == [(layer, 1) for layer in range(1, 49)]
    assert all(expert in (0, 1) for _layer, expert, _handles in bound)
    assert continued == [(layer, layer == 48) for layer in range(1, 49)]
    assert not executor.poisoned


def test_resident_token_executor_poison_requires_request_teardown():
    cache = _fake_resident_cache(n_layers=1)
    cache.configure_moe_layer = lambda *_args: None
    cache.bind_moe_expert = lambda *_args: None
    cache.begin_moe_layer = lambda *_args: (2,)
    cache.continue_moe_layer = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("continuation must not run without handles")
    )
    executor = qk_cuda.ResidentMoeTokenExecutor(
        cache, first_layer=0, layer_count=1, n_experts=4, n_used=2,
        d_model=256, expert_ffn=256, fw=24, eps=1,
        lookup_expert=lambda *_args: None, load_expert=lambda *_args: None,
    )
    specs = (qk_cuda.ResidentMoeLayerSpec(0, 1, 2, 3, 4, 5, 6, None, False),)
    with pytest.raises(qk_cuda.CudaContextError, match="handle triplet"):
        executor.run(np.zeros((1, 256), dtype=np.int64), specs)
    assert executor.poisoned
    with pytest.raises(qk_cuda.CudaContextError, match="poisoned"):
        executor.run(np.zeros((1, 256), dtype=np.int64), specs)


def test_engine_resident_token_entry_is_explicit_opt_in_and_not_generate(monkeypatch):
    from nmc.engine import Engine

    engine = object.__new__(Engine)
    engine.resident_layer_executor = False
    engine.resident = engine.resident_attention = True
    engine.DENSE, engine.NL = 1, 49
    engine.cfg = SimpleNamespace(d_model=256, n_experts=4, n_used=2, expert_ffn=256, eps=1)
    cache = _fake_resident_cache()
    hidden = np.zeros((1, 256), dtype=np.int64)
    with pytest.raises(RuntimeError, match="disabled"):
        engine.resident_decode_token(hidden, cache, None, None)

    engine.resident_layer_executor = True
    monkeypatch.setattr(qk_cuda, "resident_layer_available", lambda: True)
    specs = tuple(qk_cuda.ResidentMoeLayerSpec(layer, 1, 2, 3, 4, 5, 6, None, False)
                  for layer in range(1, 49))
    engine._resident_moe_layer_specs = lambda: specs
    events = []

    class _Executor:
        _engine_owner = engine

        def prepare(self, got):
            events.append(("prepare", len(got)))

        def run(self, value, got):
            events.append(("run", len(got), int(value[0, 0])))
            return value + 2

    cache._token_executor = _Executor()
    engine._block = lambda value, layer, *_args: (
        events.append(("dense", layer)) or value + 1
    )
    got = engine.resident_decode_token(hidden, cache, None, None)
    assert got.tolist() == [[3] * 256]
    assert events == [("prepare", 48), ("dense", 0), ("run", 48, 1)]
    assert "resident_decode_token" not in inspect.getsource(Engine.generate)
