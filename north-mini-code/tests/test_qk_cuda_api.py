"""CPU-only guards for the optional CUDA ctypes API."""

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


class _FailedRegistrationLib:
    @staticmethod
    def qk_register_weight(_raw, _out_f, _n_blocks, _qtype):
        return -1


def test_registration_failure_raises_without_returning_cacheable_handle(monkeypatch):
    monkeypatch.setattr(qk_cuda, "_lib", lambda: _FailedRegistrationLib())
    with pytest.raises(qk_cuda.CudaRegistrationError, match="available VRAM"):
        qk_cuda.register_weight(bytes(144), out_f=1, n_blocks=1, qtype=qk_cuda.Q4_K)


class _MoeLib:
    @staticmethod
    def qk_moe_ffn(*_args):
        return 0

    @staticmethod
    def qk_moe_ffn_batched(*_args):
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
