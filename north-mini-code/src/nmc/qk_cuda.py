"""ctypes loader for the CUDA integer kernel (Q4_K/Q6_K fused dequant + fixed-point matmul).

Per-host, arch-specific; returns None / available()=False when the .so or a usable GPU is absent, so callers
fall back to the CPU kernel (qk_native) or the numpy oracle. Every result MUST be byte-identical to the CPU
oracle — the GPU is a producer, the CPU oracle the canonical verifier (tests/test_qk_cuda.py)."""
from __future__ import annotations

import ctypes
import math
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from nmc.qk_native import _require_weight_bytes   # shared Q4_K/Q6_K superblock buffer-length guard

Q4_K, Q6_K = 0, 1
_DP4A_SAFE_LIMBS = 4
_CUDA_ABI_VERSION = 5
PROFILE_Q, PROFILE_K, PROFILE_V, PROFILE_O, PROFILE_OTHER = 1, 2, 3, 4, 0
_PROFILE_METRICS = (
    "registration_calls", "registration_ns",
    "h2d_calls", "h2d_bytes", "h2d_ns",
    "d2h_calls", "d2h_bytes", "d2h_ns",
    "allocation_calls", "allocation_bytes", "allocation_ns", "free_calls",
    "native_calls", "resident_apply_calls", "grouped_apply_calls",
    "moe_calls", "moe_batched_calls", "moe_batched_dp4a_calls",
    "q_projection_calls", "q_projection_ns", "k_projection_calls", "k_projection_ns",
    "v_projection_calls", "v_projection_ns", "o_projection_calls", "o_projection_ns",
    "other_projection_calls", "other_projection_ns",
)
# The resident registry and reusable activation workspaces are process-global
# in the CUDA library.  CDLL calls release the GIL, so the bridge must enforce
# the single-caller contract rather than merely documenting it.
_CUDA_LOCK = threading.RLock()
_REQUIRED_CUDA_SYMBOLS = (
    "qk_cuda_abi_version",
    "qk_linear_cuda",
    "qk_cuda_available",
    "qk_register_weight",
    "qk_register_i64",
    "qk_resident_reserve",
    "qk_apply_resident",
    "qk_apply_resident_grouped",
    "qk_rmsnorm_router",
    "qk_attention_bank_create",
    "qk_attention_bank_import",
    "qk_attention_bank_apply",
    "qk_attention_bank_moe_configure",
    "qk_attention_bank_moe_bind",
    "qk_attention_bank_moe_begin",
    "qk_attention_bank_moe_continue",
    "qk_attention_bank_moe_export",
    "qk_attention_bank_workspace_bytes",
    "qk_attention_bank_reset",
    "qk_attention_bank_destroy",
    "qk_free_all",
    "qk_resident_count",
    "qk_resident_capacity",
    "qk_moe_workspace_allocations",
    "qk_moe_workspace_bytes",
    "qk_moe_workspace_release",
    "qk_moe_ffn",
    "qk_moe_ffn_batched",
    "qk_moe_ffn_batched_dp4a",
    "qk_profile_reset",
    "qk_profile_set_enabled",
    "qk_profile_snapshot",
)


class CudaRegistrationError(RuntimeError):
    """A resident weight could not be committed to the CUDA registry/VRAM."""


class CudaContextError(RuntimeError):
    """A resident request context failed validation, allocation, or an exact envelope guard."""


def _has_current_abi(lib) -> bool:
    """Return whether a loaded per-host library implements this bridge's exact runtime contract."""
    try:
        if any(not hasattr(lib, name) for name in _REQUIRED_CUDA_SYMBOLS):
            return False
        lib.qk_cuda_abi_version.restype = ctypes.c_int
        with _CUDA_LOCK:
            return int(lib.qk_cuda_abi_version()) == _CUDA_ABI_VERSION
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def _so_path() -> Path:
    env = os.environ.get("BONSAI_BIN_DIR")
    home = os.environ.get("BONSAI_NOTARY_HOME") or str(Path.home() / ".local/integer_inference_engine/north-mini-code")
    cands = [Path(env) / "libnmc_qk_cuda.so"] if env else []
    cands += [Path(home) / "bin" / "libnmc_qk_cuda.so",
              Path(__file__).resolve().parents[2] / "tools" / "libnmc_qk_cuda.so"]
    for c in cands:
        if c.exists():
            return c
    return cands[-1]


@lru_cache(maxsize=1)
def _lib():
    p = _so_path()
    if not p.exists():
        return None
    try:
        lib = ctypes.CDLL(str(p))
        if not _has_current_abi(lib):
            return None
    except (OSError, AttributeError):
        return None
    lib.qk_linear_cuda.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                   ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.qk_linear_cuda.restype = ctypes.c_int
    lib.qk_cuda_available.restype = ctypes.c_int
    # resident-weight register API (optional; guarded so an older .so without these symbols still loads)
    try:
        lib.qk_register_weight.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
        lib.qk_register_weight.restype = ctypes.c_int64
        lib.qk_register_i64.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        lib.qk_register_i64.restype = ctypes.c_int64
        lib.qk_resident_reserve.argtypes = [ctypes.c_int64]
        lib.qk_resident_reserve.restype = ctypes.c_int
        lib.qk_apply_resident.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p]
        lib.qk_apply_resident.restype = ctypes.c_int
        P64 = ctypes.POINTER(ctypes.c_int64)
        P32 = ctypes.POINTER(ctypes.c_int)
        lib.qk_apply_resident_grouped.argtypes = [P64, P32, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64,
                                                  ctypes.c_int, ctypes.c_int64, ctypes.c_void_p]
        lib.qk_apply_resident_grouped.restype = ctypes.c_int
        lib.qk_rmsnorm_router.argtypes = [
            ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_int,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.qk_rmsnorm_router.restype = ctypes.c_int
        lib.qk_attention_bank_create.argtypes = [
            ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.qk_attention_bank_create.restype = ctypes.c_int64
        lib.qk_attention_bank_import.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_void_p,
                                                  ctypes.c_void_p, ctypes.c_int64]
        lib.qk_attention_bank_import.restype = ctypes.c_int
        lib.qk_attention_bank_apply.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
                                                 ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int,
                                                 ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
                                                 ctypes.c_void_p]
        lib.qk_attention_bank_apply.restype = ctypes.c_int
        lib.qk_attention_bank_moe_configure.argtypes = [
            ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
        ]
        lib.qk_attention_bank_moe_configure.restype = ctypes.c_int
        lib.qk_attention_bank_moe_bind.argtypes = [
            ctypes.c_int64, ctypes.c_int, ctypes.c_int,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ]
        lib.qk_attention_bank_moe_bind.restype = ctypes.c_int
        lib.qk_attention_bank_moe_begin.argtypes = [
            ctypes.c_int64, ctypes.c_int,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int,
            P32, P32,
        ]
        lib.qk_attention_bank_moe_begin.restype = ctypes.c_int
        lib.qk_attention_bank_moe_continue.argtypes = [
            ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_void_p,
        ]
        lib.qk_attention_bank_moe_continue.restype = ctypes.c_int
        lib.qk_attention_bank_moe_export.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        lib.qk_attention_bank_moe_export.restype = ctypes.c_int
        lib.qk_attention_bank_workspace_bytes.argtypes = [ctypes.c_int64]
        lib.qk_attention_bank_workspace_bytes.restype = ctypes.c_uint64
        lib.qk_attention_bank_reset.argtypes = [ctypes.c_int64]
        lib.qk_attention_bank_reset.restype = ctypes.c_int
        lib.qk_attention_bank_destroy.argtypes = [ctypes.c_int64]
        lib.qk_attention_bank_destroy.restype = None
        lib.qk_free_all.restype = None
        lib.qk_resident_count.restype = ctypes.c_int64
        if hasattr(lib, "qk_resident_capacity"):
            lib.qk_resident_capacity.restype = ctypes.c_int64
        if hasattr(lib, "qk_moe_workspace_allocations"):
            lib.qk_moe_workspace_allocations.restype = ctypes.c_int64
        lib.qk_moe_workspace_bytes.restype = ctypes.c_uint64
        lib.qk_moe_workspace_release.restype = None
        lib.qk_moe_ffn.argtypes = [P64, P64, P64, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.qk_moe_ffn.restype = ctypes.c_int
        _dp4a_args = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.qk_apply_resident_q6k_dp4a.argtypes = _dp4a_args
        lib.qk_apply_resident_q6k_dp4a.restype = ctypes.c_int
        lib.qk_apply_resident_q4k_dp4a.argtypes = _dp4a_args
        lib.qk_apply_resident_q4k_dp4a.restype = ctypes.c_int
        lib.qk_moe_ffn_dp4a.argtypes = [P64, P64, P64, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.qk_moe_ffn_dp4a.restype = ctypes.c_int
        lib.qk_moe_ffn_batched.argtypes = [P64, P64, P64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_void_p]
        lib.qk_moe_ffn_batched.restype = ctypes.c_int
        lib.qk_moe_ffn_batched_dp4a.argtypes = lib.qk_moe_ffn_batched.argtypes
        lib.qk_moe_ffn_batched_dp4a.restype = ctypes.c_int
        lib.qk_profile_reset.restype = None
        lib.qk_profile_set_enabled.argtypes = [ctypes.c_int]
        lib.qk_profile_set_enabled.restype = None
        lib.qk_profile_snapshot.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]
        lib.qk_profile_snapshot.restype = ctypes.c_int
    except AttributeError:
        pass
    try:
        with _CUDA_LOCK:
            if lib.qk_cuda_available() != 0:
                return None                           # lib loaded but no usable GPU -> fall back
    except OSError:
        return None
    return lib


def available() -> bool:
    return _lib() is not None


def qk_linear(weight_raw: bytes, x_int: np.ndarray, out_f: int, n_blocks: int, fw: int, qtype: int):
    """GPU y[t,o] = (Σ_i W_fixed[o,i]·x[t,i]) >> fw from raw Q4_K/Q6_K bytes. x_int [T, in_f] int64.
    Returns y [T, out_f] int64, or None if the kernel/GPU is unavailable or a launch fails (caller falls back)."""
    lib = _lib()
    if lib is None:
        return None
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_int, dtype=np.int64)))
    T, in_f = x.shape
    assert in_f == n_blocks * 256, (in_f, n_blocks)
    _require_weight_bytes(weight_raw, out_f, n_blocks, qtype)   # fail loud on a short buffer (no OOB D2H copy)
    wbuf = (ctypes.c_char * len(weight_raw)).from_buffer_copy(weight_raw)
    out = np.empty((T, out_f), dtype=np.int64)
    rc = lib.qk_linear_cuda(ctypes.cast(wbuf, ctypes.c_void_p), x.ctypes.data, T, out_f, n_blocks,
                            int(fw), int(qtype), out.ctypes.data)
    return None if rc != 0 else out


def resident_available() -> bool:
    lib = _lib()
    return lib is not None and hasattr(lib, "qk_register_weight")


def register_weight(weight_raw: bytes, out_f: int, n_blocks: int, qtype: int):
    """Upload one weight tensor's raw bytes to VRAM once.

    Returns ``None`` only when the resident API is unavailable. Once that API is selected, a failed registry
    growth/allocation/upload raises instead of returning a sentinel that an engine could cache as a handle.
    """
    lib = _lib()
    if lib is None or not hasattr(lib, "qk_register_weight"):
        return None
    _require_weight_bytes(weight_raw, out_f, n_blocks, qtype)   # fail loud on a short buffer (no OOB VRAM upload)
    wbuf = (ctypes.c_char * len(weight_raw)).from_buffer_copy(weight_raw)
    with _CUDA_LOCK:
        h = lib.qk_register_weight(ctypes.cast(wbuf, ctypes.c_void_p), out_f, n_blocks, qtype)
    if h < 0:
        raise CudaRegistrationError(
            f"CUDA resident weight registration failed for shape [{out_f}, {n_blocks * 256}] "
            f"(qtype={qtype}); check dimensions and available VRAM"
        )
    return int(h)


def resident_preprocess_available() -> bool:
    """Whether the exact resident RMSNorm/router/top-k boundary is usable."""
    lib = _lib()
    return lib is not None and hasattr(lib, "qk_register_i64") and hasattr(lib, "qk_rmsnorm_router")


def register_i64(matrix: np.ndarray):
    """Upload a dense fixed-int64 row-major matrix for resident preprocessing.

    A one-dimensional gain is registered as a single-row matrix.  Dense
    handles are a separate native kind and cannot be passed to quantized
    Q4_K/Q6_K projection APIs.
    """
    lib = _lib()
    if lib is None or not hasattr(lib, "qk_register_i64"):
        return None
    value = np.asarray(matrix, dtype=np.int64)
    if value.ndim == 1:
        value = value.reshape(1, -1)
    if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError("resident int64 registration requires a non-empty vector or 2-D matrix")
    value = np.ascontiguousarray(value)
    rows, cols = (int(n) for n in value.shape)
    with _CUDA_LOCK:
        handle = int(lib.qk_register_i64(value.ctypes.data, rows, cols))
    if handle < 0:
        raise CudaRegistrationError(
            f"CUDA resident int64 registration failed for shape [{rows}, {cols}]; "
            "check dimensions and available VRAM"
        )
    return handle


_I128_MAX = (1 << 127) - 1


def _rmsnorm_i128_safe(x: np.ndarray) -> bool:
    """Prove each RMSNorm sum-of-squares fits the native signed-i128 envelope.

    Normal rows take the inexpensive max-magnitude bound.  Only a row near
    int64 extremes needs an exact Python-integer sum, preserving acceptance
    for values whose coarse ``max² * width`` bound is inconclusive.
    """
    if x.ndim != 2 or x.shape[1] <= 0:
        return False
    width = int(x.shape[1])
    for row in x:
        maxabs = max(abs(int(row.min())), abs(int(row.max())))
        if maxabs * maxabs * width <= _I128_MAX:
            continue
        total = 0
        for value in row:
            total += int(value) * int(value)
            if total > _I128_MAX:
                return False
    return True


def rmsnorm_router(gain_handle: int, router_handle: int, x_int: np.ndarray, n_used: int,
                   fa: int, fw: int, eps: int = 1):
    """Exact resident RMSNorm, router projection, stable top-k, and sigmoid.

    Returns ``(normalized_h, selected_ids, gates)`` with shapes ``[T,D]``,
    ``[T,K]``, and ``[T,K]``.  ``None`` means the CUDA API is unavailable or
    the activation left the signed-i128 RMSNorm envelope, allowing the caller
    to execute the unchanged arbitrary-precision host path.  Native shape or
    runtime failures raise instead of exposing partial outputs.
    """
    lib = _lib()
    if lib is None or not hasattr(lib, "qk_rmsnorm_router"):
        return None
    x = np.asarray(x_int, dtype=np.int64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] <= 0:
        raise ValueError("resident RMSNorm/router requires a non-empty vector or 2-D activation")
    x = np.ascontiguousarray(x)
    n_used, fa, fw, eps = int(n_used), int(fa), int(fw), int(eps)
    if not 0 < n_used <= np.iinfo(np.int32).max:
        raise ValueError("resident RMSNorm/router requires n_used in the native int32 range")
    if not 1 <= fa <= 29 or not 0 <= fw <= 62 or not 0 <= eps <= (1 << 64) - 1:
        raise ValueError("resident RMSNorm/router fixed-point parameters are outside the native envelope")
    if not _rmsnorm_i128_safe(x):
        return None
    rows, width = (int(n) for n in x.shape)
    if rows > np.iinfo(np.intp).max // width or rows > np.iinfo(np.intp).max // n_used:
        raise ValueError("resident RMSNorm/router activation shape is too large")
    h = np.empty_like(x)
    ids = np.empty((rows, n_used), dtype=np.int32)
    gates = np.empty((rows, n_used), dtype=np.int64)
    with _CUDA_LOCK:
        rc = int(lib.qk_rmsnorm_router(
            int(gain_handle), int(router_handle), x.ctypes.data, rows, fa, fw, eps, n_used,
            h.size, ids.size, h.ctypes.data, ids.ctypes.data, gates.ctypes.data,
        ))
    if rc == 3:
        return None
    if rc != 0:
        raise CudaContextError(f"resident RMSNorm/router failed with native status {rc}")
    return h, ids, gates


def resident_preprocess_bytes(n_layers: int, d_model: int, n_experts: int) -> int:
    """Dense VRAM bytes for per-layer int64 norm gains plus router matrices.

    Expert Q4_K/Q6_K slices are intentionally excluded: those stay lazily
    selected and registered by route, avoiding an all-expert preload.
    """
    n_layers, d_model, n_experts = int(n_layers), int(d_model), int(n_experts)
    if n_layers < 0 or d_model <= 0 or n_experts <= 0:
        raise ValueError("resident preprocessing memory dimensions must be non-negative layers and positive widths")
    return n_layers * (n_experts + 1) * d_model * np.dtype(np.int64).itemsize


def apply_resident(handle: int, x_int: np.ndarray, out_f: int, fw: int):
    """Apply a resident weight: y[t,o]=(Σ_i W_fixed[o,i]·x[t,i])>>fw. Only x crosses PCIe. y [T,out_f] int64."""
    lib = _lib()
    if lib is None:
        return None
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_int, dtype=np.int64)))
    T = x.shape[0]
    out = np.empty((T, out_f), dtype=np.int64)
    with _CUDA_LOCK:
        rc = lib.qk_apply_resident(int(handle), x.ctypes.data, T, int(fw), out.ctypes.data)
    return None if rc != 0 else out


def grouped_available() -> bool:
    lib = _lib()
    return lib is not None and hasattr(lib, "qk_apply_resident_grouped")


def apply_resident_grouped(handles, x_int: np.ndarray, out_features, fw: int, phase_ids=None):
    """Apply same-input resident projections with one H2D and one D2H boundary.

    Results are returned as a tuple in handle order.  Each result is exactly
    the corresponding :func:`apply_resident` array; grouping changes neither
    an output row's kernel nor its arithmetic order.  ``phase_ids`` feeds the
    optional native Q/K/V/O event telemetry and has no execution semantics.
    """
    lib = _lib()
    if lib is None or not hasattr(lib, "qk_apply_resident_grouped"):
        return None
    hs = tuple(int(h) for h in handles)
    ofs = tuple(int(n) for n in out_features)
    if not hs or len(hs) != len(ofs) or len(hs) > 16 or any(n <= 0 for n in ofs):
        raise ValueError("grouped resident apply requires 1..16 handles and matching positive output sizes")
    phases = tuple(PROFILE_OTHER for _ in hs) if phase_ids is None else tuple(int(p) for p in phase_ids)
    if len(phases) != len(hs) or any(p not in (PROFILE_OTHER, PROFILE_Q, PROFILE_K, PROFILE_V, PROFILE_O)
                                    for p in phases):
        raise ValueError("grouped resident phase_ids must match handles and use known Q/K/V/O identifiers")
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_int, dtype=np.int64)))
    rows = int(x.shape[0])
    total = rows * sum(ofs)
    if total <= 0 or total > np.iinfo(np.intp).max // np.dtype(np.int64).itemsize:
        raise ValueError("grouped resident output size is invalid")
    flat = np.empty(total, dtype=np.int64)
    harr = (ctypes.c_int64 * len(hs))(*hs)
    parr = (ctypes.c_int * len(phases))(*phases)
    with _CUDA_LOCK:
        rc = lib.qk_apply_resident_grouped(harr, parr, len(hs), x.ctypes.data, rows, int(fw), total,
                                           flat.ctypes.data)
    if rc != 0:
        return None
    result, offset = [], 0
    for out_f in ofs:
        count = rows * out_f
        result.append(flat[offset:offset + count].reshape(rows, out_f))
        offset += count
    return tuple(result)


def resident_attention_available() -> bool:
    lib = _lib()
    return lib is not None and hasattr(lib, "qk_attention_bank_apply")


def resident_layer_available() -> bool:
    """Whether the request-scoped cold-route MoE continuation ABI is usable."""
    lib = _lib()
    return lib is not None and all(hasattr(lib, name) for name in (
        "qk_attention_bank_moe_configure", "qk_attention_bank_moe_bind",
        "qk_attention_bank_moe_begin", "qk_attention_bank_moe_continue",
        "qk_attention_bank_moe_export", "qk_attention_bank_workspace_bytes",
    ))


class ResidentLayerHidden:
    """Opaque marker for an M=1 residual retained by a request CUDA bank.

    The marker deliberately exposes no host activation.  It is accepted only
    by the same bank and only at the generation immediately following the
    continuation that created it.
    """

    __slots__ = ("_cache", "_generation", "shape")

    def __init__(self, cache, generation: int):
        self._cache = cache
        self._generation = int(generation)
        self.shape = (1, cache.d_model)


class ResidentAttentionCache:
    """Request-scoped device K/V bank and exact M=1 resident attention executor.

    Prefill remains on the established batched path and is imported once. Each
    subsequent decode call keeps Q/K/V, RoPE, K/V append, attention scratch,
    and O projection device-side. A failed native call poisons that layer and
    raises; callers must close the request context instead of retrying against
    partially written device state.
    """

    def __init__(self, n_layers: int, max_length: int, d_model: int, n_heads: int,
                 n_kv: int, head_dim: int, fa: int, cos, sin):
        lib = _lib()
        if lib is None or not hasattr(lib, "qk_attention_bank_create"):
            raise CudaContextError("resident attention context API is unavailable")
        values = (
            int(n_layers), int(max_length), int(d_model), int(n_heads),
            int(n_kv), int(head_dim), int(fa),
        )
        n_layers, max_length, d_model, n_heads, n_kv, head_dim, fa = values
        q_width = n_heads * head_dim
        if n_layers <= 0 or max_length <= 0 or d_model <= 0 or n_heads <= 0 or n_kv <= 0 or \
                n_heads % n_kv or head_dim <= 0 or head_dim % 2 or not 1 <= fa <= 29 or \
                d_model % 256 or q_width % 256:
            raise ValueError("invalid resident attention dimensions")
        c = np.ascontiguousarray(np.asarray(cos, dtype=np.int64))
        s = np.ascontiguousarray(np.asarray(sin, dtype=np.int64))
        expected = (max_length, head_dim // 2)
        if c.shape != expected or s.shape != expected:
            raise ValueError(f"resident attention RoPE tables must both have shape {expected}")
        with _CUDA_LOCK:
            handle = int(lib.qk_attention_bank_create(
                n_layers, max_length, d_model, n_heads, n_kv, head_dim, fa,
                c.ctypes.data, s.ctypes.data,
            ))
        if handle < 0:
            raise CudaContextError("resident attention context creation failed; check dimensions and VRAM")
        self.handle = handle
        self.n_layers, self.max_length = n_layers, max_length
        self.n_heads, self.n_kv, self.head_dim, self.fa = n_heads, n_kv, head_dim, fa
        self.d_model, self.q_width = d_model, q_width
        self._lengths = [0] * n_layers
        self._moe_configs = {}
        self._retained_generation = 0
        self._pending_moe_layer = None
        self._token_executor = None
        self._closed = False

    def length(self, layer=0):
        return self._lengths[layer]

    def import_layer(self, layer: int, k, v) -> None:
        if self._closed:
            raise CudaContextError("resident attention context is closed")
        layer = int(layer)
        kk = np.ascontiguousarray(np.asarray(k, dtype=np.int64))
        vv = np.ascontiguousarray(np.asarray(v, dtype=np.int64))
        if kk.shape != vv.shape or kk.ndim != 3 or kk.shape[0] != self.n_kv or kk.shape[2] != self.head_dim:
            raise ValueError("resident attention import expects matching [n_kv, length, head_dim] K/V")
        length = int(kk.shape[1])
        if not 0 <= layer < self.n_layers or length > self.max_length:
            raise ValueError("resident attention import exceeds layer or context bounds")
        lib = _lib()
        with _CUDA_LOCK:
            rc = lib.qk_attention_bank_import(self.handle, layer, kk.ctypes.data, vv.ctypes.data, length)
        if rc != 0:
            raise CudaContextError(f"resident attention K/V import failed for layer {layer}")
        self._lengths[layer] = length

    def apply(self, layer: int, q_handle: int, k_handle: int, v_handle: int, o_handle: int,
              hidden, fw: int, window: int | None, rope: bool):
        if self._closed:
            raise CudaContextError("resident attention context is closed")
        layer = int(layer)
        if not 0 <= layer < self.n_layers or self._lengths[layer] >= self.max_length:
            raise ValueError("resident attention decode exceeds layer or context bounds")
        x = np.ascontiguousarray(np.asarray(hidden, dtype=np.int64))
        if x.shape != (1, self.d_model):
            raise ValueError(f"resident attention apply expects hidden shape (1, {self.d_model})")
        out = np.empty((1, self.d_model), dtype=np.int64)
        inv_sqrt = int(round((1.0 / math.sqrt(self.head_dim)) * (1 << self.fa)))
        native_window = -1 if window is None else int(window)
        lib = _lib()
        with _CUDA_LOCK:
            rc = int(lib.qk_attention_bank_apply(
                self.handle, layer, int(q_handle), int(k_handle), int(v_handle), int(o_handle),
                x.ctypes.data, int(fw), native_window, int(bool(rope)), inv_sqrt,
                out.size, out.ctypes.data,
            ))
        if rc != 0:
            detail = "exact overflow guard fired" if rc == 3 else f"native status {rc}"
            raise CudaContextError(f"resident attention apply failed for layer {layer}: {detail}")
        self._lengths[layer] += 1
        return out

    def configure_moe_layer(self, layer: int, n_experts: int, n_used: int,
                            d_model: int, expert_ffn: int) -> None:
        """Allocate compact request metadata for one MoE layer, not weights."""
        if self._closed:
            raise CudaContextError("resident attention context is closed")
        values = tuple(int(value) for value in (layer, n_experts, n_used, d_model, expert_ffn))
        layer, n_experts, n_used, d_model, expert_ffn = values
        if not 0 <= layer < self.n_layers or n_experts <= 0 or not 0 < n_used <= n_experts or \
                d_model != self.d_model or expert_ffn <= 0 or d_model % 256 or expert_ffn % 256:
            raise ValueError("invalid resident MoE layer dimensions")
        config = (n_experts, n_used, d_model, expert_ffn)
        existing = self._moe_configs.get(layer)
        if existing is not None and existing != config:
            raise ValueError("resident MoE layer cannot be reconfigured with different dimensions")
        lib = _lib()
        with _CUDA_LOCK:
            rc = int(lib.qk_attention_bank_moe_configure(
                self.handle, layer, n_experts, n_used, d_model, expert_ffn,
            ))
        if rc != 0:
            raise CudaContextError(f"resident MoE layer configuration failed for layer {layer}")
        self._moe_configs[layer] = config

    def bind_moe_expert(self, layer: int, expert: int, gate_handle: int,
                        up_handle: int, down_handle: int) -> None:
        """Bind one already-resident selected expert triplet to this request."""
        if self._closed:
            raise CudaContextError("resident attention context is closed")
        layer, expert = int(layer), int(expert)
        if layer not in self._moe_configs or not 0 <= expert < self._moe_configs[layer][0]:
            raise ValueError("resident MoE expert binding is outside the configured layer")
        lib = _lib()
        with _CUDA_LOCK:
            rc = int(lib.qk_attention_bank_moe_bind(
                self.handle, layer, expert, int(gate_handle), int(up_handle), int(down_handle),
            ))
        if rc != 0:
            raise CudaContextError(
                f"resident MoE expert binding failed for layer {layer}, expert {expert}"
            )

    def begin_moe_layer(self, layer: int, gain_handle: int, router_handle: int,
                        q_handle: int, k_handle: int, v_handle: int, o_handle: int,
                        hidden, fw: int, eps: int, window: int | None, rope: bool) -> tuple[int, ...]:
        """Retain preprocessing/attention state and return only unbound expert IDs."""
        if self._closed:
            raise CudaContextError("resident attention context is closed")
        layer = int(layer)
        if layer not in self._moe_configs or self._pending_moe_layer is not None or \
                self._lengths[layer] >= self.max_length:
            raise ValueError("resident MoE layer begin violates configuration, pending, or context bounds")
        use_retained = isinstance(hidden, ResidentLayerHidden)
        if use_retained:
            if hidden._cache is not self or hidden._generation != self._retained_generation:
                raise ValueError("resident hidden marker is stale or belongs to another request")
            x_ptr = None
        else:
            value = np.ascontiguousarray(np.asarray(hidden, dtype=np.int64))
            if value.shape != (1, self.d_model):
                raise ValueError(f"resident MoE layer begin expects hidden shape (1, {self.d_model})")
            x_ptr = value.ctypes.data
        n_used = self._moe_configs[layer][1]
        cold = np.empty(n_used, dtype=np.int32)
        cold_count = ctypes.c_int(0)
        inv_sqrt = int(round((1.0 / math.sqrt(self.head_dim)) * (1 << self.fa)))
        native_window = -1 if window is None else int(window)
        lib = _lib()
        with _CUDA_LOCK:
            rc = int(lib.qk_attention_bank_moe_begin(
                self.handle, layer, int(gain_handle), int(router_handle),
                int(q_handle), int(k_handle), int(v_handle), int(o_handle),
                x_ptr, int(use_retained), int(fw), int(eps), native_window, int(bool(rope)),
                inv_sqrt, cold.size, ctypes.byref(cold_count), cold.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ))
        if rc != 0:
            detail = "exact overflow guard fired" if rc == 3 else f"native status {rc}"
            raise CudaContextError(f"resident MoE layer begin failed for layer {layer}: {detail}")
        count = int(cold_count.value)
        if not 0 <= count <= n_used:
            raise CudaContextError("resident MoE layer returned an invalid cold-route count")
        self._lengths[layer] += 1
        self._pending_moe_layer = layer
        return tuple(int(expert) for expert in cold[:count])

    def continue_moe_layer(self, layer: int, fw: int, *, publish: bool = False):
        """Run selected MoE + residual addition, retaining output unless published."""
        if self._closed:
            raise CudaContextError("resident attention context is closed")
        layer = int(layer)
        if self._pending_moe_layer != layer or layer not in self._moe_configs:
            raise ValueError("resident MoE continuation has no matching prepared layer")
        out = np.empty((1, self.d_model), dtype=np.int64) if publish else None
        out_ptr = None if out is None else out.ctypes.data
        capacity = 0 if out is None else out.size
        lib = _lib()
        with _CUDA_LOCK:
            rc = int(lib.qk_attention_bank_moe_continue(
                self.handle, layer, int(fw), int(bool(publish)), capacity, out_ptr,
            ))
        if rc != 0:
            detail = "selected cold experts remain unbound" if rc == 4 else \
                ("exact overflow guard fired" if rc == 3 else f"native status {rc}")
            raise CudaContextError(f"resident MoE continuation failed for layer {layer}: {detail}")
        self._pending_moe_layer = None
        self._retained_generation += 1
        return out if publish else ResidentLayerHidden(self, self._retained_generation)

    def export_moe_hidden(self) -> np.ndarray:
        """Explicitly publish the current retained residual for fallback/debugging."""
        if self._closed or self._pending_moe_layer is not None:
            raise CudaContextError("resident MoE hidden cannot be exported from this context state")
        out = np.empty((1, self.d_model), dtype=np.int64)
        lib = _lib()
        with _CUDA_LOCK:
            rc = int(lib.qk_attention_bank_moe_export(self.handle, out.size, out.ctypes.data))
        if rc != 0:
            raise CudaContextError("resident MoE hidden export failed")
        return out

    def workspace_bytes(self) -> int:
        """Return device bytes owned by this request bank (not weights/global scratch)."""
        if self._closed:
            raise CudaContextError("resident attention context is closed")
        lib = _lib()
        with _CUDA_LOCK:
            value = int(lib.qk_attention_bank_workspace_bytes(self.handle))
        if value == (1 << 64) - 1:
            raise CudaContextError("resident attention workspace accounting failed")
        return value

    def reset(self) -> None:
        if self._closed:
            raise CudaContextError("resident attention context is closed")
        lib = _lib()
        with _CUDA_LOCK:
            rc = int(lib.qk_attention_bank_reset(self.handle))
        if rc != 0:
            raise CudaContextError("resident attention context reset failed")
        self._lengths[:] = [0] * self.n_layers
        self._retained_generation += 1
        self._pending_moe_layer = None
        self._token_executor = None

    def close(self) -> None:
        if not self._closed:
            lib = _lib()
            if lib is not None and hasattr(lib, "qk_attention_bank_destroy"):
                with _CUDA_LOCK:
                    lib.qk_attention_bank_destroy(self.handle)
            self._closed = True
            self._pending_moe_layer = None
            self._retained_generation += 1
            self._token_executor = None


@dataclass(frozen=True)
class ResidentMoeLayerSpec:
    """Resident dense-handle inputs for one M=1 MoE layer."""

    layer: int
    gain_handle: int
    router_handle: int
    q_handle: int
    k_handle: int
    v_handle: int
    o_handle: int
    window: int | None
    rope: bool


class ResidentMoeTokenExecutor:
    """Fail-closed retained token chain over consecutive MoE layers.

    ``lookup_expert`` must be side-effect free and return an already-resident
    ``(gate, up, down)`` handle triplet or ``None``. ``load_expert`` is called
    only for IDs compacted by the device as unbound. Any error poisons this
    executor; callers must destroy/reset the request bank rather than combine
    partially advanced K/V state with a fallback path.
    """

    def __init__(self, cache: ResidentAttentionCache, *, first_layer: int, layer_count: int,
                 n_experts: int, n_used: int, d_model: int, expert_ffn: int,
                 fw: int, eps: int, lookup_expert, load_expert):
        if not isinstance(cache, ResidentAttentionCache) or cache._closed:
            raise ValueError("resident token executor requires an open attention bank")
        values = tuple(int(value) for value in (
            first_layer, layer_count, n_experts, n_used, d_model, expert_ffn, fw, eps,
        ))
        first_layer, layer_count, n_experts, n_used, d_model, expert_ffn, fw, eps = values
        if first_layer < 0 or layer_count <= 0 or first_layer + layer_count > cache.n_layers or \
                n_experts <= 0 or not 0 < n_used <= n_experts or d_model != cache.d_model or \
                d_model % 256 or expert_ffn <= 0 or expert_ffn % 256 or \
                not 0 <= fw <= 62 or not 0 <= eps <= (1 << 64) - 1:
            raise ValueError("invalid resident token executor dimensions")
        if not callable(lookup_expert) or not callable(load_expert):
            raise TypeError("resident token executor requires expert lookup and lazy-load callables")
        self.cache = cache
        self.first_layer, self.layer_count = first_layer, layer_count
        self.n_experts, self.n_used = n_experts, n_used
        self.d_model, self.expert_ffn = d_model, expert_ffn
        self.fw, self.eps = fw, eps
        self.lookup_expert, self.load_expert = lookup_expert, load_expert
        self._configured = set()
        self._scanned = set()
        self._bound = set()
        self._poisoned = False

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @staticmethod
    def _handle_triplet(value, *, allow_none: bool, label: str):
        if value is None and allow_none:
            return None
        try:
            handles = tuple(int(handle) for handle in value)
        except (TypeError, ValueError):
            handles = ()
        if len(handles) != 3 or any(handle < 0 for handle in handles):
            raise CudaContextError(f"{label} did not return a valid gate/up/down handle triplet")
        return handles

    def _configure_and_bind_known(self, layer: int) -> None:
        if layer not in self._configured:
            self.cache.configure_moe_layer(
                layer, self.n_experts, self.n_used, self.d_model, self.expert_ffn,
            )
            self._configured.add(layer)
        if layer in self._scanned:
            return
        for expert in range(self.n_experts):
            handles = self._handle_triplet(
                self.lookup_expert(layer, expert), allow_none=True,
                label=f"resident expert lookup for layer {layer}, expert {expert}",
            )
            if handles is not None:
                self.cache.bind_moe_expert(layer, expert, *handles)
                self._bound.add((layer, expert))
        self._scanned.add(layer)

    def _validated_specs(self, layers):
        specs = tuple(layers)
        expected = tuple(range(self.first_layer, self.first_layer + self.layer_count))
        if len(specs) != self.layer_count or any(not isinstance(spec, ResidentMoeLayerSpec) for spec in specs) or \
                tuple(int(spec.layer) for spec in specs) != expected:
            raise ValueError(f"resident token executor requires consecutive layers {expected[0]}..{expected[-1]}")
        return specs

    def prepare(self, layers) -> tuple[ResidentMoeLayerSpec, ...]:
        """Configure bounded layer metadata and bind known weights before K/V mutation."""
        if self._poisoned:
            raise CudaContextError("resident token executor is poisoned; destroy or reset the request bank")
        specs = self._validated_specs(layers)
        try:
            for spec in specs:
                self._configure_and_bind_known(int(spec.layer))
            return specs
        except Exception:
            self._poisoned = True
            raise

    def run(self, hidden, layers) -> np.ndarray:
        if self._poisoned:
            raise CudaContextError("resident token executor is poisoned; destroy or reset the request bank")
        specs = self.prepare(layers)
        value = np.ascontiguousarray(np.asarray(hidden, dtype=np.int64))
        if value.shape != (1, self.d_model):
            raise ValueError(f"resident token executor expects hidden shape (1, {self.d_model})")
        state = value
        try:
            for index, spec in enumerate(specs):
                layer = int(spec.layer)
                self._configure_and_bind_known(layer)
                cold = self.cache.begin_moe_layer(
                    layer, int(spec.gain_handle), int(spec.router_handle),
                    int(spec.q_handle), int(spec.k_handle), int(spec.v_handle), int(spec.o_handle),
                    state, self.fw, self.eps, spec.window, bool(spec.rope),
                )
                if len(cold) != len(set(cold)) or any(not 0 <= expert < self.n_experts for expert in cold):
                    raise CudaContextError(f"resident layer {layer} returned invalid cold expert IDs")
                for expert in cold:
                    handles = self._handle_triplet(
                        self.load_expert(layer, expert), allow_none=False,
                        label=f"lazy expert loader for layer {layer}, expert {expert}",
                    )
                    self.cache.bind_moe_expert(layer, expert, *handles)
                    self._bound.add((layer, expert))
                state = self.cache.continue_moe_layer(
                    layer, self.fw, publish=index + 1 == self.layer_count,
                )
            if not isinstance(state, np.ndarray) or state.shape != (1, self.d_model):
                raise CudaContextError("resident token executor did not publish its final residual")
            return state
        except Exception:
            self._poisoned = True
            raise


def profile_reset(*, enabled: bool = True) -> bool:
    """Reset and enable/disable process-global CUDA boundary telemetry."""
    lib = _lib()
    if lib is None or not hasattr(lib, "qk_profile_reset"):
        return False
    with _CUDA_LOCK:
        lib.qk_profile_set_enabled(0)
        lib.qk_profile_reset()
        lib.qk_profile_set_enabled(int(bool(enabled)))
    return True


def profile_set_enabled(enabled: bool) -> bool:
    lib = _lib()
    if lib is None or not hasattr(lib, "qk_profile_set_enabled"):
        return False
    with _CUDA_LOCK:
        lib.qk_profile_set_enabled(int(bool(enabled)))
    return True


def profile_snapshot() -> dict | None:
    """Return native allocation/copy/call counters for the current profile window."""
    lib = _lib()
    if lib is None or not hasattr(lib, "qk_profile_snapshot"):
        return None
    values = (ctypes.c_uint64 * len(_PROFILE_METRICS))()
    with _CUDA_LOCK:
        required = int(lib.qk_profile_snapshot(values, len(values)))
    if required != len(_PROFILE_METRICS):
        raise RuntimeError(f"CUDA profile metric ABI mismatch: native={required}, python={len(_PROFILE_METRICS)}")
    return {name: int(values[i]) for i, name in enumerate(_PROFILE_METRICS)}


def free_all():
    lib = _lib()
    if lib is not None and hasattr(lib, "qk_free_all"):
        with _CUDA_LOCK:
            lib.qk_free_all()


def resident_count() -> int:
    lib = _lib()
    if lib is None or not hasattr(lib, "qk_resident_count"):
        return 0
    with _CUDA_LOCK:
        return int(lib.qk_resident_count())


def resident_capacity() -> int:
    """Current host metadata capacity of the growable resident registry (diagnostic only)."""
    lib = _lib()
    if lib is None or not hasattr(lib, "qk_resident_capacity"):
        return 0
    with _CUDA_LOCK:
        return int(lib.qk_resident_capacity())


def reserve_resident_capacity(count: int) -> bool:
    """Reserve registry metadata only; no weight bytes or VRAM are allocated."""
    count = int(count)
    if count < 0:
        raise ValueError("resident capacity must be non-negative")
    lib = _lib()
    if lib is None or not hasattr(lib, "qk_resident_reserve"):
        return False
    with _CUDA_LOCK:
        return lib.qk_resident_reserve(count) == 0


def moe_workspace_allocations() -> int | None:
    """Number of device-buffer grows since ``free_all``; ``None`` for older CUDA libraries."""
    lib = _lib()
    if lib is None or not hasattr(lib, "qk_moe_workspace_allocations"):
        return None
    with _CUDA_LOCK:
        return int(lib.qk_moe_workspace_allocations())


def moe_workspace_bytes() -> int | None:
    """Currently retained device bytes across the reusable MoE buffer roles."""
    lib = _lib()
    if lib is None or not hasattr(lib, "qk_moe_workspace_bytes"):
        return None
    with _CUDA_LOCK:
        return int(lib.qk_moe_workspace_bytes())


def release_moe_workspace() -> bool:
    """Release prefill-sized MoE scratch while preserving every resident weight handle."""
    lib = _lib()
    if lib is None or not hasattr(lib, "qk_moe_workspace_release"):
        return False
    with _CUDA_LOCK:
        lib.qk_moe_workspace_release()
    return True


def dp4a_available() -> bool:
    lib = _lib()
    return lib is not None and hasattr(lib, "qk_apply_resident_q6k_dp4a") and hasattr(lib, "qk_apply_resident_q4k_dp4a")


def _balanced_capacity(L: int) -> int:
    """Max |x| that L balanced base-256 digits (each in [-128, 127]) can represent EXACTLY via the greedy
    signed-low-byte decomposition = 127*(256^L - 1)/255. (256^L - 1 is always divisible by 255, so exact.)"""
    return 127 * ((256 ** L - 1) // 255)


def limbs_needed(maxabs: int) -> int:
    """Smallest L of balanced base-256 digits that represents ±|x| exactly, capped at 8.

    The greedy decomposition (make_limbs) keeps L signed digits and DISCARDS the final carry, so it is exact
    only when |x| <= 127*(256^L - 1)/255 — NOT the naive 2^(8L-1) bound, which over-claims a high band and
    silently corrupts the dot product (e.g. L=2 fails on [32640, 32767]: 32640 -> [-128,-128] = -32896 =
    x - 65536). Raises rather than truncating if |x| exceeds the 8-digit capacity."""
    m = int(maxabs)
    L = 1
    while L < 8 and m > _balanced_capacity(L):
        L += 1
    if m > _balanced_capacity(8):
        raise OverflowError(f"|x| max {m} exceeds 8 balanced base-256 digits ({_balanced_capacity(8)})")
    return L


def _activation_absmax(x: np.ndarray) -> int:
    """Return max(abs(x)) as a Python int without ``abs(INT64_MIN)`` wrapping."""
    return max(abs(int(x.min())), abs(int(x.max()))) if x.size else 0


def _dp4a_limb_count(x: np.ndarray, requested: int | None = None) -> int | None:
    """Validate the activation decomposition used by the CUDA DP4A kernels.

    The balanced-byte decomposition itself supports up to eight limbs, but the
    current kernels recombine each weighted limb in int64.  Four limbs are the
    largest envelope for which every Q4_K/Q6_K subgroup term is provably in
    range.  Return ``None`` above that envelope so callers use the exact int128
    kernel instead of silently wrapping.  An explicitly undersized/invalid limb
    count is a caller error and fails loudly.
    """
    try:
        needed = limbs_needed(_activation_absmax(x))
    except OverflowError:
        if requested is None:
            return None
        raise ValueError("activation is not representable by the DP4A balanced-byte decomposition") from None
    if requested is None:
        return needed if needed <= _DP4A_SAFE_LIMBS else None
    requested = int(requested)
    if not 1 <= requested <= _DP4A_SAFE_LIMBS:
        raise ValueError(
            f"DP4A limb count must be in [1, {_DP4A_SAFE_LIMBS}], got {requested}; "
            "larger decompositions are not int64-safe in the current CUDA recombination"
        )
    if requested < needed:
        raise ValueError(f"DP4A limb count {requested} is too small; activation requires {needed}")
    return requested


def apply_resident_dp4a(handle: int, x_int: np.ndarray, out_f: int, fw: int, qtype: int, ln: int = None):
    """DP4A apply of a resident Q4_K (qtype 0) or Q6_K (qtype 1) weight — byte-identical to apply_resident,
    faster on the big matmuls. ln (activation limbs) defaults to the minimum covering max|x|. [T,out_f] or None."""
    if qtype not in (Q4_K, Q6_K):
        raise ValueError(f"unknown qtype {qtype} (expected Q4_K={Q4_K} or Q6_K={Q6_K})")
    lib = _lib()
    fn = getattr(lib, "qk_apply_resident_q4k_dp4a" if qtype == Q4_K else "qk_apply_resident_q6k_dp4a", None)
    if lib is None or fn is None:
        return None
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_int, dtype=np.int64)))
    T = x.shape[0]
    ln = _dp4a_limb_count(x, ln)
    if ln is None:
        return None                                     # exact int128 fallback: outside DP4A's safe envelope
    out = np.empty((T, out_f), dtype=np.int64)
    with _CUDA_LOCK:
        rc = fn(int(handle), x.ctypes.data, T, int(fw), int(ln), out.ctypes.data)
    return None if rc != 0 else out


def apply_resident_q6k_dp4a(handle: int, x_int: np.ndarray, out_f: int, fw: int, ln: int = None):
    return apply_resident_dp4a(handle, x_int, out_f, fw, Q6_K, ln)


def moe_ffn_available() -> bool:
    lib = _lib()
    return lib is not None and hasattr(lib, "qk_moe_ffn")


def moe_ffn_dp4a_available() -> bool:
    lib = _lib()
    return lib is not None and hasattr(lib, "qk_moe_ffn_dp4a")


def moe_ffn_batched_available() -> bool:
    lib = _lib()
    return lib is not None and hasattr(lib, "qk_moe_ffn_batched")


def moe_ffn_batched_dp4a_available() -> bool:
    lib = _lib()
    return lib is not None and hasattr(lib, "qk_moe_ffn_batched_dp4a")


def moe_ffn_batched(gate_h, up_h, down_h, m: int, k: int, h, gates, d_model: int, e_ffn: int,
                    fa: int, fw: int, *, dp4a: bool = False):
    """Batched MoE over all m·k (token, selected-expert) pairs in one set of kernels (the prefill win — collapses
    m per-token qk_moe_ffn calls into ~6 launches). gate_h/up_h/down_h are flattened token-major (pair = t*k+j);
    h is [m, d_model]; gates is [m*k]. ``dp4a`` uses the exact guarded batched-DP4A entry when available and
    falls back inside the same locked call to the int128 entry if either h or the device-computed gate×up leaves
    its proven four-limb envelope. Returns [m, d_model] int64 or None on a genuine runtime failure."""
    lib = _lib()
    if lib is None or not hasattr(lib, "qk_moe_ffn_batched"):
        return None
    P = m * k
    if m <= 0 or k <= 0:
        raise ValueError(f"batched MoE requires positive m and k, got m={m}, k={k}")
    if len(gate_h) != P or len(up_h) != P or len(down_h) != P:
        raise ValueError(f"batched MoE requires exactly m*k={P} handles for each projection")
    arr = lambda hs: (ctypes.c_int64 * P)(*[int(x) for x in hs])
    hh = np.ascontiguousarray(np.asarray(h, np.int64).reshape(-1))
    gg = np.ascontiguousarray(np.asarray(gates, np.int64).reshape(-1))
    if hh.size != m * d_model or gg.size != P:
        raise ValueError(f"batched MoE expected h.size={m * d_model} and gates.size={P}, got {hh.size} and {gg.size}")
    out = np.empty((m, d_model), dtype=np.int64)
    with _CUDA_LOCK:
        args = (arr(gate_h), arr(up_h), arr(down_h), int(m), int(k), hh.ctypes.data, gg.ctypes.data,
                int(d_model), int(e_ffn), int(fa), int(fw), out.ctypes.data)
        use_dp4a = bool(dp4a and hasattr(lib, "qk_moe_ffn_batched_dp4a") and _dp4a_limb_count(hh) is not None)
        rc = lib.qk_moe_ffn_batched_dp4a(*args) if use_dp4a else lib.qk_moe_ffn_batched(*args)
        if use_dp4a and rc == 2:                    # exact envelope miss after device-computed gate×up
            rc = lib.qk_moe_ffn_batched(*args)
    return None if rc != 0 else out


def moe_ffn(gate_h, up_h, down_h, h, gates, d_model: int, e_ffn: int, fa: int, fw: int, dp4a: bool = False):
    """Fused batched MoE expert FFN on the GPU: out[d_model] = Σ_e gate_e·down_e(silu(gate_e·h)*up_e·h).
    gate_h/up_h/down_h: resident handles of the n_e selected experts. dp4a=True runs the DP4A path (byte-
    identical, faster — the per-token decode bulk). Returns [d_model] int64 or None."""
    lib = _lib()
    fn = getattr(lib, "qk_moe_ffn_dp4a" if dp4a else "qk_moe_ffn", None) if lib is not None else None
    if fn is None:
        return None
    n_e = len(gate_h)
    if n_e <= 0 or len(up_h) != n_e or len(down_h) != n_e:
        raise ValueError("MoE requires the same positive number of gate, up, and down handles")
    arr = lambda hs: (ctypes.c_int64 * n_e)(*[int(x) for x in hs])
    hh = np.ascontiguousarray(np.asarray(h, np.int64).reshape(-1))
    gg = np.ascontiguousarray(np.asarray(gates, np.int64).reshape(-1))
    if hh.size != d_model or gg.size != n_e:
        raise ValueError(f"MoE expected h.size={d_model} and gates.size={n_e}, got {hh.size} and {gg.size}")
    out = np.empty(int(d_model), dtype=np.int64)
    with _CUDA_LOCK:
        rc = fn(arr(gate_h), arr(up_h), arr(down_h), n_e, hh.ctypes.data, gg.ctypes.data,
                int(d_model), int(e_ffn), int(fa), int(fw), out.ctypes.data)
    return None if rc != 0 else out
