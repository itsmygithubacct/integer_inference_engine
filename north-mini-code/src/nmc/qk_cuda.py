"""ctypes loader for the CUDA integer kernel (Q4_K/Q6_K fused dequant + fixed-point matmul).

Per-host, arch-specific; returns None / available()=False when the .so or a usable GPU is absent, so callers
fall back to the CPU kernel (qk_native) or the numpy oracle. Every result MUST be byte-identical to the CPU
oracle — the GPU is a producer, the CPU oracle the canonical verifier (tests/test_qk_cuda.py)."""
from __future__ import annotations

import ctypes
import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from nmc.qk_native import _require_weight_bytes   # shared Q4_K/Q6_K superblock buffer-length guard

Q4_K, Q6_K = 0, 1
_DP4A_SAFE_LIMBS = 4
_CUDA_ABI_VERSION = 2
# The resident registry and reusable activation workspaces are process-global
# in the CUDA library.  CDLL calls release the GIL, so the bridge must enforce
# the single-caller contract rather than merely documenting it.
_CUDA_LOCK = threading.RLock()
_REQUIRED_CUDA_SYMBOLS = (
    "qk_cuda_abi_version",
    "qk_linear_cuda",
    "qk_cuda_available",
    "qk_register_weight",
    "qk_resident_reserve",
    "qk_apply_resident",
    "qk_free_all",
    "qk_resident_count",
    "qk_resident_capacity",
    "qk_moe_workspace_allocations",
    "qk_moe_workspace_bytes",
    "qk_moe_workspace_release",
    "qk_moe_ffn",
    "qk_moe_ffn_batched",
)


class CudaRegistrationError(RuntimeError):
    """A resident weight could not be committed to the CUDA registry/VRAM."""


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
        lib.qk_resident_reserve.argtypes = [ctypes.c_int64]
        lib.qk_resident_reserve.restype = ctypes.c_int
        lib.qk_apply_resident.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p]
        lib.qk_apply_resident.restype = ctypes.c_int
        lib.qk_free_all.restype = None
        lib.qk_resident_count.restype = ctypes.c_int64
        if hasattr(lib, "qk_resident_capacity"):
            lib.qk_resident_capacity.restype = ctypes.c_int64
        if hasattr(lib, "qk_moe_workspace_allocations"):
            lib.qk_moe_workspace_allocations.restype = ctypes.c_int64
        lib.qk_moe_workspace_bytes.restype = ctypes.c_uint64
        lib.qk_moe_workspace_release.restype = None
        P64 = ctypes.POINTER(ctypes.c_int64)
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


def moe_ffn_batched(gate_h, up_h, down_h, m: int, k: int, h, gates, d_model: int, e_ffn: int, fa: int, fw: int):
    """Batched MoE over all m·k (token, selected-expert) pairs in one set of kernels (the prefill win — collapses
    m per-token qk_moe_ffn calls into ~6 launches). gate_h/up_h/down_h are flattened token-major (pair = t*k+j);
    h is [m, d_model]; gates is [m*k]. Byte-identical to per-token moe_ffn. Returns [m, d_model] int64 or None."""
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
        rc = lib.qk_moe_ffn_batched(arr(gate_h), arr(up_h), arr(down_h), int(m), int(k), hh.ctypes.data,
                                    gg.ctypes.data, int(d_model), int(e_ffn), int(fa), int(fw), out.ctypes.data)
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
