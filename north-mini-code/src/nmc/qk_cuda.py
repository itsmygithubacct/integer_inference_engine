"""ctypes loader for the CUDA integer kernel (Q4_K/Q6_K fused dequant + fixed-point matmul).

Per-host, arch-specific; returns None / available()=False when the .so or a usable GPU is absent, so callers
fall back to the CPU kernel (qk_native) or the numpy oracle. Every result MUST be byte-identical to the CPU
oracle — the GPU is a producer, the CPU oracle the canonical verifier (tests/test_qk_cuda.py)."""
from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

Q4_K, Q6_K = 0, 1


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
        lib.qk_linear_cuda; lib.qk_cuda_available  # noqa: B018 — probe symbols
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
        lib.qk_apply_resident.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p]
        lib.qk_apply_resident.restype = ctypes.c_int
        lib.qk_free_all.restype = None
        lib.qk_resident_count.restype = ctypes.c_int64
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
        if lib.qk_cuda_available() != 0:
            return None                               # lib loaded but no usable GPU -> fall back
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
    wbuf = (ctypes.c_char * len(weight_raw)).from_buffer_copy(weight_raw)
    out = np.empty((T, out_f), dtype=np.int64)
    rc = lib.qk_linear_cuda(ctypes.cast(wbuf, ctypes.c_void_p), x.ctypes.data, T, out_f, n_blocks,
                            int(fw), int(qtype), out.ctypes.data)
    return None if rc != 0 else out


def resident_available() -> bool:
    lib = _lib()
    return lib is not None and hasattr(lib, "qk_register_weight")


def register_weight(weight_raw: bytes, out_f: int, n_blocks: int, qtype: int):
    """Upload one weight tensor's raw bytes to VRAM ONCE; return a handle (>=0) or None. Reused across applies."""
    lib = _lib()
    if lib is None or not hasattr(lib, "qk_register_weight"):
        return None
    wbuf = (ctypes.c_char * len(weight_raw)).from_buffer_copy(weight_raw)
    h = lib.qk_register_weight(ctypes.cast(wbuf, ctypes.c_void_p), out_f, n_blocks, qtype)
    return None if h < 0 else int(h)


def apply_resident(handle: int, x_int: np.ndarray, out_f: int, fw: int):
    """Apply a resident weight: y[t,o]=(Σ_i W_fixed[o,i]·x[t,i])>>fw. Only x crosses PCIe. y [T,out_f] int64."""
    lib = _lib()
    if lib is None:
        return None
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_int, dtype=np.int64)))
    T = x.shape[0]
    out = np.empty((T, out_f), dtype=np.int64)
    rc = lib.qk_apply_resident(int(handle), x.ctypes.data, T, int(fw), out.ctypes.data)
    return None if rc != 0 else out


def free_all():
    lib = _lib()
    if lib is not None and hasattr(lib, "qk_free_all"):
        lib.qk_free_all()


def resident_count() -> int:
    lib = _lib()
    return int(lib.qk_resident_count()) if lib is not None and hasattr(lib, "qk_resident_count") else 0


def dp4a_available() -> bool:
    lib = _lib()
    return lib is not None and hasattr(lib, "qk_apply_resident_q6k_dp4a") and hasattr(lib, "qk_apply_resident_q4k_dp4a")


def limbs_needed(maxabs: int) -> int:
    """Smallest L of balanced base-256 digits that represents |x|<=maxabs exactly (2^(8L-1) > maxabs), capped 8."""
    L = 1
    while L < 8 and (1 << (8 * L - 1)) <= int(maxabs):
        L += 1
    return L


def apply_resident_dp4a(handle: int, x_int: np.ndarray, out_f: int, fw: int, qtype: int, ln: int = None):
    """DP4A apply of a resident Q4_K (qtype 0) or Q6_K (qtype 1) weight — byte-identical to apply_resident,
    faster on the big matmuls. ln (activation limbs) defaults to the minimum covering max|x|. [T,out_f] or None."""
    lib = _lib()
    fn = getattr(lib, "qk_apply_resident_q4k_dp4a" if qtype == Q4_K else "qk_apply_resident_q6k_dp4a", None)
    if lib is None or fn is None:
        return None
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_int, dtype=np.int64)))
    T = x.shape[0]
    if ln is None:
        ln = limbs_needed(int(np.abs(x).max()) if x.size else 0)
    out = np.empty((T, out_f), dtype=np.int64)
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
    arr = lambda hs: (ctypes.c_int64 * P)(*[int(x) for x in hs])
    hh = np.ascontiguousarray(np.asarray(h, np.int64).reshape(-1))
    gg = np.ascontiguousarray(np.asarray(gates, np.int64).reshape(-1))
    out = np.empty((m, d_model), dtype=np.int64)
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
    arr = lambda hs: (ctypes.c_int64 * n_e)(*[int(x) for x in hs])
    hh = np.ascontiguousarray(np.asarray(h, np.int64).reshape(-1))
    gg = np.ascontiguousarray(np.asarray(gates, np.int64).reshape(-1))
    out = np.empty(int(d_model), dtype=np.int64)
    rc = fn(arr(gate_h), arr(up_h), arr(down_h), n_e, hh.ctypes.data, gg.ctypes.data,
            int(d_model), int(e_ffn), int(fa), int(fw), out.ctypes.data)
    return None if rc != 0 else out
