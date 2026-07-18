"""ctypes loader for the CPU integer kernel (fused Q4_K/Q6_K dequant + fixed-point matmul).

Mirrors the Bonsai `q1_native` pattern: loads `libnmc_qk.so` and returns None when absent, so callers fall back
to the numpy oracle. Every kernel result MUST be byte-identical to that oracle (tests/test_qk_native.py)."""
from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

Q4_K, Q6_K = 0, 1
_BLOCK_BYTES = {Q4_K: 144, Q6_K: 210}   # ggml superblock sizes (256 weights): Q4_K=144B, Q6_K=210B


def _require_weight_bytes(weight_raw, out_f: int, n_blocks: int, qtype: int) -> int:
    """Bytes the kernel will read = out_f*n_blocks*block_bytes. Validate the buffer is at least that long, so
    a truncated/corrupt GGUF (gguf.read_raw does no length check) fails loud instead of an OOB heap read."""
    bs = _BLOCK_BYTES.get(int(qtype))
    if bs is None:
        raise ValueError(f"unknown qtype {qtype} (expected Q4_K={Q4_K} or Q6_K={Q6_K})")
    out_f, n_blocks = int(out_f), int(n_blocks)
    if out_f <= 0 or n_blocks <= 0:
        raise ValueError(f"out_f and n_blocks must be positive, got out_f={out_f}, n_blocks={n_blocks}")
    need = out_f * n_blocks * bs
    if len(weight_raw) < need:
        raise ValueError(f"weight buffer too short: {len(weight_raw)} bytes < required {need} "
                         f"(out_f={out_f}, n_blocks={n_blocks}, qtype={qtype}) — truncated or corrupt GGUF")
    return need


def _so_path() -> Path:
    env = os.environ.get("BONSAI_BIN_DIR")
    home = os.environ.get("BONSAI_NOTARY_HOME") or str(Path.home() / ".local/integer_inference_engine/north-mini-code")
    cands = [Path(env) / "libnmc_qk.so"] if env else []
    cands += [Path(home) / "bin" / "libnmc_qk.so", Path(__file__).resolve().parents[2] / "tools" / "libnmc_qk.so"]
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
        lib.qk_linear  # noqa: B018 — probe symbol
    except (OSError, AttributeError):
        return None
    lib.qk_linear.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                              ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.qk_linear.restype = None
    return lib


def available() -> bool:
    return _lib() is not None


def qk_linear(weight_raw: bytes, x_int: np.ndarray, out_f: int, n_blocks: int, fw: int, qtype: int):
    """y[t,o] = (Σ_i W_fixed[o,i]·x[t,i]) >> fw, fused from raw Q4_K/Q6_K weight bytes. x_int [T, in_f] int64.
    Returns y [T, out_f] int64, or None if the kernel is unavailable (caller falls back to numpy)."""
    lib = _lib()
    if lib is None:
        return None
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_int, dtype=np.int64)))
    T, in_f = x.shape
    assert in_f == n_blocks * 256, (in_f, n_blocks)
    _require_weight_bytes(weight_raw, out_f, n_blocks, qtype)   # fail loud on a short buffer (no OOB read)
    wbuf = (ctypes.c_char * len(weight_raw)).from_buffer_copy(weight_raw)
    out = np.empty((T, out_f), dtype=np.int64)
    lib.qk_linear(ctypes.cast(wbuf, ctypes.c_void_p), x.ctypes.data, T, out_f, n_blocks, int(fw), int(qtype),
                  out.ctypes.data)
    return out
