"""Optional CUDA Q1_0 linear kernel for Bonsai (per-host opt-in; default OFF).

Mirrors ``q1_native._load_lib``: loads ``tools/libbonsai_q1_gpu.so`` and degrades to "unavailable"
(returns ``None``) on a missing/stale/wrong-arch ``.so``, a missing CUDA runtime, or no usable GPU — so it
**never** breaks the committed CPU build. With the lib absent (the committed default — the GPU ``.so`` is
gitignored, per-host, arch-specific) every entry point here is a no-op that returns ``None`` and the caller
falls back to the CPU native/oracle path.

Contract: every result MUST be byte-identical to the int64 CPU oracle (``reference_bonsai.q1_linear_ref``); the
GPU is a *producer*, the CPU oracle the canonical *verifier*. The wrapper returns ``None`` on any integer
overflow / launch ``rc`` so a GPU hiccup degrades to CPU rather than aborting a notarized run. See
``research/bonsai-notary/IMPLEMENT-GPU-MODE.md`` for the build plan and the parity gate that must pass before
``--gpu`` is permitted to emit a receipt, and ``Q1-BITMATMUL-REFORMULATION.md`` for the kernel math.

NOTE: ``tools/bonsai_q1_gpu.cu`` (the actual kernel) is implemented and byte-checked against the CPU oracle.
The GPU ``.so`` is a per-host build artifact (gitignored) — build it with ``tools/build_bonsai_q1_gpu.sh``.
Until that ``libbonsai_q1_gpu.so`` is present on the host, ``gpu_available()`` is ``False`` and this module is
inert (callers transparently fall back to the CPU native/oracle path).
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..notary_paths import gpu_kernel_so

_ROOT = Path(__file__).resolve().parents[3]          # repo root (same parents[3] as q1_native._LIB)
_LIB = Path(gpu_kernel_so())   # prefers ~/.local/trinote/bin, falls back to <repo>/tools (back-compat)


@lru_cache(maxsize=1)
def _load_lib():
    """Load the GPU kernel lib, or return None (→ CPU fallback) if it is absent/stale/wrong-arch or no GPU."""
    if not _LIB.exists():
        return None
    try:
        lib = ctypes.CDLL(str(_LIB))
    except OSError:
        return None                                   # no CUDA runtime / wrong arch -> unavailable
    try:
        fn = lib.bonsai_q1_linear_gpu
    except AttributeError:
        return None                                   # stale/partial .so missing the core symbol
    fn.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                  # x_fp(int64), bits(uint8), scale(int64)
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,     # tokens, out_features, n_blocks, frac
        ctypes.c_void_p,                                                    # out(int64)
    ]
    fn.restype = ctypes.c_int
    # DP4A Q1 apply (compute lever; optional, guarded).
    try:
        dp4a = lib.bonsai_q1_linear_dp4a_gpu
    except AttributeError:
        pass
    else:
        dp4a.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,              # x, bits, scale
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, # tokens, out_f, n_blocks, frac
            ctypes.c_int, ctypes.c_void_p,                                  # L, out
        ]
        dp4a.restype = ctypes.c_int
    # Weight-residency API (optional; guarded so an older .so without these symbols still loads).
    try:
        reg = lib.bonsai_q1_register_weight
        ap = lib.bonsai_q1_apply_resident
        free = lib.bonsai_q1_free_weights
    except AttributeError:
        pass
    else:
        reg.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]  # bits, scale, out_f, n_blocks
        reg.restype = ctypes.c_int64                                       # handle (>=0) or -1
        ap.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]  # handle, x, tokens, frac, out
        ap.restype = ctypes.c_int
        free.argtypes = []
        free.restype = None
        try:
            reg32 = lib.bonsai_q1_register_weight_i32
        except AttributeError:
            pass
        else:
            reg32.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64
            ]
            reg32.restype = ctypes.c_int64
        try:
            reg32_gpu = lib.bonsai_q1_register_weight_i32_gpu_layout
        except AttributeError:
            pass
        else:
            reg32_gpu.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64
            ]
            reg32_gpu.restype = ctypes.c_int64
        try:
            reg32_bmma = lib.bonsai_q1_register_weight_i32_bmma
        except AttributeError:
            pass
        else:
            reg32_bmma.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64
            ]
            reg32_bmma.restype = ctypes.c_int64
        try:
            mem = lib.bonsai_gpu_mem_info
            weight_bytes = lib.bonsai_q1_resident_weight_bytes
        except AttributeError:
            pass
        else:
            mem.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            mem.restype = ctypes.c_int
            weight_bytes.argtypes = []
            weight_bytes.restype = ctypes.c_uint64
        try:
            reserve = lib.bonsai_gpu_reservation_create
            release = lib.bonsai_gpu_reservation_free
        except AttributeError:
            pass
        else:
            reserve.argtypes = [
                ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
            ]
            reserve.restype = ctypes.c_int64
            release.argtypes = [ctypes.c_int64]
            release.restype = None
    # M2 RMSNorm (optional; guarded so an older .so without it still loads).
    try:
        rms = lib.bonsai_rmsnorm_gpu
    except AttributeError:
        pass
    else:
        rms.argtypes = [
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # x, rows, cols, frac, eps
            ctypes.c_void_p, ctypes.c_void_p,                                                  # gain (nullable), out
        ]
        rms.restype = ctypes.c_int
    # M3 prefill attention (optional; guarded).
    try:
        attp = lib.bonsai_attention_prefill_gpu
    except AttributeError:
        pass
    else:
        attp.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                  # q, k, v
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,                     # H, Hkv, hd
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,                     # M, L, start
            ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,                    # frac, inv_sqrt_fp, out
        ]
        attp.restype = ctypes.c_int
    # M3 monolith: resident buffer upload + on-device prefill forward (optional; guarded).
    try:
        bufup = lib.bonsai_buf_upload_i64
        mono = lib.bonsai_prefill_forward_gpu
    except AttributeError:
        pass
    else:
        bufup.argtypes = [ctypes.c_void_p, ctypes.c_int64]                      # host int64 ptr, n
        bufup.restype = ctypes.c_int64                                          # handle or -1
        mono.argtypes = [
            ctypes.c_void_p,                                                    # x_embed
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,                     # T, d, n_layers
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,     # H, Hkv, hd, dff
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,                     # frac, eps, inv_sqrt_fp
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, # wq,wk,wv,wo handle arrays
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                  # w1,wu,w2 handle arrays
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, # n1g,n2g,qng,kng handle arrays
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,     # finalg, out_head, cos_h, sin_h
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                  # out_logits, out_k (nullable), out_v (nullable)
        ]
        mono.restype = ctypes.c_int
    # Optional device probe: a real .so should export this and return 0 only if a usable GPU exists.
    try:
        probe = lib.bonsai_gpu_available
    except AttributeError:
        pass
    else:
        probe.restype = ctypes.c_int
        try:
            if probe() != 0:
                return None                            # CUDA present but no usable device -> CPU fallback
        except OSError:
            return None
    return lib


def gpu_available() -> bool:
    """True iff the GPU Q1 kernel lib loaded and a usable GPU is present."""
    return _load_lib() is not None


def q1_apply_gpu(x_fp: np.ndarray, bits: np.ndarray, scale_fp: np.ndarray, frac: int) -> "np.ndarray | None":
    """GPU packed-Q1_0 linear `x @ W.T`, byte-identical to ``q1_linear_ref`` / ``q1_linear_native``.

    Returns ``None`` when the GPU lib is unavailable OR a tile overflows the kernel's integer envelope
    (rc != 0), so the caller falls back to the CPU native/oracle path. Uses the committed int64 scale (the
    int32 scale-cache / lut32 are CPU-bandwidth tricks irrelevant on GPU)."""
    lib = _load_lib()
    if lib is None:
        return None
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_fp, dtype=np.int64)))
    b = np.ascontiguousarray(np.asarray(bits, dtype=np.uint8))
    src = np.asarray(scale_fp)
    if not np.issubdtype(src.dtype, np.integer):       # match _contiguous_q1_weight's loud reject
        raise TypeError(f"Q1_0 scale must be an integer dtype, got {src.dtype}")
    s = np.ascontiguousarray(src, dtype=np.int64)      # GPU kernel reads int64 scale (no int32 cache on GPU)
    out_f, n_blocks = s.shape
    if x.shape[1] != n_blocks * 128:
        raise ValueError(f"Q1_0 contraction mismatch: x has {x.shape[1]}, weight expects {n_blocks * 128}")
    out = np.empty((x.shape[0], out_f), dtype=np.int64)
    rc = lib.bonsai_q1_linear_gpu(
        x.ctypes.data, b.ctypes.data, s.ctypes.data,
        ctypes.c_int64(x.shape[0]), ctypes.c_int64(int(out_f)),
        ctypes.c_int64(int(n_blocks)), ctypes.c_int64(int(frac)),
        out.ctypes.data,
    )
    if rc != 0:                                        # overflow / launch failure -> CPU fallback (no raise)
        return None
    return out


_L4_LO, _L4_HI = -2155905152, 2139062143   # balanced base-256 L=4 envelope (Q1-BITMATMUL §4.4)


def q1_apply_dp4a_gpu(x_fp: np.ndarray, bits: np.ndarray, scale_fp: np.ndarray, frac: int,
                      L: "int | None" = None) -> "np.ndarray | None":
    """DP4A Q1 apply (int8 dp4a hot loop), byte-identical to q1_linear_ref. L=4 when the activations fit the
    balanced base-256 range (the committed envelope), else 8.

    Returns None if the kernel is unavailable or declines (rc != 0). An EXPLICIT L=4 with activations OUTSIDE
    the balanced base-256 envelope RAISES: the device range guard (range_guard_l4_kernel) is enforced host-side
    here, so an out-of-envelope L=4 can never silently wrap (was a real hazard — the finding's dead-guard)."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_q1_linear_dp4a_gpu"):
        return None
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_fp, dtype=np.int64)))
    b = np.ascontiguousarray(np.asarray(bits, dtype=np.uint8))
    s = np.ascontiguousarray(np.asarray(scale_fp, dtype=np.int64))
    out_f, n_blocks = s.shape
    if x.shape[1] != n_blocks * 128:
        raise ValueError(f"Q1_0 contraction mismatch: x has {x.shape[1]}, weight expects {n_blocks * 128}")
    if L is None:
        mn, mx = int(x.min()), int(x.max())
        L = 4 if (_L4_LO <= mn and mx <= _L4_HI) else 8   # else L=8 (always exact)
    elif int(L) == 4 and x.size:
        mn, mx = int(x.min()), int(x.max())
        if not (_L4_LO <= mn and mx <= _L4_HI):
            raise ValueError(
                f"q1_apply_dp4a_gpu: explicit L=4 requires activations in [{_L4_LO}, {_L4_HI}], got [{mn}, {mx}] "
                f"— the L=4 digits would not reconstruct x and the kernel would silently wrap (rc=0). "
                f"Use L=8 (always exact) or L=None (auto-selects a safe L).")
    out = np.empty((x.shape[0], out_f), dtype=np.int64)
    rc = lib.bonsai_q1_linear_dp4a_gpu(
        x.ctypes.data, b.ctypes.data, s.ctypes.data,
        ctypes.c_int64(x.shape[0]), ctypes.c_int64(int(out_f)),
        ctypes.c_int64(int(n_blocks)), ctypes.c_int64(int(frac)), ctypes.c_int(int(L)), out.ctypes.data,
    )
    if rc != 0:
        return None
    return out


def residency_available() -> bool:
    """True iff the loaded .so exposes the weight-residency API (register/apply_resident)."""
    lib = _load_lib()
    return lib is not None and hasattr(lib, "bonsai_q1_register_weight")


def q1_register_weight(bits: np.ndarray, scale_fp: np.ndarray, *,
                       gpu_coalesced: bool = False,
                       gpu_bmma: bool = False) -> "int | None":
    """Upload a projection's bits+scale to the device ONCE; return an opaque handle (>=0), or None on
    failure / unavailability. Reuse the handle with ``q1_apply_resident`` to skip the per-call weight upload
    (the dominant decode cost). The committed int64 scale is used (no on-GPU int32 cache)."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_q1_register_weight"):
        return None
    b = np.ascontiguousarray(np.asarray(bits, dtype=np.uint8))
    src = np.asarray(scale_fp)
    if not np.issubdtype(src.dtype, np.integer):
        raise TypeError(f"Q1_0 scale must be an integer dtype, got {src.dtype}")
    # Qwen3.5 artifacts commit losslessly narrowed int32 scales.  Preserve that
    # representation when the CUDA library exposes the scale32 ABI: expanding
    # it here would consume ~800 MiB of otherwise avoidable RTX 3070 memory.
    use_i32 = src.dtype.itemsize <= 4 and hasattr(lib, "bonsai_q1_register_weight_i32")
    if gpu_coalesced and gpu_bmma:
        raise ValueError("choose one GPU Q1 runtime layout")
    if (gpu_coalesced or gpu_bmma) and not use_i32:
        raise ValueError("GPU-coalesced Q1 registration currently requires committed int32 scales")
    if gpu_coalesced and not hasattr(lib, "bonsai_q1_register_weight_i32_gpu_layout"):
        return None
    if gpu_bmma and not hasattr(lib, "bonsai_q1_register_weight_i32_bmma"):
        return None
    s = np.ascontiguousarray(src, dtype=np.int32 if use_i32 else np.int64)
    out_f, n_blocks = s.shape
    # keep refs alive only for the duration of the call; the device copy is what persists
    if gpu_bmma:
        fn = lib.bonsai_q1_register_weight_i32_bmma
    elif gpu_coalesced:
        fn = lib.bonsai_q1_register_weight_i32_gpu_layout
    else:
        fn = lib.bonsai_q1_register_weight_i32 if use_i32 else lib.bonsai_q1_register_weight
    h = fn(b.ctypes.data, s.ctypes.data,
           ctypes.c_int64(int(out_f)), ctypes.c_int64(int(n_blocks)))
    return None if int(h) < 0 else int(h)


def gpu_memory_info() -> "dict[str, int] | None":
    """Return allocator-visible CUDA memory, or ``None`` on an older/unavailable library.

    ``resident_weight_bytes`` is tracked from successful live registrations,
    which makes the Qwen3.5 memory-feasibility report auditable independently
    of CUDA context and allocator overhead.
    """
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_gpu_mem_info"):
        return None
    free = ctypes.c_uint64()
    total = ctypes.c_uint64()
    if lib.bonsai_gpu_mem_info(ctypes.byref(free), ctypes.byref(total)) != 0:
        return None
    weight_bytes = 0
    if hasattr(lib, "bonsai_q1_resident_weight_bytes"):
        weight_bytes = int(lib.bonsai_q1_resident_weight_bytes())
    return {
        "free_bytes": int(free.value),
        "total_bytes": int(total.value),
        "used_bytes": int(total.value - free.value),
        "resident_weight_bytes": weight_bytes,
    }


def gpu_reservation_create(component_bytes) -> "tuple[int, int] | None":
    """Reserve several exact device allocations as one failure-atomic probe.

    Returns ``(handle, allocated_bytes)`` on success.  A failed reservation is
    fully unwound inside CUDA and returns ``None``; no partial allocations are
    left behind.  This API is intentionally low-level and is used by the 27B
    memory-feasibility tool rather than by inference itself.
    """
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_gpu_reservation_create"):
        return None
    sizes = np.ascontiguousarray(np.asarray(tuple(component_bytes), dtype=np.uint64))
    if sizes.ndim != 1 or sizes.size == 0 or np.any(sizes == 0):
        raise ValueError("GPU reservation components must be a non-empty list of positive byte counts")
    allocated = ctypes.c_uint64()
    handle = lib.bonsai_gpu_reservation_create(
        sizes.ctypes.data, ctypes.c_int64(int(sizes.size)), ctypes.byref(allocated)
    )
    if int(handle) < 0:
        return None
    return int(handle), int(allocated.value)


def gpu_reservation_free(handle: int) -> None:
    """Release a feasibility reservation; safe to call more than once."""
    lib = _load_lib()
    if lib is not None and hasattr(lib, "bonsai_gpu_reservation_free"):
        lib.bonsai_gpu_reservation_free(ctypes.c_int64(int(handle)))


def q1_apply_resident(handle: int, x_fp: np.ndarray, out_features: int, n_blocks: int,
                      frac: int) -> "np.ndarray | None":
    """Apply a registered weight (``handle``) to fresh activations — byte-identical to ``q1_apply_gpu`` /
    the oracle, but uploads only x and downloads only out (the weight is resident). Returns None on failure
    so the caller falls back to the CPU path."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_q1_apply_resident"):
        return None
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_fp, dtype=np.int64)))
    if x.shape[1] != int(n_blocks) * 128:
        raise ValueError(f"Q1_0 contraction mismatch: x has {x.shape[1]}, weight expects {int(n_blocks) * 128}")
    out = np.empty((x.shape[0], int(out_features)), dtype=np.int64)
    rc = lib.bonsai_q1_apply_resident(ctypes.c_int64(int(handle)), x.ctypes.data,
                                      ctypes.c_int64(x.shape[0]), ctypes.c_int64(int(frac)), out.ctypes.data)
    if rc != 0:
        return None
    return out


def q1_free_weights() -> None:
    """Free all resident device weights (optional; process exit frees them regardless)."""
    lib = _load_lib()
    if lib is not None and hasattr(lib, "bonsai_q1_free_weights"):
        lib.bonsai_q1_free_weights()


def rmsnorm_gpu(x_fp: np.ndarray, frac: int, eps: int = 1,
                gain_q: "np.ndarray | None" = None) -> "np.ndarray | None":
    """GPU fixed-point RMSNorm, byte-identical to ``fixed_point_rmsnorm`` / ``bonsai_rmsnorm_i64``.
    Returns ``None`` when the GPU lib is unavailable OR any row overflows 128 bits / leaves the gain envelope
    (rc 4 → caller falls back to the CPU big-int oracle, which raises or computes exactly)."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_rmsnorm_gpu"):
        return None
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_fp, dtype=np.int64)))
    rows, cols = x.shape
    out = np.empty((rows, cols), dtype=np.int64)
    g_ptr = 0
    g = None
    if gain_q is not None:
        g = np.ascontiguousarray(np.asarray(gain_q, dtype=np.int64))
        if g.shape[0] != cols:
            raise ValueError(f"RMSNorm gain has {g.shape[0]} elems, expected cols={cols}")
        g_ptr = g.ctypes.data
    rc = lib.bonsai_rmsnorm_gpu(
        x.ctypes.data, ctypes.c_int64(rows), ctypes.c_int64(cols),
        ctypes.c_int64(int(frac)), ctypes.c_int64(int(eps)),
        ctypes.c_void_p(g_ptr), out.ctypes.data,
    )
    if rc != 0:
        return None
    return out


def attention_prefill_gpu(q_fp: np.ndarray, k_fp: np.ndarray, v_fp: np.ndarray,
                          start: int, frac: int, inv_sqrt_fp: int) -> "np.ndarray | None":
    """GPU causal M=N prefill attention, byte-identical to ``attention_prefill_native`` / the NumPy causal path.
    q:(H,M,hd) post q-norm+RoPE; k/v:(Hkv,L,hd), L==start+M. Returns (H,M,hd) int64, or None when unavailable
    or a head overflows the int64 bound (→ caller falls back to the CPU loud path — no silent wrap)."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_attention_prefill_gpu"):
        return None
    q = np.ascontiguousarray(q_fp, dtype=np.int64)
    k = np.ascontiguousarray(k_fp, dtype=np.int64)
    v = np.ascontiguousarray(v_fp, dtype=np.int64)
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(f"prefill attn shapes: q{q.shape} k{k.shape} v{v.shape}")
    H, M, hd = q.shape
    Hkv, L, hd_k = k.shape
    if hd_k != hd or v.shape != (Hkv, L, hd) or L != int(start) + M:
        raise ValueError(f"prefill attn shape/length mismatch: q{q.shape} k{k.shape} v{v.shape} start={start}")
    out = np.empty((H, M, hd), dtype=np.int64)
    rc = lib.bonsai_attention_prefill_gpu(
        q.ctypes.data, k.ctypes.data, v.ctypes.data,
        ctypes.c_int64(H), ctypes.c_int64(Hkv), ctypes.c_int64(hd),
        ctypes.c_int64(M), ctypes.c_int64(L), ctypes.c_int64(int(start)),
        ctypes.c_int64(int(frac)), ctypes.c_int64(int(inv_sqrt_fp)), out.ctypes.data,
    )
    if rc == 2:
        return None
    if rc != 0:
        return None
    return out


def attention_decode_batched_gpu(q_fp, Kc, Vc, lengths, frac, inv_sqrt_fp):
    """Standalone batched M=1 decode attention (parity gate). q:(B,H,hd); Kc/Vc padded (B,Hkv,cap,hd); lengths
    (B,). Byte-identical to attention_decode_batched_native. Returns (B,H,hd) or None (unavailable/overflow)."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_attention_decode_batched_gpu"):
        return None
    q = np.ascontiguousarray(q_fp, dtype=np.int64)
    K = np.ascontiguousarray(Kc, dtype=np.int64)
    V = np.ascontiguousarray(Vc, dtype=np.int64)
    L = np.ascontiguousarray(np.asarray(lengths, dtype=np.int64))
    B, H, hd = q.shape
    _, Hkv, cap, _ = K.shape
    out = np.empty((B, H, hd), dtype=np.int64)
    fn = lib.bonsai_attention_decode_batched_gpu
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 5 + [ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    rc = fn(q.ctypes.data, K.ctypes.data, V.ctypes.data, L.ctypes.data,
            ctypes.c_int64(B), ctypes.c_int64(H), ctypes.c_int64(Hkv), ctypes.c_int64(hd), ctypes.c_int64(cap),
            ctypes.c_int64(int(frac)), ctypes.c_int64(int(inv_sqrt_fp)), out.ctypes.data)
    return None if rc != 0 else out


def monolith_available() -> bool:
    """True iff the .so exposes the M3 on-device prefill monolith (true x-residency)."""
    lib = _load_lib()
    return lib is not None and hasattr(lib, "bonsai_prefill_forward_gpu")


def batched_decode_available() -> bool:
    """True iff the .so exposes the stateful M=B fully-resident batched decode context."""
    lib = _load_lib()
    return lib is not None and hasattr(lib, "bonsai_decode_ctx_create")


def decode_ctx_create(dims, w, g, frac, eps, inv_sqrt_fp, cap):
    """Create a device decode context (KV cache + scratch + handles). dims=(B,n_layers,H,Hkv,hd,d,dff,vocab);
    w/g are the same handle bundles as prefill_forward_gpu. Returns a ctx handle (>=0) or None."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_decode_ctx_create"):
        return None
    B, n_layers, H, Hkv, hd, d, dff, vocab = dims
    a = {k: np.ascontiguousarray(np.asarray(v, dtype=np.int64)) for k, v in
         dict(wq=w["wq"], wk=w["wk"], wv=w["wv"], wo=w["wo"], w1=w["w1"], wu=w["wu"], w2=w["w2"],
              n1g=g["n1g"], n2g=g["n2g"], qng=g["qng"], kng=g["kng"]).items()}
    fn = lib.bonsai_decode_ctx_create
    fn.argtypes = ([ctypes.c_int64] * 12 + [ctypes.c_void_p] * 11 + [ctypes.c_int64] * 4)
    fn.restype = ctypes.c_int64
    h = fn(ctypes.c_int64(B), ctypes.c_int64(n_layers), ctypes.c_int64(H), ctypes.c_int64(Hkv),
           ctypes.c_int64(hd), ctypes.c_int64(d), ctypes.c_int64(dff), ctypes.c_int64(cap),
           ctypes.c_int64(vocab), ctypes.c_int64(int(frac)), ctypes.c_int64(int(eps)),
           ctypes.c_int64(int(inv_sqrt_fp)),
           a["wq"].ctypes.data, a["wk"].ctypes.data, a["wv"].ctypes.data, a["wo"].ctypes.data,
           a["w1"].ctypes.data, a["wu"].ctypes.data, a["w2"].ctypes.data,
           a["n1g"].ctypes.data, a["n2g"].ctypes.data, a["qng"].ctypes.data, a["kng"].ctypes.data,
           ctypes.c_int64(int(g["finalg"])), ctypes.c_int64(int(w["out_head"])),
           ctypes.c_int64(int(g["cos_h"])), ctypes.c_int64(int(g["sin_h"])))
    # keep the handle arrays alive on the returned object isn't needed: create copies them into the ctx (C side)
    return None if int(h) < 0 else int(h)


def decode_ctx_seed_seq(ctx_h, b, k_arr, v_arr):
    """Seed sequence b's prefilled KV (each (n_layers,Hkv,Lb,hd) int64) into the device cache. Returns True/False."""
    lib = _load_lib()
    if lib is None:
        return False
    k = np.ascontiguousarray(np.asarray(k_arr, dtype=np.int64))
    v = np.ascontiguousarray(np.asarray(v_arr, dtype=np.int64))
    Lb = int(k.shape[2])
    fn = lib.bonsai_decode_ctx_seed_seq
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    fn.restype = ctypes.c_int
    rc = fn(ctypes.c_int64(int(ctx_h)), ctypes.c_int64(int(b)), k.ctypes.data, v.ctypes.data, ctypes.c_int64(Lb))
    return rc == 0


def decode_step(ctx_h, x_in, pos, vocab):
    """One M=B decode step. x_in (B,d) int64 new-token residual; pos (B,) absolute positions. Returns
    logits (B,vocab) int64, or None on overflow/failure (→ caller falls back to CPU)."""
    lib = _load_lib()
    if lib is None:
        return None
    x = np.ascontiguousarray(np.asarray(x_in, dtype=np.int64))
    p = np.ascontiguousarray(np.asarray(pos, dtype=np.int64))
    B = x.shape[0]
    out = np.empty((B, int(vocab)), dtype=np.int64)
    fn = lib.bonsai_decode_step
    fn.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    rc = fn(ctypes.c_int64(int(ctx_h)), x.ctypes.data, p.ctypes.data, out.ctypes.data)
    return None if rc != 0 else out


def decode_ctx_free(ctx_h):
    lib = _load_lib()
    if lib is not None and hasattr(lib, "bonsai_decode_ctx_free"):
        fn = lib.bonsai_decode_ctx_free
        fn.argtypes = [ctypes.c_int64]; fn.restype = None
        fn(ctypes.c_int64(int(ctx_h)))


def buf_upload_i64(arr: np.ndarray) -> "int | None":
    """Upload an int64 array to the device once; return a buffer handle (>=0) or None. For gains/cos/sin."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_buf_upload_i64"):
        return None
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.int64).ravel())
    h = lib.bonsai_buf_upload_i64(a.ctypes.data, ctypes.c_int64(a.size))
    return None if int(h) < 0 else int(h)


def prefill_forward_gpu(x_embed, dims, w, g, frac, eps, inv_sqrt_fp, *, export_kv=False):
    """M3 true-residency prefill forward, fully on-device — byte-identical to forward(..., last_only=True).
    x_embed: (T,d) int64 embedded residual. dims: (T,d,n_layers,H,Hkv,hd,dff). w: dict of handle ndarrays
    (wq,wk,wv,wo,w1,wu,w2) + scalars (out_head). g: dict of handle ndarrays (n1g,n2g,qng,kng) + scalars
    (finalg, cos_h, sin_h). Returns logits (vocab,) int64, or None on overflow/unavailable (→ CPU forward).
    With export_kv=True, also returns the per-layer KV cache: (logits, k_arr, v_arr) where k/v are
    (n_layers, Hkv, T, hd) int64 = the post-RoPE K / raw V to seed generative-decode prefill."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_prefill_forward_gpu"):
        return (None, None, None) if export_kv else None
    T, d, n_layers, H, Hkv, hd, dff = dims
    x = np.ascontiguousarray(np.asarray(x_embed, dtype=np.int64))
    arrs = {k: np.ascontiguousarray(np.asarray(v, dtype=np.int64)) for k, v in
            dict(wq=w["wq"], wk=w["wk"], wv=w["wv"], wo=w["wo"], w1=w["w1"], wu=w["wu"], w2=w["w2"],
                 n1g=g["n1g"], n2g=g["n2g"], qng=g["qng"], kng=g["kng"]).items()}
    out = np.empty((int(w["out_head_vocab"]),), dtype=np.int64)
    k_arr = v_arr = None
    k_ptr = v_ptr = 0
    if export_kv:
        k_arr = np.empty((n_layers, Hkv, T, hd), dtype=np.int64)
        v_arr = np.empty((n_layers, Hkv, T, hd), dtype=np.int64)
        k_ptr, v_ptr = k_arr.ctypes.data, v_arr.ctypes.data
    rc = lib.bonsai_prefill_forward_gpu(
        x.ctypes.data,
        ctypes.c_int64(T), ctypes.c_int64(d), ctypes.c_int64(n_layers),
        ctypes.c_int64(H), ctypes.c_int64(Hkv), ctypes.c_int64(hd), ctypes.c_int64(dff),
        ctypes.c_int64(int(frac)), ctypes.c_int64(int(eps)), ctypes.c_int64(int(inv_sqrt_fp)),
        arrs["wq"].ctypes.data, arrs["wk"].ctypes.data, arrs["wv"].ctypes.data, arrs["wo"].ctypes.data,
        arrs["w1"].ctypes.data, arrs["wu"].ctypes.data, arrs["w2"].ctypes.data,
        arrs["n1g"].ctypes.data, arrs["n2g"].ctypes.data, arrs["qng"].ctypes.data, arrs["kng"].ctypes.data,
        ctypes.c_int64(int(g["finalg"])), ctypes.c_int64(int(w["out_head"])),
        ctypes.c_int64(int(g["cos_h"])), ctypes.c_int64(int(g["sin_h"])),
        out.ctypes.data, ctypes.c_void_p(k_ptr), ctypes.c_void_p(v_ptr),
    )
    if rc != 0:
        return (None, None, None) if export_kv else None
    return (out, k_arr, v_arr) if export_kv else out
