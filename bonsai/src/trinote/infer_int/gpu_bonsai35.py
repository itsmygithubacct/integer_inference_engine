"""Resident CUDA support and memory feasibility for Bonsai-27B/Qwen3.5.

This module is deliberately separate from :mod:`reference_bonsai35`.  The CPU
model remains the canonical verifier and can be imported on machines without
CUDA.  The GPU side is an optional producer which must prove that the complete
packed artifact, persistent hybrid caches, and a bounded scratch arena fit
before graph construction is attempted.

The feasibility probe uses real ``cudaMalloc`` allocations after uploading all
Q1 weights and static integer buffers.  It therefore includes CUDA context and
allocator overhead and catches fragmentation caused by the hundreds of model
allocations; it is not just a spreadsheet estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from time import monotonic
import ctypes
import hashlib

import numpy as np

from .gpu_native import (
    _load_lib,
    buf_upload_i64,
    gpu_available,
    gpu_memory_info,
    gpu_reservation_create,
    gpu_reservation_free,
    q1_free_weights,
    q1_register_weight,
)

GIB = 1 << 30
DEFAULT_DEVICE_CEILING = int(7.5 * GIB)
# Loading the CUDA module and instantiating hundreds of allocator objects has a
# real device-memory cost which is not present in the artifact tensors.  On the
# RTX 3070 release gate the measured delta is about 0.88 GiB.  Budget a full
# GiB before the first model upload so another resident process (notably the
# regular Prism Bonsai-27B REPL) is rejected immediately instead of failing
# after several gigabytes have already crossed PCIe.
CUDA_PREFLIGHT_RUNTIME_RESERVE = 1 << 30


def _checked_product(*values: int) -> int:
    out = 1
    for value in values:
        value = int(value)
        if value < 0:
            raise ValueError("negative dimension in Qwen3.5 GPU memory plan")
        out *= value
        if out > (1 << 63) - 1:
            raise OverflowError("Qwen3.5 GPU memory-plan size exceeds int64")
    return out


def _q1_tensors(artifact: dict):
    """Yield ``(logical_name, bits, scale)`` in stable graph order."""
    yield "embed", artifact["embed_bits"], artifact["embed_scale_fp"]
    for li, layer in enumerate(artifact["layers"]):
        for key in layer:
            if not key.endswith("_bits"):
                continue
            stem = key[:-5]
            scale_key = f"{stem}_scale_fp"
            if scale_key not in layer:
                raise ValueError(f"layer {li} Q1 tensor {key} has no {scale_key}")
            yield f"layers.{li}.{stem}", layer[key], layer[scale_key]
    yield "output", artifact["output_bits"], artifact["output_scale_fp"]


def _q1_tensor_sha256(bits: np.ndarray, scale: np.ndarray) -> str:
    """Commit one logical packed-Q1 tensor without allocating a byte copy."""
    h = hashlib.sha256()
    for value in (bits, scale):
        arr = np.asarray(value)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        h.update(arr.dtype.str.encode("ascii"))
        h.update(repr(tuple(int(x) for x in arr.shape)).encode("ascii"))
        h.update(memoryview(arr).cast("B"))
    return h.hexdigest()


def qwen35_embedding_output_identity(artifact: dict) -> dict[str, object]:
    """Digest and prove whether embedding/output may share one GPU handle.

    Alias decisions use shape, dtype, byte length, and SHA-256—not a model-name
    assumption.  The release Bonsai-27B tensors are distinct, while genuinely
    tied artifacts register only the embedding allocation and reuse its handle.
    """
    eb, es = np.asarray(artifact["embed_bits"]), np.asarray(artifact["embed_scale_fp"])
    ob, os = np.asarray(artifact["output_bits"]), np.asarray(artifact["output_scale_fp"])
    embed_sha = _q1_tensor_sha256(eb, es)
    output_sha = _q1_tensor_sha256(ob, os)
    compatible = (
        eb.shape == ob.shape and es.shape == os.shape
        and eb.dtype == ob.dtype and es.dtype == os.dtype
        and eb.nbytes == ob.nbytes and es.nbytes == os.nbytes
    )
    return {
        "embedding_sha256": embed_sha,
        "output_sha256": output_sha,
        "tied": bool(compatible and embed_sha == output_sha),
        "embedding_bytes": int(eb.nbytes + es.nbytes),
        "output_bytes": int(ob.nbytes + os.nbytes),
    }


def _static_i64_tensors(artifact: dict):
    """Yield non-Q1 arrays needed by a resident hybrid graph."""
    for key in (
        "final_norm_gain_fp", "cos_fp", "sin_fp",
        "softplus_lut_fp", "exp_neg_lut_fp",
    ):
        yield key, artifact[key]
    for li, layer in enumerate(artifact["layers"]):
        for key, value in layer.items():
            if key == "kind" or key.endswith("_bits") or key.endswith("_scale_fp"):
                continue
            yield f"layers.{li}.{key}", value


def qwen35_workspace_components(
    artifact: dict,
    *,
    kv_bits: int = 64,
    capture_trace: bool = False,
) -> dict[str, int]:
    """Return exact allocation sizes for a one-sequence, 4K-capable graph.

    The arena aliases mutually exclusive recurrent/full-attention temporaries,
    but does not alias persistent state, convolution history, or KV.  The
    diagnostic residual buffer is included only when ``capture_trace`` is
    requested.  A separately allocated graph/scheduler reserve keeps the proof
    conservative until driver graph storage is measured by the final executor.
    """
    if kv_bits not in (32, 64):
        raise ValueError("kv_bits must be 32 or 64")
    cfg = artifact["config"]
    if str(cfg.get("architecture")) != "qwen35":
        raise ValueError("Qwen3.5 GPU planning requires architecture='qwen35'")
    layers = artifact["layers"]
    recurrent = sum(layer["kind"] == "recurrent" for layer in layers)
    attention = sum(layer["kind"] == "attention" for layer in layers)
    d = int(cfg["dModel"])
    dff = int(cfg["dFfn"])
    h = int(cfg["n_heads"])
    hkv = int(cfg["n_heads_kv"])
    hd = int(cfg["head_dim"])
    cap = int(cfg["context_len"])
    inner = int(cfg["ssmInnerSize"])
    state = int(cfg["ssmStateSize"])
    groups = int(cfg["ssmGroupCount"])
    value_heads = int(cfg["ssmTimeStepRank"])
    conv_k = int(cfg["ssmConvKernel"])
    conv_dim = 2 * groups * state + inner
    vocab = int(cfg["vocab"])
    qg_width = 2 * h * hd
    full_group = qg_width + 2 * hkv * hd
    recurrent_group = conv_dim + inner + 2 * value_heads
    max_q1_input = max(d, dff, inner)

    return {
        "recurrent_state_q30": _checked_product(recurrent, value_heads, state, state, 8),
        "recurrent_conv_history": _checked_product(recurrent, conv_k - 1, conv_dim, 8),
        "attention_k_cache": _checked_product(attention, hkv, cap, hd, kv_bits // 8),
        "attention_v_cache": _checked_product(attention, hkv, cap, hd, kv_bits // 8),
        # Per-attention-layer/KV-head monotone maxabs guards for K and V.
        # These are updated from only the newly appended row at decode time.
        "attention_guard_maxima": _checked_product(attention, hkv, 2, 8),
        "token_id_input": 8,
        # Four d-wide ping-pong/norm/projection buffers.
        "residual_arena": _checked_product(4, d, 8),
        # Same-input grouped projection destinations, whichever layer kind is larger.
        "projection_arena": _checked_product(max(full_group, recurrent_group), 8),
        # q/k/v, recurrent normalized/gated values, and attention head output.
        "head_arena": _checked_product(max(6 * value_heads * state, 4 * h * hd), 8),
        "ffn_arena": _checked_product(3, dff, 8),
        "attention_scores": _checked_product(h, cap, 8),
        "q1_bmma_activation_bitplanes": _checked_product(max_q1_input // 128, 4, 128),
        "output_logits": _checked_product(vocab, 8),
        "layer_descriptors": _checked_product(len(layers), 32, 8),
        "debug_layer_trace": (
            _checked_product(len(layers) + 1, d, 8) if capture_trace else 0
        ),
        # CUDA graph instantiation on the real 64-layer schedule consumes
        # substantially more driver memory than its tensor arenas.  Reserve
        # 576 MiB (measured graph overhead plus margin), not an optimistic
        # token-arena-only estimate.
        "cuda_graph_scheduler_reserve": 576 << 20,
    }


@dataclass(frozen=True)
class Bonsai35GpuFeasibility:
    available: bool
    feasible: bool
    reason: str
    kv_bits: int
    weight_count: int
    static_buffer_count: int
    expected_weight_bytes: int
    resident_weight_bytes: int
    static_buffer_bytes: int
    workspace_bytes: int
    baseline_used_bytes: int
    baseline_free_bytes: int
    preflight_required_bytes: int
    preflight_runtime_reserve_bytes: int
    peak_used_bytes: int
    post_cleanup_used_bytes: int
    total_device_bytes: int
    ceiling_bytes: int
    elapsed_s: float
    components: dict[str, int]
    logical_weight_bytes: int = 0
    aliased_weight_count: int = 0
    embedding_output_tied: bool = False
    embedding_weight_sha256: str = ""
    output_weight_sha256: str = ""

    @property
    def safety_margin_bytes(self) -> int:
        return self.ceiling_bytes - self.peak_used_bytes

    def as_dict(self) -> dict:
        result = dict(self.__dict__)
        result["safety_margin_bytes"] = self.safety_margin_bytes
        result["kv_allocated_bytes"] = (
            int(self.components.get("attention_k_cache", 0))
            + int(self.components.get("attention_v_cache", 0))
        )
        return result


def _unavailable(reason: str, kv_bits: int, ceiling: int, elapsed: float = 0.0):
    return Bonsai35GpuFeasibility(
        available=False, feasible=False, reason=reason, kv_bits=kv_bits,
        weight_count=0, static_buffer_count=0, expected_weight_bytes=0,
        resident_weight_bytes=0, static_buffer_bytes=0, workspace_bytes=0,
        baseline_used_bytes=0, baseline_free_bytes=0,
        preflight_required_bytes=0,
        preflight_runtime_reserve_bytes=CUDA_PREFLIGHT_RUNTIME_RESERVE,
        peak_used_bytes=0, post_cleanup_used_bytes=0,
        total_device_bytes=0, ceiling_bytes=ceiling, elapsed_s=elapsed,
        components={},
    )


def prove_bonsai35_gpu_memory(
    artifact: dict,
    *,
    kv_bits: int = 64,
    ceiling_bytes: int = DEFAULT_DEVICE_CEILING,
    retain: bool = False,
    capture_trace: bool = False,
) -> "tuple[Bonsai35GpuFeasibility, dict | None]":
    """Upload the real model and prove a complete resident allocation.

    When ``retain`` is false (the default), every weight, static buffer, and
    reservation is released before return.  With ``retain=True`` the caller
    receives opaque handles for immediate executor construction and assumes
    cleanup responsibility via :func:`release_bonsai35_gpu_residency`.
    ``capture_trace`` includes the optional post-layer residual allocation in
    that proof.
    """
    started = monotonic()
    if not gpu_available():
        return _unavailable("CUDA Q1 library/device unavailable", kv_bits, ceiling_bytes), None
    baseline = gpu_memory_info()
    if baseline is None:
        return _unavailable("CUDA memory-query ABI unavailable", kv_bits, ceiling_bytes), None

    components = qwen35_workspace_components(
        artifact,
        kv_bits=kv_bits,
        capture_trace=capture_trace,
    )
    logical_weight_bytes = sum(
        int(np.asarray(bits).nbytes + np.asarray(scale).nbytes)
        for _, bits, scale in _q1_tensors(artifact)
    )
    tied_identity = qwen35_embedding_output_identity(artifact)
    aliased_weight_count = 1 if bool(tied_identity["tied"]) else 0
    expected_weight_bytes = logical_weight_bytes - (
        int(tied_identity["output_bytes"]) if aliased_weight_count else 0
    )
    static_bytes = sum(
        int(np.asarray(value).size * 8)
        for _, value in _static_i64_tensors(artifact)
    )
    preflight_required = (
        expected_weight_bytes
        + static_bytes
        + sum(components.values())
        + CUDA_PREFLIGHT_RUNTIME_RESERVE
    )
    predicted_peak = int(baseline["used_bytes"]) + preflight_required
    conflicts = []
    if preflight_required > int(baseline["free_bytes"]):
        conflicts.append(
            f"need {preflight_required} free bytes, device has {baseline['free_bytes']}"
        )
    if predicted_peak > int(ceiling_bytes):
        conflicts.append(
            f"predicted used bytes {predicted_peak} exceed ceiling {ceiling_bytes}"
        )
    if conflicts:
        reason = (
            "GPU exclusivity conflict: " + "; ".join(conflicts)
            + f" (includes {CUDA_PREFLIGHT_RUNTIME_RESERVE}-byte CUDA runtime reserve); "
              "no model upload attempted"
        )
        report = Bonsai35GpuFeasibility(
            available=True,
            feasible=False,
            reason=reason,
            kv_bits=kv_bits,
            weight_count=0,
            static_buffer_count=0,
            expected_weight_bytes=expected_weight_bytes,
            resident_weight_bytes=0,
            static_buffer_bytes=static_bytes,
            workspace_bytes=sum(components.values()),
            baseline_used_bytes=int(baseline["used_bytes"]),
            baseline_free_bytes=int(baseline["free_bytes"]),
            preflight_required_bytes=preflight_required,
            preflight_runtime_reserve_bytes=CUDA_PREFLIGHT_RUNTIME_RESERVE,
            peak_used_bytes=int(baseline["used_bytes"]),
            post_cleanup_used_bytes=int(baseline["used_bytes"]),
            total_device_bytes=int(baseline["total_bytes"]),
            ceiling_bytes=int(ceiling_bytes),
            elapsed_s=monotonic() - started,
            components=components,
            logical_weight_bytes=logical_weight_bytes,
            aliased_weight_count=aliased_weight_count,
            embedding_output_tied=bool(tied_identity["tied"]),
            embedding_weight_sha256=str(tied_identity["embedding_sha256"]),
            output_weight_sha256=str(tied_identity["output_sha256"]),
        )
        return report, None

    handles: dict[str, dict[str, int] | int] = {"weights": {}, "buffers": {}}
    weight_count = 0
    buffer_count = 0
    reservation_handle = None
    reason = "ok"
    feasible = False
    peak = baseline
    try:
        for name, bits, scale in _q1_tensors(artifact):
            if name == "output" and aliased_weight_count:
                handles["weights"][name] = handles["weights"]["embed"]
                continue
            b = np.asarray(bits)
            s = np.asarray(scale)
            if b.dtype != np.uint8 or s.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
                raise TypeError(f"{name}: expected uint8 bits and int32/int64 scales, got {b.dtype}/{s.dtype}")
            handle = q1_register_weight(b, s, gpu_bmma=True)
            if handle is None:
                reason = f"CUDA OOM/error registering Q1 tensor {name}"
                break
            handles["weights"][name] = handle
            weight_count += 1
        else:
            for name, value in _static_i64_tensors(artifact):
                arr = np.asarray(value)
                if not np.issubdtype(arr.dtype, np.integer):
                    raise TypeError(f"{name}: resident graph buffer is not integer ({arr.dtype})")
                # Static graph kernels consume int64.  Most are already int64;
                # count the actual widened bytes for an honest device budget.
                handle = buf_upload_i64(arr)
                if handle is None:
                    reason = f"CUDA OOM/error registering static tensor {name}"
                    break
                handles["buffers"][name] = handle
                buffer_count += 1
            else:
                reserved = gpu_reservation_create(
                    value for value in components.values() if value > 0
                )
                if reserved is None:
                    reason = f"CUDA OOM reserving complete Qwen3.5 graph ({kv_bits}-bit KV)"
                else:
                    reservation_handle, allocated = reserved
                    if allocated != sum(components.values()):
                        raise RuntimeError("CUDA reservation byte count drifted from memory plan")
                    handles["reservation"] = reservation_handle
                    peak = gpu_memory_info() or peak
                    tracked = int(peak["resident_weight_bytes"])
                    feasible = (
                        tracked == expected_weight_bytes
                        and int(peak["used_bytes"]) <= int(ceiling_bytes)
                    )
                    if tracked != expected_weight_bytes:
                        reason = (
                            f"resident weight tracker reports {tracked}, expected {expected_weight_bytes}"
                        )
                    elif int(peak["used_bytes"]) > int(ceiling_bytes):
                        reason = (
                            f"peak {peak['used_bytes']} exceeds configured ceiling {ceiling_bytes}"
                        )
        if not feasible:
            peak = gpu_memory_info() or peak
    finally:
        if not retain or not feasible:
            if reservation_handle is not None:
                gpu_reservation_free(reservation_handle)
            q1_free_weights()

    post = gpu_memory_info() or baseline
    report = Bonsai35GpuFeasibility(
        available=True,
        feasible=feasible,
        reason=reason,
        kv_bits=kv_bits,
        weight_count=weight_count,
        static_buffer_count=buffer_count,
        expected_weight_bytes=expected_weight_bytes,
        resident_weight_bytes=int(peak["resident_weight_bytes"]),
        static_buffer_bytes=static_bytes,
        workspace_bytes=sum(components.values()),
        baseline_used_bytes=int(baseline["used_bytes"]),
        baseline_free_bytes=int(baseline["free_bytes"]),
        preflight_required_bytes=preflight_required,
        preflight_runtime_reserve_bytes=CUDA_PREFLIGHT_RUNTIME_RESERVE,
        peak_used_bytes=int(peak["used_bytes"]),
        post_cleanup_used_bytes=int(post["used_bytes"]),
        total_device_bytes=int(peak["total_bytes"]),
        ceiling_bytes=int(ceiling_bytes),
        elapsed_s=monotonic() - started,
        components=components,
        logical_weight_bytes=logical_weight_bytes,
        aliased_weight_count=aliased_weight_count,
        embedding_output_tied=bool(tied_identity["tied"]),
        embedding_weight_sha256=str(tied_identity["embedding_sha256"]),
        output_weight_sha256=str(tied_identity["output_sha256"]),
    )
    return report, handles if retain and feasible else None


def release_bonsai35_gpu_residency(handles: dict | None) -> None:
    """Release retained feasibility/executor allocations, safely and exactly."""
    if handles and "reservation" in handles:
        gpu_reservation_free(int(handles["reservation"]))
    # The current CUDA registry is process-global; Qwen3.5 is deliberately the
    # sole resident model on an 8 GiB card.  This frees weights and buffers.
    q1_free_weights()


def recurrent_step_gpu(
    q_fp: np.ndarray,
    k_fp: np.ndarray,
    v_fp: np.ndarray,
    z_fp: np.ndarray,
    alpha_fp: np.ndarray,
    beta_fp: np.ndarray,
    state_fp: np.ndarray,
    dt_bias_fp: np.ndarray,
    ssm_a_fp: np.ndarray,
    norm_gain_fp: np.ndarray,
    artifact: dict,
) -> "tuple[np.ndarray, np.ndarray] | None":
    """Run one exact M=1 Gated DeltaNet update on CUDA.

    Inputs are the post-convolution q/k/v slices and projected z/alpha/beta.
    Returns ``(gated_q16, updated_state_q30)``.  The caller's state is never
    mutated: any CUDA/range failure returns ``None`` so CPU fallback starts from
    the untouched pre-step cache.
    """
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai35_recurrent_step_gpu"):
        return None
    cfg = artifact["config"]
    frac = int(cfg["frac"])
    state_frac = int(cfg["ssmStateFrac"])
    key_heads = int(cfg["ssmGroupCount"])
    value_heads = int(cfg["ssmTimeStepRank"])
    state_size = int(cfg["ssmStateSize"])

    def arr(value, shape, name):
        out = np.ascontiguousarray(np.asarray(value, dtype=np.int64))
        if out.shape != shape:
            raise ValueError(f"{name} shape {out.shape}, expected {shape}")
        return out

    q = arr(q_fp, (key_heads, state_size), "q")
    k = arr(k_fp, (key_heads, state_size), "k")
    v = arr(v_fp, (value_heads, state_size), "v")
    z = arr(z_fp, (value_heads, state_size), "z")
    alpha = arr(alpha_fp, (value_heads,), "alpha")
    beta = arr(beta_fp, (value_heads,), "beta")
    state = arr(state_fp, (value_heads, state_size, state_size), "state").copy()
    dt = arr(dt_bias_fp, (value_heads,), "dt_bias")
    ssm_a = arr(ssm_a_fp, (value_heads,), "ssm_a")
    gain = arr(norm_gain_fp, (state_size,), "norm_gain")
    soft = np.ascontiguousarray(np.asarray(artifact["softplus_lut_fp"], dtype=np.int64))
    exp = np.ascontiguousarray(np.asarray(artifact["exp_neg_lut_fp"], dtype=np.int64))
    gated = np.empty((value_heads, state_size), dtype=np.int64)

    fn = lib.bonsai35_recurrent_step_gpu
    if not getattr(fn, "_trinote_typed", False):
        fn.argtypes = (
            [ctypes.c_void_p] * 11
            + [ctypes.c_int64, ctypes.c_void_p]
            + [ctypes.c_int64] * 13
            + [ctypes.c_void_p]
        )
        fn.restype = ctypes.c_int
        fn._trinote_typed = True
    rc = fn(
        q.ctypes.data, k.ctypes.data, v.ctypes.data, z.ctypes.data,
        alpha.ctypes.data, beta.ctypes.data, state.ctypes.data,
        dt.ctypes.data, ssm_a.ctypes.data, gain.ctypes.data,
        soft.ctypes.data, ctypes.c_int64(soft.size), exp.ctypes.data,
        ctypes.c_int64(exp.size), ctypes.c_int64(key_heads),
        ctypes.c_int64(value_heads), ctypes.c_int64(state_size),
        ctypes.c_int64(frac), ctypes.c_int64(state_frac),
        ctypes.c_int64(int(cfg["softplusLutMinFp"])),
        ctypes.c_int64(int(cfg["lutStepFp"])),
        ctypes.c_int64(int(cfg["softplusLutMaxFp"])),
        ctypes.c_int64(int(cfg["expNegLutMinFp"])),
        ctypes.c_int64(int(cfg["lutStepFp"])),
        ctypes.c_int64(int(cfg["gdnScaleFp"])),
        ctypes.c_int64(int(cfg["ssmRmsEpsilonFp2"])),
        gated.ctypes.data,
    )
    return None if rc != 0 else (gated, state)


def attention_decode_gpu(
    qg_fp: np.ndarray,
    k_fp: np.ndarray,
    v_fp: np.ndarray,
    k_prefix_fp: "np.ndarray | None",
    v_prefix_fp: "np.ndarray | None",
    q_norm_gain_fp: np.ndarray,
    k_norm_gain_fp: np.ndarray,
    cos_fp: np.ndarray,
    sin_fp: np.ndarray,
    artifact: dict,
) -> "tuple[np.ndarray, np.ndarray] | None":
    """Run one exact gated Qwen3.5 full-attention decode primitive.

    The returned tuple is ``(gated_heads, transformed_k_row)``.  Prefix caches
    and all caller inputs remain untouched on failure, permitting a clean CPU
    fallback before cache mutation.
    """
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai35_attention_decode_gpu"):
        return None
    cfg = artifact["config"]
    h, hkv, hd = int(cfg["n_heads"]), int(cfg["n_heads_kv"]), int(cfg["head_dim"])
    n_rot, frac = int(cfg["ropeRotDim"]), int(cfg["frac"])

    def arr(value, shape, name):
        out = np.ascontiguousarray(np.asarray(value, dtype=np.int64))
        if out.shape != shape:
            raise ValueError(f"{name} shape {out.shape}, expected {shape}")
        return out

    qg = arr(qg_fp, (h, 2, hd), "qg")
    k = arr(k_fp, (hkv, hd), "k")
    v = arr(v_fp, (hkv, hd), "v")
    qgain = arr(q_norm_gain_fp, (hd,), "q_norm_gain")
    kgain = arr(k_norm_gain_fp, (hd,), "k_norm_gain")
    cos = arr(cos_fp, (n_rot // 2,), "cos")
    sin = arr(sin_fp, (n_rot // 2,), "sin")
    if k_prefix_fp is None or v_prefix_fp is None:
        if k_prefix_fp is not None or v_prefix_fp is not None:
            raise ValueError("K and V prefixes must both be None or both arrays")
        prefix_len = 0
        kp = vp = None
        kp_ptr = vp_ptr = 0
    else:
        kp0 = np.asarray(k_prefix_fp)
        prefix_len = int(kp0.shape[1]) if kp0.ndim == 3 else -1
        kp = arr(k_prefix_fp, (hkv, prefix_len, hd), "K prefix")
        vp = arr(v_prefix_fp, (hkv, prefix_len, hd), "V prefix")
        kp_ptr, vp_ptr = kp.ctypes.data, vp.ctypes.data
    gated = np.empty((h, hd), dtype=np.int64)
    krow = np.empty((hkv, hd), dtype=np.int64)
    fn = lib.bonsai35_attention_decode_gpu
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 8 + [ctypes.c_void_p] * 6
    fn.restype = ctypes.c_int
    rc = fn(
        qg.ctypes.data, k.ctypes.data, v.ctypes.data,
        ctypes.c_void_p(kp_ptr), ctypes.c_void_p(vp_ptr),
        ctypes.c_int64(prefix_len), ctypes.c_int64(h), ctypes.c_int64(hkv),
        ctypes.c_int64(hd), ctypes.c_int64(n_rot), ctypes.c_int64(frac),
        ctypes.c_int64(int(cfg["rmsEpsilonFp2"])),
        ctypes.c_int64(int(cfg["attentionScaleFp"])),
        qgain.ctypes.data, kgain.ctypes.data, cos.ctypes.data, sin.ctypes.data,
        gated.ctypes.data, krow.ctypes.data,
    )
    return None if rc != 0 else (gated, krow)


def kv_i32_roundtrip_gpu(values: np.ndarray) -> "np.ndarray | None":
    """Exercise the guarded resident KV narrowing contract on CUDA.

    Returns an exactly sign-extended int64 copy when every value lies in the
    closed int32 interval.  Any out-of-range value returns ``None``, matching
    the resident graph's fail-loud poison/fallback decision.
    """
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai35_kv_i32_roundtrip_gpu"):
        return None
    source = np.ascontiguousarray(np.asarray(values, dtype=np.int64).reshape(-1))
    if source.size == 0:
        raise ValueError("KV int32 roundtrip needs at least one value")
    output = np.empty_like(source)
    fn = lib.bonsai35_kv_i32_roundtrip_gpu
    fn.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    rc = fn(source.ctypes.data, ctypes.c_int64(source.size), output.ctypes.data)
    return output if rc == 0 else None


def kv_i32_transaction_gpu(
    k_values: np.ndarray,
    v_values: np.ndarray,
    k_initial: np.ndarray,
    v_initial: np.ndarray,
) -> "tuple[bool, np.ndarray, np.ndarray] | None":
    """Run the resident paired KV preflight/commit against sentinel rows.

    The returned boolean is true only when both rows were committed.  On an
    unsafe lane the two returned rows must exactly equal their initial values,
    making the no-partial-write contract directly testable.
    """
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai35_kv_i32_transaction_gpu"):
        return None

    def wide(value, name):
        out = np.ascontiguousarray(np.asarray(value, dtype=np.int64).reshape(-1))
        if not out.size:
            raise ValueError(f"{name} must not be empty")
        return out

    k = wide(k_values, "K values")
    v = wide(v_values, "V values")
    if v.shape != k.shape:
        raise ValueError("K/V transaction rows must have the same shape")
    ki64 = wide(k_initial, "initial K")
    vi64 = wide(v_initial, "initial V")
    if ki64.shape != k.shape or vi64.shape != k.shape:
        raise ValueError("initial K/V rows must match the candidate rows")
    if np.any(ki64 < np.iinfo(np.int32).min) or np.any(ki64 > np.iinfo(np.int32).max):
        raise ValueError("initial K sentinel leaves int32")
    if np.any(vi64 < np.iinfo(np.int32).min) or np.any(vi64 > np.iinfo(np.int32).max):
        raise ValueError("initial V sentinel leaves int32")
    ki = np.ascontiguousarray(ki64.astype(np.int32))
    vi = np.ascontiguousarray(vi64.astype(np.int32))
    ko = np.empty_like(k)
    vo = np.empty_like(v)
    fn = lib.bonsai35_kv_i32_transaction_gpu
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] + [ctypes.c_void_p] * 2
    fn.restype = ctypes.c_int
    rc = fn(k.ctypes.data, v.ctypes.data, ki.ctypes.data, vi.ctypes.data,
            ctypes.c_int64(k.size), ko.ctypes.data, vo.ctypes.data)
    if rc not in (0, 4):
        return None
    return rc == 0, ko, vo


class _Gpu35Config(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in (
        "n_layers", "d", "dff", "H", "Hkv", "hd", "vocab", "cap", "frac", "eps", "n_rot",
        "key_heads", "value_heads", "state_size", "state_frac", "inner", "conv_k",
        "gdn_scale", "attn_scale", "ssm_eps", "soft_min", "soft_step", "soft_max", "exp_min", "exp_step",
        "embed", "final_gain", "out_head", "cos_buf", "sin_buf", "soft_buf", "soft_n", "exp_buf", "exp_n",
    )]


class _Gpu35Layer(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in (
        "kind", "slot", "n1", "n2", "w1", "wu", "w2",
        "wqkv", "wz", "walpha", "wbeta", "wout", "conv", "dt_bias", "ssm_a", "ssm_norm",
        "wqg", "wk", "wv", "wo", "q_norm", "k_norm",
    )]


class Bonsai35GpuExecutor:
    """Fully resident, CUDA-graph-captured M=1 hybrid executor.

    Packed Q1 weights, Q30 recurrent state, convolution history, and attention
    KV stay on device.  Each token is one CUDA graph submission; production
    sends one int64 token ID, expands the resident Q1 embedding row on device,
    and receives one logits row.  Any fail-loud
    arithmetic guard poisons the context, and callers must discard it and
    replay on the CPU oracle—no partially mutated GPU cache is reused.
    """

    def __init__(
        self,
        artifact: dict,
        report: Bonsai35GpuFeasibility,
        handles: dict,
        ctx: int,
        *,
        capture_trace: bool,
    ):
        self.artifact = artifact
        self.report = report
        self.handles = handles
        self.ctx = int(ctx)
        self.capture_trace = bool(capture_trace)
        self.position = 0
        self.closed = False

    @classmethod
    def try_create(
        cls,
        artifact: dict,
        *,
        ceiling_bytes: int = DEFAULT_DEVICE_CEILING,
        capture_trace: bool = False,
    ) -> "Bonsai35GpuExecutor | None":
        """Return a ready executor or ``None`` after complete cleanup/fallback.

        ``capture_trace`` retains every post-layer residual for parity
        diagnostics.  Normal inference leaves it disabled so the decode graph
        does not schedule 65 otherwise-unused copies for the 64-layer release
        model.
        """
        executor, _ = cls.try_create_reported(
            artifact,
            ceiling_bytes=ceiling_bytes,
            capture_trace=capture_trace,
        )
        return executor

    @classmethod
    def try_create_reported(
        cls,
        artifact: dict,
        *,
        ceiling_bytes: int = DEFAULT_DEVICE_CEILING,
        capture_trace: bool = False,
    ) -> "tuple[Bonsai35GpuExecutor | None, Bonsai35GpuFeasibility]":
        """Return both the optional executor and its auditable launch report."""
        lib = _load_lib()
        required_graph_abi = (
            "bonsai35_ctx_create",
            "bonsai35_ctx_set_trace",
            "bonsai35_ctx_graph_stats",
        )
        if lib is None or any(not hasattr(lib, name) for name in required_graph_abi):
            return None, _unavailable(
                "CUDA Qwen3.5 graph ABI is unavailable or stale; rebuild libbonsai_q1_gpu.so",
                32,
                ceiling_bytes,
            )
        report, handles = prove_bonsai35_gpu_memory(
            artifact,
            kv_bits=32,
            ceiling_bytes=ceiling_bytes,
            retain=True,
            capture_trace=capture_trace,
        )
        if not report.feasible or handles is None:
            return None, report
        # Replace the conservative reservation with the real context.  Weights
        # and static buffers remain resident and their handles stay valid.
        reservation = handles.pop("reservation", None)
        if reservation is not None:
            gpu_reservation_free(int(reservation))
        w, b = handles["weights"], handles["buffers"]
        cfg = artifact["config"]
        descs = []
        rec_slot = att_slot = 0

        def wh(li, name):
            return int(w[f"layers.{li}.{name}"])

        def bh(li, name):
            return int(b[f"layers.{li}.{name}"])

        for li, layer in enumerate(artifact["layers"]):
            common = dict(
                n1=bh(li, "n1_gain_fp"), n2=bh(li, "n2_gain_fp"),
                w1=wh(li, "w1"), wu=wh(li, "wu"), w2=wh(li, "w2"),
            )
            absent = dict(
                wqkv=-1, wz=-1, walpha=-1, wbeta=-1, wout=-1,
                conv=-1, dt_bias=-1, ssm_a=-1, ssm_norm=-1,
                wqg=-1, wk=-1, wv=-1, wo=-1, q_norm=-1, k_norm=-1,
            )
            if layer["kind"] == "recurrent":
                absent.update(
                    wqkv=wh(li, "wqkv"), wz=wh(li, "wz"),
                    walpha=wh(li, "walpha"), wbeta=wh(li, "wbeta"),
                    wout=wh(li, "wout"), conv=bh(li, "conv_weight_fp"),
                    dt_bias=bh(li, "dt_bias_fp"), ssm_a=bh(li, "ssm_a_fp"),
                    ssm_norm=bh(li, "ssm_norm_gain_fp"),
                )
                kind, slot = 0, rec_slot
                rec_slot += 1
            else:
                absent.update(
                    wqg=wh(li, "wqg"), wk=wh(li, "wk"), wv=wh(li, "wv"),
                    wo=wh(li, "wo"), q_norm=bh(li, "q_norm_gain_fp"),
                    k_norm=bh(li, "k_norm_gain_fp"),
                )
                kind, slot = 1, att_slot
                att_slot += 1
            descs.append(_Gpu35Layer(kind=kind, slot=slot, **common, **absent))

        c = _Gpu35Config(
            n_layers=len(descs), d=int(cfg["dModel"]), dff=int(cfg["dFfn"]),
            H=int(cfg["n_heads"]), Hkv=int(cfg["n_heads_kv"]), hd=int(cfg["head_dim"]),
            vocab=int(cfg["vocab"]), cap=int(cfg["context_len"]), frac=int(cfg["frac"]),
            eps=int(cfg["rmsEpsilonFp2"]), n_rot=int(cfg["ropeRotDim"]),
            key_heads=int(cfg["ssmGroupCount"]), value_heads=int(cfg["ssmTimeStepRank"]),
            state_size=int(cfg["ssmStateSize"]), state_frac=int(cfg["ssmStateFrac"]),
            inner=int(cfg["ssmInnerSize"]), conv_k=int(cfg["ssmConvKernel"]),
            gdn_scale=int(cfg["gdnScaleFp"]), attn_scale=int(cfg["attentionScaleFp"]),
            ssm_eps=int(cfg["ssmRmsEpsilonFp2"]), soft_min=int(cfg["softplusLutMinFp"]),
            soft_step=int(cfg["lutStepFp"]), soft_max=int(cfg["softplusLutMaxFp"]),
            exp_min=int(cfg["expNegLutMinFp"]), exp_step=int(cfg["lutStepFp"]),
            embed=int(w["embed"]), final_gain=int(b["final_norm_gain_fp"]), out_head=int(w["output"]),
            cos_buf=int(b["cos_fp"]), sin_buf=int(b["sin_fp"]),
            soft_buf=int(b["softplus_lut_fp"]), soft_n=int(artifact["softplus_lut_fp"].size),
            exp_buf=int(b["exp_neg_lut_fp"]), exp_n=int(artifact["exp_neg_lut_fp"].size),
        )
        array_type = _Gpu35Layer * len(descs)
        desc_array = array_type(*descs)
        fn = lib.bonsai35_ctx_create
        fn.argtypes = [ctypes.POINTER(_Gpu35Config), ctypes.POINTER(_Gpu35Layer)]
        fn.restype = ctypes.c_int64
        ctx = int(fn(ctypes.byref(c), desc_array))
        if ctx < 0:
            q1_free_weights()
            return None, report
        set_trace = lib.bonsai35_ctx_set_trace
        set_trace.argtypes = [ctypes.c_int64, ctypes.c_int]
        set_trace.restype = ctypes.c_int
        if set_trace(ctypes.c_int64(ctx), ctypes.c_int(bool(capture_trace))) != 0:
            free_ctx = lib.bonsai35_ctx_free
            free_ctx.argtypes = [ctypes.c_int64]
            free_ctx.restype = None
            free_ctx(ctypes.c_int64(ctx))
            q1_free_weights()
            return None, replace(
                report,
                feasible=False,
                reason="CUDA Qwen3.5 context rejected its pre-capture trace mode",
            )
        return cls(
            artifact,
            report,
            handles,
            ctx,
            capture_trace=capture_trace,
        ), report

    def decode_embedded(self, x_fp: np.ndarray) -> "np.ndarray | None":
        if self.closed:
            raise RuntimeError("Bonsai35GpuExecutor is closed")
        cfg = self.artifact["config"]
        x = np.ascontiguousarray(np.asarray(x_fp, dtype=np.int64).reshape(-1))
        if x.shape != (int(cfg["dModel"]),):
            raise ValueError(f"embedded row shape {x.shape}, expected {(int(cfg['dModel']),)}")
        out = np.empty((int(cfg["vocab"]),), dtype=np.int64)
        lib = _load_lib()
        fn = lib.bonsai35_decode_step
        fn.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
        fn.restype = ctypes.c_int
        rc = fn(ctypes.c_int64(self.ctx), x.ctypes.data,
                ctypes.c_int64(self.position), out.ctypes.data)
        if rc != 0:
            return None
        self.position += 1
        return out

    def decode_token(self, token_id: int) -> "np.ndarray | None":
        if self.closed:
            raise RuntimeError("Bonsai35GpuExecutor is closed")
        token = int(token_id)
        vocab = int(self.artifact["config"]["vocab"])
        if token < 0 or token >= vocab:
            raise ValueError(f"token ID {token} is outside [0,{vocab})")
        out = np.empty((vocab,), dtype=np.int64)
        lib = _load_lib()
        fn = lib.bonsai35_decode_token
        fn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        fn.restype = ctypes.c_int
        rc = fn(ctypes.c_int64(self.ctx), ctypes.c_int64(token),
                ctypes.c_int64(self.position), out.ctypes.data)
        if rc != 0:
            return None
        self.position += 1
        return out

    def prefill(self, token_ids) -> "np.ndarray | None":
        last = None
        for token_id in token_ids:
            last = self.decode_token(int(token_id))
            if last is None:
                return None
        return last

    def generate(self, input_ids, n_new, pick, *, eos=None, on_token=None):
        """Generate with device-resident caches; return ``(tokens, complete)``."""
        if not self.reset():
            return [], False
        seq = list(input_ids)
        logits = self.prefill(seq)
        if logits is None:
            return [], False
        out = []
        for step in range(int(n_new)):
            tok = int(pick(logits, len(seq), seq))
            seq.append(tok)
            out.append(tok)
            if on_token is not None:
                on_token(tok)
            if eos is not None and tok == int(eos):
                return out, True
            if step + 1 < int(n_new):
                logits = self.decode_token(tok)
                if logits is None:
                    return out, False
        return out, True

    def reset(self) -> bool:
        if self.closed:
            return False
        lib = _load_lib()
        fn = lib.bonsai35_ctx_reset
        fn.argtypes = [ctypes.c_int64]
        fn.restype = ctypes.c_int
        ok = fn(ctypes.c_int64(self.ctx)) == 0
        if ok:
            self.position = 0
        return ok

    def debug_snapshot(self) -> "dict[str, np.ndarray] | None":
        """Export compact caches and every post-layer residual for parity tests."""
        if self.closed:
            return None
        if not self.capture_trace:
            raise RuntimeError(
                "layer trace capture is disabled; create the executor with capture_trace=True"
            )
        return self._export_snapshot(include_trace=True)

    def state_snapshot(self) -> "dict[str, np.ndarray] | None":
        """Export recurrent/conv/KV state without requiring diagnostic layer traces."""
        if self.closed:
            return None
        return self._export_snapshot(include_trace=False)

    def _export_snapshot(self, *, include_trace: bool) -> "dict[str, np.ndarray] | None":
        cfg = self.artifact["config"]
        nrec = sum(layer["kind"] == "recurrent" for layer in self.artifact["layers"])
        natt = len(self.artifact["layers"]) - nrec
        vh, ss = int(cfg["ssmTimeStepRank"]), int(cfg["ssmStateSize"])
        groups, inner = int(cfg["ssmGroupCount"]), int(cfg["ssmInnerSize"])
        conv_dim = 2 * groups * ss + inner
        ck = int(cfg["ssmConvKernel"])
        hkv, hd = int(cfg["n_heads_kv"]), int(cfg["head_dim"])
        state = np.empty((nrec, vh, ss, ss), dtype=np.int64)
        conv = np.empty((nrec, ck - 1, conv_dim), dtype=np.int64)
        k = np.empty((natt, hkv, self.position, hd), dtype=np.int64)
        v = np.empty_like(k)
        trace = (
            np.empty((len(self.artifact["layers"]) + 1, int(cfg["dModel"])), dtype=np.int64)
            if include_trace else None
        )
        lib = _load_lib()
        fn = lib.bonsai35_ctx_export
        fn.argtypes = [ctypes.c_int64] + [ctypes.c_void_p] * 5
        fn.restype = ctypes.c_int
        rc = fn(
            ctypes.c_int64(self.ctx), state.ctypes.data, conv.ctypes.data,
            k.ctypes.data if k.size else ctypes.c_void_p(),
            v.ctypes.data if v.size else ctypes.c_void_p(),
            trace.ctypes.data if trace is not None else ctypes.c_void_p(),
        )
        if rc != 0:
            return None
        result = {"state": state, "conv": conv, "k": k, "v": v}
        if trace is not None:
            result["trace"] = trace
        return result

    def graph_metadata(self) -> dict[str, int | bool]:
        """Return the exact captured CUDA graph shape and trace overhead."""
        if self.closed:
            raise RuntimeError("Bonsai35GpuExecutor is closed")
        if not self.stats()["graph_ready"]:
            raise RuntimeError("CUDA graph is not captured; decode at least one token first")
        lib = _load_lib()
        fn = lib.bonsai35_ctx_graph_stats
        fn.argtypes = [ctypes.c_int64] + [ctypes.c_void_p] * 6
        fn.restype = ctypes.c_int
        values = [ctypes.c_int64() for _ in range(5)]
        trace_enabled = ctypes.c_int()
        rc = fn(
            ctypes.c_int64(self.ctx),
            *(ctypes.byref(value) for value in values),
            ctypes.byref(trace_enabled),
        )
        if rc != 0:
            raise RuntimeError("cannot query Bonsai35 CUDA graph metadata")
        layers = len(self.artifact["layers"])
        d_model = int(self.artifact["config"]["dModel"])
        trace_copy_nodes = layers + 1 if trace_enabled.value else 0
        return {
            "graph_nodes": int(values[0].value),
            "kernel_nodes": int(values[1].value),
            "memcpy_nodes": int(values[2].value),
            "memset_nodes": int(values[3].value),
            "other_nodes": int(values[4].value),
            "trace_enabled": bool(trace_enabled.value),
            "trace_copy_nodes_per_launch": trace_copy_nodes,
            "trace_copy_bytes_per_launch": trace_copy_nodes * d_model * 8,
        }

    def stats(self) -> dict[str, int | bool | str]:
        lib = _load_lib()
        fn = lib.bonsai35_ctx_stats
        fn.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_void_p]
        fn.restype = ctypes.c_int
        launches = ctypes.c_int64()
        position = ctypes.c_int64()
        ready = ctypes.c_int()
        poisoned = ctypes.c_int()
        rc = fn(ctypes.c_int64(self.ctx), ctypes.byref(launches), ctypes.byref(position),
                ctypes.byref(ready), ctypes.byref(poisoned))
        if rc != 0:
            raise RuntimeError("cannot query Bonsai35 CUDA context stats")
        result: dict[str, int | bool | str] = {
            "graph_launches": int(launches.value), "position": int(position.value),
            "graph_ready": bool(ready.value), "poisoned": bool(poisoned.value),
        }
        if hasattr(lib, "bonsai35_ctx_input_stats"):
            io = lib.bonsai35_ctx_input_stats
            io.argtypes = [ctypes.c_int64] + [ctypes.c_void_p] * 4
            io.restype = ctypes.c_int
            mode = ctypes.c_int64()
            token_submissions = ctypes.c_int64()
            embedded_submissions = ctypes.c_int64()
            host_bytes = ctypes.c_int64()
            if io(ctypes.c_int64(self.ctx), ctypes.byref(mode), ctypes.byref(token_submissions),
                  ctypes.byref(embedded_submissions), ctypes.byref(host_bytes)) != 0:
                raise RuntimeError("cannot query Bonsai35 CUDA input stats")
            result.update({
                "input_mode": "token_id" if mode.value == 2 else (
                    "embedded_row" if mode.value == 1 else "uncaptured"
                ),
                "token_input_submissions": int(token_submissions.value),
                "embedded_input_submissions": int(embedded_submissions.value),
                "model_input_host_bytes": int(host_bytes.value),
            })
        return result

    def close(self) -> None:
        if self.closed:
            return
        lib = _load_lib()
        if lib is not None and hasattr(lib, "bonsai35_ctx_free"):
            fn = lib.bonsai35_ctx_free
            fn.argtypes = [ctypes.c_int64]
            fn.restype = None
            fn(ctypes.c_int64(self.ctx))
        q1_free_weights()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def cpu_oracle_trace(
    artifact: dict,
    token_ids,
    *,
    capture_argmax: bool = False,
    accelerated_native: bool = False,
) -> dict[str, np.ndarray]:
    """CPU trace matching :meth:`Bonsai35GpuExecutor.debug_snapshot`.

    The default is the canonical pure NumPy/Python oracle.  Exact native
    kernels are available only through the explicitly named
    ``accelerated_native=True`` diagnostic opt-in; release acceptance must not
    use that shortcut.
    """
    from .reference_bonsai import _rmsnorm, q1_rows_fp
    from .reference_bonsai35 import _Qwen35Cache, _ffn, _full_attention, _recurrent_attention, _project

    cfg = artifact["config"]
    frac, eps = int(cfg["frac"]), int(cfg["rmsEpsilonFp2"])
    native = False
    if accelerated_native:
        from .q1_native import q1_native_available
        native = bool(q1_native_available())
    cache = _Qwen35Cache(len(artifact["layers"]))
    trace = None
    last_x = None
    argmax_ids: list[int] = []
    logits = None
    for token_id in token_ids:
        start = int(cache.t)
        x = q1_rows_fp(
            artifact["embed_bits"], artifact["embed_scale_fp"],
            np.asarray([int(token_id)], dtype=np.int64), frac,
        )
        rows = [x[0].copy()]
        for li, layer in enumerate(artifact["layers"]):
            n1 = _rmsnorm(x, frac, layer["n1_gain_fp"], native=native, eps=eps)
            if layer["kind"] == "recurrent":
                attn = _recurrent_attention(n1, layer, artifact, cache, li, native=native)
            else:
                attn = _full_attention(n1, layer, artifact, cache, li, start, native=native)
            x = x + attn
            n2 = _rmsnorm(x, frac, layer["n2_gain_fp"], native=native, eps=eps)
            x = x + _ffn(n2, layer, frac, native=native)
            rows.append(x[0].copy())
        cache.t = start + 1
        trace = np.stack(rows)
        last_x = x
        if capture_argmax:
            final = _rmsnorm(
                x, frac, artifact["final_norm_gain_fp"], native=native, eps=eps
            )
            logits = _project(final, artifact, "output", frac, native=native)[0]
            argmax_ids.append(int(np.argmax(logits)))
    if trace is None or last_x is None:
        raise ValueError("CPU Qwen3.5 trace needs at least one token")
    recurrent = [i for i, layer in enumerate(artifact["layers"]) if layer["kind"] == "recurrent"]
    attention = [i for i, layer in enumerate(artifact["layers"]) if layer["kind"] == "attention"]
    state = np.stack([cache.state[i] for i in recurrent])
    conv = np.stack([cache.conv[i] for i in recurrent])
    k = np.stack([cache.k[i] for i in attention])
    v = np.stack([cache.v[i] for i in attention])
    if logits is None:
        final = _rmsnorm(
            last_x, frac, artifact["final_norm_gain_fp"], native=native, eps=eps
        )
        logits = _project(final, artifact, "output", frac, native=native)[0]
    result = {"state": state, "conv": conv, "k": k, "v": v,
              "trace": trace, "logits": logits}
    if capture_argmax:
        result["argmax_ids"] = np.asarray(argmax_ids, dtype=np.int64)
    return result
