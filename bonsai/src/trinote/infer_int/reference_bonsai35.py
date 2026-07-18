"""Deterministic integer reference engine for Bonsai-27B / Qwen3.5 Q1_0.

The model alternates causal Gated DeltaNet recurrent blocks with gated full
attention.  All data-dependent execution in this module is integer/fixed-point:

* Q1_0 matrices use the same packed bit kernel as Bonsai-8B;
* RMSNorm, SiLU, sigmoid, softmax and attention use the established integer
  primitives;
* softplus and exp read committed integer lookup tables;
* convolution and DeltaNet state updates use explicit fixed-point products.

This is a new canonical graph.  It is intentionally not presented as
bit-identical to llama.cpp's floating-point Qwen3.5 implementation; the
artifact plus this graph define the re-executable integer model.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..determinism.fixedpoint import (
    fixed_point_matmul,
    fixed_point_sigmoid,
    fixed_point_softmax,
)
from .q1_native import (
    Bonsai35NativeExecutor,
    _validate_b35_token_id,
    _validate_b35_token_ids,
    attention_decode_native,
    attention_prefill_native,
    gdn_prefill_native,
    q1_argmax_native,
    q1_linear_native,
    q1_prepare_apply_many_native,
    q1_linear_prepared_many_native,
    q1_native_available,
    q1_prepare_native,
    q1_weight_group,
)
from .reference_bonsai import (
    _head_rmsnorm,
    _rmsnorm,
    fixed_point_silu,
    q1_linear_ref,
    q1_rows_fp,
)
from .sampler import apply_rep_penalty

_NEG_INF_SHIFT = 30


def _q1_chunk_rows() -> int:
    """Bound activation-LUT workspace during prefill (decode is one row)."""
    try:
        return max(1, int(os.environ.get("TRINOTE_BONSAI35_Q1_CHUNK", "8")))
    except ValueError:
        return 8


def _native_gdn_enabled() -> bool:
    """Allow an exact replay of the pre-native-GDN Python graph.

    Native GDN was added after the frozen Bonsai-27B diagnostic profile.  The
    benchmark harness uses this opt-out only for its ``legacy-native`` lane;
    normal native inference remains default-on.
    """
    value = os.environ.get("TRINOTE_BONSAI35_NATIVE_GDN")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


class _Qwen35NativeRuntime:
    """Per-model validated Q1 descriptors and reusable decode workspaces."""

    __slots__ = ("groups", "head_maps", "fused", "lut32_mode")

    def __init__(self, artifact: dict):
        self.groups = {}
        self.head_maps = {}
        self.fused = os.environ.get("TRINOTE_BONSAI35_Q1_FUSED", "1").strip().lower() not in {
            "0", "false", "no", "off",
        }
        self.lut32_mode = os.environ.get(
            "TRINOTE_BONSAI35_Q1_LUT32", "auto"
        ).strip().lower()
        for layer in artifact["layers"]:
            self._add(layer, ("w1", "wu"))
            self._add(layer, ("w2",))
            if layer["kind"] == "recurrent":
                self._add(layer, ("wqkv", "wz", "walpha", "wbeta"))
                self._add(layer, ("wout",))
            elif layer["kind"] == "attention":
                self._add(layer, ("wqg", "wk", "wv"))
                self._add(layer, ("wo",))
            else:
                raise ValueError(f"unknown Qwen3.5 layer kind {layer['kind']!r}")
        cfg = artifact["config"]
        value_heads = int(cfg["ssmTimeStepRank"])
        key_heads = int(cfg["ssmGroupCount"])
        mapping = np.arange(value_heads, dtype=np.int64) % key_heads
        mapping.flags.writeable = False
        self.head_maps[(value_heads, key_heads)] = mapping

    def _add(self, owner: dict, names: tuple[str, ...]) -> None:
        weights = tuple(
            (owner[f"{name}_bits"], owner[f"{name}_scale_fp"])
            for name in names
        )
        self.groups[(id(owner), names)] = q1_weight_group(weights)

    def group(self, owner: dict, names: tuple[str, ...]):
        return self.groups.get((id(owner), names))

    def head_map(self, value_heads: int, key_heads: int) -> np.ndarray:
        return self.head_maps[(int(value_heads), int(key_heads))]

    def use_lut32(self, group) -> bool:
        if self.lut32_mode in {"0", "false", "no", "off"}:
            return False
        if self.lut32_mode in {"1", "true", "yes", "on", "all"}:
            return True
        # The C dispatcher chooses scalar narrow loads at 40/48 blocks and
        # AVX2 gathers at 136 blocks.  That measured hybrid is faster end to
        # end than uint64 on the target host; every block remains guarded and
        # transparently retries uint64 on rc=5.
        return True


def _project_many(
    x_fp: np.ndarray,
    owner: dict,
    names: tuple[str, ...],
    frac: int,
    *,
    native: bool,
    runtime: _Qwen35NativeRuntime | None = None,
) -> tuple[np.ndarray, ...]:
    """Apply same-input Q1 projections through a fused validated native group."""
    x = np.atleast_2d(np.asarray(x_fp, dtype=np.int64))
    if not names:
        return ()
    chunk = _q1_chunk_rows()
    group = runtime.group(owner, names) if runtime is not None else None
    outputs = None
    if x.shape[0] > chunk:
        outputs = tuple(
            np.empty((x.shape[0], int(owner[f"{name}_scale_fp"].shape[0])), dtype=np.int64)
            for name in names
        )
    for lo in range(0, x.shape[0], chunk):
        xc = np.ascontiguousarray(x[lo:lo + chunk])
        ys = None
        if native:
            try:
                use_fused = runtime is None or runtime.fused
                prefer_lut32 = (
                    runtime.use_lut32(group)
                    if runtime is not None
                    else os.environ.get(
                        "TRINOTE_BONSAI35_Q1_LUT32", "auto"
                    ).strip().lower() not in {"0", "false", "no", "off"}
                )
                if use_fused and group is not None:
                    ys = q1_prepare_apply_many_native(
                        xc, group, frac, prefer_lut32=prefer_lut32
                    )
                elif use_fused:
                    weights = tuple(
                        (owner[f"{name}_bits"], owner[f"{name}_scale_fp"])
                        for name in names
                    )
                    ys = q1_prepare_apply_many_native(
                        xc, weights, frac, prefer_lut32=prefer_lut32
                    )
                # Older installed kernels do not expose the fused ABI.  Keep
                # the prepared-multi path as an exact compatibility fallback.
                if ys is not None:
                    pass
                else:
                    n_blocks = int(owner[f"{names[0]}_scale_fp"].shape[1])
                    prepared = q1_prepare_native(xc, n_blocks)
                    if prepared is not None:
                        weights = tuple(
                            (owner[f"{name}_bits"], owner[f"{name}_scale_fp"])
                            for name in names
                        )
                        ys = q1_linear_prepared_many_native(prepared, weights, frac)
            except (MemoryError, RuntimeError):
                ys = None
        if ys is None:
            one: list[np.ndarray] = []
            for name in names:
                y = None
                if native:
                    try:
                        y = q1_linear_native(
                            xc,
                            owner[f"{name}_bits"],
                            owner[f"{name}_scale_fp"],
                            frac,
                        )
                    except (MemoryError, RuntimeError):
                        y = None
                if y is None:
                    y = q1_linear_ref(
                        xc,
                        owner[f"{name}_bits"],
                        owner[f"{name}_scale_fp"],
                        frac,
                    )
                one.append(y)
            ys = tuple(one)
        if outputs is None:
            return ys
        for dst, y in zip(outputs, ys):
            dst[lo:lo + y.shape[0]] = y
    return outputs


def _project(x_fp: np.ndarray, owner: dict, name: str, frac: int, *, native: bool,
             runtime: _Qwen35NativeRuntime | None = None) -> np.ndarray:
    return _project_many(x_fp, owner, (name,), frac, native=native, runtime=runtime)[0]


def _lut_interp(x_fp: np.ndarray, lut: np.ndarray, *, minimum: int, step: int) -> np.ndarray:
    """Integer linear interpolation over a uniformly-spaced committed LUT."""
    x = np.asarray(x_fp, dtype=np.int64)
    table = np.asarray(lut, dtype=np.int64)
    if table.ndim != 1 or table.size < 2 or step <= 0:
        raise ValueError("malformed committed Qwen3.5 lookup table")
    maximum = minimum + step * (table.size - 1)
    xc = np.clip(x, minimum, maximum)
    pos = xc - np.int64(minimum)
    idx = np.minimum(pos // step, table.size - 2).astype(np.int64)
    rem = pos - idx * step
    lo = table[idx]
    hi = table[idx + 1]
    return lo + ((hi - lo) * rem) // step


def fixed_point_softplus_lut(x_fp: np.ndarray, artifact: dict) -> np.ndarray:
    cfg = artifact["config"]
    x = np.asarray(x_fp, dtype=np.int64)
    minimum = int(cfg["softplusLutMinFp"])
    maximum = int(cfg["softplusLutMaxFp"])
    mid = _lut_interp(
        x,
        artifact["softplus_lut_fp"],
        minimum=minimum,
        step=int(cfg["lutStepFp"]),
    )
    # log(1+exp(x)) -> 0 on the negative tail and -> x on the positive tail.
    return np.where(x <= minimum, 0, np.where(x >= maximum, x, mid)).astype(np.int64)


def fixed_point_exp_negative_lut(x_fp: np.ndarray, artifact: dict) -> np.ndarray:
    cfg = artifact["config"]
    x = np.asarray(x_fp, dtype=np.int64)
    minimum = int(cfg["expNegLutMinFp"])
    mid = _lut_interp(
        x,
        artifact["exp_neg_lut_fp"],
        minimum=minimum,
        step=int(cfg["lutStepFp"]),
    )
    fp = np.int64(1 << int(cfg["frac"]))
    return np.where(x <= minimum, 0, np.where(x >= 0, fp, mid)).astype(np.int64)


def _fixed_l2_norm(x_fp: np.ndarray, frac: int) -> np.ndarray:
    """L2-normalize rows with exact integer sums and integer square roots."""
    x = np.atleast_2d(np.asarray(x_fp, dtype=np.int64))
    out = np.empty_like(x)
    for i, row in enumerate(x):
        values = row.astype(object)
        ssq = int(np.dot(values, values))
        norm = math.isqrt(ssq)
        if norm == 0:
            out[i] = 0
        else:
            out[i] = ((values << frac) // norm).astype(np.int64)
    return out


def _apply_partial_neox_rope(
    heads: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    frac: int,
    n_rot: int,
) -> np.ndarray:
    """Apply Qwen3.5 text IMRoPE to the first ``n_rot`` head channels.

    llama.cpp maps text positions to (p,p,p,0).  Every active section therefore
    sees the same p, making IMRoPE exactly a NeoX rotate-half over the first
    n_rot channels; the remaining head channels are copied unchanged.
    """
    x = np.asarray(heads, dtype=np.int64)
    if n_rot <= 0 or n_rot > x.shape[-1] or n_rot % 2:
        raise ValueError(f"invalid partial RoPE width {n_rot} for head width {x.shape[-1]}")
    half = n_rot // 2
    if cos.shape[-1] != half or sin.shape != cos.shape:
        raise ValueError("committed Qwen3.5 RoPE table has the wrong shape")
    c = cos.reshape((1,) * (x.ndim - 2) + cos.shape)
    s = sin.reshape((1,) * (x.ndim - 2) + sin.shape)
    out = x.copy()
    x0 = x[..., :half]
    x1 = x[..., half:n_rot]
    out[..., :half] = (x0 * c - x1 * s) >> frac
    out[..., half:n_rot] = (x0 * s + x1 * c) >> frac
    return out


class _Qwen35Cache:
    """Attention KV plus convolution and DeltaNet state for one sequence."""

    __slots__ = ("k", "v", "k_buf", "v_buf", "lengths", "conv", "state", "t")

    def __init__(self, n_layers: int):
        self.k = [None] * n_layers
        self.v = [None] * n_layers
        self.k_buf = [None] * n_layers
        self.v_buf = [None] * n_layers
        self.lengths = [0] * n_layers
        self.conv = [None] * n_layers
        self.state = [None] * n_layers
        self.t = 0

    def checkpoint(self) -> tuple:
        """Shallow semantic checkpoint for transactional cached execution.

        Recurrent updates replace state arrays rather than mutating committed
        ones, and attention only appends beyond each committed length.  List
        snapshots therefore restore the complete semantic cache without a
        multi-hundred-MiB deep copy on every decoded token.
        """

        return (
            list(self.k), list(self.v), list(self.k_buf), list(self.v_buf),
            list(self.lengths), list(self.conv), list(self.state), int(self.t),
        )

    def restore(self, checkpoint: tuple) -> None:
        (self.k, self.v, self.k_buf, self.v_buf, self.lengths,
         self.conv, self.state, self.t) = checkpoint

    def extend_attention(self, li: int, kh: np.ndarray, vh: np.ndarray) -> None:
        old = self.lengths[li]
        add = int(kh.shape[1])
        need = old + add
        kb, vb = self.k_buf[li], self.v_buf[li]
        if kb is None:
            cap = max(need, 16)
            kb = np.empty((kh.shape[0], cap, kh.shape[2]), dtype=np.int64)
            vb = np.empty((vh.shape[0], cap, vh.shape[2]), dtype=np.int64)
        elif need > kb.shape[1]:
            cap = max(need, kb.shape[1] * 2)
            nk = np.empty((kb.shape[0], cap, kb.shape[2]), dtype=np.int64)
            nv = np.empty((vb.shape[0], cap, vb.shape[2]), dtype=np.int64)
            nk[:, :old] = kb[:, :old]
            nv[:, :old] = vb[:, :old]
            kb, vb = nk, nv
        kb[:, old:need] = kh
        vb[:, old:need] = vh
        self.k_buf[li], self.v_buf[li], self.lengths[li] = kb, vb, need
        self.k[li], self.v[li] = kb[:, :need], vb[:, :need]


def _full_attention(
    x_fp: np.ndarray,
    layer: dict,
    artifact: dict,
    cache: _Qwen35Cache,
    li: int,
    start: int,
    *,
    native: bool,
    runtime: _Qwen35NativeRuntime | None = None,
    debug: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    cfg = artifact["config"]
    frac = int(cfg["frac"])
    rms_eps = int(cfg.get("rmsEpsilonFp2", 1))
    m = int(x_fp.shape[0])
    h = int(cfg["n_heads"])
    hkv = int(cfg["n_heads_kv"])
    hd = int(cfg["head_dim"])
    rep = h // hkv
    qg, k, v = _project_many(
        x_fp, layer, ("wqg", "wk", "wv"), frac,
        native=native, runtime=runtime,
    )
    if debug is not None:
        debug["qg"] = np.ascontiguousarray(qg[-1:].reshape(1, h, 2, hd)).copy()
        debug["kProj"] = np.ascontiguousarray(k[-1:].reshape(1, hkv, hd)).copy()
        debug["v"] = np.ascontiguousarray(v[-1:].reshape(1, hkv, hd)).copy()

    # GGUF Q projection layout is [head, query-or-gate, channel], not
    # [all queries, all gates].  This matches qwen35.cpp's strided views.
    qg = qg.reshape(m, h, 2, hd)
    qh = qg[:, :, 0, :].transpose(1, 0, 2)
    gate = qg[:, :, 1, :]
    kh = k.reshape(m, hkv, hd).transpose(1, 0, 2)
    vh = v.reshape(m, hkv, hd).transpose(1, 0, 2)
    qh = _head_rmsnorm(
        qh, frac, layer["q_norm_gain_fp"], native=native, eps=rms_eps
    )
    kh = _head_rmsnorm(
        kh, frac, layer["k_norm_gain_fp"], native=native, eps=rms_eps
    )

    c = artifact["cos_fp"][start:start + m]
    s = artifact["sin_fp"][start:start + m]
    n_rot = int(cfg["ropeRotDim"])
    qh = _apply_partial_neox_rope(qh, c, s, frac, n_rot)
    kh = _apply_partial_neox_rope(kh, c, s, frac, n_rot)
    if debug is not None:
        debug["qRope"] = np.ascontiguousarray(qh[:, -1, :][None, ...]).copy()
        debug["kRope"] = np.ascontiguousarray(kh[:, -1, :][None, ...]).copy()
    cache_snapshot = (
        cache.k[li], cache.v[li], cache.k_buf[li], cache.v_buf[li], cache.lengths[li]
    )
    cache.extend_attention(li, kh, vh)
    length = start + m
    inv_sqrt_fp = int(cfg["attentionScaleFp"])

    out_heads = None
    if debug is None and native and m == 1:
        try:
            out_heads = attention_decode_native(
                qh[:, 0, :], cache.k[li], cache.v[li], frac, inv_sqrt_fp
            )
            if out_heads is not None:
                out_heads = out_heads.reshape(h, 1, hd)
        except (MemoryError, RuntimeError):
            out_heads = None
    if debug is None and native and m > 1 and out_heads is None:
        try:
            out_heads = attention_prefill_native(
                qh, cache.k[li], cache.v[li], start, frac, inv_sqrt_fp
            )
        except (MemoryError, RuntimeError):
            out_heads = None
    if out_heads is None:
        try:
            neg = -(1 << (frac + _NEG_INF_SHIFT))
            mask = np.arange(length)[None, :] > (start + np.arange(m))[:, None]
            out_heads = np.empty((h, m, hd), dtype=np.int64)
            debug_scores = (
                np.empty((h, length), dtype=np.int64) if debug is not None else None
            )
            debug_probs = (
                np.empty((h, length), dtype=np.int64) if debug is not None else None
            )
            for hi in range(h):
                kv = hi // rep
                scores = fixed_point_matmul(qh[hi], cache.k[li][kv].T, frac)
                scores = (scores * inv_sqrt_fp) >> frac
                scores = np.where(mask, neg, scores)
                probs = fixed_point_softmax(scores, frac)
                if debug is not None:
                    debug_scores[hi] = scores[-1]
                    debug_probs[hi] = probs[-1]
                out_heads[hi] = fixed_point_matmul(probs, cache.v[li][kv], frac)
            if debug is not None:
                debug["scores"] = debug_scores
                debug["probs"] = debug_probs
        except Exception:
            (cache.k[li], cache.v[li], cache.k_buf[li],
             cache.v_buf[li], cache.lengths[li]) = cache_snapshot
            raise

    try:
        attn = out_heads.transpose(1, 0, 2)
        attn = (attn * fixed_point_sigmoid(gate, frac)) >> frac
        if debug is not None:
            debug["head"] = np.ascontiguousarray(attn[-1:]).copy()
        projected = _project(
            attn.reshape(m, h * hd), layer, "wo", frac,
            native=native, runtime=runtime,
        )
    except BaseException:
        (cache.k[li], cache.v[li], cache.k_buf[li],
         cache.v_buf[li], cache.lengths[li]) = cache_snapshot
        raise
    return projected


def _recurrent_attention(
    x_fp: np.ndarray,
    layer: dict,
    artifact: dict,
    cache: _Qwen35Cache,
    li: int,
    *,
    native: bool,
    runtime: _Qwen35NativeRuntime | None = None,
    debug: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    cfg = artifact["config"]
    frac = int(cfg["frac"])
    m = int(x_fp.shape[0])
    state_size = int(cfg["ssmStateSize"])
    key_heads = int(cfg["ssmGroupCount"])
    value_heads = int(cfg["ssmTimeStepRank"])
    inner = int(cfg["ssmInnerSize"])
    conv_k = int(cfg["ssmConvKernel"])
    key_dim = key_heads * state_size
    conv_dim = 2 * key_dim + inner

    qkv, z, alpha, beta = _project_many(
        x_fp, layer, ("wqkv", "wz", "walpha", "wbeta"), frac,
        native=native, runtime=runtime,
    )
    if qkv.shape[1] != conv_dim:
        raise ValueError(f"recurrent QKV width {qkv.shape[1]} != {conv_dim}")
    if debug is not None:
        debug["qkv"] = np.ascontiguousarray(qkv[-1:]).copy()
        debug["z"] = np.ascontiguousarray(z[-1:]).copy()
        debug["alphaRaw"] = np.ascontiguousarray(
            alpha[-1:].reshape(1, value_heads)
        ).copy()
        debug["betaRaw"] = np.ascontiguousarray(
            beta[-1:].reshape(1, value_heads)
        ).copy()

    old_conv = cache.conv[li]
    if old_conv is None:
        old_conv = np.zeros((conv_k - 1, conv_dim), dtype=np.int64)
    window = np.concatenate((old_conv, qkv), axis=0)
    kernel = np.asarray(layer["conv_weight_fp"], dtype=np.int64)
    if kernel.shape != (conv_dim, conv_k):
        raise ValueError(f"convolution weight shape {kernel.shape} != {(conv_dim, conv_k)}")
    conv_acc = np.zeros((m, conv_dim), dtype=np.int64)
    for j in range(conv_k):
        conv_acc += window[j:j + m] * kernel[:, j]
    conv = fixed_point_silu(conv_acc >> frac, frac, native=native)
    if debug is not None:
        debug["conv"] = np.ascontiguousarray(conv[-1:]).copy()
    next_conv = np.ascontiguousarray(window[-(conv_k - 1):])

    q = conv[:, :key_dim].reshape(m, key_heads, state_size)
    k = conv[:, key_dim:2 * key_dim].reshape(m, key_heads, state_size)
    v = conv[:, 2 * key_dim:].reshape(m, value_heads, state_size)
    q = _fixed_l2_norm(q.reshape(-1, state_size), frac).reshape(m, key_heads, state_size)
    k = _fixed_l2_norm(k.reshape(-1, state_size), frac).reshape(m, key_heads, state_size)
    head_map = (
        runtime.head_map(value_heads, key_heads)
        if runtime is not None
        else np.arange(value_heads, dtype=np.int64) % key_heads
    )
    q = q[:, head_map, :]
    k = k[:, head_map, :]
    if debug is not None:
        debug["q"] = np.ascontiguousarray(q[-1:]).copy()
        debug["k"] = np.ascontiguousarray(k[-1:]).copy()

    beta = fixed_point_sigmoid(beta.reshape(m, value_heads), frac)
    softplus = fixed_point_softplus_lut(
        alpha.reshape(m, value_heads) + layer["dt_bias_fp"][None, :], artifact
    )
    gate = (softplus * layer["ssm_a_fp"][None, :]) >> frac
    if np.any(gate > 0):
        raise OverflowError("Qwen3.5 recurrent decay gate became positive")
    decay = fixed_point_exp_negative_lut(gate, artifact)
    if debug is not None:
        debug["decay"] = np.ascontiguousarray(decay[-1:]).copy()
        debug["beta"] = np.ascontiguousarray(beta[-1:]).copy()

    state = cache.state[li]
    if state is None:
        state = np.zeros((value_heads, state_size, state_size), dtype=np.int64)
    else:
        # Never mutate a committed recurrent state in place.  This lets the
        # caller roll back a later layer/gate/projection failure by restoring
        # only cache references.
        state = np.ascontiguousarray(state.copy())
    output = np.empty((m, value_heads, state_size), dtype=np.int64)
    state_frac = int(cfg.get("ssmStateFrac", min(30, frac + 14)))
    outer_shift = 2 * frac - state_frac
    if outer_shift < 0:
        raise ValueError(
            f"ssmStateFrac {state_frac} exceeds the exact Q{frac} outer-product width {2 * frac}"
        )
    inv_sqrt_fp = int(cfg["gdnScaleFp"])
    native_output = None
    if debug is None and native and _native_gdn_enabled():
        try:
            native_output = gdn_prefill_native(
                state, q, k, v, beta, decay,
                frac, state_frac, outer_shift, inv_sqrt_fp,
            )
        except MemoryError:
            native_output = None
    if native_output is not None:
        output[:] = native_output
    else:
        for t in range(m):
            state[:] = (state * decay[t, :, None, None]) >> frac
            # state is Q(state_frac), while q/k/v/beta are Q(frac).
            # Prediction is brought back to Q(frac) to form delta with v.
            pred = np.einsum("hij,hi->hj", state, k[t], optimize=True) >> state_frac
            delta = ((v[t] - pred) * beta[t, :, None]) >> frac
            state += (k[t, :, :, None] * delta[:, None, :]) >> outer_shift
            if debug is not None and t == m - 1:
                debug["pred"] = np.ascontiguousarray(pred[None, ...]).copy()
                debug["delta"] = np.ascontiguousarray(delta[None, ...]).copy()
                debug["state"] = np.ascontiguousarray(state).copy()
            # Preserve Q(state_frac) through the tiny pre-normalization GDN score.
            out = np.einsum("hij,hi->hj", state, q[t], optimize=True) >> frac
            output[t] = (out * inv_sqrt_fp) >> frac
    normed = _rmsnorm(
        output.reshape(m * value_heads, state_size),
        frac,
        layer["ssm_norm_gain_fp"],
        native=native,
        eps=int(cfg.get(
            "ssmRmsEpsilonFp2",
            round(float(cfg.get("rmsEpsilon", 1e-6)) * (1 << (2 * state_frac))),
        )),
    ).reshape(m, value_heads, state_size)
    zg = fixed_point_silu(z.reshape(m, value_heads, state_size), frac, native=native)
    gated = (normed * zg) >> frac
    if debug is not None:
        debug["gated"] = np.ascontiguousarray(gated[-1:]).copy()
    projected = _project(
        gated.reshape(m, inner), layer, "wout", frac,
        native=native, runtime=runtime,
    )
    cache.state[li] = state
    cache.conv[li] = next_conv
    return projected


def _ffn(x_fp: np.ndarray, layer: dict, frac: int, *, native: bool,
         runtime: _Qwen35NativeRuntime | None = None) -> np.ndarray:
    gate, up = _project_many(
        x_fp, layer, ("w1", "wu"), frac, native=native, runtime=runtime
    )
    hidden = (fixed_point_silu(gate, frac, native=native) * up) >> frac
    return _project(hidden, layer, "w2", frac, native=native, runtime=runtime)


@dataclass
class BonsaiQwen35ReferenceModel:
    """Native fixed-point Bonsai-27B model with recurrent/KV cached decode."""

    receipt_verify_cached_threshold: ClassVar[int] = 8
    artifact: dict

    def __post_init__(self) -> None:
        if str(self.cfg.get("architecture")) != "qwen35":
            raise ValueError(f"BonsaiQwen35ReferenceModel needs architecture='qwen35', got {self.cfg.get('architecture')!r}")
        self._native = False
        self._native_runtime = None
        self._model_executor = None

    @property
    def cfg(self) -> dict:
        return self.artifact["config"]

    def enable_native(self) -> bool:
        self._native = bool(q1_native_available())
        if self._native and self._native_runtime is None:
            self._native_runtime = _Qwen35NativeRuntime(self.artifact)
        executor_enabled = os.environ.get(
            "TRINOTE_BONSAI35_MODEL_EXECUTOR", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        if self._native and executor_enabled and self._model_executor is None:
            self._model_executor = Bonsai35NativeExecutor.create(self.artifact)
        return self._native

    def native_executor_stats(self) -> dict[str, int] | None:
        return self._model_executor.stats() if self._model_executor is not None else None

    def debug_layer_intermediates(
        self,
        token_ids: list[int] | np.ndarray,
        layer_index: int,
        *,
        native_primitives: bool | None = None,
    ) -> dict[str, np.ndarray]:
        """Return exact layer boundaries plus bounded last-token internals.

        This deliberately bypasses the resident model executor, making it a
        named-intermediate parity oracle for that executor's debug-only trace.
        """

        ids = _validate_b35_token_ids(
            token_ids, int(self.cfg["vocab"]), where="Bonsai-27B layer tracing"
        )
        layer_index = int(layer_index)
        if layer_index < 0 or layer_index >= len(self.artifact["layers"]):
            raise IndexError(f"Bonsai-27B trace layer is out of range: {layer_index}")
        if ids.size > int(self.artifact["cos_fp"].shape[0]):
            raise ValueError("Bonsai-27B layer trace exceeds committed context")
        native = self._native if native_primitives is None else bool(native_primitives)
        runtime = self._native_runtime if native else None
        cache = _Qwen35Cache(len(self.artifact["layers"]))
        frac = int(self.cfg["frac"])
        rms_eps = int(self.cfg.get("rmsEpsilonFp2", 1))
        x = q1_rows_fp(
            self.artifact["embed_bits"],
            self.artifact["embed_scale_fp"],
            ids,
            frac,
        )
        for li, layer in enumerate(self.artifact["layers"]):
            internals: dict[str, np.ndarray] | None = (
                {} if li == layer_index else None
            )
            n1 = _rmsnorm(
                x, frac, layer["n1_gain_fp"], native=native, eps=rms_eps
            )
            if layer["kind"] == "recurrent":
                branch = _recurrent_attention(
                    n1, layer, self.artifact, cache, li,
                    native=native, runtime=runtime, debug=internals,
                )
            elif layer["kind"] == "attention":
                branch = _full_attention(
                    n1, layer, self.artifact, cache, li, 0,
                    native=native, runtime=runtime, debug=internals,
                )
            else:
                raise ValueError(f"unknown Qwen3.5 layer kind {layer['kind']!r}")
            residual = x + branch
            n2 = _rmsnorm(
                residual, frac, layer["n2_gain_fp"], native=native, eps=rms_eps
            )
            ffn = _ffn(n2, layer, frac, native=native, runtime=runtime)
            output = residual + ffn
            if li == layer_index:
                boundaries = {
                    "n1": np.ascontiguousarray(n1),
                    "branch": np.ascontiguousarray(branch),
                    "residual": np.ascontiguousarray(residual),
                    "n2": np.ascontiguousarray(n2),
                    "ffn": np.ascontiguousarray(ffn),
                    "output": np.ascontiguousarray(output),
                }
                assert internals is not None
                boundaries.update(internals)
                return boundaries
            x = output
        raise AssertionError("unreachable Bonsai-27B trace layer")

    def enable_fast(self, *, check_ram: bool = True, cache_output: bool = True) -> bool:
        # Expanding 27B of signs to int8 would consume tens of GiB.  The packed
        # native kernel is the only intentional fast path for this model.
        del check_ram, cache_output
        return self.enable_native()

    def _output_linear(self, x_fp: np.ndarray) -> np.ndarray:
        return _project(
            x_fp,
            self.artifact,
            "output",
            int(self.cfg["frac"]),
            native=self._native,
            runtime=self._native_runtime,
        )

    def _output_argmax(self, x_fp: np.ndarray) -> np.ndarray:
        frac = int(self.cfg["frac"])
        if self._native:
            try:
                ids = q1_argmax_native(
                    x_fp,
                    self.artifact["output_bits"],
                    self.artifact["output_scale_fp"],
                    frac,
                    lut32=(
                        self._native_runtime.use_lut32(None)
                        if self._native_runtime is not None else True
                    ),
                )
            except (MemoryError, RuntimeError):
                ids = None
            if ids is not None:
                return ids
        return np.asarray(self._output_linear(x_fp).argmax(axis=1), dtype=np.int64)

    def _run_layers(self, token_ids, cache: _Qwen35Cache) -> np.ndarray:
        ids = _validate_b35_token_ids(
            token_ids, int(self.cfg["vocab"]), where="Qwen3.5 inference"
        )
        checkpoint = cache.checkpoint()
        try:
            return self._run_layers_validated(ids, cache)
        except BaseException:
            cache.restore(checkpoint)
            raise

    def _run_layers_validated(self, ids: np.ndarray, cache: _Qwen35Cache) -> np.ndarray:
        start = int(cache.t)
        if start + ids.size > int(self.artifact["cos_fp"].shape[0]):
            raise ValueError("Qwen3.5 run exceeds committed context/RoPE rows")
        frac = int(self.cfg["frac"])
        rms_eps = int(self.cfg.get("rmsEpsilonFp2", 1))
        x = q1_rows_fp(
            self.artifact["embed_bits"], self.artifact["embed_scale_fp"], ids, frac
        )
        for li, layer in enumerate(self.artifact["layers"]):
            n1 = _rmsnorm(
                x, frac, layer["n1_gain_fp"], native=self._native, eps=rms_eps
            )
            if layer["kind"] == "recurrent":
                attn = _recurrent_attention(
                    n1, layer, self.artifact, cache, li, native=self._native,
                    runtime=self._native_runtime,
                )
            elif layer["kind"] == "attention":
                attn = _full_attention(
                    n1, layer, self.artifact, cache, li, start, native=self._native,
                    runtime=self._native_runtime,
                )
            else:
                raise ValueError(f"unknown Qwen3.5 layer kind {layer['kind']!r}")
            x = x + attn
            n2 = _rmsnorm(
                x, frac, layer["n2_gain_fp"], native=self._native, eps=rms_eps
            )
            x = x + _ffn(
                n2, layer, frac, native=self._native, runtime=self._native_runtime
            )
        cache.t = start + int(ids.size)
        return x

    def forward(self, token_ids: list[int] | np.ndarray, *, last_only: bool = False) -> np.ndarray:
        if self._model_executor is not None:
            x = self._model_executor.prefill(token_ids)
        else:
            cache = _Qwen35Cache(len(self.artifact["layers"]))
            x = self._run_layers(token_ids, cache)
        frac = int(self.cfg["frac"])
        x = _rmsnorm(
            x,
            frac,
            self.artifact["final_norm_gain_fp"],
            native=False,
            eps=int(self.cfg.get("rmsEpsilonFp2", 1)),
        )
        return self._output_linear(x[-1:] if last_only else x)

    def forward_fast(self, token_ids: list[int] | np.ndarray, *, last_only: bool = False) -> np.ndarray:
        return self.forward(token_ids, last_only=last_only)

    def teacher_forced_logits(self, token_ids: list[int] | np.ndarray) -> np.ndarray:
        return self.forward(token_ids)

    def prefill_logits(self, token_ids) -> np.ndarray:
        if self._model_executor is not None:
            return self._model_executor.prefill_logits(token_ids)
        cache = _Qwen35Cache(len(self.artifact["layers"]))
        x = self._run_layers(token_ids, cache)
        frac = int(self.cfg["frac"])
        last = _rmsnorm(
            x[-1:],
            frac,
            self.artifact["final_norm_gain_fp"],
            native=self._native,
            eps=int(self.cfg.get("rmsEpsilonFp2", 1)),
        )
        return self._output_linear(last)

    def generate_cached(self, token_ids, n_new, pick, *, eos=None, on_token=None) -> list[int]:
        prompt = _validate_b35_token_ids(
            token_ids, int(self.cfg["vocab"]), where="Qwen3.5 generation"
        ).tolist()
        if int(n_new) <= 0:
            return []
        window = min(int(self.cfg["context_len"]), int(self.artifact["cos_fp"].shape[0]))
        if len(prompt) + int(n_new) > window:
            return self._generate_uncached(prompt, n_new, pick, eos, on_token)
        if self._model_executor is not None:
            row = self._model_executor.prefill_logits(prompt)[0]
        else:
            cache = _Qwen35Cache(len(self.artifact["layers"]))
            x = self._run_layers(prompt, cache)
        seq = list(prompt)
        out: list[int] = []
        frac = int(self.cfg["frac"])
        for step in range(int(n_new)):
            if self._model_executor is None:
                last = _rmsnorm(
                    x[-1:],
                    frac,
                    self.artifact["final_norm_gain_fp"],
                    native=self._native,
                    eps=int(self.cfg.get("rmsEpsilonFp2", 1)),
                )
                row = self._output_linear(last)[0]
            tok = _validate_b35_token_id(
                pick(row, len(seq), seq), int(self.cfg["vocab"]),
                where="Qwen3.5 generated",
            )
            seq.append(tok)
            out.append(tok)
            if on_token is not None:
                on_token(tok)
            if eos is not None and tok == int(eos):
                break
            if step + 1 < int(n_new):
                if self._model_executor is not None:
                    row = self._model_executor.decode_logits(tok)[0]
                else:
                    x = self._run_layers([tok], cache)
        return out

    def generate_greedy_tokens_cached(self, token_ids, n_new, *, eos=None, on_token=None) -> list[int]:
        prompt = _validate_b35_token_ids(
            token_ids, int(self.cfg["vocab"]), where="Qwen3.5 generation"
        ).tolist()
        if int(n_new) <= 0:
            return []
        window = min(int(self.cfg["context_len"]), int(self.artifact["cos_fp"].shape[0]))
        if len(prompt) + int(n_new) > window:
            return self._generate_uncached(
                prompt,
                n_new,
                lambda row, _pos, _hist: int(np.asarray(row).argmax()),
                eos,
                on_token,
            )
        if self._model_executor is not None:
            tok = self._model_executor.prefill_argmax(prompt)
        else:
            cache = _Qwen35Cache(len(self.artifact["layers"]))
            x = self._run_layers(prompt, cache)
        out: list[int] = []
        frac = int(self.cfg["frac"])
        for step in range(int(n_new)):
            if self._model_executor is None:
                last = _rmsnorm(
                    x[-1:],
                    frac,
                    self.artifact["final_norm_gain_fp"],
                    native=self._native,
                    eps=int(self.cfg.get("rmsEpsilonFp2", 1)),
                )
                tok = int(self._output_argmax(last)[0])
            out.append(tok)
            if on_token is not None:
                on_token(tok)
            if eos is not None and tok == int(eos):
                break
            if step + 1 < int(n_new):
                if self._model_executor is not None:
                    tok = self._model_executor.decode_argmax(tok)
                else:
                    x = self._run_layers([tok], cache)
        return out

    def _generate_uncached(self, token_ids, n_new, pick, eos, on_token) -> list[int]:
        ctx = int(self.cfg["context_len"])
        seq, out = list(token_ids), []
        for _ in range(int(n_new)):
            row = self.forward(seq[-ctx:], last_only=True)[0]
            tok = _validate_b35_token_id(
                pick(row, len(seq), seq), int(self.cfg["vocab"]),
                where="Qwen3.5 generated",
            )
            seq.append(tok)
            out.append(tok)
            if on_token is not None:
                on_token(tok)
            if eos is not None and tok == int(eos):
                break
        return out

    def generate_greedy(
        self,
        token_ids: list[int],
        n_new: int,
        *,
        rep_penalty_fp: int = 0,
        no_repeat_ngram: int = 0,
    ) -> list[int]:
        frac = int(self.cfg["frac"])
        prompt = _validate_b35_token_ids(
            token_ids, int(self.cfg["vocab"]), where="Qwen3.5 generation"
        ).tolist()

        def pick(row, _pos, hist):
            if rep_penalty_fp or no_repeat_ngram:
                row = apply_rep_penalty(row, hist, rep_penalty_fp, no_repeat_ngram, frac)
            return int(np.asarray(row).argmax())

        return prompt + self.generate_cached(prompt, n_new, pick)


def random_bonsai35_artifact(
    cfg_params: dict | None = None,
    *,
    seq_len: int = 32,
    seed: int = 0,
) -> dict:
    """Small hybrid artifact used by architecture and round-trip tests."""
    from ..model.rope_v2 import build_rope_tables
    from .import_bonsai35_gguf import build_bonsai35_luts

    p = dict(cfg_params or {})
    d = int(p.get("dModel", 128))
    heads = int(p.get("nHeads", 2))
    hkv = int(p.get("nHeadsKv", 1))
    hd = int(p.get("headDim", d // heads))
    dff = int(p.get("dFfn", 128))
    vocab = int(p.get("vocab", 128))
    n_layers = int(p.get("nLayers", 4))
    frac = int(p.get("fpFracBits", 16))
    interval = int(p.get("fullAttentionInterval", 4))
    state = int(p.get("ssmStateSize", 32))
    groups = int(p.get("ssmGroupCount", 2))
    value_heads = int(p.get("ssmTimeStepRank", 4))
    inner = state * value_heads
    conv_k = int(p.get("ssmConvKernel", 4))
    n_rot = int(p.get("ropeRotDim", min(32, hd)))
    rope_base = int(p.get("ropeBase", 10_000_000))
    rng = np.random.default_rng(seed)

    def gain(n):
        return np.full(n, 1 << frac, dtype=np.int64)

    def q1(out_f, in_f):
        if in_f % 128:
            raise ValueError("random Qwen3.5 Q1 input widths must be divisible by 128")
        blocks = in_f // 128
        bits = rng.integers(0, 256, size=(out_f, blocks, 16), dtype=np.uint8)
        scale = np.full((out_f, blocks), round(0.01 * (1 << frac)), dtype=np.int32)
        return bits, scale

    layers = []
    key_dim = groups * state
    conv_dim = 2 * key_dim + inner
    for i in range(n_layers):
        recurrent = (i + 1) % interval != 0
        layer = {"kind": "recurrent" if recurrent else "attention", "n1_gain_fp": gain(d), "n2_gain_fp": gain(d)}
        for name, out_f, in_f in (("w1", dff, d), ("wu", dff, d), ("w2", d, dff)):
            layer[f"{name}_bits"], layer[f"{name}_scale_fp"] = q1(out_f, in_f)
        if recurrent:
            for name, out_f, in_f in (
                ("wqkv", conv_dim, d), ("wz", inner, d),
                ("walpha", value_heads, d), ("wbeta", value_heads, d),
                ("wout", d, inner),
            ):
                layer[f"{name}_bits"], layer[f"{name}_scale_fp"] = q1(out_f, in_f)
            layer.update({
                "conv_weight_fp": rng.integers(-256, 257, size=(conv_dim, conv_k), dtype=np.int64),
                "dt_bias_fp": np.zeros(value_heads, dtype=np.int64),
                "ssm_a_fp": np.full(value_heads, -(1 << (frac - 1)), dtype=np.int64),
                "ssm_norm_gain_fp": gain(state),
            })
        else:
            for name, out_f, in_f in (
                ("wqg", 2 * heads * hd, d), ("wk", hkv * hd, d),
                ("wv", hkv * hd, d), ("wo", d, heads * hd),
            ):
                layer[f"{name}_bits"], layer[f"{name}_scale_fp"] = q1(out_f, in_f)
            layer.update({"q_norm_gain_fp": gain(hd), "k_norm_gain_fp": gain(hd)})
        layers.append(layer)

    embed_bits, embed_scale = q1(vocab, d)
    output_bits, output_scale = q1(vocab, d)
    cos, sin = build_rope_tables(seq_len, n_rot, rope_base, frac)
    soft, exp_neg, lut_cfg = build_bonsai35_luts(frac)
    return {
        "config": {
            "architecture": "qwen35", "modelName": "tiny-bonsai35",
            "dModel": d, "nLayers": n_layers, "n_heads": heads, "n_heads_kv": hkv,
            "head_dim": hd, "dFfn": dff, "vocab": vocab, "context_len": seq_len,
            "sourceContextLen": seq_len, "frac": frac, "rmsEpsilon": 1e-6,
            "rmsEpsilonFp2": round(1e-6 * (1 << (2 * frac))),
            "ropeBase": rope_base, "ropeRotDim": n_rot,
            "ropeSections": [n_rot // 6 + (1 if j < n_rot // 2 % 3 else 0) for j in range(3)] + [0],
            "ropeType": "imrope-text", "fullAttentionInterval": interval,
            "ssmConvKernel": conv_k, "ssmGroupCount": groups, "ssmInnerSize": inner,
            "ssmStateSize": state, "ssmTimeStepRank": value_heads,
            "ssmStateFrac": min(30, frac + 14),
            "ssmRmsEpsilonFp2": round(1e-6 * (1 << (2 * min(30, frac + 14)))),
            "attentionScaleFp": round((1.0 / math.sqrt(hd)) * (1 << frac)),
            "gdnScaleFp": round((1.0 / math.sqrt(state)) * (1 << frac)),
            **lut_cfg,
        },
        "embed_bits": embed_bits, "embed_scale_fp": embed_scale,
        "output_bits": output_bits, "output_scale_fp": output_scale,
        "final_norm_gain_fp": gain(d), "cos_fp": cos, "sin_fp": sin,
        "softplus_lut_fp": soft, "exp_neg_lut_fp": exp_neg, "layers": layers,
    }
