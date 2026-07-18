"""Canonical tensor/cache traces for Bonsai-27B optimization parity gates.

The optimized CPU and GPU engines are allowed to change scheduling, tiling,
storage width (under an exact range guard), and runtime layout.  They are not
allowed to change the committed integer graph.  This module records compact
SHA-256 checkpoints for that graph so a fast path can be compared at layer and
cache boundaries without persisting multi-gigabyte tensors.
"""
from __future__ import annotations

import hashlib
import json
import sys
from typing import Any

import numpy as np

from .reference_bonsai import _rmsnorm, q1_rows_fp
from .q1_native import _validate_b35_token_ids
from .reference_bonsai35 import (
    _Qwen35Cache,
    _ffn,
    _full_attention,
    _project,
    _recurrent_attention,
)

TRACE_FORMAT = "trinote-bonsai35-trace/1"
CACHE_COMMITMENT_FORMAT = "trinote-bonsai35-cache-commitment/1"


def tensor_digest(value: np.ndarray) -> str:
    """Hash an ndarray with an explicit dtype/shape and little-endian payload.

    Shape and dtype are part of the commitment, so a flattened/transposed or
    silently narrowed tensor cannot compare equal merely because its byte
    payload happens to match.  Canonical little endian makes the trace portable
    even though the production Bonsai-27B launcher currently targets Linux
    x86-64.
    """

    a = np.asarray(value)
    if a.dtype.hasobject:
        raise TypeError("object arrays are not valid canonical trace tensors")
    dtype_le = a.dtype.newbyteorder("<")
    if a.dtype.byteorder == ">" or (a.dtype.byteorder == "=" and sys.byteorder == "big"):
        a = a.byteswap().view(dtype_le)
    else:
        a = a.astype(dtype_le, copy=False)
    a = np.ascontiguousarray(a)
    descriptor = json.dumps(
        {"dtype": dtype_le.str, "shape": list(a.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    h = hashlib.sha256()
    h.update(b"trinote-tensor-trace-v1\0")
    h.update(len(descriptor).to_bytes(8, "little"))
    h.update(descriptor)
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def canonical_cache_record(
    layer_kinds: list[str] | tuple[str, ...],
    *,
    position: int,
    tensor_for,
    last_residual: np.ndarray,
) -> dict[str, Any]:
    """Describe executor state without encoding a producer's storage layout.

    ``tensor_for(layer, name)`` must return the logical int64 tensor named by
    ``name``: recurrent layers expose ``state`` and ``conv``; attention layers
    expose only the valid prefix of ``k`` and ``v``.  Native executors can copy
    their strided resident caches into those shapes, while the Python oracle
    supplies its arrays directly.  The resulting record is therefore directly
    comparable across producers, unlike runtime-specific diagnostic hashes.

    Only the final residual row is committed.  A prompt prefill may retain all
    residual rows in the oracle while the resident executor intentionally keeps
    one, so committing the common final row preserves the semantic boundary.
    """

    if type(position) is not int or position < 0:
        raise ValueError("cache commitment position must be a non-negative integer")
    residual = np.asarray(last_residual)
    if residual.dtype != np.int64 or residual.ndim not in (1, 2) or residual.size == 0:
        raise ValueError("cache commitment residual must be a non-empty int64 vector/matrix")
    if residual.ndim == 1:
        residual = residual.reshape(1, -1)
    else:
        residual = residual[-1:]
    residual = np.ascontiguousarray(residual)

    rows: list[dict[str, Any]] = []
    for layer, kind_value in enumerate(layer_kinds):
        kind = str(kind_value)
        names = ("state", "conv") if kind == "recurrent" else ("k", "v") if kind == "attention" else ()
        if not names:
            raise ValueError(f"unknown Qwen3.5 layer kind {kind!r} at layer {layer}")
        row: dict[str, Any] = {"kind": kind, "layer": layer}
        for name in names:
            value = np.asarray(tensor_for(layer, name))
            if value.dtype != np.int64:
                raise ValueError(
                    f"cache commitment layer {layer} {name} must be int64"
                )
            # Oracle KV grows in a capacity-backed [Hkv, capacity, hd]
            # allocation and exposes only [:, :position].  With Hkv > 1 that
            # logical prefix is intentionally strided even though it contains
            # exactly the cache values being committed.  Commit logical shape
            # and values, independent of capacity/layout, just as
            # tensor_digest already does for Fortran and endian variants.
            value = np.ascontiguousarray(value)
            row[name] = tensor_digest(value)
        rows.append(row)
    return {
        "format": CACHE_COMMITMENT_FORMAT,
        "lastResidual": tensor_digest(residual),
        "layers": rows,
        "position": position,
    }


def canonical_cache_digest(record: dict[str, Any]) -> str:
    """Hash a canonical cache record using its portable JSON encoding."""

    if not isinstance(record, dict) or record.get("format") != CACHE_COMMITMENT_FORMAT:
        raise ValueError("not a canonical Bonsai-27B cache commitment record")
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _cache_checkpoint(cache: _Qwen35Cache, li: int, kind: str) -> dict[str, Any]:
    if kind == "recurrent":
        state = cache.state[li]
        conv = cache.conv[li]
        if state is None or conv is None:
            raise AssertionError(f"recurrent cache for layer {li} was not populated")
        return {
            "state": tensor_digest(state),
            "stateShape": list(state.shape),
            "conv": tensor_digest(conv),
            "convShape": list(conv.shape),
        }
    if kind == "attention":
        k = cache.k[li]
        v = cache.v[li]
        if k is None or v is None:
            raise AssertionError(f"attention cache for layer {li} was not populated")
        return {
            "k": tensor_digest(k),
            "kShape": list(k.shape),
            "v": tensor_digest(v),
            "vShape": list(v.shape),
        }
    raise ValueError(f"unknown Qwen3.5 layer kind {kind!r}")


def _run_layers_with_trace(
    artifact: dict,
    token_ids: list[int] | np.ndarray,
    cache: _Qwen35Cache,
    *,
    native: bool,
    include_intermediates: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    ids = _validate_b35_token_ids(
        token_ids, int(artifact["config"]["vocab"]), where="Bonsai-27B tracing"
    )
    checkpoint = cache.checkpoint()
    try:
        return _run_layers_with_trace_validated(
            artifact, ids, cache, native=native,
            include_intermediates=include_intermediates,
        )
    except BaseException:
        cache.restore(checkpoint)
        raise


def _run_layers_with_trace_validated(
    artifact: dict,
    ids: np.ndarray,
    cache: _Qwen35Cache,
    *,
    native: bool,
    include_intermediates: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cfg = artifact["config"]
    frac = int(cfg["frac"])
    start = int(cache.t)
    if start + ids.size > int(artifact["cos_fp"].shape[0]):
        raise ValueError("trace exceeds the committed RoPE/context rows")
    rms_eps = int(cfg.get("rmsEpsilonFp2", 1))
    x = q1_rows_fp(
        artifact["embed_bits"], artifact["embed_scale_fp"], ids, frac
    )
    rows: list[dict[str, Any]] = []
    for li, layer in enumerate(artifact["layers"]):
        kind = str(layer["kind"])
        n1 = _rmsnorm(
            x, frac, layer["n1_gain_fp"], native=native, eps=rms_eps
        )
        if kind == "recurrent":
            branch = _recurrent_attention(
                n1, layer, artifact, cache, li, native=native
            )
        elif kind == "attention":
            branch = _full_attention(
                n1, layer, artifact, cache, li, start, native=native
            )
        else:
            raise ValueError(f"unknown Qwen3.5 layer kind {kind!r}")
        residual = x + branch
        n2 = _rmsnorm(
            residual, frac, layer["n2_gain_fp"], native=native, eps=rms_eps
        )
        ffn = _ffn(n2, layer, frac, native=native)
        x = residual + ffn
        row: dict[str, Any] = {
            "layer": li,
            "kind": kind,
            "output": tensor_digest(x),
            "outputShape": list(x.shape),
            "cache": _cache_checkpoint(cache, li, kind),
        }
        if include_intermediates:
            row.update({
                "n1": tensor_digest(n1),
                "branch": tensor_digest(branch),
                "residual": tensor_digest(residual),
                "n2": tensor_digest(n2),
                "ffn": tensor_digest(ffn),
            })
        rows.append(row)
    cache.t = start + int(ids.size)
    return x, rows


def trace_prefill(
    artifact: dict,
    token_ids: list[int] | np.ndarray,
    *,
    native: bool = False,
    include_intermediates: bool = True,
) -> dict[str, Any]:
    """Trace one fresh-cache prefill and its next greedy token."""

    ids = _validate_b35_token_ids(
        token_ids, int(artifact["config"]["vocab"]), where="Bonsai-27B tracing"
    ).tolist()
    cache = _Qwen35Cache(len(artifact["layers"]))
    x, layers = _run_layers_with_trace(
        artifact,
        ids,
        cache,
        native=native,
        include_intermediates=include_intermediates,
    )
    cfg = artifact["config"]
    frac = int(cfg["frac"])
    final = _rmsnorm(
        x[-1:],
        frac,
        artifact["final_norm_gain_fp"],
        native=native,
        eps=int(cfg.get("rmsEpsilonFp2", 1)),
    )
    logits = _project(final, artifact, "output", frac, native=native)
    next_token = int(np.asarray(logits[0]).argmax())
    return {
        "format": TRACE_FORMAT,
        "architecture": str(cfg.get("architecture", "")),
        "frac": frac,
        "inputIds": ids,
        "inputLength": len(ids),
        "embedding": tensor_digest(q1_rows_fp(
            artifact["embed_bits"], artifact["embed_scale_fp"],
            np.asarray(ids, dtype=np.int64), frac,
        )),
        "layers": layers,
        "finalNorm": tensor_digest(final),
        "logits": tensor_digest(logits),
        "nextGreedyToken": next_token,
    }


def trace_cached_greedy(
    artifact: dict,
    token_ids: list[int] | np.ndarray,
    n_new: int,
    *,
    native: bool = False,
    include_layer_traces: bool = False,
) -> dict[str, Any]:
    """Trace a cached greedy continuation, including cache commitments per step.

    `include_layer_traces=False` keeps a 32-token real-model manifest compact by
    recording an aggregate commitment for each step.  Set it for parity triage.
    """

    ids = _validate_b35_token_ids(
        token_ids, int(artifact["config"]["vocab"]), where="Bonsai-27B tracing"
    ).tolist()
    if int(n_new) < 0:
        raise ValueError("cached trace requires n_new >= 0")
    cache = _Qwen35Cache(len(artifact["layers"]))
    x, prefill_layers = _run_layers_with_trace(
        artifact, ids, cache, native=native, include_intermediates=False
    )
    cfg = artifact["config"]
    frac = int(cfg["frac"])
    rms_eps = int(cfg.get("rmsEpsilonFp2", 1))
    output_ids: list[int] = []
    steps: list[dict[str, Any]] = []
    for step in range(int(n_new)):
        final = _rmsnorm(
            x[-1:], frac, artifact["final_norm_gain_fp"],
            native=native, eps=rms_eps,
        )
        logits = _project(final, artifact, "output", frac, native=native)
        token = int(np.asarray(logits[0]).argmax())
        output_ids.append(token)
        item: dict[str, Any] = {
            "step": step,
            "token": token,
            "finalNorm": tensor_digest(final),
            "logits": tensor_digest(logits),
        }
        # Consume every generated token, including the final one.  Ordinary
        # generation can stop before feeding the last token back because no
        # further logits are needed, but a golden *state* trace must commit
        # the recurrent/conv/KV state after all n_new cached tokens.
        x, layer_rows = _run_layers_with_trace(
            artifact, [token], cache, native=native,
            include_intermediates=False,
        )
        layer_commit = hashlib.sha256(
            json.dumps(layer_rows, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        item["postDecodeLayers"] = layer_commit
        if include_layer_traces:
            item["layers"] = layer_rows
        steps.append(item)
    return {
        "format": TRACE_FORMAT,
        "architecture": str(cfg.get("architecture", "")),
        "frac": frac,
        "inputIds": ids,
        "prefillLayers": prefill_layers if include_layer_traces else hashlib.sha256(
            json.dumps(prefill_layers, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "outputIds": output_ids,
        "steps": steps,
    }


def assert_trace_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Raise a path-localized assertion for two canonical JSON traces."""

    def walk(a: Any, e: Any, path: str) -> None:
        if type(a) is not type(e):
            raise AssertionError(f"trace mismatch at {path}: {type(a).__name__} != {type(e).__name__}")
        if isinstance(e, dict):
            if set(a) != set(e):
                raise AssertionError(
                    f"trace keys differ at {path}: actual-only={sorted(set(a) - set(e))}, "
                    f"expected-only={sorted(set(e) - set(a))}"
                )
            for key in sorted(e):
                walk(a[key], e[key], f"{path}.{key}")
            return
        if isinstance(e, list):
            if len(a) != len(e):
                raise AssertionError(f"trace length differs at {path}: {len(a)} != {len(e)}")
            for i, (av, ev) in enumerate(zip(a, e)):
                walk(av, ev, f"{path}[{i}]")
            return
        if a != e:
            raise AssertionError(f"trace mismatch at {path}: {a!r} != {e!r}")

    walk(actual, expected, "$ ")
