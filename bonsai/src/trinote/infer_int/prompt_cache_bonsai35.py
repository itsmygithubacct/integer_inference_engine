"""Content-addressed deterministic prefix caches for Bonsai-27B.

The cache is an operational optimization, not part of the committed model.  A
cache entry is accepted only when its artifact digest, exact input IDs, graph
format, tensor shapes, and aggregate tensor commitment all match.  Loaded
values are the same int64 recurrent/KV state the canonical graph produced.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from ..notary_paths import notary_home
from .reference_bonsai import _rmsnorm
from .reference_bonsai35 import BonsaiQwen35ReferenceModel, _Qwen35Cache
from .trace_bonsai35 import tensor_digest

PROMPT_CACHE_FORMAT = "trinote-bonsai35-prompt-cache/1"


@dataclass
class Bonsai35PromptState:
    artifact_digest: str
    input_ids: tuple[int, ...]
    last_x: np.ndarray
    cache: _Qwen35Cache


def prompt_cache_key(artifact_digest: str, input_ids: list[int] | tuple[int, ...]) -> str:
    body = json.dumps(
        {
            "artifactDigest": str(artifact_digest),
            "format": PROMPT_CACHE_FORMAT,
            "inputIds": [int(v) for v in input_ids],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(b"trinote-bonsai35-prompt-cache-key\0" + body).hexdigest()


def default_prompt_cache_path(
    artifact_digest: str, input_ids: list[int] | tuple[int, ...]
) -> Path:
    return notary_home() / "prompt-cache" / "bonsai35" / (
        prompt_cache_key(artifact_digest, input_ids) + ".safetensors"
    )


def build_prompt_state(
    model: BonsaiQwen35ReferenceModel,
    input_ids: list[int] | tuple[int, ...],
    artifact_digest: str,
) -> Bonsai35PromptState:
    ids = tuple(int(v) for v in input_ids)
    if not ids:
        raise ValueError("cannot cache an empty Bonsai-27B prompt")
    cache = _Qwen35Cache(len(model.artifact["layers"]))
    x = model._run_layers(ids, cache)
    return Bonsai35PromptState(
        artifact_digest=str(artifact_digest),
        input_ids=ids,
        last_x=np.ascontiguousarray(x[-1:], dtype=np.int64),
        cache=cache,
    )


def _flatten_state(state: Bonsai35PromptState, artifact: dict) -> dict[str, np.ndarray]:
    tensors = {"last_x": np.ascontiguousarray(state.last_x, dtype=np.int64)}
    for li, layer in enumerate(artifact["layers"]):
        kind = str(layer["kind"])
        if kind == "recurrent":
            recurrent = state.cache.state[li]
            conv = state.cache.conv[li]
            if recurrent is None or conv is None:
                raise ValueError(f"prompt state lacks recurrent cache for layer {li}")
            tensors[f"layers.{li}.state"] = np.ascontiguousarray(recurrent, dtype=np.int64)
            tensors[f"layers.{li}.conv"] = np.ascontiguousarray(conv, dtype=np.int64)
        elif kind == "attention":
            k, v = state.cache.k[li], state.cache.v[li]
            if k is None or v is None:
                raise ValueError(f"prompt state lacks attention cache for layer {li}")
            tensors[f"layers.{li}.k"] = np.ascontiguousarray(k, dtype=np.int64)
            tensors[f"layers.{li}.v"] = np.ascontiguousarray(v, dtype=np.int64)
        else:
            raise ValueError(f"unknown Qwen3.5 layer kind {kind!r}")
    return tensors


def _tensor_commitment(tensors: dict[str, np.ndarray]) -> str:
    rows = [(name, tensor_digest(tensors[name])) for name in sorted(tensors)]
    body = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(b"trinote-bonsai35-prompt-cache-tensors\0" + body).hexdigest()


def save_prompt_state(
    state: Bonsai35PromptState,
    artifact: dict,
    path: str | Path | None = None,
) -> Path:
    if str(artifact.get("config", {}).get("architecture")) != "qwen35":
        raise ValueError("Bonsai-27B prompt cache requires a qwen35 artifact")
    expected_key = prompt_cache_key(state.artifact_digest, state.input_ids)
    out = Path(path) if path is not None else default_prompt_cache_path(
        state.artifact_digest, state.input_ids
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tensors = _flatten_state(state, artifact)
    metadata = {
        "format": PROMPT_CACHE_FORMAT,
        "key": expected_key,
        "artifactDigest": state.artifact_digest,
        "inputIds": list(state.input_ids),
        "t": int(state.cache.t),
        "layerKinds": [str(layer["kind"]) for layer in artifact["layers"]],
        "tensorCommitment": _tensor_commitment(tensors),
    }
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    try:
        save_file(
            tensors,
            str(tmp),
            metadata={"trinote": json.dumps(metadata, sort_keys=True, separators=(",", ":"))},
        )
        os.replace(tmp, out)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return out


def _required_shapes(artifact: dict, prompt_len: int) -> dict[str, tuple[int, ...]]:
    cfg = artifact["config"]
    shapes: dict[str, tuple[int, ...]] = {"last_x": (1, int(cfg["dModel"]))}
    state_size = int(cfg["ssmStateSize"])
    value_heads = int(cfg["ssmTimeStepRank"])
    key_dim = int(cfg["ssmGroupCount"]) * state_size
    conv_dim = 2 * key_dim + int(cfg["ssmInnerSize"])
    for li, layer in enumerate(artifact["layers"]):
        if layer["kind"] == "recurrent":
            shapes[f"layers.{li}.state"] = (value_heads, state_size, state_size)
            shapes[f"layers.{li}.conv"] = (int(cfg["ssmConvKernel"]) - 1, conv_dim)
        else:
            kv = (int(cfg["n_heads_kv"]), prompt_len, int(cfg["head_dim"]))
            shapes[f"layers.{li}.k"] = kv
            shapes[f"layers.{li}.v"] = kv
    return shapes


def load_prompt_state(
    path: str | Path,
    artifact: dict,
    artifact_digest: str,
) -> Bonsai35PromptState:
    src = Path(path)
    tensors: dict[str, np.ndarray] = {}
    with safe_open(str(src), framework="numpy") as f:
        raw = f.metadata() or {}
        for name in f.keys():
            tensors[name] = f.get_tensor(name)
    try:
        metadata = json.loads(raw["trinote"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Bonsai-27B prompt cache metadata in {src}") from exc
    if metadata.get("format") != PROMPT_CACHE_FORMAT:
        raise ValueError(f"unsupported prompt cache format {metadata.get('format')!r}")
    if metadata.get("artifactDigest") != str(artifact_digest):
        raise ValueError("prompt cache artifact digest does not match the loaded model")
    ids = tuple(int(v) for v in metadata.get("inputIds", []))
    if not ids or int(metadata.get("t", -1)) != len(ids):
        raise ValueError("prompt cache input IDs/time position are malformed")
    if metadata.get("key") != prompt_cache_key(str(artifact_digest), ids):
        raise ValueError("prompt cache content-address key does not match its inputs")
    kinds = [str(layer["kind"]) for layer in artifact["layers"]]
    if metadata.get("layerKinds") != kinds:
        raise ValueError("prompt cache layer graph does not match the loaded artifact")
    required = _required_shapes(artifact, len(ids))
    if set(tensors) != set(required):
        raise ValueError("prompt cache tensor set does not match the Qwen3.5 graph")
    for name, shape in required.items():
        value = tensors[name]
        if value.dtype != np.int64 or value.shape != shape:
            raise ValueError(
                f"prompt cache tensor {name} has {value.dtype}{value.shape}, expected int64{shape}"
            )
    if metadata.get("tensorCommitment") != _tensor_commitment(tensors):
        raise ValueError("prompt cache tensor commitment mismatch")

    cache = _Qwen35Cache(len(artifact["layers"]))
    cache.t = len(ids)
    for li, layer in enumerate(artifact["layers"]):
        if layer["kind"] == "recurrent":
            cache.state[li] = np.ascontiguousarray(tensors[f"layers.{li}.state"])
            cache.conv[li] = np.ascontiguousarray(tensors[f"layers.{li}.conv"])
        else:
            k = np.ascontiguousarray(tensors[f"layers.{li}.k"])
            v = np.ascontiguousarray(tensors[f"layers.{li}.v"])
            cap = max(len(ids), 16)
            kb = np.empty((k.shape[0], cap, k.shape[2]), dtype=np.int64)
            vb = np.empty((v.shape[0], cap, v.shape[2]), dtype=np.int64)
            kb[:, :len(ids)] = k
            vb[:, :len(ids)] = v
            cache.k_buf[li], cache.v_buf[li] = kb, vb
            cache.lengths[li] = len(ids)
            cache.k[li], cache.v[li] = kb[:, :len(ids)], vb[:, :len(ids)]
    return Bonsai35PromptState(
        artifact_digest=str(artifact_digest),
        input_ids=ids,
        last_x=np.ascontiguousarray(tensors["last_x"]),
        cache=cache,
    )


def generate_from_prompt_state(
    model: BonsaiQwen35ReferenceModel,
    state: Bonsai35PromptState,
    n_new: int,
    pick: Callable[[np.ndarray, int, list[int]], int],
    *,
    eos: int | None = None,
    on_token: Callable[[int], None] | None = None,
    keep_reusable: bool = True,
) -> list[int]:
    if int(n_new) <= 0:
        return []
    x = state.last_x
    seq = list(state.input_ids)
    out: list[int] = []
    frac = int(model.cfg["frac"])
    rms_eps = int(model.cfg.get("rmsEpsilonFp2", 1))
    for step in range(int(n_new)):
        final = _rmsnorm(
            x[-1:], frac, model.artifact["final_norm_gain_fp"],
            native=model._native, eps=rms_eps,
        )
        row = model._output_linear(final)[0]
        token = int(pick(row, len(seq), seq))
        seq.append(token)
        out.append(token)
        if on_token is not None:
            on_token(token)
        # A state that will be persisted must consume the final sampled token
        # too, so cache.t, input_ids, and last_x describe the same prefix.  A
        # one-shot CLI generation can skip that otherwise-unused final pass.
        if keep_reusable or step + 1 < int(n_new):
            x = model._run_layers([token], state.cache)
        if eos is not None and token == int(eos):
            break
    state.last_x = np.ascontiguousarray(x[-1:])
    state.input_ids = tuple(seq if keep_reusable else seq[:state.cache.t])
    return out
