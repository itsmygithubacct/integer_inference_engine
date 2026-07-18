"""Persist/load Bonsai Q1_0 reference artifacts.

Both the original dense Qwen3 (Bonsai-8B) format and the hybrid Qwen3.5
(Bonsai-27B) format store Q1_0 weights in their natural packed form:
`*_bits` is uint8 with shape `(out_features, n_blocks, 16)` and `*_scale_fp` is fixed-point int64 with
shape `(out_features, n_blocks)`, one scale per 128-weight group. This keeps the artifact close to the
GGUF's 1-bit storage instead of expanding every weight to int8.

Qwen3.5 artifacts may commit the exactly equivalent scales as int32.  Q1_0
scales in the release are tiny fixed-point integers (the importer validates
the narrowing), and the native kernel has a byte-identical scale32 path.  The
narrow representation saves roughly 800 MiB for the 27B model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save, save_file

from ..hashing.sha import sha256_file, sha256_hex
from .artifact_io import ArtifactError

ARTIFACT_FORMAT_BONSAI = "trinote-artifact-bonsai-qwen3/1"
ARTIFACT_FORMAT_BONSAI_QWEN35 = "trinote-artifact-bonsai-qwen35/1"

_TOP_ARRAYS = (
    "embed_bits", "embed_scale_fp",
    "output_bits", "output_scale_fp",
    "final_norm_gain_fp", "cos_fp", "sin_fp",
)
_LAYER_ARRAYS = (
    "n1_gain_fp", "n2_gain_fp", "q_norm_gain_fp", "k_norm_gain_fp",
    "wq_bits", "wq_scale_fp", "wk_bits", "wk_scale_fp", "wv_bits", "wv_scale_fp",
    "wo_bits", "wo_scale_fp", "w1_bits", "w1_scale_fp", "wu_bits", "wu_scale_fp",
    "w2_bits", "w2_scale_fp",
)

_QWEN35_TOP_ARRAYS = (
    "embed_bits", "embed_scale_fp",
    "output_bits", "output_scale_fp",
    "final_norm_gain_fp", "cos_fp", "sin_fp",
    "softplus_lut_fp", "exp_neg_lut_fp",
)
_QWEN35_COMMON_LAYER_ARRAYS = (
    "n1_gain_fp", "n2_gain_fp",
    "w1_bits", "w1_scale_fp", "wu_bits", "wu_scale_fp",
    "w2_bits", "w2_scale_fp",
)
_QWEN35_ATTN_LAYER_ARRAYS = (
    "q_norm_gain_fp", "k_norm_gain_fp",
    "wqg_bits", "wqg_scale_fp", "wk_bits", "wk_scale_fp",
    "wv_bits", "wv_scale_fp", "wo_bits", "wo_scale_fp",
)
_QWEN35_RECURRENT_LAYER_ARRAYS = (
    "wqkv_bits", "wqkv_scale_fp", "wz_bits", "wz_scale_fp",
    "walpha_bits", "walpha_scale_fp", "wbeta_bits", "wbeta_scale_fp",
    "wout_bits", "wout_scale_fp",
    "conv_weight_fp", "dt_bias_fp", "ssm_a_fp", "ssm_norm_gain_fp",
)


def _is_qwen35(artifact: dict) -> bool:
    return str(artifact.get("config", {}).get("architecture", "")) == "qwen35"


def _qwen35_layer_keys(kind: str) -> tuple[str, ...]:
    if kind == "attention":
        return _QWEN35_COMMON_LAYER_ARRAYS + _QWEN35_ATTN_LAYER_ARRAYS
    if kind == "recurrent":
        return _QWEN35_COMMON_LAYER_ARRAYS + _QWEN35_RECURRENT_LAYER_ARRAYS
    raise ValueError(f"unknown Qwen3.5 layer kind {kind!r}")


def flatten_artifact_bonsai_qwen35(artifact: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Flatten a versioned Qwen3.5 hybrid artifact without changing Qwen3/8B."""
    tensors: dict[str, np.ndarray] = {}
    for k in _QWEN35_TOP_ARRAYS:
        tensors[k] = np.ascontiguousarray(artifact[k])
    layer_types: list[str] = []
    for i, layer in enumerate(artifact["layers"]):
        kind = str(layer.get("kind", ""))
        layer_types.append(kind)
        for k in _qwen35_layer_keys(kind):
            tensors[f"layers.{i}.{k}"] = np.ascontiguousarray(layer[k])
    return tensors, {
        "format": ARTIFACT_FORMAT_BONSAI_QWEN35,
        "n_layers": len(artifact["layers"]),
        "layer_types": layer_types,
        "config": artifact["config"],
    }


def unflatten_artifact_bonsai_qwen35(tensors: dict[str, np.ndarray], meta: dict) -> dict:
    n_layers = int(meta["n_layers"])
    layer_types = list(meta.get("layer_types", []))
    if len(layer_types) != n_layers:
        raise ValueError(
            f"Qwen3.5 artifact layer_types has {len(layer_types)} entries; expected {n_layers}"
        )
    layers = []
    for i, kind in enumerate(layer_types):
        layer = {k: tensors[f"layers.{i}.{k}"] for k in _qwen35_layer_keys(str(kind))}
        layer["kind"] = str(kind)
        layers.append(layer)
    artifact = {k: tensors[k] for k in _QWEN35_TOP_ARRAYS}
    artifact.update({"config": meta["config"], "layers": layers})
    return artifact


def flatten_artifact_bonsai(artifact: dict) -> tuple[dict[str, np.ndarray], dict]:
    if _is_qwen35(artifact):
        return flatten_artifact_bonsai_qwen35(artifact)
    tensors: dict[str, np.ndarray] = {}
    for k in _TOP_ARRAYS:
        tensors[k] = np.ascontiguousarray(artifact[k])
    for i, layer in enumerate(artifact["layers"]):
        for k in _LAYER_ARRAYS:
            tensors[f"layers.{i}.{k}"] = np.ascontiguousarray(layer[k])
    meta = {
        "format": ARTIFACT_FORMAT_BONSAI,
        "n_layers": len(artifact["layers"]),
        "config": artifact["config"],
    }
    return tensors, meta


def unflatten_artifact_bonsai(tensors: dict[str, np.ndarray], meta: dict) -> dict:
    if meta.get("format") == ARTIFACT_FORMAT_BONSAI_QWEN35:
        return unflatten_artifact_bonsai_qwen35(tensors, meta)
    layers = []
    for i in range(int(meta["n_layers"])):
        layers.append({k: tensors[f"layers.{i}.{k}"] for k in _LAYER_ARRAYS})
    return {
        "config": meta["config"],
        "embed_bits": tensors["embed_bits"],
        "embed_scale_fp": tensors["embed_scale_fp"],
        "output_bits": tensors["output_bits"],
        "output_scale_fp": tensors["output_scale_fp"],
        "final_norm_gain_fp": tensors["final_norm_gain_fp"],
        "cos_fp": tensors["cos_fp"],
        "sin_fp": tensors["sin_fp"],
        "layers": layers,
    }


def _serialize_bonsai(artifact: dict, provenance: dict | None = None) -> bytes:
    tensors, meta = flatten_artifact_bonsai(artifact)
    if provenance is not None:
        meta["provenance"] = provenance
    return save(tensors, metadata={"trinote": json.dumps(meta, sort_keys=True)})


def artifact_digest_bonsai(artifact: dict, *, provenance: dict | None = None) -> str:
    return sha256_hex(_serialize_bonsai(artifact, provenance))


def save_artifact_bonsai(artifact: dict, path: str | Path, *, provenance: dict | None = None) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_qwen35(artifact):
        # `save()` returns the entire serialized file as one bytes object.  That
        # is convenient for the 8B artifact, but needlessly duplicates a ~4 GiB
        # Qwen3.5 artifact in RAM.  `save_file()` writes the same safetensors
        # representation directly and we hash the completed file.
        tensors, meta = flatten_artifact_bonsai_qwen35(artifact)
        if provenance is not None:
            meta["provenance"] = provenance
        save_file(tensors, str(path), metadata={"trinote": json.dumps(meta, sort_keys=True)})
        return sha256_file(path)
    data = _serialize_bonsai(artifact, provenance)
    path.write_bytes(data)
    return sha256_hex(data)


_LOADED_ARTIFACT_SHA256 = "_trinote_loaded_artifact_sha256"
_LOADED_CONFIG_SHA256 = "_trinote_loaded_config_sha256"


def _config_sha256(config: object) -> str:
    """Digest the JSON model configuration exactly as loaded from metadata."""

    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_hex(encoded)


class _LazyInfo(dict):
    __slots__ = ("_path", "_artifact")

    def __init__(self, base: dict, path: Path, artifact: dict | None = None):
        super().__init__(base)
        self._path = path
        self._artifact = artifact

    def __missing__(self, key):
        if key == "digest":
            d = sha256_file(self._path)
            dict.__setitem__(self, key, d)
            # Preserve the digest of the bytes that actually produced this
            # in-memory artifact.  The 27B shared receipt API uses this loader
            # attestation to bind both its optimized producer and separately
            # loaded oracle to ``model_digest``.  Private keys are ignored by
            # the artifact flattener and therefore cannot affect inference or
            # a subsequent serialization.
            if self._artifact is not None:
                self._artifact[_LOADED_ARTIFACT_SHA256] = d
                self._artifact[_LOADED_CONFIG_SHA256] = _config_sha256(
                    self._artifact.get("config")
                )
            return d
        raise KeyError(key)


def load_artifact_bonsai(path: str | Path) -> tuple[dict, dict]:
    path = Path(path)
    if not path.exists():
        raise ArtifactError(f"artifact not found: {path}")
    try:
        tensors: dict[str, np.ndarray] = {}
        with safe_open(str(path), framework="numpy") as f:
            raw = f.metadata()
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    except Exception as e:
        raise ArtifactError(f"cannot read Bonsai artifact {path}: {e}") from e
    if not raw or "trinote" not in raw:
        raise ArtifactError(f"{path} is not a trinote artifact (no 'trinote' metadata)")
    try:
        meta = json.loads(raw["trinote"])
    except json.JSONDecodeError as e:
        raise ArtifactError(f"corrupt 'trinote' metadata in {path}: {e}") from e
    if meta.get("format") not in {ARTIFACT_FORMAT_BONSAI, ARTIFACT_FORMAT_BONSAI_QWEN35}:
        raise ArtifactError(f"not a Bonsai artifact: format {meta.get('format')!r} "
                            f"(need {ARTIFACT_FORMAT_BONSAI} or {ARTIFACT_FORMAT_BONSAI_QWEN35})")
    try:
        artifact = unflatten_artifact_bonsai(tensors, meta)
    except (KeyError, ValueError) as e:
        raise ArtifactError(f"Bonsai artifact {path} missing tensor {e}") from e
    info = _LazyInfo(
        {"format": meta["format"], "provenance": meta.get("provenance")},
        path,
        artifact,
    )
    return artifact, info


def read_artifact_info_bonsai(path: str | Path) -> dict:
    """Read Bonsai artifact metadata without materializing the weight tensors."""
    path = Path(path)
    if not path.exists():
        raise ArtifactError(f"artifact not found: {path}")
    try:
        with safe_open(str(path), framework="numpy") as f:
            raw = f.metadata()
    except Exception as e:
        raise ArtifactError(f"cannot read Bonsai artifact {path}: {e}") from e
    if not raw or "trinote" not in raw:
        raise ArtifactError(f"{path} is not a trinote artifact (no 'trinote' metadata)")
    try:
        meta = json.loads(raw["trinote"])
    except json.JSONDecodeError as e:
        raise ArtifactError(f"corrupt 'trinote' metadata in {path}: {e}") from e
    if meta.get("format") not in {ARTIFACT_FORMAT_BONSAI, ARTIFACT_FORMAT_BONSAI_QWEN35}:
        raise ArtifactError(f"not a Bonsai artifact: format {meta.get('format')!r} "
                            f"(need {ARTIFACT_FORMAT_BONSAI} or {ARTIFACT_FORMAT_BONSAI_QWEN35})")
    return _LazyInfo({"format": meta["format"], "provenance": meta.get("provenance"),
                      "config": meta.get("config")}, path)
