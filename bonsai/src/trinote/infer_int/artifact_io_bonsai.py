"""Persist/load the Bonsai Qwen3 Q1_0 reference artifact.

The Bonsai artifact stores Q1_0 weights in their natural packed form:
`*_bits` is uint8 with shape `(out_features, n_blocks, 16)` and `*_scale_fp` is fixed-point int64 with
shape `(out_features, n_blocks)`, one scale per 128-weight group. This keeps the artifact close to the
GGUF's 1-bit storage instead of expanding every weight to int8.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save

from ..hashing.sha import sha256_file, sha256_hex
from .artifact_io import ArtifactError

ARTIFACT_FORMAT_BONSAI = "trinote-artifact-bonsai-qwen3/1"

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


def flatten_artifact_bonsai(artifact: dict) -> tuple[dict[str, np.ndarray], dict]:
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
    data = _serialize_bonsai(artifact, provenance)
    path.write_bytes(data)
    return sha256_hex(data)


class _LazyInfo(dict):
    __slots__ = ("_path",)

    def __init__(self, base: dict, path: Path):
        super().__init__(base)
        self._path = path

    def __missing__(self, key):
        if key == "digest":
            d = sha256_file(self._path)
            dict.__setitem__(self, key, d)
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
    if meta.get("format") != ARTIFACT_FORMAT_BONSAI:
        raise ArtifactError(f"not a Bonsai artifact: format {meta.get('format')!r} "
                            f"(need {ARTIFACT_FORMAT_BONSAI})")
    try:
        artifact = unflatten_artifact_bonsai(tensors, meta)
    except KeyError as e:
        raise ArtifactError(f"Bonsai artifact {path} missing tensor {e}") from e
    info = _LazyInfo({"format": meta["format"], "provenance": meta.get("provenance")}, path)
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
    if meta.get("format") != ARTIFACT_FORMAT_BONSAI:
        raise ArtifactError(f"not a Bonsai artifact: format {meta.get('format')!r} "
                            f"(need {ARTIFACT_FORMAT_BONSAI})")
    return _LazyInfo({"format": meta["format"], "provenance": meta.get("provenance"),
                      "config": meta.get("config")}, path)
