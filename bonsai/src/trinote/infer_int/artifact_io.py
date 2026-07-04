"""Persist the in-memory reference artifact to a single self-describing .safetensors file.

The reference artifact (`infer_int/reference.py`) is a nested dict of numpy arrays + python-int
per-layer gammas + a scalar config dict. There was no way to save/load it; this adds one, so a
trained (or demo) model becomes a single file whose sha256 is its stable identity.

safetensors stores only tensors, so we flatten the arrays to a flat name->ndarray map and stash
every non-array value (config, n_layers, untrained flag, per-layer gammas) as one JSON string in
the file's metadata under key 'trinote'. The round-trip is bit-identical: `ReferenceModel.forward` on a
loaded artifact equals forward on the original. Pure numpy + safetensors — no torch, no pickle.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save

from ..hashing.sha import sha256_file, sha256_hex

ARTIFACT_FORMAT = "trinote-artifact/1"

_TOP_ARRAYS = ("embed_fp", "final_norm_gain_fp", "cos_fp", "sin_fp")
_LAYER_ARRAYS = ("n1_gain_fp", "n2_gain_fp",
                 "wq_codes", "wk_codes", "wv_codes", "wo_codes", "w1_codes", "w2_codes")
_LAYER_SCALARS = ("wq_gamma_fp", "wk_gamma_fp", "wv_gamma_fp", "wo_gamma_fp",
                  "w1_gamma_fp", "w2_gamma_fp")


class ArtifactError(Exception):
    """Raised for a missing file, wrong/absent format tag, or malformed artifact metadata."""


def flatten_artifact(artifact: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Split the nested artifact into (flat tensors, json-able metadata). Pure / testable."""
    tensors: dict[str, np.ndarray] = {}
    for k in _TOP_ARRAYS:
        tensors[k] = np.ascontiguousarray(artifact[k])
    scalars: dict[str, int] = {}
    for i, layer in enumerate(artifact["layers"]):
        for k in _LAYER_ARRAYS:
            tensors[f"layers.{i}.{k}"] = np.ascontiguousarray(layer[k])
        for k in _LAYER_SCALARS:
            scalars[f"layers.{i}.{k}"] = int(layer[k])
    meta = {
        "format": ARTIFACT_FORMAT,
        "n_layers": len(artifact["layers"]),
        "config": artifact["config"],   # VERBATIM (mixed-style keys; ReferenceModel reads as-is)
        "scalars": scalars,
    }
    return tensors, meta


def unflatten_artifact(tensors: dict[str, np.ndarray], meta: dict) -> dict:
    """Inverse of flatten_artifact: rebuild the nested artifact ReferenceModel expects."""
    n_layers = int(meta["n_layers"])
    scalars = meta.get("scalars", {})
    layers = []
    for i in range(n_layers):
        layer: dict = {k: tensors[f"layers.{i}.{k}"] for k in _LAYER_ARRAYS}
        for k in _LAYER_SCALARS:
            layer[k] = int(scalars[f"layers.{i}.{k}"])
        layers.append(layer)
    return {
        "config": meta["config"],
        "embed_fp": tensors["embed_fp"],
        "final_norm_gain_fp": tensors["final_norm_gain_fp"],
        "cos_fp": tensors["cos_fp"],
        "sin_fp": tensors["sin_fp"],
        "layers": layers,
    }


def _serialize(artifact: dict, untrained: bool, provenance: dict | None = None) -> bytes:
    """The single canonical byte serialization — so an in-memory digest equals the on-disk sha256.

    `provenance` (training metadata: profile, tokens, losses, trainingConfigHash, datasetRoot…) is
    additive: when None the bytes are identical to before (a --demo artifact's digest is unchanged);
    when present it becomes part of the artifact's hashed identity.
    """
    tensors, meta = flatten_artifact(artifact)
    meta["untrained"] = bool(untrained)
    if provenance is not None:
        meta["provenance"] = provenance
    return save(tensors, metadata={"trinote": json.dumps(meta, sort_keys=True)})


def artifact_digest(artifact: dict, *, untrained: bool = False, provenance: dict | None = None) -> str:
    """sha256 of the exact bytes `save_artifact` would write — the model's stable identity, computed
    WITHOUT touching disk (so a live --demo model's receipt digest matches its persisted file)."""
    return sha256_hex(_serialize(artifact, untrained, provenance))


def save_artifact(artifact: dict, path: str | Path, *, untrained: bool = False,
                  provenance: dict | None = None) -> str:
    """Write the artifact as one .safetensors file; return its sha256 (the stable artifactDigest)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _serialize(artifact, untrained, provenance)
    path.write_bytes(data)
    return sha256_hex(data)              # == sha256_file(path) and == artifact_digest(artifact, ...)


def load_artifact(path: str | Path) -> tuple[dict, dict]:
    """Load a saved artifact -> (artifact, info{untrained,digest,format}). Raises ArtifactError."""
    path = Path(path)
    if not path.exists():
        raise ArtifactError(f"artifact not found: {path}")
    try:
        tensors: dict[str, np.ndarray] = {}
        with safe_open(str(path), framework="numpy") as f:
            raw = f.metadata()
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    except Exception as e:  # not a safetensors file, truncated, etc.
        raise ArtifactError(f"cannot read artifact {path}: {e}") from e
    if not raw or "trinote" not in raw:
        raise ArtifactError(f"{path} is not a trinote artifact (no 'trinote' metadata)")
    try:
        meta = json.loads(raw["trinote"])
    except json.JSONDecodeError as e:
        raise ArtifactError(f"corrupt 'trinote' metadata in {path}: {e}") from e
    if meta.get("format") != ARTIFACT_FORMAT:
        raise ArtifactError(f"unsupported artifact format {meta.get('format')!r} (need {ARTIFACT_FORMAT})")
    try:
        artifact = unflatten_artifact(tensors, meta)
    except KeyError as e:
        raise ArtifactError(f"artifact {path} missing tensor/scalar {e}") from e
    info = {"untrained": bool(meta.get("untrained", False)),
            "digest": sha256_file(path), "format": meta["format"],
            "provenance": meta.get("provenance")}
    return artifact, info
