"""Import PrismML Bonsai-8B Q1_0 GGUF into the Bonsai Qwen3 reference artifact.

This importer is intentionally separate from `import_gguf_v2.py`: Bonsai is Qwen3 dense with GGUF
`Q1_0` binary weights, while the flagship BitNet path is `bitnet-b1.58` with `i2_s` ternary weights.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..hashing.sha import sha256_file
from ..model.rope_v2 import build_rope_tables, build_yarn_rope_tables
from .import_gguf_v2 import _GGML_F16, _GGML_F32, _GGUFReader, _Tensor, _dequant_float

_GGML_Q1_0 = 41
_QK1_0 = 128
_Q1_BLOCK_BYTES = 2 + (_QK1_0 // 8)       # fp16 scale + 128 sign bits
# Upper bound on context length: the RoPE table build allocates two (ctx, head_dim//2) int64 arrays in a
# Python loop, so an unbounded attacker-supplied qwen3.context_length is an OOM/hang vector. 1M rows is far
# above any real model and ~tens of MB.
_MAX_CTX = 1_048_576


def _req_int(kv: dict, key: str) -> int:
    """Read a REQUIRED integer GGUF KV with a clear error (vs a bare int() KeyError/ValueError/TypeError)."""
    if key not in kv:
        raise ValueError(f"GGUF missing required key {key!r}")
    try:
        return int(kv[key])
    except (TypeError, ValueError) as e:
        raise ValueError(f"GGUF key {key!r} must be an integer, got {kv[key]!r}") from e


def _req_int_from_float(kv: dict, key: str) -> int:
    """Read a REQUIRED numeric KV stored as a float (e.g. rope.freq_base=1000000.0), validating integrality
    (parity with the BitNet importer) rather than silently truncating a fractional value."""
    if key not in kv:
        raise ValueError(f"GGUF missing required key {key!r}")
    try:
        f = float(kv[key])
    except (TypeError, ValueError) as e:
        raise ValueError(f"GGUF key {key!r} must be numeric, got {kv[key]!r}") from e
    if f != int(f):
        raise ValueError(f"GGUF key {key!r} must be integral, got {f}")
    return int(f)


def _nbytes_bonsai(t: _Tensor) -> int:
    # math.prod over exact Python ints (no float/int32 overflow that np.prod can hide).
    ne = math.prod(int(d) for d in t.shape)
    if t.ggml_type == _GGML_F32:
        return ne * 4
    if t.ggml_type == _GGML_F16:
        return ne * 2
    if t.ggml_type == _GGML_Q1_0:
        if int(t.shape[0]) % _QK1_0 != 0:
            raise ValueError(f"{t.name}: Q1_0 ne0 {t.shape[0]} not divisible by {_QK1_0}")
        return (ne // _QK1_0) * _Q1_BLOCK_BYTES
    raise ValueError(f"unsupported ggml type {t.ggml_type} for Bonsai tensor {t.name}")


def _dequant_q1_packed(r: _GGUFReader, t: _Tensor, frac: int) -> tuple[np.ndarray, np.ndarray]:
    """Q1_0 tensor -> (`bits`, `scale_fp`) with rows `(out, in/128, 16)` and `(out, in/128)`."""
    if len(t.shape) != 2:
        raise ValueError(f"{t.name}: Q1_0 weight must be 2-D, got shape {t.shape}")
    in_f, out_f = int(t.shape[0]), int(t.shape[1])
    if t.ggml_type != _GGML_Q1_0:
        raise ValueError(f"{t.name}: expected GGML_TYPE_Q1_0 ({_GGML_Q1_0}), got {t.ggml_type}")
    raw = r.raw(t, _nbytes_bonsai(t)).reshape(out_f, in_f // _QK1_0, _Q1_BLOCK_BYTES)
    scale_f16 = raw[:, :, :2].copy().view("<f2").reshape(out_f, in_f // _QK1_0)
    scale_fp = np.rint(scale_f16.astype(np.float64) * (1 << frac)).astype(np.int64)
    bits = raw[:, :, 2:].copy()
    return bits, scale_fp


def _gain(r: _GGUFReader, name: str, frac: int) -> np.ndarray:
    return np.rint(_dequant_float(r, r.tensors[name]).reshape(-1) * (1 << frac)).astype(np.int64)


def _q1(r: _GGUFReader, name: str, frac: int) -> tuple[np.ndarray, np.ndarray]:
    return _dequant_q1_packed(r, r.tensors[name], frac)


def import_bonsai_gguf_to_artifact(gguf_path: str | Path, *, context_len: int | None = None,
                                   frac: int = 16, progress=print) -> dict:
    r = _GGUFReader(gguf_path)
    arch = r.kv.get("general.architecture", "?")
    if arch != "qwen3":
        raise ValueError(f"expected general.architecture='qwen3' for Bonsai, got {arch!r}")
    d = _req_int(r.kv, "qwen3.embedding_length")
    n_layers = _req_int(r.kv, "qwen3.block_count")
    n_heads = _req_int(r.kv, "qwen3.attention.head_count")
    n_heads_kv = _req_int(r.kv, "qwen3.attention.head_count_kv")
    head_dim = int(r.kv.get("qwen3.attention.key_length", d // n_heads))
    value_dim = int(r.kv.get("qwen3.attention.value_length", head_dim))
    if value_dim != head_dim:
        raise ValueError(f"unsupported Bonsai value_length {value_dim}; expected {head_dim}")
    d_ffn = _req_int(r.kv, "qwen3.feed_forward_length")
    rope_base = _req_int_from_float(r.kv, "qwen3.rope.freq_base")
    ctx = int(context_len or r.kv.get("qwen3.context_length", 65_536))
    if ctx <= 0 or ctx > _MAX_CTX:
        raise ValueError(f"qwen3.context_length {ctx} out of range (1..{_MAX_CTX}); refusing to build a "
                         f"{ctx}-row RoPE table")
    rope_scaling_type = str(r.kv.get("qwen3.rope.scaling.type", "none"))
    rope_scaling_factor = float(r.kv.get("qwen3.rope.scaling.factor", 1.0))
    rope_orig = int(r.kv.get("qwen3.rope.scaling.original_context_length", ctx))
    yarn_beta_fast = float(r.kv.get("qwen3.rope.scaling.yarn_beta_fast", 32.0))
    yarn_beta_slow = float(r.kv.get("qwen3.rope.scaling.yarn_beta_slow", 1.0))
    yarn_ext_factor = float(r.kv.get("qwen3.rope.scaling.yarn_ext_factor",
                                     1.0 if rope_scaling_type == "yarn" else 0.0))
    rope_attn_factor = float(r.kv.get("qwen3.rope.scaling.attn_factor", 1.0))

    progress(f"[bonsai-import] arch={arch} d={d} layers={n_layers} heads={n_heads}/{n_heads_kv} "
             f"head_dim={head_dim} d_ffn={d_ffn} ctx={ctx} q=Q1_0")

    embed_bits, embed_scale = _q1(r, "token_embd.weight", frac)
    output_bits, output_scale = _q1(r, "output.weight", frac)
    vocab = embed_bits.shape[0]
    if output_bits.shape[0] != vocab:
        raise ValueError(f"output vocab {output_bits.shape[0]} != embedding vocab {vocab}")

    layers = []
    for i in range(n_layers):
        wq_b, wq_s = _q1(r, f"blk.{i}.attn_q.weight", frac)
        wk_b, wk_s = _q1(r, f"blk.{i}.attn_k.weight", frac)
        wv_b, wv_s = _q1(r, f"blk.{i}.attn_v.weight", frac)
        wo_b, wo_s = _q1(r, f"blk.{i}.attn_output.weight", frac)
        w1_b, w1_s = _q1(r, f"blk.{i}.ffn_gate.weight", frac)
        wu_b, wu_s = _q1(r, f"blk.{i}.ffn_up.weight", frac)
        w2_b, w2_s = _q1(r, f"blk.{i}.ffn_down.weight", frac)
        layers.append({
            "n1_gain_fp": _gain(r, f"blk.{i}.attn_norm.weight", frac),
            "n2_gain_fp": _gain(r, f"blk.{i}.ffn_norm.weight", frac),
            "q_norm_gain_fp": _gain(r, f"blk.{i}.attn_q_norm.weight", frac),
            "k_norm_gain_fp": _gain(r, f"blk.{i}.attn_k_norm.weight", frac),
            "wq_bits": wq_b, "wq_scale_fp": wq_s,
            "wk_bits": wk_b, "wk_scale_fp": wk_s,
            "wv_bits": wv_b, "wv_scale_fp": wv_s,
            "wo_bits": wo_b, "wo_scale_fp": wo_s,
            "w1_bits": w1_b, "w1_scale_fp": w1_s,
            "wu_bits": wu_b, "wu_scale_fp": wu_s,
            "w2_bits": w2_b, "w2_scale_fp": w2_s,
        })
        if (i + 1) % 4 == 0 or i == n_layers - 1:
            progress(f"[bonsai-import] layer {i + 1}/{n_layers}")

    if rope_scaling_type == "yarn":
        # llama.cpp stores `rope.scaling.factor` and passes its inverse as `freq_scale`.
        cos, sin = build_yarn_rope_tables(
            ctx, head_dim,
            base=rope_base,
            freq_scale=1.0 / rope_scaling_factor,
            n_ctx_orig=rope_orig,
            ext_factor=yarn_ext_factor,
            attn_factor=rope_attn_factor,
            beta_fast=yarn_beta_fast,
            beta_slow=yarn_beta_slow,
            frac_bits=frac,
        )
    else:
        cos, sin = build_rope_tables(ctx, head_dim, rope_base, frac)
    return {
        "config": {
            "architecture": "qwen3",
            "dModel": d,
            "nLayers": n_layers,
            "n_heads": n_heads,
            "n_heads_kv": n_heads_kv,
            "head_dim": head_dim,
            "dFfn": d_ffn,
            "vocab": vocab,
            "context_len": ctx,
            "frac": frac,
            "ropeBase": rope_base,
            "ropeScalingType": rope_scaling_type,
            "ropeScalingFactor": rope_scaling_factor,
            "ropeOriginalContextLen": rope_orig,
            "ropeFreqScale": 1.0 / rope_scaling_factor if rope_scaling_factor else 1.0,
            "yarnExtFactor": yarn_ext_factor,
            "yarnAttnFactor": rope_attn_factor,
            "yarnBetaFast": yarn_beta_fast,
            "yarnBetaSlow": yarn_beta_slow,
        },
        "embed_bits": embed_bits,
        "embed_scale_fp": embed_scale,
        "output_bits": output_bits,
        "output_scale_fp": output_scale,
        "final_norm_gain_fp": _gain(r, "output_norm.weight", frac),
        "cos_fp": cos,
        "sin_fp": sin,
        "layers": layers,
    }


def bonsai_gguf_provenance(gguf_path: str | Path, *, source: str = "prism-ml/Bonsai-8B-gguf",
                           license: str = "Apache-2.0") -> dict:
    p = Path(gguf_path)
    return {
        "kind": "imported-weights",
        "source": source,
        "license": license,
        "ggufFile": p.name,
        "ggufSha256": sha256_file(p),
        "importer": "trinote.infer_int.import_bonsai_gguf",
        "quant": "GGUF Q1_0 g128",
    }
