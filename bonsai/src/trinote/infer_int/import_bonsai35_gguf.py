"""Import prism-ml/Bonsai-27B Q1_0 GGUF into the native Qwen3.5 artifact.

Qwen3.5 is a hybrid decoder: every fourth block is gated full attention and
the other blocks are causal-convolution + Gated DeltaNet recurrent blocks.
The large matrices are ordinary GGUF Q1_0 tensors, so they reuse Trinote's
packed integer kernel.  F32 norms, convolution kernels, and recurrent
parameters are committed as fixed-point integers.

The importer intentionally emits a different artifact format from Bonsai-8B.
That keeps the already-notarized Qwen3 graph immutable while making the new
architecture explicit in the model hash.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..hashing.sha import sha256_file
from ..model.rope_v2 import build_rope_tables
from .import_gguf_v2 import _GGML_F32, _GGUFReader, _dequant_float
from .import_bonsai_gguf import (
    _GGML_Q1_0,
    _dequant_q1_packed,
    _req_int,
    _req_int_from_float,
    _rint_to_fixed_i64,
)

_MAX_CTX = 262_144


def _q1_i32(r: _GGUFReader, name: str, frac: int, *, in_f: int, out_f: int) -> tuple[np.ndarray, np.ndarray]:
    """Read one Q1_0 matrix and losslessly narrow its fixed-point scales."""
    if name not in r.tensors:
        raise ValueError(f"GGUF missing required tensor {name!r}")
    t = r.tensors[name]
    if t.ggml_type != _GGML_Q1_0:
        raise ValueError(f"{name}: expected Q1_0 type {_GGML_Q1_0}, got {t.ggml_type}")
    expected = (int(in_f), int(out_f))
    if tuple(map(int, t.shape)) != expected:
        raise ValueError(f"{name}: shape {t.shape} does not match expected {expected}")
    bits, scale64 = _dequant_q1_packed(r, t, frac)
    info = np.iinfo(np.int32)
    if scale64.size and (int(scale64.min()) < info.min or int(scale64.max()) > info.max):
        raise ValueError(f"{name}: fixed-point Q1_0 scale cannot be represented exactly as int32")
    return bits, np.ascontiguousarray(scale64.astype(np.int32))


def _fixed_tensor(r: _GGUFReader, name: str, frac: int, *, shape: tuple[int, ...]) -> np.ndarray:
    if name not in r.tensors:
        raise ValueError(f"GGUF missing required tensor {name!r}")
    t = r.tensors[name]
    if t.ggml_type != _GGML_F32:
        raise ValueError(f"{name}: expected F32 tensor, got ggml type {t.ggml_type}")
    if tuple(map(int, t.shape)) != tuple(shape):
        raise ValueError(f"{name}: shape {t.shape} does not match expected {shape}")
    arr = _dequant_float(r, t)
    return _rint_to_fixed_i64(arr, frac, name)


def _fixed_vector(r: _GGUFReader, name: str, frac: int, size: int) -> np.ndarray:
    return _fixed_tensor(r, name, frac, shape=(int(size),)).reshape(-1)


def build_bonsai35_luts(frac: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build committed softplus and negative-exp lookup tables.

    Inference uses only integer indexing and interpolation.  The libm calls
    occur once here and their rounded outputs become part of the artifact
    hash, just like the committed RoPE table.  A 2^-10 grid is materially more
    accurate than the Q16 activation quantum after linear interpolation while
    remaining under 600 KiB for both tables.
    """
    frac = int(frac)
    grid_bits = min(frac, 10)
    step_fp = 1 << (frac - grid_bits)
    fp = 1 << frac
    soft_min_fp, soft_max_fp = -16 * fp, 16 * fp
    exp_min_fp, exp_max_fp = -32 * fp, 0

    soft_x = np.arange(soft_min_fp, soft_max_fp + step_fp, step_fp, dtype=np.int64)
    exp_x = np.arange(exp_min_fp, exp_max_fp + step_fp, step_fp, dtype=np.int64)
    soft = np.fromiter(
        (round(math.log1p(math.exp(int(x) / fp)) * fp) for x in soft_x),
        dtype=np.int64,
        count=soft_x.size,
    )
    exp_neg = np.fromiter(
        (round(math.exp(int(x) / fp) * fp) for x in exp_x),
        dtype=np.int64,
        count=exp_x.size,
    )
    return soft, exp_neg, {
        "lutStepFp": int(step_fp),
        "softplusLutMinFp": int(soft_min_fp),
        "softplusLutMaxFp": int(soft_max_fp),
        "expNegLutMinFp": int(exp_min_fp),
        "expNegLutMaxFp": int(exp_max_fp),
    }


def import_bonsai35_gguf_to_artifact(
    gguf_path: str | Path,
    *,
    context_len: int = 4096,
    frac: int = 16,
    progress=print,
) -> dict:
    """Convert the official Bonsai-27B Q1_0 GGUF to a native artifact."""
    if not (1 <= int(frac) <= 29):
        raise ValueError(f"frac must be in [1, 29], got {frac}")
    r = _GGUFReader(gguf_path)
    arch = str(r.kv.get("general.architecture", "?"))
    if arch != "qwen35":
        raise ValueError(f"expected general.architecture='qwen35', got {arch!r}")

    d = _req_int(r.kv, "qwen35.embedding_length")
    n_layers = _req_int(r.kv, "qwen35.block_count")
    n_heads = _req_int(r.kv, "qwen35.attention.head_count")
    n_heads_kv = _req_int(r.kv, "qwen35.attention.head_count_kv")
    head_dim = _req_int(r.kv, "qwen35.attention.key_length")
    value_dim = _req_int(r.kv, "qwen35.attention.value_length")
    d_ffn = _req_int(r.kv, "qwen35.feed_forward_length")
    interval = _req_int(r.kv, "qwen35.full_attention_interval")
    n_rot = _req_int(r.kv, "qwen35.rope.dimension_count")
    rope_base = _req_int_from_float(r.kv, "qwen35.rope.freq_base")
    sections = [int(x) for x in r.kv.get("qwen35.rope.dimension_sections", [])]
    conv_kernel = _req_int(r.kv, "qwen35.ssm.conv_kernel")
    group_count = _req_int(r.kv, "qwen35.ssm.group_count")
    inner_size = _req_int(r.kv, "qwen35.ssm.inner_size")
    state_size = _req_int(r.kv, "qwen35.ssm.state_size")
    dt_rank = _req_int(r.kv, "qwen35.ssm.time_step_rank")
    source_ctx = _req_int(r.kv, "qwen35.context_length")
    ctx = int(context_len)

    if value_dim != head_dim:
        raise ValueError(f"Qwen3.5 value_length {value_dim} != key_length {head_dim}")
    if d % 128 or d_ffn % 128 or inner_size % 128:
        raise ValueError("Qwen3.5 Q1_0 contraction widths must be divisible by 128")
    if n_heads % n_heads_kv:
        raise ValueError(f"attention heads {n_heads} not divisible by KV heads {n_heads_kv}")
    if inner_size % dt_rank:
        raise ValueError(f"ssm.inner_size {inner_size} not divisible by time_step_rank {dt_rank}")
    if inner_size // dt_rank != state_size:
        raise ValueError(
            f"unsupported Qwen3.5 value-head size {inner_size // dt_rank}; expected state_size {state_size}"
        )
    if len(sections) != 4 or sum(sections) * 2 != n_rot:
        raise ValueError(f"invalid Qwen3.5 MRoPE sections {sections} for n_rot={n_rot}")
    if not (0 < ctx <= min(source_ctx, _MAX_CTX)):
        raise ValueError(f"context_len {ctx} outside 1..{min(source_ctx, _MAX_CTX)}")
    if interval <= 0:
        raise ValueError(f"full_attention_interval must be positive, got {interval}")

    key_dim = state_size * group_count
    conv_dim = 2 * key_dim + inner_size
    progress(
        f"[bonsai35-import] d={d} layers={n_layers} full-every={interval} "
        f"heads={n_heads}/{n_heads_kv} hd={head_dim} recurrent={inner_size}/{state_size} "
        f"ctx={ctx} q=Q1_0"
    )

    embed_bits, embed_scale = _q1_i32(r, "token_embd.weight", frac, in_f=d, out_f=_req_vocab(r))
    vocab = int(embed_scale.shape[0])
    output_bits, output_scale = _q1_i32(r, "output.weight", frac, in_f=d, out_f=vocab)

    layers: list[dict] = []
    for i in range(n_layers):
        recurrent = (i + 1) % interval != 0
        prefix = f"blk.{i}."
        layer = {
            "kind": "recurrent" if recurrent else "attention",
            "n1_gain_fp": _fixed_vector(r, prefix + "attn_norm.weight", frac, d),
            "n2_gain_fp": _fixed_vector(r, prefix + "post_attention_norm.weight", frac, d),
        }
        for slot, tensor, inf, outf in (
            ("w1", "ffn_gate.weight", d, d_ffn),
            ("wu", "ffn_up.weight", d, d_ffn),
            ("w2", "ffn_down.weight", d_ffn, d),
        ):
            layer[f"{slot}_bits"], layer[f"{slot}_scale_fp"] = _q1_i32(
                r, prefix + tensor, frac, in_f=inf, out_f=outf
            )

        if recurrent:
            for slot, tensor, inf, outf in (
                ("wqkv", "attn_qkv.weight", d, conv_dim),
                ("wz", "attn_gate.weight", d, inner_size),
                ("walpha", "ssm_alpha.weight", d, dt_rank),
                ("wbeta", "ssm_beta.weight", d, dt_rank),
                ("wout", "ssm_out.weight", inner_size, d),
            ):
                layer[f"{slot}_bits"], layer[f"{slot}_scale_fp"] = _q1_i32(
                    r, prefix + tensor, frac, in_f=inf, out_f=outf
                )
            layer.update({
                "conv_weight_fp": _fixed_tensor(
                    r, prefix + "ssm_conv1d.weight", frac, shape=(conv_kernel, conv_dim)
                ),
                "dt_bias_fp": _fixed_vector(r, prefix + "ssm_dt.bias", frac, dt_rank),
                "ssm_a_fp": _fixed_vector(r, prefix + "ssm_a", frac, dt_rank),
                "ssm_norm_gain_fp": _fixed_vector(r, prefix + "ssm_norm.weight", frac, state_size),
            })
            if np.any(layer["ssm_a_fp"] > 0):
                raise ValueError(f"{prefix}ssm_a contains a positive decay coefficient")
        else:
            for slot, tensor, inf, outf in (
                ("wqg", "attn_q.weight", d, 2 * n_heads * head_dim),
                ("wk", "attn_k.weight", d, n_heads_kv * head_dim),
                ("wv", "attn_v.weight", d, n_heads_kv * head_dim),
                ("wo", "attn_output.weight", n_heads * head_dim, d),
            ):
                layer[f"{slot}_bits"], layer[f"{slot}_scale_fp"] = _q1_i32(
                    r, prefix + tensor, frac, in_f=inf, out_f=outf
                )
            layer.update({
                "q_norm_gain_fp": _fixed_vector(r, prefix + "attn_q_norm.weight", frac, head_dim),
                "k_norm_gain_fp": _fixed_vector(r, prefix + "attn_k_norm.weight", frac, head_dim),
            })
        layers.append(layer)
        if (i + 1) % 4 == 0 or i + 1 == n_layers:
            progress(f"[bonsai35-import] layer {i + 1}/{n_layers}")

    cos, sin = build_rope_tables(ctx, n_rot, rope_base, frac)
    softplus_lut, exp_neg_lut, lut_cfg = build_bonsai35_luts(frac)
    state_frac = min(30, int(frac) + 14)
    rms_epsilon = float(r.kv.get("qwen35.attention.layer_norm_rms_epsilon", 1e-6))
    config = {
        "architecture": "qwen35",
        "modelName": str(r.kv.get("general.name", "Bonsai-27B")),
        "dModel": d,
        "nLayers": n_layers,
        "n_heads": n_heads,
        "n_heads_kv": n_heads_kv,
        "head_dim": head_dim,
        "dFfn": d_ffn,
        "vocab": vocab,
        "context_len": ctx,
        "sourceContextLen": source_ctx,
        "frac": int(frac),
        "rmsEpsilon": rms_epsilon,
        "rmsEpsilonFp2": int(round(rms_epsilon * (1 << (2 * int(frac))))),
        "ropeBase": rope_base,
        "ropeRotDim": n_rot,
        "ropeSections": sections,
        "ropeType": "imrope-text",
        "fullAttentionInterval": interval,
        "ssmConvKernel": conv_kernel,
        "ssmGroupCount": group_count,
        "ssmInnerSize": inner_size,
        "ssmStateSize": state_size,
        "ssmTimeStepRank": dt_rank,
        # DeltaNet emits ~1e-4 values immediately before an RMSNorm.  Q16
        # would quantize a large fraction of them to zero and RMSNorm would
        # amplify the damage.  Keep the private recurrent state/score domain
        # at Q30; public residual activations and every Q1 contraction remain
        # at the engine-wide `frac` scale.
        "ssmStateFrac": state_frac,
        "ssmRmsEpsilonFp2": int(round(rms_epsilon * (1 << (2 * state_frac)))),
        "attentionScaleFp": int(round((1.0 / math.sqrt(head_dim)) * (1 << frac))),
        "gdnScaleFp": int(round((1.0 / math.sqrt(state_size)) * (1 << frac))),
        **lut_cfg,
    }
    return {
        "config": config,
        "embed_bits": embed_bits,
        "embed_scale_fp": embed_scale,
        "output_bits": output_bits,
        "output_scale_fp": output_scale,
        "final_norm_gain_fp": _fixed_vector(r, "output_norm.weight", frac, d),
        "cos_fp": cos,
        "sin_fp": sin,
        "softplus_lut_fp": softplus_lut,
        "exp_neg_lut_fp": exp_neg_lut,
        "layers": layers,
    }


def _req_vocab(r: _GGUFReader) -> int:
    t = r.tensors.get("token_embd.weight")
    if t is None or len(t.shape) != 2:
        raise ValueError("GGUF missing 2-D token_embd.weight")
    return int(t.shape[1])


def bonsai35_gguf_provenance(
    gguf_path: str | Path,
    *,
    source: str = "prism-ml/Bonsai-27B-gguf",
    license: str = "Apache-2.0",
) -> dict:
    p = Path(gguf_path)
    return {
        "kind": "imported-weights",
        "source": source,
        "license": license,
        "ggufFile": p.name,
        "ggufSha256": sha256_file(p),
        "importer": "trinote.infer_int.import_bonsai35_gguf",
        "quant": "GGUF Q1_0 g128; fixed scales committed as lossless int32",
    }
