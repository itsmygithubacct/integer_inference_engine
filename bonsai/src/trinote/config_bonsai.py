"""Bonsai-8B Qwen3 configuration for a separate ATLAS notarized identity.

This is not a BitNet b1.58 variant. Bonsai-8B is Qwen3-8B dense with Q1_0 binary weights, so it gets its
own config and engine tag instead of reusing the generic 2B4T config.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class BonsaiQwen3Config:
    name: str = "ATLAS-Notarized-Bonsai-8B"
    source_repo: str = "prism-ml/Bonsai-8B-gguf"
    source_file: str = "Bonsai-8B-Q1_0.gguf"
    architecture: str = "qwen3"

    vocab_size: int = 151_669
    d_model: int = 4096
    n_layers: int = 36
    n_heads: int = 32
    n_heads_kv: int = 8
    head_dim: int = 128
    d_ffn: int = 12_288
    context_len: int = 65_536
    tie_embeddings: bool = False

    tokenizer: str = "qwen2-gpt2-bpe"
    pos_encoding: str = "rope-yarn"
    rope_base: int = 1_000_000
    rope_scaling_type: str = "yarn"
    rope_scaling_factor: float = 4.0
    rope_original_context_len: int = 16_384
    rope_convention: str = "neox"

    ffn_activation: str = "silu"
    ffn_gated: bool = True
    norm: str = "rmsnorm-qk"
    rms_eps: float = 1e-6

    quant: str = "q1_0-g128"
    quant_bits_effective: float = 1.125
    fp_frac_bits: int = 16
    inference_engine: str = "int-ref@bonsai-qwen3"

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_heads_kv == 0
        assert self.head_dim == self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_heads_kv * self.head_dim

    def param_count(self) -> int:
        embed = self.vocab_size * self.d_model
        output = 0 if self.tie_embeddings else self.vocab_size * self.d_model
        per_layer_attn = (
            self.d_model * self.d_model
            + 2 * self.d_model * self.kv_dim
            + self.d_model * self.d_model
        )
        per_layer_ffn = 3 * self.d_model * self.d_ffn
        per_layer_norm = 2 * self.d_model + 2 * self.head_dim
        return embed + output + self.n_layers * (per_layer_attn + per_layer_ffn + per_layer_norm) + self.d_model

    def as_params_block(self) -> dict:
        return {
            "name": self.name,
            "sourceRepo": self.source_repo,
            "sourceFile": self.source_file,
            "architecture": self.architecture,
            "vocab": self.vocab_size,
            "dModel": self.d_model,
            "nLayers": self.n_layers,
            "nHeads": self.n_heads,
            "nHeadsKv": self.n_heads_kv,
            "headDim": self.head_dim,
            "dFfn": self.d_ffn,
            "contextLen": self.context_len,
            "tieEmbeddings": self.tie_embeddings,
            "tokenizer": self.tokenizer,
            "posEncoding": self.pos_encoding,
            "ropeBase": self.rope_base,
            "ropeScalingType": self.rope_scaling_type,
            "ropeScalingFactor": self.rope_scaling_factor,
            "ropeOriginalContextLen": self.rope_original_context_len,
            "ropeConvention": self.rope_convention,
            "ffnActivation": self.ffn_activation,
            "ffnGated": self.ffn_gated,
            "norm": self.norm,
            "rmsEps": self.rms_eps,
            "quant": self.quant,
            "quantBitsEffective": self.quant_bits_effective,
            "fpFracBits": self.fp_frac_bits,
            "inferenceEngine": self.inference_engine,
        }

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


ATLAS_NOTARIZED_BONSAI_8B = BonsaiQwen3Config()
