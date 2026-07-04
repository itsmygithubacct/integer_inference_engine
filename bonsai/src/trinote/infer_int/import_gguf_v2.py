"""Import a BitNet b1.58 GGUF (the real flagship weights) → an Atlas-v2 `int-ref@v2` reference artifact.

This is the keystone of the "Notarizing the flagship" path (docs/ATLAS-V2.md): the flagship's weights are
ALREADY ternary (Microsoft did the QAT), and — after the faithfulness work — the `int-ref@v2` graph is
architecturally identical to the flagship (GQA-5, gated ReLU², SubLN, tied, NeoX θ=500000, param==2.41B).
So the flagship's tensors map one-to-one onto a v2 artifact that `ReferenceModelV2` runs unchanged,
giving a 2B4T-quality model with the full ATLAS glass box (bit-exact int-ref + receipts) for ~$0.

Self-contained: a minimal GGUF container reader + the bitnet.cpp `i2_s` dequant (verified against
ggml.c / ggml-quants.c in the vendored llama.cpp), then fixed-point conversion into the v2 artifact
schema.

i2_s layout (verified): a weight tensor of logical shape (ne1=out, ne0=in), in divisible by 128, stores
`out*in/4` packed bytes followed by a 32-byte tail whose first 4 bytes are the per-tensor float32 scale.
Each byte packs 4 ternary codes at bit-pairs [6:7],[4:5],[2:3],[0:1] (codes 0/1/2/3 → −1/0/+1/0); within
each 128-element block, byte `gp` (0..31) fills row positions gp, gp+32, gp+64, gp+96.

HONEST SCOPE: the int-ref path is NOT bit-identical to bitnet.cpp (fixed-point integer vs float/TL
kernels) — it DEFINES a new canonical path. Validate quality with `quality_gate_v2` before minting.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..determinism.fixedpoint import to_fixed_point
from ..model.rope_v2 import build_rope_tables
from ..hashing.sha import sha256_file

# ggml type ids (from 3rdparty/llama.cpp/ggml/include/ggml.h)
_GGML_F32, _GGML_F16, _GGML_I2_S = 0, 1, 36

# GGUF metadata value-type ids (gguf spec v2/v3)
_GT_U8, _GT_I8, _GT_U16, _GT_I16, _GT_U32, _GT_I32, _GT_F32, _GT_BOOL, _GT_STR, _GT_ARR, \
    _GT_U64, _GT_I64, _GT_F64 = range(13)
_FIXED = {_GT_U8: ("<B", 1), _GT_I8: ("<b", 1), _GT_U16: ("<H", 2), _GT_I16: ("<h", 2),
          _GT_U32: ("<I", 4), _GT_I32: ("<i", 4), _GT_F32: ("<f", 4), _GT_BOOL: ("<B", 1),
          _GT_U64: ("<Q", 8), _GT_I64: ("<q", 8), _GT_F64: ("<d", 8)}


# ── minimal GGUF container reader ───────────────────────────────────────────────

@dataclass
class _Tensor:
    name: str
    shape: tuple          # ggml ne (ne0 fastest); for a linear weight = (in, out)
    ggml_type: int
    offset: int           # relative to data_start


class _GGUFReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.buf = np.memmap(self.path, dtype=np.uint8, mode="r")
        self._mv = memoryview(self.buf)
        self.kv: dict = {}
        self.tensors: dict[str, _Tensor] = {}
        self._parse()

    def _u(self, fmt, n):
        v = struct.unpack_from(fmt, self._mv, self.p)[0]
        self.p += n
        return v

    def _string(self) -> str:
        n = self._u("<Q", 8)
        # GGUF is untrusted: bound the attacker-controlled 64-bit length against the bytes that
        # actually remain before allocating/slicing, then advance by bytes ACTUALLY consumed.
        remaining = len(self._mv) - self.p
        if n > remaining:
            raise ValueError(f"GGUF string length {n} exceeds {remaining} remaining bytes")
        chunk = bytes(self._mv[self.p:self.p + n])
        self.p += len(chunk)
        try:
            return chunk.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"GGUF string is not valid UTF-8: {e}") from e

    def _read_value(self, vtype):
        if vtype in _FIXED:
            fmt, n = _FIXED[vtype]
            return self._u(fmt, n)
        if vtype == _GT_STR:
            return self._string()
        if vtype == _GT_ARR:
            elem_t = self._u("<I", 4)
            count = self._u("<Q", 8)
            # Bound the attacker-controlled element count: each element consumes at least 1 byte
            # (a 1-byte fixed type or a string's 8-byte length header), so it cannot exceed the
            # bytes that remain. min_elem is the smallest possible per-element footprint.
            min_elem = _FIXED[elem_t][1] if elem_t in _FIXED else (8 if elem_t == _GT_STR else None)
            if min_elem is None:
                raise ValueError(f"unsupported GGUF array element type {elem_t}")
            remaining = len(self._mv) - self.p
            if count > remaining // min_elem:
                raise ValueError(
                    f"GGUF array count {count} exceeds capacity for {remaining} remaining bytes "
                    f"(min {min_elem} bytes/element)")
            return [self._read_value(elem_t) for _ in range(count)]
        raise ValueError(f"unknown GGUF metadata value type {vtype}")

    def _parse(self):
        self.p = 0
        magic = bytes(self._mv[0:4])
        if magic != b"GGUF":
            raise ValueError(f"{self.path} is not a GGUF file (magic={magic!r})")
        self.p = 4
        self.version = self._u("<I", 4)
        n_tensors = self._u("<Q", 8)
        n_kv = self._u("<Q", 8)
        # GGUF is untrusted: clamp the attacker-controlled header counts to conservative caps
        # (far above any real model: the flagship has ~hundreds of tensors and dozens of KVs).
        # Each KV needs >= 8 (key-len) + 4 (vtype) + 1 (smallest value) bytes; each tensor entry
        # needs >= 8 (name-len) + 4 (ndim) + 4 (type) + 8 (offset) bytes. Cross-check the implied
        # minimum byte consumption against the remaining length before iterating.
        _MAX_KV, _MAX_TENSORS, _MAX_NDIM = 100_000, 1_000_000, 8
        if n_kv > _MAX_KV:
            raise ValueError(f"GGUF n_kv {n_kv} exceeds cap {_MAX_KV}")
        if n_tensors > _MAX_TENSORS:
            raise ValueError(f"GGUF n_tensors {n_tensors} exceeds cap {_MAX_TENSORS}")
        remaining = len(self._mv) - self.p
        min_bytes = n_kv * 13 + n_tensors * 24
        if min_bytes > remaining:
            raise ValueError(
                f"GGUF header implies >= {min_bytes} bytes but only {remaining} remain "
                f"(n_kv={n_kv}, n_tensors={n_tensors})")
        for _ in range(n_kv):
            key = self._string()
            vtype = self._u("<I", 4)
            self.kv[key] = self._read_value(vtype)
        for _ in range(n_tensors):
            name = self._string()
            ndim = self._u("<I", 4)
            if ndim > _MAX_NDIM:
                raise ValueError(f"GGUF tensor {name!r} ndim {ndim} exceeds cap {_MAX_NDIM}")
            shape = tuple(self._u("<Q", 8) for _ in range(ndim))
            gtype = self._u("<I", 4)
            off = self._u("<Q", 8)
            self.tensors[name] = _Tensor(name, shape, gtype, off)
        align = int(self.kv.get("general.alignment", 32))
        # general.alignment is attacker-controlled KV; the GGUF spec requires a positive power of two. Guard
        # it: 0 → ZeroDivisionError, negative → nonsensical data_start. Fail loud instead.
        if align <= 0 or (align & (align - 1)) != 0:
            raise ValueError(f"GGUF general.alignment must be a positive power of two, got {align}")
        self.data_start = (self.p + align - 1) // align * align

    def raw(self, t: _Tensor, nbytes: int) -> np.ndarray:
        # t.offset is attacker-controlled: validate the full span lies within the buffer BEFORE
        # slicing, else a NumPy slice would silently truncate (returning a short array).
        start = self.data_start + int(t.offset)
        end = start + int(nbytes)
        if start < 0 or end > len(self.buf):
            raise ValueError(
                f"tensor {t.name!r} span [{start}:{end}] out of bounds for buffer of "
                f"{len(self.buf)} bytes")
        return np.asarray(self.buf[start:end])


def _nbytes(t: _Tensor) -> int:
    # math.prod over exact Python ints (no float/int32 overflow that np.prod can hide).
    ne = math.prod(int(d) for d in t.shape)
    if t.ggml_type == _GGML_F32:
        return ne * 4
    if t.ggml_type == _GGML_F16:
        return ne * 2
    if t.ggml_type == _GGML_I2_S:
        return ne // 4 + 32          # packed 2-bit + 32-byte tail (scale in first 4 bytes)
    raise ValueError(f"unsupported ggml type {t.ggml_type} for tensor {t.name}")


# ── dequant ─────────────────────────────────────────────────────────────────────

_I2S_LUT = np.array([-1, 0, 1, 0], dtype=np.int8)   # 2-bit code -> ternary (3 is unused/0)


def _dequant_float(r: _GGUFReader, t: _Tensor) -> np.ndarray:
    """F32/F16 tensor → float64 array of shape (ne1, ne0) (C-order: each ne1 row is a contiguous ne0 vec)."""
    raw = r.raw(t, _nbytes(t))
    dt = np.float32 if t.ggml_type == _GGML_F32 else np.float16
    flat = raw.view(dt).astype(np.float64)
    rows = t.shape[1] if len(t.shape) > 1 else 1
    cols = t.shape[0]
    return flat.reshape(rows, cols)


def _dequant_i2s(r: _GGUFReader, t: _Tensor) -> tuple[np.ndarray, float]:
    """i2_s weight → (codes int8 (out,in) ∈ {-1,0,1}, scale float). out=ne1, in=ne0 (in % 128 == 0)."""
    if len(t.shape) != 2:
        raise ValueError(f"{t.name}: i2_s weight must be 2-D, got shape {t.shape}")
    in_f, out_f = int(t.shape[0]), int(t.shape[1])
    # Explicit raise (not assert): block-size validation must survive `python -O`.
    if in_f % 128 != 0:
        raise ValueError(f"{t.name}: in_features {in_f} not divisible by 128 (i2_s block size)")
    raw = r.raw(t, _nbytes(t))
    n_packed = out_f * in_f // 4
    scale = float(np.frombuffer(bytes(raw[n_packed:n_packed + 4]), dtype="<f4")[0])
    pk = np.asarray(raw[:n_packed]).reshape(out_f, in_f // 128, 32)        # (out, blocks, 32 bytes)
    g0 = _I2S_LUT[(pk >> 6) & 3]                                          # byte bits [6:7] -> group 0
    g1 = _I2S_LUT[(pk >> 4) & 3]
    g2 = _I2S_LUT[(pk >> 2) & 3]
    g3 = _I2S_LUT[pk & 3]
    codes = np.stack([g0, g1, g2, g3], axis=2).reshape(out_f, in_f)       # [grp0|grp1|grp2|grp3] per block
    return codes.astype(np.int8), scale


# ── name mapping: flagship GGUF → v2 artifact slots ─────────────────────────────

_ATTN = {"attn_q": "wq", "attn_k": "wk", "attn_v": "wv", "attn_output": "wo"}
_FFN = {"ffn_gate": "w1", "ffn_up": "wu", "ffn_down": "w2"}            # gate, up, down


def import_gguf_to_v2_artifact(gguf_path: str | Path, *, context_len: int = 4096,
                               frac: int = 16, progress=print) -> dict:
    """Build a v2 `int-ref@v2` reference artifact from a BitNet b1.58 GGUF (tied LM head)."""
    r = _GGUFReader(gguf_path)
    arch = r.kv.get("general.architecture", "?")
    if arch != "bitnet-b1.58":
        raise ValueError(f"expected general.architecture=bitnet-b1.58, got {arch!r}")
    d = int(r.kv["bitnet-b1.58.embedding_length"])
    n_layers = int(r.kv["bitnet-b1.58.block_count"])
    n_heads = int(r.kv["bitnet-b1.58.attention.head_count"])
    n_heads_kv = int(r.kv["bitnet-b1.58.attention.head_count_kv"])
    d_ffn = int(r.kv["bitnet-b1.58.feed_forward_length"])
    # rope.freq_base is a GGUF float32; int() truncates. It is EXACT for the integral flagship
    # value (θ=500000); validate integrality so a non-integral base can't silently truncate.
    _rope_base_f = float(r.kv["bitnet-b1.58.rope.freq_base"])
    if _rope_base_f != int(_rope_base_f):
        raise ValueError(f"non-integral rope.freq_base {_rope_base_f!r}")
    rope_base = int(_rope_base_f)
    head_dim = d // n_heads

    def tern(name):
        codes, scale = _dequant_i2s(r, r.tensors[name])
        return codes, int(round(scale * (1 << frac)))

    def gain(name):
        return to_fixed_point(_dequant_float(r, r.tensors[name]).reshape(-1), frac)

    progress(f"[import] arch={arch} d={d} layers={n_layers} heads={n_heads}/{n_heads_kv} "
             f"d_ffn={d_ffn} rope_base={rope_base}")
    layers = []
    for i in range(n_layers):
        layer = {"n1_gain_fp": gain(f"blk.{i}.attn_norm.weight"),
                 "n2_gain_fp": gain(f"blk.{i}.ffn_norm.weight"),
                 "sa_gain_fp": gain(f"blk.{i}.attn_sub_norm.weight"),
                 "sf_gain_fp": gain(f"blk.{i}.ffn_sub_norm.weight")}
        for gname, slot in {**_ATTN, **_FFN}.items():
            codes, gamma = tern(f"blk.{i}.{gname}.weight")
            layer[f"{slot}_codes"] = codes
            layer[f"{slot}_gamma_fp"] = gamma
        layers.append(layer)
        if (i + 1) % 5 == 0 or i == n_layers - 1:
            progress(f"[import] layer {i + 1}/{n_layers}")

    embed = to_fixed_point(_dequant_float(r, r.tensors["token_embd.weight"]), frac)   # (vocab, d), tied head
    vocab = embed.shape[0]
    cos, sin = build_rope_tables(context_len, head_dim, rope_base, frac)
    progress(f"[import] embed {embed.shape} vocab={vocab}; rope tables {cos.shape}")
    return {
        "config": {"dModel": d, "nLayers": n_layers, "n_heads": n_heads, "n_heads_kv": n_heads_kv,
                   "head_dim": head_dim, "dFfn": d_ffn, "vocab": vocab, "context_len": context_len,
                   "frac": frac, "ropeBase": rope_base},
        "embed_fp": embed,
        "final_norm_gain_fp": gain("output_norm.weight"),
        "cos_fp": cos, "sin_fp": sin, "layers": layers,
    }


def gguf_provenance(gguf_path: str | Path, *, source: str = "microsoft/bitnet-b1.58-2B-4T",
                    license: str = "MIT") -> dict:
    """The weight-provenance attestation that REPLACES `trainingConfigHash` for a notarized import:
    it binds *which* weights were imported (sha256 of the source GGUF) + their origin + licence."""
    p = Path(gguf_path)
    return {"kind": "imported-weights", "source": source, "license": license,
            "ggufFile": p.name, "ggufSha256": sha256_file(p),
            "importer": "trinote.infer_int.import_gguf_v2"}
