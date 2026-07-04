"""Minimal GGUF reader for north-mini-code (Stage 6): parse the KV header + tensor index, and dequantize
individual tensors (F32 / F16 / Q4_K / Q6_K) via the Stage-2 codec — float reference or integer fixed-point.

Reads tensor data lazily by offset, so a single tensor can be pulled (and optionally only its first
`max_blocks` super-blocks sampled) without materializing the 18 GB file. Stdlib + numpy only.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from . import qk_codec as qk

# GGML tensor type ids we handle.
F32, F16, Q4_K, Q6_K = 0, 1, 12, 14
TYPE_NAME = {F32: "F32", F16: "F16", Q4_K: "Q4_K", Q6_K: "Q6_K"}
_BLK = {Q4_K: 144, Q6_K: 210}                  # bytes per 256-element super-block
_FIXED = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4),
          6: ("f", 4), 7: ("B", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}


class GGUF:
    """Parsed GGUF header. `kv` = metadata dict; `tensors[name]` = {type, shape, n, offset, nbytes}."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            assert f.read(4) == b"GGUF", "not a GGUF file"
            self.version = self._u32(f); ntensor = self._u64(f); nkv = self._u64(f)
            self.kv = {}
            for _ in range(nkv):
                k = self._str(f); self.kv[k] = self._val(f, self._u32(f))
            self.tensors: dict[str, dict] = {}
            for _ in range(ntensor):
                name = self._str(f); nd = self._u32(f)
                shape = [self._u64(f) for _ in range(nd)]
                tt = self._u32(f); off = self._u64(f)
                n = int(np.prod(shape)) if shape else 0
                self.tensors[name] = {"type": tt, "shape": shape, "n": n, "offset": off}
            data_pos = f.tell()
        align = int(self.kv.get("general.alignment", 32))
        self.data_start = (data_pos + align - 1) // align * align
        for name, t in self.tensors.items():
            t["nbytes"] = self._nbytes(t["type"], t["n"])

    # ---- header value readers --------------------------------------------------------------------------
    def _u32(self, f): return struct.unpack("<I", f.read(4))[0]
    def _u64(self, f): return struct.unpack("<Q", f.read(8))[0]
    def _str(self, f): return f.read(self._u64(f)).decode("utf-8", "replace")

    def _val(self, f, t):
        if t == 8:
            return self._str(f)
        if t == 9:                                   # array
            et = self._u32(f); n = self._u64(f)
            if et == 8:
                vals = [self._str(f) for _ in range(n)]
                return vals if n <= 64 else {"_arr_str_n": n}
            fmt, sz = _FIXED[et]
            if n <= 4096:
                return list(struct.unpack(f"<{n}{fmt}", f.read(n * sz)))
            f.seek(n * sz, 1); return {"_arr_n": n}
        fmt, sz = _FIXED[t]
        return struct.unpack(f"<{fmt}", f.read(sz))[0]

    @staticmethod
    def _nbytes(tt, n):
        if tt in (F32, F16):
            return n * (4 if tt == F32 else 2)
        if tt in _BLK:
            return (n // qk.QK_K) * _BLK[tt]
        raise ValueError(f"unsupported tensor type {tt}")

    # ---- tensor access ---------------------------------------------------------------------------------
    def read_raw(self, name: str, nbytes: int | None = None) -> bytes:
        t = self.tensors[name]
        want = t["nbytes"] if nbytes is None else min(nbytes, t["nbytes"])
        with open(self.path, "rb") as f:
            f.seek(self.data_start + t["offset"])
            return f.read(want)

    def dequant(self, name: str, frac: int | None = None, max_blocks: int | None = None) -> np.ndarray:
        """Dequantize a tensor. frac=None → float32 reference; frac=int → integer fixed-point at 2**frac.
        `max_blocks` samples only the first N super-blocks (Q4_K/Q6_K) — for validating large tensors cheaply."""
        t = self.tensors[name]; tt = t["type"]
        if tt in (F32, F16):
            nb = t["nbytes"] if max_blocks is None else min(t["nbytes"], max_blocks * 256 * (4 if tt == F32 else 2))
            arr = np.frombuffer(self.read_raw(name, nb), dtype=(np.float32 if tt == F32 else np.float16)).astype(np.float64)
            return arr.astype(np.float32) if frac is None else np.round(arr * (1 << frac)).astype(np.int64)
        nblocks = t["n"] // qk.QK_K
        if max_blocks is not None:
            nblocks = min(nblocks, max_blocks)
        raw = self.read_raw(name, nblocks * _BLK[tt])
        fn = qk.dequant_q4k_tensor if tt == Q4_K else qk.dequant_q6k_tensor
        return fn(raw, nblocks * qk.QK_K, frac)

    def _read_at(self, offset: int, nbytes: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(offset); return f.read(nbytes)

    def weight(self, name: str, frac: int | None = None) -> np.ndarray:
        """Dequantize a full tensor, reshaped to its LOGICAL shape [out, in] (or [n_experts, out, in]).
        GGUF stores ne0 fastest, so the logical (row-major) shape is `shape[::-1]`."""
        t = self.tensors[name]
        return self.dequant(name, frac).reshape(t["shape"][::-1])

    def raw_rows(self, name: str, row0: int, nrows: int) -> bytes:
        """Raw bytes for a contiguous range of logical rows [row0, row0+nrows) of a quantized tensor
        [ne0=in, ne1=out] — used for embedding lookup (rows = vocab entries)."""
        t = self.tensors[name]; ne0 = t["shape"][0]
        nb_row = ne0 // qk.QK_K * _BLK[t["type"]]
        return self._read_at(self.data_start + t["offset"] + row0 * nb_row, nrows * nb_row)

    def expert_raw(self, name: str, e: int):
        """Raw bytes of expert e of a 3-D expert tensor [ne0=in, ne1=out, n_experts], plus (in, out, type).
        The slice is row-major [out, in] in blocks — directly consumable by the qk_linear kernel."""
        t = self.tensors[name]; tt = t["type"]; ne0, ne1, _ = t["shape"]
        per = ne0 * ne1; nb = per // qk.QK_K; blk = _BLK[tt]
        raw = self._read_at(self.data_start + t["offset"] + e * nb * blk, nb * blk)
        return raw, ne0, ne1, tt

    def expert(self, name: str, e: int, frac: int | None = None) -> np.ndarray:
        """Dequantize ONE expert e of a 3-D expert tensor ([ne0=in, ne1=out, n_experts]) → [out, in].
        Reads only that expert's contiguous slice (avoids materializing all 128 experts)."""
        t = self.tensors[name]; tt = t["type"]; ne0, ne1, _ = t["shape"]
        per = ne0 * ne1                                            # elements per expert (contiguous)
        nb = per // qk.QK_K; blk = _BLK[tt]
        raw = self._read_at(self.data_start + t["offset"] + e * nb * blk, nb * blk)
        fn = qk.dequant_q4k_tensor if tt == Q4_K else qk.dequant_q6k_tensor
        return fn(raw, per, frac).reshape(ne1, ne0)               # [out, in]

    def summary(self):
        return {"version": self.version, "n_tensors": len(self.tensors), "n_kv": len(self.kv),
                "data_start": self.data_start, "arch": self.kv.get("general.architecture")}
