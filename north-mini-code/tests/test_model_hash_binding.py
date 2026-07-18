"""Regression (review-3 HIGH + MEDIUM): modelHash must bind the exact integer RoPE tables and the GGUF
header (tensor index), not just the tensor-data region and a RoPE label string.

model-free: a tiny fake engine drives the real receipts_runtime.model_hash / _rope_table_hash without the
18GB weights (full generate/model_hash parity is exercised on a GPU box separately)."""
import hashlib

import numpy as np
import pytest

from nmc import cohere2 as c2
from nmc.receipts_runtime import _MAX_ROPE_TABLE_LEN, _rope_table_hash, model_hash


class _FakeG:
    def __init__(self, path, data_start):
        self.path = str(path)
        self.data_start = int(data_start)


class _FakeEng:
    """Minimal stand-in exposing exactly what model_hash / _rope_table_hash read."""
    def __init__(self, path, data_start, cfg, *, context_length=32, drift_last_row=False):
        self.g = _FakeG(path, data_start)
        self.cfg = cfg
        self.NL = 4
        self.DENSE = 1
        self.context_length = int(context_length)
        self.drift_last_row = bool(drift_last_row)

    def _rope(self, n):
        cos, sin = c2.build_rope_tables(n, self.cfg.head_dim, base=int(self.cfg.rope_base),
                                        frac_bits=self.cfg.fa)
        if self.drift_last_row:
            cos[-1, 0] += 1
        return cos, sin


def _cfg(**over):
    base = dict(d_model=8, n_heads=2, n_kv=1, head_dim=4, ffn=16, vocab=32,
                rope_base=50000.0, fa=16, fw=24)
    base.update(over)
    return c2.Cfg(**base)


def _write_gguf(tmp_path, header: bytes, data: bytes, name: str = "fake.gguf"):
    p = tmp_path / name
    p.write_bytes(header + data)
    return p, len(header)


def test_rope_table_hash_deterministic_and_endianness_pinned(tmp_path):
    p, ds = _write_gguf(tmp_path, b"HDR-16-bytes!!!!", b"\x01\x02\x03\x04")
    eng = _FakeEng(p, ds, _cfg())
    a = _rope_table_hash(eng)
    b = _rope_table_hash(eng)
    assert a == b                                   # deterministic
    # equals a direct recomputation over the engine's own tables (the exact bytes hashed)
    cos, sin = eng._rope(eng.context_length)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(cos, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(sin, dtype="<i8").tobytes())
    assert a == h.hexdigest()


def test_modelhash_binds_rope_tables(tmp_path):
    """Two engines identical except rope_base -> different RoPE tables -> different modelHash. Before the
    fix modelHash only committed the string 'interleaved-norm', so this would have collided."""
    p, ds = _write_gguf(tmp_path, b"HDR-16-bytes!!!!", b"\x01\x02\x03\x04")
    mh1, art1 = model_hash(_FakeEng(p, ds, _cfg(rope_base=50000.0)))
    mh2, art2 = model_hash(_FakeEng(p, ds, _cfg(rope_base=10000.0)))
    assert art1 == art2                             # same tensor-data region -> same artifactDigest
    assert mh1 != mh2                               # but different tables -> different modelHash


def test_modelhash_binds_header(tmp_path):
    """Two files with identical tensor DATA but a different header (tensor index) must not share modelHash."""
    data = b"\xaa" * 64
    p1, ds1 = _write_gguf(tmp_path, b"HEADER-A-16bytes", data, name="a.gguf")
    p2, ds2 = _write_gguf(tmp_path, b"HEADER-B-16bytes", data, name="b.gguf")
    mh1, art1 = model_hash(_FakeEng(p1, ds1, _cfg()))
    mh2, art2 = model_hash(_FakeEng(p2, ds2, _cfg()))
    assert art1 == art2                             # identical tensor data
    assert mh1 != mh2                               # different header -> different modelHash


def test_modelhash_binds_every_supported_rope_row(tmp_path):
    """A rounding drift at the final supported position must change modelHash, not hide beyond a fixed prefix."""
    p, ds = _write_gguf(tmp_path, b"HDR-16-bytes!!!!", b"\x01\x02\x03\x04")
    clean = _FakeEng(p, ds, _cfg(), context_length=40)
    drift = _FakeEng(p, ds, _cfg(), context_length=40, drift_last_row=True)
    assert model_hash(clean)[0] != model_hash(drift)[0]


def test_modelhash_rejects_invalid_context_or_truncated_header(tmp_path):
    p, ds = _write_gguf(tmp_path, b"HEADER", b"DATA")
    with pytest.raises(ValueError, match="RoPE table length"):
        _rope_table_hash(_FakeEng(p, ds, _cfg(), context_length=_MAX_ROPE_TABLE_LEN + 1))
    with pytest.raises(ValueError, match="outside file size"):
        model_hash(_FakeEng(p, p.stat().st_size + 1, _cfg()))
