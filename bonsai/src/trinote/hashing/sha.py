"""SHA-256 helpers — canonical hashing used everywhere (matches the BSV/priscilla convention)."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_concat(*hex_parts: str) -> str:
    """Hash the concatenation of raw bytes of several hex digests (Merkle parent rule)."""
    h = hashlib.sha256()
    for p in hex_parts:
        h.update(bytes.fromhex(p))
    return h.hexdigest()


def double_sha256(data: bytes) -> bytes:
    """SHA-256(SHA-256(data)) — Bitcoin's hash256."""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def txid_of(raw_tx_hex: str) -> str:
    """The Bitcoin txid of a raw transaction = hash256(raw), displayed in reversed (big-endian) hex."""
    return double_sha256(bytes.fromhex(raw_tx_hex))[::-1].hex()
