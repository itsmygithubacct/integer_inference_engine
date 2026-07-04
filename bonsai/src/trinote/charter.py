"""Parse the charter's machine params block and enforce parameters == prose (Ricardian Eq.1).

The delimited block in CHARTER.md is the single source of truth that must equal
`ModelConfig.as_params_block()`. This module reads the block and compares; together with
`trinote.hashing.sha` it backs the model-hash / charter-params gate (a mismatch is a different
identity, never a warning).
"""
from __future__ import annotations

import json
from pathlib import Path

from .hashing.sha import sha256_hex

_BEGIN = "<!-- ricardian:params:begin -->"
_END = "<!-- ricardian:params:end -->"
_SEP = "\x1f"


def _coerce(v: str):
    s = v.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if s.lstrip("-").isdigit():
        return int(s)
    try:
        return float(s)
    except ValueError:
        pass
    return s


def parse_params_block(charter_path: str | Path) -> dict:
    """Return the {key: typed_value} map from CHARTER.md's ricardian:params block."""
    text = Path(charter_path).read_text(encoding="utf-8")
    if _BEGIN not in text or _END not in text:
        raise ValueError("charter is missing the ricardian:params delimiters")
    block = text.split(_BEGIN, 1)[1].split(_END, 1)[0]
    params: dict = {}
    for raw in block.splitlines():
        line = raw.split("#", 1)[0].strip()        # drop inline comments
        if not line or line.startswith("```"):     # blank / code-fence lines are structural
            continue
        if "=" not in line:
            # A non-blank, non-fence line without '=' is decoy/contradictory prose sitting INSIDE the
            # authoritative machine block — it would be invisible to the gate. Fail closed: the
            # Ricardian premise is that the block a human reads IS the block that gets hashed.
            raise ValueError(f"charter params block has a non-assignment line: {line!r}")
        key, val = line.split("=", 1)
        key = key.strip()
        if key in params:
            # Last-wins would let the visibly-stated value diverge from the hashed/gated one.
            raise ValueError(f"charter params block has a duplicate key {key!r}")
        params[key] = _coerce(val)
    return params


def assert_matches(charter_path: str | Path, config_params: dict) -> None:
    """Hard gate: the charter params block must equal config.as_params_block(), exactly."""
    parsed = parse_params_block(charter_path)
    if parsed != config_params:
        diffs = []
        for k in sorted(set(parsed) | set(config_params)):
            if parsed.get(k) != config_params.get(k):
                diffs.append(f"  {k}: charter={parsed.get(k)!r} config={config_params.get(k)!r}")
        raise AssertionError("charter params != config (different identity):\n" + "\n".join(diffs))


def canonical_prose_bytes(charter_path: str | Path) -> bytes:
    """The charter prose with the §2 params block REMOVED — UTF-8 bytes of (everything before the
    begin-delimiter) ‖ (everything after the end-delimiter). The params are hashed separately, so each
    is counted ONCE (CHARTER.md §3: 'ricardianHash binds both this prose and §2, each counted once')."""
    text = Path(charter_path).read_text(encoding="utf-8")
    if _BEGIN not in text or _END not in text:
        raise ValueError("charter is missing the ricardian:params delimiters")
    before = text.split(_BEGIN, 1)[0]
    after = text.split(_END, 1)[1]
    return (before + after).encode("utf-8")


def canonical_params_bytes(config_params: dict) -> bytes:
    """Canonical serialization of the params block: JSON, sorted keys, no whitespace — the single,
    reproducible hashed form (drawn from config.as_params_block(), the source of truth)."""
    return json.dumps(config_params, sort_keys=True, separators=(",", ":")).encode("utf-8")


def ricardian_hash(charter_path: str | Path, config_params: dict) -> str:
    """ricardianHash = H(prose ‖ params)  (CHARTER.md §3).

    Gates params==charter FIRST (`assert_matches`) — a mismatch is a different identity, never a
    silent hash — so the value can only be computed for an in-sync charter. Reproducible: edit one byte
    of the prose OR the params and the hash changes (Grigg identity-binding)."""
    assert_matches(charter_path, config_params)
    return sha256_hex(canonical_prose_bytes(charter_path) + _SEP.encode()
                      + canonical_params_bytes(config_params))
