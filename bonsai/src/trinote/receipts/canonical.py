"""Canonical serialization + commitment helpers for TEA receipts.

Every commitment in a receipt is sha256 over ONE canonical byte encoding, so any party recomputes
the identical digest (the trustless-third-entry premise, docs/receipts/RECEIPTS.md). Canonical JSON
here = sorted keys, no insignificant whitespace, UTF-8, `ensure_ascii` off (the bytes are the literal
text). This is the same "the bytes are the contract" discipline as the inference-to-chain JSON-artifact
interface (docs/receipts/RECEIPTS.md 'Scope'). Digests are bare lowercase hex (the repo-wide
convention: ricardianHash / datasetRoot / weightsRoot / artifactDigest are all bare hex — NOT the
`sha256:` display prefix used in display contexts).

FINITE VALUES ONLY: committed objects must contain only finite JSON values. `canonical_bytes` passes
`allow_nan=False`, so NaN / Infinity FAIL CLOSED (raise ValueError) rather than emitting the invalid
`NaN`/`Infinity` tokens that no other JSON parser would recompute the same way. Note also that any
float field's bytes depend on CPython's float `repr`, so a re-deriver must use the same repr to match;
prefer committing fixed-point INTEGERS (as the sampler's `repPenalty` already does) for cross-impl
reproducibility.

DOMAIN SEPARATION / ROLE BINDING (#9 — read before "fixing" the bare commit): `token_commit` hashes the
bare canonical id-list with NO role tag, so `inputCommit` and `outputCommit` over an IDENTICAL id-list are
the SAME 32-byte digest. This is INTENTIONALLY NOT changed here: these digests are anchored ON-CHAIN and
pinned by golden vectors + a cross-language receipt hash, so a wire-level domain tag inside the commit
would break byte-exact protocol compatibility. The role separation that makes an identical-id collision
NON-EXPLOITABLE lives one level up, in the SIGNED ENVELOPE: receipt.py signs/hashes objects whose FIELD
LABELS bind each digest to its role — the model entry signs
`{"modelHash","inputCommit","outputCommit","traceCommit"}` and the receipt body (→ receiptHash) carries
`inputCommit` and `outputCommit` as distinct keys. So even if the two raw digests collide, swapping the
input and output roles changes the signed/hashed bytes and is rejected; the bare commit never stands alone
as the authenticated object. A wire-level domain tag on the commit itself is DEFERRED on purpose to
preserve on-chain byte-exactness — do not add one without a coordinated protocol/vector migration.
"""
from __future__ import annotations

import json

from ..hashing.sha import sha256_hex


def canonical_bytes(obj) -> bytes:
    """The single canonical encoding of a JSON-able object: sorted keys, compact, UTF-8.

    `allow_nan=False` makes non-finite floats (NaN/Infinity) fail closed instead of producing invalid
    JSON; this is a no-op for every finite value, so committed bytes are unchanged for real inputs."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def commit(obj) -> str:
    """sha256 hex of an object's canonical bytes — a content commitment."""
    return sha256_hex(canonical_bytes(obj))


def token_commit(ids) -> str:
    """Commit to a token-id sequence (the 'canonical token ids of prompt/completion').

    Canonical form = the JSON array of ints, so the commitment is order-sensitive, inspectable, and
    language-neutral (it commits the ids, never the raw text — text stays off-chain; see
    docs/receipts/RECEIPTS.md).
    """
    return commit([int(i) for i in ids])
