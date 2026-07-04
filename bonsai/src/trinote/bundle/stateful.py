"""Recompute the on-chain AgentTea `executeAction` receipt hash (the stateful Third Entry), in pure Python.

When a Bonsai inference is notarized *under a stateful identity* (chain/src/contracts-next/agentTea.ts), the
0-sat OP_RETURN it lands is NOT the standalone `tag | modelHash | receiptHash` mark. It is a single 32-byte
hash over the action's eight committed fields:

    receipt = ricardianHash(32) || agent(33) || counterparty(33)
            || int2ByteString(amount, 8) || actionHash(32) || provenanceHash(32)
            || int2ByteString(txCount, 8) || int2ByteString(now, 4)
    receiptHash = sha256(receipt)

This is the exact byte layout asserted in `AgentTea.executeAction` and rebuilt in `bindActionBuilder`
(chain/src/agentTeaTxBuilder.ts). The Bonsai integration binds **actionHash = the trinote receiptHash** and
**provenanceHash = the trinote modelHash** (docs/receipts/THIRD-ENTRY.md), so recomputing this hash and
matching it to the on-chain OP_RETURN proves the inference receipt is the one the identity committed.

`txCount` is the PRE-increment value (the contract reads `this.txCount` for the receipt, then increments).

sCrypt's `int2ByteString(n, size)` is fixed-width little-endian sign-magnitude. For the non-negative,
in-range values these fields carry (amount/txCount fit 8 bytes, a Unix `now` fits 4 bytes with the high bit
clear until 2038) that is identical to plain little-endian, which is what `_le()` emits; we fail closed on a
negative or oversized value rather than silently diverge from the contract's encoding.
"""
from __future__ import annotations

from ..hashing.sha import sha256_hex

_RICARDIAN_LEN = 32      # Sha256
_PUBKEY_LEN = 33         # compressed secp256k1 PubKey
_HASH_LEN = 32           # actionHash / provenanceHash (Sha256)


def _le(n: int, size: int) -> bytes:
    """Fixed-width little-endian bytes — matches scrypt int2ByteString for non-negative, in-range n.

    Reserves the SIGN bit of the top byte: scrypt int2ByteString / C int2bytestring_sized treat the
    top-byte high bit as the sign and refuse a magnitude that sets it (BNS_ERANGE). We match that so a
    top-bit-set value is rejected here rather than silently producing a digest the chain encoder would
    never emit (fidelity parity; review-2 #18). In-range fields are unaffected: amount < MAX_MONEY
    (2.1e15 < 2^51), txCount, and a pre-2038 `now` (< 2^31) never set the reserved bit."""
    if n < 0:
        raise ValueError(f"negative int2ByteString operand not supported: {n}")
    if n >= (1 << (8 * size - 1)):
        raise ValueError(f"value {n} sets the reserved sign bit of a {size}-byte int2ByteString field")
    return int(n).to_bytes(size, "little")


def _hex_field(name: str, value: str, n_bytes: int) -> bytes:
    raw = bytes.fromhex(value)
    if len(raw) != n_bytes:
        raise ValueError(f"{name} must be {n_bytes} bytes ({n_bytes * 2} hex chars), got {len(raw)}")
    return raw


def agent_action_receipt_hash(
    *,
    ricardian_hash: str,
    agent_pubkey: str,
    counterparty_pubkey: str,
    amount: int,
    action_hash: str,
    provenance_hash: str,
    tx_count: int,
    lock_time: int,
) -> str:
    """Recompute the 32-byte hex receiptHash an AgentTea.executeAction commits in its OP_RETURN.

    All hash/pubkey args are bare lowercase hex (no `0x`). `tx_count` is the pre-increment counter the
    receipt commits. `lock_time` is the tx nLockTime (Unix seconds) the action used. Raises ValueError on a
    mis-sized field or an out-of-range integer (fail closed — never emit a hash that can't match the chain).
    """
    preimage = (
        _hex_field("ricardianHash", ricardian_hash, _RICARDIAN_LEN)
        + _hex_field("agentPubKey", agent_pubkey, _PUBKEY_LEN)
        + _hex_field("counterpartyPubKey", counterparty_pubkey, _PUBKEY_LEN)
        + _le(amount, 8)
        + _hex_field("actionHash", action_hash, _HASH_LEN)
        + _hex_field("provenanceHash", provenance_hash, _HASH_LEN)
        + _le(tx_count, 8)
        + _le(lock_time, 4)
    )
    return sha256_hex(preimage)
