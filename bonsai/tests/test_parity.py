"""Cross-language (Python <-> C <-> TS) differential parity goldens.

Coverage-gap closure: parity was previously reasoned about analytically but never validated by
feeding IDENTICAL inputs through the Python and C paths and comparing bytes. This pins the goldens
so any future divergence in an encoder (field order, int2ByteString width/endianness, pubkey/hash
lengths, or a protocol tag) breaks a test on at least one side.

The C side pins the SAME goldens in chain_c/tests/test_parity.c; the on-chain reference is
~/bonsai-notarized-bitnet/chain (agentTea.ts executeAction receipt + the OP_RETURN tag).
"""
from __future__ import annotations

from trinote.bundle.stateful import agent_action_receipt_hash


# AgentTea executeAction receiptHash (the stateful Third Entry). Identical 8-field inputs are fed
# through the C agent_tea_receipt_hash in test_parity.c, which asserts this same digest. The byte
# layout matches AgentTea.executeAction in agentTea.ts:
#   ricardianHash + agent + counterparty + int2ByteString(amount,8) + actionHash + provenanceHash
#   + int2ByteString(txCount,8) + int2ByteString(now,4)
_AGENT_ACTION_GOLDEN = "56e6f591625f7927cf3bb3ef1a97d5bf3ad84d94c4df843cf3136cd2fc2b7d33"


def test_agent_action_receipt_hash_golden():
    got = agent_action_receipt_hash(
        ricardian_hash="f6" * 32,
        agent_pubkey="02" + "11" * 32,
        counterparty_pubkey="03" + "22" * 32,
        amount=1000,
        action_hash="aa" * 32,
        provenance_hash="bb" * 32,
        tx_count=7,
        lock_time=1700000000,
    )
    assert got == _AGENT_ACTION_GOLDEN, (
        f"Python AgentTea action receiptHash {got} != pinned golden {_AGENT_ACTION_GOLDEN}; "
        "Python<->C<->TS stateful-receipt encoder parity is broken"
    )


def test_op_return_tag_constant_is_trinote_r1():
    """The on-chain OP_RETURN tag is fixed at 'trinote/r1' on BOTH sides (chain_c emits it; the
    Python verifier/broadcaster pin it). A drift here silently breaks anchor verification."""
    import trinote.bundle.verify as V
    import trinote.receipts.broadcast as B
    # The verifier pins the literal; assert the source agrees (cheap guard against an accidental edit).
    import inspect
    assert "trinote/r1" in inspect.getsource(V._verify_onchain)
    assert "trinote/r1" in inspect.getsource(B)


def test_agent_action_receipt_hash_fails_closed_on_bad_field():
    """A mis-sized field or out-of-range int must raise, never emit a hash that can't match chain."""
    import pytest
    with pytest.raises(ValueError):
        agent_action_receipt_hash(
            ricardian_hash="f6" * 31,  # 31 bytes, wrong
            agent_pubkey="02" + "11" * 32, counterparty_pubkey="03" + "22" * 32,
            amount=1000, action_hash="aa" * 32, provenance_hash="bb" * 32,
            tx_count=7, lock_time=1700000000,
        )
    with pytest.raises(ValueError):
        agent_action_receipt_hash(
            ricardian_hash="f6" * 32, agent_pubkey="02" + "11" * 32,
            counterparty_pubkey="03" + "22" * 32,
            amount=1 << 64,  # overflows int2ByteString(_, 8)
            action_hash="aa" * 32, provenance_hash="bb" * 32,
            tx_count=7, lock_time=1700000000,
        )


def test_le_reserves_sign_bit_like_scrypt():
    """_le matches scrypt int2ByteString / C int2bytestring_sized: in-range values encode, but a
    magnitude that sets the top byte's SIGN bit is rejected (review-2 #18)."""
    import pytest
    from trinote.bundle.stateful import _le
    assert _le(1000, 8) == (1000).to_bytes(8, "little")
    assert _le(1700000000, 4) == (1700000000).to_bytes(4, "little")  # pre-2038 now is fine
    with pytest.raises(ValueError):
        _le(1 << 63, 8)        # sets bit 63
    with pytest.raises(ValueError):
        _le(0x80000000, 4)     # 2^31 sets the sign bit of a 4-byte field
