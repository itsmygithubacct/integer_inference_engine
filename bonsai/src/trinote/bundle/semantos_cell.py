"""semantos_cell.py — the Trinote side of `onchain.kind = "semantos-cell"`.

## Why a fourth bundle kind

Both systems can write to chain. Left alone, an integrated request gets anchored twice:
semantos publishes a verified result cell, Trinote publishes a Third Entry, and the two
marks describe the same computation at two costs with two ways to disagree.

So semantos publishes, and a Trinote bundle records evidence pointing at what it
published. The division of labour is deliberate and is the reason this module is small:

  * **semantos** proves the transaction is in a block — BEEF/SPV inclusion.
  * **Trinote** proves the anchored cell describes *this receipt* — content binding.

Neither check substitutes for the other. An included transaction that commits somebody
else's computation is not evidence about this one, and a perfectly bound cell that was
never mined is not evidence at all. This module does the second and refuses to pretend
about the first.

## The state everyone conflates

A dry run is not an anchor. An accepted broadcast is not a confirmed inclusion. In this
record the difference is carried by `inclusionProofRef`: null means submitted, a
reference means inclusion was proven and can be rechecked by someone else. It is a
separate field from `txid` on purpose — holding a transaction id says a broadcast
happened, which is a different claim from the transaction being in a block.

## Cross-language agreement

`evidence_commit` must equal `evidenceCommit()` in the semantos cartridge's
`anchor-evidence.ts`, byte for byte, or the two sides commit to different things while
believing they agree. Every field is therefore restricted to the value domain where
Python's `json.dumps(sort_keys=True, separators=(",",":"))` and RFC 8785 JCS provably
coincide: 64-char lowercase hex, bounded printable-ASCII, safe integers, and null.
"""

from __future__ import annotations

import re

from ..receipts.canonical import canonical_bytes, commit

EVIDENCE_KIND = "semantos-cell"

FIELDS = ("kind", "txid", "vout", "cellHash", "typeHash", "receiptHash",
          "modelBindingHash", "inclusionProofRef")

HASH_FIELDS = ("txid", "cellHash", "typeHash", "receiptHash", "modelBindingHash")

#: `inclusionProofRef` is an opaque locator, not a digest — bounded so it stays inside
#: the domain both canonical encoders agree on
MAX_PROOF_REF_CHARS = 256
SAFE_INT = 2 ** 53 - 1

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_ASCII = re.compile(r"\A[\x20-\x7e]*\Z")


class EvidenceError(ValueError):
    """Malformed semantos-cell evidence. `code` is bounded and safe to record."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def validate_evidence(evidence) -> None:
    """Refuse anything that is not this evidence form, before it is hashed or trusted.

    Validation happens before hashing rather than only on construction: a value outside
    the shared domain would still hash, and the two languages' encoders disagree there.
    A digest that differs by encoder is worse than a refusal, because both sides believe
    they committed the same thing.
    """
    # The order and the codes mirror `evidenceCommit()` in the semantos cartridge. Both
    # sides must refuse the same values *for the same stated reason*, or a caller that
    # branches on the code behaves differently depending on which implementation it
    # happened to reach — and a reason code is part of the contract, not a log message.
    if not isinstance(evidence, dict):
        raise EvidenceError("bad-type", "evidence must be an object")

    for key in evidence:                                    # unknown before missing
        if key not in FIELDS:
            raise EvidenceError("unknown-field", key)
    for key in FIELDS:
        if key not in evidence:
            raise EvidenceError("missing-field", key)

    if evidence["kind"] != EVIDENCE_KIND:
        raise EvidenceError("bad-kind", str(evidence["kind"]))

    for name in HASH_FIELDS:
        value = evidence[name]
        if not isinstance(value, str):
            raise EvidenceError("bad-type", name)
        if len(value) != 64:
            raise EvidenceError("bad-length", f"{name} is {len(value)} chars, want 64")
        if not _HEX64.match(value):
            raise EvidenceError("bad-hex", name)

    vout = evidence["vout"]
    if isinstance(vout, bool) or not isinstance(vout, int):
        raise EvidenceError("bad-number", "vout")
    if abs(vout) > SAFE_INT:
        raise EvidenceError("bad-number", "vout outside safe-integer range")
    if vout < 0:
        raise EvidenceError("bad-number", "vout must be non-negative")

    ref = evidence["inclusionProofRef"]
    if ref is not None:
        if not isinstance(ref, str):
            raise EvidenceError("bad-type", "inclusionProofRef")
        if not _ASCII.match(ref):
            raise EvidenceError("bad-charset", "inclusionProofRef")
        if len(ref) == 0:
            # an empty reference is not "no proof" — that is what null says
            raise EvidenceError("bad-length", "inclusionProofRef must not be empty")
        if len(ref) > MAX_PROOF_REF_CHARS:
            raise EvidenceError("bad-length", f"inclusionProofRef is {len(ref)} chars")


def evidence_commit(evidence) -> str:
    """The digest a Trinote bundle records for this evidence. Bare lowercase hex.

    Must equal `evidenceCommit()` in the semantos cartridge for the same object.
    """
    validate_evidence(evidence)
    return commit(evidence)


def is_confirmed(evidence) -> bool:
    """Has inclusion actually been proven, as opposed to merely broadcast?

    Written down once here rather than re-derived at each call site, because "we have a
    txid" reads like success and is not.
    """
    validate_evidence(evidence)
    return evidence["inclusionProofRef"] is not None


def evidence_binds_receipt(evidence, expected_receipt_hash: str,
                           expected_model_binding_hash: str | None = None) -> bool:
    """Does this evidence describe the receipt that was replayed?

    The anchored cell must reference the same receipt — and, when the caller knows it,
    the same model binding. `expected_model_binding_hash` is optional because a Trinote
    verifier cannot derive it: it is a semantos record. Optional means "not checked
    here", never "checked and passed", so a caller that has the value must pass it.
    """
    validate_evidence(evidence)
    if not isinstance(expected_receipt_hash, str) or not _HEX64.match(expected_receipt_hash):
        raise EvidenceError("bad-hex", "expected_receipt_hash")
    if evidence["receiptHash"] != expected_receipt_hash:
        return False
    if expected_model_binding_hash is not None:
        if not _HEX64.match(expected_model_binding_hash or ""):
            raise EvidenceError("bad-hex", "expected_model_binding_hash")
        if evidence["modelBindingHash"] != expected_model_binding_hash:
            return False
    return True


def evidence_bytes(evidence) -> bytes:
    """Canonical bytes of the evidence — exposed for cross-language vector checks."""
    validate_evidence(evidence)
    return canonical_bytes(evidence)
