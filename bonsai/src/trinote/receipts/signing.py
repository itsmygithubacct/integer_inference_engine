"""Local receipt signing — a stdlib HMAC "vouch", the LOCAL stand-in for the on-chain scheme.

The on-chain receipt entries are specified as **Rabin** signatures verified in-script on BSV (BSV has
no OP_CHECKDATASIG). This repo runs LOCALLY with on-chain emission **disabled**, so the
1st/2nd receipt entries are signed with a deterministic keyed **HMAC-SHA256**, labeled `local-hmac@v1`
so it can NEVER be mistaken for the on-chain Rabin scheme. It is a plumbing vouch (the v1 counterparty
may be a self-counterparty — proves the wiring, not yet adversarially meaningful; see
docs/receipts/RECEIPTS.md). Real asymmetric signing belongs to the chain integration and is intentionally
not wired here.

HONEST LIMITATION: HMAC is **symmetric** — verifying a signature needs the same secret that produced
it. That is fine for local self-counterparty plumbing; it is NOT a public, third-party-verifiable
signature. The *trustless* part of a receipt is the re-executable commitment chain (commits +
receiptHash + bit-exact re-run, receipts/verify.py), which needs no secret at all.

ASYMMETRIC OPTION (recommended for real deployments): `signing_ec.py` provides a third-party-verifiable
secp256k1 ECDSA scheme (`secp256k1-ecdsa@v1`) whose signature embeds the public key, so anyone can verify
with NO shared secret. `verify_signature()` below dispatches on the scheme prefix, so a receipt may carry
either an HMAC vouch (legacy/plumbing) or an EC signature. The ON-CHAIN third entry itself uses **Rabin**
signatures verified in-script by the chain service — that is the chain layer's concern, distinct
from this off-chain receipt vouch.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass

SCHEME = "local-hmac@v1"


@dataclass(frozen=True)
class LocalKey:
    """A local HMAC signing key. The `secret` is off-chain material — never put it in a receipt."""
    secret: bytes
    label: str = ""        # human tag, e.g. "model" or "counterparty"

    @property
    def key_id(self) -> str:
        """A 64-bit ROUTING/DISPLAY tag = sha256(secret)[:16]. Reveals nothing about the secret.

        This is a short hint for selecting/labeling a key in a signature string, NOT a security
        boundary: the authoritative check is the full-MAC constant-time compare in `verify`, which is
        always called with one specific key. Two distinct secrets could share a key_id (64-bit
        truncation), but that never lets a forged MAC pass — the compare uses the verifier's own secret
        over the full HMAC-SHA256 output."""
        return hashlib.sha256(self.secret).hexdigest()[:16]

    def to_json(self) -> dict:
        """Serializable form (carries the SECRET — store local-only, like any private key)."""
        return {"scheme": SCHEME, "label": self.label,
                "secret": self.secret.hex(), "keyId": self.key_id}

    @classmethod
    def from_json(cls, d: dict) -> "LocalKey":
        if d.get("scheme") != SCHEME:
            raise ValueError(f"unsupported key scheme {d.get('scheme')!r} (need {SCHEME})")
        return cls(secret=bytes.fromhex(d["secret"]), label=d.get("label", ""))

    def sign(self, payload: bytes) -> str:
        """Polymorphic with ECKey.sign so `build_receipt` can take either key type."""
        return sign(self, payload)


def keygen(*, label: str = "", secret_hex: str | None = None) -> LocalKey:
    """Generate (or rebuild) a local signing key. `secret_hex` makes it reproducible (tests/CI)."""
    secret = bytes.fromhex(secret_hex) if secret_hex is not None else os.urandom(32)
    return LocalKey(secret=secret, label=label)


def sign(key: LocalKey, payload: bytes) -> str:
    """`local-hmac@v1:<keyId>:<hexmac>` — deterministic in (secret, payload)."""
    mac = hmac.new(key.secret, payload, hashlib.sha256).hexdigest()
    return f"{SCHEME}:{key.key_id}:{mac}"


def verify(key: LocalKey, payload: bytes, signature: str) -> bool:
    """Constant-time check that `signature` is `key`'s local-hmac vouch over `payload`."""
    parts = signature.split(":")
    if len(parts) != 3:
        return False
    scheme, key_id, mac = parts
    if scheme != SCHEME or key_id != key.key_id:
        return False
    expected = hmac.new(key.secret, payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(mac, expected)


def verify_signature(payload: bytes, signature: str, *, key: "LocalKey | None" = None,
                     expected_pubkey: str | None = None) -> bool:
    """Scheme-dispatching verify over a self-describing signature string.

    * `secp256k1-ecdsa@v1:` → ASYMMETRIC: verified with the public key embedded in the signature (and, if
      `expected_pubkey` is given, that key must match). Needs NO secret — third-party-verifiable.
    * `local-hmac@v1:`      → SYMMETRIC: needs the shared `key` (the producing secret).
    """
    s = signature or ""
    from .signing_ec import SCHEME_EC          # lazy: keeps signing.py importable without ecdsa present
    if s.startswith(SCHEME_EC + ":"):
        from .signing_ec import verify_ec
        return verify_ec(payload, signature, expected_pubkey_hex=expected_pubkey)
    if s.startswith(SCHEME + ":"):
        return key is not None and verify(key, payload, signature)
    return False
