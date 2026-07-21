"""Asymmetric, third-party-verifiable receipt signatures — secp256k1 ECDSA (`secp256k1-ecdsa@v1`).

This is the PUBLIC-key counterpart to the symmetric `local-hmac@v1` vouch in `signing.py`. Where HMAC can
only be verified by a holder of the shared secret (so it proves wiring, not authenticity), an EC signature is
verifiable by ANYONE holding only the signer's PUBLIC key — which is exactly the third-party-verifiable
property the triple-entry/Ricardian theory requires of the entries (Grigg; Sgantzos et al.). secp256k1 is
deliberate: it is Bitcoin/BSV's curve, so the SAME key that signs a receipt can later control the on-chain
agent UTXO / OP_RETURN third entry — one identity across the off-chain receipt and the on-chain notarization.

Signatures are DETERMINISTIC (RFC 6979) and low-S canonical (Bitcoin policy): byte-identical in (key, message),
consistent with the project's determinism contract. The wire form embeds the compressed public key so a
verifier needs nothing but the receipt itself:

    secp256k1-ecdsa@v1:<33-byte-compressed-pubkey-hex>:<64-byte-r||s-hex>

The private key is off-chain material (like any signing key) — never put the secret in a receipt; receipts
carry only the public key (`sigModelPubKey`) and the signature.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ecdsa import BadSignatureError, SECP256k1, SigningKey, VerifyingKey
from ecdsa.util import sigdecode_string, sigencode_string_canonize

SCHEME_EC = "secp256k1-ecdsa@v1"
_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _verifying_key(pub_hex: str) -> VerifyingKey:
    """Decode a compressed (or uncompressed) secp256k1 public key from hex."""
    return VerifyingKey.from_string(bytes.fromhex(pub_hex), curve=SECP256k1, hashfunc=hashlib.sha256)


def _secret_from_wif(wif: str) -> str:
    """Decode a compressed mainnet/testnet WIF and return its 32-byte secret.

    This deliberately stays dependency-free: wallet keyfiles are an optional
    input format for receipt signing, not a reason to add a Bitcoin SDK to the
    deterministic receipt core.
    """
    if not isinstance(wif, str) or not wif:
        raise ValueError("WIF must be a non-empty string")
    value = 0
    for char in wif:
        try:
            digit = _B58_ALPHABET.index(char)
        except ValueError as exc:
            raise ValueError("WIF contains a non-base58 character") from exc
        value = value * 58 + digit
    body = value.to_bytes((value.bit_length() + 7) // 8, "big") if value else b""
    body = b"\x00" * (len(wif) - len(wif.lstrip("1"))) + body
    if len(body) < 5:
        raise ValueError("WIF is too short")
    payload, checksum = body[:-4], body[-4:]
    expected = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    if checksum != expected:
        raise ValueError("WIF checksum mismatch")
    if len(payload) != 34 or payload[0] not in (0x80, 0xEF) or payload[-1] != 0x01:
        raise ValueError("receipt signing requires a compressed mainnet/testnet WIF")
    return payload[1:33].hex()


@dataclass(frozen=True)
class ECKey:
    """A secp256k1 signing key. `sk` is the private `SigningKey` (off-chain material — never serialized
    into a receipt). The public key (compressed) is what receipts and verifiers carry."""
    sk: SigningKey
    label: str = ""

    @classmethod
    def from_secret_hex(cls, secret_hex: str, *, label: str = "") -> "ECKey":
        return cls(sk=SigningKey.from_string(bytes.fromhex(secret_hex), curve=SECP256k1, hashfunc=hashlib.sha256),
                   label=label)

    @classmethod
    def generate(cls, *, label: str = "", secret_hex: str | None = None) -> "ECKey":
        """Generate (or rebuild from `secret_hex`) a key. `secret_hex` makes it reproducible (tests/CI)."""
        if secret_hex is not None:
            return cls.from_secret_hex(secret_hex, label=label)
        return cls(sk=SigningKey.from_string(os.urandom(32), curve=SECP256k1, hashfunc=hashlib.sha256), label=label)

    @property
    def public_hex(self) -> str:
        """Compressed (33-byte) public key hex — the public identity committed in receipts."""
        return self.sk.get_verifying_key().to_string("compressed").hex()

    @property
    def secret_hex(self) -> str:
        return self.sk.to_string().hex()

    @property
    def key_id(self) -> str:
        """Public routing/display tag = sha256(compressed pubkey)[:16]. Reveals nothing about the secret."""
        return hashlib.sha256(bytes.fromhex(self.public_hex)).hexdigest()[:16]

    def sign(self, payload: bytes) -> str:
        """`secp256k1-ecdsa@v1:<pubkey>:<sig>` — deterministic (RFC 6979), low-S canonical, over sha256(payload).
        The embedded public key makes the signature third-party-verifiable with no shared secret."""
        sig = self.sk.sign_deterministic(payload, hashfunc=hashlib.sha256, sigencode=sigencode_string_canonize)
        return f"{SCHEME_EC}:{self.public_hex}:{sig.hex()}"

    def to_json(self) -> dict:
        """Serializable form — carries the SECRET; store local-only (like any private key)."""
        return {"scheme": SCHEME_EC, "label": self.label, "secret": self.secret_hex, "pubKey": self.public_hex,
                "keyId": self.key_id}

    @classmethod
    def from_json(cls, d: dict) -> "ECKey":
        if d.get("scheme") == SCHEME_EC:
            key = cls.from_secret_hex(d["secret"], label=d.get("label", ""))
        elif d.get("private_key_hex"):
            # chain_c's historical ``chain/test_bsv.json`` schema.
            key = cls.from_secret_hex(str(d["private_key_hex"]), label=d.get("label", ""))
        elif d.get("wif"):
            # bonsai-notary wallet keyfile schema ({wif,address,publicKeyHex,...}).
            key = cls.from_secret_hex(_secret_from_wif(str(d["wif"])), label=d.get("label", ""))
        else:
            raise ValueError(
                f"unsupported key scheme {d.get('scheme')!r}; need {SCHEME_EC}, "
                "a chain_c private_key_hex keyfile, or a compressed-WIF wallet keyfile"
            )
        claimed_public = (
            d.get("pubKey") or d.get("publicKeyHex") or d.get("public_key_hex")
        )
        if claimed_public and str(claimed_public).lower() != key.public_hex.lower():
            raise ValueError("private key does not match the keyfile's claimed public key")
        return key

    def save(self, path: str | Path) -> None:
        import json
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n"
        fd = -1
        tmp = ""
        try:
            # mkstemp creates the file as 0600 before a single secret byte is
            # written.  Writing the destination first and chmod'ing afterwards
            # leaves a local disclosure window under a permissive process umask.
            fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=p.parent)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fd = -1  # owned by fh from here on
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, p)
            tmp = ""
            # Persist the rename as well as the file contents. Directory fsync
            # is supported on the Linux hosts on which the notary runs.
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            dir_fd = os.open(p.parent, flags)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            if fd >= 0:
                os.close(fd)
            if tmp:
                try:
                    os.unlink(tmp)
                except FileNotFoundError:
                    pass

    @classmethod
    def load_or_generate(cls, path: str | Path, *, label: str = "") -> "ECKey":
        """Load an issuer/BSV-wallet key, generating the native schema if absent."""
        import json
        p = Path(path)
        if p.exists():
            return cls.from_json(json.loads(p.read_text()))
        key = cls.generate(label=label)
        key.save(p)
        return key


def ec_keygen(*, label: str = "", secret_hex: str | None = None) -> ECKey:
    return ECKey.generate(label=label, secret_hex=secret_hex)


def verify_ec(payload: bytes, signature: str, *, expected_pubkey_hex: str | None = None) -> bool:
    """Verify a `secp256k1-ecdsa@v1` signature using ONLY the public key (no secret). If
    `expected_pubkey_hex` is given, the signature's embedded key must equal it (identity binding).

    Enforces the CANONICAL wire form the scheme promises (33-byte compressed pubkey, 64-byte r||s, low-S):
    an uncompressed key or a high-S re-encoding of an otherwise-valid signature is rejected. This is
    spec-conformance/anti-malleability hardening — it does not affect committed receipts (the signature is
    bound inside receiptHash), but it stops a non-canonical re-encoding from verifying as valid."""
    parts = (signature or "").split(":")
    if len(parts) != 3:
        return False
    scheme, pub_hex, sig_hex = parts
    if scheme != SCHEME_EC:
        return False
    if expected_pubkey_hex is not None and pub_hex.lower() != expected_pubkey_hex.lower():
        return False
    try:
        raw_pub = bytes.fromhex(pub_hex)
        if len(raw_pub) != 33 or raw_pub[0] not in (0x02, 0x03):
            return False                                  # require COMPRESSED pubkey (the committed form)
        raw_sig = bytes.fromhex(sig_hex)
        if len(raw_sig) != 64:
            return False                                  # require fixed 32-byte r || 32-byte s
        s_int = int.from_bytes(raw_sig[32:], "big")
        if s_int == 0 or s_int > SECP256k1.order // 2:
            return False                                  # require low-S (BIP-0062 / Bitcoin policy)
        vk = _verifying_key(pub_hex)
        return bool(vk.verify(raw_sig, payload, hashfunc=hashlib.sha256, sigdecode=sigdecode_string))
    except (BadSignatureError, ValueError):
        return False
