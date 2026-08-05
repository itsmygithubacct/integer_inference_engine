"""remote_counterparty.py — a counterparty signature from a key this host does not hold.

## What the second signature is for

A receipt carries two vouches. The model signature says "this engine produced this
output"; the counterparty signature says "someone other than the engine agrees". The
second one is the whole reason a receipt is worth more than a log line — and it is
worth exactly nothing if both keys live on the producer, because then one party
signed twice and the receipt merely looks like it has two.

Nothing in a receipt can express that. Two keys on one disk have two distinct key ids
and two valid signatures; the bytes are indistinguishable from genuine two-party
attestation. The independence has to be arranged at production time or not at all.

## How this arranges it

The producer holds no counterparty secret. It sends the *named fields* it wants
vouched to a service that holds the key, and the service rebuilds the canonical
message itself and signs that. So the producer cannot use the counterparty as a
signing oracle for arbitrary bytes: anything that is not a counterparty message comes
back as a refusal.

The transport defaults to running a command (an SSH invocation, typically), so the
connection is outbound from the producer and the counterparty needs no listener.

## The pin

`public_hex` comes from policy, never from the service. A signature carries the key
that made it, which proves only that *someone* signed — the same reasoning the
semantos side applies in `verifier-signature.ts`. A response signed by any other key
is a refusal here, not a receipt.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from typing import Callable

from .signing import verify_signature

REQUEST_SCHEMA = "trinote.counterparty-signing-request/v1"

CANONICAL_SEPARATORS = (",", ":")

#: the exact shapes a counterparty entry may take — v2 unbound, v3 request-bound
V2_FIELDS = frozenset({"modelHash", "inputCommit", "outputCommit"})
V3_FIELDS = V2_FIELDS | {"contextCommit"}

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_HEX66 = re.compile(r"\A[0-9a-f]{66}\Z")


class SigningRefused(ValueError):
    """A counterparty signature was not obtained. `code` is bounded and safe to log."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class CounterpartyNotConfigured(SigningRefused):
    """No independent counterparty, and none was explicitly waived."""


def canonical_bytes(obj) -> bytes:
    """The engine's canonical JSON. Must match `receipts.canonical`, and the service's."""
    return json.dumps(obj, sort_keys=True, separators=CANONICAL_SEPARATORS,
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def check_message_shape(msg) -> None:
    """Refuse anything that is not a counterparty entry, before it leaves this host.

    The service checks this too — that is the check that matters, since it is the one
    holding the key. This one exists so an obvious mistake is caught without a round
    trip, and so a producer bug cannot quietly become a network request.
    """
    if not isinstance(msg, dict):
        raise SigningRefused("bad-payload", "not an object")
    keys = frozenset(msg)
    if keys not in (V2_FIELDS, V3_FIELDS):
        raise SigningRefused("not-a-counterparty-message", ", ".join(sorted(keys)))
    for name, value in sorted(msg.items()):
        if not isinstance(value, str) or not _HEX64.match(value):
            raise SigningRefused("bad-hex", name)


def key_id_for(public_hex: str) -> str:
    """`ECKey.key_id` derived from the public key alone — sha256(pubkey)[:16].

    Deriving it locally is what lets the producer state the counterparty's identity in
    the receipt without asking the counterparty who it is. Learning an identity from
    the party being identified is trust-on-first-use wearing a protocol.
    """
    if not isinstance(public_hex, str) or not _HEX66.match(public_hex):
        raise SigningRefused("bad-pinned-key",
                             "counterparty public key must be 33 compressed bytes as lowercase hex")
    return hashlib.sha256(bytes.fromhex(public_hex)).hexdigest()[:16]


@dataclass
class RemoteCounterpartySigner:
    """Satisfies the shape `build_receipt` expects while holding no secret.

    `public_hex` is pinned by policy. `command` is run with the request on stdin and
    the response expected on stdout; `transport` replaces it in tests.
    """

    public_hex: str
    command: list[str] | None = None
    transport: Callable[[bytes], bytes] | None = None
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self.public_hex = self.public_hex.lower() if isinstance(self.public_hex, str) else self.public_hex
        key_id_for(self.public_hex)                  # validate the pin at construction

    @property
    def key_id(self) -> str:
        return key_id_for(self.public_hex)

    def sign(self, payload: bytes) -> str:
        """Called by `build_receipt` with the canonical payload it assembled.

        The bytes are parsed, not forwarded. What travels is the set of named fields,
        and the service rebuilds the canonical form before signing — so a producer
        that offers something other than a counterparty message gets a refusal, and
        one that offers a valid message cannot make the service sign different bytes
        than the ones it rebuilt.
        """
        try:
            msg = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SigningRefused("bad-payload", str(exc)) from exc

        check_message_shape(msg)

        # Both sides must already agree on the canonical form, because the signature
        # will be verified against these bytes and stored against the engine's. A
        # disagreement here is silent otherwise: the service signs its rendering, the
        # verifier checks the engine's, and the receipt fails much later with nothing
        # pointing at the encoder.
        if canonical_bytes(msg) != payload:
            raise SigningRefused("non-canonical-payload",
                                 "the engine's canonical form is not this module's")

        raw, transport_error = self._send(canonical_bytes({"schema": REQUEST_SCHEMA, "message": msg}))

        # The answer is read before the exit status is judged. A counterparty that
        # refuses exits non-zero *and* says why on stdout; treating the exit code as
        # the verdict turns "this counterparty does not co-sign unbound receipts" into
        # "transport-failed", which sends the operator to the network instead of to
        # the policy. Only an unusable answer is a transport problem.
        resp = None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                resp = parsed
        except json.JSONDecodeError:
            resp = None

        if resp is None:
            if transport_error is not None:
                raise SigningRefused("transport-failed", transport_error)
            raise SigningRefused("bad-response", "not a response object")

        if not resp.get("ok"):
            raise SigningRefused(str(resp.get("code", "refused")), str(resp.get("detail", ""))[:200])

        signature = resp.get("signature")
        if not isinstance(signature, str) or not signature:
            raise SigningRefused("bad-response", "no signature")

        # the pin. A service that answers with a valid signature from a key we did not
        # accept is a stranger who signed correctly, which is not the same as our
        # counterparty — and would put an unaccountable second vouch in the receipt
        if not verify_signature(payload, signature, expected_pubkey=self.public_hex):
            raise SigningRefused("unpinned-counterparty",
                                 "the signature is not from the pinned counterparty key")
        return signature

    def _send(self, request: bytes) -> tuple[bytes, str | None]:
        """Return (stdout, transport error or None). A non-zero exit is reported, not
        raised: the caller decides whether stdout already explained it."""
        if self.transport is not None:
            return self.transport(request), None
        if not self.command:
            raise SigningRefused("no-transport", "set command or transport")
        try:
            proc = subprocess.run(self.command, input=request, capture_output=True,
                                  timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise SigningRefused("transport-timeout", f"{self.timeout}s") from exc
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", "replace").strip()[:200] or f"exit {proc.returncode}"
            return proc.stdout, detail
        return proc.stdout, None
