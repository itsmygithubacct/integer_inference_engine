"""Runtime helpers for the ATLAS-Notarized-Bonsai-8B native path."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import warnings
from pathlib import Path
from typing import Callable

from ..notary_paths import ledger_default, model_key_default, counterparty_key_default
from ..receipts import keygen, build_receipt, ECKey
from ..receipts.emit import emit_receipt
from ..receipts.ledger import LocalLedger
from ..receipts.verify import verify_receipt
from .sampler import SamplerConfig, sample_token

_DEMO_KEY_WARNED = False


def _maybe_warn_demo_keys() -> None:
    """Warn ONCE if a receipt is being signed with the hardcoded demo HMAC keys (no authenticity). Silent
    under pytest and when TRINOTE_DEMO_KEYS_OK is set (the constants are load-bearing for deterministic
    tests/snapshots — see the call site)."""
    global _DEMO_KEY_WARNED
    if _DEMO_KEY_WARNED or os.environ.get("TRINOTE_DEMO_KEYS_OK") or "PYTEST_CURRENT_TEST" in os.environ:
        return
    _DEMO_KEY_WARNED = True
    warnings.warn(
        "Bonsai receipt signed with the DEMO local-hmac keys (hardcoded PUBLIC constants): the 1st/2nd-entry "
        "vouches carry NO authenticity. Pass model_key/counterparty_key (e.g. receipts.ec_keygen) for a real, "
        "third-party-verifiable signature. Set TRINOTE_DEMO_KEYS_OK=1 to silence.",
        stacklevel=3,
    )

BONSAI_LABEL = "ATLAS-Notarized-Bonsai-8B"


def load_or_generate_signing_keys(model_key_path: str | Path | None = None,
                                  counterparty_key_path: str | Path | None = None) -> tuple[ECKey, ECKey]:
    """Load (or generate + persist, chmod 0600) the real secp256k1 receipt signing keys.

    These are THIRD-PARTY-VERIFIABLE: a receipt carries the signer's PUBLIC key, so anyone can verify the
    1st/2nd-entry vouches with no shared secret (unlike the legacy demo HMAC). Keys default to
    ``~/.local/trinote/keys/`` and are created on first use if absent — so a deployment "just works" and is
    authentic, while a caller can supply pre-provisioned key paths instead. Same curve (secp256k1) as the BSV
    chain, so one identity spans the off-chain receipt and the on-chain third entry. Returns (model, counterparty)."""
    mp = Path(model_key_path) if model_key_path else Path(model_key_default())
    cp = Path(counterparty_key_path) if counterparty_key_path else Path(counterparty_key_default())
    model = ECKey.load_or_generate(mp, label=BONSAI_LABEL + " model")
    counterparty = ECKey.load_or_generate(cp, label=BONSAI_LABEL + " counterparty")
    return model, counterparty


def _demo_keys_requested() -> bool:
    """Use the deterministic legacy HMAC demo keys ONLY for tests/snapshots (pytest) or when explicitly opted
    in (TRINOTE_DEMO_KEYS_OK) — those need byte-stable receiptHashes. Real runs get authentic EC keys."""
    return bool(os.environ.get("TRINOTE_DEMO_KEYS_OK")) or "PYTEST_CURRENT_TEST" in os.environ


def identity_model_hash(identity_path: str | Path | None) -> str | None:
    """The modelHash an identity binds the receipt to, or None when NO identity is requested.

    Fail-closed distinction (review finding #5): `identity_path is None` means binding is off and
    returns None; a path that is SUPPLIED but missing / unreadable / malformed RAISES rather than
    returning None. Otherwise a typo'd or not-yet-minted identity path would silently skip the
    modelHash binding (callers treat None as 'binding off') and emit/broadcast a receipt that is
    not bound to the on-chain minted identity. The CLI's FileNotFoundError handler is meant to be
    fatal here."""
    if identity_path is None:
        return None
    p = Path(identity_path)
    if not p.exists():
        raise FileNotFoundError(f"identity file not found: {p} (binding requested but unreadable)")
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"identity file {p} is unreadable/malformed: {exc}") from exc
    # Require a non-empty STRING modelHash: an explicit JSON null (or a non-string) would otherwise
    # return None and be read as 'binding off', silently emitting an unbound receipt for a
    # partially-minted/template identity (review-2 #5/#12). Close null/empty/non-string uniformly.
    mh = data.get("modelHash") if isinstance(data, dict) else None
    if not isinstance(mh, str) or not mh:
        raise ValueError(f"identity file {p} has no usable string 'modelHash' (got {mh!r})")
    return mh


def generate_bonsai_tokens(model, input_ids: list[int], max_new: int, *,
                           sampler: SamplerConfig, eos: int | None = None,
                           on_token: Callable[[int], None] | None = None) -> list[int]:
    """Generate new tokens on the native deterministic Bonsai reference path."""
    frac = int(model.cfg["frac"])
    if (sampler.mode == "greedy" and sampler.rep_penalty == 0 and sampler.no_repeat_ngram == 0
            and hasattr(model, "generate_greedy_tokens_cached")):
        return model.generate_greedy_tokens_cached(input_ids, max_new, eos=eos, on_token=on_token)
    if hasattr(model, "generate_cached"):
        return model.generate_cached(
            input_ids,
            max_new,
            lambda row, pos, hist: sample_token(row, sampler, position=pos, frac_bits=frac, history_ids=hist),
            eos=eos,
            on_token=on_token,
        )
    seq = list(input_ids)
    out: list[int] = []
    ctx = min(int(model.cfg["context_len"]), int(model.artifact["cos_fp"].shape[0]))
    for _ in range(max_new):
        row = model.forward(seq[-ctx:], last_only=True)[0]
        tok = sample_token(row, sampler, position=len(seq), frac_bits=frac, history_ids=seq)
        seq.append(tok)
        out.append(tok)
        if eos is not None and int(tok) == int(eos):
            break
        if on_token is not None:
            on_token(tok)
    return out


def emit_and_verify_bonsai_receipt(model, *, input_ids, output_ids, model_digest: str,
                                   sampler: SamplerConfig | dict,
                                   verifier_model=None,
                                   verifier_mode: str = "fast-local",
                                   identity_path: str | Path | None = None,
                                   ledger_path: str | Path | None = None,
                                   broadcast_log: str | Path | None = None,
                                   broadcast_to_log: bool = True,
                                   chain_artifacts_dir: str | Path | None = None,
                                   model_key=None, counterparty_key=None,
                                   enable_chain: bool = False, chain_backend=None,
                                   tx_log: str | Path | None = None,
                                   ts: str | None = None) -> tuple[dict, dict, dict]:
    """Build, verify, and emit one Bonsai receipt.

    Fails closed when an identity file is supplied and its `modelHash` does not match the loaded artifact
    digest. `verifier_model` lets the caller re-execute on a fresh slow oracle while the producer uses
    fast/native kernels. Returns `(bundle, verification, emission)`.

    `model_key`/`counterparty_key` select the receipt signature scheme. Pass `ECKey`s for a real deployment
    (third-party-verifiable secp256k1 — verified from the committed public key, no shared secret). If omitted,
    the legacy symmetric HMAC demo constants are used (back-compat; the vouch proves wiring, not authenticity).
    """
    if verifier_mode not in {"fast-local", "fresh-oracle"}:
        raise ValueError(f"unknown Bonsai verifier mode {verifier_mode!r}")
    if verifier_mode == "fresh-oracle" and verifier_model is None:
        raise ValueError("verifier_mode='fresh-oracle' requires verifier_model")
    # Generated state lives OUTSIDE the repo (default ~/.local/trinote/receipts); a bare
    # call never pollutes the working tree. broadcast_log stays None here -> emit_receipt
    # resolves it the same way (broadcast_log_default) only when broadcast_to_log is on.
    if ledger_path is None:
        ledger_path = ledger_default()
    bound_hash = identity_model_hash(identity_path)
    if bound_hash is not None and bound_hash != model_digest:
        raise ValueError(
            f"artifact digest {model_digest} != identity modelHash {bound_hash}"
        )
    model_hash = bound_hash or model_digest
    # Key selection. Explicit keys ("given") always win. Otherwise:
    #   * real runs  -> authentic secp256k1 EC keys, load-or-generated under ~/.local/trinote/keys (created on
    #     first use) — third-party-verifiable from the committed public key, no shared secret.
    #   * tests/snapshots (pytest or TRINOTE_DEMO_KEYS_OK) -> the deterministic legacy HMAC demo constants,
    #     which keep receiptHashes byte-stable. These are HARDCODED PUBLIC CONSTANTS (no authenticity) and are
    #     load-bearing for the snapshot tests — do NOT randomize them.
    if model_key is None or counterparty_key is None:
        if _demo_keys_requested():
            dmk = keygen(label="atlas-notarized-bonsai", secret_hex="11" * 32)
            dck = keygen(label="counterparty", secret_hex="22" * 32)
        else:
            dmk, dck = load_or_generate_signing_keys()
    mk = model_key if model_key is not None else dmk
    ck = counterparty_key if counterparty_key is not None else dck
    bundle = build_receipt(
        model_hash=model_hash,
        input_ids=input_ids,
        output_ids=output_ids,
        sampler=sampler,
        model_key=mk,
        counterparty_key=ck,
        model_label=BONSAI_LABEL,
        artifact_digest=model_digest,
        fp_frac_bits=int(model.cfg["frac"]),   # v2: commit the sampler at the engine's fixed-point scale
    )
    # Asymmetric keys verify from the committed public key (pin the signer = identity binding); symmetric
    # HMAC keys are passed through so the legacy vouch can still be checked.
    verification = verify_receipt(
        bundle,
        model=verifier_model if verifier_model is not None else model,
        model_digest=model_digest,
        model_key=None if getattr(mk, "public_hex", None) else mk,
        counterparty_key=None if getattr(ck, "public_hex", None) else ck,
        model_pubkey=getattr(mk, "public_hex", None),
        counterparty_pubkey=getattr(ck, "public_hex", None),
    )
    verification["verificationMode"] = verifier_mode
    if not verification["ok"]:
        return bundle, verification, {}
    emission = emit_receipt(
        bundle["receipt"],
        ledger=LocalLedger(ledger_path),
        ts=ts or datetime.now(timezone.utc).isoformat(),
        chain_artifacts_dir=chain_artifacts_dir,
        broadcast_to_log=broadcast_to_log,
        broadcast_log=broadcast_log,
        enable_chain=enable_chain,
        chain_backend=chain_backend,
        tx_log=tx_log,
    )
    return bundle, verification, emission
