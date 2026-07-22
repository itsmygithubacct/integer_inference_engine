"""Runtime helpers for the ATLAS-Notarized-Bonsai-8B native path."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Callable

from ..notary_paths import ledger_default, model_key_default, counterparty_key_default
from ..receipts import keygen, build_receipt, ECKey
from ..receipts.emit import emit_receipt
from ..receipts.ledger import LocalLedger
from ..receipts.verify import verify_receipt
from .artifact_io_bonsai import (
    _LOADED_ARTIFACT_SHA256,
    _LOADED_CONFIG_SHA256,
    _config_sha256,
)
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
BONSAI35_LABEL = "ATLAS-Notarized-Bonsai-27B"
BONSAI35_IDENTITY_FORMAT = "trinote-bonsai35-identity/1"
BONSAI35_IDENTITY_ENGINE = "int-ref@bonsai-qwen35"
BONSAI35_RELEASE_ARTIFACT_SHA256 = "7eab414ceff3fff1489053d415d0c6adb1e646e552d091cc1a898d0456adf3fb"
BONSAI35_RELEASE_GGUF_SHA256 = "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0"
BONSAI35_GATE_METRIC = "teacher-forced-top1-agreement-vs-prismml-libllama"
BONSAI35_PRISM_RUNTIME_RELEASE = (
    "prism-b9591-62061f9",
    "62061f91088281e65071cc38c5f69ee95c39f14e",
    "67c64046abcf73bf489e27c9ebe7525f5b77c58db9490d1d711efe6e17bf2975",
)
BONSAI35_WEIGHT_PROVENANCE = {
    "ggufFile": "Bonsai-27B-Q1_0.gguf",
    "ggufSha256": BONSAI35_RELEASE_GGUF_SHA256,
    "importer": "trinote.infer_int.import_bonsai35_gguf",
    "kind": "imported-weights",
    "quant": "GGUF Q1_0 g128",
    "source": "prism-ml/Bonsai-27B-gguf",
}


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _gate_count(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"Bonsai-27B quality gate {field} must be an integer >= {minimum}")
    return value


def _gate_ratio(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Bonsai-27B quality gate {field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"Bonsai-27B quality gate {field} must be finite and in [0, 1]")
    return result


def _label_for_model(model) -> str:
    return BONSAI35_LABEL if str(model.cfg.get("architecture", "")) == "qwen35" else BONSAI_LABEL


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


def validate_bonsai35_receipt_identity(
    identity_path: str | Path | None,
    artifact_digest: str,
) -> dict:
    """Fail closed unless a 27B identity carries consistent release evidence.

    Merely writing ``{"modelHash": ...}`` is sufficient for the generic Bonsai
    model-binding primitive, but it is intentionally *not* sufficient to turn
    on Bonsai-27B receipt issuance.  The 27B optimization contract requires a
    distinct engine identity and a stored, hash-linked PrismML fidelity gate.

    This validates a local integrity/provenance record, not authenticity: the
    identity and gate JSON are unsigned and an actor able to replace both can
    forge a mutually consistent pair.  Receipt signatures authenticate the
    receipt signer; they do not make this local quality evidence unforgeable.
    """

    if identity_path is None:
        raise ValueError("Bonsai-27B receipts require an explicit 27B identity file")
    if not _is_sha256(artifact_digest):
        raise ValueError("Bonsai-27B loaded artifact digest is not a lowercase SHA-256")
    path = Path(identity_path)
    try:
        identity = json.loads(path.read_text())
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as exc:
        raise ValueError(f"Bonsai-27B identity is unreadable/malformed: {exc}") from exc
    if not isinstance(identity, dict):
        raise ValueError("Bonsai-27B identity must be a JSON object")
    if identity.get("format") != BONSAI35_IDENTITY_FORMAT:
        raise ValueError("Bonsai-27B identity has the wrong format")
    if identity.get("modelHash") != str(artifact_digest):
        raise ValueError("Bonsai-27B identity modelHash does not match the loaded artifact")
    if artifact_digest != BONSAI35_RELEASE_ARTIFACT_SHA256:
        raise ValueError("Bonsai-27B identity is not for the pinned release artifact")
    if identity.get("inferenceEngine") != BONSAI35_IDENTITY_ENGINE:
        raise ValueError("Bonsai-27B identity has the wrong inferenceEngine")
    if identity.get("name") != BONSAI35_LABEL:
        raise ValueError("Bonsai-27B identity has the wrong model name")

    provenance = identity.get("weightProvenance")
    if not isinstance(provenance, dict):
        raise ValueError("Bonsai-27B identity lacks release weight provenance")
    for field, expected in BONSAI35_WEIGHT_PROVENANCE.items():
        if provenance.get(field) != expected:
            raise ValueError(f"Bonsai-27B identity has the wrong weightProvenance.{field}")
    gguf_digest = BONSAI35_RELEASE_GGUF_SHA256

    gate_ref = identity.get("qualityGate")
    if not isinstance(gate_ref, dict) or gate_ref.get("verdict") != "PASS":
        raise ValueError("Bonsai-27B identity does not declare a passing quality gate")
    gate_name = gate_ref.get("gateFile")
    gate_hash = gate_ref.get("gateHash")
    if (not isinstance(gate_name, str) or not gate_name or Path(gate_name).name != gate_name
            or gate_name in {".", ".."} or not _is_sha256(gate_hash)):
        raise ValueError("Bonsai-27B identity quality-gate reference is malformed")
    gate_path = path.parent / gate_name
    try:
        gate_bytes = gate_path.read_bytes()
        gate = json.loads(gate_bytes)
    except FileNotFoundError as exc:
        raise ValueError(f"Bonsai-27B quality-gate file is missing: {gate_path}") from exc
    except (OSError, ValueError) as exc:
        raise ValueError(f"Bonsai-27B quality-gate file is unreadable/malformed: {exc}") from exc
    if hashlib.sha256(gate_bytes).hexdigest() != gate_hash:
        raise ValueError("Bonsai-27B quality-gate file digest does not match the identity")
    if not isinstance(gate, dict) or gate.get("architecture") != "qwen35":
        raise ValueError("Bonsai-27B quality gate is for the wrong architecture")
    if gate.get("artifactSha256") != str(artifact_digest) or gate.get("verdict") != "PASS":
        raise ValueError("Bonsai-27B quality gate is not a PASS for the loaded artifact")
    if gate.get("ggufSha256") != gguf_digest:
        raise ValueError("Bonsai-27B quality gate GGUF digest does not match the identity")
    if gate.get("metric") != BONSAI35_GATE_METRIC:
        raise ValueError("Bonsai-27B quality gate did not use the pinned libllama logits metric")
    if gate.get("generatedOnly") is not False or gate.get("producer") != "native":
        raise ValueError("Bonsai-27B quality gate did not compare the optimized producer to libllama logits")
    prism = gate.get("prism")
    if (not isinstance(prism, dict)
            or not _is_sha256(prism.get("teacherHarnessSha256"))
            or tuple(prism.get("runtimeRelease") or ()) != BONSAI35_PRISM_RUNTIME_RELEASE):
        raise ValueError("Bonsai-27B quality gate lacks pinned Prism runtime/harness identity")
    count = _gate_count(gate.get("count"), field="count", minimum=10)
    cases = gate.get("cases")
    if not isinstance(cases, list) or len(cases) < 5:
        raise ValueError("Bonsai-27B quality gate is only a smoke sample, not an identity gate")
    value = _gate_ratio(gate.get("value"), field="value")
    threshold = _gate_ratio(gate.get("threshold"), field="threshold")
    target_value = _gate_ratio(gate.get("targetAgreement"), field="targetAgreement")
    target_threshold = _gate_ratio(gate.get("targetThreshold"), field="targetThreshold")
    if threshold < 0.80 or target_threshold < 0.50:
        raise ValueError("Bonsai-27B quality gate weakened the release thresholds")
    if gate.get("top1Pass") is not True or gate.get("targetPass") is not True:
        raise ValueError("Bonsai-27B quality gate pass flags are not both true")

    top1_matches = _gate_count(gate.get("top1Matches"), field="top1Matches")
    target_matches = _gate_count(gate.get("targetMatches"), field="targetMatches")
    if top1_matches > count or target_matches > count:
        raise ValueError("Bonsai-27B quality gate aggregate matches exceed count")
    if (not math.isclose(top1_matches / count, value, rel_tol=0.0, abs_tol=1e-15)
            or not math.isclose(
                target_matches / count, target_value, rel_tol=0.0, abs_tol=1e-15
            )):
        raise ValueError("Bonsai-27B quality gate aggregate ratios are inconsistent")

    case_count = 0
    case_top1_matches = 0
    case_target_matches = 0
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Bonsai-27B quality gate case {index} is malformed")
        current_count = _gate_count(
            case.get("count"), field=f"cases[{index}].count", minimum=1
        )
        current_top1 = _gate_count(
            case.get("top1Matches"), field=f"cases[{index}].top1Matches"
        )
        current_target = _gate_count(
            case.get("targetMatches"), field=f"cases[{index}].targetMatches"
        )
        if current_top1 > current_count or current_target > current_count:
            raise ValueError(f"Bonsai-27B quality gate case {index} matches exceed count")
        current_top1_ratio = _gate_ratio(
            case.get("top1Agreement"), field=f"cases[{index}].top1Agreement"
        )
        current_target_ratio = _gate_ratio(
            case.get("targetAgreement"), field=f"cases[{index}].targetAgreement"
        )
        if (not math.isclose(
                current_top1 / current_count,
                current_top1_ratio,
                rel_tol=0.0,
                abs_tol=1e-15,
            ) or not math.isclose(
                current_target / current_count,
                current_target_ratio,
                rel_tol=0.0,
                abs_tol=1e-15,
            )):
            raise ValueError(f"Bonsai-27B quality gate case {index} ratios are inconsistent")
        case_count += current_count
        case_top1_matches += current_top1
        case_target_matches += current_target
    if (case_count != count or case_top1_matches != top1_matches
            or case_target_matches != target_matches):
        raise ValueError("Bonsai-27B quality gate case counts are inconsistent")
    if value < threshold:
        raise ValueError("Bonsai-27B quality gate top-1 agreement is below threshold")
    if target_value < target_threshold:
        raise ValueError("Bonsai-27B quality gate generated-target agreement is below threshold")

    # The identity repeats the release-critical summaries so a reader does not
    # have to trust stale display fields.  Require every repeated field to be
    # exactly consistent with the hash-linked gate.
    summary_exact = {
        "caseCount": len(cases),
        "count": count,
        "metric": BONSAI35_GATE_METRIC,
        "prismRuntimeRelease": list(BONSAI35_PRISM_RUNTIME_RELEASE),
        "producer": "native",
        "targetPass": True,
        "teacherHarnessSha256": prism["teacherHarnessSha256"],
        "top1Pass": True,
        "verdict": "PASS",
    }
    for field, expected in summary_exact.items():
        if gate_ref.get(field) != expected:
            raise ValueError(f"Bonsai-27B identity qualityGate.{field} disagrees with its gate")
    for field, expected in (
        ("value", value),
        ("threshold", threshold),
        ("targetAgreement", target_value),
        ("targetThreshold", target_threshold),
    ):
        actual = _gate_ratio(gate_ref.get(field), field=f"identity.qualityGate.{field}")
        if actual != expected:
            raise ValueError(f"Bonsai-27B identity qualityGate.{field} disagrees with its gate")
    return identity


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


def _validate_bonsai35_fresh_oracle(
    producer,
    verifier,
    *,
    model_digest: str,
) -> None:
    """Require a separately loaded, unaccelerated canonical 27B oracle.

    Python callers are inside the trust boundary and can mutate private state;
    this is a fail-closed API invariant, not a sandbox or authenticity proof.
    """

    # Import lazily so the 8B receipt path does not pay for the 27B graph and to
    # keep this shared helper independent of model-module import order.
    from .reference_bonsai35 import BonsaiQwen35ReferenceModel

    if type(producer) is not BonsaiQwen35ReferenceModel:
        raise ValueError("Bonsai-27B receipt producer must be the canonical Qwen3.5 model class")
    if type(verifier) is not BonsaiQwen35ReferenceModel:
        raise ValueError("Bonsai-27B receipt verifier must be the exact canonical Qwen3.5 model class")
    if verifier is producer:
        raise ValueError("Bonsai-27B receipt verifier must be a distinct fresh oracle instance")
    if verifier.artifact is producer.artifact:
        raise ValueError("Bonsai-27B receipt verifier must use a separately loaded artifact")
    if str(verifier.cfg.get("architecture", "")) != "qwen35":
        raise ValueError("Bonsai-27B receipt verifier has the wrong architecture")

    sentinel = object()
    if (getattr(verifier, "_native", sentinel) is not False
            or getattr(verifier, "_native_runtime", sentinel) is not None
            or getattr(verifier, "_model_executor", sentinel) is not None):
        raise ValueError(
            "Bonsai-27B receipt verifier must be a fresh canonical CPU oracle "
            "with no native runtime"
        )

    producer_digest = producer.artifact.get(_LOADED_ARTIFACT_SHA256)
    verifier_digest = verifier.artifact.get(_LOADED_ARTIFACT_SHA256)
    if producer_digest != model_digest or verifier_digest != model_digest:
        raise ValueError(
            "Bonsai-27B producer/verifier must both be loaded from the committed artifact digest"
        )
    producer_config_digest = producer.artifact.get(_LOADED_CONFIG_SHA256)
    verifier_config_digest = verifier.artifact.get(_LOADED_CONFIG_SHA256)
    if (producer_config_digest != _config_sha256(producer.cfg)
            or verifier_config_digest != _config_sha256(verifier.cfg)
            or producer_config_digest != verifier_config_digest
            or producer.cfg != verifier.cfg):
        raise ValueError("Bonsai-27B producer/verifier loaded artifact configs do not match")


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
                                   ts: str | None = None,
                                   telemetry: dict | None = None) -> tuple[dict, dict, dict]:
    """Build, verify, and emit one Bonsai receipt.

    Fails closed when an identity file is supplied and its `modelHash` does not match the loaded artifact
    digest. `verifier_model` lets the caller re-execute on a fresh slow oracle while the producer uses
    fast/native kernels. Returns `(bundle, verification, emission)`.

    `model_key`/`counterparty_key` select the receipt signature scheme. Pass `ECKey`s for a real deployment
    (third-party-verifiable secp256k1 — verified from the committed public key, no shared secret). If omitted,
    the legacy symmetric HMAC demo constants are used (back-compat; the vouch proves wiring, not authenticity).
    """
    preparation_started = time.monotonic()
    if verifier_mode not in {"fast-local", "fresh-oracle"}:
        raise ValueError(f"unknown Bonsai verifier mode {verifier_mode!r}")
    if verifier_mode == "fresh-oracle" and verifier_model is None:
        raise ValueError("verifier_mode='fresh-oracle' requires verifier_model")
    architecture = str(model.cfg.get("architecture", ""))
    if architecture == "qwen35" or model_digest == BONSAI35_RELEASE_ARTIFACT_SHA256:
        # Enforce the 27B release gate at the shared receipt API, not only in
        # run_bonsai_cli.  Otherwise a direct library caller could bypass the
        # artifact-bound fidelity evidence and independent re-execution that
        # the optimized producer contract requires.
        if verifier_mode != "fresh-oracle":
            raise ValueError("Bonsai-27B receipts require verifier_mode='fresh-oracle'")
        if verifier_model is None:
            raise ValueError("Bonsai-27B receipts require a fresh CPU oracle verifier")
        validate_bonsai35_receipt_identity(identity_path, model_digest)
        _validate_bonsai35_fresh_oracle(
            model, verifier_model, model_digest=model_digest
        )
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
    model_label = _label_for_model(model)
    if telemetry is not None:
        telemetry["receiptPreparationSeconds"] = time.monotonic() - preparation_started
    receipt_started = time.monotonic()
    bundle = build_receipt(
        model_hash=model_hash,
        input_ids=input_ids,
        output_ids=output_ids,
        sampler=sampler,
        model_key=mk,
        counterparty_key=ck,
        model_label=model_label,
        artifact_digest=model_digest,
        fp_frac_bits=int(model.cfg["frac"]),   # v2: commit the sampler at the engine's fixed-point scale
    )
    if telemetry is not None:
        telemetry["receiptConstructionSeconds"] = time.monotonic() - receipt_started
    # Asymmetric keys verify from the committed public key (pin the signer = identity binding); symmetric
    # HMAC keys are passed through so the legacy vouch can still be checked.
    verification_started = time.monotonic()
    verification = verify_receipt(
        bundle,
        model=verifier_model if verifier_model is not None else model,
        model_digest=model_digest,
        model_key=None if getattr(mk, "public_hex", None) else mk,
        counterparty_key=None if getattr(ck, "public_hex", None) else ck,
        model_pubkey=getattr(mk, "public_hex", None),
        counterparty_pubkey=getattr(ck, "public_hex", None),
    )
    if telemetry is not None:
        telemetry["verificationSeconds"] = time.monotonic() - verification_started
        telemetry["verificationStrategy"] = (verification.get("reexec") or {}).get("strategy")
        telemetry["verificationCheckedTokens"] = (verification.get("reexec") or {}).get("checked")
    verification["verificationMode"] = verifier_mode
    if not verification["ok"]:
        if telemetry is not None:
            telemetry["emissionSeconds"] = 0.0
        return bundle, verification, {}
    emission_started = time.monotonic()
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
    if telemetry is not None:
        telemetry["emissionSeconds"] = time.monotonic() - emission_started
    return bundle, verification, emission
