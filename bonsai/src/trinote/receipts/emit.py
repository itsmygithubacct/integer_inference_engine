"""Finalize a receipt — record it LOCALLY, then publish the 3rd entry. Mainnet broadcast is GATED.

`emit_receipt` always records the receipt to the local hash-linked ledger (the local 3rd entry) and
always produces the canonical inference-to-chain artifact JSON (docs/receipts/RECEIPTS.md). Then it
PUBLISHES the chain artifact via one of three modes, by a two-key interlock:

  enable_chain=False, broadcast_to_log=True   (DEFAULT) → LogBroadcastBackend: dry-run "broadcast" to a
                                                          local JSONL log; status "logged". No network.
  enable_chain=False, broadcast_to_log=False            → no publish; status "disabled".
  enable_chain=True                                     → a REAL `chain_backend`; status from the backend
                                                          ("broadcast"/"dry-run"). Missing → ChainDisabledError.
                                                          Backends: LocalNodeChainBackend (standalone
                                                          OP_RETURN) or LocalNodeTeaBackend (stateful
                                                          RicardianTea executeTea third entry — local
                                                          dry-run core, see broadcast.py / docs/receipts/RECEIPTS.md).

WhatsOnChain is mainnet-only (real money), so a real send needs BOTH `enable_chain=True` AND a backend
whose own confirm gate is on (`LocalNodeChainBackend(confirm=True)` → the vendored TS gate
`CONFIRM_MAINNET_BROADCAST=yes`). The default is local + log only; nothing leaves the box.
"""
from __future__ import annotations

from pathlib import Path

from ..notary_paths import broadcast_log_default
from .broadcast import LogBroadcastBackend
from .canonical import canonical_bytes
from .ledger import LocalLedger

CHAIN_TAG = "trinote/r1"            # the OP_RETURN protocol tag a real 3rd entry carries (RECEIPTS.md)
CHAIN_SCHEMA = "trinote.chain-receipt/v2"   # v2: also carries the committed sampler mode + seed (the draw nonce)


class ChainDisabledError(RuntimeError):
    """Raised when a real on-chain broadcast is requested (enable_chain=True) but no backend is wired."""


def chain_artifact(receipt: dict) -> dict:
    """The canonical inference-to-chain artifact the broadcaster turns into an OP_RETURN (the third entry).

    Carries the commitments (modelHash, receiptHash) PLUS the reproducibility nonce — the committed sampler
    `mode` and `seed`, read from the receipt's trace. This records ON-CHAIN *how* the output was drawn, so a
    randomized seed is notarized in the third entry itself (not only off-chain in the preimage). The seed is
    already bound transitively via receiptHash — recompute it from the preimage to prove the on-chain seed is
    the one that produced the output; this surfaces it directly. NOTE (standalone repo): the vendored OP_RETURN
    builder lives in `chain/` (parent repo, not bundled — see docs/receipts/RECEIPTS.md 'Scope'); to land `seed` in the
    literal OP_RETURN bytes that script must encode the field. The default LogBroadcastBackend logs the full
    artifact (seed included)."""
    s = (receipt.get("trace") or {}).get("sampler") or {}
    return {"schema": CHAIN_SCHEMA, "tag": CHAIN_TAG,
            "modelHash": receipt["modelHash"], "receiptHash": receipt["receiptHash"],
            "samplerMode": s.get("mode", "greedy"), "seed": int(s.get("seed", 0))}


def emit_receipt(receipt: dict, *, ledger: LocalLedger, ts: str | None = None,
                 chain_artifacts_dir=None, enable_chain: bool = False,
                 broadcast_to_log: bool = True, chain_backend=None, broadcast_log=None,
                 tx_log=None) -> dict:
    """Record `receipt` locally and publish its chain artifact per the gate. Returns the emission record.

    When `tx_log` is set and the broadcast produced a real transaction (it carries a `rawTx` / a real txid),
    the full transaction is ALSO appended to that off-chain JSONL transaction log (receipts/txlog.py) — a
    complete, re-broadcastable audit trail alongside the artifact `broadcast_log`."""
    entry = ledger.record(receipt, ts=ts)               # 3rd entry, local edition (always)

    artifact = chain_artifact(receipt)                  # produced regardless of publish mode
    artifact_path = None
    if chain_artifacts_dir is not None:
        d = Path(chain_artifacts_dir)
        d.mkdir(parents=True, exist_ok=True)
        artifact_path = d / f"receipt-{receipt['receiptHash']}.json"
        artifact_path.write_bytes(canonical_bytes(artifact))

    if enable_chain:
        if chain_backend is None:
            raise ChainDisabledError(
                "enable_chain=True but no chain_backend is wired. Pass an enabled chain backend — e.g. "
                "LocalNodeChainBackend(confirm=True) for a real broadcast (the two-key interlock; see "
                "docs/receipts/RECEIPTS.md), or leave enable_chain=False for the local log. Receipts "
                "stay local until you do.")
        result = chain_backend.broadcast(artifact, ts=ts)
        onchain = {"status": result.get("status", "broadcast"), **result}
    elif broadcast_to_log:
        result = LogBroadcastBackend(broadcast_log or broadcast_log_default()).broadcast(artifact, ts=ts)
        onchain = {"status": "logged", **result,
                   "reason": "dry-run broadcast to local log (enable_chain=False)"}
    else:
        onchain = {"status": "disabled",
                   "reason": "no publish (enable_chain=False, broadcast_to_log=False); 3rd entry is the local ledger"}

    tx_log_record = None
    if tx_log is not None:
        from .txlog import log_transaction
        tx_log_record = log_transaction(tx_log, onchain, kind="standalone", ts=ts)

    return {"thirdEntry": "local-ledger", "ledgerEntry": entry, "chainArtifact": artifact,
            "chainArtifactPath": str(artifact_path) if artifact_path else None, "onchain": onchain,
            "txLogPath": str(tx_log) if (tx_log is not None and tx_log_record is not None) else None}
