"""Off-chain TRANSACTION log — record every third-entry transaction we build or broadcast.

This complements the artifact `broadcast.log` (which records the chain *artifact*: modelHash / receiptHash /
samplerMode / seed). The transaction log records the *transaction itself* — its raw hex, txid, fee, size,
the OP_RETURN script, and broadcast status — so there is a complete, inspectable, re-broadcastable off-chain
audit trail of every tx, independent of the chain and of the artifact log.

Append-only JSONL, `fcntl`-locked + `fsync`'d (the same discipline as the hash-linked ledger). Each record
self-checks `txid == hash256(rawTx)` at write time when the raw tx is present, so a malformed/mismatched
entry is flagged the moment it lands (`txidVerified`).
"""
from __future__ import annotations

import os
try:
    import fcntl
except ImportError:  # pragma: no cover - Linux target, kept portable.
    fcntl = None
import json
from pathlib import Path

from .canonical import canonical_bytes
from ..hashing.sha import txid_of

TX_LOG_SCHEMA = "trinote.tx-log/v1"

# The keys copied verbatim from a backend result into the tx record (when present).
_PASSTHROUGH = ("network", "status", "txid", "fee", "sizeBytes", "satPerKb", "opReturn", "rawTx",
                "source", "changeAddress", "modelHash", "receiptHash", "broadcast")


def tx_record(onchain: dict, *, kind: str, ts: str | None = None) -> dict:
    """Normalize a broadcast-backend result (or agentd action record) into a tx-log record.

    Sets `txidVerified` to whether `hash256(rawTx)` reproduces the recorded `txid` (None if no rawTx)."""
    rec = {"schema": TX_LOG_SCHEMA, "kind": kind, "ts": ts}
    for k in _PASSTHROUGH:
        if k in onchain and onchain[k] is not None:
            rec[k] = onchain[k]
    raw = rec.get("rawTx")
    if isinstance(raw, str) and raw:
        try:
            rec["txidVerified"] = (txid_of(raw) == rec.get("txid"))
        except ValueError:
            rec["txidVerified"] = False
    else:
        rec["txidVerified"] = None
    return rec


def append_tx_log(path, record: dict) -> dict:
    """Append `record` to the JSONL tx log at `path` (locked + fsync'd). Returns the record."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_bytes(record).decode("utf-8") + "\n"
    with open(p, "a", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return record


def log_transaction(path, onchain: dict, *, kind: str, ts: str | None = None) -> dict | None:
    """Build + append a tx record from a backend result, IFF it carries a real transaction.

    A real tx has a `rawTx`, or a `txid` that is not the LogBroadcastBackend synthetic `log:` id. Returns the
    written record, or None if there was nothing to log (e.g. the local dry-run artifact log)."""
    if not isinstance(onchain, dict):
        return None
    txid = onchain.get("txid", "")
    if not onchain.get("rawTx") and (not txid or str(txid).startswith("log:")):
        return None
    return append_tx_log(path, tx_record(onchain, kind=kind, ts=ts))


def read_tx_log(path) -> list[dict]:
    """Read the tx log into a list of records (empty if the file is absent)."""
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text("utf-8").splitlines() if ln.strip()]
