"""Per-inference triple-entry TEA receipts — LOCAL build / record / verify / broadcast (P4).

Implements the receipt design in docs/receipts/RECEIPTS.md entirely in this project: a receipt's two payload entries are signed
locally (receipts/signing.py, a labeled HMAC vouch), the third entry is a local hash-linked ledger
(receipts/ledger.py), and the chain artifact is published per a two-key gate (receipts/emit.py +
broadcast.py). The trustless core — recompute the commitments and re-run the bit-exact reference engine
— needs no key and no chain (receipts/verify.py).

  build_receipt   → {"receipt": <on-chain-committable>, "preimage": <off-chain ids/trace>}
  emit_receipt    → record to the local ledger + publish: dry-run to a local log (default) OR, with
                    enable_chain=True + a LocalNodeChainBackend, broadcast via the vendored chain/ TS
                    (DRY-RUN unless confirm=True). The mainnet send is a deliberate two-key interlock.
  verify_receipt  → recompute commits + receiptHash + bit-exact re-execution (+ optional sigs)
"""
from __future__ import annotations

from .canonical import canonical_bytes, commit, token_commit
from .signing import LocalKey, keygen, sign, verify, verify_signature
from .signing_ec import ECKey, ec_keygen, verify_ec, SCHEME_EC
from .receipt import build_receipt, receipt_hash, sampler_to_block, SCHEMA, PREIMAGE_SCHEMA
from .ledger import LocalLedger, GENESIS
from .broadcast import (LogBroadcastBackend, LocalNodeChainBackend, LocalNodeTeaBackend,
                        WalletThirdEntryBackend, ChainBroadcastError)
from .emit import emit_receipt, chain_artifact, ChainDisabledError, CHAIN_TAG
from .txlog import append_tx_log, log_transaction, tx_record, read_tx_log, TX_LOG_SCHEMA
from .verify import verify_receipt

__all__ = [
    "canonical_bytes", "commit", "token_commit",
    "LocalKey", "keygen", "sign", "verify", "verify_signature",
    "ECKey", "ec_keygen", "verify_ec", "SCHEME_EC",
    "build_receipt", "receipt_hash", "sampler_to_block", "SCHEMA", "PREIMAGE_SCHEMA",
    "LocalLedger", "GENESIS",
    "LogBroadcastBackend", "LocalNodeChainBackend", "LocalNodeTeaBackend", "WalletThirdEntryBackend",
    "ChainBroadcastError",
    "emit_receipt", "chain_artifact", "ChainDisabledError", "CHAIN_TAG",
    "append_tx_log", "log_transaction", "tx_record", "read_tx_log", "TX_LOG_SCHEMA",
    "verify_receipt",
]
