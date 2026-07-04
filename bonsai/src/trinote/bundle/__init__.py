"""Portable receipt bundles — package an inference receipt + its on-chain anchor for offline audit.

A *receipt bundle* is the self-contained, content-addressed artifact a third party needs to verify a
notarized Bonsai inference WITHOUT trusting the producer: the receipt (the on-chain-committable half),
the preimage (the off-chain token ids + sampler/trace), the chain artifact, and a description of where
the third entry landed on BSV (standalone OP_RETURN, or a stateful AgentTea `executeAction`).

  pack_bundle    → write a bundle dir (or .tar.gz) with a manifest that commits every file (bundleHash)
  verify_bundle  → three independent layers, each optional and separately reported:
                     1. OFFLINE  — recompute file hashes, receiptHash, commitments, bundleHash (stdlib only)
                     2. ON-CHAIN — fetch the tx from WhatsOnChain, parse the OP_RETURN, bind it to the
                                   receipt (stateful: recompute the AgentTea action hash + walk to genesis)
                     3. RE-EXEC  — load the model and re-run the bit-exact reference engine (verify_receipt)

The split mirrors docs/receipts/RECEIPTS.md: the trustless core (layers 1 + 3) needs no key and no chain;
layer 2 proves the third entry is published and immutable on the public ledger.
"""
from __future__ import annotations

from .pack import pack_bundle, BUNDLE_SCHEMA, BundleError
from .verify import verify_bundle, load_bundle
from .stateful import agent_action_receipt_hash

__all__ = [
    "pack_bundle", "verify_bundle", "load_bundle",
    "agent_action_receipt_hash", "BUNDLE_SCHEMA", "BundleError",
]
