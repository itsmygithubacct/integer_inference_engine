"""Local hash-linked receipt ledger — the LOCAL stand-in for the on-chain 3rd entry.

On-chain, the third entry is an OP_RETURN on BSV the model cannot delete or reorder
(docs/receipts/RECEIPTS.md). On-chain emission is DISABLED here, so finalized receipts are appended to
a local, append-only, hash-LINKED JSONL ledger: each entry commits the previous entry's hash, so the
local log is **tamper-evident RELATIVE TO AN EXTERNALLY-TRUSTED HEAD** — editing, reordering, or
deleting any interior entry breaks the chain against a head a verifier already trusts, and
`verify_chain` localizes the break.

HONEST SCOPE: this is the local analog of the ledger's append-only property, WITHOUT a chain. It makes
tampering DETECTABLE, not IMPOSSIBLE, and only against a trusted reference point:
  * A file's HOLDER can rewrite an interior entry and then recompute that entry's `entryHash` plus
    every downstream `prevHash`/`entryHash` — the result still satisfies `verify_chain` (it is
    internally consistent). Detection requires comparing the head against an independently-held value;
    this is the same class of limit as the acknowledged tail-truncation gap below.
  * A local file's holder can still truncate the whole tail or discard the file.
Non-repudiation against the *operator* is exactly what the (disabled) on-chain 3rd entry would add (an
append-only public log no single party can rewrite); a local ledger cannot provide it.

WHAT IS COMMITTED: each entry stores a `receiptHash` COMMITMENT, not the receipt body itself. So
`verify_chain` attests the ORDERING and integrity of the hash chain — it does NOT attest the validity,
re-derivability, or even availability of the underlying receipts (that is receipts/verify.py's job).
"""
from __future__ import annotations

import os
try:
    import fcntl
except ImportError:  # pragma: no cover
    # POSIX-only: the inter-process flock around record() is unavailable off POSIX (e.g. Windows). Single-
    # process use — the normal verifier/runner path — is unaffected. With no lock, two CONCURRENT writer
    # processes could both append the same index, forking the chain; that corruption is NOT silent — the next
    # verify_chain() fails loud on the duplicate index. The portable re-verification path (the pure-NumPy
    # oracle) never reads this ledger. See docs/architecture/DETERMINISM.md (platform scope).
    fcntl = None
import json
from pathlib import Path

from .canonical import commit

GENESIS = "0" * 64


def _core(index: int, prev: str, receipt_hash: str, model_hash, ts) -> dict:
    """The fields covered by `entryHash` — kept in one place so record/verify can't drift."""
    return {"index": index, "prevHash": prev, "receiptHash": receipt_hash,
            "modelHash": model_hash, "ts": ts}


class LocalLedger:
    def __init__(self, path):
        self.path = Path(path)

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(ln) for ln in self.path.read_text().splitlines() if ln.strip()]

    def head(self) -> str:
        es = self.entries()
        return es[-1]["entryHash"] if es else GENESIS

    def record(self, receipt: dict, *, ts: str | None = None) -> dict:
        """Append one receipt as a new hash-linked entry; returns the entry. `ts` is optional/explicit
        (the library stays deterministic — the CLI stamps wall-clock time)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a+", encoding="utf-8") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                es = [json.loads(ln) for ln in f.read().splitlines() if ln.strip()]
                core = _core(len(es), es[-1]["entryHash"] if es else GENESIS,
                             receipt["receiptHash"], receipt.get("modelHash"), ts)
                entry = dict(core, entryHash=commit(core))
                f.seek(0, os.SEEK_END)
                f.write(json.dumps(entry, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return entry

    def verify_chain(self, *, expected_head: str | None = None,
                     expected_count: int | None = None) -> dict:
        """Walk the ledger; confirm each entry's index, prevHash link, and entryHash recomputation.

        Internal consistency alone CANNOT detect a tail-truncation or a full from-scratch rewrite — GENESIS
        is a public anchor, so a shorter/rebuilt-but-consistent chain still passes (see the module docstring).
        A verifier who pinned a trusted `head()` out-of-band (or relies on the on-chain OP_RETURN) can pass it
        as `expected_head` (and/or `expected_count`) to catch a dropped suffix or a wholesale rewrite. The
        result always includes the computed `head` so a caller can capture it for next time.
        """
        prev = GENESIS
        es = self.entries()
        for i, e in enumerate(es):
            core = _core(e.get("index"), e.get("prevHash"), e.get("receiptHash"),
                         e.get("modelHash"), e.get("ts"))
            if e.get("index") != i or e.get("prevHash") != prev or commit(core) != e.get("entryHash"):
                return {"ok": False, "brokenAt": i, "count": len(es), "head": prev}
            prev = e["entryHash"]
        head = es[-1]["entryHash"] if es else GENESIS
        if expected_count is not None and len(es) != expected_count:
            return {"ok": False, "brokenAt": None, "count": len(es), "head": head,
                    "reason": f"count {len(es)} != expected {expected_count} (truncation/rewrite)"}
        if expected_head is not None and head != expected_head:
            return {"ok": False, "brokenAt": None, "count": len(es), "head": head,
                    "reason": "head != expected (truncation/rewrite)"}
        return {"ok": True, "brokenAt": None, "count": len(es), "head": head}
