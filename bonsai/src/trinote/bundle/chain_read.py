"""Read-only BSV chain access for bundle verification — stdlib only (urllib + hashlib), no bsv-sdk.

The verifier must be runnable by anyone, with nothing beyond the Python standard library, so it can confirm
a third entry is really on-chain without installing the wallet stack. This module fetches a raw transaction
from WhatsOnChain, parses its outputs, decodes OP_RETURN data pushes, and binds them to a receipt:

  * standalone third entry  → OP_FALSE OP_RETURN <tag> <modelHash> <receiptHash>  (data items = 3)
  * stateful  third entry   → OP_FALSE OP_RETURN <receiptHash>                     (data items = 1)
    where receiptHash is the AgentTea action hash recomputed in stateful.py.

It also walks an identity UTXO chain BACKWARD (input[0] → previous identity tx) to prove a stateful action
sits on the contract's official history, all the way to the deploy/genesis tx. Network access is OPT-IN:
nothing here runs unless the caller asks for it.

Consumers / spec: the on-chain layer of `verify.py::_verify_onchain` calls these; the OP_RETURN formats and
the AgentTea action-hash preimage are documented in `docs/receipts/THIRD-ENTRY.md`,
`docs/receipts/RECEIPT-BUNDLE.md`, and `stateful.py`.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

_WOC = "https://api.whatsonchain.com/v1/bsv"
_OP_FALSE = 0x00
_OP_RETURN = 0x6a
_VALID_NETWORKS = frozenset({"main", "test", "stn"})


class ChainReadError(RuntimeError):
    """Network/parse failure while reading the chain (fail closed: the caller treats this as not-verified)."""


def _validate_network(network: str) -> str:
    """Reject any network not served by WhatsOnChain BEFORE it is interpolated into a
    request URL. The `network` reaching here is untrusted (caller- or bundle-supplied):
    an unvalidated value is both a path/query-injection vector and — via 'test' — a way
    to downgrade a costly mainnet anchor to a free testnet write. Fail closed."""
    if network not in _VALID_NETWORKS:
        raise ChainReadError(f"invalid network {network!r} (expected one of {sorted(_VALID_NETWORKS)})")
    return network


# --------------------------------------------------------------------------------------------------
# WhatsOnChain fetch (read-only; no key, no broadcast)
# --------------------------------------------------------------------------------------------------
def fetch_raw_tx(txid: str, network: str = "main", *, timeout: float = 20.0) -> str:
    """Return the raw transaction hex for `txid`, or raise ChainReadError. Read-only WoC GET."""
    if not (len(txid) == 64 and all(c in "0123456789abcdef" for c in txid.lower())):
        raise ChainReadError(f"not a txid: {txid!r}")
    _validate_network(network)
    url = f"{_WOC}/{network}/tx/{txid}/hex"
    req = urllib.request.Request(url, headers={"User-Agent": "trinote-receipt-bundle/1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("ascii").strip()
    except urllib.error.HTTPError as exc:
        raise ChainReadError(f"WoC {exc.code} for tx {txid} on {network}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ChainReadError(f"WoC unreachable for tx {txid}: {exc}") from exc


def fetch_tx_status(txid: str, network: str = "main", *, timeout: float = 20.0) -> dict:
    """Return WoC's tx status JSON ({blockheight, confirmations, ...}) or raise ChainReadError."""
    if not (len(txid) == 64 and all(c in "0123456789abcdef" for c in txid.lower())):
        raise ChainReadError(f"not a txid: {txid!r}")
    _validate_network(network)
    url = f"{_WOC}/{network}/tx/hash/{txid}"
    req = urllib.request.Request(url, headers={"User-Agent": "trinote-receipt-bundle/1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"confirmations": 0, "blockheight": None, "state": "mempool-or-unknown"}
        raise ChainReadError(f"WoC {exc.code} status for {txid}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise ChainReadError(f"WoC status unreachable for {txid}: {exc}") from exc


# --------------------------------------------------------------------------------------------------
# Minimal raw-tx parsing (just enough to read outputs + input[0] prevout)
# --------------------------------------------------------------------------------------------------
class _Cursor:
    __slots__ = ("b", "i")

    def __init__(self, b: bytes):
        self.b, self.i = b, 0

    def take(self, n: int) -> bytes:
        if self.i + n > len(self.b):
            raise ChainReadError("raw tx truncated")
        chunk = self.b[self.i:self.i + n]
        self.i += n
        return chunk

    def u8(self) -> int:
        return self.take(1)[0]

    def le(self, n: int) -> int:
        return int.from_bytes(self.take(n), "little")

    def varint(self) -> int:
        first = self.u8()
        if first < 0xfd:
            return first
        if first == 0xfd:
            return self.le(2)
        if first == 0xfe:
            return self.le(4)
        return self.le(8)


def parse_tx(raw_hex: str) -> dict:
    """Parse a raw BSV tx into {inputs:[{prevTxid,vout}], outputs:[{satoshis,scriptHex}]}.

    Only the fields the verifier needs are returned; witness/segwit do not exist on BSV so the legacy
    layout is exact.
    """
    cur = _Cursor(bytes.fromhex(raw_hex))
    cur.le(4)  # version
    n_in = cur.varint()
    inputs = []
    for _ in range(n_in):
        prev = cur.take(32)[::-1].hex()   # internal little-endian → display txid
        vout = cur.le(4)
        slen = cur.varint()
        cur.take(slen)                    # scriptSig (ignored)
        cur.le(4)                         # sequence
        inputs.append({"prevTxid": prev, "vout": vout})
    n_out = cur.varint()
    outputs = []
    for _ in range(n_out):
        sats = cur.le(8)
        slen = cur.varint()
        script = cur.take(slen).hex()
        outputs.append({"satoshis": sats, "scriptHex": script})
    return {"inputs": inputs, "outputs": outputs}


def op_return_items(script_hex: str) -> list[str] | None:
    """If `script_hex` is an OP_RETURN (optionally OP_FALSE-prefixed), return its data pushes as hex.

    Returns None if the script is not an OP_RETURN. Handles direct pushes (<0x4c bytes) and OP_PUSHDATA1/2/4.
    """
    b = bytes.fromhex(script_hex)
    i = 0
    if i < len(b) and b[i] == _OP_FALSE:
        i += 1
    if i >= len(b) or b[i] != _OP_RETURN:
        return None
    i += 1
    items: list[str] = []
    while i < len(b):
        op = b[i]; i += 1
        if op < 0x4c:
            n = op
        elif op == 0x4c:
            n = b[i]; i += 1
        elif op == 0x4d:
            n = int.from_bytes(b[i:i + 2], "little"); i += 2
        elif op == 0x4e:
            n = int.from_bytes(b[i:i + 4], "little"); i += 4
        else:
            # An opcode (e.g. OP_0 push of empty) — record empty and continue.
            items.append("")
            continue
        items.append(b[i:i + n].hex())
        i += n
    return items


def find_op_return(outputs: list[dict]) -> tuple[int, list[str]] | None:
    """Return (vout, data_items) for the first OP_RETURN output, or None."""
    for vout, out in enumerate(outputs):
        items = op_return_items(out["scriptHex"])
        if items is not None:
            return vout, items
    return None


# --------------------------------------------------------------------------------------------------
# Anchor checks (bind an on-chain OP_RETURN to a receipt)
# --------------------------------------------------------------------------------------------------
def read_standalone_anchor(txid: str, network: str = "main") -> dict:
    """Fetch `txid` and parse the standalone third entry. Returns {found, tag, modelHash, receiptHash, vout}."""
    tx = parse_tx(fetch_raw_tx(txid, network))
    hit = find_op_return(tx["outputs"])
    if hit is None:
        return {"found": False, "reason": "no OP_RETURN output in tx"}
    vout, items = hit
    if len(items) != 3:
        return {"found": False, "reason": f"OP_RETURN has {len(items)} items, expected 3 (tag/model/receipt)",
                "vout": vout, "items": items}
    tag_hex, model_hash, receipt_hash = items
    try:
        tag = bytes.fromhex(tag_hex).decode("utf-8")
    except UnicodeDecodeError:
        tag = tag_hex
    return {"found": True, "vout": vout, "tag": tag,
            "modelHash": model_hash, "receiptHash": receipt_hash}


def read_stateful_anchor(txid: str, network: str = "main") -> dict:
    """Fetch a stateful action `txid` and return its single 32-byte OP_RETURN receipt hash."""
    tx = parse_tx(fetch_raw_tx(txid, network))
    hit = find_op_return(tx["outputs"])
    if hit is None:
        return {"found": False, "reason": "no OP_RETURN output in tx"}
    vout, items = hit
    data = [it for it in items if it]
    if len(data) != 1 or len(data[0]) != 64:
        return {"found": False, "reason": f"OP_RETURN items={items}, expected one 32-byte hash",
                "vout": vout, "items": items}
    return {"found": True, "vout": vout, "receiptHash": data[0], "inputs": tx["inputs"]}


def walk_identity_to_genesis(action_txid: str, genesis_txid: str, network: str = "main",
                             *, max_hops: int = 4096) -> dict:
    """Walk an AgentTea identity chain BACKWARD from `action_txid` to `genesis_txid` via input[0].

    Every AgentTea spend pins the identity at input 0 (checkInputZero) and recreates it at output 0, so the
    chain of input[0].prevTxid links each action back to the previous identity tx, terminating at the deploy
    (genesis) tx. Returns {ok, hops, path, reason}. ok=True means action_txid provably descends from genesis.
    """
    path = [action_txid]
    cur = action_txid
    for hop in range(max_hops):
        if cur == genesis_txid:
            return {"ok": True, "hops": hop, "path": path}
        tx = parse_tx(fetch_raw_tx(cur, network))
        if not tx["inputs"]:
            return {"ok": False, "hops": hop, "path": path, "reason": f"{cur} has no inputs"}
        cur = tx["inputs"][0]["prevTxid"]
        path.append(cur)
        if cur == genesis_txid:
            return {"ok": True, "hops": hop + 1, "path": path}
    return {"ok": False, "hops": max_hops, "path": path,
            "reason": f"did not reach genesis {genesis_txid} within {max_hops} hops"}
