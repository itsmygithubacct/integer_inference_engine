"""Broadcast backends for a receipt's chain artifact — the 3rd-entry *publish* step.

NOTE (standalone repo): the on-chain `chain/` TS toolchain that `LocalNodeChainBackend` /
`LocalNodeTeaBackend` drive IS bundled here (see docs/receipts/RECEIPTS.md 'Scope'), but it is default-OFF.
Those backends only engage the vendored `chain/` scripts when `enable_chain=True` (dry-run unless the
second key `CONFIRM_MAINNET_BROADCAST=yes` is set); the default `LogBroadcastBackend` (network-free dry-run
log) is the publish path that functions with zero network.

All local to this project. Backends, picked by `emit_receipt` (receipts/emit.py):

  * `LogBroadcastBackend` — the DEFAULT testing path (`broadcast_to_log=True`). Exercises the whole
    build→emit→broadcast flow with ZERO network: appends the would-be-broadcast chain artifact to a
    local JSONL "broadcast log" and returns a synthetic `log:<digest>` txid. This is how the broadcast
    mechanism is tested without touching BSV.
  * `LocalNodeChainBackend` — the standalone-OP_RETURN path (only when `enable_chain=True`). Runs the
    VENDORED TS broadcaster `chain/scripts/broadcastOpenLmReceipt.ts` locally via node, piping the
    canonical chain artifact to its stdin and parsing the returned `{txid,…}`. It builds the
    OP_RETURN third entry (`OP_FALSE OP_RETURN <tag> <modelHash> <receiptHash>`; see
    docs/receipts/RECEIPTS.md) funded by any key
    and broadcasts to WhatsOnChain — DRY-RUN unless `confirm=True` (→ the TS gate
    `CONFIRM_MAINNET_BROADCAST=yes`). An identity-less public timestamp.
  * `LocalNodeTeaBackend` — the STATEFUL RicardianTea path (only when `enable_chain=True`). Runs the
    vendored `chain/scripts/executeTeaReceipt.ts`, which threads the same artifact through
    `RicardianTea.executeTea` so the third entry is a state transition of a reputation-bearing identity
    (modelHash→invoiceHash, trinote receiptHash→provenanceHash; see `chain/src/openLmReceiptTea.ts`).
    This is the LOCAL DRY-RUN CORE: it runs the contract's in-script verification fully offline against
    an ephemeral identity (no deployment, no network). A real stateful spend — deployed identity UTXO,
    real keys, UTXO-tip management — is a deliberately deferred, separately-gated step
    (docs/receipts/RECEIPTS.md), so
    this backend refuses `confirm=True`.

WhatsOnChain is mainnet-only (real money), so a real `LocalNodeChainBackend` send needs BOTH
`enable_chain=True` AND `confirm=True` — a deliberate two-key interlock. Tests use the log backend or a
fake subprocess; they never shell out to node and never broadcast.
"""
from __future__ import annotations

try:
    import fcntl
except ImportError:  # pragma: no cover - this project targets Linux, but keep imports portable.
    fcntl = None
import json
import os
import shutil
import subprocess
from pathlib import Path

from .canonical import canonical_bytes, commit

_REPO_ROOT = Path(__file__).resolve().parents[3]


class ChainBroadcastError(RuntimeError):
    """A real broadcast attempt failed (node/TS/WoC error, or a malformed broadcaster response)."""


class LogBroadcastBackend:
    """Dry-run 'broadcast' to a local JSONL log — the safe, network-free testing path."""

    def __init__(self, log_path):
        self.log_path = Path(log_path)

    def broadcast(self, artifact: dict, *, ts: str | None = None) -> dict:
        # The dry-run txid is a PURE CONTENT COMMIT to the artifact ("log:" + commit(artifact)[:32]) with
        # NO nonce: two identical artifacts produce the same txid BY DESIGN (it is deterministic and
        # reproducible, not a unique broadcast id). Ordering / anti-replay is the ledger's job, not this
        # synthetic id's. (Do not add a nonce — the demo snapshot pins this exact scheme.)
        txid = "log:" + commit(artifact)[:32]           # synthetic, clearly-not-real, reproducible
        record = {"txid": txid, "network": "log", "broadcast": False, "ts": ts, "artifact": artifact}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                # Write the canonical encoding (consumers parse each line as JSON — verify_package.py —
                # never byte-compare the raw line, so this only tightens consistency with commit()).
                f.write(canonical_bytes(record).decode("utf-8") + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return {"txid": txid, "network": "log", "broadcast": False, "dryRun": True,
                "logPath": str(self.log_path)}


class WalletThirdEntryBackend:
    """Publish the chain artifact as a public OP_RETURN Third Entry via bonsai-notary's OWN BSV HD wallet
    (`wallet/notary_wallet.py`, bsv-sdk) — the self-contained path, NOT the vendored TS chain layer. This
    project's wallet builds + signs + broadcasts `OP_FALSE OP_RETURN <tag> <modelHash> <receiptHash>`, funded
    by a pre-split UTXO (`source_index` on the receive path), change to a DERIVED change address, fee at
    `sat_per_kb` (100 = 0.1 sat/byte). DRY-RUN unless `confirm=True` (the same two-key interlock:
    `enable_chain=True` in emit_receipt AND `confirm=True` here). It SUBPROCESSES the wallet's own uv venv so
    the numpy-only notary runtime never imports bsv-sdk. See docs/receipts/THIRD-ENTRY.md."""

    def __init__(self, *, source_index: int = 20, change_index: int = 2, sat_per_kb: int = 100,
                 confirm: bool = False, change_to_source: bool = False, allow_unconfirmed: bool = False,
                 wallet_python=None, wallet_script=None, timeout: int = 120):
        self.source_index = source_index
        self.change_index = change_index
        self.sat_per_kb = sat_per_kb
        self.confirm = confirm
        # For a live, multi-inference session: route change back to the source and spend mempool UTXOs so a
        # single funded receive UTXO self-rolls and emits receipts back-to-back without a confirmation wait.
        self.change_to_source = change_to_source
        self.allow_unconfirmed = allow_unconfirmed
        self.wallet_python = Path(wallet_python) if wallet_python else _REPO_ROOT / ".venv_wallet/bin/python"
        self.wallet_script = Path(wallet_script) if wallet_script else _REPO_ROOT / "wallet/notary_wallet.py"
        self.timeout = timeout

    def command(self, artifact: dict) -> list[str]:
        """The exact wallet CLI invocation (asserted in tests; no subprocess is spawned there)."""
        cmd = [str(self.wallet_python), str(self.wallet_script), "third-entry",
               "--source-index", str(self.source_index),
               "--model-hash", str(artifact["modelHash"]), "--receipt-hash", str(artifact["receiptHash"]),
               "--tag", str(artifact.get("tag", "trinote/r1")),
               "--change-index", str(self.change_index), "--sat-per-kb", str(self.sat_per_kb), "--json"]
        if self.change_to_source:
            cmd.append("--change-to-source")
        if self.allow_unconfirmed:
            cmd.append("--allow-unconfirmed")
        if self.confirm:
            cmd.append("--broadcast")
        return cmd

    def broadcast(self, artifact: dict, *, ts: str | None = None) -> dict:
        if not self.wallet_python.exists():
            raise ChainBroadcastError(
                f"wallet venv not found at {self.wallet_python} — create it with `uv venv .venv_wallet` + "
                f"`uv pip install -r requirements_wallet.txt`, then `notary_wallet.py gen-mnemonic`")
        try:
            proc = subprocess.run(self.command(artifact), capture_output=True, cwd=str(_REPO_ROOT),
                                  timeout=self.timeout)
        except (OSError, subprocess.SubprocessError) as e:
            raise ChainBroadcastError(f"wallet third-entry failed to launch: {e}") from e
        if proc.returncode != 0:
            raise ChainBroadcastError(
                f"wallet third-entry exited {proc.returncode}: {proc.stderr.decode('utf-8', 'replace')[:500]}")
        try:
            out = json.loads(proc.stdout.decode("utf-8").strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as e:
            raise ChainBroadcastError(f"wallet third-entry returned non-JSON: {proc.stdout[:300]!r}") from e
        if "txid" not in out:
            raise ChainBroadcastError(f"wallet third-entry response missing txid: {out}")
        out.setdefault("network", "main")
        out.setdefault("status", "broadcast" if out.get("broadcast") else "dry-run")
        return out


def _discover_node_bin() -> str | None:
    """Directory holding `node` — PATH first, then the nvm default (this box manages node via nvm)."""
    node = shutil.which("node")
    if node:
        return str(Path(node).parent)
    for cand in sorted((Path.home() / ".config/nvm/versions/node").glob("*/bin"), reverse=True):
        if (cand / "node").exists():
            return str(cand)
    return None


class _LocalNodeBackend:
    """Shared driver for the vendored `chain/` TS scripts: locate node, build the ts-node command,
    set the confirm/key env, pipe the canonical artifact in, parse the one-line JSON out. DRY-RUN unless
    `confirm=True`. Subclasses fix the `script` and a human `label` for error messages."""

    script = "scripts/broadcastOpenLmReceipt.ts"
    label = "broadcaster"

    def __init__(self, *, chain_dir=None, confirm: bool = False, key_file=None,
                 node_bin=None, timeout: int = 180):
        self.chain_dir = Path(chain_dir or (_REPO_ROOT / "chain"))
        self.confirm = confirm
        self.key_file = key_file
        self.node_bin = node_bin if node_bin is not None else _discover_node_bin()
        self.timeout = timeout

    def _env(self) -> dict:
        env = dict(os.environ)
        if self.node_bin:
            env["PATH"] = f"{self.node_bin}:{env.get('PATH', '')}"
        env["CONFIRM_MAINNET_BROADCAST"] = "yes" if self.confirm else "no"
        if self.key_file:
            env["KEY_FILE"] = str(self.key_file)
        return env

    def command(self) -> list[str]:
        """The exact local command (asserted in tests; no node is spawned there)."""
        ts_node = self.chain_dir / "node_modules/.bin/ts-node"
        return [str(ts_node), "--transpile-only", self.script]

    def broadcast(self, artifact: dict, *, ts: str | None = None) -> dict:
        if not (self.chain_dir / "node_modules").exists():
            raise ChainBroadcastError(
                f"chain deps missing at {self.chain_dir}/node_modules — the on-chain `chain/` toolchain is "
                f"bundled but its node deps are not installed (run `npm install` in {self.chain_dir}; see "
                f"docs/receipts/RECEIPTS.md 'Scope'), or leave enable_chain=False for the local log backend")
        try:
            proc = subprocess.run(self.command(), input=canonical_bytes(artifact),
                                  capture_output=True, cwd=str(self.chain_dir),
                                  env=self._env(), timeout=self.timeout)
        except (OSError, subprocess.SubprocessError) as e:
            raise ChainBroadcastError(f"local node {self.label} failed to launch: {e}") from e
        if proc.returncode != 0:
            raise ChainBroadcastError(
                f"{self.label} exited {proc.returncode}: {proc.stderr.decode('utf-8', 'replace')[:500]}")
        try:
            out = json.loads(proc.stdout.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise ChainBroadcastError(f"{self.label} returned non-JSON: {proc.stdout[:300]!r}") from e
        if "txid" not in out:
            raise ChainBroadcastError(f"{self.label} response missing txid: {out}")
        out.setdefault("status", "broadcast" if out.get("broadcast") else "dry-run")
        return out


class LocalNodeChainBackend(_LocalNodeBackend):
    """Drive the vendored standalone-OP_RETURN broadcaster. DRY-RUN unless confirm=True."""

    script = "scripts/broadcastOpenLmReceipt.ts"
    label = "broadcaster"

    def __init__(self, *, chain_dir=None, script: str = "scripts/broadcastOpenLmReceipt.ts",
                 confirm: bool = False, key_file=None, node_bin=None, timeout: int = 180):
        super().__init__(chain_dir=chain_dir, confirm=confirm, key_file=key_file,
                         node_bin=node_bin, timeout=timeout)
        self.script = script


class LocalNodeTeaBackend(_LocalNodeBackend):
    """Drive the vendored stateful `executeTea` receipt runner — the LOCAL DRY-RUN core.

    Threads the trinote receipt through `RicardianTea.executeTea` and runs the contract's in-script
    verification fully offline (ephemeral identity, DummyProvider). The TS script REFUSES
    `CONFIRM_MAINNET_BROADCAST=yes`, and this backend refuses `confirm=True` at construction — a real
    stateful spend is a separately-gated, deferred step (docs/receipts/RECEIPTS.md). Returns the runner's
    `{txid, contractReceiptHash, mapping, identity, status:"dry-run", …}`.
    """

    script = "scripts/executeTeaReceipt.ts"
    label = "executeTea runner"

    def __init__(self, *, chain_dir=None, node_bin=None, timeout: int = 240, confirm: bool = False):
        if confirm:
            raise ValueError(
                "LocalNodeTeaBackend is the local dry-run core and has no mainnet leg. A real stateful "
                "executeTea spend (deployed identity, real keys, UTXO-tip state) is a separately-gated, "
                "deferred step — see docs/receipts/RECEIPTS.md. Refusing confirm=True.")
        super().__init__(chain_dir=chain_dir, confirm=False, key_file=None,
                         node_bin=node_bin, timeout=timeout)
