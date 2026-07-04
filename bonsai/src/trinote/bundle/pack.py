"""Write a portable, content-addressed receipt bundle.

A bundle is a directory (optionally tarred to `.tar.gz`) whose `manifest.json` commits every other file by
sha256, and a `bundleHash` that commits the manifest itself. Tampering with any file changes its digest →
changes the bundleHash, so a single 32-byte value pins the whole package.

Layout::

    <bundle>/
      manifest.json          # {schema, kind, modelHash, receiptHash, files:{name:sha256}, bundleHash}
      receipt.json           # the on-chain-committable half (commitments + signatures + receiptHash)
      preimage.json          # the off-chain half (token ids + sampler + trace) — needed to re-verify
      chain-artifact.json     # {schema, tag, modelHash, receiptHash, samplerMode, seed}
      onchain.json           # where/how the third entry landed (standalone OP_RETURN or stateful action)
      ledger-head.json       # OPTIONAL local hash-linked ledger entry for this receipt
      identity.json          # STATEFUL ONLY — the AgentTea identity the action ran under

Files are written as canonical bytes (sorted keys, compact) so digests are reproducible by any party.
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

from ..receipts.canonical import canonical_bytes, commit
from ..receipts.emit import chain_artifact

BUNDLE_SCHEMA = "trinote.receipt-bundle/v1"


class BundleError(ValueError):
    """A bundle is malformed (missing required half, wrong kind, mismatched fields)."""


def _assert_ascii(value, field: str) -> None:
    """Fail closed if a free-text manifest field is non-ASCII. `bundleHash` commits the manifest via
    canonical_bytes(ensure_ascii=False); a non-ASCII byte would make the digest depend on the JSON
    implementation's escaping (a non-Python re-deriver that emits \\uXXXX would compute a different
    bundleHash and falsely report tampering). The committed on-chain receiptHash carries no free text, so
    this only guards the off-chain packaging digest — keep it byte-reproducible across languages."""
    if isinstance(value, str):
        try:
            value.encode("ascii")
        except UnicodeEncodeError as exc:
            raise BundleError(f"{field} must be ASCII for a cross-implementation-stable bundleHash: {exc}") from exc


def _write(path: Path, obj) -> str:
    """Write `obj` as canonical bytes and return its sha256 hex (the value recorded in the manifest)."""
    raw = canonical_bytes(obj)
    path.write_bytes(raw)
    return commit(obj)


def _write_text(path: Path, text: str) -> str:
    """Write raw UTF-8 text and return its sha256 hex (for human-readable bundle files like transcript.md)."""
    import hashlib
    raw = text.encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _render_transcript_md(t: dict) -> str:
    """Render a human-readable transcript. NOTE: the AUTHORITATIVE content is the committed token ids in
    preimage.json (inputCommit/outputCommit bind them into receiptHash); this plaintext is a convenience and
    is re-derivable from the token ids with the model's tokenizer."""
    return (
        f"# Bonsai notarized inference transcript\n\n"
        f"- model: {t.get('modelLabel', '')}\n"
        f"- modelHash: `{t.get('modelHash', '')}`\n"
        f"- receiptHash: `{t.get('receiptHash', '')}`\n"
        f"- sampler: {t.get('sampler', '')}\n"
        f"- tokens: {t.get('inputTokenCount', '?')} in / {t.get('outputTokenCount', '?')} out\n\n"
        f"## Prompt\n\n{t.get('prompt', '')}\n\n"
        f"## Output\n\n{t.get('output', '')}\n\n"
        f"> Plaintext is for human readability; the authoritative record is the committed token ids in "
        f"preimage.json (bound into receiptHash). Re-derive this text from those ids with the model tokenizer.\n"
    )


def pack_bundle(
    *,
    bundle: dict,
    onchain: dict | None = None,
    out_dir: str | Path,
    ledger_entry: dict | None = None,
    identity: dict | None = None,
    model_label: str = "",
    created: str | None = None,
    transcript: dict | None = None,
    as_tar: bool = False,
) -> dict:
    """Pack `bundle` (`{receipt, preimage}` from build_receipt) into a portable, content-addressed bundle.

    `onchain` describes the third entry: pass a dict with "kind" of "standalone" or "stateful" (for
    "stateful" an `identity` dict is required) to bundle an on-chain mark, OR pass `onchain=None` for a
    LOCAL bundle (kind="local") whose third entry is the local hash-linked ledger — no BSV, verifiable by
    offline commitments + bit-exact re-execution. `transcript` (optional {prompt, output, ...}) is written
    as a human-readable transcript.json + transcript.md (committed by bundleHash, but NOT part of receiptHash,
    which binds only token ids). Returns {path, bundleHash, manifest}. With `as_tar`, `out_dir` is the .tar.gz
    path to write and the staging dir is removed.
    """
    receipt = bundle.get("receipt")
    preimage = bundle.get("preimage")
    if not isinstance(receipt, dict) or not isinstance(preimage, dict):
        raise BundleError("bundle must contain 'receipt' and 'preimage' dicts (build_receipt output)")
    if onchain is None:
        kind = "local"
    else:
        kind = onchain.get("kind")
        if kind not in ("standalone", "stateful"):
            raise BundleError(f"onchain.kind must be 'standalone', 'stateful', or omitted (local), got {kind!r}")
        if kind == "stateful":
            if not isinstance(identity, dict):
                raise BundleError("stateful bundle requires an 'identity' dict (ricardianHash, genesisTxid, pubkeys)")
            # genesisTxid is load-bearing: verify._verify_onchain walks the identity UTXO chain
            # back to it to prove the action sits on THIS identity's official history. A stateful
            # bundle without it cannot be soundly verified, so refuse to build one (fail closed at
            # pack time rather than emit a bundle that silently skips the provenance walk).
            if not identity.get("genesisTxid"):
                raise BundleError("stateful bundle requires identity.genesisTxid (proves on-chain identity provenance)")

    _assert_ascii(model_label or preimage.get("modelLabel", ""), "modelLabel")
    _assert_ascii(created, "created")

    out_path = Path(out_dir)
    if not as_tar and out_path.exists() and not out_path.is_dir():
        raise BundleError(f"bundle out_dir exists and is not a directory: {out_path}")
    work = out_path.with_suffix(out_path.suffix + ".d") if as_tar else out_path
    work.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    files["receipt.json"] = _write(work / "receipt.json", receipt)
    files["preimage.json"] = _write(work / "preimage.json", preimage)
    files["chain-artifact.json"] = _write(work / "chain-artifact.json", chain_artifact(receipt))
    if onchain is not None:
        files["onchain.json"] = _write(work / "onchain.json", onchain)
    if ledger_entry is not None:
        files["ledger-head.json"] = _write(work / "ledger-head.json", ledger_entry)
    if identity is not None:
        files["identity.json"] = _write(work / "identity.json", identity)
    if transcript is not None:
        files["transcript.json"] = _write(work / "transcript.json", transcript)
        files["transcript.md"] = _write_text(work / "transcript.md", _render_transcript_md(transcript))

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "kind": kind,
        "created": created,
        "modelLabel": model_label or preimage.get("modelLabel", ""),
        "modelHash": receipt["modelHash"],
        "receiptHash": receipt["receiptHash"],
        "files": files,
    }
    manifest["bundleHash"] = commit(manifest)   # commits schema/kind/metadata + every file digest
    (work / "manifest.json").write_bytes(canonical_bytes(manifest))

    if as_tar:
        try:
            _tar_dir(work, out_path)
        except BaseException:
            out_path.unlink(missing_ok=True)   # drop any partial archive
            raise
        finally:
            # Always remove the staging dir, even if tarring failed or was interrupted.
            for child in sorted(work.iterdir()):
                child.unlink()
            try:
                work.rmdir()
            except OSError:
                pass
        final = out_path
    else:
        final = work
    return {"path": str(final), "bundleHash": manifest["bundleHash"], "manifest": manifest}


def _tar_dir(src: Path, dest: Path) -> None:
    """Tar the bundle files into `dest` (.tar.gz), each entry under the bundle's stem dir, sorted + reproducible."""
    stem = dest.name
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    with tarfile.open(dest, "w:gz") as tar:
        for child in sorted(src.iterdir()):
            data = child.read_bytes()
            info = tarfile.TarInfo(name=f"{stem}/{child.name}")
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
