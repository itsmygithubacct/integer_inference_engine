"""Single source of truth for the notary's out-of-repo state directory.

Every generated artifact — receipt ledgers, broadcast/transaction logs, packaged
receipt bundles, JSON session logs, and perf-debug traces — is written under
``$BONSAI_NOTARY_HOME`` (default ``~/.local/trinote``), NEVER
inside the checked-out repository. Keeping generated state out of the tree means an
``scp``/``rsync`` of the repo stays "clean" (akin to a fresh clone) and day-to-day runs
never pollute the working copy. Secrets (signing keys, wallet mnemonic, chain keys) live
here too — see ``SECURITY.md`` and ``INSTALL.md`` (§6).

Resolution is intentionally lazy (a function call per lookup) so a process that sets
``$BONSAI_NOTARY_HOME`` after import — or a test that monkeypatches it — still gets
the right path. Writers create the directories on demand; nothing is created here.
"""
from __future__ import annotations

import os
from pathlib import Path

# Default state home: ~/.local/trinote (XDG-style, under ~/.local; the shared home for the
# composed notary — engine + chain_c + bsv_third_entry). Override with $BONSAI_NOTARY_HOME.
_DEFAULT_HOME_RELPATH = (".local", "trinote")


def notary_home() -> Path:
    """Resolve ``$BONSAI_NOTARY_HOME`` (default ``~/.local/trinote``)."""
    env = os.environ.get("BONSAI_NOTARY_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home().joinpath(*_DEFAULT_HOME_RELPATH)


def config_path() -> Path:
    """Engine settings file: ``$BONSAI_CONFIG`` if set, else ``$BONSAI_NOTARY_HOME/bonsai.toml``."""
    env = os.environ.get("BONSAI_CONFIG")
    return Path(env).expanduser() if env else notary_home() / "bonsai.toml"


def receipts_dir() -> Path:
    """Directory holding the local receipt ledger and broadcast/transaction logs."""
    return notary_home() / "receipts"


def ledger_default() -> str:
    """Default local hash-linked receipt ledger path."""
    return str(receipts_dir() / "notarized.ledger.jsonl")


def broadcast_log_default() -> str:
    """Default dry-run broadcast log (the off-chain third-entry log)."""
    return str(receipts_dir() / "broadcast.log")


def tx_log_default() -> str:
    """Default off-chain transaction log (raw tx of every on-chain third entry)."""
    return str(receipts_dir() / "transactions.log")


def bundles_dir() -> Path:
    """Directory for auto-packaged portable receipt bundles."""
    return notary_home() / "bundles"


def keys_dir() -> Path:
    """Directory for the receipt signing keys (secp256k1 PRIVATE keys; written chmod 0600). A SECRET location,
    like the wallet — lives under the notary home, never in the repo."""
    return notary_home() / "keys"


def model_key_default() -> str:
    """Default path of the model (issuer) receipt signing key — generated on first use if absent."""
    return str(keys_dir() / "model.key.json")


def counterparty_key_default() -> str:
    """Default path of the counterparty receipt signing key — generated on first use if absent."""
    return str(keys_dir() / "counterparty.key.json")


def sessions_log_default() -> str:
    """Default JSON-mode reproduction/session log."""
    return str(notary_home() / "sessions" / "bonsai-json.jsonl")


def debug_dir_default() -> str:
    """Default perf-debug trace directory (honors ``$BONSAI_DEBUG_DIR`` first)."""
    return os.environ.get("BONSAI_DEBUG_DIR", str(notary_home() / "debug"))


# --- Non-receipt data homes (weights, benchmark outputs, built kernels, vendored llama.cpp) ----------------
#
# Historically these lived INSIDE the checked-out repo (``<repo>/models``, ``<repo>/benchmarks/results``,
# ``<repo>/tools/*.so``, and an author-specific ``~/research/refs/...`` for llama.cpp). They now default under
# ``$BONSAI_NOTARY_HOME`` too, so the repo tree holds only source. Each has a dedicated env override (mirroring
# ``$BONSAI_DEBUG_DIR``), and the ``default_*`` resolvers PREFER the notary location but fall back to the
# legacy in-repo / dev path when that is where the data still is — so existing installs keep working until the
# data is migrated (see ``scripts/migrate_to_notary_home.sh``).

_REPO_ROOT = Path(__file__).resolve().parents[2]          # <repo> = src/trinote/notary_paths.py -> parents[2]
_GGUF_NAME = "Bonsai-8B-Q1_0.gguf"
_ARTIFACT_NAME = "atlas-notarized-bonsai-8b.safetensors"
_IDENTITY_NAME = "atlas-notarized-bonsai-8b.identity.json"
_KERNEL_SO = "libbonsai_q1_kernel.so"
_GPU_SO = "libbonsai_q1_gpu.so"


def _env_dir(var: str, default: Path) -> Path:
    v = os.environ.get(var)
    return Path(v).expanduser() if v else default


def _resolve(primary: Path, *fallbacks: Path) -> Path:
    """Return ``primary`` unless it is missing and a fallback exists (then the first existing fallback)."""
    if primary.exists():
        return primary
    for fb in fallbacks:
        if fb.exists():
            return fb
    return primary


# Downloaded / imported model weights ($BONSAI_MODELS_DIR -> ~/.local/trinote/models)
def models_dir() -> Path:
    return _env_dir("BONSAI_MODELS_DIR", notary_home() / "models")


def default_gguf() -> str:
    """Source GGUF weights; prefers the notary models dir, falls back to legacy ``<repo>/models``."""
    return str(_resolve(models_dir() / _GGUF_NAME, _REPO_ROOT / "models" / _GGUF_NAME))


def default_artifact() -> str:
    """Imported safetensors artifact; prefers the notary models dir, falls back to ``<repo>/artifacts/model``."""
    return str(_resolve(models_dir() / _ARTIFACT_NAME, _REPO_ROOT / "artifacts" / "model" / _ARTIFACT_NAME))


def default_identity() -> str:
    """Minted identity record. This is a TRACKED source artifact, so it stays in ``<repo>/artifacts`` with the
    notary models dir as a secondary location."""
    return str(_resolve(_REPO_ROOT / "artifacts" / _IDENTITY_NAME, models_dir() / _IDENTITY_NAME))


# Generated benchmark outputs ($BONSAI_BENCHMARKS_DIR -> ~/.local/trinote/benchmarks)
def benchmarks_dir() -> Path:
    return _env_dir("BONSAI_BENCHMARKS_DIR", notary_home() / "benchmarks")


def results_dir() -> Path:
    """Default benchmark results root (was ``<repo>/benchmarks/results``)."""
    return benchmarks_dir() / "results"


# Built native binaries ($BONSAI_BIN_DIR -> ~/.local/trinote/bin) — build artifacts are not source.
def bin_dir() -> Path:
    return _env_dir("BONSAI_BIN_DIR", notary_home() / "bin")


def kernel_so(name: str = _KERNEL_SO) -> str:
    """Resolve a built kernel ``.so`` by filename; prefers the notary bin dir, falls back to ``<repo>/tools``."""
    return str(_resolve(bin_dir() / name, _REPO_ROOT / "tools" / name))


def gpu_kernel_so() -> str:
    return kernel_so(_GPU_SO)


# Vendored PrismML llama.cpp providing ``llama-tokenize`` ($BONSAI_LLAMA_DIR -> ~/.local/trinote/vendor/llama.cpp)
def llama_dir() -> Path:
    return _env_dir("BONSAI_LLAMA_DIR", notary_home() / "vendor" / "llama.cpp")


def default_bin_dir() -> str:
    """``llama-tokenize`` build/bin dir; prefers the notary vendor tree, falls back to the legacy author path
    ``~/research/refs/PrismML-llama.cpp/build/bin`` if that is where it was built."""
    return str(_resolve(llama_dir() / "build" / "bin",
                        Path.home() / "research" / "refs" / "PrismML-llama.cpp" / "build" / "bin"))
