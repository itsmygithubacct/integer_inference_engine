"""GGUF tokenizer glue for the Atlas-v2 quality gate — encode via the flagship's own tokenizer.

The flagship GGUF embeds a GPT-2-style byte-level BPE (128,256 tokens). To compare the int-ref@v2
imported model against bitnet.cpp fairly, BOTH must see the SAME token ids. We get exact flagship
tokenization by shelling out to the built `llama-tokenize` (zero BPE reimplementation, guaranteed to
match bitnet.cpp), and implement only the trivial byte-level *decode* (id → token string → bytes) here
to render int-ref output back to text. Used only by the quality gate, never on the receipt path.
"""
from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from pathlib import Path

from .import_gguf_v2 import _GGUFReader
from ..notary_paths import default_bin_dir

# Fallback only — callers (run_bonsai_cli, the bench tools) pass an explicit --bin-dir resolved via
# notary_paths. Defaults under $BONSAI_NOTARY_HOME with back-compat to the legacy dev path.
_DEFAULT_BIN_DIR = Path(default_bin_dir())


@lru_cache(maxsize=1)
def _byte_decoder() -> dict:
    """Inverse of GPT-2 bytes→unicode: maps each visible unicode char back to its raw byte."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def load_gguf_tokens(gguf_path: str | Path) -> list[str]:
    return list(_GGUFReader(gguf_path).kv["tokenizer.ggml.tokens"])


def decode(ids, tokens: list[str], *, skip_special_from: int = 128000) -> str:
    """Decode token ids → text (byte-level). Special/control tokens (id >= skip_special_from) are dropped."""
    return b"".join(token_bytes(i, tokens, skip_special_from=skip_special_from) for i in ids).decode(
        "utf-8", errors="replace"
    )


def token_bytes(token_id: int, tokens: list[str], *, skip_special_from: int = 128000) -> bytes:
    """Decode one token id to raw bytes; special/control tokens return empty bytes."""
    i = int(token_id)
    if not 0 <= i < skip_special_from:
        return b""
    bd = _byte_decoder()
    return bytes(bd.get(c, 0) for c in tokens[i])


def llama_tokenize(prompt: str, gguf_path: str | Path, *, bin_dir: str | Path = _DEFAULT_BIN_DIR) -> list[int]:
    """Exact flagship tokenization (incl. BOS) via the built `llama-tokenize`. Returns the id list."""
    exe = Path(bin_dir) / "llama-tokenize"
    try:
        proc = subprocess.run([str(exe), "-m", str(gguf_path), "-p", prompt],
                              capture_output=True, text=True, timeout=300)
    except FileNotFoundError as e:
        raise RuntimeError(f"llama-tokenize not found at {exe}") from e
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"llama-tokenize failed with exit {proc.returncode}: {msg[:500]}")
    ids = []
    for line in proc.stdout.splitlines():
        m = re.match(r"\s*(\d+)\s*->", line)
        if m:
            ids.append(int(m.group(1)))
    if not ids:
        msg = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"llama-tokenize produced no token ids: {msg[:500]}")
    return ids


def llama_generate(prompt: str, gguf_path: str | Path, n_new: int, *,
                   bin_dir: str | Path = _DEFAULT_BIN_DIR, threads: int = 4) -> str:
    """Greedy (temp 0) continuation TEXT from bitnet.cpp `llama-cli` — the reference to compare against."""
    exe = Path(bin_dir) / "llama-cli"
    try:
        r = subprocess.run([str(exe), "-m", str(gguf_path), "-p", prompt,
                            "-n", str(n_new), "-t", str(threads), "--temp", "0"],
                           capture_output=True, text=True, timeout=600)
    except FileNotFoundError as e:
        raise RuntimeError(f"llama-cli not found at {exe}") from e
    if r.returncode != 0:
        msg = (r.stderr or r.stdout).strip()
        raise RuntimeError(f"llama-cli failed with exit {r.returncode}: {msg[:500]}")
    # llama-cli prints prompt+continuation to stdout (logs go to stderr); collapse to a single line.
    return " ".join(r.stdout.split())


def llama_complete(prompt: str, gguf_path: str | Path, n_new: int, *,
                   bin_dir: str | Path = _DEFAULT_BIN_DIR, threads: int = 4,
                   ctx_size: int = 2048) -> str:
    """Greedy raw continuation text via llama.cpp `llama-completion`.

    Newer PrismML llama.cpp keeps `llama-cli` in chat mode for chat-template models; `llama-completion`
    is the raw completion binary needed for fair teacher-forced gates.
    """
    exe = Path(bin_dir) / "llama-completion"
    try:
        r = subprocess.run([
            str(exe), "-m", str(gguf_path), "-p", prompt,
            "-n", str(n_new), "-t", str(threads), "-c", str(ctx_size),
            "-no-cnv", "--simple-io", "--no-display-prompt", "--temp", "0",
        ], capture_output=True, text=True, timeout=600)
    except FileNotFoundError as e:
        raise RuntimeError(f"llama-completion not found at {exe}") from e
    if r.returncode != 0:
        msg = (r.stderr or r.stdout).strip()
        raise RuntimeError(f"llama-completion failed with exit {r.returncode}: {msg[:500]}")
    return r.stdout
