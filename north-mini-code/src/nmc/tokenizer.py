"""gpt2 byte-level BPE tokenizer for north-mini-code (Stage 6b).

Loads the vocab + merges extracted from the GGUF (`tools/extract_tokenizer.py`) and implements the standard
GPT-2 byte-level BPE: bytes → a printable-unicode alphabet, GPT-2 regex pre-tokenization, greedy rank-ordered
merges, vocab lookup. Decode is the exact inverse. Lossless **round-trip** is guaranteed by construction.

PARITY: the pre-tokenizer is llama.cpp's `cohere2moe` → `LLAMA_VOCAB_PRE_TYPE_TINY_AYA` (below). Verified
three ways — (1) exact token-COUNT parity with the live Ollama model (**12/12**, `tools/check_tok_parity.py`);
(2) **byte-exact** against llama.cpp's own `TINY_AYA` `regex_exprs` in llama-vocab.cpp: `_DIGIT_RX`/`_MAIN_RX`
are char-identical to the two applied regexes, and `_pretokenize` reproduces the true `unicode_regex_split`
cascade piece-for-piece (regression: `tests/test_tokenizer.py::test_pretokenize_tiny_aya_golden`); and (3)
**exact token-ID parity (5/5, with and without BOS)** against llama.cpp's own `llama-tokenize` (vocab_only),
using a 3-line patch that registers the `cohere2moe` arch so the reference build can load the GGUF — see
`docs/TOKENIZER-EXACT-ID-VERIFICATION.md` and `tools/llama-cpp-cohere2moe-vocab.patch`. (`prompt_eval_count`
alone is a weaker oracle than exact IDs, hence (2)/(3).) Round-trip is lossless by construction.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import regex  # \p{Lu} etc. need the third-party `regex` module (stdlib `re` lacks unicode property classes)

# cohere2moe pre-tokenizer = llama.cpp LLAMA_VOCAB_PRE_TYPE_TINY_AYA (verified against llama-vocab.cpp). Two
# regexes applied in order: (1) thousands-grouping of digit runs; (2) the GPT-4o-style main pattern
# (case-aware letters, contractions, \p{N}{1,3}, punctuation+trailing newlines, newline runs, whitespace).
_DIGIT_RX = regex.compile(r"\d{1,3}(?=(?:\d{3})*\b)")
_MAIN_RX = regex.compile(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _pretokenize(text: str) -> list[str]:
    """Apply the TINY_AYA pre-tokenizer: the digit-grouping regex first (its matches are tokens), with the
    gaps between digit runs split by the GPT-4o-style main regex (llama.cpp's ordered regex_exprs semantics)."""
    out: list[str] = []
    last = 0
    for m in _DIGIT_RX.finditer(text):
        if m.start() > last:
            out += _MAIN_RX.findall(text[last:m.start()])
        out.append(m.group())
        last = m.end()
    if last < len(text):
        out += _MAIN_RX.findall(text[last:])
    return out


@lru_cache(maxsize=1)
def _byte_to_unicode() -> dict[int, str]:
    """The reversible GPT-2 byte↔unicode map: every byte 0..255 ↦ a distinct printable unicode char."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class Tokenizer:
    def __init__(self, vocab: list[str], merges: list[str], meta: dict):
        self.id_to_token = vocab
        self.token_to_id = {t: i for i, t in enumerate(vocab)}
        self.bpe_ranks = {tuple(m.split(" ")): i for i, m in enumerate(merges) if " " in m}
        self.meta = meta
        self.bos_id = meta.get("bos_token_id")
        self.eos_id = meta.get("eos_token_id")
        self.add_bos = bool(meta.get("add_bos_token"))
        self.b2u = _byte_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self._cache: dict[str, tuple[str, ...]] = {}

    # ---- construction --------------------------------------------------------------------------------
    @classmethod
    def from_dir(cls, d: str | Path) -> "Tokenizer":
        d = Path(d)
        return cls(json.loads((d / "vocab.json").read_text()),
                   json.loads((d / "merges.json").read_text()),
                   json.loads((d / "meta.json").read_text()))

    # ---- BPE -----------------------------------------------------------------------------------------
    def _bpe(self, token: str) -> tuple[str, ...]:
        if token in self._cache:
            return self._cache[token]
        word = tuple(token)
        pairs = _get_pairs(word)
        while pairs:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new, i = [], 0
            while i < len(word):
                j = word.index(first, i) if first in word[i:] else len(word)
                new.extend(word[i:j])
                if j < len(word) and word[j] == first and j + 1 < len(word) and word[j + 1] == second:
                    new.append(first + second); i = j + 2
                else:
                    if j == len(word):
                        break
                    new.append(word[j]); i = j + 1
            word = tuple(new)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        self._cache[token] = word
        return word

    # ---- public API ----------------------------------------------------------------------------------
    def encode(self, text: str, *, add_bos: bool | None = None) -> list[int]:
        out: list[int] = []
        for piece in _pretokenize(text):
            enc = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            for sub in self._bpe(enc):
                out.append(self.token_to_id[sub])           # KeyError ⇒ vocab/merge mismatch (fail loud)
        if add_bos if add_bos is not None else self.add_bos:
            if self.bos_id is not None:
                out.insert(0, int(self.bos_id))
        return out

    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        special = {self.bos_id, self.eos_id, self.meta.get("pad_token_id"), self.meta.get("unk_token_id")}
        chars = "".join(self.id_to_token[i] for i in ids if not (skip_special and i in special))
        return bytes(self.u2b[c] for c in chars).decode("utf-8", errors="replace")
