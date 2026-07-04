"""Stage-6b gates for the gpt2-BPE tokenizer: lossless round-trip + structure. (Exact llama.cpp parity is a
separate check against Ollama — see tools/check_tok_parity.py.)"""
import json
from pathlib import Path

import pytest

from nmc.tokenizer import Tokenizer, _byte_to_unicode, _pretokenize

TOKDIR = Path.home() / ".local/integer_inference_engine/north-mini-code/tokenizer"
pytestmark = pytest.mark.skipif(not (TOKDIR / "vocab.json").exists(),
                                reason="tokenizer not extracted (run tools/extract_tokenizer.py under sudo)")


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.from_dir(TOKDIR)


def test_byte_unicode_reversible():
    m = _byte_to_unicode()
    assert len(m) == 256 and len(set(m.values())) == 256        # bijection over all bytes


# Golden pre-tokenizer pieces, validated byte-exact against llama.cpp's `LLAMA_VOCAB_PRE_TYPE_TINY_AYA`:
# nmc's two regexes are char-identical to llama-vocab.cpp's applied TINY_AYA regex_exprs, and `_pretokenize`
# reproduces the true `unicode_regex_split` cascade piece-for-piece (verified 2026-07-01). These vectors pin
# the distinctive TINY_AYA behaviour: thousands-grouping of digit runs and case-aware letter splitting.
_TINY_AYA_GOLDEN = {
    "The capital of France is": ["The", " capital", " of", " France", " is"],
    "def add(a, b):\n    return a + b\n":
        ["def", " add", "(a", ",", " b", "):\n", "   ", " return", " a", " +", " b", "\n"],
    "Numbers like 1,234,567 and 89000.":
        ["Numbers", " like", " ", "1", ",", "234", ",", "567", " and", " ", "89", "000", "."],
    "café déjà vu — naïve façade": ["café", " déjà", " vu", " —", " naïve", " façade"],
    "CamelCaseIDENTifier_v2": ["Camel", "Case", "IDENTifier", "_v", "2"],
}


def test_pretokenize_tiny_aya_golden():
    for text, pieces in _TINY_AYA_GOLDEN.items():
        assert _pretokenize(text) == pieces, f"TINY_AYA pre-token drift: {text!r}"
        assert "".join(_pretokenize(text)) == text                # contiguous partition (lossless coverage)


SAMPLES = [
    "Hello, world!",
    "def fib(n):\n    return n if n < 2 else fib(n-1)+fib(n-2)",
    "Numbers 1234567890 and symbols !@#$%^&*()",
    "Unicode: café — naïve — 日本語 — 🚀",
    "    leading + \ttabs +  double  spaces",
    "",
]


@pytest.mark.parametrize("s", SAMPLES)
def test_round_trip_lossless(tok, s):
    assert tok.decode(tok.encode(s)) == s


def test_bos_and_validity(tok):
    ids = tok.encode("hello there")
    assert ids[0] == tok.bos_id                                  # add_bos
    assert all(0 <= i < len(tok.id_to_token) for i in ids)
    assert tok.encode("") == [tok.bos_id]


def test_no_bos_option(tok):
    assert tok.encode("hello", add_bos=False)[0] != tok.bos_id or tok.bos_id is None


def test_deterministic(tok):
    assert tok.encode("the quick brown fox") == tok.encode("the quick brown fox")
