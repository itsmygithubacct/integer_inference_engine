#!/usr/bin/env python3
"""Token-COUNT parity check: our gpt2-BPE encode vs Ollama's `prompt_eval_count` (raw mode) for the live model.

A strong signal the tokenization matches the reference WITHOUT needing a tokenize endpoint. Matches on plain
text; differences on indented code reveal the cohere2moe pre-tokenizer's whitespace handling differs from the
GPT-2 default regex (the known parity gap — round-trip is unaffected).

    PYTHONPATH=src .venv/bin/python tools/check_tok_parity.py
"""
import json
import sys
import urllib.request
from pathlib import Path

from nmc.tokenizer import Tokenizer

MODEL = "north-mini-code-1.0:latest"
tok = Tokenizer.from_dir(Path.home() / ".local/integer_inference_engine/north-mini-code/tokenizer")


def ollama_prompt_tokens(p: str) -> int:
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps({"model": MODEL, "prompt": p, "raw": True, "stream": False,
                         "options": {"num_predict": 1}}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600)).get("prompt_eval_count")


PROMPTS = [
    "Hello, world!",
    "The capital of France is",
    "def add(a, b):\n    return a + b\n",
    "for i in range(10):\n        print(i)",
    "A short sentence without code.",
]
n_match = 0
for p in PROMPTS:
    mine, oll = len(tok.encode(p)), ollama_prompt_tokens(p)
    ok = mine == oll; n_match += ok
    print(f"  mine={mine:3d}  ollama={oll}  {'OK ' if ok else 'DIFF'}  {p[:36]!r}")
print(f"\n{n_match}/{len(PROMPTS)} exact-count matches "
      f"(plain text matches; indented-code diffs ⇒ cohere2moe pre-tokenizer regex needed for full parity)")
sys.exit(0)
