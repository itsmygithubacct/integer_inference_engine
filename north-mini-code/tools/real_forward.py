#!/usr/bin/env python3
"""Stage-6b: a REAL-weight forward of north-mini-code and an architecture check vs the live Ollama model.

Runs the full 49-layer cohere2moe graph (the Stage 3-5 float reference blocks) on the actual GGUF weights —
loaded + dequantized per layer to bound RAM — for a prompt, and compares the greedy next token to Ollama.
This validates that our architecture implementation matches the real model (the integer determinism is proven
separately, per-component). FLOAT here = correctness check; an integer forward at 30B needs kernels (perf stage).

    sudo env PYTHONPATH=src .venv/bin/python tools/real_forward.py <blob> "prompt"
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from nmc.gguf import GGUF
from nmc.tokenizer import Tokenizer
from nmc import cohere2 as c2

blob, prompt = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "The capital of France is")
TOK = os.environ.get("NMC_TOKENIZER",
                     str(Path.home() / ".local/integer_inference_engine/north-mini-code/tokenizer"))

g = GGUF(blob)
kv = g.kv
A = "cohere2moe"
cfg = c2.Cfg(
    d_model=kv[f"{A}.embedding_length"], n_heads=kv[f"{A}.attention.head_count"],
    n_kv=kv[f"{A}.attention.head_count_kv"], head_dim=kv[f"{A}.attention.key_length"],
    ffn=kv[f"{A}.feed_forward_length"], vocab=kv[f"{A}.vocab_size"],
    sliding_window=kv[f"{A}.attention.sliding_window"], n_experts=kv[f"{A}.expert_count"],
    n_used=kv[f"{A}.expert_used_count"], expert_ffn=kv[f"{A}.expert_feed_forward_length"],
    rope_base=float(kv[f"{A}.rope.freq_base"]),
)
n_layers = kv[f"{A}.block_count"]; dense = kv[f"{A}.leading_dense_block_count"]
print(f"cfg: {cfg.d_model}d {n_layers}L {cfg.n_heads}/{cfg.n_kv}h {cfg.n_experts}e top{cfg.n_used} "
      f"vocab={cfg.vocab} dense_blocks={dense}", flush=True)

tok = Tokenizer.from_dir(TOK)
ids = tok.encode(prompt)
print(f"prompt={prompt!r}  tokens={len(ids)}  ids[:8]={ids[:8]}", flush=True)

t0 = time.time()
embed = g.weight("token_embd.weight", None)            # [vocab, d] float32 (~2GB) — input embeds + tied head
print(f"loaded embeddings {embed.shape} in {time.time()-t0:.1f}s", flush=True)
x = embed[np.asarray(ids)].astype(np.float64)          # [T, d]

for li in range(n_layers):
    t1 = time.time()
    p = f"blk.{li}."
    W = {"attn_norm": g.weight(p + "attn_norm.weight"),
         "wq": g.weight(p + "attn_q.weight"), "wk": g.weight(p + "attn_k.weight"),
         "wv": g.weight(p + "attn_v.weight"), "wo": g.weight(p + "attn_output.weight")}
    window = c2.window_for_layer(cfg, li)
    if li < dense:
        W |= {"gate": g.weight(p + "ffn_gate.weight"), "up": g.weight(p + "ffn_up.weight"),
              "down": g.weight(p + "ffn_down.weight")}
        x = c2.dense_block_float(x, W, cfg, window=window)
    else:
        We = {"router": g.weight(p + "ffn_gate_inp.weight"), "gate": g.weight(p + "ffn_gate_exps.weight"),
              "up": g.weight(p + "ffn_up_exps.weight"), "down": g.weight(p + "ffn_down_exps.weight")}
        x = c2.moe_block_float(x, W, We, cfg, window=window)
        del We
    del W
    print(f"  layer {li:2d} ({'dense' if li < dense else 'moe'}, {'full' if window is None else 'swa'}) "
          f"{time.time()-t1:.1f}s", flush=True)

logits = c2.tied_head_float(x, {"output_norm": g.weight("output_norm.weight"), "token_embd": embed})
nxt = int(logits[-1].argmax())
print(f"\nforward {time.time()-t0:.1f}s total", flush=True)
print(f"OUR next token: id={nxt}  text={tok.decode([nxt])!r}")

# Ollama greedy first token for the same raw prompt.
req = urllib.request.Request("http://127.0.0.1:11434/api/generate",
    data=json.dumps({"model": "north-mini-code-1.0:latest", "prompt": prompt, "raw": True, "stream": False,
                     "options": {"temperature": 0, "top_k": 1, "num_predict": 1}}).encode(),
    headers={"Content-Type": "application/json"})
oll = json.load(urllib.request.urlopen(req, timeout=600)).get("response", "")
print(f"OLLAMA next token: text={oll!r}")
print("MATCH" if tok.decode([nxt]) == oll else "DIFF")
