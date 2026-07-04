#!/usr/bin/env python3
"""Stage-7: a FAST, all-INTEGER real-weight forward using the qk_linear kernel — and check vs Ollama.

Same cohere2moe graph as tools/real_forward.py but fully integer fixed-point (activations fa=16, weights fw=24)
with the heavy matmuls done by the C kernel (fused Q4_K/Q6_K dequant + __int128 matmul, byte-exact to the numpy
oracle). RMSNorm/RoPE/softmax/SiLU reuse the Bonsai integer primitives. Demonstrates the kernel makes a 30B
integer forward tractable, and reports whether the integer engine's greedy token matches Ollama.

    sudo env PYTHONPATH=src .venv/bin/python tools/real_forward_int.py <blob> "prompt"
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from nmc.gguf import GGUF, Q4_K, Q6_K
from nmc.tokenizer import Tokenizer
from nmc import cohere2 as c2
from nmc import qk_native, qk_cuda
from nmc import qk_codec as qkc

# Backend for the heavy Q4_K/Q6_K linears: NMC_BACKEND=cuda|cuda-resident|cpu (default: cuda if usable).
# All are byte-identical to the integer oracle, so the forward output is the same regardless of backend.
# cuda-resident uploads each weight to VRAM once (register API) and reuses it — needs a ≥24GB GPU.
_want = os.environ.get("NMC_BACKEND", "cuda")
_RESIDENT = False
if _want.startswith("cuda") and qk_cuda.available():
    kn = qk_cuda
    _RESIDENT = (_want == "cuda-resident") and qk_cuda.resident_available()
    _BNAME = "cuda-resident" if _RESIDENT else "cuda"
else:
    kn, _BNAME = qk_native, ("cpu" if _want != "cuda" else "cpu (cuda unavailable)")
_handles: dict = {}                          # resident: name/(name,e) -> (handle, out_f)

blob, prompt = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "The capital of France is")
TOK = os.environ.get("NMC_TOKENIZER",
                     str(Path.home() / ".local/integer_inference_engine/north-mini-code/tokenizer"))
FA, FW = 16, 24
_QT = {Q4_K: kn.Q4_K, Q6_K: kn.Q6_K}

g = GGUF(blob); kv = g.kv; A = "cohere2moe"
cfg = c2.Cfg(d_model=kv[f"{A}.embedding_length"], n_heads=kv[f"{A}.attention.head_count"],
             n_kv=kv[f"{A}.attention.head_count_kv"], head_dim=kv[f"{A}.attention.key_length"],
             ffn=kv[f"{A}.feed_forward_length"], vocab=kv[f"{A}.vocab_size"],
             sliding_window=kv[f"{A}.attention.sliding_window"], n_experts=kv[f"{A}.expert_count"],
             n_used=kv[f"{A}.expert_used_count"], expert_ffn=kv[f"{A}.expert_feed_forward_length"],
             rope_base=float(kv[f"{A}.rope.freq_base"]), fa=FA, fw=FW)
NL = kv[f"{A}.block_count"]; DENSE = kv[f"{A}.leading_dense_block_count"]
tok = Tokenizer.from_dir(TOK); ids = tok.encode(prompt)
print(f"prompt={prompt!r} tokens={len(ids)}  backend={_BNAME}  cfg {cfg.d_model}d {NL}L "
      f"{cfg.n_experts}e/top{cfg.n_used}", flush=True)


assert kn.available(), "qk kernel .so not found — build it (tools/build_nmc_kernel.sh) and ensure it is on the path"


def klin(name, x):                       # kernel linear from a full tensor name; x [T,in] int64 -> [T,out]
    t = g.tensors[name]; ne0, ne1, qt = t["shape"][0], t["shape"][1], _QT[t["type"]]
    if _RESIDENT:
        if name not in _handles:
            h = qk_cuda.register_weight(g.read_raw(name), ne1, ne0 // 256, qt); assert h is not None, name
            _handles[name] = (h, ne1)
        h, of = _handles[name]; return qk_cuda.apply_resident(h, x, of, FW)
    return kn.qk_linear(g.read_raw(name), x, ne1, ne0 // 256, FW, qt)


def klin_expert(name, e, x):
    raw, ne0, ne1, tt = g.expert_raw(name, e); qt = _QT[tt]
    if _RESIDENT:
        key = (name, e)
        if key not in _handles:
            h = qk_cuda.register_weight(raw, ne1, ne0 // 256, qt); assert h is not None
            _handles[key] = (h, ne1)
        h, of = _handles[key]; return qk_cuda.apply_resident(h, x, of, FW)
    return kn.qk_linear(raw, x, ne1, ne0 // 256, FW, qt)


def norm(x, gname):                      # integer RMSNorm with F32 gain -> fixed-point gain at fa
    return c2.fixed_point_rmsnorm(x, FA, cfg.eps, gain_q=c2.to_fixed(g.weight(gname, None), FA))


t0 = time.time()
# embeddings of the prompt tokens, dequantized to fixed-point at fa (each is 8 Q6_K rows of token_embd)
x = np.stack([qkc.dequant_q6k_tensor(g.raw_rows("token_embd.weight", i, 1), cfg.d_model, FA) for i in ids])
cos, sin = c2.build_rope_tables(len(ids), cfg.head_dim, base=int(cfg.rope_base), frac_bits=FA)

for li in range(NL):
    p = f"blk.{li}."; window = c2.window_for_layer(cfg, li)
    h = norm(x, p + "attn_norm.weight")
    q = klin(p + "attn_q.weight", h); k = klin(p + "attn_k.weight", h); v = klin(p + "attn_v.weight", h)
    attn = klin(p + "attn_output.weight",
                c2.attention_int(q, k, v, cfg, cos, sin, window, rope=window is not None or li < DENSE))
    if li < DENSE:
        gg = c2.silu_int(klin(p + "ffn_gate.weight", h), FA); uu = klin(p + "ffn_up.weight", h)
        gu = ((gg.astype(object) * uu.astype(object)) >> FA).astype(np.int64)
        ffn = klin(p + "ffn_down.weight", gu)
    else:
        rl = c2.linear(h, c2.to_fixed(g.weight(p + "ffn_gate_inp.weight", None), FW), FW)   # F32 router (small)
        ffn = np.zeros((len(ids), cfg.d_model), dtype=object)
        for t in range(len(ids)):
            sel = c2._topk_lowidx(rl[t], cfg.n_used)
            gates = c2.fixed_point_sigmoid(rl[t][sel].astype(np.int64), FA)
            ht = h[t:t + 1]
            for j, e in enumerate(sel):
                gg = c2.silu_int(klin_expert(p + "ffn_gate_exps.weight", e, ht), FA)
                uu = klin_expert(p + "ffn_up_exps.weight", e, ht)
                gu = ((gg.astype(object) * uu.astype(object)) >> FA).astype(np.int64)
                eo = klin_expert(p + "ffn_down_exps.weight", e, gu)[0].astype(object)
                ffn[t] += (eo * int(gates[j])) >> FA
        ffn = ffn.astype(np.int64)
    x = x + attn + ffn
    if li % 8 == 0 or li == NL - 1:
        print(f"  layer {li:2d} ({'full' if window is None else 'swa'})  t={time.time()-t0:.1f}s", flush=True)

hn = norm(x, "output_norm.weight")
logits = klin("token_embd.weight", hn)             # tied head via the kernel (Q6_K)
nxt = int(logits[-1].argmax())
print(f"\nINTEGER forward {time.time()-t0:.1f}s   next id={nxt} text={tok.decode([nxt])!r}"
      + (f"  (resident: {qk_cuda.resident_count()} weights in VRAM)" if _RESIDENT else ""), flush=True)
if _RESIDENT:
    qk_cuda.free_all()                       # release VRAM before the ollama reference loads the model

req = urllib.request.Request("http://127.0.0.1:11434/api/generate",
    data=json.dumps({"model": "north-mini-code-1.0:latest", "prompt": prompt, "raw": True, "stream": False,
                     "options": {"temperature": 0, "top_k": 1, "num_predict": 1}}).encode(),
    headers={"Content-Type": "application/json"})
oll = json.load(urllib.request.urlopen(req, timeout=600)).get("response", "")
print(f"OLLAMA next text={oll!r}")
print("MATCH" if tok.decode([nxt]) == oll else "DIFF")
