#!/usr/bin/env python3
"""KV-cached multi-token GENERATION on the real model with the CUDA(-resident) kernel — and check vs Ollama.

Prefill the prompt (populating the per-layer KV cache), then decode token-by-token: each step embeds the last
token, runs the cache-aware block (attend over the cache within the per-layer SWA/full window), and samples
greedily. With NMC_BACKEND=cuda-resident the weights are uploaded to VRAM ONCE (register API) and reused across
every decoded token — the resident-decode win (the per-call backend re-uploads weights every token). All
integer, byte-identical to the numpy oracle. Reports decode tok/s and matches the greedy text to Ollama.

    sudo env PYTHONPATH=src NMC_BACKEND=cuda-resident .venv/bin/python tools/real_generate.py <blob> "prompt" [n_new]
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

_want = os.environ.get("NMC_BACKEND", "cuda")
_RESIDENT = False
if _want.startswith("cuda") and qk_cuda.available():
    kn = qk_cuda
    _RESIDENT = (_want == "cuda-resident") and qk_cuda.resident_available()
    _BNAME = "cuda-resident" if _RESIDENT else "cuda"
else:
    kn, _BNAME = qk_native, ("cpu" if _want != "cuda" else "cpu (cuda unavailable)")
_handles: dict = {}
_FUSED = _RESIDENT and qk_cuda.moe_ffn_available()    # one batched GPU call per MoE layer (lever 1+2)

blob = sys.argv[1]
prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
N_NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 16
TOK = os.environ.get("NMC_TOKENIZER", str(Path.home() / ".local/integer_inference_engine/north-mini-code/tokenizer"))
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
assert kn.available(), "qk kernel .so not found"
print(f"prompt={prompt!r} tokens={len(ids)} n_new={N_NEW}  backend={_BNAME}  {cfg.d_model}d {NL}L "
      f"{cfg.n_experts}e/top{cfg.n_used}", flush=True)


def klin(name, x):
    t = g.tensors[name]; ne0, ne1, qt = t["shape"][0], t["shape"][1], _QT[t["type"]]
    if _RESIDENT:
        if name not in _handles:
            _handles[name] = (qk_cuda.register_weight(g.read_raw(name), ne1, ne0 // 256, qt), ne1)
        h, of = _handles[name]; return qk_cuda.apply_resident(h, x, of, FW)
    return kn.qk_linear(g.read_raw(name), x, ne1, ne0 // 256, FW, qt)


def klin_expert(name, e, x):
    raw, ne0, ne1, tt = g.expert_raw(name, e); qt = _QT[tt]
    if _RESIDENT:
        key = (name, e)
        if key not in _handles:
            _handles[key] = (qk_cuda.register_weight(raw, ne1, ne0 // 256, qt), ne1)
        h, of = _handles[key]; return qk_cuda.apply_resident(h, x, of, FW)
    return kn.qk_linear(raw, x, ne1, ne0 // 256, FW, qt)


def expert_handle(name, e):                       # get-or-register one expert weight; return its resident handle
    key = (name, e)
    if key not in _handles:
        raw, ne0, ne1, tt = g.expert_raw(name, e)
        _handles[key] = (qk_cuda.register_weight(raw, ne1, ne0 // 256, _QT[tt]), ne1)
    return _handles[key][0]


def norm(x, gname):
    return c2.fixed_point_rmsnorm(x, FA, cfg.eps, gain_q=c2.to_fixed(g.weight(gname, None), FA))


_router_w: dict = {}                              # cache the F32 router weights (fixed-point) per layer
_router_checked = [False]


def router_logits(h, p):
    """MoE router logits, int64 (the F32 router has no kernel). d_model·2**40 < 2**63 so int64 == big-int;
    a one-shot self-check vs the object path guarantees byte-exactness at runtime."""
    if p not in _router_w:
        _router_w[p] = c2.to_fixed(g.weight(p + "ffn_gate_inp.weight", None), FW)
    W = _router_w[p]
    rl = ((np.asarray(h, np.int64) @ W.T) >> FW).astype(np.int64)
    if not _router_checked[0]:
        assert np.array_equal(rl, c2.linear(h, W, FW)), "router int64 != big-int (overflow)"
        _router_checked[0] = True
    return rl


def embed(token_ids):                        # dequant token_embd rows -> [m, d] fixed-point at fa
    return np.stack([qkc.dequant_q6k_tensor(g.raw_rows("token_embd.weight", int(t), 1), cfg.d_model, FA)
                     for t in token_ids])


def decode_block(x_new, li, cache, cos, sin):
    """Cache-aware parallel block for m new positions (prefill m=len(prompt) or decode m=1)."""
    p = f"blk.{li}."; window = c2.window_for_layer(cfg, li); start = cache.length(li); m = x_new.shape[0]
    h = norm(x_new, p + "attn_norm.weight")
    q = klin(p + "attn_q.weight", h).reshape(m, cfg.n_heads, cfg.head_dim)
    k = klin(p + "attn_k.weight", h).reshape(m, cfg.n_kv, cfg.head_dim)
    v = klin(p + "attn_v.weight", h).reshape(m, cfg.n_kv, cfg.head_dim)
    if window is not None or li < DENSE:          # cohere2 NoPE: RoPE only on SWA + dense-prefix layers
        q = c2._rope_int(q, cos, sin, FA, start); k = c2._rope_int(k, cos, sin, FA, start)
    cache.append(li, np.transpose(k, (1, 0, 2)), np.transpose(v, (1, 0, 2)))
    attn = klin(p + "attn_output.weight", c2.attention_cached(q, cache.k[li], cache.v[li], start, cfg, window))
    if li < DENSE:
        gg = c2.silu_int(klin(p + "ffn_gate.weight", h), FA); uu = klin(p + "ffn_up.weight", h)
        gu = ((gg.astype(object) * uu.astype(object)) >> FA).astype(np.int64); ffn = klin(p + "ffn_down.weight", gu)
    else:
        rl = router_logits(h, p)
        if _FUSED:                                # one batched on-GPU call per token: gate+silu+up+down+combine
            ffn = np.empty((m, cfg.d_model), dtype=np.int64)
            for t in range(m):
                sel = c2._topk_lowidx(rl[t], cfg.n_used)
                gates = c2.fixed_point_sigmoid(rl[t][sel].astype(np.int64), FA)
                gh = [expert_handle(p + "ffn_gate_exps.weight", e) for e in sel]
                uh = [expert_handle(p + "ffn_up_exps.weight", e) for e in sel]
                dh = [expert_handle(p + "ffn_down_exps.weight", e) for e in sel]
                ffn[t] = qk_cuda.moe_ffn(gh, uh, dh, h[t], gates, cfg.d_model, cfg.expert_ffn, FA, FW)
        else:
            ffn = np.zeros((m, cfg.d_model), dtype=object)
            for t in range(m):
                sel = c2._topk_lowidx(rl[t], cfg.n_used)
                gates = c2.fixed_point_sigmoid(rl[t][sel].astype(np.int64), FA)
                ht = h[t:t + 1]
                for jj, e in enumerate(sel):
                    gg = c2.silu_int(klin_expert(p + "ffn_gate_exps.weight", e, ht), FA)
                    uu = klin_expert(p + "ffn_up_exps.weight", e, ht)
                    gu = ((gg.astype(object) * uu.astype(object)) >> FA).astype(np.int64)
                    eo = klin_expert(p + "ffn_down_exps.weight", e, gu)[0].astype(object)
                    ffn[t] += (eo * int(gates[jj])) >> FA
            ffn = ffn.astype(np.int64)
    return np.asarray(x_new, np.int64) + attn + ffn


cos, sin = c2.build_rope_tables(len(ids) + N_NEW + 1, cfg.head_dim, base=int(cfg.rope_base), frac_bits=FA)
cache = c2.KVCache(NL)
t0 = time.time()
x = embed(ids)
for li in range(NL):
    x = decode_block(x, li, cache, cos, sin)
last = x[-1:]
t_prefill = time.time() - t0
print(f"prefill {t_prefill:.1f}s ({len(ids)} tok)", flush=True)

out = []
t1 = time.time()
for step in range(N_NEW):
    logits = klin("token_embd.weight", norm(last, "output_norm.weight"))
    nxt = int(logits[0].argmax())
    out.append(nxt)
    if step + 1 >= N_NEW:
        break
    last = embed([nxt])
    for li in range(NL):
        last = decode_block(last, li, cache, cos, sin)
t_decode = time.time() - t1
gen_text = tok.decode(out)
print(f"decode {t_decode:.1f}s for {N_NEW} tok = {N_NEW / t_decode:.2f} tok/s"
      + (f"  (resident: {qk_cuda.resident_count()} weights in VRAM)" if _RESIDENT else ""), flush=True)
print(f"OURS  : {gen_text!r}", flush=True)

# Optional: prove the KV-cached decode is byte-identical to re-prefilling from scratch each step ON THE REAL
# MODEL (separates a decode bug from fixed-point fidelity drift vs Ollama). Re-prefills with a fresh cache.
if os.environ.get("NMC_VERIFY"):
    ref, seq = [], list(ids)
    for _ in range(len(out)):
        fresh = c2.KVCache(NL); xx = embed(seq)
        for li in range(NL):
            xx = decode_block(xx, li, fresh, cos, sin)
        nx = int(klin("token_embd.weight", norm(xx[-1:], "output_norm.weight"))[0].argmax())
        ref.append(nx); seq.append(nx)
    print("VERIFY", "OK — cached decode == from-scratch prefill (byte-exact)" if ref == out
          else f"FAIL — cached {out} != reprefill {ref}", flush=True)
if _RESIDENT:
    qk_cuda.free_all()

if os.environ.get("NMC_VERIFY"):     # verify is self-contained; skip the ollama reference (avoids VRAM contention)
    sys.exit(0)

req = urllib.request.Request("http://127.0.0.1:11434/api/generate",
    data=json.dumps({"model": "north-mini-code-1.0:latest", "prompt": prompt, "raw": True, "stream": False,
                     "options": {"temperature": 0, "top_k": 1, "num_predict": N_NEW}}).encode(),
    headers={"Content-Type": "application/json"})
oll = json.load(urllib.request.urlopen(req, timeout=600)).get("response", "")
print(f"OLLAMA: {oll!r}")
# greedy both sides; compare on the common length (Ollama may stop at eos / tokenize the tail differently)
ok = gen_text.startswith(oll) or oll.startswith(gen_text) or gen_text == oll
print("MATCH" if ok else "DIFF")
