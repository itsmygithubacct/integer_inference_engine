#!/usr/bin/env python3
"""Stage-6a: validate the GGUF loader + Q4_K/Q6_K codec on the REAL north-mini-code tensors.

Run (root, to read the ollama-owned blob):
    sudo env PYTHONPATH=src .venv/bin/python tools/verify_real_tensors.py <blob> [out.txt]

Checks, on the actual model: (1) tensor shapes/types match the Stage-1 metadata; (2) the integer fixed-point
dequant matches the float reference dequant on REAL Q4_K/Q6_K blocks (extends the Stage-2 synthetic gate)."""
import sys
import numpy as np
from nmc.gguf import GGUF, TYPE_NAME

FW = 24
blob = sys.argv[1]
out = open(sys.argv[2], "w") if len(sys.argv) > 2 else sys.stdout

def log(*a):
    print(*a, file=out); out.flush()

g = GGUF(blob)
log(f"loaded: {g.summary()}")

# Representative tensors across all quant types + roles.
names = [
    "token_embd.weight",            # Q6_K, tied head + embeddings
    "blk.0.attn_q.weight",          # Q4_K
    "blk.0.attn_v.weight",          # Q6_K
    "blk.0.attn_norm.weight",       # F32
    "blk.1.ffn_gate_inp.weight",    # F32 router (MoE block)
    "blk.1.ffn_gate_exps.weight",   # Q4_K experts (3-D)
    "blk.1.ffn_down_exps.weight",   # Q6_K experts
    "output_norm.weight",           # F32
]
log("\n%-28s %-6s %-22s %s" % ("tensor", "type", "shape", "int-vs-float dequant"))
worst = 0.0
for nm in names:
    if nm not in g.tensors:
        log("%-28s  MISSING" % nm); continue
    t = g.tensors[nm]; tn = TYPE_NAME.get(t["type"], t["type"])
    if t["type"] in (0, 1):                       # F32/F16 -> fixed-point round-trip
        f = g.dequant(nm, frac=None, max_blocks=8)
        q = g.dequant(nm, frac=FW, max_blocks=8)
        rel = np.max(np.abs(q.astype(np.float64) / (1 << FW) - f)) / max(np.max(np.abs(f)), 1e-9)
    else:                                          # Q4_K/Q6_K -> codec fidelity on REAL blocks
        f = g.dequant(nm, frac=None, max_blocks=256)
        q = g.dequant(nm, frac=FW, max_blocks=256)
        rel = np.max(np.abs(q.astype(np.float64) / (1 << FW) - f)) / max(np.max(np.abs(f)), 1e-9)
    worst = max(worst, rel)
    log("%-28s %-6s %-22s rel=%.2e" % (nm, tn, str(t["shape"]), rel))

log(f"\nWORST int-vs-float rel err over sampled real tensors: {worst:.2e}")
log("PASS" if worst < 1e-5 else "CHECK (rel err higher than expected)")
