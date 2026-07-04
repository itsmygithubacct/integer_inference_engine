#!/usr/bin/env python3
"""Benchmark the resident register-API vs per-call streaming on a head-sized weight (no model needed).

Per-call re-uploads the weight bytes every apply; resident uploads once then reuses. Measures the residency
win (amortized weight H2D) on this GPU — and reports byte-equality (the determinism contract). Synthetic
Q6_K weight shaped like the tied head (out_f=262144, in_f=2048) — the heaviest matmul in the forward.

    PYTHONPATH=src .venv/bin/python tools/bench_resident.py [N] [out_f] [n_blocks]
"""
import sys
import time

import numpy as np

from nmc import qk_cuda

if not qk_cuda.resident_available():
    print("resident register API unavailable (build tools/build_nmc_cuda.sh on a GPU host)"); sys.exit(1)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
out_f = int(sys.argv[2]) if len(sys.argv) > 2 else 262144      # tied-head rows
nb = int(sys.argv[3]) if len(sys.argv) > 3 else 8             # in_f = 2048
QTYPE, FW, T = qk_cuda.Q6_K, 24, 1
rng = np.random.default_rng(0)
raw = rng.integers(0, 256, size=out_f * nb * 210, dtype=np.uint8).tobytes()   # Q6_K block bytes
x = rng.integers(-(1 << 14), 1 << 14, size=(T, nb * 256), dtype=np.int64)
print(f"head-sized weight: out_f={out_f} in_f={nb*256} ({len(raw)/1e6:.0f}MB Q6_K)  N={N} applies")

qk_cuda.qk_linear(raw, x, out_f, nb, FW, QTYPE)               # warm up the CUDA context
t = time.time()
for _ in range(N):
    r_pc = qk_cuda.qk_linear(raw, x, out_f, nb, FW, QTYPE)    # per-call: re-uploads the weight each time
percall = (time.time() - t) / N

h = qk_cuda.register_weight(raw, out_f, nb, QTYPE)            # upload ONCE
assert h is not None
t = time.time()
for _ in range(N):
    r_res = qk_cuda.apply_resident(h, x, out_f, FW)           # resident: weight already in VRAM
resident = (time.time() - t) / N
qk_cuda.free_all()

print(f"per-call : {percall*1000:7.1f} ms/apply  (H2D weight + kernel + D2H, every call)")
print(f"resident : {resident*1000:7.1f} ms/apply  (kernel + tiny x H2D/D2H only)")
print(f"speedup  : {percall/resident:6.2f}x   byte-identical: {np.array_equal(r_pc, r_res)}")
