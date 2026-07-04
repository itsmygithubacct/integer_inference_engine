#!/usr/bin/env python3
"""Lever-1 measurement: batched-MoE prefill (one set of kernels over all m·k token-expert pairs) vs the
per-token qk_moe_ffn loop, byte-exact. (The DP4A A/B was measured separately as ~neutral end-to-end.)

    sudo env PYTHONPATH=src NMC_BACKEND=cuda-resident .venv/bin/python tools/bench_dp4a_e2e.py <blob> [P]
"""
import sys
import time

from nmc.engine import Engine

blob = sys.argv[1]
P = int(sys.argv[2]) if len(sys.argv) > 2 else 256          # prompt length (≥64 to exercise the prefill MoE batch)

eng = Engine(blob)
text = "The history and future of deterministic, verifiable machine inference, told at some length. " * 400
ids = eng.encode(text)[:P]
print(f"[bench] backend={eng.bname} fused={eng.fused} prompt={len(ids)}tok", flush=True)

eng.generate(ids, 1)                                        # warm: register all touched weights (1 prefill)


def tb(batch):
    eng.batch_moe = batch
    t = time.time(); o = eng.generate(ids, 1); return time.time() - t, o


b_on, ob = tb(True)
b_off, op = tb(False)
if eng.resident:
    eng.free()
print(f"[bench] PREFILL {P} tok: batched-MoE {b_on:.2f}s   per-token-MoE {b_off:.2f}s   "
      f"speedup {b_off/b_on:.2f}x   byte-exact {ob == op}")
print("[bench] done")
