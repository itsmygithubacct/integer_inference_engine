#!/usr/bin/env python3
"""Emit structured cold/warm North Mini Code inference telemetry as JSON.

Usage:
    PYTHONPATH=src NMC_BACKEND=cuda-resident python tools/profile_nmc.py MODEL.gguf \
        --prompt "The capital of France is" --new-tokens 4

The report includes Python phase wall time/call counts plus native registration,
H2D/D2H, allocation, resident-call, grouped-projection, and Q/K/V event timing.
It never silently upgrades a CPU fallback into a resident-GPU measurement:
``--require-resident`` is enabled by default and must be explicitly disabled.
"""
from __future__ import annotations

import argparse
import json
import sys

from nmc.engine import Engine


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blob", help="North Mini Code GGUF path")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--new-tokens", type=int, default=4)
    parser.add_argument("--allow-fallback", action="store_true",
                        help="allow a non-resident backend (report remains labelled with the actual backend)")
    parser.add_argument("--resident-preprocess", action="store_true",
                        help="profile the opt-in exact device RMSNorm/router/top-k boundary")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    engine = Engine(args.blob, profile=True, resident_preprocess=args.resident_preprocess)
    try:
        if not engine.resident and not args.allow_fallback:
            raise RuntimeError(f"resident CUDA required for this profile, actual backend: {engine.bname}")
        ids = engine.encode(args.prompt)
        generated = engine.generate(ids, args.new_tokens)
        report = engine.profile_snapshot()
        report["engine"] = {
            "backend": engine.bname,
            "resident": engine.resident,
            "group_projections": engine.group_projections,
            "resident_preprocess": engine.resident_preprocess,
            "batched_moe": engine.batch_moe,
            "batched_moe_dp4a": engine.dp4a_batch_moe,
            "prompt_tokens": len(ids),
            "generated_tokens": len(generated),
        }
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    finally:
        engine.free()


if __name__ == "__main__":
    main()
