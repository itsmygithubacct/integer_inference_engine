#!/usr/bin/env bash
set -euo pipefail
# build_nmc_kernel.sh — per-host CPU integer kernel (Q4_K/Q6_K fused dequant + fixed-point matmul).
# NOT a committed artifact; result MUST be byte-identical to the numpy oracle (tests/test_qk_native.py).
# No -ffast-math (integer only, but keep the floating fp16->fixed conversion strictly IEEE).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$ROOT/tools/nmc_qk_kernel.c"
BIN_DIR="${BONSAI_BIN_DIR:-${BONSAI_NOTARY_HOME:-$HOME/.local/integer_inference_engine/north-mini-code}/bin}"
mkdir -p "$BIN_DIR"
out="$BIN_DIR/libnmc_qk.so"
cc="${CC:-gcc}"
exec "$cc" -O3 -fopenmp -fPIC -shared "$src" -o "$out" -lm
