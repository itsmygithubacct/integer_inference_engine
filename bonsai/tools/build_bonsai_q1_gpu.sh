#!/usr/bin/env bash
set -euo pipefail
# build_bonsai_q1_gpu.sh — PER-HOST OPT-IN GPU Q1_0 kernel (NOT a committed/portable artifact).
# Compiles tools/bonsai_q1_gpu.cu -> tools/libbonsai_q1_gpu.so for THIS host's GPU only.
# Like the AVX-512 stance: never the committed default. The result MUST be byte-identical to the int64 CPU
# oracle (q1_linear_ref); the CPU oracle stays the canonical verifier. -arch=sm_86 = RTX 3070 (Ampere).
# See research/bonsai-notary/IMPLEMENT-GPU-MODE.md (build order + parity gate) and Q1-BITMATMUL-REFORMULATION.md.
usage() { sed -n '2,8p' "$0"; }
for arg in "$@"; do case "$arg" in -h|--help) usage; exit 0;; esac; done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$ROOT/tools/bonsai_q1_gpu.cu"
# Built kernel goes to $BONSAI_NOTARY_HOME/bin (build artifacts are not source); the loader prefers it and
# falls back to <repo>/tools for back-compat. Override with $BONSAI_BIN_DIR.
BIN_DIR="${BONSAI_BIN_DIR:-${BONSAI_NOTARY_HOME:-$HOME/.local/trinote}/bin}"; mkdir -p "$BIN_DIR"
out="$BIN_DIR/libbonsai_q1_gpu.so"
nvcc="${NVCC:-nvcc}"
arch="${CUDA_ARCH:-sm_86}"          # RTX 3070; override for other hosts, never bake into a committed build

if ! command -v "$nvcc" >/dev/null 2>&1; then
  echo "build_bonsai_q1_gpu.sh: nvcc not found on PATH (set NVCC=...). GPU build is opt-in; the CPU path stays." >&2
  exit 1
fi
if [ ! -f "$src" ]; then
  echo "build_bonsai_q1_gpu.sh: $src not written yet — see IMPLEMENT-GPU-MODE.md milestone M1." >&2
  echo "  (Until the kernel exists, gpu_native.gpu_available() is False and --gpu falls back to CPU.)" >&2
  exit 2
fi

# NO --use_fast_math / -ffast-math: would relax/reorder arithmetic. Integer kernel only, but assert it.
# -O3, -fPIC -shared (ctypes), no -arch=native (pin sm explicitly so a wrong-arch .so can't ship).
exec "$nvcc" -O3 -arch="$arch" -Xcompiler -fPIC -shared "$src" -o "$out"
