#!/usr/bin/env bash
set -euo pipefail
# build_nmc_cuda.sh — per-host CUDA integer kernel (Q4_K/Q6_K fused dequant + fixed-point matmul).
# PER-HOST, arch-specific, NOT committed. Result MUST be byte-identical to the CPU oracle (tests/test_qk_cuda.py).
# Arch auto-detected from the GPU (sm_<compute_cap>); override with CUDA_ARCH. No --use_fast_math (keep the
# fp16->fixed conversion strictly IEEE; integer math is exact regardless).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$ROOT/tools/nmc_qk_cuda.cu"
BIN_DIR="${BONSAI_BIN_DIR:-${BONSAI_NOTARY_HOME:-$HOME/.local/integer_inference_engine/north-mini-code}/bin}"
mkdir -p "$BIN_DIR"
out="$BIN_DIR/libnmc_qk_cuda.so"
nvcc="${NVCC:-nvcc}"
if ! command -v "$nvcc" >/dev/null 2>&1; then
  echo "build_nmc_cuda.sh: nvcc not found (CUDA -devel toolkit needed). GPU kernel is opt-in; CPU path stays." >&2
  exit 1
fi
arch="${CUDA_ARCH:-}"
if [ -z "$arch" ] && command -v nvidia-smi >/dev/null 2>&1; then
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
  [ -n "$cc" ] && arch="sm_${cc}"
fi
arch="${arch:-sm_86}"     # RTX 3070/3090 default
echo "build_nmc_cuda.sh: nvcc $($nvcc --version | grep -oE 'release [0-9.]+' | head -1)  arch=$arch -> $out"
# --cudart static: self-contained .so (no runtime libcudart.so dependency) so ctypes loads it even when CUDA
# isn't on the default loader path (e.g. vast.ai -devel containers without ldconfig'd /usr/local/cuda/lib64).
exec "$nvcc" -O3 -arch="$arch" --cudart static -Xcompiler -fPIC -shared "$src" -o "$out"
