#!/usr/bin/env bash
set -euo pipefail
# build_bonsai_q1_gpu.sh — PER-HOST OPT-IN GPU Q1_0 kernel (NOT a committed/portable artifact).
# Compiles tools/bonsai_q1_gpu.cu -> tools/libbonsai_q1_gpu.so for THIS host's GPU only.
# Like the AVX-512 stance: never the committed default. The result MUST be byte-identical to the int64 CPU
# oracle (q1_linear_ref); the CPU oracle stays the canonical verifier. CUDA_ARCH overrides auto-detection.
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
arch="${CUDA_ARCH:-}"

if ! command -v "$nvcc" >/dev/null 2>&1; then
  echo "build_bonsai_q1_gpu.sh: nvcc not found on PATH (set NVCC=...). GPU build is opt-in; the CPU path stays." >&2
  exit 1
fi
if [ ! -f "$src" ]; then
  echo "build_bonsai_q1_gpu.sh: $src not written yet — see IMPLEMENT-GPU-MODE.md milestone M1." >&2
  echo "  (Until the kernel exists, gpu_native.gpu_available() is False and --gpu falls back to CPU.)" >&2
  exit 2
fi

if [ -z "$arch" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "build_bonsai_q1_gpu.sh: cannot auto-detect CUDA architecture; set CUDA_ARCH=sm_XX" >&2
    exit 2
  fi
  capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed -n '1p' | tr -d '[:space:].')"
  if [[ ! "$capability" =~ ^[0-9]{2,3}$ ]]; then
    echo "build_bonsai_q1_gpu.sh: invalid compute capability from nvidia-smi; set CUDA_ARCH=sm_XX" >&2
    exit 2
  fi
  arch="sm_$capability"
fi
if [[ ! "$arch" =~ ^sm_[0-9]{2,3}$ ]]; then
  echo "build_bonsai_q1_gpu.sh: CUDA_ARCH must look like sm_86, got: $arch" >&2
  exit 2
fi
arch_number="${arch#sm_}"
if ((10#$arch_number < 75)); then
  echo "build_bonsai_q1_gpu.sh: $arch is unsupported; exact BMMA kernels require sm_75 or newer" >&2
  exit 2
fi
echo "[bonsai-gpu-build] target architecture: $arch${CUDA_ARCH:+ (override)}"

# NO --use_fast_math / -ffast-math: would relax/reorder arithmetic. Integer kernel only, but assert it.
# -O3, -fPIC -shared (ctypes), no -arch=native (pin sm explicitly so a wrong-arch .so can't ship).
# Build beside the destination and publish only after checking the dynamic ABI:
# CUDA 12.8/GCC 13 has otherwise produced a successful .so with ctx_create
# internalized, which makes the resident path unavailable at runtime.
if ! command -v nm >/dev/null 2>&1; then
  echo "build_bonsai_q1_gpu.sh: nm not found; cannot verify the CUDA ctypes ABI" >&2
  exit 1
fi
tmp="${out}.tmp.$$"
cleanup() { rm -f "$tmp"; }
trap cleanup EXIT HUP INT TERM
"$nvcc" -O3 -arch="$arch" -Xcompiler -fPIC -shared "$src" -o "$tmp"
for symbol in \
  bonsai_gpu_abi_version \
  bonsai_gpu_abi_manifest \
  bonsai_gpu_last_error \
  bonsai_gpu_last_error_code \
  bonsai_gpu_device_probe \
  bonsai_q1_linear_gpu \
  bonsai35_ctx_create \
  bonsai35_ctx_set_trace \
  bonsai35_ctx_set_projection_grouping \
  bonsai35_ctx_projection_stats \
  bonsai35_ctx_graph_stats \
  bonsai35_ctx_profile_stats \
  bonsai35_ctx_execution_stats \
  bonsai35_ctx_sampler_stats \
  bonsai35_profile_decode_token \
  bonsai35_decode_batch \
  bonsai35_fused_silu_mul_gpu \
  bonsai35_prefill_tokens \
  bonsai35_decode_token_device \
  bonsai35_sample_prepare \
  bonsai35_sample_select
do
  if ! nm -D --defined-only "$tmp" | awk -v wanted="$symbol" '$3 == wanted { found=1 } END { exit !found }'; then
    echo "build_bonsai_q1_gpu.sh: required dynamic symbol is missing: $symbol" >&2
    exit 1
  fi
done

# Loading and probing the just-built temporary verifies more than dynsym: it
# catches wrong-architecture objects, stale driver/runtime combinations, and a
# manifest/version mismatch before the old known-good library is replaced.
if [ "${BONSAI_SKIP_DEVICE_PROBE:-0}" != "1" ]; then
  py="${PYTHON:-python3}"
  if ! command -v "$py" >/dev/null 2>&1; then
    echo "build_bonsai_q1_gpu.sh: python3 is required for the device/ABI probe (or set BONSAI_SKIP_DEVICE_PROBE=1)" >&2
    exit 1
  fi
  "$py" - "$tmp" "$arch_number" <<'PY'
import ctypes
import json
import os
import sys

path, expected_arch = sys.argv[1], int(sys.argv[2])
lib = ctypes.CDLL(path)
lib.bonsai_gpu_abi_version.restype = ctypes.c_int
version = int(lib.bonsai_gpu_abi_version())
if version != 3:
    raise SystemExit(f"build_bonsai_q1_gpu.sh: expected ABI version 3, got {version}")
lib.bonsai_gpu_abi_manifest.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
lib.bonsai_gpu_abi_manifest.restype = ctypes.c_size_t
n = int(lib.bonsai_gpu_abi_manifest(None, 0))
buf = ctypes.create_string_buffer(n + 1)
if int(lib.bonsai_gpu_abi_manifest(buf, len(buf))) != n:
    raise SystemExit("build_bonsai_q1_gpu.sh: ABI manifest length changed during probe")
manifest = json.loads(buf.value)
if manifest.get("abi_version") != version:
    raise SystemExit("build_bonsai_q1_gpu.sh: ABI manifest/version mismatch")
major = ctypes.c_int()
minor = ctypes.c_int()
lib.bonsai_gpu_device_probe.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
lib.bonsai_gpu_device_probe.restype = ctypes.c_int
rc = int(lib.bonsai_gpu_device_probe(ctypes.byref(major), ctypes.byref(minor)))
if rc != 0:
    lib.bonsai_gpu_last_error.restype = ctypes.c_char_p
    detail = (lib.bonsai_gpu_last_error() or b"unknown CUDA error").decode("utf-8", "replace")
    raise SystemExit(f"build_bonsai_q1_gpu.sh: device probe failed: {detail}")
runtime_arch = major.value * 10 + minor.value
if runtime_arch < 75:
    raise SystemExit(f"build_bonsai_q1_gpu.sh: runtime device sm_{runtime_arch} cannot execute BMMA")
if expected_arch != runtime_arch and os.environ.get("BONSAI_ALLOW_CUDA_ARCH_MISMATCH") != "1":
    raise SystemExit(
        f"build_bonsai_q1_gpu.sh: target sm_{expected_arch} does not match runtime device sm_{runtime_arch}; "
        "set BONSAI_ALLOW_CUDA_ARCH_MISMATCH=1 only for an intentional cross-build"
    )
PY
fi
mv -f "$tmp" "$out"
trap - EXIT HUP INT TERM
