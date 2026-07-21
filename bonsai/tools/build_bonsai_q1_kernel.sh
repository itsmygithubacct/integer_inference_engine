#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
build_bonsai_q1_kernel.sh — build the Bonsai Q1_0 native inference kernel

Purpose:
  Compile tools/bonsai_q1_kernel.c into the shared library
  tools/libbonsai_q1_kernel.so used by the deterministic integer-inference
  engine. The build uses only a portable baseline ISA and byte-exact
  arithmetic flags so the resulting .so produces identical output across
  hosts and never SIGILLs on CPUs lacking newer ISA extensions.

Options:
  -h, --help    Print this help and exit 0 (no build, no side effects).

Environment variables:
  CC            C compiler to use (default: gcc). Must be on PATH.

Notes:
  Baseline ISA is selected from `uname -m`:
    x86_64  -> -march=x86-64-v2
    aarch64 -> -march=armv8-a
    other   -> (no -march; rely on compiler default)
  OpenMP (-fopenmp) is added only if a real compile-test succeeds.
  -march=native and -ffast-math are intentionally NOT used: they would
  change emitted code/arithmetic and break byte-exact, redistributable output.
  -fwrapv -fno-strict-overflow ARE used: the kernel deliberately reproduces
  NumPy's two's-complement wrap on signed int64 overflow (the byte-exact
  contract), so every signed overflow must wrap rather than be treated as
  UB a compiler may optimize on. This makes wrap defined regardless of
  compiler/version at negligible -O3 cost.

Example:
  tools/build_bonsai_q1_kernel.sh
  CC=clang tools/build_bonsai_q1_kernel.sh
EOF
}

# Handle --help BEFORE any work: no build, no model load, no broadcast, no network.
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cc="${CC:-gcc}"

# Guard: the C compiler must exist on PATH.
if ! command -v "$cc" >/dev/null 2>&1; then
  echo "build_bonsai_q1_kernel.sh: C compiler '$cc' not found on PATH (set CC=...)" >&2
  exit 1
fi

# OpenMP: use a real compile-test (not a GCC-internal preprocessor probe) so the
# result reflects whether this compiler can actually build/link an OpenMP object.
openmp_flags=()
tmp_omp="$(mktemp --suffix=.c 2>/dev/null || mktemp)"
trap 'rm -f "$tmp_omp" "${tmp_omp%.c}.o" "${tmp_omp%.c}.march.o"' EXIT
cat >"$tmp_omp" <<'EOF'
#include <omp.h>
int main(void) {
  int n = 0;
  #pragma omp parallel
  { n += omp_get_thread_num(); }
  return n & 0;
}
EOF
if "$cc" -fopenmp -c "$tmp_omp" -o "${tmp_omp%.c}.o" >/dev/null 2>&1; then
  openmp_flags=(-fopenmp)
fi

# Portable baseline ISA selected by machine arch. Using -march=native would bake in
# THIS host's ISA extensions (AVX-512, etc.) and the resulting .so would SIGILL on any
# CPU lacking them — bad for a committed, redistributable kernel. NOTE: the committed
# tools/libbonsai_q1_kernel.so is NOT rebuilt by the review; re-running this script
# regenerates it with the portable flags below (and that fresh .so is what subsequent
# runs adopt). Do NOT add -ffast-math or -march=native: they change emitted arithmetic
# and would break byte-exact output.
march_flags=()
arch="$(uname -m)"
case "$arch" in
  x86_64)
    # GCC 9 (the stock compiler on Ubuntu 20.04) predates the x86-64-v2
    # spelling.  Compile-test it and retain a portable x86-64 fallback so a
    # fresh Turing-era host can still build the canonical CPU oracle.
    if "$cc" -march=x86-64-v2 -c "$tmp_omp" -o "${tmp_omp%.c}.march.o" >/dev/null 2>&1; then
      march_flags=(-march=x86-64-v2)
    else
      march_flags=(-march=x86-64)
    fi
    ;;
  aarch64)
    march_flags=(-march=armv8-a)
    ;;
  *)
    march_flags=()
    ;;
esac

# -fwrapv -fno-strict-overflow: make signed int64 overflow a DEFINED two's-complement wrap (matching the
# NumPy oracle), not UB. The kernel relies on wrap at e.g. the attention score scale and the SiLU distance;
# without these flags a different compiler/version could legally misoptimize and diverge from the committed
# byte-exact result.
# Built kernel goes to $BONSAI_NOTARY_HOME/bin (build artifacts are not source); the loader prefers it and
# falls back to <repo>/tools for back-compat. Override the location with $BONSAI_BIN_DIR.
BIN_DIR="${BONSAI_BIN_DIR:-${BONSAI_NOTARY_HOME:-$HOME/.local/trinote}/bin}"; mkdir -p "$BIN_DIR"
rm -f "$tmp_omp" "${tmp_omp%.c}.o" "${tmp_omp%.c}.march.o"
trap - EXIT
exec "$cc" -O3 "${march_flags[@]}" -mtune=generic -fwrapv -fno-strict-overflow -fPIC -shared "${openmp_flags[@]}" \
  "$ROOT/tools/bonsai_q1_kernel.c" \
  -o "$BIN_DIR/libbonsai_q1_kernel.so"
