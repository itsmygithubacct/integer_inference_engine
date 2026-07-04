#!/usr/bin/env bash
# install_bonsai_receipted.sh — one-shot, IDEMPOTENT setup for the receipted Bonsai-8B engine.
#
# Runs every download/build/setup step needed to produce notarized, byte-exact receipts:
#   1. uv venv                                  (.venv/)                         — skipped if present
#   2. uv pip install -r requirements.txt       (numpy, safetensors, ecdsa)      — no-op if satisfied
#   3. build the CPU native kernel              (libbonsai_q1_kernel.so)         — skipped if up to date
#   4. build the CUDA kernel (if nvcc present)  (libbonsai_q1_gpu.so)            — optional / per-host
#   5. fetch model weights                      (scripts/fetch_weights.sh)       — skips verified copies
#   6. import GGUF -> reference artifact         (.safetensors)                   — only if artifact absent
#   7. generate receipt signing keys            (secp256k1, 0600)                — created only if absent
#
# Re-running is safe: each step is a no-op when its output already exists and is current.
# Built kernels, weights, and keys live under $BONSAI_NOTARY_HOME
# (default ~/.local/trinote), never in the source tree.
#
# Usage:  ./install_bonsai_receipted.sh [--force] [--no-weights] [--no-gpu] [-h]
#   --force        rebuild kernels and re-download weights even if present
#   --no-weights   skip the (large) weight download — engine + tests still work on the synthetic model
#   --no-gpu       skip the optional CUDA kernel build
#
# Weights download from HuggingFace by default (prism-ml/Bonsai-8B-gguf, public). Override the repo
# with HF_REPO (and HF_TOKEN for a private mirror), e.g.
#   HF_REPO=<org>/Bonsai-8B-gguf ./install_bonsai_receipted.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

FORCE=0; DO_WEIGHTS=1; DO_GPU=1
for a in "$@"; do
  case "$a" in
    --force) FORCE=1 ;;
    --no-weights) DO_WEIGHTS=0 ;;
    --no-gpu) DO_GPU=0 ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "install_bonsai_receipted.sh: unknown arg '$a' (try --help)" >&2; exit 2 ;;
  esac
done

NOTARY_HOME="${BONSAI_NOTARY_HOME:-$HOME/.local/trinote}"
BIN_DIR="${BONSAI_BIN_DIR:-$NOTARY_HOME/bin}"
PY="$ROOT/.venv/bin/python"

bold() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32m✓\033[0m %s\n' "$*"; }
skip() { printf '    \033[33m·\033[0m %s\n' "$*"; }
warn() { printf '    \033[31m!\033[0m %s\n' "$*" >&2; }

# True if the build output SO is missing or older than its source SRC (or --force).
needs_build() {  # SRC SO
  [ "$FORCE" = "1" ] && return 0
  [ -f "$2" ] || return 0
  [ "$1" -nt "$2" ] && return 0
  return 1
}

# ---- 1. venv ----------------------------------------------------------------
bold "1/7  Python virtualenv (uv)"
command -v uv >/dev/null 2>&1 || { warn "uv not found on PATH — install uv first (https://docs.astral.sh/uv/)"; exit 1; }
if [ -x "$PY" ]; then
  skip ".venv already present ($("$PY" --version 2>&1))"
else
  uv venv --python 3.12 >/dev/null
  ok "created .venv"
fi

# ---- 2. requirements --------------------------------------------------------
bold "2/7  Runtime dependencies"
uv pip install -q -r requirements.txt
ok "requirements.txt satisfied (numpy, safetensors, ecdsa)"

# ---- 3. CPU native kernel ---------------------------------------------------
bold "3/7  CPU native kernel"
CPU_SO="$BIN_DIR/libbonsai_q1_kernel.so"
if needs_build "tools/bonsai_q1_kernel.c" "$CPU_SO"; then
  bash tools/build_bonsai_q1_kernel.sh >/dev/null && ok "built $CPU_SO"
else
  skip "up to date ($CPU_SO)"
fi

# ---- 4. CUDA kernel (optional, per-host) ------------------------------------
bold "4/7  CUDA kernel (optional)"
GPU_SO="$BIN_DIR/libbonsai_q1_gpu.so"
if [ "$DO_GPU" = "0" ]; then
  skip "skipped (--no-gpu)"
elif ! command -v "${NVCC:-nvcc}" >/dev/null 2>&1; then
  skip "no nvcc on PATH — GPU build skipped; --gpu will fall back to the CPU oracle"
elif needs_build "tools/bonsai_q1_gpu.cu" "$GPU_SO"; then
  if bash tools/build_bonsai_q1_gpu.sh >/dev/null 2>&1; then ok "built $GPU_SO"; else warn "CUDA build failed (non-fatal; CPU path still works)"; fi
else
  skip "up to date ($GPU_SO)"
fi

# ---- 5. model weights -------------------------------------------------------
bold "5/7  Model weights"
if [ "$DO_WEIGHTS" = "0" ]; then
  skip "skipped (--no-weights)"
else
  fw_args=(); [ "$FORCE" = "1" ] && fw_args=(--force)
  if bash scripts/fetch_weights.sh "${fw_args[@]+"${fw_args[@]}"}"; then
    ok "weights present and verified"
  else
    warn "weight fetch incomplete — set HF_REPO (and HF_TOKEN if private) and re-run, or copy the files into $NOTARY_HOME/models. Engine + tests still run on the synthetic model."
  fi
fi

# ---- 6. import GGUF -> reference artifact (only if the .safetensors is absent) ---------------
bold "6/7  Import GGUF -> reference artifact"
ART="$(PYTHONPATH=src "$PY" -c 'from trinote.notary_paths import default_artifact; print(default_artifact())')"
GGUF="$(PYTHONPATH=src "$PY" -c 'from trinote.notary_paths import default_gguf; print(default_gguf())')"
if [ "$FORCE" != "1" ] && [ -f "$ART" ]; then
  skip "artifact present ($ART)"
elif [ ! -f "$GGUF" ]; then
  skip "no GGUF present to import (fetch weights first, or set HF_REPO)"
else
  echo "    importing GGUF -> int reference artifact (dequantize; ~1-3 min) ..."
  if PYTHONPATH=src "$PY" -m trinote.cli.import_bonsai_gguf_cli >/dev/null; then
    ok "imported $ART"
  else
    warn "GGUF import failed — run 'PYTHONPATH=src .venv/bin/python -m trinote.cli.import_bonsai_gguf_cli' to see the error"
  fi
fi

# ---- 7. receipt signing keys ------------------------------------------------
bold "7/7  Receipt signing keys (secp256k1, 0600)"
PYTHONPATH=src "$PY" - <<'PYEOF'
import os
from trinote.infer_int.bonsai_runtime import load_or_generate_signing_keys
from trinote.notary_paths import model_key_default, counterparty_key_default
paths = [model_key_default(), counterparty_key_default()]
existed = {p: os.path.exists(p) for p in paths}     # check BEFORE generating
mk, ck = load_or_generate_signing_keys()
for p in paths:
    print(f"    {'present' if existed[p] else 'created'}: {p}")
print(f"    model pubkey: {mk.public_hex[:24]}…  keyId {mk.key_id}")
PYEOF
ok "signing keys ready"

# ---- summary ----------------------------------------------------------------
bold "Done — receipted Bonsai-8B engine ready"
PYTHONPATH=src "$PY" - <<'PYEOF'
import os
from trinote.notary_paths import notary_home, default_gguf, default_artifact, kernel_so, gpu_kernel_so
def mark(p): return ("present" if os.path.exists(p) else "MISSING")
print(f"    state home : {notary_home()}")
print(f"    cpu kernel : {mark(kernel_so())}  ({kernel_so()})")
print(f"    gpu kernel : {mark(gpu_kernel_so())}")
print(f"    gguf       : {mark(default_gguf())}")
print(f"    artifact   : {mark(default_artifact())}")
try:
    from trinote.infer_int.reference_bonsai import BonsaiReferenceModel  # import sanity
    print("    engine     : import OK")
except Exception as e:
    print(f"    engine     : IMPORT FAILED — {e}")
PYEOF
echo
echo "    Run:   ../bonsai-cli \"What is a tensor?\" --receipts --verbose"
echo "    REPL:  ../bonsai-cli repl --receipts"
