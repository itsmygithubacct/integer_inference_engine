#!/usr/bin/env bash
# bonsai.sh — launch ATLAS-Notarized-Bonsai-8B in one of its model/states.
#
# A thin dispatcher over the real CLI (`python -m trinote.cli.run_bonsai_cli`); each mode is just a curated
# flag set, so anything the CLI accepts can still be appended after the prompt. Run from anywhere — the repo
# root is resolved from this script's location.
#
#   scripts/bonsai.sh <mode> [PROMPT] [extra CLI flags...]      # PROMPT may also come last, or use -p "..."
#
# Modes:
#   json            structured JSON output {thinking,answer,bonsai,receipt,bundle} (deterministic + receipted)
#   repl            interactive REPL (omit PROMPT) — deterministic + receipted, type 'quit' to exit
#   deterministic   deterministic integer engine, NO receipt (fastest receipt-free reproducible run)   [alias: det]
#   receipted       deterministic + local notarized receipt (byte-exact re-execution + verify)         [alias: rcpt]
#   onchain         receipted + BSV OP_RETURN Third Entry — DRY-RUN by default (no spend/broadcast)
#   bonsai27        Bonsai-27B Q1 GGUF through PrismML llama.cpp (Linux CUDA, inference-only, NO receipt) [alias: 27b]
#
# GPU:  native modes use the GPU (resident-monolith prefill + KV-export, byte-identical) when available.
#       Set BONSAI_GPU=0 to force CPU.
# Onchain: builds the tx but does NOT broadcast unless you append --chain-confirm (it spends real BSV).
#
# Examples:
#   scripts/bonsai.sh receipted "What is the capital of France?" -n 64
#   scripts/bonsai.sh json "List three primes." --answer
#   scripts/bonsai.sh repl
#   scripts/bonsai.sh deterministic "Hello" --sampler greedy
#   scripts/bonsai.sh onchain "Notarize this." --chain-confirm        # real BSV broadcast
#   BONSAI_GPU=0 scripts/bonsai.sh receipted "..."                     # force CPU
#   BONSAI_DRYRUN=1 scripts/bonsai.sh json "..."                       # print the command, don't run
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="python3"                              # fall back to system python3 if no venv
GPU="${BONSAI_GPU:-1}"                                    # 1 = use GPU on native modes (default), 0 = CPU

# Print the header comment block (lines after the shebang up to the first non-comment line) as help.
usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next} {exit}' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

[ $# -ge 1 ] || usage 1
mode="$1"; shift

# The 27B GGUF model has its own launcher and argument surface. Dispatch before the native launcher's positional
# prompt normalization so model-option values cannot be mistaken for the prompt.
case "$mode" in bonsai27|27b) exec "$REPO/../bonsai-27b-cli" "$@" ;; esac

# Optional positional PROMPT, accepted EITHER first ("prompt" --flags) OR last (--flags "prompt"). Using -p
# explicitly always works and takes precedence. (If you end the line with a value-taking flag and give no
# prompt, pass it as -p to avoid the trailing token being read as the prompt.)
args=("$@")
have_p=0
for a in "${args[@]+"${args[@]}"}"; do case "$a" in -p|--prompt) have_p=1 ;; esac; done
prompt_args=()
if [ "$mode" != "repl" ] && [ ${#args[@]} -ge 1 ] && [ "${args[0]:0:1}" != "-" ]; then # prompt first
    prompt_args=(-p "${args[0]}"); args=("${args[@]:1}")
elif [ "$mode" != "repl" ] && [ $have_p -eq 0 ] && [ ${#args[@]} -ge 1 ]; then # else prompt last
    last="${args[$((${#args[@]}-1))]}"
    if [ "${last:0:1}" != "-" ]; then
        prompt_args=(-p "$last"); unset 'args[$((${#args[@]}-1))]'; args=("${args[@]+"${args[@]}"}")
    fi
fi
set -- "${args[@]+"${args[@]}"}"

# --gpu for native (integer-engine) modes only, gated by BONSAI_GPU.
gpu_args=()
[ "$GPU" = "1" ] && gpu_args=(--gpu)

case "$mode" in
    json)                  base=(--json --fast "${gpu_args[@]}") ;;
    repl)                  base=(--repl --chat --fast "${gpu_args[@]}") ;;
    deterministic|det)     base=(--fast "${gpu_args[@]}" --no-receipt) ;;
    receipted|rcpt)        base=(--fast "${gpu_args[@]}" --receipt) ;;
    onchain)               base=(--fast "${gpu_args[@]}" --receipt --onchain) ;;  # dry-run unless --chain-confirm
    -h|--help|help)        usage 0 ;;
    *) echo "bonsai.sh: unknown mode '$mode' (try: json repl deterministic receipted onchain bonsai27)" >&2; exit 2 ;;
esac

cmd=("$PY" -m trinote.cli.run_bonsai_cli "${base[@]}" "${prompt_args[@]}" "$@")

if [ "$mode" = "onchain" ]; then
    case " $* " in *" --chain-confirm "*) echo "[bonsai.sh] ONCHAIN: --chain-confirm set — this BROADCASTS a real BSV tx (spends sats)." >&2 ;;
                   *) echo "[bonsai.sh] onchain DRY-RUN (builds the Third Entry tx but does not broadcast). Append --chain-confirm to spend." >&2 ;; esac
fi

if [ "${BONSAI_DRYRUN:-0}" = "1" ]; then
    printf '[bonsai.sh] would run:'; printf ' %q' PYTHONPATH=src "${cmd[@]}"; printf '\n'; exit 0
fi

cd "$REPO"
exec env PYTHONPATH=src "${cmd[@]}"
