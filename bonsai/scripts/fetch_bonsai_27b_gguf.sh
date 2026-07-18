#!/usr/bin/env bash
# Download and verify the pinned prism-ml/Bonsai-27B-gguf Q1 model.
# The public repository downloads anonymously; HF_TOKEN/BONSAI_TOKEN or HF_TOKEN_FILE is honored if set.
#
# Usage: scripts/fetch_bonsai_27b_gguf.sh [--dry-run] [--force]
set -euo pipefail

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next} {exit}' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

REPO="prism-ml/Bonsai-27B-gguf"
REVISION="0cf7e3d21581b169b4df1de8bf01316000e2fbb7"
FILENAME="Bonsai-27B-Q1_0.gguf"
EXPECTED_SIZE="3803452480"
EXPECTED_SHA256="17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0"
URL="https://huggingface.co/$REPO/resolve/$REVISION/$FILENAME"
NOTARY_HOME="${BONSAI_NOTARY_HOME:-$HOME/.local/trinote}"
MODELS_DIR="${BONSAI_MODELS_DIR:-$NOTARY_HOME/models}"
DEST="${BONSAI_27B_GGUF:-$MODELS_DIR/$FILENAME}"
DRY_RUN=0
FORCE=0

while (($#)); do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
        -h|--help|help) usage 0 ;;
        *) echo "fetch_bonsai_27b_gguf.sh: unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

TOKEN="${HF_TOKEN:-${BONSAI_TOKEN:-}}"
TOKEN_FILE="${HF_TOKEN_FILE:-}"
if [ -z "$TOKEN" ] && [ -z "$TOKEN_FILE" ] && [ -r "$HOME/.hugging_face_token.txt" ]; then
    TOKEN_FILE="$HOME/.hugging_face_token.txt"
fi
if [ -z "$TOKEN" ] && [ -n "$TOKEN_FILE" ]; then
    [ -r "$TOKEN_FILE" ] || { echo "fetch_bonsai_27b_gguf.sh: cannot read HF_TOKEN_FILE=$TOKEN_FILE" >&2; exit 2; }
    TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"
fi

if ((DRY_RUN)); then
    printf '[dry-run] model: %s@%s/%s\n' "$REPO" "$REVISION" "$FILENAME"
    printf '[dry-run] download: %s\n' "$URL"
    printf '[dry-run] size: %s bytes\n' "$EXPECTED_SIZE"
    printf '[dry-run] sha256: %s\n' "$EXPECTED_SHA256"
    printf '[dry-run] destination: %s\n' "$DEST"
    [ -n "$TOKEN" ] && printf '[dry-run] Hugging Face authentication: configured (secret not shown)\n' \
                     || printf '[dry-run] Hugging Face authentication: anonymous (repository is public)\n'
    exit 0
fi

for tool in curl sha256sum stat; do
    command -v "$tool" >/dev/null 2>&1 || { echo "fetch_bonsai_27b_gguf.sh: $tool is required" >&2; exit 2; }
done

if [ -f "$DEST" ]; then
    ACTUAL="$(sha256sum "$DEST" | awk '{print $1}')"
    if [ "$ACTUAL" = "$EXPECTED_SHA256" ]; then
        echo "[bonsai-27b] model already present and verified: $DEST"
        exit 0
    fi
    if (( ! FORCE )); then
        echo "fetch_bonsai_27b_gguf.sh: existing model failed checksum: $DEST" >&2
        echo "Re-run with --force to preserve it as a timestamped backup and fetch the pinned model." >&2
        exit 1
    fi
    mv "$DEST" "$DEST.invalid.$(date +%Y%m%d%H%M%S)"
fi

mkdir -p "$(dirname "$DEST")"
PART="$DEST.part"
if [ -f "$PART" ] && [ "$(stat -c %s "$PART")" -gt "$EXPECTED_SIZE" ]; then
    mv "$PART" "$PART.invalid.$(date +%Y%m%d%H%M%S)"
fi
echo "[bonsai-27b] downloading pinned Q1 GGUF (3.80 GB; resumable) ..."
if [ -n "$TOKEN" ]; then
    printf 'header = "Authorization: Bearer %s"\n' "$TOKEN" | \
        curl --config - --location --fail --retry 5 --retry-delay 2 --continue-at - --output "$PART" "$URL"
else
    curl --location --fail --retry 5 --retry-delay 2 --continue-at - --output "$PART" "$URL"
fi

ACTUAL_SIZE="$(stat -c %s "$PART")"
[ "$ACTUAL_SIZE" = "$EXPECTED_SIZE" ] || {
    echo "fetch_bonsai_27b_gguf.sh: size mismatch: expected $EXPECTED_SIZE, got $ACTUAL_SIZE" >&2
    exit 1
}
ACTUAL_SHA256="$(sha256sum "$PART" | awk '{print $1}')"
[ "$ACTUAL_SHA256" = "$EXPECTED_SHA256" ] || {
    echo "fetch_bonsai_27b_gguf.sh: checksum mismatch: expected $EXPECTED_SHA256, got $ACTUAL_SHA256" >&2
    exit 1
}
mv "$PART" "$DEST"
echo "[bonsai-27b] model verified: $DEST"
