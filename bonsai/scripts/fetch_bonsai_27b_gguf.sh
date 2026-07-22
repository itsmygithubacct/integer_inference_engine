#!/usr/bin/env bash
# Download and verify the pinned prism-ml/Bonsai-27B-gguf Q1 model.
# The public repository downloads anonymously; HF_TOKEN/BONSAI_TOKEN or HF_TOKEN_FILE is honored if set.
#
# Usage: scripts/fetch_bonsai_27b_gguf.sh [--dry-run] [--force] [--segments N]
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
SEGMENTS="${BONSAI_FETCH_SEGMENTS:-1}"
MAX_RETRIES="${BONSAI_FETCH_RETRIES:-8}"
RETRY_MAX_TIME="${BONSAI_FETCH_RETRY_MAX_TIME:-900}"
CONNECT_TIMEOUT="${BONSAI_FETCH_CONNECT_TIMEOUT:-30}"
LOW_SPEED_LIMIT="${BONSAI_FETCH_LOW_SPEED_LIMIT:-1024}"
LOW_SPEED_TIME="${BONSAI_FETCH_LOW_SPEED_TIME:-90}"

while (($#)); do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
        --segments)
            (($# >= 2)) || { echo "fetch_bonsai_27b_gguf.sh: --segments requires a value" >&2; exit 2; }
            SEGMENTS="$2"; shift ;;
        -h|--help|help) usage 0 ;;
        *) echo "fetch_bonsai_27b_gguf.sh: unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

is_positive_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
for pair in \
    "segments:$SEGMENTS" \
    "retries:$MAX_RETRIES" \
    "retry max time:$RETRY_MAX_TIME" \
    "connect timeout:$CONNECT_TIMEOUT" \
    "low-speed limit:$LOW_SPEED_LIMIT" \
    "low-speed time:$LOW_SPEED_TIME"
do
    label="${pair%%:*}"; value="${pair#*:}"
    is_positive_int "$value" || {
        echo "fetch_bonsai_27b_gguf.sh: $label must be a positive integer, got: $value" >&2
        exit 2
    }
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
    printf '[dry-run] transfer: %s segment(s), retry-all-errors=%s, low-speed=%sB/s for %ss\n' \
        "$SEGMENTS" "$MAX_RETRIES" "$LOW_SPEED_LIMIT" "$LOW_SPEED_TIME"
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
if [ -f "$PART" ]; then
    partial_size="$(stat -c %s "$PART")"
    if [ "$partial_size" -gt "$EXPECTED_SIZE" ]; then
        mv "$PART" "$PART.invalid.$(date +%Y%m%d%H%M%S)"
    elif [ "$partial_size" = "$EXPECTED_SIZE" ]; then
        partial_sha256="$(sha256sum "$PART" | awk '{print $1}')"
        if [ "$partial_sha256" = "$EXPECTED_SHA256" ]; then
            mv -f "$PART" "$DEST"
            echo "[bonsai-27b] completed download already verified and promoted: $DEST"
            exit 0
        fi
        mv "$PART" "$PART.invalid.$(date +%Y%m%d%H%M%S)"
    fi
fi

# Keep credentials out of argv/process listings.  curl receives the optional
# Authorization header through a transient stdin config and never persists it.
curl_fetch() {
    if [ -n "$TOKEN" ]; then
        printf 'header = "Authorization: Bearer %s"\n' "$TOKEN" | curl --config - "$@"
    else
        curl "$@"
    fi
}

CURL_COMMON=(
    --location --fail --show-error
    --retry "$MAX_RETRIES" --retry-all-errors --retry-max-time "$RETRY_MAX_TIME"
    --connect-timeout "$CONNECT_TIMEOUT"
    --speed-limit "$LOW_SPEED_LIMIT" --speed-time "$LOW_SPEED_TIME"
)

echo "[bonsai-27b] downloading pinned Q1 GGUF (3.80 GB; resumable, $SEGMENTS segment(s)) ..."
if ((SEGMENTS == 1)); then
    curl_fetch "${CURL_COMMON[@]}" --continue-at - --output "$PART" "$URL"
else
    SEGMENT_DIR="$PART.segments"
    mkdir -p "$SEGMENT_DIR"
    segment_size=$(( (EXPECTED_SIZE + SEGMENTS - 1) / SEGMENTS ))
    for ((segment=0; segment<SEGMENTS; ++segment)); do
        start=$(( segment * segment_size ))
        ((start < EXPECTED_SIZE)) || break
        end=$(( start + segment_size - 1 ))
        ((end < EXPECTED_SIZE)) || end=$(( EXPECTED_SIZE - 1 ))
        segment_path="$SEGMENT_DIR/$(printf '%06d.part' "$segment")"
        expected_segment_size=$(( end - start + 1 ))
        actual_segment_size=0
        if [ -f "$segment_path" ]; then
            actual_segment_size="$(stat -c %s "$segment_path")"
            if ((actual_segment_size > expected_segment_size)); then
                mv "$segment_path" "$segment_path.invalid.$(date +%Y%m%d%H%M%S)"
                actual_segment_size=0
            elif ((actual_segment_size == expected_segment_size)); then
                continue
            fi
        fi
        # Request only the missing suffix and append it.  A failed curl leaves
        # its received bytes in place, so the next run resumes that segment.
        range_start=$(( start + actual_segment_size ))
        curl_fetch "${CURL_COMMON[@]}" --range "$range_start-$end" "$URL" >> "$segment_path"
        actual_segment_size="$(stat -c %s "$segment_path")"
        [ "$actual_segment_size" = "$expected_segment_size" ] || {
            echo "fetch_bonsai_27b_gguf.sh: segment $segment size mismatch: expected $expected_segment_size, got $actual_segment_size" >&2
            exit 1
        }
    done
    ASSEMBLED="$PART.assemble.$$"
    cleanup_assembled() { rm -f "$ASSEMBLED"; }
    trap cleanup_assembled EXIT HUP INT TERM
    : > "$ASSEMBLED"
    for ((segment=0; segment<SEGMENTS; ++segment)); do
        segment_path="$SEGMENT_DIR/$(printf '%06d.part' "$segment")"
        [ -f "$segment_path" ] || continue
        dd if="$segment_path" of="$ASSEMBLED" oflag=append conv=notrunc status=none
    done
    mv -f "$ASSEMBLED" "$PART"
    trap - EXIT HUP INT TERM
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
# PART and DEST are deliberately adjacent, making this a same-filesystem
# atomic rename.  Readers can observe only the old verified file or the new
# verified file—never a partial download.
mv -f "$PART" "$DEST"
if ((SEGMENTS > 1)); then
    rm -rf -- "$PART.segments"
fi
echo "[bonsai-27b] model verified: $DEST"
