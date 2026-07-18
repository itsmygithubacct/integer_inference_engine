#!/usr/bin/env bash
# Install the pinned PrismML llama.cpp CUDA 12.4 runtime used by Bonsai-27B on Linux x86_64.
# Runtime files are kept outside the checkout under $BONSAI_NOTARY_HOME by default.
#
# Usage: scripts/install_bonsai_27b_gguf.sh [--dry-run] [--force]
set -euo pipefail

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next} {exit}' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

RELEASE="prism-b9591-62061f9"
COMMIT="62061f91088281e65071cc38c5f69ee95c39f14e"
ASSET="llama-prism-b9591-62061f9-bin-linux-cuda-12.4-x64.tar.gz"
ARCHIVE_SHA256="67c64046abcf73bf489e27c9ebe7525f5b77c58db9490d1d711efe6e17bf2975"
URL="https://github.com/PrismML-Eng/llama.cpp/releases/download/$RELEASE/$ASSET"
NOTARY_HOME="${BONSAI_NOTARY_HOME:-$HOME/.local/trinote}"
DEST="${BONSAI_27B_BIN_DIR:-$NOTARY_HOME/vendor/llama.cpp-bonsai27/$RELEASE/bin}"
DOWNLOAD_DIR="${BONSAI_DOWNLOADS_DIR:-$NOTARY_HOME/downloads}"
ARCHIVE="$DOWNLOAD_DIR/$ASSET"
DRY_RUN=0
FORCE=0

while (($#)); do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
        -h|--help|help) usage 0 ;;
        *) echo "install_bonsai_27b_gguf.sh: unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

if ((DRY_RUN)); then
    printf '[dry-run] runtime release: %s (commit %s)\n' "$RELEASE" "$COMMIT"
    printf '[dry-run] download: %s\n' "$URL"
    printf '[dry-run] sha256: %s\n' "$ARCHIVE_SHA256"
    printf '[dry-run] install: %s\n' "$DEST"
    exit 0
fi

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    echo "install_bonsai_27b_gguf.sh: this pinned runtime requires Linux x86_64." >&2
    exit 2
fi
command -v nvidia-smi >/dev/null 2>&1 || {
    echo "install_bonsai_27b_gguf.sh: NVIDIA/CUDA was not detected (nvidia-smi is required)." >&2
    exit 2
}
for tool in curl sha256sum tar; do
    command -v "$tool" >/dev/null 2>&1 || { echo "install_bonsai_27b_gguf.sh: $tool is required" >&2; exit 2; }
done

MARKER="$DEST/.runtime-release"
if [ -x "$DEST/llama-cli" ] && [ -f "$MARKER" ] && [ "$(sed -n '1p' "$MARKER")" = "$RELEASE" ]; then
    echo "[bonsai-27b] PrismML runtime already installed: $DEST"
    exit 0
fi
if [ -e "$DEST" ] && (( ! FORCE )); then
    echo "install_bonsai_27b_gguf.sh: refusing to replace existing unrecognized runtime: $DEST" >&2
    echo "Re-run with --force to preserve it as a timestamped backup and install the pinned release." >&2
    exit 1
fi

mkdir -p "$DOWNLOAD_DIR" "$(dirname "$DEST")"
if [ -f "$ARCHIVE" ] && [ "$(sha256sum "$ARCHIVE" | awk '{print $1}')" = "$ARCHIVE_SHA256" ]; then
    echo "[bonsai-27b] using verified cached runtime archive: $ARCHIVE"
else
    PART="$ARCHIVE.part"
    if ((FORCE)) && [ -f "$PART" ]; then
        mv "$PART" "$PART.invalid.$(date +%s)"
    fi
    echo "[bonsai-27b] downloading pinned PrismML CUDA runtime (about 271 MB) ..."
    curl --location --fail --retry 5 --retry-delay 2 --continue-at - --output "$PART" "$URL"
    ACTUAL="$(sha256sum "$PART" | awk '{print $1}')"
    [ "$ACTUAL" = "$ARCHIVE_SHA256" ] || {
        echo "install_bonsai_27b_gguf.sh: runtime checksum mismatch: expected $ARCHIVE_SHA256, got $ACTUAL" >&2
        exit 1
    }
    mv "$PART" "$ARCHIVE"
fi

TMP="$(mktemp -d "$(dirname "$DEST")/.bonsai27-runtime.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
tar -xzf "$ARCHIVE" -C "$TMP" --strip-components=1
[ -x "$TMP/llama-cli" ] || { echo "install_bonsai_27b_gguf.sh: archive did not contain llama-cli" >&2; exit 1; }
printf '%s\n%s\n%s\n' "$RELEASE" "$COMMIT" "$ARCHIVE_SHA256" > "$TMP/.runtime-release"

if [ -e "$DEST" ]; then
    BACKUP="$DEST.previous.$(date +%Y%m%d%H%M%S)"
    mv "$DEST" "$BACKUP"
    echo "[bonsai-27b] preserved previous runtime at $BACKUP"
fi
mv "$TMP" "$DEST"
trap - EXIT

LD_LIBRARY_PATH="$DEST${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$DEST/llama-cli" --version
echo "[bonsai-27b] installed pinned PrismML runtime: $DEST"
