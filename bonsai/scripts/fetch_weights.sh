#!/usr/bin/env bash
# fetch_weights.sh — download the Bonsai notarized model weights from HuggingFace.
#
# Two weight artifacts:
#   Bonsai-8B-Q1_0.gguf                    (~1.16 GB) — DOWNLOADED from HuggingFace ($HF_REPO,
#                                          default prism-ml/Bonsai-8B-gguf; public, Apache-2.0).
#   atlas-notarized-bonsai-8b.safetensors  (~1.5 GB) — NOT published on HF; DETERMINISTICALLY
#                                          CREATED by importing the GGUF (trinote-import-bonsai-gguf;
#                                          engine venv required).
#
# Each is checksum-verified (sha256) against the provenance hashes recorded in
# artifacts/atlas-notarized-bonsai-8b.identity.json. Existing verified files are skipped;
# mismatches are reported and (with --force) re-fetched / re-built.

set -euo pipefail

# ----------------------------------------------------------------------------
# Static metadata (resolved before any argument parsing so --help is side-effect free)
# ----------------------------------------------------------------------------
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Weights download under $BONSAI_NOTARY_HOME (default ~/.local/trinote/models), NOT into the repo tree. A
# verified copy already at the legacy in-repo location is reused (back-compat) so existing installs don't redownload.
NOTARY_HOME="${BONSAI_NOTARY_HOME:-$HOME/.local/trinote}"
MODELS_DIR="${BONSAI_MODELS_DIR:-$NOTARY_HOME/models}"

# HuggingFace source repo for the GGUF (real, public, Apache-2.0). Override with $HF_REPO.
#   e.g. HF_REPO=prism-ml/Bonsai-8B-gguf  HF_REVISION=main  ./scripts/fetch_weights.sh
HF_REPO_DEFAULT="prism-ml/Bonsai-8B-gguf"

# ----------------------------------------------------------------------------
# Usage — printed BEFORE any work; no build, no model load, no network, no broadcast.
# ----------------------------------------------------------------------------
usage() {
  cat <<EOF
${SCRIPT_NAME} — download Bonsai notarized model weights from HuggingFace.

Purpose:
  Fetch the two large (gitignored) weight artifacts into their expected paths
  and sha256-verify them against the recorded provenance hashes:
    models/Bonsai-8B-Q1_0.gguf
    artifacts/model/atlas-notarized-bonsai-8b.safetensors

Usage:
  ${SCRIPT_NAME} [options]

Options:
  -f, --force            Re-download even if a verified file already exists.
      --no-verify        Skip sha256 checksum verification (not recommended).
      --gguf-only        Fetch only the GGUF weights.
      --safetensors-only Fetch only the safetensors weights.
      --dry-run          Print what would be downloaded; do nothing.
  -h, --help             Show this help and exit (no side effects).

Environment variables:
  HF_REPO          HuggingFace repo id            (default: ${HF_REPO_DEFAULT})
  HF_REVISION      Repo revision / branch / tag   (default: pinned commit 48516770…; =main to track tip)
  HF_ENDPOINT      HuggingFace base URL           (default: https://huggingface.co)
  HF_TOKEN         Bearer token for private repos (default: unset)
  GGUF_SHA256      Expected sha256 of the GGUF    (default: from identity record)
  SAFETENSORS_SHA256
                   Expected sha256 of the safetensors (default: from identity record)
  DOWNLOADER       Force 'curl' or 'wget'         (default: auto-detect)

Example:
  HF_REPO=prism-ml/Bonsai-8B-gguf ${SCRIPT_NAME}
  HF_REPO=my-org/Bonsai-8B HF_TOKEN=hf_xxx ${SCRIPT_NAME} --force
EOF
}

# ----------------------------------------------------------------------------
# Argument parsing — handle -h/--help first, before reading env or touching disk.
# ----------------------------------------------------------------------------
FORCE=0
VERIFY=1
DO_GGUF=1
DO_SAFETENSORS=1
DRY_RUN=0

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -f|--force)
      FORCE=1
      shift
      ;;
    --no-verify)
      VERIFY=0
      shift
      ;;
    --gguf-only)
      DO_SAFETENSORS=0
      shift
      ;;
    --safetensors-only)
      DO_GGUF=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "${SCRIPT_NAME}: unknown option: $1" >&2
      echo "Try '${SCRIPT_NAME} --help'." >&2
      exit 2
      ;;
    *)
      echo "${SCRIPT_NAME}: unexpected argument: $1" >&2
      echo "Try '${SCRIPT_NAME} --help'." >&2
      exit 2
      ;;
  esac
done

# ----------------------------------------------------------------------------
# Configuration (read AFTER --help so --help has no dependency on the environment)
# ----------------------------------------------------------------------------
HF_REPO="${HF_REPO:-$HF_REPO_DEFAULT}"
# Pinned to an IMMUTABLE commit for reproducible downloads (the GGUF's CONTENT is also pinned by
# GGUF_SHA256 below). Override with HF_REVISION=main to track the branch tip.
HF_REVISION="${HF_REVISION:-48516770dd04643643e9f9019a2a349cf26c5dbd}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_TOKEN="${HF_TOKEN:-}"
DOWNLOADER="${DOWNLOADER:-}"

# Expected checksums default to the provenance hashes baked into the identity record.
#   ggufSha256 (weightProvenance) / modelHash (safetensors weightsRoot).
_GGUF_ENV_SET="${GGUF_SHA256:+1}"
_SAFE_ENV_SET="${SAFETENSORS_SHA256:+1}"
GGUF_SHA256="${GGUF_SHA256:-284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54}"
SAFETENSORS_SHA256="${SAFETENSORS_SHA256:-e5ae7bd10b103b8139f1c37e1c1d353878d4f55d8451d0b6b39aaac2943658e1}"

# Drift guard: the identity record is the single source of truth. modelHash IS
# sha256(safetensors) and weightProvenance.ggufSha256 IS sha256(gguf), so the
# pinned defaults above MUST equal the deployed identity record. This guard fails
# closed on any mismatch (it would have caught the open_lm->trinote re-import that
# left these literals stale). Explicit env overrides are exempt.
_IDENTITY_JSON="$ROOT/artifacts/atlas-notarized-bonsai-8b.identity.json"
if [[ -r "$_IDENTITY_JSON" ]] && command -v python3 >/dev/null 2>&1; then
  _ID_MH="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("modelHash") or "")' "$_IDENTITY_JSON" 2>/dev/null || true)"
  _ID_GG="$(python3 -c 'import json,sys;print((json.load(open(sys.argv[1])).get("weightProvenance") or {}).get("ggufSha256") or "")' "$_IDENTITY_JSON" 2>/dev/null || true)"
  if [[ -z "$_SAFE_ENV_SET" && -n "$_ID_MH" && "$SAFETENSORS_SHA256" != "$_ID_MH" ]]; then
    printf '[%s] ERROR: pinned SAFETENSORS_SHA256 (%s) != identity modelHash (%s); re-pin to the identity record\n' "${SCRIPT_NAME:-fetch_weights}" "$SAFETENSORS_SHA256" "$_ID_MH" >&2; exit 1
  fi
  if [[ -z "$_GGUF_ENV_SET" && -n "$_ID_GG" && "$GGUF_SHA256" != "$_ID_GG" ]]; then
    printf '[%s] ERROR: pinned GGUF_SHA256 (%s) != identity ggufSha256 (%s); re-pin to the identity record\n' "${SCRIPT_NAME:-fetch_weights}" "$GGUF_SHA256" "$_ID_GG" >&2; exit 1
  fi
fi

# Logical artifacts: "<relative-dest-path>|<sha256>|<remote-filename>"
GGUF_REL="models/Bonsai-8B-Q1_0.gguf"
GGUF_REMOTE="Bonsai-8B-Q1_0.gguf"
SAFETENSORS_REL="artifacts/model/atlas-notarized-bonsai-8b.safetensors"
SAFETENSORS_REMOTE="atlas-notarized-bonsai-8b.safetensors"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
log()  { printf '[%s] %s\n' "${SCRIPT_NAME}" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "${SCRIPT_NAME}" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "${SCRIPT_NAME}" "$*" >&2; exit 1; }

# Pick a downloader once, honoring an explicit DOWNLOADER override.
detect_downloader() {
  case "$DOWNLOADER" in
    curl)
      command -v curl >/dev/null 2>&1 || die "DOWNLOADER=curl but curl not found on PATH."
      ;;
    wget)
      command -v wget >/dev/null 2>&1 || die "DOWNLOADER=wget but wget not found on PATH."
      ;;
    "")
      if command -v curl >/dev/null 2>&1; then
        DOWNLOADER="curl"
      elif command -v wget >/dev/null 2>&1; then
        DOWNLOADER="wget"
      else
        die "Neither curl nor wget found on PATH. Install one, or set DOWNLOADER."
      fi
      ;;
    *)
      die "DOWNLOADER must be 'curl' or 'wget' (got: ${DOWNLOADER})."
      ;;
  esac
}

# Pick a sha256 tool (checksum-verify hook). Echoes the digest of $1 to stdout.
sha256_of() {
  local f="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$f" | awk '{print $1}'
  else
    die "No sha256 tool found (need sha256sum or shasum). Use --no-verify to bypass."
  fi
}

# Verify $1 against expected sha256 $2. Returns 0 on match, 1 on mismatch.
verify_checksum() {
  local file="$1" expected="$2" actual
  if ((! VERIFY)); then
    warn "checksum verification disabled (--no-verify) for $(basename "$file")"
    return 0
  fi
  if [[ -z "$expected" ]]; then
    warn "no expected sha256 for $(basename "$file"); skipping verification"
    return 0
  fi
  actual="$(sha256_of "$file")"
  if [[ "$actual" == "$expected" ]]; then
    log "checksum OK: $(basename "$file") (sha256=${actual})"
    return 0
  fi
  warn "checksum MISMATCH for $(basename "$file")"
  warn "  expected: ${expected}"
  warn "  actual:   ${actual}"
  return 1
}

# Download $url to $dest atomically (via a .part temp file).
download() {
  local url="$1" dest="$2" tmp
  tmp="${dest}.part"
  rm -f "$tmp"
  log "downloading: ${url}"
  log "         to: ${dest}"
  if [[ "$DOWNLOADER" == "curl" ]]; then
    local -a auth=()
    [[ -n "$HF_TOKEN" ]] && auth=(-H "Authorization: Bearer ${HF_TOKEN}")
    curl --fail --location --retry 3 --retry-delay 2 --continue-at - \
      "${auth[@]}" --output "$tmp" "$url" \
      || { rm -f "$tmp"; die "curl failed for ${url}"; }
  else
    local -a auth=()
    [[ -n "$HF_TOKEN" ]] && auth=(--header "Authorization: Bearer ${HF_TOKEN}")
    wget --tries=3 --continue "${auth[@]}" -O "$tmp" "$url" \
      || { rm -f "$tmp"; die "wget failed for ${url}"; }
  fi
  mv -f "$tmp" "$dest"
}

# Fetch one logical artifact: relpath, expected sha256, remote filename.
fetch_artifact() {
  local relpath="$1" expected="$2" remote="$3"
  local dest="${MODELS_DIR}/$(basename "$relpath")"
  local legacy="${ROOT}/${relpath}"
  local url="${HF_ENDPOINT}/${HF_REPO}/resolve/${HF_REVISION}/${remote}"

  if ((DRY_RUN)); then
    log "[dry-run] would fetch ${url}"
    log "[dry-run]          into ${dest} (expected sha256=${expected:-<none>})"
    return 0
  fi

  # Skip if a verified copy already exists in the new OR the legacy in-repo location (back-compat).
  if ((FORCE == 0)); then
    for cand in "$dest" "$legacy"; do
      if [[ -f "$cand" ]] && verify_checksum "$cand" "$expected"; then
        log "already present and verified: ${relpath} at ${cand} (skipping; use --force to redownload)"
        return 0
      fi
    done
  fi

  mkdir -p "$(dirname "$dest")"
  download "$url" "$dest"

  if ! verify_checksum "$dest" "$expected"; then
    die "downloaded ${relpath} failed checksum verification (file left at ${dest} for inspection)"
  fi
  log "fetched: ${relpath}"
}

# Build the atlas-notarized safetensors by importing the verified GGUF. It is NOT published on HF — it
# is the DETERMINISTIC int-ref import of the GGUF (trinote-import-bonsai-gguf), reproducible byte-for-byte
# from the GGUF on a given host. Requires this engine's venv (numpy/safetensors).
import_safetensors() {
  local fname="atlas-notarized-bonsai-8b.safetensors"
  local dest="${MODELS_DIR}/${fname}"
  local gguf="${MODELS_DIR}/Bonsai-8B-Q1_0.gguf"

  if ((DRY_RUN)); then
    log "[dry-run] would import ${gguf} -> ${dest} (deterministic GGUF import; NOT an HF download)"
    return 0
  fi
  if ((FORCE == 0)) && [[ -e "$dest" ]] && verify_checksum "$dest" "$SAFETENSORS_SHA256"; then
    log "already present, verified: ${fname} (skip; --force to rebuild)"
    return 0
  fi
  [[ -e "$gguf" ]] || die "cannot build ${fname}: GGUF missing at ${gguf} (fetch the GGUF first)"
  local py="$ROOT/.venv/bin/python"; [[ -x "$py" ]] || py="python3"
  log "importing GGUF -> ${fname} (deterministic creation step)"
  mkdir -p "$(dirname "$dest")"
  BONSAI_NOTARY_HOME="$NOTARY_HOME" PYTHONPATH="$ROOT/src" "$py" \
    -m trinote.cli.import_bonsai_gguf_cli --gguf "$gguf" --out "$dest" \
    || die "GGUF -> safetensors import failed (need the engine venv: ${ROOT}/.venv)"
  verify_checksum "$dest" "$SAFETENSORS_SHA256" \
    || die "imported ${fname} failed checksum verification (left at ${dest} for inspection)"
  log "built (imported): ${fname}"
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
main() {
  if ((DO_GGUF)) && ((! DRY_RUN)); then detect_downloader; fi

  log "GGUF repo: ${HF_REPO} @ ${HF_REVISION}  (endpoint ${HF_ENDPOINT})"
  ((DO_GGUF)) && ((! DRY_RUN)) && log "via:       ${DOWNLOADER}"

  # GGUF is the only HuggingFace artifact; the safetensors is built from it by import (above).
  ((DO_GGUF))        && fetch_artifact "$GGUF_REL" "$GGUF_SHA256" "$GGUF_REMOTE"
  ((DO_SAFETENSORS)) && import_safetensors

  ((DRY_RUN)) && { log "dry-run complete."; return 0; }
  log "done."
}

main
