#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOTARY_HOME="${BONSAI_NOTARY_HOME:-${HOME:?HOME is required}/.local/trinote}"
SOURCE_TREE="${PRISM_SOURCE_TREE:-$NOTARY_HOME/vendor/llama.cpp-qwen35-source}"
RUNTIME_BIN="${PRISM_RUNTIME_BIN:-$NOTARY_HOME/vendor/llama.cpp-bonsai27/prism-b9591-62061f9/bin}"
OUT="${PRISM_TEACHER_HARNESS:-$ROOT/tools/prismml_teacher_forced}"
TOKENIZER_OUT="${PRISM_TOKENIZER_SERVER:-$ROOT/tools/prismml_tokenizer_server}"

if [[ ! -d "$SOURCE_TREE/.git" && ! -f "$SOURCE_TREE/.git" ]]; then
  echo "error: pinned llama.cpp source checkout not found: $SOURCE_TREE" >&2
  exit 2
fi
if [[ ! -f "$RUNTIME_BIN/libllama.so" ]]; then
  echo "error: pinned PrismML libllama runtime not found: $RUNTIME_BIN" >&2
  exit 2
fi

source_revision="$(git -C "$SOURCE_TREE" rev-parse HEAD)"
runtime_revision="$(sed -n 's/^source_revision=//p' "$RUNTIME_BIN/.runtime-release" 2>/dev/null || true)"
if [[ -z "$runtime_revision" ]]; then
  runtime_revision="$(sed -n '2p' "$RUNTIME_BIN/.runtime-release" 2>/dev/null || true)"
fi
if [[ -n "$runtime_revision" && "$runtime_revision" != "$source_revision" ]]; then
  echo "error: header/runtime revision mismatch: source=$source_revision runtime=$runtime_revision" >&2
  exit 2
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
git -C "$SOURCE_TREE" archive HEAD include ggml/include | tar -x -C "$tmp"

"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$tmp/include" -I"$tmp/ggml/include" \
  "$ROOT/tools/prismml_teacher_forced.cpp" \
  -Wl,-rpath,"$RUNTIME_BIN" -Wl,--no-as-needed \
  "$RUNTIME_BIN/libllama.so" -ldl -pthread \
  -o "$OUT"
chmod 0755 "$OUT"
"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$tmp/include" -I"$tmp/ggml/include" \
  "$ROOT/tools/prismml_tokenizer_server.cpp" \
  -Wl,-rpath,"$RUNTIME_BIN" -Wl,--no-as-needed \
  "$RUNTIME_BIN/libllama.so" -ldl -pthread \
  -o "$TOKENIZER_OUT"
chmod 0755 "$TOKENIZER_OUT"
echo "built $OUT against llama.cpp $source_revision"
echo "built $TOKENIZER_OUT against llama.cpp $source_revision"
