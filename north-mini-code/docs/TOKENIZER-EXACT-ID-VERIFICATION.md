# Byte-exact tokenizer verification (nmc vs llama.cpp)

nmc's tokenizer (`src/nmc/tokenizer.py`) ports the GGUF's `gpt2` byte-level BPE with the
`cohere2moe` → `LLAMA_VOCAB_PRE_TYPE_TINY_AYA` pre-tokenizer. This note records how its
tokenization was confirmed **byte-exact against llama.cpp itself** (exact token IDs, not just
counts). `tools/check_tok_parity.py` (token *counts* vs live Ollama) is the quick gate;
`prompt_eval_count` is a weaker oracle than exact IDs, so this is the authoritative check.

## The RAM-safe oracle: `llama-tokenize --vocab-only`

llama.cpp's `tools/tokenize` loads with `model_params.vocab_only = true` — it reads **only the
vocab**, never the weights. So it tokenizes the 18 GB nmc GGUF (or the 80 GB mistral one) with a
few hundred MB of RAM, no weight load, no freeze risk on a small host. It reads the root-owned
ollama blob, so run it under sudo.

## Why the stock reference build can't load nmc (and the 3-line fix)

The nmc GGUF declares `general.architecture = cohere2moe`. An unpatched PrismML llama.cpp checkout registers
only `cohere2`, so `load_arch` throws
`unknown model architecture: 'cohere2moe'` — even in vocab-only mode. It also maps only the
pre-string `"tiny_aya"` (not `"cohere2moe"`) to `TINY_AYA`.

For **vocab-only** this is trivial to fix because `load_hparams` returns early right after
`general.name` (`llama-model.cpp`: `if (hparams.vocab_only ...) return;`) and `load_tensors` is
skipped entirely — so no hparam/tensor/graph plumbing is needed. The patch
([`tools/llama-cpp-cohere2moe-vocab.patch`](../tools/llama-cpp-cohere2moe-vocab.patch)) only:

1. registers `LLM_ARCH_COHERE2MOE` (enum in `llama-arch.h` + name `"cohere2moe"` in `llama-arch.cpp`), and
2. aliases the `"cohere2moe"` pre-string to `LLAMA_VOCAB_PRE_TYPE_TINY_AYA` in `llama-vocab.cpp`.

It is purely additive — it does not touch any existing arch or pre-type, so every other model
(e.g. mistral-medium's `mistral3` / `pre=default`) tokenizes identically.

### Apply + build

```sh
LLAMA_CPP_DIR="$(pwd)/../PrismML-llama.cpp"
cd "$LLAMA_CPP_DIR"
git checkout -b cohere2moe-vocab-tokenize          # isolate; `git checkout prism` restores pristine source
git apply ../integer_inference_engine/north-mini-code/tools/llama-cpp-cohere2moe-vocab.patch
cmake --build build --target llama-tokenize -j"$(nproc)"   # build is GGML_CUDA=OFF -> fast, CPU-only
```

### Run (sudo, to read the ollama blob)

Resolve the model blob from the ollama manifest, then dump IDs:

```sh
BLOB=$(python3 - <<'PY'
import json,glob
m=json.load(open(glob.glob('/usr/share/ollama/.ollama/models/manifests/*/*/north-mini-code-1.0/latest')[0]))
print(next('/usr/share/ollama/.ollama/models/blobs/'+l['digest'].replace('sha256:','sha256-')
           for l in m['layers'] if l['mediaType'].endswith('image.model')))
PY
)
BIN="$LLAMA_CPP_DIR/build/bin/llama-tokenize"
sudo "$BIN" -m "$BLOB" --no-escape --no-bos --ids --log-disable -p 'The capital of France is'
```

`--no-bos` gives the core stream; omit it to see the default `bos_id=2` prepended (matching
nmc's `encode(...)` default `add_bos=True`).

## Result (2026-07-01)

**5/5 exact-ID parity** between `nmc.Tokenizer.encode` and the patched `llama-tokenize`, both with
and without BOS, on the parity prompt set — plus the source-level cross-check (nmc's `_DIGIT_RX` /
`_MAIN_RX` are char-identical to llama.cpp's applied `TINY_AYA` `regex_exprs`, and `_pretokenize`
reproduces the true `unicode_regex_split` cascade piece-for-piece; see
`tests/test_tokenizer.py::test_pretokenize_tiny_aya_golden`). The tokenizer is confirmed correct.
