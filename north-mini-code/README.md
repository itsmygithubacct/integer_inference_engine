# north-mini-code — deterministic integer engine (WIP)

A work-in-progress all-integer / fixed-point engine for **`north-mini-code-1.0`** (`cohere2moe`, 30.5B MoE,
Q4_K_M), the second target after Bonsai-8B. Goal: byte-identical output across CPU/GPU/threads/batch + verifiable
receipts, preserving the Q4_K model's quality (faithful integer rebuild, not a 1-bit re-quantization).

**Design + plan:** [`north-mini-code-integer-engine.md`](../../../research/north-mini-code-integer-engine.md)
(scoping, full architecture, milestones — under `~/research/`). Generalized design:
[`integer_engine.md`](../integer_engine.md). Reuses the proven Bonsai engine at
`~/integer_inference_engine/bonsai`.

## Status
- ✅ **Stage 1 — import/inspect**: full GGUF metadata pinned (arch, SWA pattern, MoE, norm, codec types).
- ✅ **Stage 2 — Q4_K/Q6_K integer codec** (`src/nmc/qk_codec.py`): float reference dequant (ports llama.cpp) +
  integer fixed-point dequant (scalar + vectorized) + integer GEMM. **Gated:** byte-exact integer parity
  (scalar==vectorized, 200 seeds each) and float fidelity < 1e-6 at frac=24, plus contraction-order invariance.
- ✅ **Stage 3 — cohere2 dense path** (`src/nmc/cohere2.py`): the Cohere **parallel block** (one RMSNorm →
  attn + FFN summed into the residual), GQA full-causal attention with NeoX RoPE (θ=50000), SwiGLU FFN, and the
  **tied-embedding** head — all integer, composing the Stage-2 codec + reused Bonsai primitives. **Gated:**
  fidelity vs float ref (block 1.8e-3, logits 1.9e-3, **argmax agreement 8/8**), integer determinism, codec
  integration. 422 tests green.
- ✅ **Stage 4 — sliding-window / full attention interleave** (`src/nmc/cohere2.py`): windowed causal mask +
  the per-layer pattern (`is_full_layer`/`NORTH_SWA_PATTERN` — full every 4th layer, SWA(4096) elsewhere,
  matching the GGUF). **Gated:** mask semantics, SWA==full when window≥seq (byte-identical), SWA changes the
  result when window<seq, int≈float fidelity, determinism. 433 tests green.
- ✅ **Stage 5 — MoE layer** (`src/nmc/cohere2.py`): integer router → **sigmoid** gating → **top-k** (lowest-
  index tie-break) → per-expert SwiGLU → weighted combine (`expert_weights_norm=false`), in the parallel block.
  **Gated:** router top-k agreement int==float (8/8 seeds, all tokens), tie-break, MoE fidelity (4.8e-3), full
  block argmax 8/8, determinism. 457 tests green.
- ✅ **Stage 6a — GGUF loader + real-weight codec validation** (`src/nmc/gguf.py`, `tools/verify_real_tensors.py`):
  parses the real model (442 tensors, cohere2moe), shapes match the Stage-1 spec, integer dequant **exact
  (0.00 rel err at frac=24)** on real Q4_K/Q6_K/F32 tensors.
- ✅ **Stage 6b — tokenizer + real-weight forward**:
  - **gpt2-BPE tokenizer** (`src/nmc/tokenizer.py`): byte-level BPE over the real vocab (262144) + merges
    (254739), using llama.cpp's `cohere2moe → TINY_AYA` pre-tokenizer regex. Lossless round-trip + **exact
    token-count parity with live Ollama 12/12** (plain text, indented code, thousands numbers, unicode, URLs,
    case-mixed). `tools/check_tok_parity.py`.
  - **Real-weight forward** (`src/nmc/gguf.py` + vectorized dequant; `tools/real_forward.py`): the full 49-layer
    cohere2moe graph (dense block 0 + 48 MoE, per-layer SWA/full, MoE top-8) on the actual Q4_K_M weights
    (per-layer dequant). **Predicts the same next token as Ollama**: "The capital of France is" → ` Paris`
    (id 12071) == Ollama ` Paris`. Architecture validated end-to-end on the real model. 469 tests green.
- ✅ **Stage 7 — CPU integer kernel** (`tools/nmc_qk_kernel.c`, `src/nmc/qk_native.py`): fused Q4_K/Q6_K dequant
  + fixed-point matmul (`__int128` accumulate), **byte-identical** to the numpy oracle (11 parity tests) and
  **128× faster** on a head-sized matmul. Kernel-backed **all-integer forward** (`tools/real_forward_int.py`) on
  the real weights runs in **16.6 s** (vs 22 min float) and **predicts the same next token as Ollama**: "The
  capital of France is" → ` Paris`. **480 tests green.**
- ✅ **Stage 8 — GPU CUDA kernel + receipts/CLI**: the CUDA port of the fused Q4_K/Q6_K dequant +
  fixed-point matmul (`tools/nmc_qk_cuda.cu`, `src/nmc/qk_cuda.py`, parity-tested in `tests/test_qk_cuda.py`),
  and the deterministic engine + receipt stack (`src/nmc/engine.py`, `src/nmc/receipts_runtime.py`,
  `tools/emit_receipt.py`, `tools/nmc_cli.py`, the `north-mini-code-cli` launcher) — secp256k1-signed
  byte-exact receipts reusing the Bonsai stack (`requirements_receipts.txt`, `tests/test_receipts.py`), with
  a dry-run-logged on-chain 3rd-entry artifact that broadcasts via the bonsai-notary wallet behind the
  two-key interlock (`--broadcast --confirm`).
- ✅ **Stage 9 — fidelity eval harness** (`src/nmc/fidelity.py` + `tools/fidelity_eval.py`): a pure,
  unit-tested metric **library** (teacher-forced **perplexity**, top-k next-token agreement vs Ollama,
  free-running char-divergence) with a thin CLI over it. Perplexity de-scales the integer logits by `2**fa`
  (a `linear` leaves activations at `2**fa`, `logit_scale=1`) into nat units, so it is **comparable across an
  `NMC_FA` sweep** — the CLI's `--fa-sweep` (re-execs per fa), configurable corpus (`--prompts-file`), metric
  selection, and JSON output make that sweep one command. Metrics + runners are gated **offline** on synthetic
  logits + a stub engine (`tests/test_fidelity.py`, 14 tests) — no model needed. **522 tests green.**
  **Measured on the real 30.5B model** (RTX 3090, `NMC_BACKEND=cuda`, 24-prompt corpus, deploy driver
  `~/.local/trinote/deploy/nmc_fidelity.py`): next-token agreement vs Ollama **top-1 92%, top-3/5/10 = 100%**
  (the integer engine ranks the float model's token first 92% of the time, in its top-3 always); corpus
  **perplexity 18.98 @ fa=16**. The `fa` de-scaling proves its worth in the sweep — **fa=12 → ppl ≈ 3912**
  (unusable) vs **fa=16 → ppl ≈ 32** on the same 8-prompt subset, i.e. 16 activation fixed-point bits are
  necessary; fa=12 is far too coarse.
- ⏳ Next: fix the **`cuda-resident` MoE path** (crashes with a `None` expert handle in
  `qk_cuda.moe_ffn_batched` — the eval fell back to the slower non-resident `cuda` backend); then a broader
  code-eval corpus + the free-running metric (too slow on the non-resident path); packaging polish.

## Layout
```
src/nmc/qk_codec.py         # Q4_K/Q6_K codec (float ref, integer dequant, GEMM, self-test)
src/nmc/cohere2.py          # cohere2moe graph: parallel block, GQA+NeoX RoPE, SWA/full, SwiGLU, MoE
src/nmc/gguf.py             # GGUF loader + real-weight dequant
src/nmc/tokenizer.py        # gpt2-BPE tokenizer over the real vocab/merges
src/nmc/qk_native.py        # CPU kernel binding (tools/nmc_qk_kernel.c)
src/nmc/qk_cuda.py          # CUDA kernel binding (tools/nmc_qk_cuda.cu)
src/nmc/engine.py           # deterministic decode engine
src/nmc/fidelity.py         # fidelity metrics (perplexity, top-k agreement, divergence) — pure, unit-tested
src/nmc/receipts_runtime.py # secp256k1-signed byte-exact receipts (reuses the Bonsai stack)
tools/                      # kernels (nmc_qk_kernel.c, nmc_qk_cuda.cu), CLI (nmc_cli.py), forward/eval/receipt tools
tests/                      # per-stage gates (codec, cohere2 attn/dense/moe/rope/decode, native, cuda, tokenizer, receipts)
requirements.txt            # numpy  (+ requirements_test.txt: pytest; requirements_receipts.txt: ecdsa)
```

## Run
```bash
cd ~/integer_inference_engine/north-mini-code
uv venv --python 3.12 && uv pip install -r requirements.txt -r requirements_test.txt
PYTHONPATH=src .venv/bin/python -m nmc.qk_codec --frac 24 --n 128   # codec self-test (parity + fidelity)
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q                 # the gates
```

To run the full deterministic engine (one-shot / `json` / `repl`, with `--receipts` and an optional
`--broadcast`/`--confirm` on-chain 3rd entry), use the `north-mini-code-cli` launcher at the engine root
(`~/integer_inference_engine/north-mini-code-cli`); receipts need `requirements_receipts.txt` (`ecdsa`).

**Validating a receipt bundle** — `tools/verify_bundle.py` (offline: signatures + commitments, no model/GPU) and
`tools/replay_receipt.py` (byte-exact re-execution); see [docs/VALIDATING-A-BUNDLE.md](docs/VALIDATING-A-BUNDLE.md)
for the full procedure (including reproducing a bundle on a *different* machine via `nmc_gpu_test.py replay`).

Generated state/secrets, when the engine runs models, will live under `$BONSAI_NOTARY_HOME` (its own subtree),
never in this source tree — same discipline as the Bonsai engine.
