# north-mini-code — deterministic integer engine (WIP)

A work-in-progress all-integer / fixed-point engine for **`north-mini-code-1.0`** (`cohere2moe`, 30.5B MoE,
Q4_K_M), the second target after Bonsai-8B. Goal: byte-identical output across CPU/GPU/threads/batch + verifiable
receipts, preserving the Q4_K model's quality (faithful integer rebuild, not a 1-bit re-quantization).

The generalized design is documented in [`integer_engine.md`](../integer_engine.md); private scoping and
research notes intentionally remain outside the published repository. This port reuses the proven Bonsai
engine under [`../bonsai`](../bonsai).

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
- ✅ **Resident CUDA runtime hardening and boundary reduction**: growable fail-loud weight registration,
  persistent MoE workspaces, one-call batched expert prefill, exact guarded batched-DP4A, and same-input
  grouped resident projections. Q/K/V now share one activation upload and one result download; the leading
  dense gate/up pair uses the same exact grouped ABI. After batched prefill, a request-scoped device bank
  imports K/V once and keeps decode Q/K/V, exact interleaved RoPE, transactional K/V append, fixed-point
  causal/SWA attention scratch, and O projection on-device for the full attention sublayer. ABI v5 keeps the
  residual width distinct from the wider query projection used by the real model (2048 versus 4096), rejects
  stale per-host libraries and exposes
  opt-in cold/warm registration, H2D/D2H, allocation, Python/native crossing, attention/routing/MoE, and
  Q/K/V/O projection telemetry. Synthetic GPU gates compare every grouped/DP4A result byte-for-byte with
  the unchanged int128 path, including explicit envelope fallback.
- ✅ **Bounded resident MoE preprocessing (opt-in)**: `qk_register_i64` and `qk_rmsnorm_router` execute exact
  RMSNorm, the dense router, stable low-index top-k, and fixed-point sigmoid on-device. The signed-i128 and
  int64 envelopes fail closed to the arbitrary-precision host oracle, while expert Q4_K/Q6_K slices remain
  lazily registered only after selection. For the real 2048-lane/128-expert shape, guarded 256-thread CUDA
  blocks prove a complete uint64/int64 arithmetic envelope before parallel RMSNorm and router reductions;
  any unproven row or logit is recomputed by the byte-exact `__int128` kernel in the same stream. The 48-layer
  dense gain/router budget is exactly **101,449,728 bytes (96.75 MiB)**. Enable with
  `NMC_RESIDENT_PREPROCESS=1`; see
  [`docs/RESIDENT-LAYER-EXECUTOR.md`](docs/RESIDENT-LAYER-EXECUTOR.md) for the boundary and promotion gates.
  A local real-shape proxy measured **0.347823 ms** per fast call versus **2.632553 ms** for a deliberately
  forced exact fallback (**7.57×**); this is a fallback proxy, not a historical serial-library baseline.
- ✅ **Request-scoped cold-route continuation (ABI v5)**: the attention bank retains residual, normalized `h`,
  route IDs/gates, attention, selected-expert MoE intermediates, and the next residual. Only device-compacted
  unbound expert IDs are exposed; lazy gate/up/down handle binding resumes the prepared layer, while warm
  routes publish no IDs or weights. A two-layer synthetic CUDA chain is byte-exact and proves retained
  continuation plus fail-closed unbound handling. Expert weights remain route-lazy rather than preloaded.
- ✅ **Explicit dense-to-48-MoE token orchestrator (opt-in)**: `ResidentMoeTokenExecutor` preflights bounded
  per-layer metadata, binds known handles without transfer, lazily loads only device-reported cold IDs, and
  poisons the request on any partial-state failure. `Engine.resident_decode_token` composes leading dense block
  0 with exactly 48 retained MoE layers when `resident_layer_executor=True`. It is intentionally not selected
  by `Engine.generate`. Synthetic multi-layer parity is exact; a second fixed-route token performs zero new
  CUDA allocations and keeps `workspace_bytes()` constant.
- ✅ **Retained-layer hardware promotion gate (2026-07-22)**: the isolated real-model run passed exact hidden,
  full-logit, and greedy-token parity plus all seven verdict checks. Established throughput was
  **8.4294542598 tok/s** and retained throughput was **11.7969858823 tok/s**, a **1.3994958059×** ratio; combined
  peak memory was **7326 MiB** (**0.2982413** of the device). The two-key signed receipt replay passed locally
  with chain publication disabled. Evidence binds source snapshot
  `5c7f866c2c52361f8011511a36e26de101839c5b6a31bd1077e893125c1ecff0` (196 entries, base
  `d1cd09049b8ac153e2028985fef1eae32611a900`). `Engine.generate` selection remains an explicit product choice,
  not an unresolved hardware gate.

## Layout
```
src/nmc/qk_codec.py         # Q4_K/Q6_K codec (float ref, integer dequant, GEMM, self-test)
src/nmc/cohere2.py          # cohere2moe graph: parallel block, GQA+NeoX RoPE, SWA/full, SwiGLU, MoE
src/nmc/gguf.py             # GGUF loader + real-weight dequant
src/nmc/tokenizer.py        # gpt2-BPE tokenizer over the real vocab/merges
src/nmc/qk_native.py        # CPU kernel binding (tools/nmc_qk_kernel.c)
src/nmc/qk_cuda.py          # CUDA kernel binding (tools/nmc_qk_cuda.cu)
src/nmc/engine.py           # deterministic decode engine
src/nmc/profiling.py        # opt-in structured cold/warm phase telemetry
src/nmc/fidelity.py         # fidelity metrics (perplexity, top-k agreement, divergence) — pure, unit-tested
src/nmc/receipts_runtime.py # secp256k1-signed byte-exact receipts (reuses the Bonsai stack)
tools/                      # kernels (nmc_qk_kernel.c, nmc_qk_cuda.cu), CLI (nmc_cli.py), forward/eval/receipt tools
tests/                      # per-stage gates (codec, cohere2 attn/dense/moe/rope/decode, native, cuda, tokenizer, receipts)
requirements.txt            # numpy  (+ requirements_test.txt: pytest; requirements_receipts.txt: ecdsa)
```

## Run
```bash
cd integer_inference_engine/north-mini-code
uv venv --python 3.12 && uv pip install -r requirements.txt -r requirements_test.txt
PYTHONPATH=src .venv/bin/python -m nmc.qk_codec --frac 24 --n 128   # codec self-test (parity + fidelity)
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q                 # the gates
```

Build the exact per-host CUDA ABI and capture a fail-loud resident profile with:

```bash
tools/build_nmc_cuda.sh
PYTHONPATH=src NMC_BACKEND=cuda-resident .venv/bin/python tools/profile_nmc.py MODEL.gguf \
  --prompt "The capital of France is" --new-tokens 4
```

To reproduce the isolated real-model parity, memory, allocation, receipt, and
no-regression gate for the explicit retained 1+48 layer executor, run:

```bash
PYTHONPATH=src NMC_BACKEND=cuda-resident .venv/bin/python tools/gate_resident_layers.py MODEL.gguf \
  --tokenizer TOKENIZER_DIR --new-tokens 8 --expected-model-sha256 MODEL_SHA256 \
  --output resident-layer-gate.json
```

The 2026-07-22 pass measured **8.4294542598 tok/s** for the established path and
**11.7969858823 tok/s** for the retained path (**1.3994958059×**), with a
combined sampled peak of **7326 MiB / 0.2982413**. It bound the exact 196-entry
source snapshot above, matched every hidden/logit/token state, passed all seven
verdict checks, and completed signed local receipt replay without a chain
broadcast. Repo-external evidence in the operator-local deploy results
directory is named
`20260722-195649_source-snapshot.json`,
`20260722-195649_resident-layer-gate.json`, and
`20260722-195649_resident-layer-gate-evidence.tar.gz`.

The gate warms the established route set before measuring either path, checks
full hidden/logit arrays and greedy tokens exactly, requires zero allocation
growth on later retained transitions, samples combined GPU memory with
`nvidia-smi`, and atomically emits private JSON evidence. The warm-up itself
uses the public `Engine.generate` path. A successful gate also emits a local
two-key signed receipt bundle and verifies that both paths reproduce its signed
token commitments; it explicitly disables all chain and dry-run-log publish
paths.

The receipt implementation is shared with Bonsai. Remote layouts containing
only this directory must install `requirements_receipts.txt` and point
`NMC_BONSAI_SRC` at a matching deployed Bonsai `src/` tree before invoking the
gate:

```bash
python -m pip install -r requirements_receipts.txt
export NMC_BONSAI_SRC=/absolute/path/to/bonsai/src
```

The private mode-0600 report contains public signer keys and local artifact
paths only; signing-key paths and private material are never emitted. Resident
GPU state is released after the model hash is bound and before signing starts.

Profiling is disabled during ordinary inference. Setting `NMC_PROFILE=1` or passing `profile=True` to
`nmc.engine.Engine` enables it; `Engine.profile_snapshot()` returns separate cold and warm Python phases plus
native transfer/allocation/projection counters. The profiling CLI requires a resident CUDA backend unless
`--allow-fallback` is explicitly supplied, so an unavailable or stale GPU runtime cannot be mistaken for a
GPU result.

To run the full deterministic engine (one-shot / `json` / `repl`, with `--receipts` and an optional
`--broadcast`/`--confirm` on-chain 3rd entry), use the `north-mini-code-cli` launcher at the engine root
(`../north-mini-code-cli` from this directory); receipts need `requirements_receipts.txt` (`ecdsa`).

**Validating a receipt bundle** — `tools/verify_bundle.py` (offline: signatures + commitments, no model/GPU) and
`tools/replay_receipt.py` (byte-exact re-execution); see [docs/VALIDATING-A-BUNDLE.md](docs/VALIDATING-A-BUNDLE.md)
for the full procedure (including reproducing a bundle on a *different* machine via `nmc_gpu_test.py replay`).

Generated state/secrets, when the engine runs models, live OUT of this source tree under
`~/.local/integer_inference_engine/north-mini-code` (see `src/nmc/receipts_runtime.py::STATE_HOME`) — same
out-of-tree discipline as the Bonsai engine. Note: unlike Bonsai, north-mini-code does **not** currently read
`$BONSAI_NOTARY_HOME`, so setting that variable relocates Bonsai's keys but not north-mini-code's secp256k1
signing keys.
