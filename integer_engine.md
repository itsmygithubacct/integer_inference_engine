# A standalone deterministic integer inference engine for BitNet-style LLMs, with Ollama interop

> **Status: design document.** The *core* — a byte-exact, all-integer fixed-point inference engine for 1-bit
> (Q1_0) weights — is **already built and measured** in `bonsai-notarized-bitnet` (the `int-ref@bonsai-qwen3`
> engine: `src/trinote/infer_int/reference_bonsai.py`, `src/trinote/determinism/fixedpoint.py`, the CPU/CUDA
> kernels, receipts, GGUF import). This document describes extracting that core into a **standalone engine**,
> generalizing the weight codec to **BitNet-style ternary** models, and wiring it to **Ollama** for model
> distribution and a drop-in API. Sections are marked **[proven]** (exists today) or **[proposed]** (design).

---

## 1. Thesis

Local LLM inference today (Ollama / llama.cpp) is **fast and convenient but not reproducible**: it runs in
floating point, and float arithmetic is non-associative, batch-variant, atomic-order-dependent, and varies by
BLAS/driver/CPU — so even greedy `T=0` decoding drifts across machines, GPUs, and even batch sizes. You cannot
get the *same bytes* twice across hardware, so you cannot *prove* what a model produced.

A **deterministic integer inference engine** fixes this at the root: do **all** model arithmetic in
**integer fixed-point**. Integer addition is exactly associative (`(a+b)+c == a+(b+c)` with a defined mod-2⁶⁴
wrap), so the result is **byte-identical regardless of thread count, SIMD width, GPU, or batch size** — provided
no integer overflow (which a fail-loud bound guarantees). That single property unlocks **verifiable inference**:
a third party can re-execute the run on a reference CPU and get the same bytes, so a generation can carry a
cryptographic **receipt**.

**BitNet-style LLMs are the natural fit.** Their weights are already 1-bit (`{−1,+1}`, Q1_0) or 1.58-bit ternary
(`{−1,0,+1}`, BitNet b1.58) — there is no float weight to quantize away, and the "matmul" is a **signed integer
sum**, not a floating-point dot product. Integer inference is not a compromise for these models; it is the
honest implementation.

**Ollama is the distribution and UX layer.** It already manages GGUF models (`pull`, `run`, an HTTP API) and is
where users get models. The standalone engine reuses Ollama's **model store** and exposes an **Ollama-compatible
API**, swapping only the *compute backend* — from llama.cpp's float kernels to the deterministic integer engine.

> Net: **`ollama pull` a tiny BitNet model → run it through a deterministic integer engine → get byte-exact,
> reproducible, optionally receipted output**, on CPU (or a GPU as a verified producer). Edge, audit,
> compliance, agent-notarization, and "prove what the model said" use cases that float engines structurally
> cannot serve.

---

## 2. Why these three together

| Piece | What it gives | What it lacks alone |
|---|---|---|
| **Integer engine** | byte-exact reproducibility → verifiable/receiptable inference | a model format + distribution + UX |
| **BitNet-style models** | weights already integer (1-bit / ternary); tiny; CPU-friendly | a deterministic, verifiable runtime |
| **Ollama** | model distribution (`pull`), storage, a familiar API + clients | reproducibility (it is float / llama.cpp) |

The intersection is the product: **verifiable local inference of tiny integer models with a familiar UX.**

---

## 3. The determinism core **[proven]**

This is the load-bearing engineering, already implemented and byte-exact in the reference project.

### 3.1 Everything is integer fixed-point
Activations live as `int64` fixed-point with `frac` fractional bits (e.g. `frac=16`). The only floats permitted
are *scalar* constants computed **host-side once**, correctly rounded, and committed as integers (e.g.
`inv_sqrt_fp = round((1/√head_dim)·2^frac)`). There is **no float in the compute path** and **no float
reduction** — that is the actual guarantee.

### 3.2 Why integer ⇒ byte-exact, on any hardware
Every reduction in a transformer — the weight apply, the attention scores, the softmax denominator, the `@V`
accumulation, the vocab argmax — is an integer sum. Integer `+` is associative and commutative in `ℤ/2⁶⁴ℤ`, so
**any** order (thread, warp, SIMD lane, atomic, tiling, batch) yields the identical sum. The classic sources of
GPU/threaded non-determinism therefore do not apply:

- **float non-associativity** → N/A (integer).
- **batch-invariance failure** (result depends on batch size/tiling) → N/A (integer sums are tiling-independent).
- **atomic-order races** → harmless for integer sums; only `argmax` needs a fixed **lowest-index tie-break**.
- **transcendental/library variance** (`expf`, cuBLAS) → avoided: softmax/SiLU use integer **polynomials**, not
  `expf`; no BLAS.

### 3.3 The integer op set (all byte-exact, with a pure-integer oracle)
- **Weight apply** (the "matmul"): a masked integer signed-sum (see §4).
- **RMSNorm**: 128-bit sum-of-squares (the residual stream is unbounded across layers, so it exceeds int64),
  a bit-exact integer `isqrt`, and **floor-division** (toward −∞, not truncation). Fails loud if even 128 bits
  is insufficient (→ a big-int reference path).
- **Softmax**: integer max-shift, a cubic `2^-f` polynomial, integer normalize (floor-div). No `exp`.
- **SiLU**: the same integer `2^-f` polynomial sigmoid.
- **RoPE**: integer multiply-add + arithmetic floor-shift; host-computed `cos/sin` tables in fixed-point.
- **Argmax / sampling**: lowest-index tie-break; samplers (greedy/temp/top-k/top-p/min-p) are integer and
  reproducible via a counter-based (SHA-256 + Lemire) draw keyed by `(seed, position)`.

### 3.4 The fail-loud overflow contract
Integer associativity holds *only if no value silently wraps unexpectedly*. The engine carries a per-tensor
bound (`max|a|·max|b|·K ≤ 2⁶³−1`) checked **before** each multiply; a breach **raises** rather than wrapping
(a silent wrap would be "wrong-but-still-deterministic" — both producer and verifier would agree on garbage).
The Q1 weight apply is the one deliberate exception: it wraps mod-2⁶⁴ *by construction*, byte-identically to the
NumPy reference, so producer and verifier wrap the same way.

### 3.5 Producer / verifier split
The **pure-integer reference (NumPy/Python) is the canonical verifier**; accelerated kernels (CPU C/OpenMP,
optional CUDA) are **byte-identical producers**. A receipt produced on a GPU re-executes bit-for-bit on a
CPU-only verifier — so a third party never needs the producer's hardware. The parity gate
(`np.array_equal(kernel_out, oracle_out)`) is what makes aggressive kernel work safe.

---

## 4. Weight formats: Q1_0 and BitNet ternary

The "matmul" `Y = X · Wᵀ` for low-bit signed weights is a **masked integer sum**, not a float dot product. This
is where the engine and BitNet meet.

### 4.1 Q1_0 (1-bit, `{−1,+1}`, per-group int scale) **[proven]**
Per output `o`, per 128-weight group `b`, with `signᵢ ∈ {−1,+1}` and int activations `xᵢ`:

```
signed_sum_b = Σ_i signᵢ·xᵢ = 2·(Σ_{bit=1} xᵢ) − (Σ_i xᵢ) = 2·pos_sum_b − block_total_b
out_o        = Σ_b  arshift( signed_sum_b · scale[o,b],  frac )      # per-group scale, per-group floor, then sum
```
`block_total_b` is independent of `o` (compute once per token-group, reuse across all outputs); only `pos_sum_b`
depends on the weight bits — a *masked sum* of `x`, no multiply by ±1 and no branch.

### 4.2 BitNet b1.58 (1.58-bit ternary, `{−1,0,+1}`) **[proposed]**
Ternary is the same shape with a zero state that simply drops out of the sum:

```
signed_sum_b = Σ_i wᵢ·xᵢ = (Σ_{wᵢ=+1} xᵢ) − (Σ_{wᵢ=−1} xᵢ)            # zeros contribute nothing
out_o        = Σ_b  arshift( signed_sum_b · scale[o,b],  frac )
```
Two sign bits per weight (or a packed 5-trits-per-byte / `i2` layout) select `+x`, `−x`, or skip. The engine's
existing masked-sum machinery covers it directly; the only new code is the **ternary unpack** in the weight
codec. BitNet b1.58 also publishes a per-tensor (or per-row) scale that maps onto the per-group `scale[o,b]`.

### 4.3 A weight-codec abstraction **[proposed]**
Define a small interface so the engine is weight-format-agnostic:

```
class QuantCodec:
    group_size: int
    def unpack_signs(bits) -> int8[...]      # {−1,0,+1}
    def scale(o, b) -> int64                 # per-group integer scale (fixed-point)
    def apply(x_fp, weight, frac) -> int64   # masked integer signed-sum -> out  (byte-exact)
```
Implementations: `Q1_0Codec` **[proven]**, `BitNet158Codec` (ternary) **[proposed]**, and any future
per-group-scaled low-bit signed scheme. The pure-integer `apply` is the oracle; CPU/CUDA kernels are
byte-identical fast paths registered against the same parity gate.

### 4.4 Activations: the one real numerics decision
Two consistent choices, both fully integer/deterministic:
- **Fixed-point int64 activations (the engine's current model).** `x` carries `frac` bits; the apply is
  `int64`; RMSNorm uses int128. Maximally faithful (no activation quantization), at int64 cost.
- **BitNet-native int8 activations + int32 accumulation.** BitNet quantizes activations per-token (absmax →
  int8) and accumulates ternary·int8 in int32. This is **still deterministic** (integer accumulation,
  associative) and maps to `__dp4a`/int8-IMMA tensor cores. The engine can support a **per-tensor activation
  scale** path so it matches a BitNet reference *bit-for-bit*, provided the reference's quant+accumulate is
  specified exactly (absmax rounding, int32 accumulate, requant) and reproduced.

> Determinism does **not** require int64 — it requires *integer accumulation with a defined no-overflow/wrap
> contract*. int8×ternary→int32 is byte-exact iff the int32 accumulator never overflows (a static per-tensor
> bound, like §3.4). The fixed-point-int64 path is the conservative default; the int8 path is the
> performance/compat path for BitNet.

---

## 5. The bit-matmul / tensor-core mapping **[partly proven]**

The masked signed-sum maps to hardware integer MAC units (`__dp4a`, int8 IMMA tensor cores) **while staying
byte-exact**, via a base-256 limb decomposition of the (multi-bit) activations — int8 weight (`±1`/ternary) ×
int8 activation-limb → int32, recombined in int64 (exact mod-2⁶⁴). Full derivation + overflow proof:
`bonsai-notary/Q1-BITMATMUL-REFORMULATION.md`.

**Honest measured caveat [proven]:** for the *Q1_0* case on a consumer GPU, the masked sum is **add-bound, not
multiply-bound** (the ±1 weight is a sign flip, not a real multiply), so DP4A gave only ~1.15× kernel-only and
did not win end-to-end — the win came from **weight residency** + **large-M prefill**, not from tensor cores.
**BitNet ternary with int8 activations is more multiply-like** and is exactly the regime tensor cores are built
for, so the tensor-core path is more promising there — but it must be *measured*, not assumed (this project's
recurring lesson: three plausible GPU optimizations measured as regressions and were gated off).

---

## 6. Architecture (standalone) **[proposed, extracted from proven parts]**

```
                 ┌─────────────────────────────────────────────────────────────┐
  GGUF model ───▶│  Importer: GGUF → integer artifact                            │
 (Ollama blob /  │   • dequant-to-integer weights (packed signs + per-group int  │
  HF download)   │     scales), fixed-point RoPE tables, integer norm gains      │
                 └─────────────────────────────────────────────────────────────┘
                                   │  integer artifact (safetensors / mmap)
                                   ▼
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │  CORE ENGINE (all integer)                                                      │
   │   pure-integer reference  ── canonical VERIFIER (the oracle)                    │
   │   CPU native (C/OpenMP)    ┐                                                    │
   │   CUDA (per-host opt-in)   ┘ byte-identical PRODUCERS (parity-gated)            │
   │   QuantCodec: Q1_0 | BitNet-ternary | …                                         │
   │   integer ops: apply · RMSNorm · softmax · SiLU · RoPE · argmax · samplers      │
   └──────────────────────────────────────────────────────────────────────────────┘
                                   │
            ┌──────────────────────┼───────────────────────────┐
            ▼                      ▼                           ▼
   CLI (run / serve)     Ollama-compatible HTTP        Library API (embed)
                         /api/generate /api/chat        + optional RECEIPTS
                         + OpenAI /v1/chat/completions   (commit + re-exec + sign)
```

Layers:
1. **Importer** — read GGUF (incl. Ollama's blob store, §7), dequantize to the integer artifact: packed weight
   signs, per-group integer scales, fixed-point RoPE tables, integer RMSNorm gains, committed sampler/config.
2. **Core** — the pure-integer oracle + byte-identical CPU/CUDA kernels + the `QuantCodec`. (Extracted from
   `reference_bonsai.py` / `fixedpoint.py` / `q1_native.py` / `gpu_native.py` / `bonsai_q1_kernel.c` /
   `bonsai_q1_gpu.cu`.)
3. **Interfaces** — CLI, an Ollama/OpenAI-compatible server, and a library API.
4. **Receipts (optional)** — commitments + bit-exact re-execution + signatures (extracted from `receipts/`).

---

## 7. Ollama interop **[proposed]**

Ollama's backend is llama.cpp (float) — this engine does **not** plug into Ollama's runner. Instead it reuses
the two parts of Ollama that matter: its **model store** and its **API surface**. GGUF is the bridge.

### 7.1 Consume Ollama's model store (read side)
Ollama stores pulled models under `~/.ollama/models/` as OCI-style content-addressed blobs plus JSON manifests
(`manifests/<registry>/<namespace>/<model>/<tag>`). A manifest lists layers by `sha256:` digest; one layer
(media type for the model GGUF) is the weights blob. The importer:
1. resolves a model name+tag → manifest → the GGUF blob digest → `blobs/sha256-<digest>`;
2. reads that GGUF directly (no copy) and imports it to the integer artifact (cached, content-addressed by the
   blob digest so re-import is skipped).

So the user does `ollama pull <bitnet-model>` and points the engine at the model name — it finds the blob.
*(Exact manifest media-types/paths should be pinned against the installed Ollama version at implementation
time; treat the above as the shape, not a frozen spec.)*

### 7.2 Ollama-compatible API (serve side)
Expose an HTTP server matching the endpoints existing Ollama clients use:
- `POST /api/generate` (prompt → completion, streaming NDJSON)
- `POST /api/chat` (messages → chat, streaming)
- `GET /api/tags` (list available integer models)
- plus an **OpenAI-compatible** `POST /v1/chat/completions` for the broad tool ecosystem.

Requests run on the deterministic integer engine. Two value-adds over stock Ollama: responses are
**byte-reproducible** at a fixed seed, and an optional `receipt: true` request field returns (or logs) a
verifiable receipt alongside the output.

### 7.3 Packaging as an Ollama model (write side, optional)
Ship integer artifacts as GGUF (or a sidecar) with a `Modelfile` so `ollama create` registers them. The engine
advertises which GGUF quant types it can run deterministically (Q1_0 and BitNet ternary types) and refuses
others loudly rather than silently falling back to float.

### 7.4 Honest boundary
This is **interop, not a fork of Ollama**: reuse its distribution + API, replace the compute with the integer
engine. Float GGUF models (the majority) are out of scope — the engine runs **1-bit / ternary** models, where
integer is exact and the determinism story is real.

---

## 8. Verifiable inference / receipts **[proven, optional]**

Because the engine is byte-exact, each generation can emit a **receipt**: commitments over (model hash, input,
output, integer execution trace), a hash-linked ledger entry, and **third-party-verifiable signatures**
(secp256k1; the receipt carries only the public key). A verifier re-executes the receipt on the **CPU integer
oracle** and confirms the bytes — no shared secret, no GPU, no trust in the producer. This is the capability a
float engine structurally cannot offer (it can't reproduce its own bytes), and it is fully built in the
reference project (`receipts/`, `bonsai_runtime.emit_and_verify_*`). For a standalone engine it is an opt-in
feature, off by default for plain "fast local inference," on for audit/agent/compliance use.

> **Scope: re-execute the COMMITTED artifact, not re-import the GGUF.** Verification binds the committed
> `artifactDigest` to the receipt's `modelHash` and re-runs the committed integer artifact — that is
> byte-stable everywhere. It does NOT claim that re-importing the same GGUF reproduces the same artifact: the
> RoPE/YaRN tables are built with libm `cos`/`sin` + float `pow` then rounded and committed+hashed, so import
> on different hardware can yield a different `artifactDigest`/`modelHash`. "Verifiable" = re-execute the
> exact committed bytes, not re-derive them from source.

---

## 9. Performance **[proven numbers from the reference 8B Q1_0 engine]**

Honest, measured on an 8-core CPU + RTX 3070 (so directional for the standalone engine, exact for Q1_0-8B):

- **CPU decode**: the 1-bit weight gather is **L3-latency-bound** at the portable build floor; concurrency
  (threads / process-sharding) is the lever, not single-stream kernel tuning.
- **GPU (byte-exact producer)**: weight **residency** + large-M **prefill** are the wins — single-query decode
  **2.14×**, **prefill 17.6×** (T=256), end-to-end **generation 12.7–15.8×**, **batch decode 1.85×**, all
  byte-identical to the CPU oracle. Consumer-GPU **int64 is emulated** (the headwind); DP4A/IMMA helped Q1_0
  only marginally (add-bound) — expect a **better tensor-core story for BitNet int8×ternary**.
- **Determinism cost**: int128 RMSNorm + integer softmax polynomials are more work than float equivalents, but
  this is what buys reproducibility; for **tiny BitNet models** the absolute cost is small and CPU-friendly,
  which is the sweet spot.

Takeaway: the engine is competitive and *reproducible*; it trades a constant factor of arithmetic for a property
(byte-exactness) no float engine has. BitNet's small size makes that trade cheap.

---

## 10. What's proven vs. what this proposes

| Capability | Status |
|---|---|
| All-integer fixed-point transformer, byte-exact CPU-oracle ⇄ CPU-native (incl. int64 overflow boundary) | **proven, CI-gated** (Q1_0 8B) |
| Pure-integer oracle + byte-identical CPU C kernel | **proven, CI-gated** |
| CUDA kernel int64-faithful (no float atomics/`expf`/fast-math), byte-checked vs CPU oracle on GPU hardware or `BONSAI_VERIFY_GPU=1` | **proven on hardware; NOT in default CI** (no GPU in CI) |
| Q1_0 masked-sum apply, integer RMSNorm/softmax/SiLU/RoPE, integer samplers | **proven** |
| Cryptographic receipts (commit + re-exec + secp256k1 signatures) | **proven** |
| GGUF import → integer artifact (re-execute the committed artifact; re-import is not byte-stable — RoPE/YaRN libm tables) | **proven** (one model family) |
| GPU residency / prefill / batch wins, byte-checked vs CPU oracle when GPU present | **proven (measured on one GPU)** |
| **Extract the core into a standalone, format-agnostic engine** | **proposed** |
| **`BitNet158Codec` (ternary) + a generic `QuantCodec` interface** | **proposed** |
| **int8-activation BitNet path (DP4A/IMMA) with a matching parity gate** | **proposed** |
| **Read Ollama's blob store; serve `/api/generate` `/api/chat` `/v1`** | **proposed** |
| **Package integer artifacts as Ollama models (`Modelfile`)** | **proposed** |

---

## 11. Risks & open questions

1. **BitNet GGUF formats.** The exact GGUF quant types / packing for BitNet b1.58 (and bitnet.cpp's
   layouts) must be read and pinned; the ternary unpack + per-tensor activation quant must be reproduced
   **bit-for-bit** to claim byte-exactness against a BitNet reference. Until verified against a real model, the
   `BitNet158Codec` is a design, not a guarantee.
2. **Activation-quant determinism.** If matching a published BitNet reference, every rounding step (absmax,
   int8 cast, int32 accumulate, requant) must be specified and reproduced; otherwise the engine defines its own
   (still-deterministic) numerics and is self-consistent but not bit-equal to that reference.
3. **Ollama API/version drift.** Manifest media-types and API fields evolve; pin against the installed version
   and test, rather than hard-coding.
4. **Cross-hardware verification is argued, not exhaustively measured.** Integer associativity gives
   byte-exactness *by construction*, but it should be validated on ≥2 CPU arches and ≥2 GPU arches before the
   "verify anywhere" claim is load-bearing (the reference project verified launch-geometry invariance on one
   GPU only).
5. **Scope discipline.** This is for **1-bit / ternary** models. Don't promise float-GGUF parity; refuse
   unsupported quant types loudly. The value is *verifiable integer inference of integer models*, not a general
   llama.cpp replacement.

---

## 12. Minimal first milestone

1. Extract the integer core (oracle + CPU kernel + `QuantCodec`) into a standalone package; re-run the Q1_0
   parity gate to prove the extraction is byte-identical.
2. Add `BitNet158Codec`; import one real BitNet b1.58 GGUF; prove the integer oracle runs it and is
   self-reproducible (same bytes across thread counts).
3. Add the Ollama blob-store reader (`ollama pull` → engine runs the blob) and a minimal `/api/generate`.
4. Turn on optional receipts; demonstrate produce-on-GPU / verify-on-CPU byte-exact.

Each step is gated by `np.array_equal(producer, oracle)` — correctness first, speed second, exactly as the
reference engine was built.

---

*Reference implementation (the proven core this generalizes):* `bonsai-notarized-bitnet` —
`src/trinote/infer_int/` (engine, kernels, import, samplers, receipts glue),
`src/trinote/determinism/fixedpoint.py` (integer primitives), `tools/bonsai_q1_kernel.c` +
`tools/bonsai_q1_gpu.cu` (byte-identical CPU/CUDA producers), and the specs under `~/research/bonsai-notary/`
(`Q1-BITMATMUL-REFORMULATION.md`, `GPU-FEASIBILITY.md`, `IMPLEMENT-GPU-MODE.md`, `DETERMINISM`/`SAMPLER` docs).
