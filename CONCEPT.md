# The core idea: inference you can notarize

**Goal: run an LLM so that its output can be *proven* to come from a specific model — and re-checked by
anyone, on any hardware.** The enabling trick is to make the entire forward pass **byte-for-byte
reproducible** by computing it in **integer fixed-point** instead of floating point.

---

## 1. The problem: float inference can't be verified

A cryptographic receipt for "model *M* produced output *Y* from input *X*" is only useful if a third party
can **re-execute** the computation and get the *same* result to check it. Floating-point inference breaks this:

- **Reduction order isn't fixed.** A matmul/softmax sum is split across threads, SIMD lanes, GPU blocks, and
  batch tiles in hardware-dependent orders. Float addition isn't associative, so `(a+b)+c ≠ a+(b+c)` in the
  last bits — the logits differ across machines, libraries, and even batch sizes.
- **Transcendentals differ.** `exp`, `cos`, `sin`, `1/sqrt` come from vendor math kernels (libm vs CUDA vs
  fast-math) that disagree in the low bits.

So two honest parties running the *same* model on the *same* input get *different* bytes. There is nothing to
sign that the other side can reproduce. **Float inference is not attestable.**

## 2. The solution: all-integer fixed-point → byte-identical everywhere

Run every operation in integer fixed-point arithmetic (activations scaled by `2^fa`, weights by `2^fw`):

- A fixed-point value is an **integer**; the product of two is an exact integer; a contraction is an exact
  integer **sum**. Integer addition **is** associative → **reduction-order invariant**. The result is identical
  regardless of thread count, SIMD width, GPU vs CPU, or batch size.
- The hazardous transcendentals (softmax `exp`, RMSNorm `1/sqrt`, SiLU sigmoid) are replaced by **integer
  algorithms** — a fixed polynomial for `2^-x`, an exact integer `isqrt` — that are bit-identical on every
  machine (no libm, no float reduction).

The result: **the same model + input produces the same bytes on any hardware.** That byte-exactness is the
whole game — it is what a receipt commits to and what a verifier reproduces.

## 3. From byte-exactness to a receipt

With a deterministic forward, a generation can carry a verifiable receipt:

1. **Commit** the model artifact (`modelHash` over the committed integer weights + RoPE tables + config) and
   the inputs.
2. **Run** the integer forward to produce the output tokens (the producer may be a fast GPU).
3. **Sign + chain** the (commitments, input, output) into a receipt (secp256k1, hash-chained ledger).
4. **Verify**: anyone re-executes the integer forward from the committed artifact on a reference CPU and gets
   the **exact same bytes** → the receipt checks out. No trust in the producer required.

The **CPU integer oracle is the canonical verifier**; faster backends (the C kernel, the CUDA kernel) are
**producers** that must be **byte-identical** to that oracle — every kernel is gated against it.

> **What "verifiable" means precisely.** The guarantee is *re-execute the exact committed artifact and get the
> same bytes*, not *re-import the same GGUF and reproduce the artifact*. The RoPE tables are built with libm
> `cos`/`sin` + float `pow`, then rounded and **committed + hashed**, so re-importing a GGUF on different
> hardware can yield a different digest. Inference *from a committed artifact* is byte-stable; verification
> pins the committed artifact (`artifactDigest == modelHash`), not the import path.

## 4. Determinism vs. fidelity (two separate properties)

- **Determinism** — the engine reproduces *its own* output byte-for-byte anywhere. This holds **by
  construction** (integer, order-invariant) and is enforced by parity gates (CPU oracle ⇄ C/CUDA kernels).
- **Fidelity** — how close the integer engine's output is to the *float* model it approximates. This is
  **empirical** (fixed-point rounding accumulates over the layers) and is measured by the fidelity eval.
  A receipt attests determinism; fidelity is the separate question of "is this still a faithful run of the
  model." For north-mini-code, after fixing the cohere2 RoPE convention, the integer engine agrees with the
  float model (Ollama) on **92% of next tokens (100% within top-3)** — i.e. faithful as well as deterministic.

## 5. Two engines

| Engine | Model | Why |
|---|---|---|
| **bonsai** | Bonsai-8B (Qwen3 dense, 1-bit `Q1_0`) | The proven reference: deterministic integer engine + a working signed/hash-chained receipt stack. |
| **north-mini-code** | Cohere2-MoE **30.5B** (`Q4_K_M`) | Proves the approach scales past a small dense model to a large **Mixture-of-Experts** in higher-precision integer fixed-point. |

`north-mini-code` reuses bonsai's determinism primitives (fixed-point RMSNorm/softmax/sigmoid, RoPE), vendored
verbatim so it stays self-contained. It adds the MoE router/gating, the parallel block, interleaved SWA/full
attention, a GPU kernel with resident weights, and a KV-cached decode path — all under the same byte-exact
discipline.

## 6. Why it matters

Deterministic, attestable inference enables **provenance and audit** for model outputs: prove that a specific
model (not a swapped or tampered one) produced a specific output, verifiable by anyone without trusting the
operator — the basis for on-chain notarization, regulatory attestation, and trustless model marketplaces.

---

*Full design: [`integer_engine.md`](integer_engine.md). North-mini-code status + the fidelity result:
[`north-mini-code/`](north-mini-code/). Supporting research notes remain outside the published repository.*
