# integer_inference_engine

A **deterministic, integer-only inference engine** for low-bit (BitNet-style) LLMs. All model
arithmetic is integer fixed-point, so a generation is **byte-identical across hardware** (thread count,
SIMD width, GPU, batch size) — which is what makes inference *verifiable*: a third party can
re-execute a run on a reference CPU and get the same bytes, so each generation can carry a
cryptographic receipt.

See [`integer_engine.md`](integer_engine.md) for the full design (sections marked **[proven]** vs
**[proposed]**).

> **Scope of "verifiable":** the guarantee is *re-execute the exact committed artifact and get the same
> bytes*, NOT *re-import the same GGUF and reproduce the artifact*. The RoPE/YaRN tables are built with
> libm `cos`/`sin` + float `pow` then rounded and committed+hashed, so re-IMPORTING the same GGUF can yield
> a different `artifactDigest`/`modelHash` across hardware. Inference from a committed artifact is
> byte-stable; verification pins the committed artifact (`artifactDigest == modelHash`), not the import.

## The core idea — see [`CONCEPT.md`](CONCEPT.md)

**Run an LLM so its output can be *proven* to come from a specific model, and re-checked by anyone.**

- **Float inference can't be verified.** Reduction order (threads/SIMD/GPU/batch) and vendor transcendentals
  (`exp`/`cos`/`1/sqrt`) differ across machines, so the same model + input gives different bytes — nothing a
  third party can reproduce to check a receipt.
- **Integer fixed-point fixes it.** Integer addition is associative → reduction-order invariant; the
  transcendentals are replaced by exact integer algorithms. So the forward is **byte-identical on any
  hardware**.
- **That enables receipts.** Commit the model artifact + inputs, run the deterministic forward, sign +
  hash-chain it; a verifier **re-executes on a reference CPU and gets the same bytes**. The CPU integer oracle
  is the canonical verifier; the C/CUDA kernels are byte-exact *producers* gated against it.
- **Two properties, kept separate:** *determinism* (reproduces its own bytes — holds by construction) vs
  *fidelity* (how close to the float model — empirical; north-mini-code: **92% next-token / 100% top-3** vs
  Ollama after the cohere2 RoPE fix).

## Two engines

| Dir | Model | Status |
|---|---|---|
| [`bonsai/`](bonsai/) | **Bonsai-8B**, Qwen3-dense, 1-bit `Q1_0` weights | **proven** — CPU-oracle ⇄ CPU-native byte-exact and CI-gated (incl. the int64 overflow boundary); the CUDA kernel is int64-faithful (no float atomics / `expf` / fast-math) and byte-checked against the CPU oracle when GPU hardware is present or `BONSAI_VERIFY_GPU=1` (that GPU parity is NOT exercised in default CI). Cryptographic receipts, GGUF import. The reference instance. |
| [`north-mini-code/`](north-mini-code/) | **Cohere2-MoE 30.5B**, `Q4_K_M` (mixed Q4_K/Q6_K) | **working** — codec, dense+MoE blocks, real-weight forward, byte-exact CPU+CUDA kernels (resident weights), KV-cached decode, **92% next-token / 100% top-3 fidelity vs Ollama** (after fixing the cohere2 interleaved-RoPE + NoPE), and **secp256k1 receipts** (verified on a 4090). On-chain 3rd entry + CLI launcher pending. |

`north-mini-code` reuses `bonsai`'s determinism primitives (`fixedpoint`, RoPE); a small set is vendored
verbatim under `north-mini-code/src/nmc/_bonsai/` to keep it self-contained.

## Quick start (bonsai)

```bash
cd bonsai
uv venv && uv pip install -r requirements.txt
tools/build_bonsai_q1_kernel.sh                 # native packed-Q1 CPU kernel
./../bonsai-cli "What is a tensor?"             # or: bonsai-cli repl  /  bonsai-cli json "..."
```

The `bonsai-cli` launcher at the repo root wraps `bonsai/src/trinote/cli/run_bonsai_cli.py`. Model
weights and all generated state live OUTSIDE the repo under `$BONSAI_NOTARY_HOME` (default
`~/.local/trinote`); see `bonsai/README.md` and `bonsai/bonsai8b.md`.

## Quick start (north-mini-code)

```bash
ollama pull north-mini-code-1.0                 # provides the ~18GB Q4_K_M GGUF (cuda-resident wants ~24GB VRAM)
cd north-mini-code && uv venv --python 3.12
uv pip install -r requirements.txt -r requirements_receipts.txt
tools/build_nmc_kernel.sh                        # CPU kernel  (tools/build_nmc_cuda.sh for the GPU kernel)
./../north-mini-code-cli "The capital of France is" --verbose
./../north-mini-code-cli "The capital of France is" --receipts --verbose   # + a verifiable receipt
./../north-mini-code-cli repl --receipts                                    # interactive
```

`--receipts` emits a secp256k1-signed, hash-chained, content-addressed bundle that re-verifies **offline**
(no model/GPU) and carries an on-chain 3rd-entry artifact (dry-run-logged; `--broadcast` for a real BSV send via
the bonsai-notary wallet). The receipt attests the *integer engine's deterministic output* (re-verifiable by
anyone), not float-parity with llama.cpp — see [`CONCEPT.md`](CONCEPT.md). State/keys/bundles live OUTSIDE the
repo under `~/.local/integer_inference_engine/north-mini-code`. The `north-mini-code-cli` launcher reuses the
model-agnostic `trinote.*` notary stack via `$NMC_BONSAI_SRC` (default `bonsai/src`).

## Used by `bonsai-notary`

This engine is one of four pieces in the [`bonsai-notary`](../bonsai-notary) composition (deterministic
inference + on-chain notarization). That repo references this one at `engine/` and expects the
`trinote` package under `bonsai/src`.

## License

Apache-2.0 (`LICENSE`).
