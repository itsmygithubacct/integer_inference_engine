# integer_inference_engine

A **deterministic, integer-only inference engine** for low-bit (BitNet-style) LLMs. The canonical model
graph uses specified fixed-point arithmetic so a third party can re-execute a committed run on the CPU
oracle and compare exact bytes. Native CPU and CUDA implementations are *producers*: they do not become
verifiers merely by being faster, and each release/configuration needs byte-parity evidence against that
oracle before it can be used behind a receipt gate.

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
- **Integer fixed-point specifies a reproducible forward.** Modular integer reductions and exact integer
  transcendental replacements define the same result independent of scheduling. A canonical implementation
  can therefore be byte-stable across hardware; each optimized kernel still has to prove parity with it.
- **That enables receipts.** Commit the model artifact + inputs, run the deterministic forward, sign +
  hash-chain it; a verifier **re-executes on a reference CPU and gets the same bytes**. The CPU integer oracle
  is the canonical verifier; the C/CUDA kernels are byte-exact *producers* gated against it.
- **Two properties, kept separate:** *determinism* (reproduces its own bytes — holds by construction) vs
  *fidelity* (how close to the float model — empirical; north-mini-code: **92% next-token / 100% top-3** vs
  Ollama on a 24-prompt greedy-continuation set — a small, favourable sample, not a broad benchmark; see
  `north-mini-code/tools/fidelity_eval.py`).

## Two engines

| Dir | Model | Status |
|---|---|---|
| [`bonsai/`](bonsai/) | **Bonsai-8B** Qwen3-dense and **Bonsai-27B** Qwen3.5-hybrid, 1-bit `Q1_0` weights | **8B proven; 27B host acceptance passed** — the 27B resident CPU producer clears the controlled 4x steady-decode and 3x prompt-prefill gates; the resident CUDA producer clears populated-4K throughput and 128-token pure-oracle parity. A distinct artifact/quality-bound 27B identity and fresh-oracle receipt smoke pass locally; the root launcher remains receipt-off by default. |
| [`north-mini-code/`](north-mini-code/) | **Cohere2-MoE 30.5B**, `Q4_K_M` (mixed Q4_K/Q6_K) | **working** — codec, dense+MoE blocks, real-weight forward, byte-exact CPU+CUDA kernels (resident weights), KV-cached decode, **92% next-token / 100% top-3 fidelity vs Ollama** (24-prompt sample), and **secp256k1 receipts** (verified on a 4090). On-chain 3rd entry + CLI launcher pending. |

`north-mini-code` reuses `bonsai`'s determinism primitives (`fixedpoint`, RoPE); a small set is vendored
verbatim under `north-mini-code/src/nmc/_bonsai/` to keep it self-contained.

## Get it

```bash
git clone https://github.com/itsmygithubacct/integer_inference_engine.git
cd integer_inference_engine
```

Standalone — no sibling repos required. Pick an engine below (`bonsai/` is the proven reference).

## Quick start (bonsai)

```bash
cd bonsai
uv venv && uv pip install -r requirements.txt
tools/build_bonsai_q1_kernel.sh                 # native packed-Q1 CPU kernel
scripts/fetch_weights.sh                         # REQUIRED: downloads the model into $BONSAI_NOTARY_HOME (out of tree)
./../bonsai-cli "What is a tensor?"             # or: bonsai-cli repl  /  bonsai-cli json "..."
```

The `bonsai-cli` launcher at the repo root wraps `bonsai/src/trinote/cli/run_bonsai_cli.py`. Model
weights and all generated state live OUTSIDE the repo under `$BONSAI_NOTARY_HOME` (default
`~/.local/trinote`); see `bonsai/README.md` and `bonsai/bonsai8b.md`. Without the weight fetch the CLI
aborts with `[bonsai] FATAL: required file not found` — it never auto-downloads.

### Bonsai-27B on Linux

The repository exposes the 1-bit `prism-ml/Bonsai-27B-gguf` checkpoint through both the native integer engine
and the pinned PrismML llama.cpp CUDA runtime:

```bash
bonsai/scripts/install_bonsai_27b_gguf.sh
bonsai/scripts/fetch_bonsai_27b_gguf.sh
cd bonsai
tools/build_bonsai_q1_kernel.sh
PYTHONPATH=src .venv/bin/python -m trinote.cli.import_bonsai35_gguf_cli --context-len 4096
cd ..
./bonsai-integer-27b-cli "Explain Merkle proofs." -n 64
./bonsai-integer-27b-cli repl
./bonsai-27b-cli "Explain Merkle proofs." -n 256
```

The integer launcher defaults to a 1,024-token generation budget, not a model ceiling; pass `-n N` or set
`BONSAI_INTEGER_27B_MAX_NEW=N` to change it. The imported deterministic artifact currently has a 4,096-token
context, while the source GGUF advertises a larger context. The native REPL therefore selects 4,096 automatically.
The 8B REPL prefers its original 16,384-token (pre-YaRN) quality window when host memory permits. A deterministic
4-gram guard is enabled by default to stop exact loops without repetition-penalizing the EOS/control tokens.
Qwen3.5 one-shot thinking stays enabled for answer quality; the REPL answers directly by default, and `/think on`
enables reasoning for the session.

The integer launcher first attempts the optional resident deterministic CUDA producer and cleanly falls back to
the optimized CPU producer if CUDA is unavailable, the 4K memory/exclusivity preflight fails, or a device guard
poisons the context. Set `BONSAI_INTEGER_27B_GPU=0` to force CPU. The CPU executor performs a complete 64-layer
token plus final norm/output through one native ABI call and one persistent OpenMP team. Controlled,
content-bound i7-10700F runs show 4.2632x and 4.1983x median speedups for decode tokens 3-32 and 33-128,
respectively. Exact 32-token and 128-token prefill gates pass at 4.1616x and 3.6854x, with identical cache/output
commitments and sub-5% variation. Raw JSON and the fail-closed comparisons are linked from
[`bonsai/BONSAI-27B-BENCHMARKING.md`](bonsai/BONSAI-27B-BENCHMARKING.md).

The separate `bonsai-27b-cli` is a faster floating-point PrismML/CUDA path and can never issue Trinote receipts.
Its default context is now llama.cpp's model-aware hardware auto-fit (`-c 0 --fit on`) instead of a launcher-fixed
4,096. Set `BONSAI_CONTEXT_SIZE=N` or pass `--context-size N` to override either backend.
On an 8 GiB RTX 3070 it cannot coexist with the resident integer graph, whose hardened populated-4K run used
6,362,562,560 bytes live with a 6,740,049,920-byte conservative proof peak. Stop/unload PrismML or force the
integer launcher to CPU. The accepted CUDA path stores KV with a guarded int32 representation; the optimized
CPU producer still stores KV as int64. Both downloads are pinned and SHA-256 verified, and runtime/model data stays
outside the checkout. See
[`bonsai/BONSAI-27B-GGUF.md`](bonsai/BONSAI-27B-GGUF.md).

The root 27B launcher deliberately disables receipts. Direct CLI and library callers fail closed unless the
distinct artifact-bound Qwen3.5 identity, its hash-bound passing PrismML quality gate, and a fresh non-native CPU
oracle verifier are explicitly supplied. That local gate now passes, including a real receipt plus deliberate
output-corruption rejection, but receipt mode remains opt-in and PrismML can never issue a receipt.

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

This engine is one of four pieces in the [`bonsai-notary`](https://github.com/itsmygithubacct/bonsai-notary) composition (deterministic
inference + on-chain notarization). That repo references this one at `engine/` and expects the
`trinote` package under `bonsai/src`.

## License

Apache-2.0 (`LICENSE`).
