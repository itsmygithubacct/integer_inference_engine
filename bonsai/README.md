# Bonsai-8B deterministic integer inference engine (standalone)

A self-contained extraction of the **`int-ref@bonsai-qwen3`** engine from `bonsai-notarized-bitnet`: an
all-integer / fixed-point transformer that runs the **ATLAS-Notarized-Bonsai-8B** model (Qwen3, ~8.19B params,
**Q1_0** 1-bit weights, `frac=16`) with output **byte-identical across CPU/GPU/threads/batch** and an optional
**third-party-verifiable receipt** per generation.

For the full design and measured numbers see **[`bonsai8b.md`](bonsai8b.md)** (this model) and
**[`../integer_engine.md`](../integer_engine.md)** (the generalized BitNet/Ollama design).

> This is the **engine only** — the deterministic compute, kernels, GGUF import, samplers, and receipts. The
> agent-identity, BSV chain, and HD-wallet subsystems are **not** included (the `--onchain` path shells out to a
> wallet script if one is present, otherwise it is unavailable; everything else runs without them).

---

## Layout

```
bonsai/
├── src/trinote/
│   ├── infer_int/        # the engine: reference_bonsai.py (oracle+dispatch), bonsai_runtime.py,
│   │                     #   sampler.py, q1_native.py / gpu_native.py (kernel loaders), import_*.py, verify.py
│   ├── determinism/      # fixedpoint.py — integer primitives (matmul, rmsnorm, softmax, isqrt, exp2)
│   ├── model/            # rope_v2.py — fixed-point NeoX RoPE + tables
│   ├── receipts/         # commit / ledger / signing (secp256k1 + HMAC) / verify / broadcast
│   ├── bundle/           # portable receipt bundle pack/verify
│   ├── hashing/          # sha helpers
│   ├── cli/              # run_bonsai_cli.py, json_mode.py, import_bonsai_gguf_cli.py, receipt_bundle_cli.py,
│   │                     #   quality_gate_bonsai_cli.py
│   ├── config_bonsai.py  # the canonical 8B config; charter.py; notary_paths.py (out-of-repo state homes)
├── tools/                # bonsai_q1_kernel.c (CPU) + bonsai_q1_gpu.cu (CUDA) + build_*.sh
├── tests/                # test_bonsai_smoke.py, test_bonsai_gpu.py, test_receipt_bundle.py (byte-exact gates)
├── scripts/              # bonsai.sh (launcher), fetch_weights.sh
├── artifacts/            # atlas-notarized-bonsai-8b.identity.json (modelHash binding)
├── requirements.txt      # numpy, safetensors, ecdsa   (+ requirements_test.txt: pytest)
└── bonsai8b.md, README.md
```

All **generated state and secrets** live OUTSIDE this tree under `$BONSAI_NOTARY_HOME` (default
`~/.local/trinote`): receipt ledgers, bundles, the built kernel `.so` (in `bin/`), and the auto-generated
secp256k1 signing keys (`keys/`, mode 0600). Nothing a run produces is written back into the source tree.

---

## Setup

```bash
cd ~/integer_inference_engine/bonsai
uv venv --python 3.12
uv pip install -r requirements.txt            # numpy, safetensors, ecdsa  (add requirements_test.txt for pytest)

tools/build_bonsai_q1_kernel.sh               # CPU native kernel (byte-exact fast path) -> ~/.local/trinote/bin
tools/build_bonsai_q1_gpu.sh                  # OPTIONAL: per-host CUDA producer (needs nvcc); --gpu uses it

scripts/fetch_weights.sh                      # OPTIONAL: the 8B GGUF + imported safetensors (~2.6 GB, for real runs)
```

The kernel and weights are resolved at runtime via `trinote.notary_paths` (prefer `~/.local/trinote`, fall back
to the in-tree `tools/` / `artifacts/`). Without the CUDA `.so`, `--gpu` silently falls back to the CPU oracle;
without weights, the synthetic-model tests still run (the engine is exercised end-to-end on a tiny random model).

---

## Run

```bash
# launcher (curated flag sets; BONSAI_GPU=0 forces CPU, BONSAI_DRYRUN=1 prints the command)
scripts/bonsai.sh receipted "What is a tensor?" -n 64    # deterministic + notarized receipt
scripts/bonsai.sh deterministic "Hello"                  # integer engine, no receipt
scripts/bonsai.sh json "List three primes."              # structured {thinking,answer,receipt,bundle}
scripts/bonsai.sh repl                                    # interactive

# or the CLI directly
PYTHONPATH=src .venv/bin/python -m trinote.cli.run_bonsai_cli \
    --fast --receipt --chat -p "What is a tensor?" -n 64
```

Each receipted turn prints `[receipt] <hash> VERIFIED` (it self-verifies by re-executing bit-exactly before
emitting) and writes a portable bundle to `~/.local/trinote/bundles/`. Real secp256k1 signing keys are
generated on first use under `~/.local/trinote/keys/`; pass `--model-key`/`--counterparty-key` to supply your
own, or `--demo-keys` for the legacy deterministic HMAC vouch (snapshots only).

---

## Verify (byte-exactness gates)

```bash
PYTHONPATH=src:tests .venv/bin/python -m pytest tests/test_bonsai_smoke.py tests/test_receipt_bundle.py -q
PYTHONPATH=src:tests .venv/bin/python -m pytest tests/test_bonsai_gpu.py -q   # skips kernels if no GPU/.so
```

These prove every accelerated path (CPU C, CUDA) is `np.array_equal` to the pure-NumPy integer oracle — the
contract that makes the receipts meaningful. A stored bundle is independently re-verifiable offline with
`PYTHONPATH=src .venv/bin/python -m trinote.cli.receipt_bundle_cli verify <bundle.tar.gz> --reexec`.

---

## What this engine is for

Verifiable, reproducible local inference: the same prompt + seed yields the **same bytes** on any machine, and a
third party can re-run the receipt on a CPU-only host to confirm what the model produced — a property floating-
point runtimes (llama.cpp / Ollama) structurally cannot provide. Bonsai-8B is the proven instance; the codec
(`Q1_0`) and front door generalize to BitNet-ternary models and Ollama interop (see `../integer_engine.md`).
