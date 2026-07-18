# Bonsai-8B — the reference instance of the deterministic integer inference engine

> **Status: built, byte-exact, measured, in production use.** This is the
> concrete realization of the design in [`../integer_engine.md`](../integer_engine.md): an all-integer,
> byte-reproducible inference engine running a real 8B 1-bit (Q1_0) model, with cryptographic receipts. Where
> the parent doc is the generalized vision (BitNet-style + Ollama), this is the proven, specific implementation.

---

## 1. What it is

**ATLAS-Notarized-Bonsai-8B** is a Qwen3-architecture ~8.19-billion-parameter model quantized to **Q1_0** (one
signed bit per weight + one integer scale per 128-weight group), run by the **`int-ref@bonsai-qwen3`** engine —
a pure-integer / fixed-point transformer whose every output is **byte-identical across CPU, GPU, thread count,
and batch size**, and which emits a **third-party-verifiable receipt** for each generation.

| Property | Value |
|---|---|
| Architecture | Qwen3 (RMSNorm pre-norm, GQA, SwiGLU FFN, NeoX RoPE, QK-norm) |
| Parameters | 8,188,548,096 (~8.19B) |
| Quantization | `q1_0-g128` — 1 sign bit/weight, 1 int scale per 128-weight group |
| Fixed-point | `frac = 16` (activations are int64 at scale 2¹⁶) |
| d_model / layers | 4096 / 36 |
| heads (Q / KV) | 32 / 8  (GQA, rep = 4), head_dim = 128 |
| d_ffn | 12,288 |
| context / vocab | 65,536 / 151,669 |
| modelHash | `e5ae7bd10b103b8139f1c37e1c1d353878d4f55d8451d0b6b39aaac2943658e1` |
| Source weights | `Bonsai-8B-Q1_0.gguf` (PrismML GGUF) → imported to `atlas-notarized-bonsai-8b.safetensors` |

Per-layer Q1_0 contractions (the `K` of each weight matrix, in 128-wide groups): Q/K/V/O proj and FFN gate/up
contract over `d_model=4096` (32 groups); FFN down (`w2`) over `d_ffn=12288` (96 groups); the output head is
`151669 × 4096` (32 groups). Resident packed weights are ~1.5 GB (bits ≈ 1.0 GB + int scales ≈ 0.5 GB).

---

## 2. Why integer, here specifically

Bonsai's weights are already **1-bit signed** — there is no float weight to preserve. The "matmul" is therefore
a **masked integer signed-sum**, not a floating-point dot product, so doing the whole forward in **int64
fixed-point** is the *faithful* implementation, not a lossy approximation. Because integer addition is exactly
associative in `ℤ/2⁶⁴ℤ`, the result is **byte-identical regardless of execution order** (threads, SIMD, GPU,
batch) — which is what makes the receipts real: a third party re-executes on the CPU integer oracle and gets the
same bytes, with no shared secret and no GPU. Float runtimes (llama.cpp / Ollama) cannot do this — they can't
reproduce their own bytes across hardware.

---

## 3. The integer compute (concrete for Bonsai-8B)

All ops are integer; the only floats are host-computed correctly-rounded **scalars** (e.g.
`inv_sqrt_fp = round((1/√128)·2¹⁶)`), committed as ints — there is no float in the compute path and no float
reduction.

- **Q1_0 weight apply** — per output `o`, per 128-group `b`, with `signᵢ∈{−1,+1}` and int64 activations `xᵢ`:
  `signed_sum_b = Σ signᵢ·xᵢ = 2·pos_sum_b − block_total_b`, then
  `out_o = Σ_b arshift(signed_sum_b · scale[o,b], frac)` (per-group scale, per-group floor-toward-−∞, then
  summed). `block_total_b` is output-independent → computed once per token-group, reused across all 4096 / 12288
  / 151669 outputs.
- **RMSNorm** (n1/n2/q-norm/k-norm/final) — 128-bit sum-of-squares (the residual stream is unbounded across 36
  layers, so it exceeds int64), bit-exact integer `isqrt`, floor-division; fails loud → big-int oracle if even
  128 bits is insufficient.
- **Attention** — integer `q@Kᵀ`, integer softmax (cubic `2^-f` polynomial + `d_clip`, floor-div normalize, no
  `exp`), integer `@V`; GQA (32 Q heads share 8 KV heads, rep 4); causal for prefill, all-keys for decode;
  fail-loud `max|q|·max|k|·hd ≤ INT64_MAX` division-form bound.
- **SiLU / SwiGLU FFN** — integer sigmoid (same `2^-f` poly), `(silu(gate)·up)>>frac`.
- **RoPE** — NeoX rotate-half, integer multiply-add + arithmetic floor-shift, host-computed fixed-point cos/sin.
- **Output** — int64 vocab projection; **lowest-index argmax tie-break**; samplers (greedy/temp/top-k/top-p/
  min-p) integer + receipt-safe via a counter-based (SHA-256 + Lemire) draw keyed by `(seed, position)`.

---

## 4. Backends — one oracle, byte-identical accelerators

| Backend | Role | Artifact |
|---|---|---|
| **Pure-NumPy reference** | the canonical **verifier** (the oracle) | `reference_bonsai.py` (`_q1_bl_ref`, `q1_linear_ref`) |
| **CPU native (C / OpenMP)** | byte-identical **producer**, default fast path (`--fast`) | `tools/bonsai_q1_kernel.c` → `libbonsai_q1_kernel.so`, loaded by `q1_native.py` |
| **CUDA (per-host opt-in)** | byte-identical **producer** (`--gpu`) | `tools/bonsai_q1_gpu.cu` → `libbonsai_q1_gpu.so`, loaded by `gpu_native.py` |

Every accelerator is gated by `np.array_equal(kernel_out, oracle_out)` (`tests/test_bonsai_gpu.py`,
`tests/test_bonsai_smoke.py`) — including the int64 wrap boundary, GQA shapes, K=12288, forced-tie argmax, and
launch-geometry invariance. The CUDA `.so` is **per-host, arch-specific, gitignored**; absent it, `--gpu`
silently falls back to the CPU oracle, so a clean clone runs unchanged.

---

## 5. Measured performance (8-core i7-10700F + RTX 3070, OMP=15)

All byte-identical to the CPU oracle.

| Phase | Result |
|---|---|
| CPU single-query decode | ~0.23 s/tok; the 1-bit gather is **L3-latency-bound** (concurrency is the lever) |
| **GPU single-query decode** | **2.14×** (0.107 s/tok) — weight residency removes the per-call upload |
| **GPU prefill (T=256)** | **17.6×** vs CPU (resident on-device monolith) |
| **GPU generation** (`generate_cached`) | **12.7–15.8×** end-to-end (prefill on GPU + decode), byte-identical |
| **GPU batch decode (M=B)** | **1.68–1.91×** vs sequential (B rows share the weight read) |

Honest negatives (built, byte-exact, but **gated off** because they regressed on this GPU — *measure, don't
assume*): DP4A Q1 apply (~1.15× kernel-only, the masked sum is add-bound not multiply-bound), per-op-full
on-device decode, and the fully-resident batched decode (tiny-row RMSNorm occupancy at small B). The wins were
**weight residency** + **large-M prefill**, not tensor cores or "push everything onto the GPU." Consumer-GPU
**int64 is emulated** — the structural headwind.

---

## 6. Receipts — verifiable, notarized inference

Each generation can emit a **receipt**: commitments over (modelHash, input, output, integer execution trace), a
hash-linked local ledger entry, and **third-party-verifiable secp256k1 signatures** (the receipt carries only
the public key). The **CPU oracle is the canonical verifier**: a receipt produced on the GPU re-executes
byte-for-byte on a CPU-only machine, so no one needs the producer's hardware or any shared secret.

- **Real signing keys** are auto-generated at `0600` under `~/.local/trinote/keys/` on first receipted run
  (`model.key.json`, `counterparty.key.json`), or supplied with `--model-key` / `--counterparty-key`.
  (`--demo-keys` / pytest use the legacy deterministic HMAC vouch for byte-stable snapshots only.)
- **On-chain (optional):** a BSV OP_RETURN "third entry" via the notary wallet (`--onchain`, dry-run unless
  `--chain-confirm`). Same secp256k1 curve, so one identity spans the off-chain receipt and the on-chain entry.

---

## 7. How to run

Weights live out-of-repo (fetched separately); all generated state goes under `$BONSAI_NOTARY_HOME`
(`~/.local/trinote`), never into the repo tree.

```bash
cd integer_inference_engine/bonsai
scripts/fetch_weights.sh                       # GGUF + imported safetensors (one-time)
tools/build_bonsai_q1_kernel.sh                # CPU native kernel (byte-exact fast path)
tools/build_bonsai_q1_gpu.sh                   # optional: per-host CUDA producer (--gpu)

# Launch states (scripts/bonsai.sh — thin dispatcher over the CLI):
scripts/bonsai.sh receipted "What is a tensor?" -n 64   # deterministic + notarized receipt (GPU if built)
scripts/bonsai.sh json      "List three primes."        # structured {thinking,answer,receipt,bundle}
scripts/bonsai.sh repl                                  # interactive
scripts/bonsai.sh deterministic "Hello"                 # integer engine, no receipt
scripts/bonsai.sh onchain   "Notarize this." --chain-confirm   # + BSV third entry (spends BSV)
```

`BONSAI_GPU=0` forces CPU; `BONSAI_DRYRUN=1` prints the resolved command. Each receipted turn prints
`[receipt] <hash> VERIFIED` and writes a portable bundle to `~/.local/trinote/bundles/`.

---

## 8. File map (the integer engine for Bonsai-8B)

**Core compute:** `src/trinote/infer_int/reference_bonsai.py` (the engine), `src/trinote/determinism/fixedpoint.py`
(integer primitives), `src/trinote/model/rope_v2.py` (fixed-point RoPE), `src/trinote/infer_int/sampler.py`
(integer samplers).
**Accelerators:** `tools/bonsai_q1_kernel.c` + `q1_native.py` (CPU); `tools/bonsai_q1_gpu.cu` + `gpu_native.py`
(CUDA); `tools/build_bonsai_q1_*.sh`.
**Import / IO:** `infer_int/import_gguf_v2.py`, `import_bonsai_gguf.py`, `artifact_io_bonsai.py`,
`gguf_tokenizer_v2.py`.
**Runtime / receipts:** `infer_int/bonsai_runtime.py` (generate + `emit_and_verify_*` + `load_or_generate_signing_keys`),
`infer_int/verify.py`, `src/trinote/receipts/` (signing/ledger/verify/bundle).
**Entry points:** `cli/run_bonsai_cli.py`, `cli/json_mode.py`, `cli/receipt_bundle_cli.py`,
`cli/import_bonsai_gguf_cli.py`, `cli/quality_gate_bonsai_cli.py`, `scripts/bonsai.sh`, and the repo-root
`bonsai-cli` launcher.
**Tests:** `tests/test_bonsai_smoke.py` (engine/native parity, samplers, receipts), `tests/test_bonsai_gpu.py`
(GPU parity gate), `tests/test_receipt_bundle.py`, `tests/test_review_fixes.py`, `tests/test_parity.py`.
**Design notes (external, not in this repo):** the `Q1-BITMATMUL-REFORMULATION`, `GPU-FEASIBILITY`,
`IMPLEMENT-GPU-MODE`, `DETERMINISM`, and `SAMPLER-INTEGER` specs live in the author's `bonsai-notary` research
tree, not under this standalone repo — treat references to them as provenance, not in-repo paths.

---

## 9. Verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_bonsai_smoke.py -q     # engine + receipts byte-exactness
PYTHONPATH=src .venv/bin/python -m pytest tests/test_bonsai_gpu.py   -q     # GPU kernels == oracle (skips if no GPU)
```
Every generation also **self-verifies before emitting its receipt** (recomputes commitments + re-executes
bit-exactly → the `[receipt] … VERIFIED` line). A stored bundle is independently re-verifiable offline on the CPU
oracle (no console script is installed — invoke the module directly); `--reexec` requires `--artifact`:

```bash
PYTHONPATH=src .venv/bin/python -m trinote.cli.receipt_bundle_cli verify <bundle_dir> --reexec --artifact <A.safetensors>
```

---

## 10. How this maps to the standalone design

| Generalized design (`../integer_engine.md`) | Bonsai-8B instance (here) |
|---|---|
| `QuantCodec` interface | `Q1_0Codec` (1-bit, g128) — the proven codec |
| all-integer determinism core | `int-ref@bonsai-qwen3` (Qwen3, frac=16) |
| oracle + byte-identical producers | NumPy oracle + CPU-C + CUDA, parity-gated |
| receipts / verifiable inference | secp256k1 receipts + BSV third entry |
| GGUF ingest | PrismML `Bonsai-8B-Q1_0.gguf` → integer artifact |
| **BitNet ternary codec, Ollama interop** | **not yet** — the next generalization (see parent doc §4.2, §7) |

Bonsai-8B is the existence proof: a real 8B 1-bit model running byte-exact and receipted today. The standalone
engine generalizes its codec to BitNet ternary and its front door to Ollama.
