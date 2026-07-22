# Resident layer executor boundary

## Implemented boundary

CUDA ABI v5 now has two exact, bounded interfaces. The attention bank carries
the residual width separately from `n_heads * head_dim`, matching the real
model's 2048-lane hidden stream and 4096-lane query/attention projection. The established
`qk_rmsnorm_router` preprocessor remains available to the opt-in engine path.
The new request-scoped layer continuation removes its large round trip:

```text
host residual x
    -> H2D once
    -> integer RMSNorm with resident int64 gain
    -> resident int64 router projection
    -> stable low-index top-k + fixed-point sigmoid
    -> normalized h + K IDs + K gates retained in the attention bank
    -> D2H only compact IDs whose expert triplets are not bound
    -> lazy registration/binding of only those cold Q4_K/Q6_K slices
    -> resident attention, then continuation: selected MoE + residual on-device
    -> retained residual consumed by the next layer, or explicit final D2H
```

`qk_register_i64` gives dense gain/router tensors a registry kind that the
quantized projection APIs reject. `qk_rmsnorm_router` mirrors the canonical
Python arithmetic: exact sum-of-squares and integer square root, Python floor
division for negative normalized lanes, guarded gain multiplication, an exact
signed-int64 router contraction, stable low-index ties, and the committed
integer sigmoid polynomial.

At the committed `d_model=2048`, `n_experts=128` shape, preprocessing first
uses guarded 256-thread CUDA blocks. RMSNorm reduces exact uint64 squares only
after proving the entire row, shifted numerator, and gain product fit; the
router reduces signed int64 products only after proving an absolute contraction
bound for each row/expert pair. A failed fast proof is not a model error: a
per-row/per-logit flag dispatches only that item to the existing serial
`__int128` kernel in the same CUDA stream. The exact kernel remains responsible
for true canonical-envelope errors, so parallel reduction cannot change a bit
or weaken fail-closed behavior.

A local RTX 3070 standalone proxy benchmark at the real shape, with 30 warm
iterations and 300 timed calls, measured **0.347823 ms** per guarded fast call
versus **2.632553 ms** per deliberately forced exact-fallback call, or
**7.57×**. This is explicitly a fallback proxy, not a historical serial-library
comparison: the fallback measurement includes the guard kernels as well as the
serial exact work, and the standalone API also includes its transfer and
synchronization boundaries.

`qk_attention_bank_moe_configure` allocates only compact pointer/type metadata
for one layer. `qk_attention_bank_moe_bind` transactionally publishes one
already-registered gate/up/down triplet, with the device bound byte written
last. `qk_attention_bank_moe_begin` fuses preprocessing with resident
attention, retains the original residual and all routing metadata, and uses a
device compaction kernel so warm route IDs are never exposed to Python. Route
and cold-ID discovery is queued before attention and shares the attention
guard synchronization. The exact unresolved-ID set is then retained in host
request metadata; successful binds retire those entries, so
`qk_attention_bank_moe_continue` can refuse an incomplete bind without
rerunning GPU cold discovery or adding another route-ID round trip. After that
check, continuation gathers only selected handles, runs the unchanged exact
Q4_K/Q6_K expert kernels, and retains two's-complement-exact residual bits in
a ping-pong buffer. `qk_attention_bank_moe_export` is the explicit escape
hatch for a final result or a future controlled fallback.

The bridge proves each row's sum-of-squares fits signed i128 before dispatch.
The exact fallback repeats that guard and also rejects a router accumulation or
gain multiply outside the canonical int64 envelope. A true exact-envelope
rejection returns the standalone API to the arbitrary-precision host
implementation; the retained begin boundary fails loudly and poisons the
request without publishing partial metadata. Shape, allocation, and runtime
errors follow the same fail-loud publication rule. The begin boundary completes
all projection and storage preflight before it queues preprocessing. Once CUDA
work has been queued, any later host-side failure drains the default stream and
poisons the layer before returning, so reset or destruction cannot race kernels
that still reference request scratch or K/V storage.

The original preprocessor's engine integration is opt-in with either
`Engine(..., resident_preprocess=True)` or
`NMC_RESIDENT_PREPROCESS=1`. It applies to the 48 MoE layers. The profiling
driver exposes the same switch as `--resident-preprocess`. ABI-v5 layer
continuation now has two opt-in orchestration surfaces:

- `ResidentMoeTokenExecutor` runs an explicitly committed consecutive M=1 MoE
  layer sequence and is the synthetic parity surface.
- `Engine(..., resident_layer_executor=True).resident_decode_token(...)`
  preflights the real architecture, executes leading dense block 0, then runs
  exactly layers 1..48 through that retained executor.

`resident_decode_token` requires an already imported `ResidentAttentionCache`
and is deliberately not called by `Engine.generate`. Setting
`NMC_RESIDENT_LAYER_EXECUTOR=1` only enables this explicit method; it does not
change production generation. That separation prevents a synthetic proof from
being mistaken for real-model parity or a performance result.

## Memory bound

Only the norm gain and router matrix are dense int64 residents per MoE layer:

```text
bytes = layers * (experts + 1) * d_model * 8
      = 48 * 129 * 2048 * 8
      = 101,449,728 bytes (96.75 MiB)
```

Expert gate/up/down tensors are not preloaded. The existing `_ehandle` path
registers a slice only after its expert ID is selected and reuses that handle
on later tokens. This avoids treating the growable registry as permission to
upload all 18,432 possible expert slices; the project's compact full-weight
estimate is about 18 GB before K/V and execution workspaces.

The new request metadata is bounded independently of expert weights. For each
configured layer it stores three device-pointer tables, three quantization-kind
tables, and one bound byte per expert. M=1 execution scratch grows only for the
selected `K` experts (`2*K*expert_ffn + K*d_model` int64 intermediates), plus
five `d_model` rows, router logits, and compact route metadata. Repeating a
warm shape reuses those buffers. Attention probability scratch reserves the
request's committed maximum length on first use rather than reallocating one
position at a time. `ResidentAttentionCache.workspace_bytes()` exposes the
request-owned RoPE, K/V, binding, and continuation allocation total; registry
weights and serialized process-global scratch remain separately measurable by
the existing telemetry.

## Why this remains an explicit 1+48-layer orchestrator

The cold synchronization boundary is now compact: Python sees only IDs whose
triplets are not already bound, locates those slices in GGUF, registers them,
and resumes the prepared native layer. Warm routes have no weight transfer and
no route-ID publication. An isolated warm `L`-layer MoE chain performs exactly
`3*L + 1` D2H calls: one combined preprocessing/attention guard, one cold
count, and one continuation guard per layer, plus the final residual row. It
uses two explicit synchronization barriers per layer: the shared
preprocessing/cold-discovery/attention guard and the MoE continuation guard.
Forcing all expert weights resident would remove even the compact cold lookup,
but would violate the bounded-memory design and cannot run on the local 8 GB
development GPU.

The hardware promotion blocker was cleared on 2026-07-22. The isolated
real-model run bound source snapshot
`5c7f866c2c52361f8011511a36e26de101839c5b6a31bd1077e893125c1ecff0`
(196 entries, base commit
`d1cd09049b8ac153e2028985fef1eae32611a900`) and passed exact hidden,
full-vocabulary-logit, and greedy-token parity at every measured step. Warm
throughput increased from **8.4294542598 tok/s** on the established path to
**11.7969858823 tok/s** on the retained path, a **1.3994958059×** ratio. Combined
sampled peak memory was **7326 MiB**, or **0.2982413** of the device.

All seven report verdict checks passed: isolated GPU at startup, complete GPU
sampling, exact hidden/logit/token parity, warm allocation stability, no warm
throughput regression, combined peak below the limit, and local signed receipt
replay. The two-key secp256k1 receipt reproduced both signed commitments and
the established replay exactly; offline verification passed and chain
publication/broadcast remained disabled. The gate therefore no longer blocks
selection on hardware evidence. `Engine.generate` still does not select this
path automatically; changing that default is a separate product decision, not
unfinished parity, memory, receipt, or throughput validation.

The bound repo-external files in the operator-local deploy results directory
are
`20260722-195649_source-snapshot.json`,
`20260722-195649_resident-layer-gate.json`, and
`20260722-195649_resident-layer-gate-evidence.tar.gz`.

`tools/gate_resident_layers.py` remains the fail-closed reproducibility gate.
It runs a route-loading reference pass, a warmed measurement
of the established resident-attention path, and then the explicit retained
layer path in one process. Every hidden row, full-vocabulary logit row, and
greedy token must match exactly. The report also gates warm CUDA-allocation and
request-workspace stability, combined sampled GPU memory, and warm token rate.
It requires an isolated GPU at startup and publishes strict mode-0600 JSON by
atomic replacement. Use at least four generated tokens so the retained path
has a cold transition and two measured warm transitions:

```bash
PYTHONPATH=src NMC_BACKEND=cuda-resident python tools/gate_resident_layers.py \
  MODEL.gguf --tokenizer TOKENIZER_DIR --new-tokens 8 \
  --expected-model-sha256 MODEL_SHA256 \
  --output resident-layer-gate.json
```

The route warm-up is an actual call to the established `Engine.generate`, not
a second copy of gate-only orchestration. Its tokens must equal both measured
paths. Whenever exact established, retained, and public replay parity is
established, the gate attempts a two-key secp256k1-signed local receipt replay
independently of the throughput verdict. It disables both chain publication
and the dry-run broadcast log, offline-verifies the bundle, and proves that the
established replay and retained output reproduce the signed input/output
commitments. Receipt replay remains an independent required promotion check;
a throughput failure does not misreport it as an unattempted signature
failure.

Receipt support imports the shared Bonsai notary implementation. A deployment
that copies only `north-mini-code/` must separately place the matching Bonsai
`src/` tree on the host and install the receipt requirement before running:

```bash
python -m pip install -r requirements_receipts.txt
export NMC_BONSAI_SRC=/absolute/path/to/bonsai/src
test -f "$NMC_BONSAI_SRC/trinote/receipts/receipt.py"
```

The bundle defaults beside the JSON report as `<report-stem>-bundle/`; use
`--bundle-dir` to select an explicit artifact path. Signing keys remain local,
owner-readable state and only their public keys are included in evidence. The
mode-0600 gate report records the bundle/ledger paths for retrieval, but never
records either private-key path or private scalar. Resident GPU handles are
freed immediately after `model_hash` is computed and before keys are loaded.

## Promotion gates — passed 2026-07-22

The source-bound hardware report passed all seven verdict checks:

- exact full-model hidden, full-logit, and greedy-token parity;
- isolated GPU at startup;
- complete GPU sampling;
- stable warm allocations;
- retained warm throughput with no regression;
- combined compact-weight, K/V, preprocessing, attention, and MoE peak below
  the configured device limit;
- two-key local signed receipt replay with offline verification.

The receipt's additional local-only check also passed, with no chain broadcast.

Synthetic CUDA parity is covered by `tests/test_qk_cuda.py`, including a
two-layer retained chain that fails on an unbound cold route, resumes after a
lazy bind, plus a reusable three-layer token chain that matches the host oracle
byte-for-byte. Its second warm token adds zero CUDA allocations and leaves
request-owned workspace bytes unchanged. CPU-only ABI, lock, state, envelope,
memory, explicit opt-in, and exact 1+48 orchestration proofs are covered by
`tests/test_qk_cuda_api.py`.
