# Bonsai-27B controlled performance and idle diagnostics

The canonical Bonsai-27B verifier is the deterministic Python/NumPy CPU graph. The optimized resident CPU and
resident integer CUDA implementations are producers that must match it. The normal integer launcher attempts
resident CUDA first and falls back to CPU; `tools/bench_bonsai35.py --producer native` deliberately measures the
resident **CPU** executor so CUDA availability cannot silently change a CPU result. Compare any path with PrismML
only after removing other inference processes: idle teams and GPU residency can invalidate TTFT/decode numbers.

## Runtime thread policy

`../bonsai-integer-27b-cli` establishes thread policy before Python imports NumPy or loads the OpenMP runtime:

| Variable | Default | Purpose |
|---|---:|---|
| `OPENBLAS_NUM_THREADS` | `1` | prevent a second BLAS scheduler |
| `MKL_NUM_THREADS` | `1` | prevent a second BLAS scheduler |
| `NUMEXPR_NUM_THREADS` | `1` | prevent a second expression scheduler |
| `OMP_NUM_THREADS` | physical cores visible to the process | native Q1 worker team |
| `OMP_DYNAMIC` | `FALSE` | stable team size |
| `OMP_WAIT_POLICY` | `PASSIVE` | sleep between native regions |
| `OMP_PLACES` / `OMP_PROC_BIND` | `cores` / `close` | stable core placement |
| `OMP_MAX_ACTIVE_LEVELS` | `1` | no nested OpenMP teams |
| `GOMP_SPINCOUNT` / `KMP_BLOCKTIME` | `0` / `0` | stop GNU/Intel post-region spinning |
| `TRINOTE_ORACLE_Q1_THREADS` | `4` | pure-NumPy verifier output-row workers; no native producer handle |

An already exported standard variable is treated as an explicit override. `BONSAI_INTEGER_27B_THREADS` provides
a launcher-specific OpenMP override; `BONSAI_INTEGER_27B_BLAS_THREADS` changes all three math-backend defaults.
The CLI's `--threads N` has highest precedence and is resolved by the shell launcher before Python starts.

```bash
BONSAI_INTEGER_27B_THREADS=4 ./bonsai-integer-27b-cli repl
BONSAI_INTEGER_27B_BLAS_THREADS=2 ./bonsai-integer-27b-cli repl
./bonsai-integer-27b-cli repl --threads 8
BONSAI_DRYRUN=1 ./bonsai-integer-27b-cli repl --threads 8
```

Changing a thread count changes scheduling only; native integer output remains byte-identical. The oracle's Q1
workers own disjoint output-row chunks and never share a reduction, so their scheduling is also byte-invariant.
Set `TRINOTE_ORACLE_Q1_THREADS=1` for the historical serial-oracle baseline. Use one BLAS thread unless a
controlled benchmark proves otherwise.

The Python entry points also accept `--cpu-threads N` (an alias of `--threads`) and set OpenMP, OpenBLAS, MKL,
BLIS, NumExpr, VecLib, and the pure-oracle Q1 row workers consistently before model/shared-library load. This is the appropriate interface for
notary automation that does not enter through the shell launcher.

## Verified-token latency and deterministic routing

Generation throughput alone is not the product latency: a receipt-bound turn also constructs the receipt,
re-executes every committed output with a fresh oracle, and emits the ledger entry. Use `--run-report PATH` on
`trinote-run-bonsai` or `trinote-receipt-bundle verify` to atomically retain a `receipt-run/v1` record with these
phases separated. The record contains hashes, engine/strategy paths, thread/resource bounds, token counts, and
cleanup state; it deliberately excludes prompt text and signing-key paths, redacts host-local absolute paths,
and is created mode `0600`.

The release verifier workload is checked in as
`tests/fixtures/bonsai35_19x64_golden.json`. Its 19 input IDs, 64 output IDs, visible-byte commitment, artifact
digest, and GGUF digest are the acceptance boundary. Benchmark the exact 20-output product baseline across fresh
processes (required for trustworthy OpenMP settings) with:

```bash
PYTHONPATH=src .venv/bin/python tools/bench_bonsai35_verifier.py \
  --artifact "$BONSAI_INTEGER_27B_ARTIFACT" \
  --threads 1,2,4,8,16 --output-counts 20 \
  --out /tmp/bonsai35-verifier-benchmark.json \
  --policy-out /tmp/bonsai35-verifier-policy.json \
  --oracle-policy-out /tmp/bonsai35-verifier-oracle-policy.json
```

Every cell must reproduce all committed tokens. The variants are pure-oracle teacher forcing, byte-identical
native teacher forcing, and byte-identical native cached replay. The generated `receipt-verifier-policy/v1` is
bound to the artifact SHA-256 and routes deterministically by committed input/output token counts. Generated
policies fail closed for a token-count pair absent from the measured matrix, or when that point does not select
an exact first-match rule with the measured thread count. Each benchmark cell records requested and effective
workers separately and cannot publish a policy when they differ. Add the product lengths you intend to verify
to `--output-counts` instead of extrapolating a timing winner. Apply it with:

```bash
trinote-receipt-bundle verify receipt.tar.gz --reexec \
  --artifact "$BONSAI_INTEGER_27B_ARTIFACT" \
  --strategy-policy /tmp/bonsai35-verifier-policy.json \
  --run-report /tmp/receipt-verify-run.json
```

An explicit `--verifier-engine native` fails if the native kernel cannot be enabled; it never silently turns a
measured native route into an oracle run. Strategy selection changes only the exact recomputation schedule and
never enables sampled/partial verification. The policy also pins the single measured process-wide thread count;
an explicit conflicting `--cpu-threads` is rejected instead of silently applying an unmeasured performance route.
When a policy is supplied, explicit verifier-engine/strategy overrides are rejected: the measured route is
authoritative, and run evidence never labels an overridden route as policy-applied.

Fresh-oracle receipt issuance can use the same routing machinery without relaxing its independent-verifier gate.
Pass the benchmark's oracle-only policy as
`trinote-run-bonsai --verify-mode fresh-oracle --receipt-verify-policy POLICY.json`. Issuance rejects a policy
whose selected rule names the native engine; native policies remain valid for later third-party bundle replay.

For the producer, `--require-gpu` implies `--gpu` and fails on residency failure or a later CUDA range/launch
guard. Normal `--gpu` retains canonical CPU replay as a portability policy. Release acceptance and CI must use
the fail-closed flag so a CPU fallback cannot be mislabeled as GPU evidence.

## Controlled JSON benchmark

`tools/bench_bonsai35.py` is a controller/worker harness. The controller imports no NumPy. It sets thread policy,
samples background CPU, rejects another native engine, pins a physical-core or SMT affinity set, and only then
starts workers. Results default to
`$BONSAI_BENCHMARKS_DIR/results/bonsai35/` (normally
`~/.local/trinote/benchmarks/results/bonsai35/`), never the checkout.

Start with a raw, one-token greedy baseline and an exact chat run:

```bash
cd bonsai
.venv/bin/python tools/bench_bonsai35.py \
  --raw-ids 12675 --max-new 32 --sampler greedy \
  --producer native --condition second-turn --repetitions 5 --threads 8

.venv/bin/python tools/bench_bonsai35.py \
  --prompt 'Hi' --chat --max-new 32 --sampler bonsai27-rec \
  --condition second-turn --repetitions 5 --threads 8
```

Use `--producer oracle` for the controlled canonical NumPy baseline and `--producer native` for the optimized
resident CPU path. The native worker requires the release model ABI and records resident call/team, Q1-group,
LUT hit/fallback, selected ISA/LUT width, and cache-width counters; it must not silently fall back to the
Python-orchestrated graph. Both implement the same committed integer graph, but only `oracle` is the verifier.
Do not compare an isolated clean native run against an older contaminated profile.

For a matched before/after CPU comparison, use `--producer legacy-native`. This is a controlled reconstruction
of the pre-fusion Python/native-primitives architecture behind the documented 15.3--16.1 second diagnostic
range: it requires the packed native Q1 library, retains Python graph/cache orchestration and the historical
separate Q1 prepare plus prepared-multi apply calls, and disables the resident model executor, fused Q1
prepare/apply ABI, guarded int32 LUT, scale cache, and post-profile native GDN. Native RMSNorm, SiLU, and decode
attention remain enabled because they were already separately dispatched in the captured profile; disabling
them would manufacture a slower baseline that was never measured. The JSON records the previous, forced, and
effective value of every replay toggle plus the actual source, kernel build, compiler, Python, and NumPy
identities.

```bash
.venv/bin/python tools/bench_bonsai35.py \
  --raw-ids 12675 --max-new 1 --sampler greedy \
  --producer legacy-native --condition second-turn --repetitions 5 --threads 8
```

This lane does not claim to be the old binary: it runs current code under explicit compatibility controls and
controlled thread/affinity policy. It intentionally does not recreate the original busy host or active thread
pools, so 15.3--16.1 seconds is provenance context rather than a pass band. Compare it only with a `native` run
using the same artifact/kernel identity, host controls, prompt, condition, and repetition count.

Conditions have deliberately different process lifetimes:

- `cold-process`: each measured repetition loads a fresh process and artifact, with no warm-up;
- `warm-process`: each repetition gets its own process and one explicitly unmeasured full turn first;
- `second-turn`: one process loads once, performs one unmeasured turn, then runs all repetitions with allocator
  and native workspaces resident.

All three conditions start every measured prompt from an empty semantic model cache. `warm-process` and
`second-turn` preserve the loaded artifact, native handle, packed weights, descriptors, arenas, and allocator
state. Each resident prefill ABI resets recurrent state, convolution history, and position as part of the timed
prefill call; valid KV is defined by the reset position. The unmeasured warm-up is never a prefix of the measured
request.

The JSON separates GGUF metadata load, artifact load, native enable, artifact hashing, tokenizer startup/reuse, prompt
prefill, compute and end-to-end
TTFT, first cached decode, token 3-32 and 33-128 steady windows, final norm, output projection, and sampling. Each
iteration includes exact input/output IDs, peak RSS, CPU time/utilization, page faults, context switches, and device
memory. Both oracle and resident iterations use the same producer-independent cache commitment: ordered logical
int64 state/conv/K/V tensors (only the valid KV prefix), cache position, and final residual. The resident ABI
exports its context-strided caches into those oracle shapes outside the timed region; its FNV fingerprints remain
diagnostics and are not acceptance commitments. Environment identity includes source
revision/dirty state, artifact SHA-256, kernel SHA-256 and ELF build
ID/compiler comment, Python/NumPy/platform, CPU topology/affinity, GPU identity, and all relevant thread variables.
For the resident path, final norm/output are fused into the prefill/decode ABI, so their standalone timing fields are
zero and their cost is already included in prompt prefill or decode graph time.

The aggregate contains median, p10, p90, and coefficient of variation. Its five-percent variation gate requires
at least five samples; a one-run smoke is explicitly ineligible. The gate is evidence,
not a reason to hide a result: raw repetitions remain in the file. A contaminated run is rejected before model load.
`--allow-busy` is available for diagnostics, but the JSON remains annotated with every rejection reason.

The synthetic parity suite has exercised 144 thread/tile/LUT/ISA configurations, but that is correctness evidence,
not controlled throughput evidence. To build the still-pending selected-executor performance matrix, vary relevant
runtime controls independently rather than combining unrelated runs:

```bash
for threads in 1 2 4 8 12 15; do
  .venv/bin/python tools/bench_bonsai35.py --raw-ids 12675 --max-new 32 \
    --threads "$threads" --condition second-turn --repetitions 5
done
```

Use explicit raw ID lists to benchmark exact prompt lengths. Do not report prefill tokens/second as decode
tokens/second; the harness keeps these fields separate. The resident model executor owns its scheduling and does
not use the Python `_project_many` Q1 chunk setting. `--prefill-q1-chunk` remains useful for primitive/legacy
Python-orchestrated diagnostics, but varying it is not evidence about the selected resident executor.

## Accepted CPU timing evidence

The final records use the same artifact, installed CPU library, content-bound source set, physical cores 0-7,
one BLAS thread, passive OpenMP waits, greedy sampling, and exact input IDs on the i7-10700F. The independent
comparison tool rejects a source/kernel/workload mismatch, fewer than five repetitions, failed variation gates,
or unequal output/cache commitments.

| Gate | Legacy median | Resident median | Speedup | Variation |
|---|---:|---:|---:|---|
| Raw-1 prefill | 1.317650 s | 0.292663 s | 4.5023x | pass, 5 reps |
| Raw-1 compute TTFT | 1.358838 s | 0.292665 s | 4.6430x | pass, 5 reps |
| Decode tokens 3-32 | 1.476946 s/token | 0.346441 s/token | 4.2632x | pass, 5 reps |
| Decode tokens 33-128 | 1.474023 s/token | 0.351101 s/token | 4.1983x | pass, 5 reps |
| 32-token prefill | 40.120097 s | 9.640637 s | 4.1616x | pass, 7 warm-process reps |
| 32-token compute TTFT | 40.169670 s | 9.640641 s | 4.1667x | pass, 7 warm-process reps |
| 128-token prefill | 158.813912 s | 43.092396 s | 3.6854x | pass, 5 second-turn reps |
| 128-token compute TTFT | 158.863708 s | 43.092400 s | 3.6866x | pass, 5 second-turn reps |

The three fail-closed comparison records are:

- `results/bonsai35-cpu-raw1-out128-comparison-acceptance.json`, SHA-256
  `903bd008c6d8b23de3da11af04b48d850d19c9ca8163c1fffa9b1cdd6c86455e`;
- `results/bonsai35-cpu-prompt32-out1-comparison-acceptance.json`, SHA-256
  `4587a2c188a1b5790b796508bc4b38b01b8a34709ec152aef76fcce4bcdaf830`;
- `results/bonsai35-cpu-prompt128-out1-comparison-acceptance.json`, SHA-256
  `6f51b96605f567cf919b3029868fdc06138433f702145405a369d9c5cd2b2700`.

The raw acceptance JSON stays outside the published repository in the operator's external results directory;
the SHA-256 values above bind those records. Linux `perf` was unavailable on the acceptance host (`perf` not
installed and `kernel.perf_event_paranoid=3`), so the JSON preserves that machine-readable limitation rather
than estimating PMU counters. Logical executor counters show AVX2, 32-bit activation LUT selection, one resident
team entry per prefill/decode call, and zero LUT fallback in the accepted native runs.

The resident CPU executor stores KV as int64. The hardened CUDA producer instead uses guarded int32 KV and its
populated-4K record reports `kv_bits=32`; paired K/V guards preflight every lane before committing either cache
row. A CPU benchmark must still report `cache_width_bits=64`: CUDA narrow-KV evidence does not establish a CPU
int32-KV path.

For GPU measurements, the regular PrismML Bonsai-27B process (about 4.17 GiB) and the resident integer graph
(6,362,562,560 bytes observed live, with a 6,740,049,920-byte proof peak) are mutually exclusive on the 8 GiB
RTX 3070. Stop/unload PrismML before a resident integer GPU run, record that transition, and restore it only after
cleanup. A launcher result that fell back to CPU because PrismML owned the card is a CPU result, not a GPU benchmark.

## Accepted GPU hardware evidence

The fail-closed GPU hardware gate passed on 2026-07-22 on an RTX 4090 compiled for `sm_89`. It was bound to
source snapshot `5c7f866c2c52361f8011511a36e26de101839c5b6a31bd1077e893125c1ecff0` (base `d1cd090`,
196 entries) and artifact SHA-256 `7eab414ceff3fff1489053d415d0c6adb1e646e552d091cc1a898d0456adf3fb`.
The operator-local deploy results are `20260722-195649_bonsai-hardware-gate-summary.json` and
`20260722-195649_bonsai-hardware-gate-evidence.tar.gz`. Their three bound reports passed: `golden19x64`
(`ef50f06773716d2431c64d1337a169611c1b3f675edc9aa437c63eeca9676847`), `full128Parity`
(`fcbb405898c176b989e6849c71649df372373e0ce70e3f803284cb445fe9f351`), and `populated4kThroughput`
(`2291fdde3ef830adb9ef5618139c3ddb22bc1977d7e784675af2d019cb2dc1e2`).

The 19-input/64-output golden record passed all 16 gates. Its output-ID SHA-256 is
`6528f94af6398d9bfa8c85099b5d1e2047ac133eafb0b20be00f494d71f125bf`, and its visible-byte SHA-256 is
`0326877dd8a72d427519acf2c33070dbe5049dd91136c29b5e6cbee7ff306c05`.

The full-128 record passed all seven acceptance fields: `array_parity`, `artifact_bound`,
`device_resident_token_embedding`, `device_under_7_5_gib_ceiling`, `generated_token_parity`,
`one_graph_submission_per_consumed_token`, and `raw_hi_next_token_11`. For prompt IDs `[12675]`, CPU and GPU
consumed and generated exactly 128 tokens. The generated int64-LE token SHA-256 was
`8faec4eb5a7a17d0d472474f76a8113c74974bf725c4729b13be04998ade6704`; the first eight IDs were
`[11, 353, 2688, 4313, 310, 1791, 264, 4145]`, and the last eight were
`[2081, 13, 353, 599, 264, 1103, 314, 3470]`. Every CPU/GPU int64 array was equal:

| Array | Shape | Shared SHA-256 |
|---|---|---|
| convolution history | `[48, 3, 10240]` | `8062c45545594ff8ddbe1e3fd1691292952603949d148d4e8b8593565eb16a0a` |
| K cache | `[16, 4, 128, 256]` | `2225963b49a54382fd7be1c2ea2b75ab854c14e2f0073b1700b7c0783a4ebdc5` |
| logits | `[248320]` | `0daa98539fea4101c4f2d7a8f76916acd74a3936b6db467ff7a8f301e4358ac3` |
| recurrent state | `[48, 48, 128, 128]` | `b1b6839a6b1487efa0e54eaaf98272f331e36060651d03160c8514daf53fa011` |
| layer trace | `[65, 5120]` | `c76b8559102bceca02361d5fdbda546df24be63a934adecd4f4b9889ce14c2db` |
| V cache | `[16, 4, 128, 256]` | `95792b453b08fa251fe7e2f322119cd54261e0af7d817df5eb3e967a283e5a8b` |

The canonical NumPy oracle took 1,646.6276 seconds; steady GPU decode reached 22.6762 tokens/second. The populated
4K record passed all 15 acceptance fields: `consumed_id_count_exact`, `context_not_poisoned`,
`context_position_exact`, `device_embedding_only`, `device_under_7_5_gib_ceiling`, `diagnostic_trace_disabled`,
`generated_id_count_exact`, `graph_ready`, `last_32_at_least_10_tokens_per_second`,
`live_memory_queries_available`, `model_input_host_bytes_exact`, `one_graph_submission_per_consumed_token`,
`projection_grouping_enabled`, `raw_hi_next_token_11`, and `token_id_input_mode`. It completed 4,096 consumed
positions and 4,096 graph launches, with a last-32 median of 18.7055 tokens/second. Live use was 6,202,458,112
bytes and the proof peak was 6,795,952,128 bytes, both below the 8,053,063,680-byte ceiling.

This was local hardware acceptance evidence; no chain transaction was broadcast. It does not complete or replace
the still-pending selected-executor CPU tuning matrix described above.

## Q1 hardware counters

The same harness provides isolated Q1 preparation and prepared-apply modes at actual model widths. It records logical
prepare/apply/OpenMP-region counts and LUT bytes even without privileged counters:

```bash
.venv/bin/python tools/bench_bonsai35.py --mode q1-prepare \
  --q1-in-features 17408 --q1-out-features 5120 --q1-iterations 20 --perf

.venv/bin/python tools/bench_bonsai35.py --mode q1-apply \
  --q1-in-features 5120 --q1-out-features 17408 --q1-iterations 20 --perf
```

`--perf` uses Linux `perf stat` only when a permission probe succeeds. The result records cycles, instructions,
cache/branch events, frontend/backend stalls, context switches, migrations, and page faults, or records why counters
were unavailable. Override `--perf-events` for host-specific L1/L2/LLC or uncore IMC events. DRAM bandwidth and
OpenMP barrier time require host-specific uncore/OMPT tooling and are explicitly reported as unavailable rather than
invented from wall time.

On the acceptance host, `perf` was not installed and `kernel.perf_event_paranoid` was `3`. Therefore only logical
engine counters and ordinary process/resource measurements have been collected; privileged cycles, instructions,
cache/branch misses, stalls, uncore DRAM bandwidth, and OMPT barrier timing remain unavailable. A future privileged
run must record the probe and raw counter output—it must not backfill these fields from wall time.

## Idle acceptance diagnostic

This launch-and-own test measures an independent CPU command, starts the real REPL on a pseudo-terminal, runs a
one-token `Hi` turn so every native pool has actually executed, waits for the next prompt, lets pools settle for five
seconds, samples idle CPU for five seconds, then repeats the command while the REPL remains idle:

```bash
cd bonsai
BONSAI_INTEGER_27B_GPU=0 .venv/bin/python tools/diagnose_bonsai_idle.py
```

Forcing CPU makes this an unambiguous OpenMP/BLAS idle result. Omit the override only when intentionally testing
the default CUDA-attempt/fallback launcher policy and record which producer actually started.

Acceptance requires less than one percent of one CPU core and no more than ten-percent slowdown of the independent
command. The JSON includes actual thread count/states, context switches, and the process's thread environment. To
inspect an existing tmux REPL without sending it any signal:

```bash
.venv/bin/python tools/diagnose_bonsai_idle.py --pid "$(pgrep -n -f 'trinote.cli.run_bonsai_cli.*--engine native')"
```

The existing-PID mode only performs the idle CPU check because it has no pre-launch competitor baseline.
