# Bonsai-27B GGUF on Linux

This project supports [`prism-ml/Bonsai-27B-gguf`](https://huggingface.co/prism-ml/Bonsai-27B-gguf)
on Linux in two ways:

- `bonsai-integer-27b-cli` runs the Qwen3.5 hybrid graph in Trinote's deterministic fixed-point engine. It
  attempts the optional resident CUDA producer by default and cleanly falls back to the optimized resident
  CPU executor. `BONSAI_INTEGER_27B_GPU=0` forces CPU.
- `bonsai-27b-cli` runs the original GGUF through PrismML's llama.cpp CUDA fork. It is much faster, but its
  floating-point execution is not receipt-capable.

## Install

From the `integer_inference_engine` checkout:

```bash
bonsai/scripts/install_bonsai_27b_gguf.sh
bonsai/scripts/fetch_bonsai_27b_gguf.sh
./bonsai-27b-cli "Explain Merkle proofs." -n 256
```

The runtime and model stay outside the checkout under `$BONSAI_NOTARY_HOME` (default
`~/.local/trinote`). The installer is Linux x86-64/CUDA-specific and uses PrismML's official CUDA 12.4
binary. The model repository is public, but `HF_TOKEN`, `BONSAI_TOKEN`, and `HF_TOKEN_FILE` are supported;
`~/.hugging_face_token.txt` is detected automatically when present.

The downloads are pinned and verified before installation:

| Component | Revision | SHA-256 |
|---|---|---|
| `Bonsai-27B-Q1_0.gguf` | `0cf7e3d21581b169b4df1de8bf01316000e2fbb7` | `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0` |
| PrismML llama.cpp CUDA archive | `prism-b9591-62061f9` / `62061f91088281e65071cc38c5f69ee95c39f14e` | `67c64046abcf73bf489e27c9ebe7525f5b77c58db9490d1d711efe6e17bf2975` |

The launcher uses the model-card sampling preset (`temperature=0.7`, `top-p=0.95`, `top-k=20`,
`min-p=0`) and fully offloads the model (`-ngl 99`). Its default 4096-token context is chosen to fit an
8 GB GPU; override it with `BONSAI_27B_CTX_SIZE`, and use `BONSAI_27B_NGL` to change offload behavior.

```bash
BONSAI_27B_CTX_SIZE=8192 ./bonsai-27b-cli "Summarize this design." -n 128
./bonsai-27b-cli repl
BONSAI_DRYRUN=1 ./bonsai-27b-cli "show the resolved command"
```

## Native integer engine

Build the packed-Q1 CPU kernel, then import the downloaded GGUF into the Qwen3.5 artifact format once:

```bash
cd bonsai
tools/build_bonsai_q1_kernel.sh
tools/build_prismml_teacher_forced.sh
PYTHONPATH=src .venv/bin/python -m trinote.cli.import_bonsai35_gguf_cli \
  --gguf ~/.local/trinote/models/Bonsai-27B-Q1_0.gguf \
  --out ~/.local/trinote/models/Bonsai-27B-Q1_0-int-qwen35.safetensors \
  --context-len 4096
cd ..

./bonsai-integer-27b-cli "Explain Merkle proofs." -n 64
./bonsai-integer-27b-cli repl
```

### Optimized CPU producer

The release-shape CPU executor validates and pins artifact pointers once and owns the recurrent state,
convolution history, KV cache, and reusable workspaces. A prefill/decode call enters one native ABI and one
persistent OpenMP team for embedding, all 64 layers, final RMSNorm, and either logits or exact greedy argmax.
The Python/NumPy Qwen3.5 graph remains the canonical verifier; the C executor is an optimized producer and
never verifies itself.

Real raw/chat traces, logits, generated IDs, portable/AVX2 dispatch, and guarded int32/uint64 activation-LUT
paths match that oracle. The final controlled i7-10700F publication records use eight physical OpenMP workers,
one BLAS thread, passive waits, exact prompt IDs, at least five repetitions, and fail-closed provenance and
commitment checks:

| Metric | Frozen legacy producer | Resident CPU producer | Speedup |
|---|---:|---:|---:|
| Decode tokens 3-32 | 1.476946 s/token | 0.346441 s/token | 4.2632x |
| Decode tokens 33-128 | 1.474023 s/token | 0.351101 s/token | 4.1983x |
| 32-token prefill | 40.120097 s | 9.640637 s | 4.1616x |
| 128-token prefill | 158.813912 s | 43.092396 s | 3.6854x |

The matched producer pairs have identical output/cache commitments and passing variation gates. The benchmark
guide gives TTFT values, raw JSON paths, hashes, and the unavailable-PMU qualification.

### Resident deterministic CUDA producer

On an RTX 3070, build the optional per-host CUDA library before launching:

```bash
cd bonsai
tools/build_bonsai_q1_gpu.sh
cd ..
./bonsai-integer-27b-cli repl
```

The 27B launcher tries the resident CUDA graph by default and performs a clean CPU replay when CUDA is
missing, the memory proof fails, or an exact integer range guard fires. Set `BONSAI_INTEGER_27B_GPU=0` to
force the CPU producer. Runtime-only BMMA weight packing does not alter the artifact or receipt digest.

The hardened real 4K-context allocation proof on the RTX 3070 registered all 498 Q1 matrices with their
committed int32 scales (4,202,086,400 unique resident bytes) and 357 static integer buffers (23,785,488 bytes).
The release artifact does not tie embedding and output: their committed Q1 digests are respectively
`d84aa718ba9568a9a7f13cb66b0aad72957c24176f92fac5299dcba502c8ecf9` and
`40148b5dd2437a798ebd5280b5bd777aca48165fae3d8283e94cccbbbbec1306`, so no false alias is claimed. The
same identity check aliases a genuinely tied synthetic fixture to one resident handle.

Production decode transfers one 8-byte token ID per step and expands the selected packed-Q1 embedding row on
device; it does not construct or upload a 5,120-element host embedding. K and V use transactional guarded
int32 storage. One preflight examines both complete rows without writing, and the following commit writes both
only if every lane is safe. An unsafe K or V lane therefore leaves both cache rows and their max guards
untouched, poisons the context, and forces canonical CPU replay. Debug export sign-extends the cache exactly.
The two caches allocate 536,870,912 bytes total, saving exactly 512 MiB versus int64 storage.

The source-and-binary-bound populated-cache run consumed all 4,096 positions with exactly 4,096 token-ID graph
submissions, 32,768 model-input host bytes, zero embedded-row submissions, and no poison. Its final 32 decodes
measured 96.131 ms/token median (10.4025 tok/s), satisfying the 10 tok/s gate at an actually populated 4K
cache. With a 362,610,688-byte live baseline, the conservative proof peaked at 6,740,049,920 bytes, leaving
1,313,013,760 bytes below the 7.5 GiB ceiling; observed live usage peaked at 6,362,562,560 bytes at position
4,096. The bound record has SHA-256
`f0cbb0a6cda982fe5cb8cdf91f7aa545740f0c490756f10c9db753d768be715a` and commits consumed/generated ID
digests `1f759eadc519ea624d9eda41ef89a2f4a2476d63c663107d45f2b77cfa3a6810` and
`30d764c1545375728de9663a0fa32e29da44bb1283d0026df828f52a00abc62e`. Prompt prefill is currently
sequential and is reported separately from cached decode.

The regular PrismML process uses about 4.17 GiB by itself. It cannot coexist with the resident integer graph
on an 8 GiB card. Before uploading any model tensor, the integer launcher queries device free/used memory and
checks the complete tensor/workspace plan plus a 1 GiB CUDA-module/allocator reserve against the 7.5 GiB
ceiling. If `bonsai27` (or another GPU model server) is already loaded, it reports a GPU exclusivity conflict
and continues on CPU without a partial multi-gigabyte upload. For an intentionally concurrent pair of REPLs,
launch the integer session with `BONSAI_INTEGER_27B_GPU=0`; stop the regular session and restart the integer
launcher with its default environment when resident integer speed is desired.

CUDA is a producer, not a weaker verifier. The hardened, binary-bound real-model soak consumed 128 tokens with
exactly 128 graph submissions and compared against native-disabled pure NumPy. Post-layer residuals, all 48 Q30
states, convolution histories, all 16 attention K/V caches, final logits, and all generated IDs were byte-exact.
The context accepted 128 token IDs, zero embedded rows, and exactly 1,024 model-input host bytes without poison.
Its generated-ID digest is `8faec4eb5a7a17d0d472474f76a8113c74974bf725c4729b13be04998ade6704`.
The canonical report has SHA-256 `914979be8aa480ee92ea24925bee0e01032ad2bb5eb55c14f9ff4a427aa8625d`.
A guarded failure poisons the context before any further output can be trusted; the CLI discards it and replays
from the original prompt on CPU. This raw-`Hi` soak is release evidence, but does not by itself generalize to
every prompt, sampler, or cross-configuration path.

Both hardened records bind CUDA library SHA-256
`3b5d72801c54dfaea2eb699212b376c82b58363caef8b2035fc10f0386893bce`, ELF build ID
`b6295728475d88da2af9ca80192bbee4c4713f8d`, and acceptance-tool SHA-256
`f6cc88ce21a14edcb1d8c172f2af5adce87489ffb8ad3e68ba480fbf003c7a63`. The focused acceptance/tool,
resident-graph, transactional-KV, tied-weight, and long-attention suite passes all 33 tests, including lengths
1, 2, 31, 32, 127, 128, 512, 1,024, 4,095, and 4,096.

Produce the default release-bound 128-output/128-submission acceptance soak with:

```bash
cd bonsai
PYTHONPATH=src .venv/bin/python tools/verify_bonsai35_gpu_real.py \
  --json-out ~/.local/trinote/benchmarks/bonsai35-gpu-real.json
```

The command writes one canonical JSON document to stdout and atomically writes the same bytes to `--json-out`.
It exits nonzero for an artifact-digest mismatch, GPU exclusivity/preflight failure, poisoned decode, graph
submission-count drift, generated-token mismatch, or any residual/state/convolution/K/V/logit mismatch. Its
CPU comparison is the pure NumPy/Python oracle; native acceleration is explicitly disabled for the release
gate. The report binds the artifact, acceptance-tool source, checkout revision/dirty-state digest, exact loaded
CUDA library SHA-256/ELF build ID, compiler metadata, Python, and NumPy. Keep the 128-output parity record and
the populated-4K throughput record separate: the former is the long correctness gate, while the latter proves
the release-context performance and memory targets.

The separate populated-cache performance procedure consumes all 4,096 positions on GPU, commits every
generated ID, and times the final 32 decode steps without running or exporting the CPU parity trace:

```bash
PYTHONPATH=src .venv/bin/python tools/verify_bonsai35_gpu_real.py \
  --throughput-context 4096 \
  --json-out ~/.local/trinote/benchmarks/bonsai35-gpu-4k-throughput.json
```

Its report includes proof and live device memory, exact graph-launch/position/poison state, every final-32
timing sample, and median tokens/second. It fails if context or launch accounting drifts, the graph is poisoned,
or the final-32 median misses the 10 tok/s RTX 3070 target. The default invocation remains the 128-output full
CPU-oracle parity gate.

The second build command compiles two small helpers against the exact pinned libllama revision: a
teacher-forced logits harness for the model-quality gate, and a persistent vocab-only tokenizer. The first
REPL tokenization starts that helper; later turns reuse the loaded vocabulary instead of reparsing the 3.5 GB
GGUF. Set `BONSAI_PERSISTENT_TOKENIZER=0` to force the slower one-shot `llama-tokenize` fallback.

For repeated fixed prefixes, `--prompt-cache` writes a content-addressed, artifact-bound recurrent/KV state
under `$BONSAI_NOTARY_HOME/prompt-cache/bonsai35`. Every tensor shape and aggregate tensor digest is verified
before reuse; a rejected cache is rebuilt, and cache bytes never enter the model or receipt commitment.

The integer launcher configures a single-threaded NumPy/BLAS backend and a passive, physical-core OpenMP team
before Python starts. The fresh pure-NumPy receipt oracle separately uses four disjoint Q1 output-row workers by
default (`TRINOTE_ORACLE_Q1_THREADS=1` restores the serial baseline); they do not invoke a native producer or share
a reduction. This avoids competing math schedulers and idle spin at the REPL. Host-aware overrides, controlled
TTFT/decode benchmarks, Q1 hardware-counter modes, and the idle acceptance test are documented in
[`BONSAI-27B-BENCHMARKING.md`](BONSAI-27B-BENCHMARKING.md).

The imported artifact is about 4.23 GB. It commits a distinct 64-layer Qwen3.5 graph: 48 Gated DeltaNet
recurrent layers, 16 gated full-attention layers, integer IMRoPE tables, and integer softplus/decay lookup
tables. Residual activations use Q16; the small recurrent state and pre-normalization scores use Q30 so they
do not collapse before the model's `1e-6` RMSNorm epsilon is applied.

The root launcher deliberately passes `--no-receipt`. The graph is deterministic and re-executable, but the
repository's original identity is bound to Bonsai-8B and must not be reused. Both the CLI and the shared receipt
API fail closed if a caller attempts to bypass this default: 27B issuance requires `--verify-mode fresh-oracle`,
a distinct `int-ref@bonsai-qwen35` identity, and a sibling quality-gate JSON whose SHA-256, artifact digest,
GGUF digest, pinned Prism runtime, thresholds, and multi-prompt PASS are all verified. The verifier object must
be a fresh Qwen3.5 CPU oracle with native producers disabled. This is a library-level hard gate, not merely a
launcher default. The identity and gate are unsigned local evidence: their SHA-256 link detects mismatch but
does not make a pair that can be replaced together cryptographically unforgeable. The accepted
identity/quality-gate/real-receipt artifacts are not currently shipped, so 27B
receipt issuance remains disabled even though deterministic inference works.

Build and run the exact libllama fidelity gate with:

```bash
tools/build_prismml_teacher_forced.sh
PYTHONPATH=src .venv/bin/python -m trinote.cli.quality_gate_bonsai_cli \
  --artifact ~/.local/trinote/models/Bonsai-27B-Q1_0-int-qwen35.safetensors \
  --gguf ~/.local/trinote/models/Bonsai-27B-Q1_0.gguf \
  --bin-dir ~/.local/trinote/vendor/llama.cpp-bonsai27/prism-b9591-62061f9/bin \
  --fast --json-out artifacts/atlas-notarized-bonsai-27b.quality-gate.json
```

The gate obtains exact greedy target IDs and top-k logits directly from libllama; it does not infer emitted
IDs by retokenizing output text.

## Backend boundary

The GGUF stores binary language weights, but llama.cpp still performs floating-point activations and
hardware-dependent CUDA reductions. Accordingly, `bonsai-27b-cli` always selects `--no-receipt`, and the
shared CLI rejects `--receipt`, `--json`, or `--onchain` with that backend. Use
`bonsai-integer-27b-cli` when deterministic integer execution is required.

The implementation roles are deliberately distinct:

| Role | Implementation | Receipt status |
|---|---|---|
| Canonical integer oracle | Python/NumPy Qwen3.5 graph with packed-Q1 reference math | verifier |
| Optimized integer CPU producer | one-call/team resident native model executor | producer only; controlled CPU gates and real-trace parity pass |
| Optimized integer CUDA producer | resident deterministic CUDA graph, with fail-closed CPU replay | populated-4K throughput and 128-token parity gates pass |
| Float behavior reference | pinned PrismML libllama/CUDA runtime | never receipt-capable |
| Receipt verifier | structural/signature/artifact checks plus fresh canonical NumPy-oracle re-execution | required for any 27B issuance; never replaced by a producer or PrismML |
