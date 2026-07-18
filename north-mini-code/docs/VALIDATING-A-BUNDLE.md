# Validating a receipt bundle

A **receipt bundle** is the portable, self-contained proof that the north-mini-code integer engine, on a specific
set of weights, with a specific sampler, deterministically produced a specific output from a specific input — and
that a producer cryptographically attested to it. This document explains how to validate one.

There are **two independent levels**, and a complete validation runs both:

| Level | What it proves | Needs the model? | Needs a GPU? | Tool |
|------|----------------|:---:|:---:|------|
| **1. Offline** | the bundle is internally consistent and signed (untampered; optionally *who* signed) | no | no | `tools/verify_bundle.py` |
| **2. Replay** | the signed output is what the deterministic engine actually computes — reproduced **byte-for-byte** | yes | yes¹ | `tools/replay_receipt.py` |

¹ A GPU is only needed to re-run the model at a useful speed; the engine is byte-identical on CPU (`NMC_BACKEND=cpu`),
so a CPU-only verifier reaches the same answer, just slower.

---

## What's in a bundle

A bundle is a directory (content-addressed by `manifest.bundleHash`):

| file | contents |
|------|----------|
| `receipt.json` | the signed, on-chain-committable half: `modelHash`, `inputCommit`, `outputCommit`, `trace`/`traceCommit`, `receiptHash`, and the two secp256k1 signatures (`sigModel`, `sigCounterparty`) with their pubkeys |
| `preimage.json` | the **authoritative record**: `inputIds`, `outputIds` (the raw token ids the commitments are over), `sampler`, `modelHash`, `artifactDigest` |
| `manifest.json` | bundle schema, `bundleHash`, and a SHA-256 digest of every other file |
| `transcript.json` / `.md` | human-readable prompt + decoded output (convenience only — *not* authoritative) |
| `chain-artifact.json` | the "3rd entry" payload for the hash-chained ledger / optional BSV broadcast |
| `ledger-head.json` | the local hash-chain head this receipt extends |

The **plaintext transcript is not the proof** — the authoritative content is the committed token ids in
`preimage.json`, bound into `receiptHash`. Re-derive the text from those ids with the model tokenizer.

---

## The commitment / trust model (read this first)

Every hash is a commitment you can recompute:

- `inputCommit  = token_commit(inputIds)`
- `outputCommit = token_commit(outputIds)`
- `traceCommit  = commit(trace)`   (sampler config, etc.)
- `modelHash    = commit({ artifactDigest = sha256(GGUF tensor data), config })` — pins **both** the exact weights
  **and** the exact integer semantics (`fa`/`fw`, arch, RoPE convention = interleaved-norm, NoPE). It is *which
  deterministic computation* the receipt is about.
- `receiptHash  = sha256(receipt body **including the signatures**)`

**The key subtlety:** `receiptHash` commits the *signed pair* (it's the payload that goes on-chain), so it is
**key-dependent**. A third-party verifier who does not hold the producer's private keys **cannot and should not
reproduce `receiptHash`**. Instead the verifier:

1. **verifies** the producer's signatures against the **public** keys, and
2. **reproduces** the `outputCommit` by re-executing the deterministic engine.

So "reproducing the receipt" means *reproducing the signed `outputCommit` byte-for-byte and confirming the
producer's signature over it* — **not** recomputing `receiptHash`. (If you try to recompute `receiptHash` with
your own keys you will get a different value; that is correct, not a failure.)

**Integrity vs authenticity.** Verifying a signature against the pubkey *embedded in the receipt* proves the
receipt was not altered after signing (**integrity**) — but a forger could self-sign fabricated content with a
fresh key. To prove a **specific** producer signed it (**authenticity**), pin their known public key
(`--model-pubkey` / `--counterparty-pubkey`). The `modelHash` + the Level-2 replay then bind that signature to a
real, reproducible computation.

---

## Level 1 — offline validation (anyone, from the bundle alone)

No model, no GPU, no network. Needs only the verifier library (`trinote.*` receipt stack + `ecdsa`).

```bash
cd north-mini-code
NMC_BONSAI_SRC=<…/bonsai/src> PYTHONPATH=src \
  .venv/bin/python tools/verify_bundle.py <bundle_dir>
```

It runs and reports two sub-layers:

- **structural** — manifest schema, `bundleHash`, **every file's digest**, `receiptHash` self-consistency, the
  `input`/`output`/`trace` commitments, and the chain artifact.
- **signature / commitments** — `structuralOk`, `signatureOk` (the ECDSA sigs validate over the canonical signed
  message), `commitMatch`, `receiptHashMatch`, and the `…Authenticated` flags.

A bundle is **offline-valid** when `structuralOk ∧ signatureOk ∧ commitMatch ∧ receiptHashMatch` are all true
(exit 0). To also assert *who* produced it, pin the expected signer:

```bash
… tools/verify_bundle.py <bundle_dir> \
    --model-pubkey 039bd741…  --counterparty-pubkey 03fd8441…
```

with the pins set, `sigModelAuthenticated` / `sigCounterpartyAuthenticated` must become true.

**What a failure means**

| failing check | meaning |
|---|---|
| `file:*` digest | a packed file was altered after bundling |
| `commitMatch` / `outputCommit` | `receipt.outputCommit` ≠ `token_commit(preimage.outputIds)` — the ids don't match the commitment |
| `receiptHashMatch` | the receipt body was altered after signing |
| `signatureOk` | the signature is not a valid ECDSA sig over the message for the (embedded/pinned) pubkey |
| `…Authenticated` (with a pin) | a *different* key signed it than the one you pinned |

---

## Level 2 — byte-exact replay (re-execution)

This is the heart of the claim: re-run the deterministic engine and confirm it reproduces the **signed output
byte-for-byte**. Because the engine is deterministic integer arithmetic pinned by `modelHash`, the *compute path*
holds on **any machine, any GPU arch, or CPU** — that is the whole point of the notarized integer engine.

> **One honest caveat (unlike Bonsai).** north-mini-code rebuilds its RoPE cos/sin tables at run time from libm
> `math.cos/sin` + a float `pow` fed to `round()` (`engine._rope`), rather than reading committed tables. Those
> tables are now bound into `modelHash` (`receipts_runtime.model_hash` hashes every row through the
> GGUF-declared context limit, and the engine refuses positions beyond that limit), so a host whose libm rounds a
> borderline entry differently produces a **different `modelHash`** — the divergence is *detected* as a
> `modelHash` mismatch (fail-closed: a genuine receipt is rejected, never forged), rather than silently emitting
> different tokens. So replay is byte-exact on any host with **matching libm rounding**; on a host that diverges,
> the mismatch is surfaced and attributable, not silent. Committing the tables into the shipped artifact (as
> Bonsai does) to make this by-construction is tracked follow-up work.

### On a machine you already have (with the GGUF)

```bash
cd north-mini-code
NMC_BONSAI_SRC=<…/bonsai/src> NMC_BACKEND=cuda-resident PYTHONPATH=src \
  .venv/bin/python tools/replay_receipt.py <bundle_dir> <gguf_blob>
```

`replay_receipt.py` performs four checks and exits 0 only if all pass:

1. **offline-verify** the producer's bundle (Level 1, sigs + commitments + chain);
2. **same model** — this box's GGUF hashes to the bundle's `modelHash` (and `artifactDigest`);
3. **byte-exact output** — re-execute `inputIds` under the recorded sampler (greedy/seed-0) and assert the output
   ids equal `preimage.outputIds` *token-for-token*;
4. **signed-commit match** — `token_commit(re-exec output) == receipt.outputCommit` (ties the re-execution to the
   producer's signed claim).

### On a fresh, *different* cloud machine (the strongest demonstration)

The deploy driver provisions a new GPU box (hardened against vast.ai create-failures), pushes the bundle + the
receipt stack, re-executes, and tears the box down. Use `--exclude-machine` to *guarantee a different physical
host* than the producer:

```bash
cd ~/.local/integer_inference_engine/deploy
python3 nmc_gpu_test.py replay --yes --gpu-name 4090 \
    --bundle <bundle_dir> --exclude-machine <producer_machine_id>
```

Look for the final lines in the pulled log (`runs_nmc_fwd_remote.log`):

```
[replay] (2) modelHash match=True   artifactDigest match=True
[replay] (3) re-executed N tokens; BYTE-EXACT output match: True
[replay] (4) re-exec outputCommit == producer's SIGNED outputCommit: True
[replay] RESULT: BYTE-EXACT REPRODUCED on a different machine ✓
```

---

## Worked example

The bundle `20260630T035654436999Z` (prompt *"what is the meaning of life?"*, 362-token greedy output,
`modelHash aeca579d…`) was produced on vast.ai machine `8334` and replayed on machine `8011`:

```
[replay] (1) offline bundle verify (sigs+commits+chain): True
[replay] (2) modelHash match=True   artifactDigest match=True
[replay] (3) re-executed 362 tokens; BYTE-EXACT output match: True
[replay] (4) re-exec outputCommit == producer's SIGNED outputCommit: True
[replay] RESULT: BYTE-EXACT REPRODUCED on a different machine ✓
```

A different physical machine, re-running the deterministic integer engine, produced the identical 362 tokens and
matched the producer's cryptographically-signed `outputCommit`.

---

## A complete validation, in one breath

1. `verify_bundle.py <bundle>` → offline-valid (integrity); add `--model-pubkey …` to assert authenticity.
2. `replay_receipt.py <bundle> <gguf>` (or `nmc_gpu_test.py replay --bundle … --exclude-machine …`) → the
   deterministic engine reproduces the signed output **byte-for-byte**.

Pass both and you have: *a named producer cryptographically attested that this exact integer computation maps this
input to this output, and an independent machine confirms the deterministic engine actually produces it.*

## See also
- `../README.md` — engine overview
- `../../CONCEPT.md`, `../../integer_engine.md` — why deterministic integer inference enables this at all
- `tools/verify_bundle.py`, `tools/replay_receipt.py`, `src/nmc/receipts_runtime.py` — the implementations
