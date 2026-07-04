# Security Policy — integer_inference_engine

The `trinote` engine (under `bonsai/src/trinote`) produces deterministic integer
LLM inference and wraps it in a cryptographic triple-entry receipt. Its security
model rests on **reproducibility** (anyone can re-execute and verify a receipt) and
**fail-closed binding** (a receipt is bound to the exact weights it names).

## Trust anchors

- **Determinism.** A receipt is only trustworthy because the computation is
  bit-exact and re-derivable. Any nondeterminism (float where fixed-point is
  intended, unseeded RNG, iteration-order or platform dependence) silently breaks
  the guarantee. The sampler draw is fully determined by `(seed, absolute-position)`
  under a fixed RNG domain tag.
- **Model binding (fail closed).** The runtime rejects an artifact whose digest does
  not equal the identity `modelHash`, and treats a supplied-but-missing/unreadable
  identity as fatal — it never emits a receipt unbound to the minted identity.
- **Canonical receipts.** Receipt/commitment serialization is deterministic and
  injective; signatures are computed and checked over the canonical bytes only.

## Keys and secrets

- Receipt signing keys are **secp256k1** and third-party-verifiable (the receipt
  carries the signer's *public* key). They are created/stored under
  `$BONSAI_NOTARY_HOME` (default `~/.local/trinote`), **outside this repository**.
- Never commit key files, WIFs, mnemonics, or funded addresses. Any private key that
  appears in a test vector is a deterministic public constant with no funds.

## Model weights

Weights are not shipped here; they are fetched and **checksum-verified** against the
identity record (`fetch_weights.sh` fails closed on a hash mismatch). A receipt only
verifies against weights whose `sha256` matches the committed `modelHash`.

## Reporting a vulnerability

Report privately via GitHub Security Advisories — use **"Report a vulnerability"** on
this repository's **Security** tab (repo-relative
[`security/advisories/new`](../../security/advisories/new)). This keeps the report
private until a fix is coordinated. **Do not** open a public issue for an unfixed
vulnerability, and do not include real key material in a report.
