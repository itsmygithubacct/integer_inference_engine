# Contributing

Thanks for your interest in the integer inference engine.

The active, notarized engine is the `trinote` package under
[`bonsai/src/trinote`](bonsai/src/trinote) (the Bonsai-8B release path).
`north-mini-code` is a work-in-progress port and is not on the release path.
(The earlier `mistral-medium` port has been moved out of the tree to
`~/.local/trinote/wip/mistral-medium`; its tokenizer verification is recorded in
[`../FIXLOG.md`](../FIXLOG.md).)

## Ground rules

- **Determinism is the whole product.** The receipt is only meaningful because the
  inference is bit-exact and reproducible. Nothing that feeds a hash or a committed
  value may depend on wall-clock, unseeded randomness, dict/set iteration order,
  locale, or platform int width. Fixed-point paths stay fixed-point; the sampler is
  fully determined by `(seed, absolute-position)`.
- **Fail closed on binding.** An artifact whose digest ≠ the identity `modelHash`
  must be rejected; a supplied-but-missing identity must raise, never silently emit
  an unbound receipt.
- **Protocol constants are frozen.** The receipt schema, on-chain tag (`trinote/r1`),
  sampler RNG domain tag, and on-disk artifact magic are wire/format constants —
  changing one is a breaking, coordinated change (see [`../RENAME.md`](../RENAME.md)).
- **No hand-rolled crypto, no committed secrets.** Use `python-ecdsa` /
  `hashlib`; never commit key files, funded addresses, or machine-specific paths.

## Developing and testing

The engine is run with `src` on the path, not pip-installed:

```sh
cd bonsai
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements_test.txt
PYTHONPATH=src python -m pytest tests -q          # offline; no weights needed
```

- New behavior gets a test; the default run must pass **offline**.
- If you touch a determinism- or protocol-relevant path, add or update the golden /
  parity vector that pins it.

## License

By contributing you agree your contributions are licensed under the Apache
License 2.0 (see [`LICENSE`](LICENSE)).
