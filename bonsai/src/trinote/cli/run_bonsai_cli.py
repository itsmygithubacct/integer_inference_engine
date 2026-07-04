"""trinote-run-bonsai — run ATLAS-Notarized-Bonsai-8B with native receipts or fast PrismML GGUF."""
from __future__ import annotations

import argparse
import ast
import codecs
import json
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

from ..infer_int.artifact_io_bonsai import load_artifact_bonsai
from ..infer_int.bonsai_runtime import (
    emit_and_verify_bonsai_receipt,
    generate_bonsai_tokens,
    load_or_generate_signing_keys,
)
from ..infer_int.gguf_tokenizer_v2 import load_gguf_tokens, llama_tokenize, token_bytes
from ..infer_int.import_gguf_v2 import _GGUFReader
from ..infer_int.reference_bonsai import BonsaiReferenceModel
from ..infer_int.sampler import SamplerConfig, is_receipt_safe, resolve_sampler
from ..config import load_config
from ..receipts import WalletThirdEntryBackend
from ..bundle import pack_bundle, verify_bundle, BundleError

# All generated state (bundles, ledgers, session logs) lives OUTSIDE the repo under
# $BONSAI_NOTARY_HOME (default ~/.local/trinote) — see trinote.notary_paths.
from ..notary_paths import (  # noqa: E402
    bundles_dir,
    ledger_default,
    broadcast_log_default,
    tx_log_default,
    sessions_log_default,
    model_key_default,
    counterparty_key_default,
    default_gguf,
    default_artifact,
    default_identity,
    default_bin_dir,
)


def _build_local_bundle(bundle: dict, emission: dict, *, out_dir: Path, model_label: str,
                        prompt: str, output_text: str, sampler: SamplerConfig) -> str:
    """Package a LOCAL (BSV-off) receipt bundle as a .tar.gz under `out_dir`, including a human-readable
    transcript (plaintext prompt + output). Returns the bundle path. The third entry is the local ledger;
    trust rests on offline commitments + bit-exact re-execution (`trinote-receipt-bundle verify --reexec`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rh = bundle["receipt"]["receiptHash"]
    pre = bundle.get("preimage", {})
    model_label = model_label or pre.get("modelLabel", "")
    transcript = {
        "modelLabel": model_label,
        "modelHash": bundle["receipt"].get("modelHash", ""),
        "receiptHash": rh,
        "sampler": sampler.mode,
        "seed": sampler.seed,
        "inputTokenCount": len(pre.get("inputIds", [])),
        "outputTokenCount": len(pre.get("outputIds", [])),
        "prompt": prompt,
        "output": output_text,
    }
    res = pack_bundle(bundle=bundle, onchain=None, out_dir=out_dir / f"bonsai-{rh}.tar.gz",
                      ledger_entry=emission.get("ledgerEntry"), model_label=model_label,
                      transcript=transcript, as_tar=True)
    return res["path"]


def _handle_repl_command(cmd: str, *, last_run: dict, ref, model_digest: str, bundle_dir: Path) -> None:
    """Handle a `:`-prefixed REPL command in the interactive chat. Commands: :bundle, :verify [path], :help."""
    parts = cmd.split()
    name = parts[0]
    if name in (":help", ":h", ":?"):
        print("[repl] :bundle            package the last receipt as a local bundle (prints the path)\n"
              "[repl] :verify [path]     verify a bundle by IN-MODEL replay (byte-exact); default = last bundle\n"
              "[repl] :help              this help     ·     quit / exit / Ctrl-D to leave", file=sys.stderr)
        return
    if name == ":bundle":
        if not last_run.get("bundle"):
            print("[repl] no run yet — ask the model something first", file=sys.stderr)
            return
        try:
            path = _build_local_bundle(last_run["bundle"], last_run["emission"], out_dir=bundle_dir,
                                       model_label=last_run.get("model_label", ""), prompt=last_run["prompt"],
                                       output_text=last_run["output_text"], sampler=last_run["sampler"])
            last_run["bundle_path"] = path
            print(f"[bundle] {path}", file=sys.stderr)
        except BundleError as exc:
            print(f"[bundle] failed: {exc}", file=sys.stderr)
        return
    if name == ":verify":
        path = parts[1] if len(parts) > 1 else last_run.get("bundle_path")
        if not path:
            print("[repl] no bundle to verify — run :bundle first, or pass a path", file=sys.stderr)
            return
        print(f"[verify] re-executing {path} in-model (byte-exact replay; this re-runs inference) ...", file=sys.stderr)
        try:
            res = verify_bundle(path, reexec=True, model=ref, model_digest=model_digest)
        except (FileNotFoundError, BundleError) as exc:
            print(f"[verify] failed: {exc}", file=sys.stderr)
            return
        rx = res.get("reexec") or {}
        print(f"[verify] {'VERIFIED' if res['ok'] else 'NOT VERIFIED'}  "
              f"offline={res['offline']['ok']} reexec={rx.get('reexecOk')} "
              f"artifactBound={rx.get('artifactBoundOk')} strategy={rx.get('strategy')} "
              f"checked={rx.get('checked')}", file=sys.stderr)
        return
    print(f"[repl] unknown command {name!r} — try :help", file=sys.stderr)

# Anchor default artifact paths to the repo root so the CLI works from any working directory (relative
# defaults crash when invoked outside the repo root). Override any of them with the matching --flag.
_REPO = Path(__file__).resolve().parents[3]
# Weights / kernels / llama.cpp default under $BONSAI_NOTARY_HOME (resolved by notary_paths), falling back to
# the legacy in-repo / dev locations when that is where the data still is, so existing installs keep working.
# Override any of them with the matching --flag (or $BONSAI_MODELS_DIR / $BONSAI_LLAMA_DIR).
_DEFAULT_BIN_DIR = Path(default_bin_dir())
_DEFAULT_GGUF = default_gguf()
_DEFAULT_ARTIFACT = default_artifact()
_DEFAULT_IDENTITY = default_identity()
_ADD_GENERATION_RE = re.compile(
    r"\{%-\s*if\s+add_generation_prompt\s*%\}\s*\{\{-\s*('(?:\\.|[^'])*')\s*\}\}\s*\{%-\s*endif\s*%\}",
    re.S,
)


def _special_token_cutoff(gguf_reader, vocab_size: int) -> int:
    """Lowest token id that is a special/control token, for decode(skip_special_from=...).

    GGUF tokenizer.ggml.token_type marks NORMAL=1 / BYTE=6 as text and CONTROL=3 / USER_DEFINED=4 /
    UNUSED=5 as special. The Bonsai/Qwen3 vocab keeps all control glyphs in a trailing block, so the
    first special id is the cutoff above which decode() suppresses control glyphs in the live stream.
    Falls back to vocab_size (drop nothing) if the metadata is missing."""
    types = gguf_reader.kv.get("tokenizer.ggml.token_type")
    if not types:
        return vocab_size
    special = {3, 4, 5}
    for i, t in enumerate(types):
        if int(t) in special:
            return i
    return vocab_size


def _sampler_from_args(args) -> SamplerConfig:
    frac_default = 16
    rep_fp = round((args.rep_penalty - 1.0) * (1 << frac_default)) if args.rep_penalty > 1.0 else 0
    return resolve_sampler(            # expands presets like 'qwen3-rec'; plain modes pass through unchanged
        args.sampler,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=getattr(args, "min_p", 0.0),
        seed=args.seed,
        rep_penalty=rep_fp,
        no_repeat_ngram=args.no_repeat_ngram,
    )


def _qwen3_chat_prompt(prompt: str, gguf_kv: dict | None = None) -> str:
    template = str((gguf_kv or {}).get("tokenizer.chat_template", "") or "")
    if template:
        required = ("<|im_start|>", "<|im_end|>", "message.role", "content", "add_generation_prompt")
        missing = [s for s in required if s not in template]
        if missing:
            raise ValueError(
                "unsupported tokenizer.chat_template for Bonsai/Qwen3 chat mode; "
                f"missing {', '.join(missing)}"
            )
        match = _ADD_GENERATION_RE.search(template)
        if not match:
            raise ValueError("unsupported tokenizer.chat_template for Bonsai/Qwen3 chat mode; "
                             "cannot extract add_generation_prompt literal")
        assistant_prompt = ast.literal_eval(match.group(1))
    else:
        assistant_prompt = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return f"<|im_start|>user\n{prompt}<|im_end|>\n" + assistant_prompt


def _validate_args(args) -> int:
    if args.sampler in ("temp", "top_k", "top_p", "min_p") and args.temp <= 0:
        print("--temp must be > 0 for sampling modes", file=sys.stderr)
        return 2
    if args.sampler == "top_k" and args.top_k <= 0:
        print("--top-k must be > 0 for --sampler top_k", file=sys.stderr)
        return 2
    if args.sampler == "top_p" and not (0.0 < args.top_p <= 1.0):
        print("--top-p must be in (0, 1] for --sampler top_p", file=sys.stderr)
        return 2
    if args.sampler == "min_p" and not (0.0 <= args.min_p <= 1.0):   # 0 = auto (resolve_sampler -> 0.1)
        print("--min-p must be in [0, 1] for --sampler min_p (0 = default 0.1)", file=sys.stderr)
        return 2
    if args.no_repeat_ngram < 0:
        print("--no-repeat-ngram must be >= 0", file=sys.stderr)
        return 2
    if args.receipt and args.engine != "native":
        print("[bonsai] --receipt requires --engine native; raw PrismML GGUF runs do not emit receipts",
              file=sys.stderr)
        return 2
    if args.fast_required and not args.fast:
        print("[bonsai] --fast-required requires --fast", file=sys.stderr)
        return 2
    if args.verify_mode == "fresh-oracle" and not args.receipt:
        print("[bonsai] --verify-mode fresh-oracle requires receipts to be enabled", file=sys.stderr)
        return 2
    return 0


def _run_native(args, cfg: SamplerConfig) -> int:
    # OMP_NUM_THREADS is set in main() before dispatch (single-query default = nproc-1).
    t_load0 = time.time()
    r = _GGUFReader(args.gguf)
    eos = int(r.kv.get("tokenizer.ggml.eos_token_id", -1))
    tokens = load_gguf_tokens(args.gguf)
    special_from = _special_token_cutoff(r, len(tokens))
    print(f"[bonsai] loading native artifact {args.artifact} ...", file=sys.stderr)
    art, info = load_artifact_bonsai(args.artifact)
    ref = BonsaiReferenceModel(art)
    t_load = time.time() - t_load0
    print(f"[bonsai] engine=int-ref@bonsai-qwen3 sampler={cfg.mode} seed={cfg.seed} "
          f"receipt-bound={is_receipt_safe(cfg)}", file=sys.stderr)
    if getattr(args, "gpu", False):
        os.environ["TRINOTE_GPU"] = "1"   # opt-in toggle read by _gpu_enabled() per Q1 apply (needs --fast)
        if not args.fast:
            print("[bonsai] note: --gpu has no effect without --fast (GPU accelerates the native engine)",
                  file=sys.stderr)
    if args.fast:
        t_fast0 = time.time()
        fast_kind = "native packed-Q1"
        fast_ok = ref.enable_native()
        if not fast_ok:
            fast_kind = "Q1 sign cache"
            fast_ok = ref.enable_fast(check_ram=True, cache_output=True)
        t_fast = time.time() - t_fast0
        if not fast_ok and args.fast_required:
            print("[bonsai] FATAL: --fast-required set but RAM-gated Bonsai fast path could not be enabled",
                  file=sys.stderr)
            return 1
        state = f"{fast_kind} enabled" if fast_ok else "unavailable; using oracle packed path"
        print(f"[bonsai] fast path {state} ({t_fast:.1f}s)", file=sys.stderr)
    verifier_ref = None
    if args.receipt and args.verify_mode == "fresh-oracle":
        print("[bonsai] loading fresh slow oracle verifier for receipts ...", file=sys.stderr)
        verifier_art, verifier_info = load_artifact_bonsai(args.artifact)
        if verifier_info["digest"] != info["digest"]:
            print("[bonsai] FATAL: verifier artifact digest drifted while loading", file=sys.stderr)
            return 1
        verifier_ref = BonsaiReferenceModel(verifier_art)
    print("[bonsai] receipt-bound native reference path; use --engine prismml.cpp only for raw "
          "non-receipt speed demos", file=sys.stderr)
    if args.bench:
        print(f"[bench] load={t_load:.3f}s", file=sys.stderr)

    prompts = [args.prompt] if args.prompt else None
    interactive = prompts is None
    status = 0
    bundle_dir = Path(args.bundle_dir)
    last_run: dict = {}
    while True:
        prompt = prompts.pop(0) if prompts else (input("\nbonsai> ") if interactive else None)
        if prompt is None or (interactive and prompt.strip() in ("", "quit", "exit")):
            break
        if interactive and prompt.strip().startswith(":"):
            _handle_repl_command(prompt.strip(), last_run=last_run, ref=ref,
                                 model_digest=info["digest"], bundle_dir=bundle_dir)
            continue
        t_tok0 = time.time()
        try:
            model_prompt = _qwen3_chat_prompt(prompt, r.kv) if args.chat else prompt
        except ValueError as exc:
            print(f"[bonsai] FATAL: {exc}", file=sys.stderr)
            status = 1
            if not interactive:
                break
            continue
        input_ids = llama_tokenize(model_prompt, args.gguf, bin_dir=args.bin_dir)
        t_tok = time.time() - t_tok0
        stream_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        out_parts: list[str] = []      # accumulate the human-visible text for the bundle transcript
        t0 = time.time()

        def on_token(tok: int) -> None:
            # Drop control glyphs (<|im_start|>, <think>, …) from the USER-VISIBLE stream by decoding
            # with the tokenizer's special-token cutoff. The committed output_ids / receipt are unaffected
            # (they carry the full id list); only what we print to stdout is filtered.
            text = stream_decoder.decode(token_bytes(tok, tokens, skip_special_from=special_from), final=False)
            if text:
                out_parts.append(text)
                sys.stdout.write(text)
                sys.stdout.flush()

        output_ids = generate_bonsai_tokens(
            ref, input_ids, args.max_new, sampler=cfg, eos=eos, on_token=on_token
        )
        tail = stream_decoder.decode(b"", final=True)
        if tail:
            out_parts.append(tail)
            sys.stdout.write(tail)
            sys.stdout.flush()
        output_text = "".join(out_parts)
        dt = time.time() - t0
        if not args.quiet:
            print(f"\n[bonsai] {len(output_ids)} tok in {dt:.1f}s · "
                  f"{dt / max(len(output_ids), 1):.1f}s/tok", file=sys.stderr)
        if args.receipt and output_ids:
            try:
                t_verify0 = time.time()
                # Real third-party-verifiable secp256k1 keys by default (load-or-generate under
                # ~/.local/trinote/keys); --demo-keys forces the legacy deterministic HMAC vouch.
                if args.demo_keys:
                    os.environ["TRINOTE_DEMO_KEYS_OK"] = "1"
                    sign_mk = sign_ck = None
                else:
                    sign_mk, sign_ck = load_or_generate_signing_keys(args.model_key, args.counterparty_key)
                bundle, verification, emission = emit_and_verify_bonsai_receipt(
                    ref,
                    input_ids=input_ids,
                    output_ids=output_ids,
                    model_digest=info["digest"],
                    sampler=cfg,
                    model_key=sign_mk,
                    counterparty_key=sign_ck,
                    verifier_model=verifier_ref,
                    verifier_mode=args.verify_mode,
                    identity_path=args.identity,
                    ledger_path=args.ledger,
                    broadcast_log=args.broadcast_log,
                    broadcast_to_log=args.broadcast_to_log,
                    chain_artifacts_dir=args.chain_artifacts_dir,
                    enable_chain=args.onchain,
                    chain_backend=(WalletThirdEntryBackend(
                        source_index=args.chain_source_index, sat_per_kb=args.chain_sat_per_kb,
                        confirm=args.chain_confirm, change_to_source=True, allow_unconfirmed=True)
                        if args.onchain else None),
                    tx_log=(args.tx_log or None),
                )
                t_verify = time.time() - t_verify0
            except ValueError as exc:
                print(f"[receipt] FATAL: {exc}", file=sys.stderr)
                status = 1
                if not interactive:
                    break
                continue
            rx = verification.get("reexec") or {}
            if not verification["ok"]:
                print(f"[receipt] {bundle['receipt']['receiptHash']} verify failed: "
                      f"mode={verification.get('verificationMode')} "
                      f"bind={verification.get('modelHashMatch')} "
                      f"commit={verification.get('commitMatch')} "
                      f"hash={verification.get('receiptHashMatch')} "
                      f"reexec={rx.get('ok')} strategy={rx.get('strategy')}", file=sys.stderr)
                status = 1
                if not interactive:
                    break
            else:
                print(f"[receipt] {bundle['receipt']['receiptHash']} VERIFIED "
                      f"mode={verification.get('verificationMode')} "
                      f"strategy={rx.get('strategy')} "
                      f"ledger={emission['ledgerEntry']['index']} "
                      f"broadcast={emission['onchain']['status']}", file=sys.stderr)
            # Remember the last run so the REPL :bundle / :verify commands can act on it.
            last_run = {"bundle": bundle, "emission": emission, "prompt": prompt,
                        "output_text": output_text, "sampler": cfg,
                        "model_label": bundle.get("preimage", {}).get("modelLabel", "")}
            # Receipt bundles are DEFAULT-ON whenever receipts are on (disable with --no-bundle): package a
            # portable, re-executable LOCAL bundle (with the human-readable transcript) and print its path.
            if args.bundle and verification["ok"]:
                try:
                    path = _build_local_bundle(bundle, emission, out_dir=bundle_dir,
                                               model_label=last_run["model_label"], prompt=prompt,
                                               output_text=output_text, sampler=cfg)
                    last_run["bundle_path"] = path
                    print(f"[bundle] {path}", file=sys.stderr)
                except BundleError as exc:
                    print(f"[bundle] failed: {exc}", file=sys.stderr)
            if args.save_bundle:
                # Also dump the raw {receipt, preimage} + emission (for `trinote-receipt-bundle pack` / debugging).
                rh = bundle["receipt"]["receiptHash"]
                d = Path(args.save_bundle)
                d.mkdir(parents=True, exist_ok=True)
                (d / f"receipt-{rh}.json").write_text(json.dumps(bundle))
                (d / f"emission-{rh}.json").write_text(json.dumps(emission))
                print(f"[receipt] saved bundle inputs → {d}/receipt-{rh}.json", file=sys.stderr)
            if args.bench:
                print(f"[bench] tokenize={t_tok:.3f}s generate_prefill_decode={dt:.3f}s "
                      f"verify_emit={t_verify:.3f}s verify_strategy={rx.get('strategy')} "
                      f"output_tok_per_s={len(output_ids) / dt if dt > 0 else 0.0:.4f}",
                      file=sys.stderr)
        elif args.bench:
            print(f"[bench] tokenize={t_tok:.3f}s generate_prefill_decode={dt:.3f}s verify_emit=0.000s "
                  f"output_tok_per_s={len(output_ids) / dt if dt > 0 else 0.0:.4f}", file=sys.stderr)
        if not interactive:
            break
    return status


def _run_prismml(args, cfg: SamplerConfig) -> int:
    cli = Path(args.bin_dir) / "llama-cli"
    cmd = [
        str(cli), "-m", args.gguf,
        "-n", str(args.max_new),
        "-t", os.environ.get("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 2) - 1))),
        "--simple-io",
        "--color", "off",
    ]
    if args.prompt:
        cmd += ["-p", args.prompt, "--single-turn"]
    else:
        cmd += ["--conversation"]
    if cfg.mode == "greedy":
        cmd += ["--temp", "0"]
    else:
        cmd += ["--temp", str(cfg.temperature), "--top-k", str(cfg.top_k), "--top-p", str(cfg.top_p)]
    if args.rep_penalty != 1.0:
        cmd += ["--repeat-penalty", str(args.rep_penalty)]
    print("[bonsai] engine=prismml.cpp (fast raw GGUF; NO receipt)", file=sys.stderr)
    return subprocess.run(cmd).returncode


def main(argv: list[str] | None = None) -> int:
    # allow_abbrev=False: the bonsai-notary launcher gates auto-funding + fresh-change on a LITERAL
    # ' --chain-confirm ' substring, so a prefix like --chain-conf must NOT resolve to the real
    # broadcast flag (that would spend with change-address hygiene skipped). Matches agent_cli.py/cli.py.
    ap = argparse.ArgumentParser(prog="trinote-run-bonsai",
                                 description="Run ATLAS-Notarized-Bonsai-8B",
                                 allow_abbrev=False)
    ap.add_argument("--gguf", default=_DEFAULT_GGUF)
    ap.add_argument("--artifact", default=_DEFAULT_ARTIFACT)
    ap.add_argument("--identity", default=_DEFAULT_IDENTITY)
    ap.add_argument("-p", "--prompt", default=None)
    ap.add_argument("-n", "--max-new", type=int, default=1024)
    ap.add_argument("--engine", choices=["native", "prismml.cpp"], default="native")
    ap.add_argument("--chat", action="store_true",
                    help="wrap native prompts with the Bonsai/Qwen3 chat template before tokenization")
    ap.add_argument("--sampler", choices=["min_p", "qwen3-rec", "greedy", "temp", "top_k", "top_p"],
                    default="qwen3-rec",
                    help="default 'qwen3-rec' = the Qwen3 vendor preset (top-p sampling at temp 0.6 / "
                         "top-k 20 / top-p 0.95); receipt-bound + byte-exactly reproducible at a fixed seed. "
                         "'min_p' = min-p sampling (keep tokens with prob >= min_p*max_prob; min_p defaults "
                         "to 0.1); 'greedy' = argmax.")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-seed", action="store_true")
    ap.add_argument("--rep-penalty", type=float, default=1.0)
    ap.add_argument("--no-repeat-ngram", type=int, default=0)
    ap.add_argument("--receipt", action="store_true")
    ap.add_argument("--no-receipt", dest="receipt", action="store_false")
    ap.set_defaults(receipt=True)
    ap.add_argument("--verify-mode", choices=["fast-local", "fresh-oracle"], default="fast-local",
                    help="receipt verifier: reuse producer model or re-load a fresh slow oracle")
    ap.add_argument("--model-key", default=None,
                    help="path to the model (issuer) secp256k1 receipt signing key (JSON). Default: "
                         f"{model_key_default()} — generated on first use if absent (third-party-verifiable).")
    ap.add_argument("--counterparty-key", default=None,
                    help=f"path to the counterparty signing key (JSON). Default: {counterparty_key_default()}.")
    ap.add_argument("--demo-keys", action="store_true",
                    help="sign with the legacy deterministic HMAC demo keys (NO authenticity; for reproducible "
                         "snapshots only). Default is real secp256k1 keys under ~/.local/trinote/keys.")
    ap.add_argument("--fast", dest="fast", action="store_true",
                    help="enable native packed-Q1 fast path, falling back to RAM-gated sign cache")
    ap.add_argument("--no-fast", dest="fast", action="store_false",
                    help="force the native oracle packed-Q1 path")
    ap.set_defaults(fast=False)
    ap.add_argument("--fast-required", action="store_true",
                    help="fail if --fast cannot be enabled because of RAM limits")
    ap.add_argument("--gpu", dest="gpu", action="store_true", default=False,
                    help="per-host opt-in: try the GPU Q1 kernel before the CPU native/oracle path "
                         "(byte-identical; silently falls back to CPU when libbonsai_q1_gpu.so / a GPU is "
                         "absent). Composes with --fast (the GPU accelerates the native engine).")
    ap.add_argument("--bench", action="store_true",
                    help="print per-stage native timing to stderr")
    ap.add_argument("--ledger", default=ledger_default())
    ap.add_argument("--broadcast-log", default=broadcast_log_default())
    ap.add_argument("--no-broadcast-log", dest="broadcast_to_log", action="store_false")
    ap.set_defaults(broadcast_to_log=True)
    ap.add_argument("--chain-artifacts-dir", default=None)
    ap.add_argument("--save-bundle", default=None,
                    help="dump each receipt's {receipt,preimage}+emission here for `trinote-receipt-bundle pack`")
    ap.add_argument("--bundle-dir", default=str(bundles_dir()),
                    help="directory for the auto-packaged local receipt bundles (default ~/.local/trinote/bundles)")
    ap.add_argument("--no-bundle", dest="bundle", action="store_false",
                    help="do NOT auto-package a receipt bundle per run (bundles are on by default with receipts)")
    ap.set_defaults(bundle=True)
    ap.add_argument("--tx-log", default=tx_log_default(),
                    help="off-chain transaction log: the full raw tx of every on-chain third entry "
                         "(in addition to the artifact broadcast-log); set empty to disable")
    ap.add_argument("--onchain", action="store_true",
                    help="build each receipt as a public BSV OP_RETURN Third Entry via the notary wallet "
                         "(DRY-RUN unless --chain-confirm)")
    ap.add_argument("--chain-source-index", type=int, default=23,
                    help="receive index of the funding UTXO for on-chain receipts (self-rolls in place)")
    ap.add_argument("--chain-sat-per-kb", type=int, default=100, help="on-chain fee in sat/KILOBYTE (default 100)")
    # Two-key interlock: DRY-RUN by default. With --onchain the tx is built + its txid computed + logged,
    # but NOT broadcast unless --chain-confirm is also given (matches trinote-agent/agentd and SECURITY.md).
    ap.add_argument("--chain-confirm", dest="chain_confirm", action="store_true",
                    help="with --onchain: actually BROADCAST the third entry to mainnet (real BSV)")
    ap.add_argument("--chain-dry-run", dest="chain_confirm", action="store_false",
                    help="with --onchain: build + compute the third-entry txid but DO NOT broadcast (default)")
    ap.set_defaults(chain_confirm=False)
    ap.add_argument("--bin-dir", default=str(_DEFAULT_BIN_DIR), help="PrismML llama.cpp build/bin")
    ap.add_argument("--threads", type=int, default=0,
                    help="OpenMP threads for a SINGLE query (1 process / 1 shard). 0 = auto = nproc-1. An "
                         "explicit OMP_NUM_THREADS in the environment (launch scripts / the parallel bench) "
                         "is respected; the parallel bench tool sets its own per-shard threads.")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="JSON mode: print progress to stderr")
    # ---- Bonsai JSON mode -------------------------------------------------------------------------------
    ap.add_argument("--json", dest="json", action="store_true",
                    help="JSON mode: ask --prompt as a question, emit one structured object "
                         "{thinking,answer,bonsai,receipt,bundle,reproduction}; record to the reproduction log")
    ap.add_argument("--fetch", action="store_true",
                    help="JSON mode: return a previously-inferenced response for an identical prompt+context "
                         "from the log WITHOUT re-running the model (on a miss, run + record)")
    ap.add_argument("--json-log", dest="json_log", default=sessions_log_default(),
                    help="JSONL reproduction log (default ~/.local/trinote/sessions/bonsai-json.jsonl)")
    ap.add_argument("--json-instr", dest="json_instr", default=None,
                    help="instruction appended to the question (default: the 'respond in json …' suffix)")
    ap.add_argument("--answer", action="store_true", help="JSON mode: print ONLY the answer field")
    ap.add_argument("--json-filter", dest="json_filter", default=None,
                    help="JSON mode: pipe the result through this jq filter (e.g. .answer)")
    ap.add_argument("--debug", action="store_true",
                    help="JSON mode: append enhanced perf logging (timing, tokens) to ~/.local/trinote/debug")
    # Config-file settings override the built-in defaults above; an explicit CLI flag still overrides the
    # config (argparse only treats set_defaults values as defaults). See trinote.config / bonsai.toml.
    ap.set_defaults(**load_config())
    args = ap.parse_args(argv)

    if args.random_seed:
        args.seed = secrets.randbits(64)
        print(f"[bonsai] random seed = {args.seed} (committed in receipt)", file=sys.stderr)
    bad = _validate_args(args)
    if bad:
        return bad
    cfg = _sampler_from_args(args)
    # Threading default for a SINGLE query (one process / one shard): nproc-1 OpenMP threads (leaves a core
    # for the system). Precedence: explicit --threads > an OMP_NUM_THREADS already in the env (launch scripts /
    # the parallel bench, which manages its own per-shard threads) > nproc-1. Set before any model/.so load so
    # OpenMP picks it up.
    if args.threads and args.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
    elif "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 2) - 1))
    try:
        if args.json:
            from .json_mode import run_json   # deferred import (avoids an import cycle)
            return run_json(args, cfg)
        if args.engine == "prismml.cpp":
            return _run_prismml(args, cfg)
        return _run_native(args, cfg)
    except FileNotFoundError as exc:
        # Missing GGUF / artifact / identity (e.g. defaults resolved outside the repo, or a wrong --path):
        # report cleanly instead of dumping a traceback, and exit non-zero.
        target = getattr(exc, "filename", None) or exc
        print(f"[bonsai] FATAL: required file not found: {target}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
