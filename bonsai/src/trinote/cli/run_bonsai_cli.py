"""Run native integer Bonsai-8B/27B or an inference-only PrismML GGUF backend."""
from __future__ import annotations

import argparse
import ast
import codecs
import contextlib
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
    validate_bonsai35_receipt_identity,
)
from ..infer_int.gguf_tokenizer_v2 import load_gguf_tokens, llama_tokenize, token_bytes
from ..infer_int.import_gguf_v2 import _GGUFReader
from ..infer_int.reference_bonsai import BonsaiReferenceModel
from ..infer_int.reference_bonsai35 import BonsaiQwen35ReferenceModel
from ..infer_int.gpu_bonsai35 import Bonsai35GpuExecutor
from ..infer_int.prompt_cache_bonsai35 import (
    build_prompt_state,
    default_prompt_cache_path,
    generate_from_prompt_state,
    load_prompt_state,
    prompt_cache_key,
    save_prompt_state,
)
from ..infer_int.sampler import SamplerConfig, is_receipt_safe, resolve_sampler, sample_token
from ..config import load_config
from ..receipts import WalletThirdEntryBackend
from ..bundle import pack_bundle, verify_bundle, BundleError
from .context_window import parse_context_size, resolve_context_window
from .conversation import ContextOverflow, Conversation
from .live_session import LiveNativeSession
from .repl import TerminalNoise, TerminalRepl, parse_command

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
                        prompt: str, output_text: str, sampler: SamplerConfig,
                        messages: list[dict[str, str]] | None = None,
                        context: dict | None = None) -> str:
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
    if messages is not None:
        transcript["messages"] = messages
    if context is not None:
        transcript["context"] = context
    res = pack_bundle(bundle=bundle, onchain=None, out_dir=out_dir / f"bonsai-{rh}.tar.gz",
                      ledger_entry=emission.get("ledgerEntry"), model_label=model_label,
                      transcript=transcript, as_tar=True)
    return res["path"]


def _handle_repl_command(cmd: str, *, last_run: dict, ref, model_digest: str,
                         bundle_dir: Path, terminal: TerminalRepl | None = None) -> None:
    """Handle a `:`-prefixed REPL command in the interactive chat. Commands: :bundle, :verify [path], :help."""
    parsed = parse_command(cmd)
    if parsed is None:
        return
    name = ":" + parsed.name
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
                                       output_text=last_run["output_text"], sampler=last_run["sampler"],
                                       messages=last_run.get("messages"), context=last_run.get("context"))
            last_run["bundle_path"] = path
            print(f"[bundle] {path}", file=sys.stderr)
        except BundleError as exc:
            print(f"[bundle] failed: {exc}", file=sys.stderr)
        return
    if name == ":verify":
        path = parsed.args[0] if parsed.args else last_run.get("bundle_path")
        if not path:
            print("[repl] no bundle to verify — run :bundle first, or pass a path", file=sys.stderr)
            return
        print(f"[verify] re-executing {path} in-model (byte-exact replay; this re-runs inference) ...", file=sys.stderr)
        try:
            guard = terminal.quarantine_input() if terminal else contextlib.nullcontext()
            with guard:
                res = verify_bundle(path, reexec=True, model=ref, model_digest=model_digest)
        except KeyboardInterrupt:
            print("\n[verify] cancelled", file=sys.stderr)
            return
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


def _print_repl_help() -> None:
    print(
        "[repl] /clear             clear retained conversation context\n"
        "[repl] /context [auto|N]  show or change the token window\n"
        "[repl] /system [TEXT]     set system text and clear the conversation\n"
        "[repl] /think [on|off]    Qwen3.5 reasoning mode (off by default here)\n"
        "[repl] /retry             regenerate the last user turn\n"
        "[repl] /history           show retained user/assistant turns\n"
        "[repl] /paste             multiline input; a single '.' submits\n"
        "[repl] /bundle            package the last receipt\n"
        "[repl] /verify [PATH]     verify a bundle by byte-exact replay\n"
        "[repl] /exit              leave (also Ctrl-D)\n"
        "[repl] Ctrl-C cancels active generation; a trailing '\\' continues input.\n"
        "[repl] ':' command aliases remain supported.",
        file=sys.stderr,
    )


def _handle_session_command(
    line: str,
    *,
    terminal: TerminalRepl,
    conversation: Conversation,
    context_profile,
    auto_context_profile,
    last_run: dict,
    ref,
    model_digest: str,
    bundle_dir: Path,
) -> tuple[bool, str | None, bool]:
    """Return ``(handled, prompt_to_run, exit_requested)``."""
    command = parse_command(line)
    if command is None:
        return False, None, False
    name = command.name
    if name == "help":
        _print_repl_help()
    elif name == "exit":
        return True, None, True
    elif name == "clear":
        conversation.clear()
        print("[repl] conversation cleared", file=sys.stderr)
    elif name == "context":
        if not command.args:
            used = conversation.retained_tokens()
            mode = "auto" if conversation.context_automatic else "explicit"
            reason = auto_context_profile.reason if conversation.context_automatic else "session override"
            print(
                f"[repl] context={conversation.context_size} ({mode}; {reason}) "
                f"input-budget={conversation.input_budget} retained={used} "
                f"output-reserve={conversation.max_new} evicted={conversation.evicted_total} "
                f"source-max={context_profile.source_max} artifact-max={context_profile.artifact_max}",
                file=sys.stderr,
            )
        elif len(command.args) != 1:
            print("[repl] usage: /context [auto|N]", file=sys.stderr)
        else:
            try:
                requested = parse_context_size(command.args[0])
                size = auto_context_profile.effective if requested is None else requested
                if context_profile.hard_max is not None and size > context_profile.hard_max:
                    raise ValueError(
                        f"context {size} exceeds hard maximum {context_profile.hard_max}"
                    )
                conversation.set_context_size(size, automatic=requested is None)
                print(
                    f"[repl] context={size}; old turns will be evicted as needed",
                    file=sys.stderr,
                )
            except ValueError as exc:
                print(f"[repl] {exc}", file=sys.stderr)
    elif name == "system":
        system_text = " ".join(command.args)
        conversation.set_system(system_text)
        state = "set" if system_text else "cleared"
        print(f"[repl] system message {state}; conversation cleared", file=sys.stderr)
    elif name == "think":
        if conversation.architecture != "qwen35":
            print("[repl] /think applies only to Qwen3.5 models", file=sys.stderr)
        elif not command.args:
            print(f"[repl] thinking={'on' if conversation.thinking else 'off'}", file=sys.stderr)
        elif len(command.args) == 1 and command.args[0].lower() in {"on", "off"}:
            conversation.thinking = command.args[0].lower() == "on"
            print(f"[repl] thinking={'on' if conversation.thinking else 'off'}", file=sys.stderr)
        else:
            print("[repl] usage: /think [on|off]", file=sys.stderr)
    elif name == "retry":
        prompt = conversation.retry()
        if prompt is None:
            print("[repl] no turn to retry", file=sys.stderr)
        else:
            print("[repl] retrying last user turn", file=sys.stderr)
            return True, prompt, False
    elif name == "history":
        if not conversation.turns:
            print("[repl] no retained turns", file=sys.stderr)
        for i, turn in enumerate(conversation.turns, 1):
            answer = " ".join(turn.output_text.strip().split())
            if len(answer) > 160:
                answer = answer[:157] + "..."
            print(f"[{i}] user: {turn.user}\n    assistant: {answer}", file=sys.stderr)
    elif name == "paste":
        text = terminal.read_paste()
        if text:
            return True, text, False
    elif name in {"bundle", "verify"}:
        _handle_repl_command(
            ":" + name + ((" " + command.raw_args) if command.raw_args else ""),
            last_run=last_run,
            ref=ref,
            model_digest=model_digest,
            bundle_dir=bundle_dir,
            terminal=terminal,
        )
    else:
        print(f"[repl] unknown command '/{name}' — try /help", file=sys.stderr)
    return True, None, False

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


def _context_size_arg(value: str) -> int:
    """argparse adapter accepting both ``auto`` and non-negative token counts."""
    try:
        parsed = parse_context_size(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return 0 if parsed is None else parsed


def _qwen3_chat_prompt(
    prompt: str,
    gguf_kv: dict | None = None,
    *,
    thinking: bool = True,
) -> str:
    kv = gguf_kv or {}
    template = str(kv.get("tokenizer.chat_template", "") or "")
    if kv.get("general.architecture") == "qwen35":
        # The Qwen3.5 template is a larger multimodal/tool Jinja program whose
        # add_generation_prompt branch contains nested conditionals, so the
        # compact Qwen3 literal extractor below is deliberately not used.  For
        # a single text user turn this is the exact rendered vendor template
        # with thinking enabled (the model's default).
        required = ("<|im_start|>", "<|im_end|>", "add_generation_prompt", "<think>\\n")
        missing = [s for s in required if s not in template]
        if template and missing:
            raise ValueError(
                "unsupported tokenizer.chat_template for Bonsai/Qwen3.5 chat mode; "
                f"missing {', '.join(missing)}"
            )
        assistant_prefix = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
        return (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_prefix}"
        )
    if not thinking:
        raise ValueError("--no-think currently supports only Bonsai-27B/Qwen3.5 chat mode")
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
    if getattr(args, "repl", False) and getattr(args, "prompt", None) is not None:
        print("[bonsai] --repl cannot be combined with --prompt", file=sys.stderr)
        return 2
    if int(getattr(args, "max_new", 1)) <= 0:
        print("[bonsai] --max-new must be > 0", file=sys.stderr)
        return 2
    if getattr(args, "think", False) and getattr(args, "no_think", False):
        print("[bonsai] --think and --no-think are mutually exclusive", file=sys.stderr)
        return 2
    if getattr(args, "no_think", False) and not args.chat:
        print("[bonsai] --no-think requires --chat", file=sys.stderr)
        return 2
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
    if args.onchain and args.engine != "native":
        print("[bonsai] --onchain requires --engine native and a verifiable receipt", file=sys.stderr)
        return 2
    if args.json and args.engine != "native":
        print("[bonsai] --json reproduction mode requires --engine native", file=sys.stderr)
        return 2
    if args.ctx_size is not None and args.ctx_size < 0:
        print("[bonsai] --ctx-size must be >= 0", file=sys.stderr)
        return 2
    if args.n_gpu_layers is not None and args.n_gpu_layers < 0:
        print("[bonsai] --n-gpu-layers must be >= 0", file=sys.stderr)
        return 2
    if args.fast_required and not args.fast:
        print("[bonsai] --fast-required requires --fast", file=sys.stderr)
        return 2
    if args.prompt_cache and args.engine != "native":
        print("[bonsai] --prompt-cache requires --engine native", file=sys.stderr)
        return 2
    if args.verify_mode == "fresh-oracle" and not args.receipt:
        print("[bonsai] --verify-mode fresh-oracle requires receipts to be enabled", file=sys.stderr)
        return 2
    return 0


def _generate_bonsai35_with_prompt_cache(
    ref: BonsaiQwen35ReferenceModel,
    input_ids: list[int],
    max_new: int,
    *,
    sampler: SamplerConfig,
    artifact_digest: str,
    cache_dir: str | None,
    eos: int | None,
    on_token,
) -> list[int]:
    """Load/build a verified deterministic prefix state, then generate from it."""

    if len(input_ids) + int(max_new) > int(ref.cfg["context_len"]):
        return generate_bonsai_tokens(
            ref, input_ids, max_new, sampler=sampler, eos=eos, on_token=on_token
        )
    if cache_dir:
        path = Path(cache_dir) / (prompt_cache_key(artifact_digest, input_ids) + ".safetensors")
    else:
        path = default_prompt_cache_path(artifact_digest, input_ids)
    state = None
    if path.exists():
        try:
            state = load_prompt_state(path, ref.artifact, artifact_digest)
            print(f"[prompt-cache] hit {path}", file=sys.stderr)
        except (OSError, ValueError) as exc:
            print(f"[prompt-cache] rejected {path}: {exc}; rebuilding", file=sys.stderr)
    if state is None:
        state = build_prompt_state(ref, input_ids, artifact_digest)
        save_prompt_state(state, ref.artifact, path)
        print(f"[prompt-cache] miss; wrote {path}", file=sys.stderr)
    frac = int(ref.cfg["frac"])
    return generate_from_prompt_state(
        ref,
        state,
        max_new,
        lambda row, position, history: sample_token(
            row,
            sampler,
            position=position,
            frac_bits=frac,
            history_ids=history,
        ),
        eos=eos,
        on_token=on_token,
        keep_reusable=False,
    )


def _generate_native_turn(
    ref,
    input_ids: list[int],
    *,
    args,
    cfg: SamplerConfig,
    eos: int | None,
    on_token,
    artifact_arch: str,
    artifact_digest: str,
    gpu_executor,
    live_session: LiveNativeSession | None = None,
) -> list[int]:
    """Dispatch one turn without changing producer/fallback semantics."""
    if live_session is not None:
        frac = int(ref.cfg["frac"])

        def fallback_notice() -> None:
            print(
                "\n[bonsai] CUDA range/launch guard fired; replaying prompt on CPU oracle",
                file=sys.stderr,
            )

        result = live_session.generate(
            input_ids,
            args.max_new,
            lambda row, pos, hist: sample_token(
                row, cfg, position=pos, frac_bits=frac, history_ids=hist
            ),
            eos=eos,
            on_token=on_token,
            on_gpu_fallback=fallback_notice,
        )
        return result.output_ids
    if gpu_executor is not None:
        frac = int(ref.cfg["frac"])
        gpu_partial: list[int] = []

        def gpu_on_token(tok: int) -> None:
            gpu_partial.append(int(tok))
            on_token(int(tok))

        output_ids, gpu_complete = gpu_executor.generate(
            input_ids,
            args.max_new,
            lambda row, pos, hist: sample_token(
                row, cfg, position=pos, frac_bits=frac, history_ids=hist
            ),
            eos=eos,
            on_token=gpu_on_token,
        )
        if gpu_complete:
            return output_ids
        # The GPU guard poisons the device context before returning an
        # untrusted row. Replay canonically from the original prompt.
        print("\n[bonsai] CUDA range/launch guard fired; replaying prompt on CPU oracle",
              file=sys.stderr)
        replay_index = 0

        def replay_on_token(tok: int) -> None:
            nonlocal replay_index
            if replay_index < len(gpu_partial):
                if int(tok) != gpu_partial[replay_index]:
                    raise RuntimeError(
                        "GPU/CPU replay diverged before fallback boundary; refusing output"
                    )
            else:
                on_token(int(tok))
            replay_index += 1

        return generate_bonsai_tokens(
            ref, input_ids, args.max_new, sampler=cfg, eos=eos,
            on_token=replay_on_token,
        )
    if args.prompt_cache and artifact_arch == "qwen35":
        return _generate_bonsai35_with_prompt_cache(
            ref,
            input_ids,
            args.max_new,
            sampler=cfg,
            artifact_digest=artifact_digest,
            cache_dir=args.prompt_cache_dir,
            eos=eos,
            on_token=on_token,
        )
    return generate_bonsai_tokens(
        ref, input_ids, args.max_new, sampler=cfg, eos=eos, on_token=on_token
    )


def _run_native(args, cfg: SamplerConfig) -> int:
    # OMP_NUM_THREADS is set in main() before dispatch (single-query default = nproc-1).
    t_load0 = time.time()
    r = _GGUFReader(args.gguf)
    eos = int(r.kv.get("tokenizer.ggml.eos_token_id", -1))
    tokens = load_gguf_tokens(args.gguf)
    special_from = _special_token_cutoff(r, len(tokens))
    print(f"[bonsai] loading native artifact {args.artifact} ...", file=sys.stderr)
    art, info = load_artifact_bonsai(args.artifact)
    gguf_arch = str(r.kv.get("general.architecture", ""))
    artifact_arch = str(art.get("config", {}).get("architecture", ""))
    if gguf_arch != artifact_arch:
        raise ValueError(
            f"GGUF architecture {gguf_arch!r} does not match native artifact architecture {artifact_arch!r}"
        )
    if artifact_arch == "qwen35":
        ref_class = BonsaiQwen35ReferenceModel
        engine_name = "int-ref@bonsai-qwen35"
    elif artifact_arch == "qwen3":
        ref_class = BonsaiReferenceModel
        engine_name = "int-ref@bonsai-qwen3"
    else:
        raise ValueError(f"unsupported native Bonsai architecture {artifact_arch!r}")
    try:
        context_profile = resolve_context_window(
            r.kv, artifact=art, backend="native", requested=args.ctx_size
        )
        auto_context_profile = (
            context_profile if context_profile.automatic else
            resolve_context_window(r.kv, artifact=art, backend="native", requested=None)
        )
    except ValueError as exc:
        print(f"[bonsai] {exc}", file=sys.stderr)
        return 2
    if int(args.max_new) >= context_profile.effective:
        print(
            f"[bonsai] --max-new {args.max_new} must be smaller than context "
            f"{context_profile.effective}",
            file=sys.stderr,
        )
        return 2
    if args.prompt_cache and artifact_arch != "qwen35":
        raise ValueError("--prompt-cache currently supports only Bonsai-27B/Qwen3.5")
    if args.receipt and artifact_arch == "qwen35":
        if args.verify_mode != "fresh-oracle":
            raise ValueError(
                "Bonsai-27B receipts require --verify-mode fresh-oracle"
            )
        validate_bonsai35_receipt_identity(args.identity, info["digest"])
    ref = ref_class(art)
    t_load = time.time() - t_load0
    print(f"[bonsai] engine={engine_name} sampler={cfg.mode} seed={cfg.seed} "
          f"receipt-bound={is_receipt_safe(cfg)}", file=sys.stderr)
    print(
        f"[bonsai] context={context_profile.effective} "
        f"({'auto: ' + context_profile.reason if context_profile.automatic else context_profile.reason}) "
        f"source={context_profile.source_max} artifact={context_profile.artifact_max} "
        f"output-reserve={args.max_new}",
        file=sys.stderr,
    )
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
        verifier_ref = ref_class(verifier_art)
    gpu_executor = None
    gpu_report = None
    if getattr(args, "gpu", False) and artifact_arch == "qwen35" and not args.prompt_cache:
        try:
            gpu_executor, gpu_report = Bonsai35GpuExecutor.try_create_reported(art)
        except (MemoryError, RuntimeError, ValueError) as exc:
            print(f"[bonsai] Qwen3.5 resident CUDA unavailable ({exc}); clean CPU replay enabled",
                  file=sys.stderr)
        if gpu_executor is None:
            if gpu_report is not None:
                print(f"[bonsai] Qwen3.5 resident CUDA not started: {gpu_report.reason}; "
                      "using canonical CPU engine", file=sys.stderr)
            else:
                print("[bonsai] Qwen3.5 resident CUDA unavailable; using canonical CPU engine",
                      file=sys.stderr)
        else:
            print(f"[bonsai] Qwen3.5 resident CUDA graph enabled; "
                  f"memory-proof peak={gpu_executor.report.peak_used_bytes} bytes",
                  file=sys.stderr)
    print("[bonsai] receipt-bound native reference path; use --engine prismml.cpp only for raw "
          "non-receipt speed demos", file=sys.stderr)
    if args.bench:
        print(f"[bench] load={t_load:.3f}s", file=sys.stderr)

    prompts = [args.prompt] if args.prompt is not None else []
    interactive = bool(getattr(args, "repl", False) or args.prompt is None)
    status = 0
    bundle_dir = Path(args.bundle_dir)
    last_run: dict = {}
    terminal = TerminalRepl() if interactive else None
    conversation = None
    live_session = None
    pending_prompt: str | None = None
    if interactive:
        conversation = Conversation(
            lambda text: llama_tokenize(text, args.gguf, bin_dir=args.bin_dir),
            architecture=artifact_arch,
            context_size=context_profile.effective,
            max_new=args.max_new,
            eos_id=eos,
            chat=args.chat,
            # Interactive Qwen3.5 answers directly unless --think is explicit.
            thinking=bool(getattr(args, "think", False)),
            system_prompt=getattr(args, "system_prompt", "") or "",
            context_automatic=context_profile.automatic,
        )
        live_session = LiveNativeSession(
            ref,
            architecture=artifact_arch,
            artifact_digest=info["digest"],
            gpu_executor=gpu_executor,
        )
        print(
            f"[repl] {context_profile.model_name} · {engine_name} · "
            f"context {context_profile.effective} · input budget {conversation.input_budget} · "
            f"thinking={'on' if conversation.thinking else 'off'} · /help for commands",
            file=sys.stderr,
        )
    while True:
        if pending_prompt is not None:
            prompt, pending_prompt = pending_prompt, None
        elif prompts:
            prompt = prompts.pop(0)
        elif interactive:
            try:
                prompt = terminal.read("\nbonsai> ")
            except TerminalNoise as exc:
                print(f"[repl] {exc}", file=sys.stderr)
                continue
        else:
            prompt = None
        if prompt is None:
            break
        if interactive:
            if not prompt.strip():
                continue
            if prompt.strip().lower() in {"quit", "exit"}:
                break
            try:
                handled, next_prompt, exit_requested = _handle_session_command(
                    prompt,
                    terminal=terminal,
                    conversation=conversation,
                    context_profile=context_profile,
                    auto_context_profile=auto_context_profile,
                    last_run=last_run,
                    ref=ref,
                    model_digest=info["digest"],
                    bundle_dir=bundle_dir,
                )
            except ValueError as exc:
                print(f"[repl] {exc}", file=sys.stderr)
                continue
            if exit_requested:
                break
            if handled:
                pending_prompt = next_prompt
                continue
        t_tok0 = time.time()
        prepared = None
        if interactive:
            try:
                prepared = conversation.prepare(prompt)
            except ContextOverflow as exc:
                print(f"[repl] {exc}", file=sys.stderr)
                continue
            input_ids = list(prepared.input_ids)
            if prepared.evicted:
                print(
                    f"[repl] context full; evicting {prepared.evicted} oldest turn(s)",
                    file=sys.stderr,
                )
        else:
            try:
                model_prompt = (
                    _qwen3_chat_prompt(
                        prompt,
                        r.kv,
                        thinking=bool(getattr(args, "think", False) or not args.no_think),
                    )
                    if args.chat else prompt
                )
            except ValueError as exc:
                print(f"[bonsai] FATAL: {exc}", file=sys.stderr)
                status = 1
                break
            input_ids = llama_tokenize(model_prompt, args.gguf, bin_dir=args.bin_dir)
            budget = context_profile.input_budget(args.max_new)
            if len(input_ids) > budget:
                print(
                    f"[bonsai] prompt is {len(input_ids)} tokens but only {budget} fit after "
                    f"the {args.max_new}-token output reserve; lower -n or raise --context-size",
                    file=sys.stderr,
                )
                return 2
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

        try:
            guard = terminal.quarantine_input() if terminal else contextlib.nullcontext()
            with guard:
                output_ids = _generate_native_turn(
                    ref,
                    input_ids,
                    args=args,
                    cfg=cfg,
                    eos=eos,
                    on_token=on_token,
                    artifact_arch=artifact_arch,
                    artifact_digest=info["digest"],
                    gpu_executor=gpu_executor,
                    live_session=live_session,
                )
        except KeyboardInterrupt:
            if live_session is not None:
                live_session.invalidate()
            print("\n[repl] generation cancelled; partial response was not added to context",
                  file=sys.stderr)
            if not interactive:
                return 130
            continue
        tail = stream_decoder.decode(b"", final=True)
        if tail:
            out_parts.append(tail)
            sys.stdout.write(tail)
            sys.stdout.flush()
        output_text = "".join(out_parts)
        if interactive and output_ids:
            conversation.commit(prepared, output_ids, output_text)
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
                guard = terminal.quarantine_input() if terminal else contextlib.nullcontext()
                with guard:
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
            if conversation is not None:
                last_run["messages"] = conversation.messages()
                last_run["context"] = {
                    "effectiveTokens": conversation.context_size,
                    "inputBudgetTokens": conversation.input_budget,
                    "inputTokens": len(input_ids),
                    "outputReserveTokens": conversation.max_new,
                    "evictedTurnsTotal": conversation.evicted_total,
                }
            # Receipt bundles are DEFAULT-ON whenever receipts are on (disable with --no-bundle): package a
            # portable, re-executable LOCAL bundle (with the human-readable transcript) and print its path.
            if args.bundle and verification["ok"]:
                try:
                    path = _build_local_bundle(bundle, emission, out_dir=bundle_dir,
                                               model_label=last_run["model_label"], prompt=prompt,
                                               output_text=output_text, sampler=cfg,
                                               messages=last_run.get("messages"),
                                               context=last_run.get("context"))
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
    if gpu_executor is not None:
        gpu_executor.close()
    return status


def _run_prismml(args, cfg: SamplerConfig) -> int:
    cli = Path(args.bin_dir) / "llama-cli"
    reader = _GGUFReader(args.gguf)
    try:
        context_profile = resolve_context_window(
            reader.kv, backend="prismml.cpp", requested=args.ctx_size
        )
    except ValueError as exc:
        print(f"[bonsai] {exc}", file=sys.stderr)
        return 2
    cmd = [
        str(cli), "-m", args.gguf,
        "-n", str(args.max_new),
        "-t", os.environ.get("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 2) - 1))),
        "--simple-io",
        "--color", "off",
        "-c", str(context_profile.effective),
    ]
    if context_profile.automatic:
        cmd += ["--fit", "on"]
    if args.n_gpu_layers is not None:
        cmd += ["-ngl", str(args.n_gpu_layers)]
    if args.flash_attn:
        cmd += ["-fa", "on"]
    if args.prompt:
        # This PrismML build auto-enables conversation mode for chat-templated GGUFs. Without --single-turn,
        # EOF on our non-interactive stdin is treated as another empty turn and llama-cli loops printing
        # prompts after generation instead of exiting.
        cmd += ["-p", args.prompt, "--single-turn"]
    else:
        cmd += ["--conversation"]
    if cfg.mode == "greedy":
        cmd += ["--temp", "0"]
    else:
        cmd += ["--temp", str(cfg.temperature), "--top-k", str(cfg.top_k), "--top-p", str(cfg.top_p),
                "--min-p", str(cfg.min_p), "--seed", str(cfg.seed)]
    if args.rep_penalty != 1.0:
        cmd += ["--repeat-penalty", str(args.rep_penalty)]
    resolved = "hardware auto-fit" if context_profile.automatic else str(context_profile.effective)
    print(
        f"[bonsai] engine=prismml.cpp (fast raw GGUF; NO receipt) "
        f"context={resolved} source={context_profile.source_max}",
        file=sys.stderr,
    )
    return subprocess.run(cmd).returncode


def main(argv: list[str] | None = None) -> int:
    # allow_abbrev=False: the bonsai-notary launcher gates auto-funding + fresh-change on a LITERAL
    # ' --chain-confirm ' substring, so a prefix like --chain-conf must NOT resolve to the real
    # broadcast flag (that would spend with change-address hygiene skipped). Matches agent_cli.py/cli.py.
    ap = argparse.ArgumentParser(prog="trinote-run-bonsai",
                                 description="Run deterministic Bonsai-8B/27B or inference-only Bonsai backends",
                                 allow_abbrev=False)
    ap.add_argument("--gguf", default=_DEFAULT_GGUF)
    ap.add_argument("--artifact", default=_DEFAULT_ARTIFACT)
    ap.add_argument("--identity", default=_DEFAULT_IDENTITY)
    ap.add_argument("-p", "--prompt", default=None)
    ap.add_argument("--repl", action="store_true",
                    help="run the contextual interactive REPL (launchers use this explicitly)")
    ap.add_argument("-n", "--max-new", type=int, default=1024)
    ap.add_argument("--engine", choices=["native", "prismml.cpp"], default="native")
    ap.add_argument("--chat", action="store_true",
                    help="wrap prompts with the selected model's chat template")
    ap.add_argument("--no-think", action="store_true",
                    help="Qwen3.5 chat: use the model template's hard non-thinking prefix")
    ap.add_argument("--think", action="store_true",
                    help="Qwen3.5 REPL: start with reasoning enabled (interactive default is off)")
    ap.add_argument("--system", dest="system_prompt", default="",
                    help="REPL system message (changing it live with /system clears retained turns)")
    ap.add_argument("--sampler", choices=["min_p", "qwen3-rec", "bonsai27-rec", "greedy", "temp", "top_k", "top_p"],
                    default="qwen3-rec",
                    help="default 'qwen3-rec' = the Qwen3 vendor preset (top-p sampling at temp 0.6 / "
                         "top-k 20 / top-p 0.95); receipt-bound + byte-exactly reproducible at a fixed seed. "
                         "'bonsai27-rec' = Bonsai-27B's recommended temp 0.7 / top-k 20 / top-p 0.95. "
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
                    help="per-host opt-in: use the byte-identical CUDA producer (Qwen3.5 uses a fully "
                         "resident hybrid CUDA graph); cleanly replay/fall back to CPU when the library, "
                         "GPU memory, or an integer range guard is unavailable. Composes with --fast.")
    ap.add_argument("--bench", action="store_true",
                    help="print per-stage native timing to stderr")
    ap.add_argument("--prompt-cache", action="store_true",
                    help="Qwen3.5: persist and verify a content-addressed deterministic prefix state")
    ap.add_argument("--prompt-cache-dir", default=None,
                    help="override the prefix-cache directory (default: $BONSAI_NOTARY_HOME/prompt-cache/bonsai35)")
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
    ap.add_argument("--context-size", "--ctx-size", dest="ctx_size", type=_context_size_arg, default=None,
                    help="context tokens for any backend; 'auto' or 0 = model/artifact/hardware-aware auto")
    ap.add_argument("--n-gpu-layers", type=int, default=None,
                    help="PrismML llama.cpp GPU layers (-ngl); 99 offloads all Bonsai-27B layers")
    ap.add_argument("--flash-attn", action="store_true",
                    help="enable PrismML llama.cpp flash attention (-fa on)")
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
    defaults = load_config()
    env_context = os.environ.get("BONSAI_CONTEXT_SIZE")
    if env_context is not None:
        try:
            defaults["ctx_size"] = 0 if parse_context_size(env_context) is None else int(env_context)
        except ValueError as exc:
            ap.error(f"invalid BONSAI_CONTEXT_SIZE: {exc}")
    ap.set_defaults(**defaults)
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
