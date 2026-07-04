"""Bonsai JSON mode — structured queries with a hashed reproduction log and a cache-fetch.

A query asks the model to emit `{"thinking", "answer"}`; the notary then injects `bonsai` (perf), `receipt`
(the verified local receipt), `bundle` (the local re-executable bundle path), and `reproduction` (the
determinism record). The final object has six fields:

    { "thinking", "answer",                         <- shipped by the LLM
      "bonsai", "receipt", "bundle", "reproduction" } <- added by the notary

REPRODUCTION LOG: every run is appended to a JSONL log keyed by

    key = sha256("bonsai-json-key/v2" + modelHash + canonical(samplerBlock) + inputCommit + maxNew)

i.e. an IDENTICAL prompt with IDENTICAL context (same committed model, same sampler, same tokenized input,
same generation budget) maps to the same key. Because the engine is byte-exact deterministic, two runs with
the same key MUST produce the same `outputCommit`; the log lets us (a) PROVE byte-exact reproduction over time
by comparing against the first recorded entry, and (b) FETCH a previously-inferenced response without
re-running the model.

INTEGRITY: the JSONL log is local and editable, so the fetch path does not blindly trust it — before serving
a cached answer it (1) confirms the on-disk artifact still hashes to the key's modelHash (the same fail-closed
identity/artifact binding the producer path enforces), and (2) re-checks the entry's own `answerCommit`. A
fetched result is explicitly marked `reExecuted: false` — the authoritative proof remains the receipt + the
re-executable bundle the run path wrote.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - POSIX only; single-process use is unaffected (see ledger.py)
    fcntl = None

# All notary state lives outside the repo under $BONSAI_NOTARY_HOME (default ~/.local/trinote).
from ..notary_paths import notary_home, sessions_log_default, debug_dir_default  # noqa: E402

_NOTARY_HOME = notary_home()
DEFAULT_JSON_LOG = sessions_log_default()
DEFAULT_JSON_INSTR = ": respond in json with a 'thinking' part and an 'answer' part"
_KEY_FRAC = 16   # Bonsai is fixed-point frac=16; the sampler block (hence the key) is computed at this scale.
# Enhanced perf logging for model-improvement analysis (~/.local/trinote/debug).
DEFAULT_DEBUG_DIR = debug_dir_default()


def cache_key(model_hash: str, sampler_block: dict, input_commit: str, max_new: int) -> str:
    """Stable key for an identical prompt+context. Same (model, sampler, tokenized input, generation budget)
    -> same key. max_new is bound because a smaller cap truncates the output to a different prefix."""
    from ..receipts.canonical import canonical_bytes
    h = hashlib.sha256()
    h.update(b"bonsai-json-key/v2\x00")
    h.update((model_hash or "").encode("utf-8") + b"\x00")
    h.update(canonical_bytes(sampler_block) + b"\x00")
    h.update((input_commit or "").encode("utf-8") + b"\x00")
    h.update(f"maxNew={int(max_new)}".encode("utf-8"))
    return h.hexdigest()


def _answer_commit(result_or_entry: dict) -> str:
    """A commitment over the human-facing (thinking, answer) so a tampered cached answer line is detectable."""
    return hashlib.sha256(json.dumps(
        {"thinking": result_or_entry.get("thinking", ""), "answer": result_or_entry.get("answer", "")},
        sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def extract_model_json(text: str) -> dict:
    """Pull the model's JSON {thinking, answer} from its (possibly messy / truncated) output; never raise.

    Returns {"thinking", "answer", "extracted"} where `extracted` is:
      "json"         a complete, balanced JSON object with an answer/thinking field was parsed;
      "json-partial" the JSON was truncated, but the `"answer": "..."` field itself was located by regex
                     (so the model DID produce an answer field — grade it);
      "none"         no answer field could be found (the model never emitted one — e.g. truncated mid-thinking).
                     `answer` is then the raw text (so the interactive --answer output still shows something),
                     but a strict grader should treat "none" as no-answer (do not grade the raw text, which
                     would let a question echo leak in).
    """
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    # 1) a complete balanced JSON object (single O(n) scan)
    depth = 0
    start = -1
    for i, c in enumerate(t):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    obj = json.loads(t[start:i + 1])
                except ValueError:
                    obj = None
                if isinstance(obj, dict) and ("answer" in obj or "thinking" in obj):
                    return {"thinking": _as_text(obj.get("thinking", "")),
                            "answer": _as_text(obj.get("answer", "")), "extracted": "json"}
    # 2) truncated JSON, but the answer field is present and closed — locate it directly
    am = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', t)
    if am:
        tm = re.search(r'"thinking"\s*:\s*"((?:[^"\\]|\\.)*)"', t)
        try:
            answer = json.loads('"' + am.group(1) + '"')
        except ValueError:
            answer = am.group(1)
        try:
            thinking = json.loads('"' + tm.group(1) + '"') if tm else ""
        except ValueError:
            thinking = tm.group(1) if tm else ""
        return {"thinking": thinking, "answer": answer, "extracted": "json-partial"}
    # 3) no answer field at all
    return {"thinking": "", "answer": t, "extracted": "none"}


def _as_text(v) -> str:
    return v if isinstance(v, str) else ("" if v is None else json.dumps(v, ensure_ascii=False))


def log_lookup(log_path: str | Path, key: str) -> tuple[dict | None, int]:
    """Return (first-recorded entry for `key`, count of entries with `key`), streaming line-by-line (peak
    memory = one line). The FIRST entry is the canonical reference a later run is compared against."""
    p = Path(log_path)
    if not p.exists():
        return None, 0
    first = None
    count = 0
    with open(p, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("key") == key:
                count += 1
                if first is None:
                    first = e
    return first, count


def log_append(log_path: str | Path, entry: dict) -> None:
    """Append one entry atomically: advisory-locked (POSIX) + flushed + fsync'd so concurrent JSON-mode runs
    do not interleave/tear lines in the reproduction log (mirrors receipts/ledger.py)."""
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n"
    with open(p, "a", encoding="utf-8") as fh:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def debug_log(record: dict, debug_dir: str | None = None) -> str:
    """Append one enhanced-perf record (timing, tokens, key, receipt) to ~/.local/trinote/debug for use in
    improving the model. Returns the log path."""
    d = Path(debug_dir or DEFAULT_DEBUG_DIR)
    d.mkdir(parents=True, exist_ok=True)
    f = d / "bonsai-debug.jsonl"
    line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
    with open(f, "a", encoding="utf-8") as fh:
        if fcntl is not None:                       # advisory lock so PARALLEL shards don't tear lines
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return str(f)


def _home_relative(path: str | None) -> str | None:
    if not path:
        return path
    home = os.path.expanduser("~")
    ap = os.path.abspath(path)
    return os.path.relpath(ap, home) if ap.startswith(home + os.sep) else path


def _emit(result: dict, args) -> None:
    """Print per the output flags: --answer (just the string), --json-filter (jq), else full JSON."""
    if getattr(args, "answer", False):
        print(result.get("answer", ""))
        return
    flt = getattr(args, "json_filter", None)
    blob = json.dumps(result, indent=2, ensure_ascii=False)
    if flt:
        try:
            proc = subprocess.run(["jq", flt], input=blob, text=True, capture_output=True)
            if proc.returncode == 0:
                sys.stdout.write(proc.stdout)
                return
            sys.stderr.write(proc.stderr)   # jq error: fall back to the unfiltered object (still valid JSON)
        except FileNotFoundError:
            print("[json] jq not found; emitting unfiltered JSON", file=sys.stderr)
    print(blob)


def _vlog(args, msg: str) -> None:
    if getattr(args, "verbose", False):
        print(msg, file=sys.stderr)


def run_json(args, cfg) -> int:
    """Orchestrate a Bonsai JSON-mode query: tokenize -> key -> (fetch cached | run+record+verify-reproduction)
    -> emit. Returns an exit status (non-zero on a reproduction MISMATCH — a determinism failure)."""
    from .run_bonsai_cli import _qwen3_chat_prompt, _special_token_cutoff, _build_local_bundle
    from ..infer_int.gguf_tokenizer_v2 import load_gguf_tokens, llama_tokenize, token_bytes
    from ..infer_int.import_gguf_v2 import _GGUFReader
    from ..infer_int.bonsai_runtime import (identity_model_hash, generate_bonsai_tokens,
                                            emit_and_verify_bonsai_receipt)
    from ..receipts.canonical import token_commit
    from ..receipts.receipt import sampler_to_block
    from ..hashing.sha import sha256_file

    if args.verify_mode == "fresh-oracle":
        print("[json] --verify-mode fresh-oracle is not supported in JSON mode (no verifier model is built); "
              "use fast-local", file=sys.stderr)
        return 2
    question = args.prompt
    if not question:
        print("[json] JSON mode needs a question via --prompt", file=sys.stderr)
        return 2
    instr = getattr(args, "json_instr", None) or DEFAULT_JSON_INSTR
    prompt = f"{question}{instr}"

    r = _GGUFReader(args.gguf)
    try:
        model_prompt = _qwen3_chat_prompt(prompt, r.kv) if args.chat else prompt
    except ValueError as exc:
        print(f"[json] FATAL: {exc}", file=sys.stderr)
        return 2
    _vlog(args, f"[json] prompt sent to model: {prompt}")
    t_tok0 = time.time()
    input_ids = llama_tokenize(model_prompt, args.gguf, bin_dir=args.bin_dir)
    t_tokenize = time.time() - t_tok0
    input_commit = token_commit(input_ids)
    sampler_block = sampler_to_block(cfg, _KEY_FRAC)
    try:
        model_hash = identity_model_hash(args.identity)
    except (ValueError, FileNotFoundError, OSError) as exc:
        # identity_model_hash now fails closed on a missing/malformed identity; surface it as a
        # clean '[json] FATAL' like the native path instead of a raw traceback (review-2 #15).
        print(f"[json] FATAL: {exc}", file=sys.stderr)
        return 1
    key = cache_key(model_hash, sampler_block, input_commit, args.max_new)
    first, count = log_lookup(args.json_log, key)

    # FETCH MODE: serve the already-inferenced response without loading the model — but only after binding it
    # to the ACTUAL installed artifact (the producer's fail-closed invariant) and re-checking the entry's own
    # commitment. Any failure falls through to a real run.
    if getattr(args, "fetch", False) and first is not None:
        actual = sha256_file(args.artifact)
        stored = first.get("result", {})
        if actual != model_hash:
            _vlog(args, f"[json] cache hit but artifact {actual[:12]} != key modelHash "
                        f"{str(model_hash)[:12]} — re-running")
        elif _answer_commit(stored) != first.get("answerCommit"):
            _vlog(args, "[json] cache hit FAILED integrity (answerCommit mismatch) — re-running")
        else:
            result = dict(stored)
            result["reproduction"] = {"key": key, "cacheHit": True, "reExecuted": False, "seenBefore": True,
                                      "reproduced": None, "priorReceiptHash": first.get("receiptHash"),
                                      "count": count}
            _vlog(args, f"[json] cache HIT {key[:16]} — returning stored response (no inference)")
            _emit(result, args)
            return 0

    # RUN: load the model and infer.
    from ..infer_int.artifact_io_bonsai import load_artifact_bonsai
    from ..infer_int.reference_bonsai import BonsaiReferenceModel
    _vlog(args, f"[json] {'cache MISS' if getattr(args, 'fetch', False) else 'running'} — loading model + inferring")
    eos = int(r.kv.get("tokenizer.ggml.eos_token_id", -1))
    tokens = load_gguf_tokens(args.gguf)
    special_from = _special_token_cutoff(r, len(tokens))
    t_load0 = time.time()
    art, info = load_artifact_bonsai(args.artifact)
    model = BonsaiReferenceModel(art)
    engine_path = "oracle"
    if args.fast:
        if model.enable_native():
            engine_path = "native"
        elif model.enable_fast(check_ram=True, cache_output=True):
            engine_path = "sign-cache"
        elif getattr(args, "fast_required", False):
            print("[json] FATAL: --fast-required set but the fast path could not be enabled", file=sys.stderr)
            return 1
    t_load = time.time() - t_load0

    # Time-to-first-token (~prefill + 1 decode step) vs steady-state decode — the key perf split for targeting
    # gains (prefill matmul vs autoregressive decode vs the verify re-execution overhead below).
    _ttft = {"t": None}

    def _on_tok(_tok):
        if _ttft["t"] is None:
            _ttft["t"] = time.time()

    t0 = time.time()
    output_ids = generate_bonsai_tokens(model, input_ids, args.max_new, sampler=cfg, eos=eos, on_token=_on_tok)
    dt = time.time() - t0
    n_out = len(output_ids)
    ttft = (_ttft["t"] - t0) if _ttft["t"] is not None else dt
    steady_per_tok = (dt - ttft) / max(n_out - 1, 1)
    finish = ("eos" if (output_ids and int(output_ids[-1]) == eos)
              else ("cap" if n_out >= args.max_new else "stop"))

    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    out_parts = [dec.decode(token_bytes(t, tokens, skip_special_from=special_from)) for t in output_ids]
    out_parts.append(dec.decode(b"", final=True))
    output_text = "".join(p for p in out_parts if p)

    t_verify0 = time.time()
    bundle, verification, emission = emit_and_verify_bonsai_receipt(
        model, input_ids=input_ids, output_ids=output_ids, model_digest=info["digest"], sampler=cfg,
        verifier_mode="fast-local", identity_path=args.identity, ledger_path=args.ledger,
        broadcast_log=args.broadcast_log, broadcast_to_log=args.broadcast_to_log,
        enable_chain=False, chain_backend=None, tx_log=(args.tx_log or None))
    t_verify = time.time() - t_verify0

    receipt = bundle["receipt"]
    output_commit = receipt["outputCommit"]
    rx = verification.get("reexec") or {}
    verified = bool(verification.get("ok"))
    ledger_entry = emission.get("ledgerEntry") or {}

    bundle_path = None
    if args.bundle and verified:
        try:
            bundle_path = _home_relative(_build_local_bundle(
                bundle, emission, out_dir=Path(args.bundle_dir), model_label="",
                prompt=question, output_text=output_text, sampler=cfg))
        except Exception as exc:   # noqa: BLE001 - bundle packaging must not break the query result
            _vlog(args, f"[json] bundle packaging failed: {exc}")

    model_json = extract_model_json(output_text)
    result = {
        "thinking": model_json["thinking"],
        "answer": model_json["answer"],
        "bonsai": {
            "engine": "int-ref@bonsai-qwen3", "sampler": cfg.mode, "seed": cfg.seed,
            "tokens": len(output_ids), "maxNew": int(args.max_new), "finish": finish,
            "seconds": round(dt, 1), "secPerTok": round(dt / max(len(output_ids), 1), 3),
            "fastPath": bool(args.fast),
        },
        "receipt": {
            "receiptHash": receipt["receiptHash"],
            "status": "VERIFIED" if verified else "FAILED",
            "mode": verification.get("verificationMode"),
            "strategy": rx.get("strategy"),
            "ledger": ledger_entry.get("index"),
            "broadcast": (emission.get("onchain") or {}).get("status"),
        },
        "bundle": bundle_path,
    }

    # REPRODUCTION: compare this run's outputCommit to the FIRST recorded run for this key.
    status = 0 if verified else 1
    if first is not None:
        reproduced = (output_commit == first.get("outputCommit"))
        result["reproduction"] = {"key": key, "cacheHit": False, "reExecuted": True, "seenBefore": True,
                                  "reproduced": reproduced, "priorReceiptHash": first.get("receiptHash"),
                                  "count": count + 1}
        if not reproduced:
            status = 1   # byte-exact reproduction FAILED — a determinism violation.
            _vlog(args, f"[json] REPRODUCTION MISMATCH for key {key[:16]}: "
                        f"{output_commit} != {first.get('outputCommit')}")
    else:
        result["reproduction"] = {"key": key, "cacheHit": False, "reExecuted": True, "seenBefore": False,
                                  "reproduced": None, "priorReceiptHash": None, "count": 1}

    log_append(args.json_log, {
        "key": key, "modelHash": model_hash, "artifactDigest": info["digest"], "inputCommit": input_commit,
        "outputCommit": output_commit, "receiptHash": receipt["receiptHash"], "sampler": sampler_block,
        "maxNew": int(args.max_new), "finish": finish, "answerCommit": _answer_commit(result),
        "question": question, "result": result,
    })
    if getattr(args, "debug", False):
        n_in = len(input_ids)
        ctx = n_in + n_out
        omp = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
        wall = t_tokenize + t_load + dt + t_verify
        rec = {
            "ts": time.time(), "schema": "bonsai-debug/v1", "key": key, "receiptHash": receipt["receiptHash"],
            "modelHash": model_hash, "question": question,
            # --- token counts / context (attention cost grows with context) ---
            "inputTokens": n_in, "outputTokens": n_out, "contextTokens": ctx,
            "maxNew": int(args.max_new), "finish": finish,
            # --- PER-PHASE wall time (where the time goes — the optimization targets) ---
            "tokenizeSeconds": round(t_tokenize, 4),       # llama-tokenize subprocess
            "loadSeconds": round(t_load, 4),               # artifact mmap + fast-path enable
            "ttftSeconds": round(ttft, 4),                 # ~prefill + first decode token
            "decodeTotalSeconds": round(dt, 4),            # full generate() (prefill + all decode)
            "steadyDecodeSecPerTok": round(steady_per_tok, 5),  # post-first-token autoregressive cost
            "verifyEmitSeconds": round(t_verify, 4),       # receipt re-execution + emit (the verify overhead)
            "wallSecondsTracked": round(wall, 4),
            # --- throughput ---
            "decodeTokPerSec": round(n_out / dt, 4) if dt > 0 else 0.0,
            "prefillTokPerSecApprox": round(n_in / ttft, 2) if ttft > 0 else 0.0,
            # --- engine / environment (perf depends on these) ---
            "enginePath": engine_path, "fastPath": bool(args.fast), "ompThreads": omp,
            "samplerMode": cfg.mode,
            # --- correctness context ---
            "verified": verified, "reproduced": result["reproduction"].get("reproduced"),
            "strategy": rx.get("strategy"), "verifyMode": verification.get("verificationMode"),
        }
        path = debug_log(rec)
        _vlog(args, f"[json] debug perf appended -> {path}")
    _emit(result, args)
    return status
