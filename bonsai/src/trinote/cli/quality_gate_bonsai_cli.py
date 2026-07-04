"""trinote-quality-gate-bonsai — teacher-forced agreement vs PrismML llama.cpp."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from ..infer_int.artifact_io_bonsai import load_artifact_bonsai
from ..infer_int.gguf_tokenizer_v2 import load_gguf_tokens, decode, llama_tokenize, llama_complete
from ..infer_int.reference_bonsai import BonsaiReferenceModel
from ..notary_paths import default_gguf, default_artifact, default_bin_dir

# A small, diverse smoke set. NOTE: a production gate needs DOZENS-to-HUNDREDS of prompts (and ideally
# held-out, domain-spanning ones) before its agreement number is statistically meaningful; this set is
# only a fast sanity check for the import.
_DEFAULT_PROMPTS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "The opposite of hot is",
    "Two plus two equals",
    "The first president of the United States was",
    "The chemical symbol for gold is",
    "The largest planet in our solar system is",
    "Roses are red, violets are",
    "The speed of light is approximately",
    "Once upon a time, there was a",
]


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _run_teacher_harness(harness: str | Path, *, gguf: str | Path, full_ids: list[int],
                         start: int, ctx_size: int, threads: int, top_k: int) -> dict:
    cmd = [
        str(harness),
        "--model", str(gguf),
        "--tokens", ",".join(str(i) for i in full_ids),
        "--start", str(start),
        "--ctx-size", str(ctx_size),
        "--threads", str(threads),
        "--top-k", str(top_k),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"teacher-forced harness failed with exit {proc.returncode}: {msg[:800]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"teacher-forced harness produced invalid JSON: {proc.stdout[:800]}") from exc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="trinote-quality-gate-bonsai",
        description="Teacher-forced agreement: native Bonsai int-ref vs PrismML llama.cpp",
    )
    ap.add_argument("--artifact", default=default_artifact())
    ap.add_argument("--gguf", default=default_gguf())
    ap.add_argument("--bin-dir", default=default_bin_dir())
    ap.add_argument("--n-new", type=int, default=4,
                    help="PrismML raw continuation length to teacher-force")
    ap.add_argument("--ctx-size", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="min lenient top-1 (vs PrismML logits) agreement to PASS")
    ap.add_argument("--target-threshold", type=float, default=0.50,
                    help="min generated-target agreement (vs PrismML's own emitted token) to PASS")
    ap.add_argument("--prompt", action="append", help="override prompts (repeatable)")
    ap.add_argument("--teacher-harness", default="tools/prismml_teacher_forced",
                    help="compiled PrismML libllama teacher-forced harness")
    ap.add_argument("--top-k", type=int, default=10, help="top-k overlap/rank reporting from PrismML logits")
    ap.add_argument("--json-out", default=None, help="optional path to write gate result JSON")
    args = ap.parse_args(argv)

    prompts = args.prompt or _DEFAULT_PROMPTS
    print(f"[bonsai-gate] loading artifact {args.artifact} ...")
    art, info = load_artifact_bonsai(args.artifact)
    prov = info.get("provenance") or {}
    print(f"[bonsai-gate] provenance: {prov.get('source')}  "
          f"ggufSha256={prov.get('ggufSha256')}")
    print("[bonsai-gate] native 8B teacher-forced forwards are slow; keep --n-new small for smoke gates")
    ref = BonsaiReferenceModel(art)
    tokens = load_gguf_tokens(args.gguf)

    total_target_match, total_top1_match, total_cnt = 0, 0, 0
    cases = []
    for p in prompts:
        prompt_ids = llama_tokenize(p, args.gguf, bin_dir=args.bin_dir)
        continuation = llama_complete(
            p, args.gguf, args.n_new,
            bin_dir=args.bin_dir,
            threads=args.threads,
            ctx_size=args.ctx_size,
        )
        full_ids = llama_tokenize(p + continuation, args.gguf, bin_dir=args.bin_dir)
        plen = _common_prefix(prompt_ids, full_ids) or len(prompt_ids)
        refs = full_ids[plen:]
        if not refs:
            print(f"[bonsai-gate] {p!r}: no continuation produced — skipped")
            continue
        prism = _run_teacher_harness(
            args.teacher_harness,
            gguf=args.gguf,
            full_ids=full_ids,
            start=plen,
            ctx_size=args.ctx_size,
            threads=args.threads,
            top_k=args.top_k,
        )
        prism_rows = prism["rows"]
        logits = ref.forward(full_ids)
        preds = [int(logits[i].argmax()) for i in range(plen - 1, len(full_ids) - 1)]
        prism_top1 = [int(r["top1"]) for r in prism_rows]
        target_ranks = [int(r["targetRank"]) for r in prism_rows]
        target_matches = sum(int(a == b) for a, b in zip(preds, refs))
        top1_matches = sum(int(a == b) for a, b in zip(preds, prism_top1))
        total_target_match += target_matches
        total_top1_match += top1_matches
        total_cnt += len(refs)
        target_agree = target_matches / len(refs)
        top1_agree = top1_matches / len(refs)
        cases.append({
            "prompt": p,
            "referenceText": decode(refs, tokens, skip_special_from=len(tokens)),
            "nativeText": decode(preds, tokens, skip_special_from=len(tokens)),
            "prismTop1Text": decode(prism_top1, tokens, skip_special_from=len(tokens)),
            "targetMatches": target_matches,
            "top1Matches": top1_matches,
            "count": len(refs),
            "targetAgreement": target_agree,
            "top1Agreement": top1_agree,
            "targetRanks": target_ranks,
        })
        print(f"\n[bonsai-gate] prompt: {p!r}")
        print(f"              PrismML generated -> {cases[-1]['referenceText']!r}")
        print(f"              PrismML logits    -> {cases[-1]['prismTop1Text']!r}   (teacher-forced top-1)")
        print(f"              native logits     -> {cases[-1]['nativeText']!r}   (teacher-forced top-1)")
        print(f"              top1 agreement    = {top1_matches}/{len(refs)} = {top1_agree:.2f}")
        print(f"              target agreement  = {target_matches}/{len(refs)} = {target_agree:.2f} "
              f"target ranks={target_ranks}")

    overall = total_top1_match / total_cnt if total_cnt else 0.0
    target_overall = total_target_match / total_cnt if total_cnt else 0.0
    # PASS requires BOTH the lenient top-1 agreement AND the stricter generated-target agreement to clear
    # their thresholds — top-1 alone (logit-argmax overlap) can look perfect while the actually-emitted
    # token diverges, so gating on both keeps the verdict honest.
    top1_ok = overall >= args.threshold
    target_ok = target_overall >= args.target_threshold
    verdict = "PASS" if (top1_ok and target_ok) else "FAIL"
    result = {
        "metric": "teacher-forced-top1-agreement-vs-prismml-libllama",
        "value": overall,
        "targetAgreement": target_overall,
        "threshold": args.threshold,
        "targetThreshold": args.target_threshold,
        "top1Pass": top1_ok,
        "targetPass": target_ok,
        "verdict": verdict,
        "top1Matches": total_top1_match,
        "targetMatches": total_target_match,
        "count": total_cnt,
        "cases": cases,
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"\n[bonsai-gate] OVERALL teacher-forced top1 agreement = "
          f"{total_top1_match}/{total_cnt} = {overall:.3f} "
          f"(threshold {args.threshold:.2f}) -> {'OK' if top1_ok else 'LOW'}")
    print(f"[bonsai-gate] generated-target agreement = {total_target_match}/{total_cnt} = {target_overall:.3f} "
          f"(threshold {args.target_threshold:.2f}) -> {'OK' if target_ok else 'LOW'}")
    print(f"[bonsai-gate] VERDICT (both must pass) -> {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
