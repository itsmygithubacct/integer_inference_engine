"""trinote-quality-gate-bonsai — teacher-forced agreement vs PrismML llama.cpp."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from ..infer_int.artifact_io_bonsai import load_artifact_bonsai
from ..infer_int.bonsai_runtime import BONSAI35_PRISM_RUNTIME_RELEASE
from ..infer_int.gguf_tokenizer_v2 import load_gguf_tokens, decode, llama_tokenize, llama_complete
from ..infer_int.reference_bonsai import BonsaiReferenceModel
from ..infer_int.reference_bonsai35 import BonsaiQwen35ReferenceModel
from ..notary_paths import default_gguf, default_artifact, default_bin_dir

_DEFAULT_TEACHER_HARNESS = Path(__file__).resolve().parents[3] / "tools" / "prismml_teacher_forced"

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


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _run_teacher_harness(harness: str | Path, *, gguf: str | Path, full_ids: list[int],
                         start: int, ctx_size: int, threads: int, top_k: int,
                         n_gpu_layers: int, n_new: int = 0) -> dict:
    cmd = [
        str(harness),
        "--model", str(gguf),
        "--tokens", ",".join(str(i) for i in full_ids),
        "--start", str(start),
        "--ctx-size", str(ctx_size),
        "--threads", str(threads),
        "--top-k", str(top_k),
        "--gpu-layers", str(n_gpu_layers),
    ]
    if n_new:
        cmd.extend(["--n-new", str(n_new)])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"teacher-forced harness failed with exit {proc.returncode}: {msg[:800]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"teacher-forced harness produced invalid JSON: {proc.stdout[:800]}") from exc


def _model_for_artifact(artifact: dict):
    architecture = str(artifact.get("config", {}).get("architecture", ""))
    if architecture == "qwen3":
        return BonsaiReferenceModel(artifact), architecture
    if architecture == "qwen35":
        return BonsaiQwen35ReferenceModel(artifact), architecture
    raise ValueError(f"unsupported Bonsai quality-gate architecture {architecture!r}")


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
    ap.add_argument("--prism-gpu-layers", type=int, default=99,
                    help="PrismML GPU-offload layers for generation and teacher forcing")
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="min lenient top-1 (vs PrismML logits) agreement to PASS")
    ap.add_argument("--target-threshold", type=float, default=0.50,
                    help="min generated-target agreement (vs PrismML's own emitted token) to PASS")
    ap.add_argument("--prompt", action="append", help="override prompts (repeatable)")
    ap.add_argument("--teacher-harness", default=str(_DEFAULT_TEACHER_HARNESS),
                    help="compiled PrismML libllama teacher-forced harness")
    ap.add_argument("--top-k", type=int, default=10, help="top-k overlap/rank reporting from PrismML logits")
    ap.add_argument("--json-out", default=None, help="optional path to write gate result JSON")
    ap.add_argument("--fast", action="store_true",
                    help="enable the byte-identical packed-Q1 producer (strongly recommended for 27B)")
    ap.add_argument("--generated-only", action="store_true",
                    help="gate only agreement with PrismML's emitted greedy targets; use when the optional "
                         "teacher-forced libllama harness is unavailable")
    args = ap.parse_args(argv)

    prompts = args.prompt or _DEFAULT_PROMPTS
    print(f"[bonsai-gate] loading artifact {args.artifact} ...")
    art, info = load_artifact_bonsai(args.artifact)
    prov = info.get("provenance") or {}
    actual_gguf_digest = _sha256_file(args.gguf)
    if prov.get("ggufSha256") != actual_gguf_digest:
        raise RuntimeError(
            "quality-gate GGUF bytes do not match the imported artifact provenance"
        )
    runtime_release_path = Path(args.bin_dir) / ".runtime-release"
    try:
        runtime_release = runtime_release_path.read_text().splitlines()
    except OSError as exc:
        raise RuntimeError(
            f"quality gate cannot read pinned Prism runtime identity: {exc}"
        ) from exc
    if tuple(runtime_release) != BONSAI35_PRISM_RUNTIME_RELEASE:
        raise RuntimeError("quality gate Prism runtime identity is not the pinned 27B release")
    print(f"[bonsai-gate] provenance: {prov.get('source')}  "
          f"ggufSha256={prov.get('ggufSha256')}")
    ref, architecture = _model_for_artifact(art)
    if args.fast:
        enabled = bool(ref.enable_fast(check_ram=True, cache_output=True))
        if not enabled:
            raise RuntimeError("--fast requested but no native packed-Q1 producer is available")
    print(f"[bonsai-gate] architecture={architecture} producer={'native' if args.fast else 'oracle'}; "
          "keep --n-new small for smoke gates")
    tokens = load_gguf_tokens(args.gguf)

    total_target_match, total_top1_match, total_cnt = 0, 0, 0
    cases = []
    for p in prompts:
        prompt_ids = llama_tokenize(p, args.gguf, bin_dir=args.bin_dir)
        if args.generated_only:
            continuation = llama_complete(
                p, args.gguf, args.n_new,
                bin_dir=args.bin_dir,
                threads=args.threads,
                ctx_size=args.ctx_size,
                n_gpu_layers=args.prism_gpu_layers,
            )
            full_ids = llama_tokenize(p + continuation, args.gguf, bin_dir=args.bin_dir)
            plen = _common_prefix(prompt_ids, full_ids) or len(prompt_ids)
            refs = full_ids[plen:]
            prism_rows = []
        else:
            plen = len(prompt_ids)
            prism = _run_teacher_harness(
                args.teacher_harness,
                gguf=args.gguf,
                full_ids=prompt_ids,
                start=plen,
                ctx_size=args.ctx_size,
                threads=args.threads,
                top_k=args.top_k,
                n_gpu_layers=args.prism_gpu_layers,
                n_new=args.n_new,
            )
            prism_rows = prism["rows"]
            refs = [int(v) for v in prism.get("generatedIds", [])]
            full_ids = prompt_ids + refs
        if not refs:
            print(f"[bonsai-gate] {p!r}: no continuation produced — skipped")
            continue
        if prism_rows and len(prism_rows) != len(refs):
            raise RuntimeError("teacher-forced harness row/generated-ID count mismatch")
        logits = ref.forward(full_ids)
        preds = [int(logits[i].argmax()) for i in range(plen - 1, len(full_ids) - 1)]
        prism_top1 = [int(r["top1"]) for r in prism_rows]
        target_ranks = [int(r["targetRank"]) for r in prism_rows]
        target_matches = sum(int(a == b) for a, b in zip(preds, refs))
        top1_matches = sum(int(a == b) for a, b in zip(preds, prism_top1)) if prism_rows else 0
        total_target_match += target_matches
        total_top1_match += top1_matches
        total_cnt += len(refs)
        target_agree = target_matches / len(refs)
        top1_agree = top1_matches / len(refs) if prism_rows else None
        cases.append({
            "prompt": p,
            "inputIds": prompt_ids,
            "referenceIds": refs,
            "nativeIds": preds,
            "prismTop1Ids": prism_top1 if prism_rows else None,
            "referenceText": decode(refs, tokens, skip_special_from=len(tokens)),
            "nativeText": decode(preds, tokens, skip_special_from=len(tokens)),
            "prismTop1Text": (decode(prism_top1, tokens, skip_special_from=len(tokens))
                              if prism_rows else None),
            "targetMatches": target_matches,
            "top1Matches": top1_matches,
            "count": len(refs),
            "targetAgreement": target_agree,
            "top1Agreement": top1_agree,
            "targetRanks": target_ranks,
        })
        print(f"\n[bonsai-gate] prompt: {p!r}")
        print(f"              PrismML generated -> {cases[-1]['referenceText']!r}")
        if prism_rows:
            print(f"              PrismML logits    -> {cases[-1]['prismTop1Text']!r}   "
                  "(teacher-forced top-1)")
        print(f"              native logits     -> {cases[-1]['nativeText']!r}   (teacher-forced top-1)")
        if prism_rows:
            print(f"              top1 agreement    = {top1_matches}/{len(refs)} = {top1_agree:.2f}")
        print(f"              target agreement  = {target_matches}/{len(refs)} = {target_agree:.2f} "
              f"target ranks={target_ranks}")

    overall = (total_top1_match / total_cnt if total_cnt else 0.0) if not args.generated_only else None
    target_overall = total_target_match / total_cnt if total_cnt else 0.0
    # PASS requires BOTH the lenient top-1 agreement AND the stricter generated-target agreement to clear
    # their thresholds — top-1 alone (logit-argmax overlap) can look perfect while the actually-emitted
    # token diverges, so gating on both keeps the verdict honest.
    top1_ok = True if args.generated_only else bool(overall is not None and overall >= args.threshold)
    target_ok = target_overall >= args.target_threshold
    verdict = "PASS" if (top1_ok and target_ok) else "FAIL"
    harness_path = Path(args.teacher_harness)
    harness_sha256 = None
    if not args.generated_only and harness_path.is_file():
        harness_sha256 = hashlib.sha256(harness_path.read_bytes()).hexdigest()
    artifact_path = Path(args.artifact)
    gguf_path = Path(args.gguf)
    bin_dir = Path(args.bin_dir)
    result = {
        "metric": ("generated-target-agreement-vs-prismml" if args.generated_only
                   else "teacher-forced-top1-agreement-vs-prismml-libllama"),
        "architecture": architecture,
        "artifactSha256": info["digest"],
        # Evidence files may be published. Record portable provenance labels,
        # never the operator's username or machine-specific absolute paths.
        "artifactPath": artifact_path.name,
        "ggufPath": gguf_path.name,
        "ggufSha256": actual_gguf_digest,
        "producer": "native" if args.fast else "oracle",
        "generatedOnly": bool(args.generated_only),
        "prism": {
            "binDir": str(Path(bin_dir.parent.name) / bin_dir.name),
            "runtimeRelease": runtime_release,
            "gpuLayers": args.prism_gpu_layers,
            "teacherHarness": (harness_path.name if not args.generated_only else None),
            "teacherHarnessSha256": harness_sha256,
        },
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
    if not args.generated_only:
        print(f"\n[bonsai-gate] OVERALL teacher-forced top1 agreement = "
              f"{total_top1_match}/{total_cnt} = {overall:.3f} "
              f"(threshold {args.threshold:.2f}) -> {'OK' if top1_ok else 'LOW'}")
    print(f"[bonsai-gate] generated-target agreement = {total_target_match}/{total_cnt} = {target_overall:.3f} "
          f"(threshold {args.target_threshold:.2f}) -> {'OK' if target_ok else 'LOW'}")
    print(f"[bonsai-gate] VERDICT (both must pass) -> {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
