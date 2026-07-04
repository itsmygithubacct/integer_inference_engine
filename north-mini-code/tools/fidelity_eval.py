#!/usr/bin/env python3
"""Fidelity eval for the all-INTEGER north-mini-code engine vs the float model.

Metrics (greedy / temperature 0), pick any subset with --metrics:
  ppl      the integer engine's teacher-forced self-perplexity over the corpus (exp mean NLL). Logits are
           fixed-point at 2**fa, de-scaled to nat units, so ppl is COMPARABLE ACROSS an fa sweep.  [no Ollama]
  topk     is Ollama's greedy next token within OUR top-k logits? top-1 == exact next-token agreement.  [Ollama]
  freerun  greedy N tokens both sides; common leading CHARACTERS before divergence.                     [Ollama]

The metric math + runners live in `nmc.fidelity` (pure, unit-tested in tests/test_fidelity.py); this file is
just CLI + model/Ollama I/O + reporting. Needs the real GGUF blob (vocab_only is NOT enough — this runs the
forward), a running Ollama with the model pulled (for topk/freerun), and — since the blob is ollama-owned —
usually sudo.

    sudo env NMC_FA=16 PYTHONPATH=src .venv/bin/python tools/fidelity_eval.py <blob> --metrics ppl,topk,freerun
    # perplexity fa sweep (re-execs itself once per fa, no Ollama needed):
    sudo env PYTHONPATH=src .venv/bin/python tools/fidelity_eval.py <blob> --fa-sweep 12,14,16 --metrics ppl --json sweep.json
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

from nmc.engine import Engine, FA
from nmc import fidelity as F

MODEL = "north-mini-code-1.0:latest"

# Default corpus: factual/short-answer + code + quote continuations (greedy-stable, single clear next token).
DEFAULT_PROMPTS = [
    "The capital of France is", "The capital of Japan is", "The opposite of hot is",
    "Water is made of hydrogen and", "The first president of the United States was",
    "2 + 2 =", "The square root of 144 is", "Roses are red, violets are",
    "The largest planet in the solar system is", "The chemical symbol for gold is",
    "def add(a, b):\n    return", "import numpy as", "for i in range(10):\n    print(",
    "The speed of light is approximately", "Once upon a time, there was a",
    "The mitochondria is the powerhouse of the", "To be or not to be, that is the",
    "The three primary colors are red, blue, and", "An apple a day keeps the",
    "The author of Romeo and Juliet is", "E equals m c", "The boiling point of water is",
    "The currency of the United Kingdom is the", "A group of lions is called a",
]


def load_prompts(path):
    """`.json` → list[str]; else one prompt per line with literal `\\n`/`\\t` escapes decoded (blank lines
    and `#` comments skipped)."""
    if path.endswith(".json"):
        return json.loads(open(path).read())
    out = []
    for line in open(path):
        s = line.rstrip("\n")
        if not s or s.lstrip().startswith("#"):
            continue
        out.append(s.encode().decode("unicode_escape"))
    return out


def make_ollama_fn(model, url):
    def ollama(prompt, n):
        req = urllib.request.Request(url,
            data=json.dumps({"model": model, "prompt": prompt, "raw": True, "stream": False,
                             "options": {"temperature": 0, "top_k": 1, "num_predict": n}}).encode(),
            headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=600)).get("response", "")
    return ollama


def run_one(blob, prompts, metrics, n_free, model, url):
    """One fa (from NMC_FA / engine module) → results dict for the requested metrics."""
    eng = Engine(blob)
    out = {"fa": FA, "backend": eng.bname, "model": model, "n_prompts": len(prompts)}
    print(f"[fidelity] fa={FA} backend={eng.bname} model={model} {len(prompts)} prompts "
          f"metrics={','.join(metrics)}", flush=True)
    ollama = make_ollama_fn(model, url)
    if "ppl" in metrics:
        t = time.time(); out["ppl"] = F.eval_ppl(eng, prompts, FA)
        print(f"[ppl]     corpus perplexity = {out['ppl']['corpus_ppl']:.3f}  "
              f"over {out['ppl']['n_tokens']} tokens  ({time.time()-t:.0f}s)")
    if "topk" in metrics:
        t = time.time(); out["topk"] = F.eval_topk(eng, prompts, ollama)
        pk = out["topk"]["pct"]
        print(f"[topk]    " + "  ".join(f"top-{k} {pk[k]:.0f}%" for k in out["topk"]["ks"])
              + f"  ({time.time()-t:.0f}s)")
        for r in out["topk"]["per_prompt"]:
            flag = "=1 " if r["rank"] == 1 else (f"~{r['rank']}".ljust(3) if r["rank"] else "X  ")
            print(f"          {flag} {r['prompt']!r:42} ours={r['ours_top1']!r:12} ollama={r['ollama']!r}")
    if "freerun" in metrics:
        t = time.time(); out["freerun"] = F.eval_freerun(eng, prompts, ollama, n_free)
        print(f"[freerun] mean common prefix = {out['freerun']['mean_common_prefix_chars']:.0f} chars  "
              f"(greedy {n_free} tok, {time.time()-t:.0f}s)")
        for r in out["freerun"]["per_prompt"]:
            print(f"          cp={r['common_prefix_chars']:3d}  {r['prompt']!r:38} "
                  f"ours={r['ours']!r} ollama={r['ollama']!r}")
    return out


def sweep(script, blob, fas, metrics, passthrough, jpath):
    """Re-exec this tool once per fa (the engine reads NMC_FA at import, so a fresh process per fa is the
    clean, isolated way) and aggregate their JSON into an fa-vs-perplexity table. `passthrough` forwards the
    corpus/limit/n-free/model flags so every fa runs on the SAME prompts."""
    results = []
    for fa in fas:
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
        env = {**os.environ, "NMC_FA": str(fa)}
        cmd = [sys.executable, script, blob, "--metrics", ",".join(metrics), "--json", tmp, *passthrough]
        try:
            if subprocess.run(cmd, env=env).returncode == 0:
                results.append(json.load(open(tmp)))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    print("\n[fa-sweep] fa   corpus_ppl" + ("   top-1%" if "topk" in metrics else ""))
    for r in results:
        line = f"           {r['fa']:<4} {r.get('ppl', {}).get('corpus_ppl', float('nan')):>9.3f}"
        if "topk" in metrics and "topk" in r:
            line += f"   {r['topk']['pct'][1]:.0f}%"
        print(line)
    if jpath:
        json.dump({"sweep": results}, open(jpath, "w"), indent=2)
        print(f"[fa-sweep] wrote {jpath}")


def main(argv=None):
    argv = argv if argv is not None else sys.argv
    ap = argparse.ArgumentParser(description="fidelity eval for the integer nmc engine vs Ollama")
    ap.add_argument("blob", help="path to the real GGUF blob (ollama-owned → usually needs sudo)")
    ap.add_argument("--metrics", default="ppl,topk,freerun", help="comma list of: ppl,topk,freerun")
    ap.add_argument("--prompts-file", help="override corpus (.json list, or one-per-line with \\n escapes)")
    ap.add_argument("--limit", type=int, help="use only the first N prompts")
    ap.add_argument("--n-free", type=int, default=12, help="tokens to generate for the freerun metric")
    ap.add_argument("--fa-sweep", help="comma list of fa values (re-execs per fa); implies ppl unless set")
    ap.add_argument("--model", default=MODEL, help="ollama model name")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/generate")
    ap.add_argument("--json", dest="jpath", help="write results as JSON to this path")
    args = ap.parse_args(argv[1:])

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    known = {"ppl", "topk", "freerun"}
    for m in metrics:
        if m not in known:
            print(f"[fidelity] warning: unknown metric {m!r} (known: {sorted(known)})", file=sys.stderr)
    prompts = load_prompts(args.prompts_file) if args.prompts_file else DEFAULT_PROMPTS
    if args.limit:
        prompts = prompts[:args.limit]

    if args.fa_sweep:
        fas = [int(x) for x in args.fa_sweep.split(",")]
        passthrough = ["--n-free", str(args.n_free), "--model", args.model, "--ollama-url", args.ollama_url]
        if args.prompts_file:
            passthrough += ["--prompts-file", args.prompts_file]
        if args.limit:
            passthrough += ["--limit", str(args.limit)]
        return sweep(os.path.abspath(argv[0]), args.blob, fas, metrics or ["ppl"], passthrough, args.jpath)

    out = run_one(args.blob, prompts, metrics, args.n_free, args.model, args.ollama_url)
    if args.jpath:
        json.dump(out, open(args.jpath, "w"), indent=2)
        print(f"[fidelity] wrote {args.jpath}")


if __name__ == "__main__":
    main()
