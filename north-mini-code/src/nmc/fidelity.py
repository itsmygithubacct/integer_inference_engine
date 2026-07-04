"""Fidelity metrics for the all-integer engine vs the float model — pure, model-free functions plus thin
runners, kept OUT of the tool so the math is reusable and unit-testable without the 30.5B model.

Logit scale: a `linear` leaves activations at fixed-point `2**fa` (weights are shifted back out) and
cohere2moe's `logit_scale = 1`, so the engine's integer logits are at scale `2**fa`. `from_fixed(logits, fa)`
recovers true-logit (nat) units — hence softmax / perplexity below de-scale by `fa` and are **comparable
across an fa sweep** (the whole point of `NMC_FA`). These metrics are offline diagnostics, not part of the
deterministic receipt path, so plain float numpy here is fine.

Runners (`eval_*`) take an injected `engine` (and, where a float reference is needed, an `ollama_fn`) and
return plain dicts (JSON-ready) — so they drive the real model in the tool and a stub engine in the tests.
"""
from __future__ import annotations

import math

import numpy as np

# ---- pure math (unit-tested on synthetic logits) ---------------------------------------------------------


def from_fixed(logits, fa):
    """Fixed-point (`2**fa`) integer logits → float true-logit (nat) units."""
    return np.asarray(logits, dtype=np.float64) / float(1 << fa)


def log_softmax(logits):
    """Numerically-stable natural-log softmax over the last axis (float logits, nat units)."""
    a = np.asarray(logits, dtype=np.float64)
    shifted = a - a.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def token_nll(logits_row, target_id, *, fa=None):
    """Negative log-likelihood (nats) of `target_id` under one logits row. If `fa` is given the row is
    integer fixed-point at `2**fa` and is de-scaled to nat units first."""
    row = from_fixed(logits_row, fa) if fa is not None else np.asarray(logits_row, np.float64)
    return float(-log_softmax(row)[int(target_id)])


def sequence_nlls(all_logits, ids, *, fa=None):
    """Teacher-forced per-token NLLs (nats) for one sequence. Row `all_logits[i]` (the prediction *after*
    token i) scores the ACTUAL next token `ids[i+1]`, for i in 0..len(ids)-2 → returns `len(ids)-1` values."""
    ids = list(ids)
    all_logits = np.asarray(all_logits)
    if all_logits.shape[0] != len(ids):
        raise ValueError(f"all_logits rows {all_logits.shape[0]} != len(ids) {len(ids)}")
    return [token_nll(all_logits[i], ids[i + 1], fa=fa) for i in range(len(ids) - 1)]


def perplexity(nlls):
    """`exp(mean NLL)` over a flat list of per-token NLLs (nats). Empty → nan."""
    nlls = list(nlls)
    return float(math.exp(sum(nlls) / len(nlls))) if nlls else float("nan")


def topk_rank(logits_row, target_id, max_k):
    """1-based rank of `target_id` among the top `max_k` logits (highest first, ties → lower index, matching
    the engine's argmax); None if outside the top `max_k`."""
    order = np.argsort(-np.asarray(logits_row), kind="stable")[:max_k]   # stable → lower index wins a tie
    hit = np.nonzero(order == int(target_id))[0]
    return int(hit[0]) + 1 if hit.size else None


def topk_hits(rank, ks):
    """dict `k → (target within top-k)` from a rank (None = outside)."""
    return {int(k): (rank is not None and rank <= k) for k in ks}


def common_prefix_len(a, b):
    """Count of shared leading elements of two sequences (strings, id lists, …)."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# ---- runners (drive the real engine in the tool, a stub engine in tests) ---------------------------------


def eval_ppl(engine, prompts, fa):
    """Integer engine's teacher-forced self-perplexity over `prompts` (one prefill each → all-position
    logits). Corpus PPL is `exp(mean NLL)` pooled over ALL tokens (not a mean of per-prompt PPLs). No
    float reference needed; its drift across an fa sweep is the precision/fidelity signal."""
    rows, all_nlls = [], []
    for pr in prompts:
        ids = engine.encode(pr)
        if len(ids) < 2:
            continue
        nlls = sequence_nlls(engine.logits_prefill(ids), ids, fa=fa)
        engine.free()
        all_nlls.extend(nlls)
        rows.append({"prompt": pr, "n_tokens": len(ids), "ppl": perplexity(nlls)})
    return {"corpus_ppl": perplexity(all_nlls), "n_tokens": len(all_nlls), "per_prompt": rows}


def eval_topk(engine, prompts, ollama_fn, ks=(1, 3, 5, 10)):
    """Is the float model's greedy next token within OUR top-k logits? top-1 == exact next-token agreement;
    a jump top-1→top-5 means we rank the right token highly and only the argmax flips on near-ties. Compares
    DECODED strings (Ollama returns text, not ids) — robust to single-token re-encoding quirks."""
    maxk = max(ks)
    hits = {int(k): 0 for k in ks}
    rows = []
    for pr in prompts:
        ids = engine.encode(pr)
        lg = engine.logits_prefill(ids)[-1]
        engine.free()
        order = np.argsort(-np.asarray(lg), kind="stable")[:maxk]
        toks = [engine.decode([int(i)]) for i in order]
        theirs = ollama_fn(pr, 1)
        rank = toks.index(theirs) + 1 if theirs in toks else None
        for k in ks:
            hits[int(k)] += (rank is not None and rank <= k)
        rows.append({"prompt": pr, "ours_top1": toks[0], "ollama": theirs, "rank": rank})
    n = len(rows) or 1
    return {"ks": [int(k) for k in ks], "hits": hits,
            "pct": {int(k): 100.0 * hits[int(k)] / n for k in ks}, "n": len(rows), "per_prompt": rows}


def eval_freerun(engine, prompts, ollama_fn, n_free):
    """Greedy `n_free` tokens on both sides; count common leading CHARACTERS before divergence (char-level is
    robust to tokenization differences; greedy compounds one near-tie flip into a different continuation)."""
    rows, cps = [], []
    for pr in prompts:
        ours = engine.decode(engine.generate(ids := engine.encode(pr), n_free))
        engine.free()
        theirs = ollama_fn(pr, n_free)
        cp = common_prefix_len(ours, theirs)
        cps.append(cp)
        rows.append({"prompt": pr, "ours": ours, "ollama": theirs, "common_prefix_chars": cp})
    return {"mean_common_prefix_chars": float(np.mean(cps)) if cps else 0.0, "per_prompt": rows}
