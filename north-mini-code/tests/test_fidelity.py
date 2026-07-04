"""Fidelity-metric gates — pure math on synthetic logits + the eval runners driven by a stub engine, so the
whole `nmc.fidelity` surface is exercised offline (no 30.5B model, no Ollama)."""
import math

import numpy as np
import pytest

from nmc import fidelity as F


# ---- pure math -------------------------------------------------------------------------------------------
def test_log_softmax_normalized():
    lg = np.array([2.0, -1.0, 0.5, 3.0])
    p = np.exp(F.log_softmax(lg))
    assert p.sum() == pytest.approx(1.0)
    # two equal logits -> ln(0.5) each
    assert F.log_softmax(np.array([0.0, 0.0])).tolist() == pytest.approx([math.log(0.5)] * 2)


def test_perplexity_uniform_equals_vocab():
    V = 50
    # uniform logits -> every token nll = ln V -> ppl == V, for any target
    nlls = [F.token_nll(np.zeros(V), t) for t in (0, 7, V - 1)]
    assert all(n == pytest.approx(math.log(V)) for n in nlls)
    assert F.perplexity(nlls) == pytest.approx(V)


def test_perplexity_confident_near_one():
    V = 32
    row = np.zeros(V); row[3] = 20.0                     # very peaked
    assert F.perplexity([F.token_nll(row, 3)]) == pytest.approx(1.0, abs=1e-6)


def test_perplexity_empty_is_nan():
    assert math.isnan(F.perplexity([]))


def test_token_nll_fa_descale_matches_float():
    fa = 16
    float_logits = np.array([2.0, 0.0, -1.0, 0.5])
    int_logits = np.round(float_logits * (1 << fa)).astype(np.int64)   # fixed-point at 2**fa
    for t in range(4):
        assert F.token_nll(int_logits, t, fa=fa) == pytest.approx(F.token_nll(float_logits, t), abs=1e-9)


def test_sequence_nlls_teacher_forcing_and_length():
    fa = 16
    ids = [5, 9, 2, 7]                                   # predictions: 5->9, 9->2, 2->7  (3 nlls)
    V = 16
    rows = []
    for i in range(len(ids)):
        r = np.zeros(V)
        r[ids[(i + 1) % len(ids)]] = 12.0               # peak at the actual next token
        rows.append(np.round(r * (1 << fa)).astype(np.int64))
    nlls = F.sequence_nlls(rows, ids, fa=fa)
    assert len(nlls) == len(ids) - 1
    assert all(n < 1e-3 for n in nlls)                  # teacher-forced peak -> tiny nll


def test_sequence_nlls_shape_mismatch_raises():
    with pytest.raises(ValueError):
        F.sequence_nlls(np.zeros((3, 8)), [1, 2, 3, 4])   # 3 rows != 4 ids


def test_topk_rank_and_ties():
    lg = np.array([3.0, 1.0, 2.0])
    assert F.topk_rank(lg, 0, 3) == 1                    # value 3 is highest
    assert F.topk_rank(lg, 2, 3) == 2                    # value 2 second
    assert F.topk_rank(lg, 1, 1) is None                # value 1 outside top-1
    # tie -> lower index wins
    assert F.topk_rank(np.array([5.0, 5.0, 1.0]), 0, 3) == 1
    assert F.topk_rank(np.array([5.0, 5.0, 1.0]), 1, 3) == 2


def test_topk_hits():
    assert F.topk_hits(2, [1, 3, 5]) == {1: False, 3: True, 5: True}
    assert F.topk_hits(None, [1, 5]) == {1: False, 5: False}


def test_common_prefix_len():
    assert F.common_prefix_len("abcXYZ", "abcQRS") == 3
    assert F.common_prefix_len([1, 2, 3], [1, 2, 4]) == 2
    assert F.common_prefix_len("", "abc") == 0
    assert F.common_prefix_len("same", "same") == 4


# ---- runners on a stub engine (no real model) ------------------------------------------------------------
class StubEngine:
    """Deterministic char-level stand-in: token id = ord(c) % V; logits peak (nat 12) at the ACTUAL next
    token so teacher-forced ppl is ~1 and top-1 is the true next token."""
    V, fa, bname = 32, 16, "stub"

    def encode(self, s):
        return [ord(c) % self.V for c in s] or [0]

    def decode(self, ids):
        return "".join(chr(97 + (i % 26)) for i in ids)

    def logits_prefill(self, ids):
        out = np.zeros((len(ids), self.V), dtype=np.int64)
        for i in range(len(ids)):
            out[i, ids[(i + 1) % len(ids)]] = 12 * (1 << self.fa)     # fixed-point at 2**fa
        return out

    def generate(self, ids, n):
        return [(ids[-1] + k + 1) % self.V for k in range(n)]

    def free(self):
        pass


def test_eval_ppl_runner():
    eng = StubEngine()
    r = F.eval_ppl(eng, ["hello", "world"], eng.fa)
    assert r["n_tokens"] == (5 - 1) + (5 - 1)             # teacher-forced tokens per prompt
    assert len(r["per_prompt"]) == 2
    assert r["corpus_ppl"] < 1.1                          # peaked stub -> ppl ~ 1


def test_eval_topk_runner_hits_and_misses():
    eng = StubEngine()
    prompts = ["abc", "defg"]
    # oracle that always returns OUR top-1 -> 100% top-1
    top1 = lambda pr, n: eng.decode([int(np.argmax(eng.logits_prefill(eng.encode(pr))[-1]))])
    r = F.eval_topk(eng, prompts, top1)
    assert r["hits"][1] == len(prompts) and all(row["rank"] == 1 for row in r["per_prompt"])
    # oracle that never matches -> 0%
    r0 = F.eval_topk(eng, prompts, lambda pr, n: " never")
    assert r0["hits"][1] == 0 and all(row["rank"] is None for row in r0["per_prompt"])


def test_eval_freerun_runner():
    eng = StubEngine()
    captured = {}
    def oracle(pr, n):
        ours = eng.decode(eng.generate(eng.encode(pr), n))
        captured[pr] = ours
        return ours[:3] + " "                        # diverge after 3 shared chars
    r = F.eval_freerun(eng, ["hello"], oracle, 6)
    assert r["per_prompt"][0]["common_prefix_chars"] == 3
    assert r["mean_common_prefix_chars"] == pytest.approx(3.0)


# ---- tool: fa-sweep flag forwarding (regression) ---------------------------------------------------------
def test_fa_sweep_forwards_corpus_flags(monkeypatch, tmp_path):
    """`--fa-sweep` re-execs one child per fa; each child must inherit the corpus/limit/n-free flags (a custom
    --prompts-file was previously dropped, so children silently ran the DEFAULT corpus) and must not recurse."""
    import sys
    import json
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))          # make tools/ importable
    try:
        import tools.fidelity_eval as T
    except Exception:                                                        # pragma: no cover
        pytest.skip("tools.fidelity_eval not importable here")

    calls = []
    class CP:
        returncode = 0
    def fake_run(cmd, env=None):
        calls.append(cmd)
        ji = cmd.index("--json")
        json.dump({"fa": int(env["NMC_FA"]), "ppl": {"corpus_ppl": 1.0, "n_tokens": 1}}, open(cmd[ji + 1], "w"))
        return CP()
    monkeypatch.setattr(T.subprocess, "run", fake_run)

    cf = tmp_path / "corpus.json"; cf.write_text(json.dumps(["a", "b", "c"]))
    T.main(["fidelity_eval.py", "/fake/blob", "--fa-sweep", "12,16", "--metrics", "ppl",
            "--prompts-file", str(cf), "--limit", "2", "--n-free", "8"])
    assert len(calls) == 2                                                   # one child per fa
    for c in calls:
        assert "--prompts-file" in c and str(cf) in c                       # corpus forwarded
        assert "--limit" in c and "--n-free" in c                           # and limit / n-free
        assert "--fa-sweep" not in c and "/fake/blob" in c                  # no recursion, blob kept


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
