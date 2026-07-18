"""Reusable all-integer cohere2moe engine on real GGUF weights — prefill + KV-cached decode, kernel-backed.

One place for the real-model forward/decode so the CLIs (real_generate.py) and the fidelity eval don't each
re-implement it. Backend via NMC_BACKEND=cuda|cuda-resident|cpu (all byte-identical to the numpy oracle).
cuda-resident keeps weights in VRAM (register API); with the fused MoE kernel the expert FFN is one batched
GPU call per layer. RMSNorm/RoPE/softmax/SiLU reuse the vendored Bonsai integer primitives.
"""
import os
from pathlib import Path

import numpy as np

from nmc.gguf import GGUF, Q4_K, Q6_K
from nmc.tokenizer import Tokenizer
from nmc import cohere2 as c2
from nmc import qk_native, qk_cuda
from nmc import qk_codec as qkc

# Activation fixed-point bits (tunable for the fidelity sweep via NMC_FA); weights stay at fw=24. The kernel
# computes (x@Wᵀ)>>fw and is fa-agnostic, so raising fa just buys activation precision (no re-quantization).
FA, FW = int(os.environ.get("NMC_FA", "16")), 24
# DP4A pays only on large (compute-bound) matmuls; below this (out_f·rows) the limb-gen overhead regresses it.
DP4A_MIN_WORK = int(os.environ.get("NMC_DP4A_MIN_WORK", str(1 << 17)))
_DEFAULT_TOK = str(Path.home() / ".local/integer_inference_engine/north-mini-code/tokenizer")
_DEFAULT_CONTEXT_LENGTH = 8192
_MAX_CONTEXT_LENGTH = 1_048_576


def select_backend(want=None):
    """(kernel_module, resident?, name) for NMC_BACKEND. Falls back to CPU if CUDA is unusable."""
    want = want or os.environ.get("NMC_BACKEND", "cuda")
    if want.startswith("cuda") and qk_cuda.available():
        resident = (want == "cuda-resident") and qk_cuda.resident_available()
        return qk_cuda, resident, ("cuda-resident" if resident else "cuda")
    return qk_native, False, ("cpu" if want != "cuda" else "cpu (cuda unavailable)")


def greedy(logits_row, pos=None, hist=None):
    return int(np.asarray(logits_row).argmax())


class Engine:
    def __init__(self, blob, tok_dir=None, backend=None):
        self.g = GGUF(blob); kv = self.g.kv; A = "cohere2moe"
        self.cfg = c2.Cfg(
            d_model=kv[f"{A}.embedding_length"], n_heads=kv[f"{A}.attention.head_count"],
            n_kv=kv[f"{A}.attention.head_count_kv"], head_dim=kv[f"{A}.attention.key_length"],
            ffn=kv[f"{A}.feed_forward_length"], vocab=kv[f"{A}.vocab_size"],
            sliding_window=kv[f"{A}.attention.sliding_window"], n_experts=kv[f"{A}.expert_count"],
            n_used=kv[f"{A}.expert_used_count"], expert_ffn=kv[f"{A}.expert_feed_forward_length"],
            rope_base=float(kv[f"{A}.rope.freq_base"]), fa=FA, fw=FW)
        self.context_length = int(kv.get(f"{A}.context_length", _DEFAULT_CONTEXT_LENGTH))
        if not 1 <= self.context_length <= _MAX_CONTEXT_LENGTH:
            raise ValueError(
                f"{A}.context_length must be in [1, {_MAX_CONTEXT_LENGTH}], got {self.context_length}"
            )
        self.NL = kv[f"{A}.block_count"]; self.DENSE = kv[f"{A}.leading_dense_block_count"]
        self.kn, self.resident, self.bname = select_backend(backend)
        self.fused = self.resident and qk_cuda.moe_ffn_available()
        self._QT = {Q4_K: self.kn.Q4_K, Q6_K: self.kn.Q6_K}
        self.tok = Tokenizer.from_dir(tok_dir or os.environ.get("NMC_TOKENIZER", _DEFAULT_TOK))
        self.dp4a_min_work = DP4A_MIN_WORK          # size gate for DP4A (settable for A/B benchmarking)
        self.batch_moe = True                       # batch the prefill MoE over tokens (settable for A/B)
        self._h = {}; self._rw = {}; self._rchecked = False
        assert self.kn.available(), "qk kernel .so not found"

    # --- tokenizer passthrough ---
    def encode(self, s): return self.tok.encode(s)
    def decode(self, ids): return self.tok.decode(ids)
    def free(self):
        if self.resident:
            qk_cuda.free_all()
            self._h.clear()          # registry is cleared -> drop the stale handle cache so the next call
            self._rchecked = False   # re-registers (else reuse after free() hits freed handles -> None)

    # --- kernel linears (resident-aware) ---
    def _klin(self, name, x):
        t = self.g.tensors[name]; ne0, ne1, qt = t["shape"][0], t["shape"][1], self._QT[t["type"]]
        if self.resident:
            if name not in self._h:
                self._h[name] = (qk_cuda.register_weight(self.g.read_raw(name), ne1, ne0 // 256, qt), ne1)
            hh, of = self._h[name]
            # DP4A only when the matmul is large enough to be compute-bound (the tied head, large-m prefill);
            # for small per-token decode matmuls it's overhead-bound and regresses, so use the int128 kernel.
            rows = x.shape[0] if getattr(x, "ndim", 1) > 1 else 1
            if qk_cuda.dp4a_available() and qt in (qk_cuda.Q4_K, qk_cuda.Q6_K) and of * rows >= self.dp4a_min_work:
                r = qk_cuda.apply_resident_dp4a(hh, x, of, FW, qt)
                if r is not None:
                    return r
            return qk_cuda.apply_resident(hh, x, of, FW)
        return self.kn.qk_linear(self.g.read_raw(name), x, ne1, ne0 // 256, FW, qt)

    def _klin_expert(self, name, e, x):
        raw, ne0, ne1, tt = self.g.expert_raw(name, e); qt = self._QT[tt]
        if self.resident:
            key = (name, e)
            if key not in self._h:
                self._h[key] = (qk_cuda.register_weight(raw, ne1, ne0 // 256, qt), ne1)
            hh, of = self._h[key]; return qk_cuda.apply_resident(hh, x, of, FW)
        return self.kn.qk_linear(raw, x, ne1, ne0 // 256, FW, qt)

    def _ehandle(self, name, e):
        key = (name, e)
        if key not in self._h:
            raw, ne0, ne1, tt = self.g.expert_raw(name, e)
            self._h[key] = (qk_cuda.register_weight(raw, ne1, ne0 // 256, self._QT[tt]), ne1)
        return self._h[key][0]

    def _norm(self, x, gname):
        return c2.fixed_point_rmsnorm(x, FA, self.cfg.eps, gain_q=c2.to_fixed(self.g.weight(gname, None), FA))

    def _embed(self, ids):
        return np.stack([qkc.dequant_q6k_tensor(self.g.raw_rows("token_embd.weight", int(t), 1), self.cfg.d_model, FA)
                         for t in ids])

    def _router(self, h, p):
        if p not in self._rw:
            self._rw[p] = c2.to_fixed(self.g.weight(p + "ffn_gate_inp.weight", None), FW)
        W = self._rw[p]
        rl = ((np.asarray(h, np.int64) @ W.T) >> FW).astype(np.int64)   # int64 == big-int (d_model·2**40 < 2**63)
        if not self._rchecked:
            assert np.array_equal(rl, c2.linear(h, W, FW)), "router int64 != big-int (overflow)"
            self._rchecked = True
        return rl

    def _block(self, x_new, li, cache, cos, sin):
        cfg = self.cfg; p = f"blk.{li}."; window = c2.window_for_layer(cfg, li)
        start = cache.length(li); m = x_new.shape[0]
        h = self._norm(x_new, p + "attn_norm.weight")
        q = self._klin(p + "attn_q.weight", h).reshape(m, cfg.n_heads, cfg.head_dim)
        k = self._klin(p + "attn_k.weight", h).reshape(m, cfg.n_kv, cfg.head_dim)
        v = self._klin(p + "attn_v.weight", h).reshape(m, cfg.n_kv, cfg.head_dim)
        if window is not None or li < self.DENSE:   # cohere2 NoPE: RoPE only on SWA + dense-prefix layers
            q = c2._rope_int(q, cos, sin, FA, start); k = c2._rope_int(k, cos, sin, FA, start)
        cache.append(li, np.transpose(k, (1, 0, 2)), np.transpose(v, (1, 0, 2)))
        attn = self._klin(p + "attn_output.weight", c2.attention_cached(q, cache.k[li], cache.v[li], start, cfg, window))
        if li < self.DENSE:
            gg = c2.silu_int(self._klin(p + "ffn_gate.weight", h), FA); uu = self._klin(p + "ffn_up.weight", h)
            c2._assert_i64_contraction(c2._absmax_int(gg), c2._absmax_int(uu), 1, "dense FFN silu(gate)*up")
            gu = ((gg * uu) >> FA)                                 # int64 (gg·uu ≲ 2**40 fits) — lever 2
            ffn = self._klin(p + "ffn_down.weight", gu)
        else:
            rl = self._router(h, p)
            if self.fused and m > 1 and self.batch_moe and qk_cuda.moe_ffn_batched_available():
                # prefill: batch all m tokens' selected experts into ONE set of kernels (lever 1) instead of
                # m per-token qk_moe_ffn calls — collapses ~m·48 launches/forward and saturates the GPU.
                gh, uh, dh, gflat = [], [], [], []
                for t in range(m):
                    sel = c2._topk_lowidx(rl[t], cfg.n_used)
                    gflat += [int(x) for x in c2.fixed_point_sigmoid(rl[t][sel].astype(np.int64), FA)]
                    gh += [self._ehandle(p + "ffn_gate_exps.weight", e) for e in sel]
                    uh += [self._ehandle(p + "ffn_up_exps.weight", e) for e in sel]
                    dh += [self._ehandle(p + "ffn_down_exps.weight", e) for e in sel]
                ffn = qk_cuda.moe_ffn_batched(gh, uh, dh, m, cfg.n_used, h, gflat, cfg.d_model, cfg.expert_ffn, FA, FW)
            elif self.fused:
                ffn = np.empty((m, cfg.d_model), dtype=np.int64)
                for t in range(m):
                    sel = c2._topk_lowidx(rl[t], cfg.n_used)
                    gates = c2.fixed_point_sigmoid(rl[t][sel].astype(np.int64), FA)
                    gh = [self._ehandle(p + "ffn_gate_exps.weight", e) for e in sel]
                    uh = [self._ehandle(p + "ffn_up_exps.weight", e) for e in sel]
                    dh = [self._ehandle(p + "ffn_down_exps.weight", e) for e in sel]
                    ffn[t] = qk_cuda.moe_ffn(gh, uh, dh, h[t], gates, cfg.d_model, cfg.expert_ffn, FA, FW)
            else:
                ffn = np.zeros((m, cfg.d_model), dtype=object)
                for t in range(m):
                    sel = c2._topk_lowidx(rl[t], cfg.n_used)
                    gates = c2.fixed_point_sigmoid(rl[t][sel].astype(np.int64), FA)
                    ht = h[t:t + 1]
                    for jj, e in enumerate(sel):
                        gg = c2.silu_int(self._klin_expert(p + "ffn_gate_exps.weight", e, ht), FA)
                        uu = self._klin_expert(p + "ffn_up_exps.weight", e, ht)
                        gu = ((gg.astype(object) * uu.astype(object)) >> FA).astype(np.int64)
                        eo = self._klin_expert(p + "ffn_down_exps.weight", e, gu)[0].astype(object)
                        ffn[t] += (eo * int(gates[jj])) >> FA
                ffn = ffn.astype(np.int64)
        return np.asarray(x_new, np.int64) + attn + ffn

    def _rope(self, n):
        return c2.build_rope_tables(n, self.cfg.head_dim, base=int(self.cfg.rope_base), frac_bits=FA)

    def _require_context(self, rows: int) -> int:
        rows = int(rows)
        if rows < 1:
            raise ValueError("inference requires at least one input token")
        if rows > self.context_length:
            raise ValueError(
                f"requested {rows} RoPE rows exceeds the committed model context {self.context_length}"
            )
        return rows

    # --- public inference ---
    def logits_prefill(self, ids):
        """Prefill `ids`; return logits at every position [len(ids), vocab] (teacher-forced predictions)."""
        cos, sin = self._rope(self._require_context(len(ids)))
        cache = c2.KVCache(self.NL); x = self._embed(ids)
        for li in range(self.NL):
            x = self._block(x, li, cache, cos, sin)
        return self._klin("token_embd.weight", self._norm(x, "output_norm.weight"))

    def next_token(self, ids):
        return int(self.logits_prefill(ids)[-1].argmax())

    def generate(self, ids, n_new, pick=greedy, on_token=None, stop_eos=True):
        """Prefill `ids` then KV-cached decode up to n_new tokens. Returns the new token ids. on_token(tok) is
        called as each token is produced (for streaming); stop_eos ends early at the tokenizer's eos (so a
        natural answer finishes instead of padding to n_new — n_new is then a cap, not a forced length)."""
        n_new = int(n_new)
        if n_new < 0:
            raise ValueError(f"n_new must be non-negative, got {n_new}")
        # To emit N tokens, RoPE is consumed by the prompt and the first N-1 generated tokens; the final
        # sampled token is not fed back through a block. Build exactly the rows that can affect this run.
        rope_rows = self._require_context(len(ids) + max(n_new - 1, 0))
        cos, sin = self._rope(rope_rows)
        cache = c2.KVCache(self.NL)
        x = self._embed(ids)
        for li in range(self.NL):
            x = self._block(x, li, cache, cos, sin)
        last = x[-1:]
        eos = self.tok.eos_id if stop_eos else None
        out, seq = [], list(ids)
        for step in range(n_new):
            tok = int(pick(self._klin("token_embd.weight", self._norm(last, "output_norm.weight"))[0], len(seq), seq))
            out.append(tok); seq.append(tok)
            if on_token is not None:
                on_token(tok)
            if tok == eos or step + 1 >= n_new:
                break
            last = self._embed([tok])
            for li in range(self.NL):
                last = self._block(last, li, cache, cos, sin)
        return out
