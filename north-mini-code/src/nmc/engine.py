"""Reusable all-integer cohere2moe engine on real GGUF weights — prefill + KV-cached decode, kernel-backed.

One place for the real-model forward/decode so the CLIs (real_generate.py) and the fidelity eval don't each
re-implement it. Backend via NMC_BACKEND=cuda|cuda-resident|cpu (all byte-identical to the numpy oracle).
cuda-resident keeps weights in VRAM (register API); with the fused MoE kernel the expert FFN is one batched
GPU call per layer. After host/batched prefill, its request-scoped attention bank keeps decode Q/K/V, RoPE,
K/V, fixed-point attention scratch, and O projection on-device. An opt-in bounded preprocessing ABI also
executes exact RMSNorm, router/top-k, and sigmoid gates on-device while expert slices remain route-lazy.
An explicit opt-in token entry composes dense block 0 with the retained 48-layer MoE continuation, but
production generate stays on its proven path until full-model parity. Host fallback and every integer floor
remain the byte-exact oracle.
"""
import os
from pathlib import Path

import numpy as np

from nmc.gguf import GGUF, Q4_K, Q6_K
from nmc.tokenizer import Tokenizer
from nmc import cohere2 as c2
from nmc import qk_native, qk_cuda
from nmc import qk_codec as qkc
from nmc.profiling import InferenceProfiler

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
    def __init__(self, blob, tok_dir=None, backend=None, *, profile=None, group_projections=True,
                 resident_attention=True, resident_preprocess=None, resident_layer_executor=None):
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
        self.dp4a_batch_moe = True                  # guarded DP4A for compute-bound batched expert prefill
        self.group_projections = bool(group_projections)
        self.resident_attention = bool(resident_attention)
        if resident_preprocess is None:
            resident_preprocess = os.environ.get("NMC_RESIDENT_PREPROCESS", "").strip().lower() \
                not in ("", "0", "false", "no", "off")
        self.resident_preprocess = bool(resident_preprocess)
        if resident_layer_executor is None:
            resident_layer_executor = os.environ.get("NMC_RESIDENT_LAYER_EXECUTOR", "").strip().lower() \
                not in ("", "0", "false", "no", "off")
        # This flag gates only the explicit resident_decode_token() entry. The
        # production generate() path remains unchanged until real-model parity.
        self.resident_layer_executor = bool(resident_layer_executor)
        self._h = {}; self._ih = {}; self._rw = {}; self._rchecked = False
        if profile is None:
            profile = os.environ.get("NMC_PROFILE", "").strip().lower() not in ("", "0", "false", "no", "off")
        self._profile = InferenceProfiler(bool(profile))
        self._native_cold = None
        if self._profile.enabled and self.resident:
            qk_cuda.profile_reset(enabled=True)
        assert self.kn.available(), "qk kernel .so not found"

    # --- tokenizer passthrough ---
    def encode(self, s): return self.tok.encode(s)
    def decode(self, ids): return self.tok.decode(ids)
    def free(self):
        if self.resident:
            with self._profile.phase("python_native.free"):
                qk_cuda.free_all()
            self._h.clear()          # registry is cleared -> drop the stale handle cache so the next call
            self._ih.clear()
            self._rchecked = False   # re-registers (else reuse after free() hits freed handles -> None)

    def reset_profile(self):
        """Start a new cold/warm measurement window without changing inference state."""
        self._profile.reset()
        self._native_cold = None
        if self.resident:
            qk_cuda.profile_reset(enabled=self._profile.enabled)

    def profile_snapshot(self):
        native = None
        if self.resident and self._profile.enabled:
            total = qk_cuda.profile_snapshot()
            if total is not None:
                cold = self._native_cold or (total if self._profile.bucket == "cold" else {})
                warm = {key: total[key] - cold.get(key, 0) for key in total}
                native = {"total": total, "cold": dict(cold), "warm": warm}
        return self._profile.snapshot(native)

    def _mark_profile_warm(self):
        if self._profile.enabled and self.resident and self._profile.bucket == "cold":
            self._native_cold = qk_cuda.profile_snapshot()
        self._profile.mark_warm()

    def _native_call(self, operation, fn, *args, **kwargs):
        with self._profile.phase("python_native_crossing"):
            with self._profile.phase(f"python_native.{operation}"):
                return fn(*args, **kwargs)

    @staticmethod
    def _cuda_result(result, operation):
        if result is None:
            raise RuntimeError(f"CUDA resident {operation} failed")
        return result

    @staticmethod
    def _projection_phase(name):
        suffixes = (
            ("attn_q.weight", "projection.q"), ("attn_k.weight", "projection.k"),
            ("attn_v.weight", "projection.v"), ("attn_output.weight", "projection.o"),
            ("ffn_gate", "projection.ffn_gate"), ("ffn_up", "projection.ffn_up"),
            ("ffn_down", "projection.ffn_down"),
        )
        for suffix, phase in suffixes:
            if suffix in name:
                return phase
        return "projection.output_head" if name == "token_embd.weight" else "projection.other"

    def _weight_handle(self, name):
        t = self.g.tensors[name]
        ne0, ne1, qt = t["shape"][0], t["shape"][1], self._QT[t["type"]]
        if name not in self._h:
            # Every first registration is cold discovery, even if a route first
            # reaches that expert during a later decode token.
            with self._profile.phase("registration.weight", bucket="cold"):
                handle = self._native_call(
                    "register_weight", qk_cuda.register_weight,
                    self.g.read_raw(name), ne1, ne0 // 256, qt,
                )
            if handle is None:
                raise RuntimeError("CUDA resident registration API became unavailable")
            self._h[name] = (handle, ne1)
        handle, out_f = self._h[name]
        return handle, out_f, qt

    # --- kernel linears (resident-aware) ---
    def _klin(self, name, x):
        phase = self._projection_phase(name)
        with self._profile.phase(phase):
            t = self.g.tensors[name]; ne0, ne1, qt = t["shape"][0], t["shape"][1], self._QT[t["type"]]
            if self.resident:
                hh, of, qt = self._weight_handle(name)
                # DP4A only when the matmul is large enough to be compute-bound (the tied head, large-m prefill);
                # for small per-token decode matmuls it's overhead-bound and regresses, so use the int128 kernel.
                rows = x.shape[0] if getattr(x, "ndim", 1) > 1 else 1
                if (qk_cuda.dp4a_available() and qt in (qk_cuda.Q4_K, qk_cuda.Q6_K)
                        and of * rows >= self.dp4a_min_work):
                    r = self._native_call("apply_resident_dp4a", qk_cuda.apply_resident_dp4a,
                                          hh, x, of, FW, qt)
                    if r is not None:
                        return r
                r = self._native_call("apply_resident", qk_cuda.apply_resident, hh, x, of, FW)
                return self._cuda_result(r, f"apply for {name}")
            return self._native_call(
                "qk_linear", self.kn.qk_linear, self.g.read_raw(name), x, ne1, ne0 // 256, FW, qt,
            )

    def _klin_grouped(self, names, x, phase_ids=None):
        if not (self.resident and self.group_projections and qk_cuda.grouped_available()):
            return tuple(self._klin(name, x) for name in names)
        handles, out_features = [], []
        for name in names:
            handle, out_f, _ = self._weight_handle(name)
            handles.append(handle); out_features.append(out_f)
            self._profile.record(self._projection_phase(name), calls=1)
        with self._profile.phase("projection.grouped"):
            result = self._native_call(
                "apply_resident_grouped", qk_cuda.apply_resident_grouped,
                handles, x, out_features, FW, phase_ids,
            )
        return self._cuda_result(result, f"grouped apply for {', '.join(names)}")

    def _klin_expert(self, name, e, x):
        raw, ne0, ne1, tt = self.g.expert_raw(name, e); qt = self._QT[tt]
        with self._profile.phase(self._projection_phase(name)):
            if self.resident:
                key = (name, e)
                if key not in self._h:
                    with self._profile.phase("registration.expert", bucket="cold"):
                        handle = self._native_call("register_expert", qk_cuda.register_weight,
                                                   raw, ne1, ne0 // 256, qt)
                    if handle is None:
                        raise RuntimeError("CUDA resident registration API became unavailable")
                    self._h[key] = (handle, ne1)
                hh, of = self._h[key]
                result = self._native_call("apply_expert", qk_cuda.apply_resident, hh, x, of, FW)
                return self._cuda_result(result, f"expert apply for {name}[{e}]")
            return self._native_call("qk_linear_expert", self.kn.qk_linear,
                                     raw, x, ne1, ne0 // 256, FW, qt)

    def _ehandle(self, name, e):
        key = (name, e)
        if key not in self._h:
            raw, ne0, ne1, tt = self.g.expert_raw(name, e)
            with self._profile.phase("registration.expert", bucket="cold"):
                handle = self._native_call("register_expert", qk_cuda.register_weight,
                                           raw, ne1, ne0 // 256, self._QT[tt])
            if handle is None:
                raise RuntimeError("CUDA resident registration API became unavailable")
            self._h[key] = (handle, ne1)
        return self._h[key][0]

    def _norm(self, x, gname):
        with self._profile.phase("normalization"):
            return c2.fixed_point_rmsnorm(x, FA, self.cfg.eps,
                                          gain_q=c2.to_fixed(self.g.weight(gname, None), FA))

    def _embed(self, ids):
        with self._profile.phase("embedding"):
            return np.stack([qkc.dequant_q6k_tensor(
                self.g.raw_rows("token_embd.weight", int(t), 1), self.cfg.d_model, FA,
            ) for t in ids])

    def _router(self, h, p):
        with self._profile.phase("routing"):
            if p not in self._rw:
                self._rw[p] = c2.to_fixed(self.g.weight(p + "ffn_gate_inp.weight", None), FW)
            W = self._rw[p]
            rl = ((np.asarray(h, np.int64) @ W.T) >> FW).astype(np.int64)   # int64 == big-int (d_model·2**40 < 2**63)
            if not self._rchecked:
                assert np.array_equal(rl, c2.linear(h, W, FW)), "router int64 != big-int (overflow)"
                self._rchecked = True
            return rl

    def _resident_preprocess_handles(self, p):
        """Register/reuse one layer's dense gain/router handles without running it."""
        if not (self.resident and qk_cuda.resident_preprocess_available()):
            raise qk_cuda.CudaContextError("resident preprocessing handle API is unavailable")
        gain_name = p + "attn_norm.weight"
        router_name = p + "ffn_gate_inp.weight"
        if gain_name not in self._ih:
            gain = c2.to_fixed(self.g.weight(gain_name, None), FA)
            with self._profile.phase("registration.preprocess_gain", bucket="cold"):
                handle = self._native_call("register_i64_gain", qk_cuda.register_i64, gain)
            if handle is None:
                raise RuntimeError("CUDA resident int64 registration API became unavailable")
            self._ih[gain_name] = handle
        if p not in self._rw:
            self._rw[p] = c2.to_fixed(self.g.weight(router_name, None), FW)
        if router_name not in self._ih:
            with self._profile.phase("registration.preprocess_router", bucket="cold"):
                handle = self._native_call("register_i64_router", qk_cuda.register_i64, self._rw[p])
            if handle is None:
                raise RuntimeError("CUDA resident int64 registration API became unavailable")
            self._ih[router_name] = handle
        return self._ih[gain_name], self._ih[router_name]

    def _resident_norm_router(self, x, p):
        """Run the exact bounded device preprocessor, or return None for host fallback."""
        if not (self.resident and self.resident_preprocess and qk_cuda.resident_preprocess_available()):
            return None
        gain_handle, router_handle = self._resident_preprocess_handles(p)
        with self._profile.phase("preprocess.resident"):
            result = self._native_call(
                "rmsnorm_router", qk_cuda.rmsnorm_router,
                gain_handle, router_handle, x, self.cfg.n_used, FA, FW, self.cfg.eps,
            )
        if result is None:
            self._profile.record("preprocess.resident_fallback")
            return None
        self._profile.record("normalization")
        self._profile.record("routing")
        return result

    @staticmethod
    def _resident_expert_tensor_names(prefix):
        return (
            prefix + "ffn_gate_exps.weight",
            prefix + "ffn_up_exps.weight",
            prefix + "ffn_down_exps.weight",
        )

    def _resident_known_expert(self, layer, expert):
        prefix = f"blk.{int(layer)}."
        entries = [self._h.get((name, int(expert))) for name in self._resident_expert_tensor_names(prefix)]
        if any(entry is None for entry in entries):
            return None
        return tuple(int(entry[0]) for entry in entries)

    def _resident_load_expert(self, layer, expert):
        prefix = f"blk.{int(layer)}."
        return tuple(self._ehandle(name, int(expert)) for name in self._resident_expert_tensor_names(prefix))

    def _resident_moe_layer_specs(self):
        specs = []
        for layer in range(self.DENSE, self.NL):
            prefix = f"blk.{layer}."
            gain_handle, router_handle = self._resident_preprocess_handles(prefix)
            projection_handles = [
                self._weight_handle(prefix + suffix)[0]
                for suffix in ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight")
            ]
            window = c2.window_for_layer(self.cfg, layer)
            specs.append(qk_cuda.ResidentMoeLayerSpec(
                layer, gain_handle, router_handle, *projection_handles,
                window, window is not None or layer < self.DENSE,
            ))
        return tuple(specs)

    def resident_decode_token(self, hidden, cache, cos, sin):
        """Explicit opt-in M=1 dense block plus retained 48-layer MoE decode.

        This method is intentionally not called by :meth:`generate`. It is a
        model-level integration surface for isolated full-model parity work;
        any failure after the dense block mutates K/V is fail-closed and the
        caller must destroy the request cache.
        """
        if not self.resident_layer_executor:
            raise RuntimeError("resident layer executor is disabled; opt in explicitly")
        if not (self.resident and self.resident_attention and qk_cuda.resident_layer_available()):
            raise qk_cuda.CudaContextError("resident layer executor requires the current CUDA resident ABI")
        if not isinstance(cache, qk_cuda.ResidentAttentionCache):
            raise TypeError("resident_decode_token requires a ResidentAttentionCache")
        if self.DENSE != 1 or self.NL - self.DENSE != 48:
            raise RuntimeError(
                f"resident token orchestrator is committed to 1 dense + 48 MoE layers, got "
                f"{self.DENSE} dense + {self.NL - self.DENSE} MoE"
            )
        value = np.ascontiguousarray(np.asarray(hidden, dtype=np.int64))
        if value.shape != (1, self.cfg.d_model):
            raise ValueError(f"resident_decode_token expects hidden shape (1, {self.cfg.d_model})")

        specs = self._resident_moe_layer_specs()
        executor = cache._token_executor
        if executor is None:
            executor = qk_cuda.ResidentMoeTokenExecutor(
                cache, first_layer=self.DENSE, layer_count=self.NL - self.DENSE,
                n_experts=self.cfg.n_experts, n_used=self.cfg.n_used,
                d_model=self.cfg.d_model, expert_ffn=self.cfg.expert_ffn,
                fw=FW, eps=self.cfg.eps,
                lookup_expert=self._resident_known_expert,
                load_expert=self._resident_load_expert,
            )
            executor._engine_owner = self
            cache._token_executor = executor
        elif getattr(executor, "_engine_owner", None) is not self:
            raise qk_cuda.CudaContextError("resident request cache is already owned by another engine")
        # Preconfigure bounded pointer metadata and bind already-resident expert
        # slices before the leading dense block advances layer-0 K/V.
        executor.prepare(specs)
        dense_out = self._block(value, 0, cache, cos, sin)
        return executor.run(dense_out, specs)

    def _moe(self, h, p, li, routed=None):
        cfg = self.cfg; m = h.shape[0]
        if routed is None:
            rl = self._router(h, p)
            route_ids = route_gates = None
        else:
            route_ids = np.asarray(routed[0], dtype=np.int32)
            route_gates = np.asarray(routed[1], dtype=np.int64)
            expected = (m, cfg.n_used)
            if route_ids.shape != expected or route_gates.shape != expected or \
                    np.any(route_ids < 0) or np.any(route_ids >= cfg.n_experts):
                raise qk_cuda.CudaContextError("resident router returned invalid selected-expert metadata")
            rl = None
        with self._profile.phase("moe"):
            if self.fused and m > 1 and self.batch_moe and qk_cuda.moe_ffn_batched_available():
                # Flatten token/expert pairs once.  The guarded DP4A entry reuses each token's limb expansion;
                # if h or the device-computed gate×up leaves the four-limb envelope it executes the unchanged
                # int128 batched path instead of truncating or wrapping.
                gh, uh, dh, gflat = [], [], [], []
                for t in range(m):
                    if route_ids is None:
                        sel = c2._topk_lowidx(rl[t], cfg.n_used)
                        gates = c2.fixed_point_sigmoid(rl[t][sel].astype(np.int64), FA)
                    else:
                        sel, gates = route_ids[t], route_gates[t]
                    gflat += [int(x) for x in gates]
                    gh += [self._ehandle(p + "ffn_gate_exps.weight", e) for e in sel]
                    uh += [self._ehandle(p + "ffn_up_exps.weight", e) for e in sel]
                    dh += [self._ehandle(p + "ffn_down_exps.weight", e) for e in sel]
                use_dp4a = (self.dp4a_batch_moe and qk_cuda.moe_ffn_batched_dp4a_available()
                            and m * cfg.n_used * cfg.expert_ffn >= self.dp4a_min_work)
                ffn = self._native_call(
                    "moe_ffn_batched_dp4a" if use_dp4a else "moe_ffn_batched",
                    qk_cuda.moe_ffn_batched, gh, uh, dh, m, cfg.n_used, h, gflat,
                    cfg.d_model, cfg.expert_ffn, FA, FW, dp4a=use_dp4a,
                )
                return self._cuda_result(ffn, f"batched MoE for layer {li}")
            if self.fused:
                ffn = np.empty((m, cfg.d_model), dtype=np.int64)
                for t in range(m):
                    if route_ids is None:
                        sel = c2._topk_lowidx(rl[t], cfg.n_used)
                        gates = c2.fixed_point_sigmoid(rl[t][sel].astype(np.int64), FA)
                    else:
                        sel, gates = route_ids[t], route_gates[t]
                    gh = [self._ehandle(p + "ffn_gate_exps.weight", e) for e in sel]
                    uh = [self._ehandle(p + "ffn_up_exps.weight", e) for e in sel]
                    dh = [self._ehandle(p + "ffn_down_exps.weight", e) for e in sel]
                    result = self._native_call(
                        "moe_ffn", qk_cuda.moe_ffn, gh, uh, dh, h[t], gates,
                        cfg.d_model, cfg.expert_ffn, FA, FW,
                    )
                    ffn[t] = self._cuda_result(result, f"MoE for layer {li}, token {t}")
                return ffn
            ffn = np.zeros((m, cfg.d_model), dtype=object)
            for t in range(m):
                if route_ids is None:
                    sel = c2._topk_lowidx(rl[t], cfg.n_used)
                    gates = c2.fixed_point_sigmoid(rl[t][sel].astype(np.int64), FA)
                else:
                    sel, gates = route_ids[t], route_gates[t]
                ht = h[t:t + 1]
                for jj, e in enumerate(sel):
                    gg = c2.silu_int(self._klin_expert(p + "ffn_gate_exps.weight", e, ht), FA)
                    uu = self._klin_expert(p + "ffn_up_exps.weight", e, ht)
                    gu = ((gg.astype(object) * uu.astype(object)) >> FA).astype(np.int64)
                    eo = self._klin_expert(p + "ffn_down_exps.weight", e, gu)[0].astype(object)
                    ffn[t] += (eo * int(gates[jj])) >> FA
            return ffn.astype(np.int64)

    def _block(self, x_new, li, cache, cos, sin):
        with self._profile.phase("layer"):
            return self._block_inner(x_new, li, cache, cos, sin)

    def _block_inner(self, x_new, li, cache, cos, sin):
        cfg = self.cfg; p = f"blk.{li}."; window = c2.window_for_layer(cfg, li)
        start = cache.length(li); m = x_new.shape[0]
        routed = None
        preprocessed = self._resident_norm_router(x_new, p) if li >= self.DENSE else None
        if preprocessed is None:
            h = self._norm(x_new, p + "attn_norm.weight")
        else:
            h, route_ids, route_gates = preprocessed
            routed = (route_ids, route_gates)
        qname, kname, vname = p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"
        oname = p + "attn_output.weight"
        if isinstance(cache, qk_cuda.ResidentAttentionCache):
            qh, _, _ = self._weight_handle(qname); kh, _, _ = self._weight_handle(kname)
            vh, _, _ = self._weight_handle(vname); oh, _, _ = self._weight_handle(oname)
            for name in (qname, kname, vname, oname):
                self._profile.record(self._projection_phase(name), calls=1)
            with self._profile.phase("attention.resident"):
                attn = self._native_call(
                    "attention_bank_apply", cache.apply, li, qh, kh, vh, oh, h, FW, window,
                    window is not None or li < self.DENSE,
                )
        else:
            q, k, v = self._klin_grouped(
                (qname, kname, vname), h,
                (qk_cuda.PROFILE_Q, qk_cuda.PROFILE_K, qk_cuda.PROFILE_V),
            )
            q = q.reshape(m, cfg.n_heads, cfg.head_dim)
            k = k.reshape(m, cfg.n_kv, cfg.head_dim)
            v = v.reshape(m, cfg.n_kv, cfg.head_dim)
            if window is not None or li < self.DENSE:   # cohere2 NoPE: RoPE only on SWA + dense-prefix layers
                with self._profile.phase("position_encoding"):
                    q = c2._rope_int(q, cos, sin, FA, start); k = c2._rope_int(k, cos, sin, FA, start)
            with self._profile.phase("kv_cache"):
                cache.append(li, np.transpose(k, (1, 0, 2)), np.transpose(v, (1, 0, 2)))
            with self._profile.phase("attention"):
                attended = c2.attention_cached(q, cache.k[li], cache.v[li], start, cfg, window)
            attn = self._klin(oname, attended)
        if li < self.DENSE:
            gate, up = self._klin_grouped((p + "ffn_gate.weight", p + "ffn_up.weight"), h)
            with self._profile.phase("dense_ffn_elementwise"):
                gg = c2.silu_int(gate, FA)
                c2._assert_i64_contraction(c2._absmax_int(gg), c2._absmax_int(up), 1,
                                           "dense FFN silu(gate)*up")
                gu = ((gg * up) >> FA)                         # exact int64 within the checked envelope
            ffn = self._klin(p + "ffn_down.weight", gu)
        else:
            ffn = self._moe(h, p, li, routed=routed)
        with self._profile.phase("residual"):
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
        with self._profile.phase("rope_tables"):
            cos, sin = self._rope(self._require_context(len(ids)))
        cache = c2.KVCache(self.NL, max_length=len(ids)); x = self._embed(ids)
        try:
            for li in range(self.NL):
                x = self._block(x, li, cache, cos, sin)
        finally:
            # Batched prefill scratch scales with prompt_tokens * selected_experts
            # (about 2 GiB at the supported 8K shape).  Keep resident weights but
            # return that transient arena before the output head or another run.
            if self.resident:
                self._native_call("release_moe_workspace", qk_cuda.release_moe_workspace)
        result = self._klin("token_embd.weight", self._norm(x, "output_norm.weight"))
        self._mark_profile_warm()
        return result

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
        with self._profile.phase("rope_tables"):
            cos, sin = self._rope(rope_rows)
        # Clamp geometric growth to this request's already-validated RoPE span.
        # Near an 8K boundary an unconstrained 1.5x growth step could otherwise
        # reserve 12,138 rows for 8,192 logical positions in every layer.
        cache = c2.KVCache(self.NL, max_length=rope_rows)
        x = self._embed(ids)
        try:
            for li in range(self.NL):
                x = self._block(x, li, cache, cos, sin)
        finally:
            if self.resident:
                self._native_call("release_moe_workspace", qk_cuda.release_moe_workspace)
        decode_cache = cache
        device_cache = None
        if (self.resident and self.resident_attention and qk_cuda.resident_attention_available()
                and n_new > 1):
            try:
                with self._profile.phase("attention.resident_setup", bucket="cold"):
                    device_cache = qk_cuda.ResidentAttentionCache(
                        self.NL, rope_rows, self.cfg.d_model, self.cfg.n_heads, self.cfg.n_kv,
                        self.cfg.head_dim, FA, cos, sin,
                    )
                    for li in range(self.NL):
                        device_cache.import_layer(li, cache.k[li], cache.v[li])
                decode_cache = device_cache
            except qk_cuda.CudaContextError:
                # Setup is transactional with respect to the host cache: no
                # decode state has consumed the device bank yet, so a capacity
                # failure can retain the established exact host path.
                if device_cache is not None:
                    device_cache.close()
                device_cache = None
                self._profile.record("attention.resident_setup_fallback")
        self._mark_profile_warm()
        last = x[-1:]
        eos = self.tok.eos_id if stop_eos else None
        out, seq = [], list(ids)
        try:
            for step in range(n_new):
                tok = int(pick(self._klin("token_embd.weight", self._norm(last, "output_norm.weight"))[0], len(seq), seq))
                out.append(tok); seq.append(tok)
                if on_token is not None:
                    on_token(tok)
                if tok == eos or step + 1 >= n_new:
                    break
                last = self._embed([tok])
                for li in range(self.NL):
                    last = self._block(last, li, decode_cache, cos, sin)
            return out
        finally:
            if device_cache is not None:
                self._native_call("attention_bank_destroy", device_cache.close)
