"""Reusable native inference state for contextual Bonsai REPL sessions.

The receipt commits the fully rendered input IDs for every turn.  This cache is
only an execution optimization: when the next rendered input is not an exact
extension of the committed prefix (retry, clear, system change, or eviction),
it is discarded and rebuilt from those same IDs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..infer_int.prompt_cache_bonsai35 import (
    Bonsai35PromptState,
    build_prompt_state,
    generate_from_prompt_state,
)
from ..infer_int.reference_bonsai import _BonsaiKVCache, _gpu_enabled, _rmsnorm


TokenPicker = Callable[[np.ndarray, int, list[int]], int]
TokenCallback = Callable[[int], None]


def _is_prefix(prefix: tuple[int, ...], values: tuple[int, ...]) -> bool:
    return len(prefix) <= len(values) and values[:len(prefix)] == prefix


@dataclass(frozen=True)
class LiveGeneration:
    output_ids: list[int]
    reused_tokens: int
    gpu_fallback: bool = False


class LiveNativeSession:
    """Keep exact Qwen3/Qwen3.5 KV/recurrent state between REPL turns."""

    def __init__(self, model, *, architecture: str, artifact_digest: str,
                 gpu_executor=None) -> None:
        self.model = model
        self.architecture = str(architecture)
        self.artifact_digest = str(artifact_digest)
        self.gpu = gpu_executor if self.architecture == "qwen35" else None

        self._gpu_ids: tuple[int, ...] = ()
        self._gpu_logits: np.ndarray | None = None
        self._cpu_ids: tuple[int, ...] = ()
        self._cpu_logits: np.ndarray | None = None
        self._q35_state: Bonsai35PromptState | None = None
        self._q3_ids: tuple[int, ...] = ()
        self._q3_cache: _BonsaiKVCache | None = None
        self._q3_last_x: np.ndarray | None = None
        self._q3_logits: np.ndarray | None = None

    def invalidate(self) -> None:
        """Forget all speculative/live state after cancellation or a guard."""
        self._gpu_ids = ()
        self._gpu_logits = None
        if self.gpu is not None:
            try:
                self.gpu.reset()
            except Exception:
                pass
        self._cpu_ids = ()
        self._cpu_logits = None
        executor = getattr(self.model, "_model_executor", None)
        if executor is not None:
            try:
                executor.reset()
            except Exception:
                pass
        self._q35_state = None
        self._q3_ids = ()
        self._q3_cache = None
        self._q3_last_x = None
        self._q3_logits = None

    def generate(
        self,
        input_ids: list[int],
        n_new: int,
        pick: TokenPicker,
        *,
        eos: int | None = None,
        on_token: TokenCallback | None = None,
        on_gpu_fallback: Callable[[], None] | None = None,
        sampler_cfg=None,
    ) -> LiveGeneration:
        ids = tuple(int(v) for v in input_ids)
        if not ids or int(n_new) <= 0:
            return LiveGeneration([], 0)
        if self.architecture == "qwen35" and self.gpu is not None:
            device_sampler = (
                sampler_cfg is not None
                and hasattr(self.gpu, "sample_device")
                and hasattr(self.gpu, "decode_token_device")
                and (hasattr(self.gpu, "prefill_device") or hasattr(self.gpu, "_prefill_device"))
            )
            if device_sampler:
                output, reused, complete = self._generate_gpu_device(
                    ids, int(n_new), sampler_cfg, eos=eos, on_token=on_token
                )
            else:
                output, reused, complete = self._generate_gpu(
                    ids, int(n_new), pick, eos=eos, on_token=on_token
                )
            if complete:
                return LiveGeneration(output, reused)
            # A failed resident graph is poisoned. Replay the exact input on the
            # canonical CPU producer and suppress the already streamed prefix.
            self._gpu_ids = ()
            self._gpu_logits = None
            if on_gpu_fallback is not None:
                on_gpu_fallback()
            replay_index = 0

            def replay_token(token: int) -> None:
                nonlocal replay_index
                token = int(token)
                if replay_index < len(output):
                    if token != output[replay_index]:
                        raise RuntimeError(
                            "GPU/CPU replay diverged before fallback boundary; refusing output"
                        )
                elif on_token is not None:
                    on_token(token)
                replay_index += 1

            cpu, cpu_reused = self._generate_qwen35_cpu(
                ids, int(n_new), pick, eos=eos, on_token=replay_token
            )
            return LiveGeneration(cpu, cpu_reused, gpu_fallback=True)
        if self.architecture == "qwen35":
            output, reused = self._generate_qwen35_cpu(
                ids, int(n_new), pick, eos=eos, on_token=on_token
            )
            return LiveGeneration(output, reused)
        if self.architecture == "qwen3":
            output, reused = self._generate_qwen3(
                ids, int(n_new), pick, eos=eos, on_token=on_token
            )
            return LiveGeneration(output, reused)
        raise ValueError(f"unsupported live-session architecture {self.architecture!r}")

    def _prepare_gpu(self, ids: tuple[int, ...]) -> tuple[np.ndarray | None, int]:
        reusable = (
            bool(self._gpu_ids)
            and _is_prefix(self._gpu_ids, ids)
            and int(getattr(self.gpu, "position", -1)) == len(self._gpu_ids)
        )
        reused = len(self._gpu_ids) if reusable else 0
        if reusable:
            suffix = ids[len(self._gpu_ids):]
            if suffix:
                logits = self.gpu.prefill(suffix)
            else:
                logits = self._gpu_logits
            if logits is not None:
                self._gpu_ids = ids
                self._gpu_logits = np.asarray(logits)
                return self._gpu_logits, reused
        if not self.gpu.reset():
            return None, 0
        logits = self.gpu.prefill(ids)
        if logits is None:
            return None, 0
        self._gpu_ids = ids
        self._gpu_logits = np.asarray(logits)
        return self._gpu_logits, 0

    def _generate_gpu(self, ids: tuple[int, ...], n_new: int, pick: TokenPicker,
                      *, eos: int | None, on_token: TokenCallback | None):
        logits, reused = self._prepare_gpu(ids)
        if logits is None:
            return [], reused, False
        seq = list(ids)
        output: list[int] = []
        for step in range(n_new):
            token = int(pick(logits, len(seq), seq))
            seq.append(token)
            output.append(token)
            if on_token is not None:
                on_token(token)
            final = (eos is not None and token == int(eos)) or step + 1 == n_new
            next_logits = self.gpu.decode_token(token)
            if next_logits is None:
                self._gpu_ids = ()
                self._gpu_logits = None
                # The sampled final token does not need another model step, so
                # failure to retain it invalidates only the optimization.
                return output, reused, final
            self._gpu_ids = tuple(seq)
            self._gpu_logits = np.asarray(next_logits)
            logits = self._gpu_logits
            if final:
                break
        return output, reused, True

    def _prepare_gpu_device(self, ids: tuple[int, ...]) -> tuple[bool, int]:
        reusable = (
            bool(self._gpu_ids)
            and _is_prefix(self._gpu_ids, ids)
            and int(getattr(self.gpu, "position", -1)) == len(self._gpu_ids)
        )
        reused = len(self._gpu_ids) if reusable else 0
        if reusable:
            for token in ids[len(self._gpu_ids):]:
                if not self.gpu.decode_token_device(int(token)):
                    self._gpu_ids = ()
                    self._gpu_logits = None
                    return False, reused
            self._gpu_ids = ids
            self._gpu_logits = None
            return True, reused
        if not self.gpu.reset():
            return False, 0
        prefill = getattr(self.gpu, "prefill_device", None)
        if prefill is None:
            prefill = self.gpu._prefill_device
        if not prefill(ids):
            self._gpu_ids = ()
            return False, 0
        self._gpu_ids = ids
        self._gpu_logits = None
        return True, 0

    def _generate_gpu_device(self, ids: tuple[int, ...], n_new: int, sampler_cfg,
                             *, eos: int | None, on_token: TokenCallback | None):
        ready, reused = self._prepare_gpu_device(ids)
        if not ready:
            return [], reused, False
        seq = list(ids)
        output: list[int] = []
        for step in range(n_new):
            token = self.gpu.sample_device(sampler_cfg, seq, len(seq))
            if token is None:
                self._gpu_ids = ()
                return output, reused, False
            token = int(token)
            seq.append(token)
            output.append(token)
            if on_token is not None:
                on_token(token)
            final = (eos is not None and token == int(eos)) or step + 1 == n_new
            if not self.gpu.decode_token_device(token):
                self._gpu_ids = ()
                self._gpu_logits = None
                # The sampled final token is already exact; only future prefix
                # reuse is lost if consuming it fails after selection.
                return output, reused, final
            self._gpu_ids = tuple(seq)
            self._gpu_logits = None
            if final:
                break
        return output, reused, True

    def _native_executor_is_current(self, executor) -> bool:
        history = getattr(executor, "_history", None)
        if history is not None:
            return tuple(int(v) for v in history) == self._cpu_ids
        try:
            return int(executor.position()) == len(self._cpu_ids)
        except Exception:
            return False

    def _prepare_native_executor(self, executor, ids: tuple[int, ...]):
        reusable = (
            bool(self._cpu_ids)
            and self._native_executor_is_current(executor)
            and _is_prefix(self._cpu_ids, ids)
        )
        reused = len(self._cpu_ids) if reusable else 0
        if reusable:
            suffix = ids[len(self._cpu_ids):]
            if suffix:
                for token in suffix[:-1]:
                    executor.decode(int(token))
                row = executor.decode_logits(int(suffix[-1]))[0]
                self._cpu_ids = ids
                self._cpu_logits = np.asarray(row)
                return self._cpu_logits, reused
            if self._cpu_logits is not None:
                return self._cpu_logits, reused
        row = executor.prefill_logits(ids)[0]
        self._cpu_ids = ids
        self._cpu_logits = np.asarray(row)
        return self._cpu_logits, 0

    def _generate_native_executor(self, executor, ids: tuple[int, ...], n_new: int,
                                  pick: TokenPicker, *, eos: int | None,
                                  on_token: TokenCallback | None):
        row, reused = self._prepare_native_executor(executor, ids)
        seq = list(ids)
        output: list[int] = []
        for step in range(n_new):
            token = int(pick(row, len(seq), seq))
            seq.append(token)
            output.append(token)
            if on_token is not None:
                on_token(token)
            final = (eos is not None and token == int(eos)) or step + 1 == n_new
            if final:
                # Consume the last token without paying for an otherwise unused
                # vocabulary projection. The next user segment supplies a
                # suffix whose final token recreates the needed logits.
                try:
                    executor.decode(token)
                    self._cpu_ids = tuple(seq)
                except Exception:
                    self._cpu_ids = ()
                    self._cpu_logits = None
                    try:
                        executor.reset()
                    except Exception:
                        pass
                self._cpu_logits = None
                break
            row = executor.decode_logits(token)[0]
            self._cpu_ids = tuple(seq)
            self._cpu_logits = np.asarray(row)
        return output, reused

    def _prepare_q35_state(self, ids: tuple[int, ...]):
        state = self._q35_state
        if state is not None and _is_prefix(state.input_ids, ids):
            reused = len(state.input_ids)
            suffix = ids[reused:]
            if suffix:
                x = self.model._run_layers(suffix, state.cache)
                state.last_x = np.ascontiguousarray(x[-1:], dtype=np.int64)
                state.input_ids = ids
            return state, reused
        state = build_prompt_state(self.model, ids, self.artifact_digest)
        self._q35_state = state
        return state, 0

    def _generate_qwen35_cpu(self, ids: tuple[int, ...], n_new: int, pick: TokenPicker,
                             *, eos: int | None, on_token: TokenCallback | None):
        executor = getattr(self.model, "_model_executor", None)
        if executor is not None:
            return self._generate_native_executor(
                executor, ids, n_new, pick, eos=eos, on_token=on_token
            )
        state, reused = self._prepare_q35_state(ids)
        output = generate_from_prompt_state(
            self.model, state, n_new, pick, eos=eos,
            on_token=on_token, keep_reusable=True,
        )
        return output, reused

    def _q3_row(self, x: np.ndarray) -> np.ndarray:
        model = self.model
        frac = int(model.cfg["frac"])
        final = _rmsnorm(
            x[-1:], frac, model.artifact["final_norm_gain_fp"],
            native=bool(getattr(model, "_native", False)),
        )
        return np.asarray(model._output_linear(
            final, frac, fast=bool(getattr(model, "_fast", False))
        )[0])

    def _prepare_qwen3(self, ids: tuple[int, ...]):
        reusable = self._q3_cache is not None and _is_prefix(self._q3_ids, ids)
        reused = len(self._q3_ids) if reusable else 0
        if reusable:
            suffix = ids[len(self._q3_ids):]
            if suffix:
                self._q3_last_x = self.model._run_layers(suffix, self._q3_cache)
                self._q3_logits = None
            self._q3_ids = ids
            return reused
        self._q3_cache = _BonsaiKVCache(len(self.model.artifact["layers"]))
        self._q3_logits = None
        # Preserve the existing byte-identical GPU prefill optimization when
        # available, then continue incremental decode from its exported cache.
        if bool(getattr(self.model, "_native", False)) and _gpu_enabled():
            try:
                seeded = self.model._gpu_prefill(ids, want_kv=True)
            except (MemoryError, RuntimeError, ValueError):
                seeded = None
            if seeded is not None:
                logits, self._q3_cache = seeded
                self._q3_logits = np.asarray(logits[0])
                self._q3_last_x = None
        if self._q3_logits is None:
            self._q3_last_x = self.model._run_layers(ids, self._q3_cache)
        self._q3_ids = ids
        return 0

    def _generate_qwen3(self, ids: tuple[int, ...], n_new: int, pick: TokenPicker,
                        *, eos: int | None, on_token: TokenCallback | None):
        reused = self._prepare_qwen3(ids)
        seq = list(ids)
        output: list[int] = []
        row = self._q3_logits
        for _step in range(n_new):
            if row is None:
                row = self._q3_row(self._q3_last_x)
            token = int(pick(row, len(seq), seq))
            seq.append(token)
            output.append(token)
            if on_token is not None:
                on_token(token)
            # Retain the sampled token so the next rendered turn appends only
            # its chat closure and new user segment.
            self._q3_last_x = self.model._run_layers([token], self._q3_cache)
            self._q3_ids = tuple(seq)
            self._q3_logits = None
            row = None
            if eos is not None and token == int(eos):
                break
        return output, reused
