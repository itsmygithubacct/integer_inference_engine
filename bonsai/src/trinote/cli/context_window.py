"""Resolve a useful context window from model, artifact, and host limits.

``context_length`` in a GGUF is a model capability, not always a sensible
interactive allocation.  Native artifacts add a stricter committed RoPE-table
limit, while llama.cpp can fit an otherwise-unset context to device memory.
This module keeps those values distinct so the CLI can report what it chose.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_GIB = 1 << 30


@dataclass(frozen=True)
class ContextWindow:
    architecture: str
    model_name: str
    source_max: int | None
    original_max: int | None
    artifact_max: int | None
    hard_max: int | None
    effective: int
    automatic: bool
    reason: str
    cache_bytes_per_token: int | None = None
    memory_max: int | None = None

    def input_budget(self, max_new: int) -> int:
        """Tokens available to the prompt after reserving the completion."""
        return max(0, int(self.effective) - max(0, int(max_new)))


def parse_context_size(value: object) -> int | None:
    """Normalize ``None``/``auto``/0 to auto and positive integers otherwise."""
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"", "auto", "model", "fit"}:
            return None
        try:
            value = int(raw, 10)
        except ValueError as exc:
            raise ValueError(f"context size must be 'auto' or an integer, got {value!r}") from exc
    size = int(value)
    if size == 0:
        return None
    if size < 0:
        raise ValueError("context size must be >= 0 (0 = auto)")
    return size


def _positive_int(value: object) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _meminfo() -> tuple[int | None, int | None]:
    """Return (MemAvailable, MemTotal) without an optional dependency."""
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            name, raw = line.split(":", 1)
            fields = raw.split()
            if fields:
                values[name] = int(fields[0]) * 1024
    except (OSError, ValueError):
        return None, None
    return values.get("MemAvailable"), values.get("MemTotal")


def native_cache_bytes_per_token(artifact: dict | None) -> int | None:
    """Estimate the native int64 attention K/V retained for one token."""
    if not artifact:
        return None
    cfg = artifact.get("config", {})
    hkv = _positive_int(cfg.get("n_heads_kv"))
    head_dim = _positive_int(cfg.get("head_dim"))
    layers = artifact.get("layers") or []
    if not hkv or not head_dim or not layers:
        return None
    attention_layers = sum(
        1 for layer in layers if str(layer.get("kind", "attention")) == "attention"
    )
    if attention_layers <= 0:
        return None
    # K and V, each int64, for every retained attention layer/head/channel.
    return attention_layers * hkv * head_dim * 2 * 8


def _memory_safe_tokens(
    bytes_per_token: int | None,
    *,
    available_bytes: int | None,
    total_bytes: int | None,
) -> int | None:
    if not bytes_per_token or not available_bytes:
        return None
    # Leave the larger of 4 GiB or 20% of RAM for weights, transient prefill,
    # receipt replay, and the rest of the host. Explicit user sizes may exceed
    # this recommendation; it is an auto-default guard, not a hard prohibition.
    reserve = max(4 * _GIB, (int(total_bytes) // 5) if total_bytes else 0)
    budget = max(0, int(available_bytes) - reserve)
    tokens = budget // int(bytes_per_token)
    if tokens >= 256:
        tokens = (tokens // 256) * 256
    return max(1, tokens)


def resolve_context_window(
    gguf_kv: dict,
    *,
    artifact: dict | None = None,
    backend: str = "native",
    requested: object = None,
    available_bytes: int | None = None,
    total_bytes: int | None = None,
) -> ContextWindow:
    """Resolve the effective window; ``effective=0`` means llama.cpp auto-fit."""
    architecture = str(gguf_kv.get("general.architecture", "") or "unknown")
    model_name = str(gguf_kv.get("general.name", "") or architecture)
    source_max = _positive_int(gguf_kv.get(f"{architecture}.context_length"))
    original_max = _positive_int(
        gguf_kv.get(f"{architecture}.rope.scaling.original_context_length")
    )

    artifact_max = None
    if artifact:
        cfg_max = _positive_int(artifact.get("config", {}).get("context_len"))
        rope = artifact.get("cos_fp")
        rope_max = _positive_int(getattr(rope, "shape", (None,))[0]) if rope is not None else None
        limits = [v for v in (cfg_max, rope_max) if v]
        artifact_max = min(limits) if limits else None

    limits = [v for v in (source_max, artifact_max) if v]
    hard_max = min(limits) if limits else None
    explicit = parse_context_size(requested)

    if explicit is not None:
        if hard_max is not None and explicit > hard_max:
            detail = f"model maximum {source_max}" if artifact_max is None else (
                f"native artifact maximum {artifact_max} (source model {source_max})"
            )
            raise ValueError(f"context size {explicit} exceeds {detail}")
        return ContextWindow(
            architecture=architecture,
            model_name=model_name,
            source_max=source_max,
            original_max=original_max,
            artifact_max=artifact_max,
            hard_max=hard_max,
            effective=explicit,
            automatic=False,
            reason="explicit override",
            cache_bytes_per_token=native_cache_bytes_per_token(artifact),
        )

    if backend != "native":
        # llama.cpp's -c 0 means model context; --fit then lowers unset values
        # to the available device memory. This is more accurate than duplicating
        # backend allocation rules here.
        return ContextWindow(
            architecture=architecture,
            model_name=model_name,
            source_max=source_max,
            original_max=original_max,
            artifact_max=None,
            hard_max=source_max,
            effective=0,
            automatic=True,
            reason="llama.cpp model context with hardware auto-fit",
        )

    if hard_max is None:
        raise ValueError("cannot resolve native context: model/artifact has no context metadata")
    if available_bytes is None and total_bytes is None:
        available_bytes, total_bytes = _meminfo()
    per_token = native_cache_bytes_per_token(artifact)
    memory_max = _memory_safe_tokens(
        per_token, available_bytes=available_bytes, total_bytes=total_bytes
    )
    # Prefer the model's unextended quality window when YaRN metadata exposes
    # one. The larger advertised window remains available as an explicit value.
    preferred = min(hard_max, original_max) if original_max else hard_max
    effective = min(preferred, memory_max) if memory_max else preferred
    effective = max(1, int(effective))
    reasons = []
    if artifact_max is not None and source_max is not None and artifact_max < source_max:
        reasons.append(f"artifact cap {artifact_max}")
    if original_max and original_max < hard_max:
        reasons.append(f"original RoPE window {original_max}")
    if memory_max and memory_max < preferred:
        reasons.append(f"host-memory recommendation {memory_max}")
    if not reasons:
        reasons.append("model/artifact maximum")
    return ContextWindow(
        architecture=architecture,
        model_name=model_name,
        source_max=source_max,
        original_max=original_max,
        artifact_max=artifact_max,
        hard_max=hard_max,
        effective=effective,
        automatic=True,
        reason=", ".join(reasons),
        cache_bytes_per_token=per_token,
        memory_max=memory_max,
    )
