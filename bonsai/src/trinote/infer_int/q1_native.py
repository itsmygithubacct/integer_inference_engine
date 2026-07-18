"""Optional native packed-Q1_0 linear kernel for Bonsai."""
from __future__ import annotations

import ctypes
import os
import threading
from functools import lru_cache, wraps
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ..notary_paths import kernel_so


_ROOT = Path(__file__).resolve().parents[3]
_LIB = Path(kernel_so())   # prefers ~/.local/trinote/bin, falls back to <repo>/tools (back-compat)
_TLS = threading.local()
_DEFAULT_WORKSPACE_MAX_MB = 64
_B35_ABI_VERSION = 2
_INT64_MAX = (1 << 63) - 1
_ISA_LOCK = threading.RLock()


class _B35Q1Desc(ctypes.Structure):
    _fields_ = [
        ("bits", ctypes.c_void_p), ("scale", ctypes.c_void_p),
        ("out_features", ctypes.c_int64), ("n_blocks", ctypes.c_int64),
    ]


class _B35LayerDesc(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int64),
        ("n1_gain", ctypes.c_void_p), ("n2_gain", ctypes.c_void_p),
        ("w1", _B35Q1Desc), ("wu", _B35Q1Desc), ("w2", _B35Q1Desc),
        ("q_norm_gain", ctypes.c_void_p), ("k_norm_gain", ctypes.c_void_p),
        ("wqg", _B35Q1Desc), ("wk", _B35Q1Desc),
        ("wv", _B35Q1Desc), ("wo", _B35Q1Desc),
        ("wqkv", _B35Q1Desc), ("wz", _B35Q1Desc),
        ("walpha", _B35Q1Desc), ("wbeta", _B35Q1Desc),
        ("wout", _B35Q1Desc),
        ("conv_weight", ctypes.c_void_p), ("dt_bias", ctypes.c_void_p),
        ("ssm_a", ctypes.c_void_p), ("ssm_norm_gain", ctypes.c_void_p),
    ]


class _B35ModelDesc(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_int64) for name in (
            "n_layers", "context_len", "frac", "d_model", "d_ff", "vocab",
            "n_heads", "n_heads_kv", "head_dim", "rope_rot_dim",
            "ssm_state_size", "ssm_group_count", "ssm_inner_size",
            "ssm_value_heads", "ssm_conv_kernel", "ssm_state_frac",
            "rms_eps", "ssm_rms_eps", "attention_scale", "gdn_scale",
            "lut_step", "softplus_min", "softplus_max", "exp_min",
            "softplus_count", "exp_count", "isa_mode",
        )
    ] + [
        ("embed", _B35Q1Desc), ("output", _B35Q1Desc),
        ("final_norm_gain", ctypes.c_void_p),
        ("cos", ctypes.c_void_p), ("sin", ctypes.c_void_p),
        ("softplus_lut", ctypes.c_void_p), ("exp_lut", ctypes.c_void_p),
        ("layers", ctypes.c_void_p),
    ]


class _B35ExecStats(ctypes.Structure):
    _fields_ = [
        ("decode_calls", ctypes.c_uint64), ("prefill_calls", ctypes.c_uint64),
        ("team_entries", ctypes.c_uint64), ("q1_groups", ctypes.c_uint64),
        ("lut32_hits", ctypes.c_uint64), ("lut32_fallbacks", ctypes.c_uint64),
        ("lut64_groups", ctypes.c_uint64),
        ("layer_major_prefills", ctypes.c_uint64),
        ("layer_major_rows", ctypes.c_uint64),
        ("prefill_tiles_40", ctypes.c_uint64),
        ("prefill_tiles_48", ctypes.c_uint64),
        ("prefill_tiles_136", ctypes.c_uint64),
        ("fused_residual_rms_calls", ctypes.c_uint64),
        ("parallel_rms_calls", ctypes.c_uint64),
        ("last_team_size", ctypes.c_int64), ("selected_isa", ctypes.c_int64),
        ("selected_lut_bits", ctypes.c_int64), ("cache_width_bits", ctypes.c_int64),
        ("prefill_tile_40", ctypes.c_int64),
        ("prefill_tile_48", ctypes.c_int64),
        ("prefill_tile_136", ctypes.c_int64),
    ]


def _b35_int(cfg: dict, name: str) -> int:
    if name not in cfg or type(cfg[name]) is not int:
        raise ValueError(f"Bonsai-27B config {name} must be a Python int")
    value = cfg[name]
    if value < -(1 << 63) or value > _INT64_MAX:
        raise ValueError(f"Bonsai-27B config {name} is outside int64")
    return value


def _b35_array(owner: dict, name: str, dtype, shape: tuple[int, ...],
               owners: list[np.ndarray]) -> np.ndarray:
    value = owner.get(name)
    if not isinstance(value, np.ndarray):
        raise ValueError(f"Bonsai-27B tensor {name} must be an ndarray")
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        raise ValueError(
            f"Bonsai-27B tensor {name} must have dtype {expected_dtype}, got {value.dtype}"
        )
    if value.shape != shape:
        raise ValueError(
            f"Bonsai-27B tensor {name} has shape {value.shape}, expected {shape}"
        )
    if not value.flags.c_contiguous:
        raise ValueError(f"Bonsai-27B tensor {name} must be C-contiguous")
    owners.append(value)
    return value


def _b35_q1_arrays(owner: dict, name: str, out_features: int,
                   in_features: int, owners: list[np.ndarray]) -> None:
    if in_features <= 0 or in_features % 128:
        raise ValueError(f"Bonsai-27B {name} input width is not Q1_0 aligned")
    blocks = in_features // 128
    _b35_array(owner, f"{name}_bits", np.uint8,
               (out_features, blocks, 16), owners)
    _b35_array(owner, f"{name}_scale_fp", np.int32,
               (out_features, blocks), owners)


def _validate_b35_release_artifact(
    artifact: dict,
) -> tuple[dict, list | tuple, tuple[np.ndarray, ...]]:
    """Validate every release extent before constructing any ctypes pointer."""

    if type(artifact) is not dict:
        raise ValueError("Bonsai-27B release artifact must be a dict")
    cfg = artifact.get("config")
    layers = artifact.get("layers")
    if type(cfg) is not dict or not isinstance(layers, (list, tuple)):
        raise ValueError("Bonsai-27B artifact needs config and layers")
    if cfg.get("architecture") != "qwen35":
        raise ValueError("Bonsai-27B release executor requires architecture='qwen35'")

    fixed = {
        "nLayers": 64,
        "frac": 16,
        "dModel": 5120,
        "dFfn": 17408,
        "n_heads": 24,
        "n_heads_kv": 4,
        "head_dim": 256,
        "ropeRotDim": 64,
        "ssmStateSize": 128,
        "ssmGroupCount": 16,
        "ssmInnerSize": 6144,
        "ssmTimeStepRank": 48,
        "ssmConvKernel": 4,
        "ssmStateFrac": 30,
        "fullAttentionInterval": 4,
    }
    values: dict[str, int] = {}
    for name, expected in fixed.items():
        values[name] = _b35_int(cfg, name)
        if values[name] != expected:
            raise ValueError(
                f"Bonsai-27B release config {name}={values[name]}, expected {expected}"
            )
    for name in (
        "context_len", "vocab", "rmsEpsilonFp2", "ssmRmsEpsilonFp2",
        "attentionScaleFp", "gdnScaleFp", "lutStepFp",
        "softplusLutMinFp", "softplusLutMaxFp",
        "expNegLutMinFp", "expNegLutMaxFp",
    ):
        values[name] = _b35_int(cfg, name)
    if values["context_len"] <= 0 or values["vocab"] <= 0:
        raise ValueError("Bonsai-27B context_len and vocab must be positive")
    if values["rmsEpsilonFp2"] < 0 or values["ssmRmsEpsilonFp2"] < 0:
        raise ValueError("Bonsai-27B RMS epsilon values must be non-negative")
    if values["attentionScaleFp"] <= 0 or values["gdnScaleFp"] <= 0:
        raise ValueError("Bonsai-27B attention/GDN scales must be positive")
    step = values["lutStepFp"]
    if step <= 0:
        raise ValueError("Bonsai-27B LUT step must be positive")
    if len(layers) != values["nLayers"]:
        raise ValueError(
            f"Bonsai-27B layer count {len(layers)} != nLayers {values['nLayers']}"
        )

    owners: list[np.ndarray] = []
    d = values["dModel"]
    dff = values["dFfn"]
    vocab = values["vocab"]
    context = values["context_len"]
    heads = values["n_heads"]
    kv_heads = values["n_heads_kv"]
    head_dim = values["head_dim"]
    state_dim = values["ssmStateSize"]
    groups = values["ssmGroupCount"]
    inner = values["ssmInnerSize"]
    value_heads = values["ssmTimeStepRank"]
    conv_dim = 2 * groups * state_dim + inner
    attention_inner = heads * head_dim
    kv_width = kv_heads * head_dim

    _b35_q1_arrays(artifact, "embed", vocab, d, owners)
    _b35_q1_arrays(artifact, "output", vocab, d, owners)
    _b35_array(artifact, "final_norm_gain_fp", np.int64, (d,), owners)
    _b35_array(artifact, "cos_fp", np.int64,
               (context, values["ropeRotDim"] // 2), owners)
    _b35_array(artifact, "sin_fp", np.int64,
               (context, values["ropeRotDim"] // 2), owners)

    def lut_shape(min_name: str, max_name: str) -> tuple[int]:
        span = values[max_name] - values[min_name]
        if span < 0 or span % step:
            raise ValueError(f"Bonsai-27B {min_name}/{max_name} do not align to LUT step")
        return (span // step + 1,)

    _b35_array(artifact, "softplus_lut_fp", np.int64,
               lut_shape("softplusLutMinFp", "softplusLutMaxFp"), owners)
    _b35_array(artifact, "exp_neg_lut_fp", np.int64,
               lut_shape("expNegLutMinFp", "expNegLutMaxFp"), owners)

    for index, layer in enumerate(layers):
        if type(layer) is not dict:
            raise ValueError(f"Bonsai-27B layer {index} must be a dict")
        expected_kind = "attention" if (index + 1) % 4 == 0 else "recurrent"
        if layer.get("kind") != expected_kind:
            raise ValueError(
                f"Bonsai-27B layer {index} kind {layer.get('kind')!r}, "
                f"expected {expected_kind!r}"
            )
        _b35_array(layer, "n1_gain_fp", np.int64, (d,), owners)
        _b35_array(layer, "n2_gain_fp", np.int64, (d,), owners)
        _b35_q1_arrays(layer, "w1", dff, d, owners)
        _b35_q1_arrays(layer, "wu", dff, d, owners)
        _b35_q1_arrays(layer, "w2", d, dff, owners)
        if expected_kind == "recurrent":
            _b35_q1_arrays(layer, "wqkv", conv_dim, d, owners)
            _b35_q1_arrays(layer, "wz", inner, d, owners)
            _b35_q1_arrays(layer, "walpha", value_heads, d, owners)
            _b35_q1_arrays(layer, "wbeta", value_heads, d, owners)
            _b35_q1_arrays(layer, "wout", d, inner, owners)
            _b35_array(layer, "conv_weight_fp", np.int64,
                       (conv_dim, values["ssmConvKernel"]), owners)
            _b35_array(layer, "dt_bias_fp", np.int64, (value_heads,), owners)
            _b35_array(layer, "ssm_a_fp", np.int64, (value_heads,), owners)
            _b35_array(layer, "ssm_norm_gain_fp", np.int64, (state_dim,), owners)
        else:
            _b35_array(layer, "q_norm_gain_fp", np.int64, (head_dim,), owners)
            _b35_array(layer, "k_norm_gain_fp", np.int64, (head_dim,), owners)
            _b35_q1_arrays(layer, "wqg", 2 * attention_inner, d, owners)
            _b35_q1_arrays(layer, "wk", kv_width, d, owners)
            _b35_q1_arrays(layer, "wv", kv_width, d, owners)
            _b35_q1_arrays(layer, "wo", d, attention_inner, owners)
    # A tuple is intentionally retained by the executor.  Replacing entries
    # in the caller's artifact/layer dicts cannot release a descriptor owner.
    return cfg, layers, tuple(owners)


def _validate_b35_abi(lib) -> None:
    try:
        version = int(lib.bonsai35_model_abi_version())
        sizes = tuple(int(lib.bonsai35_model_abi_sizeof(kind)) for kind in range(4))
    except AttributeError as exc:
        raise RuntimeError("native Bonsai-27B model executor ABI handshake is missing") from exc
    expected = (
        ctypes.sizeof(_B35ModelDesc), ctypes.sizeof(_B35LayerDesc),
        ctypes.sizeof(_B35Q1Desc), ctypes.sizeof(_B35ExecStats),
    )
    if version != _B35_ABI_VERSION or sizes != expected:
        raise RuntimeError(
            f"native Bonsai-27B ABI mismatch: version/sizes {(version, sizes)!r}, "
            f"expected {(_B35_ABI_VERSION, expected)!r}"
        )


def _validate_b35_token_ids(token_ids, vocab: int, *, where: str) -> np.ndarray:
    try:
        values = list(token_ids)
    except TypeError as exc:
        raise TypeError(f"{where} token IDs must be an iterable of Python ints") from exc
    if not values:
        raise ValueError(f"{where} requires at least one token ID")
    for index, value in enumerate(values):
        if type(value) is not int:
            raise TypeError(f"{where} token {index} must be a Python int")
        if value < 0 or value > _INT64_MAX or value >= vocab:
            raise ValueError(f"{where} token {index}={value} is outside [0, {vocab})")
    return np.ascontiguousarray(np.asarray(values, dtype=np.int64))


def _validate_b35_token_id(token_id, vocab: int, *, where: str) -> int:
    if type(token_id) is not int:
        raise TypeError(f"{where} token must be a Python int")
    if token_id < 0 or token_id > _INT64_MAX or token_id >= vocab:
        raise ValueError(f"{where} token {token_id} is outside [0, {vocab})")
    return token_id


def _b35_locked(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            self._require_open()
            return method(self, *args, **kwargs)
    return locked


@lru_cache(maxsize=1)
def _load_lib():
    if not _LIB.exists():
        return None
    # A stale/partial/wrong-arch .so should degrade to "native unavailable" (oracle fallback), not crash the
    # whole engine. Catch the load error AND a missing core symbol and return None — @lru_cache memoizes the
    # None, so q1_native_available() stays cheap and honest instead of raising on every call.
    try:
        lib = ctypes.CDLL(str(_LIB))
    except OSError:
        return None
    try:
        fn = lib.bonsai_q1_linear_i64
    except AttributeError:
        return None
    fn.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    try:
        fnw = lib.bonsai_q1_linear_i64_workspace
    except AttributeError:
        pass
    else:
        fnw.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ]
        fnw.restype = ctypes.c_int
    try:
        fna = lib.bonsai_q1_argmax_i64_workspace
    except AttributeError:
        pass
    else:
        fna.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ]
        fna.restype = ctypes.c_int
    try:
        fnp = lib.bonsai_q1_prepare_i64
    except AttributeError:
        pass
    else:
        fnp.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ]
        fnp.restype = ctypes.c_int
    try:
        fnlp = lib.bonsai_q1_linear_i64_prepared
    except AttributeError:
        pass
    else:
        fnlp.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ]
        fnlp.restype = ctypes.c_int
    try:
        fnmp = lib.bonsai_q1_linear_i64_prepared_multi
    except AttributeError:
        pass
    else:
        fnmp.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ]
        fnmp.restype = ctypes.c_int
    try:
        fnr = lib.bonsai_rmsnorm_i64
    except AttributeError:
        pass
    else:
        fnr.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        fnr.restype = ctypes.c_int
    # Optional narrow int32 scale-cache variants (Recommendation 7). Same ctypes layout as their int64
    # twins — only the C-level scale element type differs — so the int64 argtype lists are reused.
    _scale32_argtypes = {
        "bonsai_q1_linear_i64_workspace_scale32": [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ],
        "bonsai_q1_argmax_i64_workspace_scale32": [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ],
        "bonsai_q1_linear_i64_prepared_scale32": [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ],
        "bonsai_q1_linear_i64_prepared_multi_scale32": [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
        ],
    }
    for _sym, _argtypes in _scale32_argtypes.items():
        try:
            _fn = getattr(lib, _sym)
        except AttributeError:
            continue
        _fn.argtypes = _argtypes
        _fn.restype = ctypes.c_int
    # int32 activation-LUT-entry variants (same ctypes layout as the uint64 twins — the LUT pointer is a
    # void_p either way; only the C element type differs).
    _lut32_argtypes = {
        "bonsai_q1_prepare_i64_lut32": [
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ],
        "bonsai_q1_linear_i64_workspace_lut32": [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ],
        "bonsai_q1_linear_i64_prepared_lut32": [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ],
        "bonsai_q1_linear_i64_prepared_multi_lut32": [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ],
        "bonsai_q1_argmax_i64_workspace_lut32": [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ],
        "bonsai_q1_argmax_i64_workspace_lut32_scale32": [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ],
    }
    for _sym, _argtypes in _lut32_argtypes.items():
        try:
            _fn = getattr(lib, _sym)
        except AttributeError:
            continue
        _fn.argtypes = _argtypes
        _fn.restype = ctypes.c_int
    # Fused prepare + apply-many entry points.  All four symbols share one ABI;
    # the suffixes select the scale and activation-LUT storage widths in C.
    _prepare_apply_multi_argtypes = [
        ctypes.c_void_p,                                      # x
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,    # bits/scales/out-features lists
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_void_p,                                      # output pointer list
        ctypes.c_void_p, ctypes.c_size_t,                     # totals/count
        ctypes.c_void_p, ctypes.c_size_t,                     # LUT/count
    ]
    for _sym in (
        "bonsai_q1_prepare_apply_multi_i64",
        "bonsai_q1_prepare_apply_multi_i64_scale32",
        "bonsai_q1_prepare_apply_multi_i64_lut32",
        "bonsai_q1_prepare_apply_multi_i64_lut32_scale32",
    ):
        try:
            _fn = getattr(lib, _sym)
        except AttributeError:
            continue
        _fn.argtypes = _prepare_apply_multi_argtypes
        _fn.restype = ctypes.c_int
    try:
        fnisa = lib.bonsai_q1_runtime_has_avx2
    except AttributeError:
        pass
    else:
        fnisa.argtypes = []
        fnisa.restype = ctypes.c_int
        try:
            lib.bonsai_q1_set_isa_mode.argtypes = [ctypes.c_int]
            lib.bonsai_q1_set_isa_mode.restype = ctypes.c_int
            lib.bonsai_q1_get_isa_mode.argtypes = []
            lib.bonsai_q1_get_isa_mode.restype = ctypes.c_int
        except AttributeError:
            pass
    try:
        fnsilu = lib.bonsai_silu_i64
    except AttributeError:
        pass
    else:
        fnsilu.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        fnsilu.restype = ctypes.c_int
    try:
        fngdn = lib.bonsai_gdn_decode_i64
    except AttributeError:
        pass
    else:
        fngdn.argtypes = [
            ctypes.c_void_p,                         # state (mutated)
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # q/k/v
            ctypes.c_void_p, ctypes.c_void_p,        # beta/decay
            ctypes.c_int64, ctypes.c_int64,          # heads/dim
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_void_p,         # inv sqrt/out
        ]
        fngdn.restype = ctypes.c_int
    try:
        fngdnp = lib.bonsai_gdn_prefill_i64
    except AttributeError:
        pass
    else:
        fngdnp.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,  # tokens/heads/dim
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_void_p,
        ]
        fngdnp.restype = ctypes.c_int
    try:
        fnattn = lib.bonsai_attention_decode_i64
    except AttributeError:
        pass
    else:
        fnattn.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
        ]
        fnattn.restype = ctypes.c_int
    try:
        fnattp = lib.bonsai_attention_prefill_i64
    except AttributeError:
        pass
    else:
        fnattp.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                # q, k, v
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,                   # H, Hkv, hd
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,                   # M, L, start
            ctypes.c_int64, ctypes.c_int64,                                   # frac, inv_sqrt_fp
            ctypes.c_void_p,                                                  # out
        ]
        fnattp.restype = ctypes.c_int
    try:
        fnattb = lib.bonsai_attention_decode_batched_i64
    except AttributeError:
        pass
    else:
        fnattb.argtypes = [
            ctypes.c_void_p,                                                  # q (B,H,hd)
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,                # k_ptrs[B], v_ptrs[B], lengths[B]
            ctypes.c_void_p, ctypes.c_void_p,                                 # k_kv_strides[B], v_kv_strides[B]
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,   # B, H, Hkv, hd
            ctypes.c_int64, ctypes.c_int64,                                   # frac, inv_sqrt_fp
            ctypes.c_void_p,                                                  # out (B,H,hd)
        ]
        fnattb.restype = ctypes.c_int
    try:
        fnmc = lib.bonsai35_model_create
    except AttributeError:
        pass
    else:
        fnmc.argtypes = [ctypes.POINTER(_B35ModelDesc), ctypes.POINTER(ctypes.c_void_p)]
        fnmc.restype = ctypes.c_int
        try:
            lib.bonsai35_model_abi_version.argtypes = []
            lib.bonsai35_model_abi_version.restype = ctypes.c_uint64
            lib.bonsai35_model_abi_sizeof.argtypes = [ctypes.c_int]
            lib.bonsai35_model_abi_sizeof.restype = ctypes.c_size_t
        except AttributeError:
            pass
        lib.bonsai35_model_free.argtypes = [ctypes.c_void_p]
        lib.bonsai35_model_free.restype = None
        lib.bonsai35_model_reset.argtypes = [ctypes.c_void_p]
        lib.bonsai35_model_reset.restype = ctypes.c_int
        lib.bonsai35_model_decode.argtypes = [
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
        ]
        lib.bonsai35_model_decode.restype = ctypes.c_int
        lib.bonsai35_model_prefill.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
        ]
        lib.bonsai35_model_prefill.restype = ctypes.c_int
        _b35_optional = {
            "bonsai35_model_decode_logits": [
                ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
            ],
            "bonsai35_model_decode_argmax": [
                ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
            ],
            "bonsai35_model_prefill_logits": [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
            ],
            "bonsai35_model_prefill_argmax": [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
            ],
            "bonsai35_model_force_lut_fallback": [ctypes.c_void_p, ctypes.c_int],
            "bonsai35_model_debug_fail_after_mutation": [ctypes.c_void_p, ctypes.c_int],
            "bonsai35_model_cache_fingerprints": [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
            ],
            "bonsai35_model_export_tensor": [
                ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_void_p, ctypes.c_int64,
            ],
            "bonsai35_model_debug_trace_layer": [
                ctypes.c_void_p, ctypes.c_int64,
            ],
            "bonsai35_model_export_trace": [
                ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64,
            ],
            "bonsai35_model_export_internal": [
                ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64,
            ],
        }
        for _name, _types in _b35_optional.items():
            try:
                _fn = getattr(lib, _name)
            except AttributeError:
                continue
            _fn.argtypes = _types
            _fn.restype = ctypes.c_int
        try:
            lib.bonsai35_model_position.argtypes = [ctypes.c_void_p]
            lib.bonsai35_model_position.restype = ctypes.c_int64
            lib.bonsai35_model_debug_trace_rows.argtypes = [ctypes.c_void_p]
            lib.bonsai35_model_debug_trace_rows.restype = ctypes.c_int64
            lib.bonsai35_model_debug_internal_count.argtypes = [
                ctypes.c_void_p, ctypes.c_int64,
            ]
            lib.bonsai35_model_debug_internal_count.restype = ctypes.c_int64
        except AttributeError:
            pass
        lib.bonsai35_model_get_stats.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_B35ExecStats),
        ]
        lib.bonsai35_model_get_stats.restype = ctypes.c_int
    return lib


def q1_native_available() -> bool:
    return _load_lib() is not None


def q1_set_isa(mode: str) -> str:
    """Force ``auto``, ``portable``, or ``avx2`` native dispatch.

    Forced AVX2 fails closed on unsupported hardware instead of executing an
    illegal instruction or silently selecting the portable path.
    """
    normalized = str(mode).strip().lower()
    values = {"auto": 0, "portable": 1, "scalar": 1, "avx2": 2}
    if normalized not in values:
        raise ValueError("Q1 ISA must be auto, portable, or avx2")
    with _ISA_LOCK:
        lib = _load_lib()
        if lib is None or not hasattr(lib, "bonsai_q1_set_isa_mode"):
            if normalized == "avx2":
                raise RuntimeError("native AVX2 dispatch control is unavailable")
            return "portable"
        rc = lib.bonsai_q1_set_isa_mode(ctypes.c_int(values[normalized]))
        if rc == 5:
            raise RuntimeError("AVX2 was forced but is unavailable on this CPU")
        if rc != 0:
            raise RuntimeError(f"bonsai_q1_set_isa_mode failed with code {rc}")
        selected = int(lib.bonsai_q1_get_isa_mode())
        return "avx2" if selected == 2 else "portable"


def q1_selected_isa() -> str:
    with _ISA_LOCK:
        lib = _load_lib()
        if lib is None or not hasattr(lib, "bonsai_q1_get_isa_mode"):
            return "portable"
        return "avx2" if int(lib.bonsai_q1_get_isa_mode()) == 2 else "portable"


def _workspace_max_bytes() -> int:
    try:
        mb = int(os.environ.get("TRINOTE_Q1_WORKSPACE_MAX_MB", str(_DEFAULT_WORKSPACE_MAX_MB)))
    except ValueError:
        mb = _DEFAULT_WORKSPACE_MAX_MB
    return max(0, mb) * 1024 * 1024


def _workspace_arrays(total_count: int, lut_count: int):
    need_bytes = (int(total_count) + int(lut_count)) * np.dtype(np.uint64).itemsize
    if need_bytes <= 0 or need_bytes > _workspace_max_bytes():
        return None
    totals, lut = getattr(_TLS, "q1_workspace", (None, None))
    if totals is None or totals.size < total_count:
        totals = np.empty(total_count, dtype=np.uint64)
    if lut is None or lut.size < lut_count:
        lut = np.empty(lut_count, dtype=np.uint64)
    _TLS.q1_workspace = (totals, lut)
    return totals, lut


def _workspace_arrays_lut32(total_count: int, lut_count: int):
    """Like _workspace_arrays but the activation LUT is int32 (half the bytes). Separate TLS slot so it does
    not alias the uint64 LUT workspace. Sized by the int32 LUT footprint against the same MB cap."""
    need_bytes = int(total_count) * np.dtype(np.uint64).itemsize + int(lut_count) * np.dtype(np.int32).itemsize
    if need_bytes <= 0 or need_bytes > _workspace_max_bytes():
        return None
    totals, lut = getattr(_TLS, "q1_workspace_lut32", (None, None))
    if totals is None or totals.size < total_count:
        totals = np.empty(total_count, dtype=np.uint64)
    if lut is None or lut.size < lut_count:
        lut = np.empty(lut_count, dtype=np.int32)
    _TLS.q1_workspace_lut32 = (totals, lut)
    return totals, lut


_DEFAULT_ATTN_SCRATCH_MAX_MB = 128


def _attn_scratch_max_bytes() -> int:
    try:
        mb = int(os.environ.get("TRINOTE_ATTN_SCRATCH_MAX_MB", str(_DEFAULT_ATTN_SCRATCH_MAX_MB)))
    except ValueError:
        mb = _DEFAULT_ATTN_SCRATCH_MAX_MB
    return max(0, mb) * 1024 * 1024


def _attn_scratch(count: int):
    """Thread-local per-head scores/probs scratch (H*L int64) for the native attention kernel."""
    need_bytes = int(count) * np.dtype(np.int64).itemsize
    if need_bytes <= 0 or need_bytes > _attn_scratch_max_bytes():
        return None
    s = getattr(_TLS, "attn_scratch", None)
    if s is None or s.size < count:
        s = np.empty(count, dtype=np.int64)
        _TLS.attn_scratch = s
    return s


class Q1Prepared:
    """Prepared activation LUT for applying multiple Q1_0 projections to the same input."""

    __slots__ = ("x", "tokens", "n_blocks", "totals", "lut")

    def __init__(self, x: np.ndarray, tokens: int, n_blocks: int,
                 totals: np.ndarray, lut: np.ndarray):
        self.x = x
        self.tokens = int(tokens)
        self.n_blocks = int(n_blocks)
        self.totals = totals
        self.lut = lut


class Q1WeightGroup:
    """Validated same-input Q1 descriptors plus reusable output storage.

    A group belongs to one model runtime.  It keeps the NumPy owners alive,
    validates layout once, and avoids rebuilding weight pointer arrays and
    allocating projection outputs on every decoded token.  Model runtimes are
    intentionally not concurrently re-entrant; callers needing concurrency
    create one runtime/model instance per sequence.
    """

    __slots__ = (
        "packed", "out_features", "n_blocks", "scale32", "bits_ptrs",
        "scale_ptrs", "_outs", "_capacity", "_out_ptrs",
    )

    def __init__(self, weights):
        packed = []
        out_features = []
        n_blocks = None
        scale32 = True
        for bits, scale_fp in weights:
            b, s, out_f, blocks = _contiguous_q1_weight(bits, scale_fp)
            if n_blocks is None:
                n_blocks = blocks
            elif blocks != n_blocks:
                raise ValueError(
                    f"Q1_0 group block mismatch: first weight has {n_blocks}, next has {blocks}"
                )
            packed.append((b, s))
            out_features.append(out_f)
            scale32 = scale32 and s.dtype == np.int32
        if not packed:
            raise ValueError("Q1_0 projection group cannot be empty")
        # Mixed scale widths cannot share a typed C pointer list.  Canonicalize
        # the uncommon mixed case once, never in the decode loop.
        if not scale32 and any(s.dtype == np.int32 for _b, s in packed):
            packed = [
                (b, s if s.dtype == np.int64 else np.ascontiguousarray(s, dtype=np.int64))
                for b, s in packed
            ]
        self.packed = tuple(packed)
        self.out_features = np.ascontiguousarray(np.asarray(out_features, dtype=np.int64))
        self.n_blocks = int(n_blocks)
        self.scale32 = bool(scale32)
        ptr_array = ctypes.c_void_p * len(self.packed)
        self.bits_ptrs = ptr_array(*(b.ctypes.data for b, _s in self.packed))
        self.scale_ptrs = ptr_array(*(s.ctypes.data for _b, s in self.packed))
        self._outs = ()
        self._capacity = 0
        self._out_ptrs = None

    def outputs(self, tokens: int) -> tuple[np.ndarray, ...]:
        tokens = int(tokens)
        if tokens < 0:
            raise ValueError("Q1_0 projection token count cannot be negative")
        if self._capacity < tokens:
            # Grow geometrically for prompt tiles.  Decode stays at capacity 1
            # after the first call and has no projection-output allocations.
            capacity = max(tokens, 1 if self._capacity == 0 else self._capacity * 2)
            self._outs = tuple(
                np.empty((capacity, int(out_f)), dtype=np.int64)
                for out_f in self.out_features
            )
            ptr_array = ctypes.c_void_p * len(self._outs)
            self._out_ptrs = ptr_array(*(out.ctypes.data for out in self._outs))
            self._capacity = capacity
        return tuple(out[:tokens] for out in self._outs)


def q1_weight_group(weights) -> Q1WeightGroup:
    """Validate and pin a same-input projection group for repeated execution."""
    return Q1WeightGroup(weights)


_Q1_STATS = {
    "fused_calls": 0,
    "lut32_hits": 0,
    "lut32_fallbacks": 0,
    "u64_calls": 0,
}


def q1_native_stats(*, reset: bool = False) -> dict[str, int]:
    """Snapshot lightweight fused-dispatch counters used by benchmarks/tests."""
    result = dict(_Q1_STATS)
    if reset:
        for key in _Q1_STATS:
            _Q1_STATS[key] = 0
    return result


def _b35_i64_ptr(value, name: str) -> int:
    a = np.asarray(value)
    if a.dtype != np.int64 or not a.flags.c_contiguous:
        raise ValueError(f"Bonsai-27B executor tensor {name} must be contiguous int64")
    return int(a.ctypes.data)


def _b35_q1(owner: dict, name: str) -> _B35Q1Desc:
    bits = np.asarray(owner[f"{name}_bits"])
    scale = np.asarray(owner[f"{name}_scale_fp"])
    if bits.dtype != np.uint8 or not bits.flags.c_contiguous:
        raise ValueError(f"Bonsai-27B executor {name} bits must be contiguous uint8")
    if scale.dtype != np.int32 or not scale.flags.c_contiguous or scale.ndim != 2:
        raise ValueError(f"Bonsai-27B executor {name} scales must be contiguous int32")
    out_features, n_blocks = map(int, scale.shape)
    if bits.shape != (out_features, n_blocks, 16):
        raise ValueError(f"Bonsai-27B executor malformed {name} Q1 layout")
    return _B35Q1Desc(bits.ctypes.data, scale.ctypes.data, out_features, n_blocks)


class Bonsai35NativeExecutor:
    """Resident release-shape Qwen3.5 executor: one ABI call/team per step."""

    def __init__(self, artifact: dict):
        lib = _load_lib()
        if lib is None or not hasattr(lib, "bonsai35_model_create"):
            raise RuntimeError("native Bonsai-27B model executor is unavailable")
        required = (
            "bonsai35_model_abi_version", "bonsai35_model_abi_sizeof",
            "bonsai35_model_prefill_logits", "bonsai35_model_decode_logits",
            "bonsai35_model_prefill_argmax", "bonsai35_model_decode_argmax",
            "bonsai35_model_force_lut_fallback",
            "bonsai35_model_cache_fingerprints", "bonsai35_model_export_tensor",
            "bonsai35_model_position", "bonsai35_model_debug_trace_layer",
            "bonsai35_model_debug_trace_rows", "bonsai35_model_export_trace",
            "bonsai35_model_debug_internal_count", "bonsai35_model_export_internal",
        )
        if any(not hasattr(lib, symbol) for symbol in required):
            raise RuntimeError("native Bonsai-27B model executor ABI is stale")
        cfg, layers, owners = _validate_b35_release_artifact(artifact)
        _validate_b35_abi(lib)
        selected_isa = q1_set_isa(os.environ.get("TRINOTE_Q1_ISA", "auto"))
        self._lock = threading.RLock()
        self._owners = owners
        layer_array = (_B35LayerDesc * len(layers))()
        empty = _B35Q1Desc()
        for i, layer in enumerate(layers):
            recurrent = layer["kind"] == "recurrent"
            common = dict(
                kind=0 if recurrent else 1,
                n1_gain=_b35_i64_ptr(layer["n1_gain_fp"], f"layer{i}.n1"),
                n2_gain=_b35_i64_ptr(layer["n2_gain_fp"], f"layer{i}.n2"),
                w1=_b35_q1(layer, "w1"), wu=_b35_q1(layer, "wu"),
                w2=_b35_q1(layer, "w2"),
            )
            if recurrent:
                layer_array[i] = _B35LayerDesc(
                    **common, q_norm_gain=None, k_norm_gain=None,
                    wqg=empty, wk=empty, wv=empty, wo=empty,
                    wqkv=_b35_q1(layer, "wqkv"), wz=_b35_q1(layer, "wz"),
                    walpha=_b35_q1(layer, "walpha"), wbeta=_b35_q1(layer, "wbeta"),
                    wout=_b35_q1(layer, "wout"),
                    conv_weight=_b35_i64_ptr(layer["conv_weight_fp"], f"layer{i}.conv"),
                    dt_bias=_b35_i64_ptr(layer["dt_bias_fp"], f"layer{i}.dt_bias"),
                    ssm_a=_b35_i64_ptr(layer["ssm_a_fp"], f"layer{i}.ssm_a"),
                    ssm_norm_gain=_b35_i64_ptr(
                        layer["ssm_norm_gain_fp"], f"layer{i}.ssm_norm"
                    ),
                )
            else:
                layer_array[i] = _B35LayerDesc(
                    **common,
                    q_norm_gain=_b35_i64_ptr(layer["q_norm_gain_fp"], f"layer{i}.q_norm"),
                    k_norm_gain=_b35_i64_ptr(layer["k_norm_gain_fp"], f"layer{i}.k_norm"),
                    wqg=_b35_q1(layer, "wqg"), wk=_b35_q1(layer, "wk"),
                    wv=_b35_q1(layer, "wv"), wo=_b35_q1(layer, "wo"),
                    wqkv=empty, wz=empty, walpha=empty, wbeta=empty, wout=empty,
                    conv_weight=None, dt_bias=None, ssm_a=None, ssm_norm_gain=None,
                )
        desc = _B35ModelDesc(
            n_layers=int(cfg["nLayers"]), context_len=int(cfg["context_len"]),
            frac=int(cfg["frac"]), d_model=int(cfg["dModel"]),
            d_ff=int(cfg["dFfn"]), vocab=int(cfg["vocab"]),
            n_heads=int(cfg["n_heads"]), n_heads_kv=int(cfg["n_heads_kv"]),
            head_dim=int(cfg["head_dim"]), rope_rot_dim=int(cfg["ropeRotDim"]),
            ssm_state_size=int(cfg["ssmStateSize"]),
            ssm_group_count=int(cfg["ssmGroupCount"]),
            ssm_inner_size=int(cfg["ssmInnerSize"]),
            ssm_value_heads=int(cfg["ssmTimeStepRank"]),
            ssm_conv_kernel=int(cfg["ssmConvKernel"]),
            ssm_state_frac=int(cfg["ssmStateFrac"]),
            rms_eps=int(cfg["rmsEpsilonFp2"]),
            ssm_rms_eps=int(cfg["ssmRmsEpsilonFp2"]),
            attention_scale=int(cfg["attentionScaleFp"]),
            gdn_scale=int(cfg["gdnScaleFp"]), lut_step=int(cfg["lutStepFp"]),
            softplus_min=int(cfg["softplusLutMinFp"]),
            softplus_max=int(cfg["softplusLutMaxFp"]),
            exp_min=int(cfg["expNegLutMinFp"]),
            softplus_count=int(np.asarray(artifact["softplus_lut_fp"]).size),
            exp_count=int(np.asarray(artifact["exp_neg_lut_fp"]).size),
            isa_mode=2 if selected_isa == "avx2" else 1,
            embed=_b35_q1(artifact, "embed"), output=_b35_q1(artifact, "output"),
            final_norm_gain=_b35_i64_ptr(artifact["final_norm_gain_fp"], "final_norm"),
            cos=_b35_i64_ptr(artifact["cos_fp"], "cos"),
            sin=_b35_i64_ptr(artifact["sin_fp"], "sin"),
            softplus_lut=_b35_i64_ptr(artifact["softplus_lut_fp"], "softplus_lut"),
            exp_lut=_b35_i64_ptr(artifact["exp_neg_lut_fp"], "exp_lut"),
            layers=ctypes.addressof(layer_array),
        )
        self._artifact = artifact
        self._config = MappingProxyType(dict(cfg))
        self._layer_kinds = tuple(str(layer["kind"]) for layer in layers)
        self._layers = layer_array
        self._desc = desc
        self._lib = lib
        self.d_model = int(cfg["dModel"])
        self.vocab = int(cfg["vocab"])
        self._force_lut_replay = False
        self._history: list[int] = []
        self._argmax_out = np.empty(1, dtype=np.int64)
        self._trace_layer: int | None = None
        self._selected_isa = selected_isa
        self._handle = self._new_handle()

    @classmethod
    def create(cls, artifact: dict):
        try:
            return cls(artifact)
        except (MemoryError, ValueError, RuntimeError):
            return None

    def _new_handle(self) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        rc = self._lib.bonsai35_model_create(ctypes.byref(self._desc), ctypes.byref(handle))
        if rc == 5:
            raise ValueError("artifact/config is unsupported by the release-shape executor")
        if rc == 2:
            raise MemoryError("cannot allocate resident Bonsai-27B executor caches")
        if rc != 0 or not handle.value:
            raise RuntimeError(f"bonsai35_model_create failed with code {rc}")
        return handle

    def _restore_committed_prefix(self, history: tuple[int, ...]) -> None:
        """Discard a poisoned handle and reconstruct only committed tokens."""
        old = self._handle
        if old is not None and old.value:
            self._lib.bonsai35_model_free(old)
            old.value = None
        self._handle = self._new_handle()
        if self._force_lut_replay:
            rc = self._lib.bonsai35_model_force_lut_fallback(self._handle, ctypes.c_int(1))
            if rc != 0:
                raise RuntimeError(f"cannot restore LUT fallback mode (code {rc})")
        if self._trace_layer is not None:
            rc = self._lib.bonsai35_model_debug_trace_layer(
                self._handle, ctypes.c_int64(self._trace_layer)
            )
            if rc != 0:
                raise RuntimeError(f"cannot restore trace selection (code {rc})")
        if history:
            ids = np.ascontiguousarray(np.asarray(history, dtype=np.int64))
            # Replaying via the one-value argmax ABI avoids allocating a
            # context_len*d_model hidden matrix on this rare error path.
            rc = self._lib.bonsai35_model_prefill_argmax(
                self._handle, ids.ctypes.data, ctypes.c_int64(ids.size),
                self._argmax_out.ctypes.data,
            )
            if rc != 0:
                raise RuntimeError(f"committed-prefix replay failed with code {rc}")
        self._history = list(history)

    def _raise_transactional(self, operation: str, rc: int,
                             history: tuple[int, ...]) -> None:
        try:
            self._restore_committed_prefix(history)
        except Exception as recovery_error:
            raise RuntimeError(
                f"{operation} failed with code {rc}; committed-prefix recovery also failed"
            ) from recovery_error
        raise RuntimeError(
            f"{operation} failed with code {rc}; cache restored to committed prefix"
        )

    def _require_open(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is None or not handle.value:
            raise RuntimeError("Bonsai-27B native executor is closed")

    @_b35_locked
    def reset(self) -> None:
        previous = tuple(self._history)
        rc = self._lib.bonsai35_model_reset(self._handle)
        if rc != 0:
            self._raise_transactional("bonsai35_model_reset", rc, previous)
        self._history.clear()

    @_b35_locked
    def prefill(self, token_ids) -> np.ndarray:
        ids = _validate_b35_token_ids(
            token_ids, self.vocab, where="native Bonsai-27B prefill"
        )
        previous = tuple(self._history)
        out = np.empty((ids.size, self.d_model), dtype=np.int64)
        rc = self._lib.bonsai35_model_prefill(
            self._handle, ids.ctypes.data, ctypes.c_int64(ids.size), out.ctypes.data
        )
        if rc != 0:
            self._raise_transactional("bonsai35_model_prefill", rc, previous)
        self._history = [int(value) for value in ids]
        return out

    @_b35_locked
    def decode(self, token_id: int) -> np.ndarray:
        token = _validate_b35_token_id(
            token_id, self.vocab, where="native Bonsai-27B decode"
        )
        previous = tuple(self._history)
        out = np.empty((1, self.d_model), dtype=np.int64)
        rc = self._lib.bonsai35_model_decode(
            self._handle, ctypes.c_int64(token), out.ctypes.data
        )
        if rc != 0:
            self._raise_transactional("bonsai35_model_decode", rc, previous)
        self._history.append(token)
        return out

    @_b35_locked
    def prefill_logits(self, token_ids) -> np.ndarray:
        """Run the prompt and final norm/output projection in one team."""
        ids = _validate_b35_token_ids(
            token_ids, self.vocab, where="native Bonsai-27B prefill"
        )
        previous = tuple(self._history)
        out = np.empty((1, self.vocab), dtype=np.int64)
        rc = self._lib.bonsai35_model_prefill_logits(
            self._handle, ids.ctypes.data, ctypes.c_int64(ids.size), out.ctypes.data
        )
        if rc != 0:
            self._raise_transactional("bonsai35_model_prefill_logits", rc, previous)
        self._history = [int(value) for value in ids]
        return out

    @_b35_locked
    def decode_logits(self, token_id: int) -> np.ndarray:
        """Run one cached token through final logits in one ABI/team entry."""
        token = _validate_b35_token_id(
            token_id, self.vocab, where="native Bonsai-27B decode"
        )
        previous = tuple(self._history)
        out = np.empty((1, self.vocab), dtype=np.int64)
        rc = self._lib.bonsai35_model_decode_logits(
            self._handle, ctypes.c_int64(token), out.ctypes.data
        )
        if rc != 0:
            self._raise_transactional("bonsai35_model_decode_logits", rc, previous)
        self._history.append(token)
        return out

    @_b35_locked
    def prefill_argmax(self, token_ids) -> int:
        """Run the prompt through the exact greedy output in one ABI/team."""
        ids = _validate_b35_token_ids(
            token_ids, self.vocab, where="native Bonsai-27B prefill"
        )
        previous = tuple(self._history)
        out = self._argmax_out
        rc = self._lib.bonsai35_model_prefill_argmax(
            self._handle, ids.ctypes.data, ctypes.c_int64(ids.size), out.ctypes.data
        )
        if rc != 0:
            self._raise_transactional("bonsai35_model_prefill_argmax", rc, previous)
        self._history = [int(value) for value in ids]
        return int(out[0])

    @_b35_locked
    def decode_argmax(self, token_id: int) -> int:
        """Run one cached token through exact greedy argmax in one ABI/team."""
        token = _validate_b35_token_id(
            token_id, self.vocab, where="native Bonsai-27B decode"
        )
        previous = tuple(self._history)
        out = self._argmax_out
        rc = self._lib.bonsai35_model_decode_argmax(
            self._handle, ctypes.c_int64(token), out.ctypes.data
        )
        if rc != 0:
            self._raise_transactional("bonsai35_model_decode_argmax", rc, previous)
        self._history.append(token)
        return int(out[0])

    @_b35_locked
    def force_lut_fallback(self, enabled: bool) -> None:
        """Exercise the pre-output uint64 LUT replay path (tests/diagnostics)."""
        rc = self._lib.bonsai35_model_force_lut_fallback(
            self._handle, ctypes.c_int(1 if enabled else 0)
        )
        if rc != 0:
            raise RuntimeError(f"bonsai35_model_force_lut_fallback failed with code {rc}")
        self._force_lut_replay = bool(enabled)

    @_b35_locked
    def debug_fail_after_mutation(self, kind: str | None) -> None:
        """Inject a one-shot recurrent/attention post-mutation error for tests."""
        modes = {None: 0, "recurrent": 1, "attention": 2}
        if kind not in modes:
            raise ValueError("failure kind must be None, 'recurrent', or 'attention'")
        rc = self._lib.bonsai35_model_debug_fail_after_mutation(
            self._handle, ctypes.c_int(modes[kind])
        )
        if rc != 0:
            raise RuntimeError(f"bonsai35_model_debug_fail_after_mutation failed with code {rc}")

    @_b35_locked
    def debug_trace_layer(self, layer_index: int | None) -> None:
        """Select one layer whose exact boundaries and internals are retained.

        Tracing is disabled by default and allocates no trace arena. Passing
        ``None`` disables capture; a subsequent successful prefill/decode is
        required before :meth:`export_debug_trace`.
        """

        layer = -1 if layer_index is None else int(layer_index)
        if layer >= len(self._layer_kinds) or layer < -1:
            raise IndexError(f"Bonsai-27B trace layer is out of range: {layer}")
        rc = self._lib.bonsai35_model_debug_trace_layer(
            self._handle, ctypes.c_int64(layer)
        )
        if rc != 0:
            raise RuntimeError(f"bonsai35_model_debug_trace_layer failed with code {rc}")
        self._trace_layer = None if layer < 0 else layer

    @_b35_locked
    def export_debug_trace(self) -> dict[str, np.ndarray]:
        """Copy full boundaries plus bounded last-token layer internals."""

        if self._trace_layer is None:
            raise RuntimeError("Bonsai-27B debug trace is not enabled")
        rows = int(self._lib.bonsai35_model_debug_trace_rows(self._handle))
        if rows <= 0:
            raise RuntimeError("Bonsai-27B debug trace has no completed execution")
        names = ("n1", "branch", "residual", "n2", "ffn", "output")
        result: dict[str, np.ndarray] = {}
        for kind, name in enumerate(names):
            out = np.empty((rows, self.d_model), dtype=np.int64)
            rc = self._lib.bonsai35_model_export_trace(
                self._handle,
                ctypes.c_int64(kind),
                out.ctypes.data,
                ctypes.c_int64(out.size),
            )
            if rc != 0:
                raise RuntimeError(
                    f"bonsai35_model_export_trace({name}) failed with code {rc}"
                )
            result[name] = out

        cfg = self._config
        if self._layer_kinds[self._trace_layer] == "recurrent":
            heads = int(cfg["ssmTimeStepRank"])
            dim = int(cfg["ssmStateSize"])
            groups = int(cfg["ssmGroupCount"])
            inner = int(cfg["ssmInnerSize"])
            conv_dim = 2 * groups * dim + inner
            internals = (
                ("qkv", (1, conv_dim)),
                ("z", (1, inner)),
                ("alphaRaw", (1, heads)),
                ("betaRaw", (1, heads)),
                ("conv", (1, conv_dim)),
                ("q", (1, heads, dim)),
                ("k", (1, heads, dim)),
                ("decay", (1, heads)),
                ("beta", (1, heads)),
                ("pred", (1, heads, dim)),
                ("delta", (1, heads, dim)),
                ("state", (heads, dim, dim)),
                ("gated", (1, heads, dim)),
            )
        else:
            heads = int(cfg["n_heads"])
            kv_heads = int(cfg["n_heads_kv"])
            head_dim = int(cfg["head_dim"])
            score_count = int(
                self._lib.bonsai35_model_debug_internal_count(
                    self._handle, ctypes.c_int64(5)
                )
            )
            if score_count <= 0 or score_count % heads:
                raise RuntimeError(
                    "Bonsai-27B attention trace has an invalid score extent"
                )
            length = score_count // heads
            internals = (
                ("qg", (1, heads, 2, head_dim)),
                ("kProj", (1, kv_heads, head_dim)),
                ("v", (1, kv_heads, head_dim)),
                ("qRope", (1, heads, head_dim)),
                ("kRope", (1, kv_heads, head_dim)),
                ("scores", (heads, length)),
                ("probs", (heads, length)),
                ("head", (1, heads, head_dim)),
            )
        for kind, (name, shape) in enumerate(internals):
            count = int(
                self._lib.bonsai35_model_debug_internal_count(
                    self._handle, ctypes.c_int64(kind)
                )
            )
            expected = int(np.prod(shape, dtype=np.int64))
            if count != expected:
                raise RuntimeError(
                    f"Bonsai-27B internal trace {name} has {count} values, expected {expected}"
                )
            out = np.empty(shape, dtype=np.int64)
            rc = self._lib.bonsai35_model_export_internal(
                self._handle,
                ctypes.c_int64(kind),
                out.ctypes.data,
                ctypes.c_int64(count),
            )
            if rc != 0:
                raise RuntimeError(
                    f"bonsai35_model_export_internal({name}) failed with code {rc}"
                )
            result[name] = out
        return result

    @_b35_locked
    def position(self) -> int:
        return int(self._lib.bonsai35_model_position(self._handle))

    @_b35_locked
    def cache_fingerprints(self) -> tuple[int, int, int, int]:
        """Return diagnostic state/conv/K/V fingerprints without mutation."""
        raw = (ctypes.c_uint64 * 4)()
        rc = self._lib.bonsai35_model_cache_fingerprints(self._handle, raw)
        if rc != 0:
            raise RuntimeError(f"bonsai35_model_cache_fingerprints failed with code {rc}")
        return tuple(int(value) for value in raw)

    @_b35_locked
    def export_cache_tensor(self, layer_index: int, name: str) -> np.ndarray:
        """Copy one cache into the oracle's logical contiguous int64 shape."""

        layer_index = int(layer_index)
        if layer_index < 0 or layer_index >= len(self._layer_kinds):
            raise IndexError(f"Bonsai-27B cache layer is out of range: {layer_index}")
        cfg = self._config
        kind = self._layer_kinds[layer_index]
        position = self.position()
        if kind == "recurrent":
            shapes = {
                "state": (
                    int(cfg["ssmTimeStepRank"]),
                    int(cfg["ssmStateSize"]),
                    int(cfg["ssmStateSize"]),
                ),
                "conv": (
                    int(cfg["ssmConvKernel"]) - 1,
                    2 * int(cfg["ssmGroupCount"]) * int(cfg["ssmStateSize"])
                    + int(cfg["ssmInnerSize"]),
                ),
            }
            abi_kinds = {"state": 0, "conv": 1}
        elif kind == "attention":
            shapes = {
                "k": (int(cfg["n_heads_kv"]), position, int(cfg["head_dim"])),
                "v": (int(cfg["n_heads_kv"]), position, int(cfg["head_dim"])),
            }
            abi_kinds = {"k": 2, "v": 3}
        else:
            raise ValueError(f"unknown Qwen3.5 layer kind {kind!r}")
        if name not in shapes:
            raise ValueError(f"{kind} layer has no logical {name!r} cache")
        out = np.empty(shapes[name], dtype=np.int64)
        rc = self._lib.bonsai35_model_export_tensor(
            self._handle,
            ctypes.c_int64(layer_index),
            ctypes.c_int64(abi_kinds[name]),
            out.ctypes.data,
            ctypes.c_int64(out.size),
        )
        if rc != 0:
            raise RuntimeError(
                f"bonsai35_model_export_tensor({layer_index}, {name}) failed with code {rc}"
            )
        return out

    @_b35_locked
    def export_last_residual(self) -> np.ndarray:
        """Copy the final resident residual in canonical one-row shape."""

        out = np.empty((1, self.d_model), dtype=np.int64)
        rc = self._lib.bonsai35_model_export_tensor(
            self._handle,
            ctypes.c_int64(-1),
            ctypes.c_int64(4),
            out.ctypes.data,
            ctypes.c_int64(out.size),
        )
        if rc != 0:
            raise RuntimeError(f"Bonsai-27B residual export failed with code {rc}")
        return out

    @_b35_locked
    def stats(self) -> dict[str, int]:
        raw = _B35ExecStats()
        rc = self._lib.bonsai35_model_get_stats(self._handle, ctypes.byref(raw))
        if rc != 0:
            raise RuntimeError(f"bonsai35_model_get_stats failed with code {rc}")
        return {name: int(getattr(raw, name)) for name, _ctype in raw._fields_}

    def close(self) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            handle = getattr(self, "_handle", None)
            if handle is not None and handle.value:
                self._lib.bonsai35_model_free(handle)
                handle.value = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _contiguous_q1_weight(bits: np.ndarray, scale_fp: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    b = np.asarray(bits, dtype=np.uint8)
    src = np.asarray(scale_fp)
    # The scale is a fixed-point integer; a float (or otherwise non-integer) dtype here would be silently
    # truncated by ascontiguousarray(dtype=int64) and corrupt the result. Reject it loudly.
    if not np.issubdtype(src.dtype, np.integer):
        raise TypeError(f"Q1_0 scale must be an integer dtype, got {src.dtype}")
    # Preserve a narrow int32 scale cache (Recommendation 7) so the *_scale32 kernels can read it directly;
    # otherwise canonicalize to the committed int64 scale. int32 is lossless for any in-range scale and the
    # native math is byte-identical (q1_element_s32 promotes to the same 64-bit operand).
    dtype = np.int32 if src.dtype == np.int32 else np.int64
    s = np.ascontiguousarray(src, dtype=dtype)
    if not b.flags.c_contiguous:
        b = np.ascontiguousarray(b)
    out_f, n_blocks = s.shape
    if b.shape != (out_f, n_blocks, 16):
        raise ValueError(f"Q1_0 bits shape {b.shape} does not match {(out_f, n_blocks, 16)}")
    return b, s, int(out_f), int(n_blocks)


def q1_prepare_apply_many_native(
    x_fp: np.ndarray,
    weights: Q1WeightGroup | tuple,
    frac: int,
    *,
    prefer_lut32: bool = False,
) -> tuple[np.ndarray, ...] | None:
    """Prepare and apply a same-input projection group in one native call.

    ``prefer_lut32`` is a guarded optimization: the narrow builder returns rc=5
    before writing any output when a subset sum cannot fit int32, then this
    wrapper retries the exact uint64-LUT kernel.  A :class:`Q1WeightGroup`
    validates static metadata once and owns reusable decode output buffers.
    """
    lib = _load_lib()
    if lib is None:
        return None
    group = weights if isinstance(weights, Q1WeightGroup) else Q1WeightGroup(weights)
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_fp, dtype=np.int64)))
    if x.shape[1] != group.n_blocks * 128:
        raise ValueError(
            f"Q1_0 contraction mismatch: x has {x.shape[1]}, "
            f"weight expects {group.n_blocks * 128}"
        )
    total_count = int(x.shape[0]) * group.n_blocks
    lut_count = total_count * 16 * 256
    outs = group.outputs(int(x.shape[0]))
    common = (
        x.ctypes.data,
        ctypes.cast(group.bits_ptrs, ctypes.c_void_p),
        ctypes.cast(group.scale_ptrs, ctypes.c_void_p),
        group.out_features.ctypes.data,
        ctypes.c_int64(len(group.packed)),
        ctypes.c_int64(x.shape[0]),
        ctypes.c_int64(group.n_blocks),
        ctypes.c_int64(int(frac)),
        ctypes.cast(group._out_ptrs, ctypes.c_void_p),
    )

    if prefer_lut32:
        suffix = "_lut32_scale32" if group.scale32 else "_lut32"
        fn32 = getattr(lib, f"bonsai_q1_prepare_apply_multi_i64{suffix}", None)
        ws32 = _workspace_arrays_lut32(total_count, lut_count)
        if fn32 is not None and ws32 is not None:
            totals, lut = ws32
            _Q1_STATS["fused_calls"] += 1
            rc = fn32(
                *common,
                totals.ctypes.data, ctypes.c_size_t(totals.size),
                lut.ctypes.data, ctypes.c_size_t(lut.size),
            )
            if rc == 0:
                _Q1_STATS["lut32_hits"] += 1
                return outs
            if rc == 5:
                _Q1_STATS["lut32_fallbacks"] += 1
            elif rc != 3:
                raise RuntimeError(
                    f"bonsai_q1_prepare_apply_multi_i64{suffix} failed with code {rc}"
                )

    suffix = "_scale32" if group.scale32 else ""
    fn = getattr(lib, f"bonsai_q1_prepare_apply_multi_i64{suffix}", None)
    workspace = _workspace_arrays(total_count, lut_count)
    if fn is None or workspace is None:
        return None
    totals, lut = workspace
    _Q1_STATS["fused_calls"] += 1
    _Q1_STATS["u64_calls"] += 1
    rc = fn(
        *common,
        totals.ctypes.data, ctypes.c_size_t(totals.size),
        lut.ctypes.data, ctypes.c_size_t(lut.size),
    )
    if rc == 3:
        return None
    if rc != 0:
        raise RuntimeError(
            f"bonsai_q1_prepare_apply_multi_i64{suffix} failed with code {rc}"
        )
    return outs


def q1_prepare_native(x_fp: np.ndarray, n_blocks: int, *, lut32: bool = False) -> Q1Prepared | None:
    """Prepare a native activation LUT reusable across Q1_0 weights with the same input width.

    With lut32=True the LUT entries are int32 (half the gather bytes); returns None if the int32 symbol is
    absent, the workspace is too large, or a block exceeds the int32 envelope (rc 5) — the caller then
    retries with the uint64 LUT. The returned Q1Prepared's `lut.dtype` (int32 vs uint64) tells the apply
    wrappers which kernel to dispatch."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_q1_prepare_i64"):
        return None
    n_blocks = int(n_blocks)
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_fp, dtype=np.int64)))
    if x.shape[1] != n_blocks * 128:
        raise ValueError(f"Q1_0 contraction mismatch: x has {x.shape[1]}, weight expects {n_blocks * 128}")
    total_count = int(x.shape[0]) * n_blocks
    lut_count = total_count * 16 * 256
    if lut32:
        fn = getattr(lib, "bonsai_q1_prepare_i64_lut32", None)
        if fn is None:
            return None
        workspace = _workspace_arrays_lut32(total_count, lut_count)
        if workspace is None:
            return None
        totals, lut = workspace
    else:
        fn = lib.bonsai_q1_prepare_i64
        workspace = _workspace_arrays(total_count, lut_count)
        if workspace is None:
            return None
        totals, lut = workspace
    rc = fn(
        x.ctypes.data,
        ctypes.c_int64(x.shape[0]),
        ctypes.c_int64(n_blocks),
        totals.ctypes.data,
        ctypes.c_size_t(totals.size),
        lut.ctypes.data,
        ctypes.c_size_t(lut.size),
    )
    if rc == 3 or rc == 5:
        return None
    if rc != 0:
        raise RuntimeError(f"bonsai_q1_prepare_i64 failed with code {rc}")
    return Q1Prepared(x, x.shape[0], n_blocks, totals, lut)


def q1_linear_prepared_native(prepared: Q1Prepared, bits: np.ndarray,
                              scale_fp: np.ndarray, frac: int) -> np.ndarray | None:
    """Apply a packed-Q1_0 linear layer using a prepared activation LUT."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_q1_linear_i64_prepared"):
        return None
    b, s, out_f, n_blocks = _contiguous_q1_weight(bits, scale_fp)
    if n_blocks != prepared.n_blocks:
        raise ValueError(
            f"Q1_0 prepared n_blocks mismatch: prepared {prepared.n_blocks}, weight expects {n_blocks}"
        )
    out = np.empty((prepared.tokens, out_f), dtype=np.int64)
    if prepared.lut.dtype == np.int32:
        # int32-LUT prepared kernel reads an int64 scale; an int32-LUT prepare implies its apply symbol.
        fn = getattr(lib, "bonsai_q1_linear_i64_prepared_lut32", None)
        if fn is None:
            return None
        if s.dtype != np.int64:
            s = np.ascontiguousarray(s, dtype=np.int64)
    elif s.dtype == np.int32:
        fn = getattr(lib, "bonsai_q1_linear_i64_prepared_scale32", None)
        if fn is None:
            return None
    else:
        fn = lib.bonsai_q1_linear_i64_prepared
    rc = fn(
        b.ctypes.data,
        s.ctypes.data,
        ctypes.c_int64(prepared.tokens),
        ctypes.c_int64(out_f),
        ctypes.c_int64(n_blocks),
        ctypes.c_int64(int(frac)),
        out.ctypes.data,
        prepared.totals.ctypes.data,
        ctypes.c_size_t(prepared.totals.size),
        prepared.lut.ctypes.data,
        ctypes.c_size_t(prepared.lut.size),
    )
    if rc == 3:
        return None
    if rc != 0:
        raise RuntimeError(f"bonsai_q1_linear_i64_prepared failed with code {rc}")
    return out


def q1_linear_prepared_many_native(prepared: Q1Prepared, weights, frac: int) -> tuple[np.ndarray, ...] | None:
    """Apply multiple packed-Q1_0 linears against one prepared activation LUT in a single native call."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_q1_linear_i64_prepared_multi"):
        return None
    packed = []
    out_features = []
    outs = []
    narrow = True
    for bits, scale_fp in weights:
        b, s, out_f, n_blocks = _contiguous_q1_weight(bits, scale_fp)
        if n_blocks != prepared.n_blocks:
            raise ValueError(
                f"Q1_0 prepared n_blocks mismatch: prepared {prepared.n_blocks}, weight expects {n_blocks}"
            )
        packed.append((b, s))
        out_features.append(out_f)
        outs.append(np.empty((prepared.tokens, out_f), dtype=np.int64))
        if s.dtype != np.int32:
            narrow = False
    if not packed:
        return ()
    if prepared.lut.dtype == np.int32:
        # int32-LUT multi kernel reads int64 scales; only it can read the int32 LUT, so a missing symbol
        # means fall back (None) rather than handing the int32 LUT to a uint64-LUT kernel.
        fn = getattr(lib, "bonsai_q1_linear_i64_prepared_multi_lut32", None)
        if fn is None:
            return None
        packed = [(b, s if s.dtype == np.int64 else np.ascontiguousarray(s, dtype=np.int64))
                  for b, s in packed]
    else:
        fn = getattr(lib, "bonsai_q1_linear_i64_prepared_multi_scale32", None) if narrow else None
        if fn is None:
            # int64-kernel path: canonicalize EVERY scale to int64 first so an int32 cache array is never
            # handed to the int64 kernel, which would read its 4-byte entries as 8-byte scales and silently
            # corrupt the logits (the worst determinism failure). This covers a mixed int32/int64 batch and
            # the case where the scale32 symbol is absent; genuine int64 arrays pass through untouched.
            packed = [(b, s if s.dtype == np.int64 else np.ascontiguousarray(s, dtype=np.int64))
                      for b, s in packed]
            fn = lib.bonsai_q1_linear_i64_prepared_multi
    ptr_array = ctypes.c_void_p * len(packed)
    bits_ptrs = ptr_array(*(b.ctypes.data for b, _s in packed))
    scale_ptrs = ptr_array(*(s.ctypes.data for _b, s in packed))
    out_ptrs = ptr_array(*(out.ctypes.data for out in outs))
    out_f_arr = np.ascontiguousarray(np.asarray(out_features, dtype=np.int64))
    rc = fn(
        ctypes.cast(bits_ptrs, ctypes.c_void_p),
        ctypes.cast(scale_ptrs, ctypes.c_void_p),
        out_f_arr.ctypes.data,
        ctypes.c_int64(len(packed)),
        ctypes.c_int64(prepared.tokens),
        ctypes.c_int64(prepared.n_blocks),
        ctypes.c_int64(int(frac)),
        ctypes.cast(out_ptrs, ctypes.c_void_p),
        prepared.totals.ctypes.data,
        ctypes.c_size_t(prepared.totals.size),
        prepared.lut.ctypes.data,
        ctypes.c_size_t(prepared.lut.size),
    )
    if rc == 3:
        return None
    if rc != 0:
        raise RuntimeError(f"bonsai_q1_linear_i64_prepared_multi failed with code {rc}")
    return tuple(outs)


def q1_linear_native(x_fp: np.ndarray, bits: np.ndarray, scale_fp: np.ndarray, frac: int,
                     *, lut32: bool = False) -> np.ndarray | None:
    """Return native packed-Q1_0 linear output, or None when the native library is unavailable.

    With lut32=True (and an int64 scale) the int32-LUT-entry workspace kernel is tried first; it falls
    through to the uint64-LUT path if the symbol is absent, the workspace is too large, or a block exceeds
    the int32 envelope (rc 5)."""
    lib = _load_lib()
    if lib is None:
        return None
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_fp, dtype=np.int64)))
    b, s, out_f, n_blocks = _contiguous_q1_weight(bits, scale_fp)
    if x.shape[1] != n_blocks * 128:
        raise ValueError(f"Q1_0 contraction mismatch: x has {x.shape[1]}, weight expects {n_blocks * 128}")
    out = np.empty((x.shape[0], out_f), dtype=np.int64)
    if lut32 and s.dtype == np.int64:
        fn = getattr(lib, "bonsai_q1_linear_i64_workspace_lut32", None)
        if fn is not None:
            total_count = int(x.shape[0]) * int(n_blocks)
            lut_count = total_count * 16 * 256
            workspace = _workspace_arrays_lut32(total_count, lut_count)
            if workspace is not None:
                totals, lut = workspace
                rc = fn(
                    x.ctypes.data, b.ctypes.data, s.ctypes.data,
                    ctypes.c_int64(x.shape[0]), ctypes.c_int64(out_f),
                    ctypes.c_int64(n_blocks), ctypes.c_int64(int(frac)),
                    out.ctypes.data,
                    totals.ctypes.data, ctypes.c_size_t(totals.size),
                    lut.ctypes.data, ctypes.c_size_t(lut.size),
                )
                if rc == 0:
                    return out
                if rc not in (3, 5):    # 3 short ws / 5 out-of-int32 -> fall through to uint64 path
                    raise RuntimeError(f"bonsai_q1_linear_i64_workspace_lut32 failed with code {rc}")
    if s.dtype == np.int32:
        # Narrow scale cache: there is no base (non-workspace) int32 kernel, so require the workspace
        # variant and a workspace allocation, else signal None to fall back to the int64 oracle path.
        fn = getattr(lib, "bonsai_q1_linear_i64_workspace_scale32", None)
        if fn is None:
            return None
        total_count = int(x.shape[0]) * int(n_blocks)
        lut_count = total_count * 16 * 256
        workspace = _workspace_arrays(total_count, lut_count)
        if workspace is None:
            return None
        totals, lut = workspace
        rc = fn(
            x.ctypes.data,
            b.ctypes.data,
            s.ctypes.data,
            ctypes.c_int64(x.shape[0]),
            ctypes.c_int64(out_f),
            ctypes.c_int64(n_blocks),
            ctypes.c_int64(int(frac)),
            out.ctypes.data,
            totals.ctypes.data,
            ctypes.c_size_t(totals.size),
            lut.ctypes.data,
            ctypes.c_size_t(lut.size),
        )
        if rc == 3:
            return None
        if rc != 0:
            raise RuntimeError(f"bonsai_q1_linear_i64_workspace_scale32 failed with code {rc}")
        return out
    rc = None
    workspace_fn = getattr(lib, "bonsai_q1_linear_i64_workspace", None)
    if workspace_fn is not None:
        total_count = int(x.shape[0]) * int(n_blocks)
        lut_count = total_count * 16 * 256
        workspace = _workspace_arrays(total_count, lut_count)
        if workspace is not None:
            totals, lut = workspace
            rc = workspace_fn(
                x.ctypes.data,
                b.ctypes.data,
                s.ctypes.data,
                ctypes.c_int64(x.shape[0]),
                ctypes.c_int64(out_f),
                ctypes.c_int64(n_blocks),
                ctypes.c_int64(int(frac)),
                out.ctypes.data,
                totals.ctypes.data,
                ctypes.c_size_t(totals.size),
                lut.ctypes.data,
                ctypes.c_size_t(lut.size),
            )
    if rc is None or rc == 3:
        rc = lib.bonsai_q1_linear_i64(
            x.ctypes.data,
            b.ctypes.data,
            s.ctypes.data,
            ctypes.c_int64(x.shape[0]),
            ctypes.c_int64(out_f),
            ctypes.c_int64(n_blocks),
            ctypes.c_int64(int(frac)),
            out.ctypes.data,
        )
    if rc == 2:
        return None
    if rc != 0:
        raise RuntimeError(f"bonsai_q1_linear_i64 failed with code {rc}")
    return out


def q1_argmax_native(x_fp: np.ndarray, bits: np.ndarray, scale_fp: np.ndarray, frac: int,
                     *, lut32: bool = False) -> np.ndarray | None:
    """Return argmax ids for native packed-Q1_0 linear output without materializing the full logits row.

    With lut32=True (and an int64 scale) the int32-LUT-entry argmax kernel is tried first (the vocab head is
    the most LUT-reuse-bound gather); it falls through to the uint64-LUT argmax on rc 3/5."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_q1_argmax_i64_workspace"):
        return None
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_fp, dtype=np.int64)))
    b, s, out_f, n_blocks = _contiguous_q1_weight(bits, scale_fp)
    if x.shape[1] != n_blocks * 128:
        raise ValueError(f"Q1_0 contraction mismatch: x has {x.shape[1]}, weight expects {n_blocks * 128}")
    total_count = int(x.shape[0]) * int(n_blocks)
    lut_count = total_count * 16 * 256
    ids = np.empty(x.shape[0], dtype=np.int64)
    values = np.empty(x.shape[0], dtype=np.int64)
    if lut32:
        sym = (
            "bonsai_q1_argmax_i64_workspace_lut32_scale32"
            if s.dtype == np.int32
            else "bonsai_q1_argmax_i64_workspace_lut32"
        )
        fn = getattr(lib, sym, None)
        if fn is not None:
            ws32 = _workspace_arrays_lut32(total_count, lut_count)
            if ws32 is not None:
                totals32, lut32arr = ws32
                rc = fn(
                    x.ctypes.data, b.ctypes.data, s.ctypes.data,
                    ctypes.c_int64(x.shape[0]), ctypes.c_int64(out_f),
                    ctypes.c_int64(n_blocks), ctypes.c_int64(int(frac)),
                    ids.ctypes.data, values.ctypes.data,
                    totals32.ctypes.data, ctypes.c_size_t(totals32.size),
                    lut32arr.ctypes.data, ctypes.c_size_t(lut32arr.size),
                )
                if rc == 0:
                    return ids
                if rc not in (3, 5):
                    raise RuntimeError(f"{sym} failed with code {rc}")
    workspace = _workspace_arrays(total_count, lut_count)
    if workspace is None:
        return None
    totals, lut = workspace
    if s.dtype == np.int32:
        fn = getattr(lib, "bonsai_q1_argmax_i64_workspace_scale32", None)
        if fn is None:
            return None
    else:
        fn = lib.bonsai_q1_argmax_i64_workspace
    rc = fn(
        x.ctypes.data,
        b.ctypes.data,
        s.ctypes.data,
        ctypes.c_int64(x.shape[0]),
        ctypes.c_int64(out_f),
        ctypes.c_int64(n_blocks),
        ctypes.c_int64(int(frac)),
        ids.ctypes.data,
        values.ctypes.data,
        totals.ctypes.data,
        ctypes.c_size_t(totals.size),
        lut.ctypes.data,
        ctypes.c_size_t(lut.size),
    )
    if rc == 3:
        return None
    if rc != 0:
        raise RuntimeError(f"bonsai_q1_argmax_i64_workspace failed with code {rc}")
    return ids


def rmsnorm_native(x_fp: np.ndarray, frac: int, *, eps: int = 1,
                   gain_q: np.ndarray | None = None) -> np.ndarray | None:
    """Return native fixed-point RMSNorm output, or None when unavailable/outside the fast envelope."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_rmsnorm_i64"):
        return None
    x = np.ascontiguousarray(np.atleast_2d(np.asarray(x_fp, dtype=np.int64)))
    gain = None
    gain_ptr = ctypes.c_void_p(0)
    if gain_q is not None:
        gain = np.ascontiguousarray(np.asarray(gain_q, dtype=np.int64))
        if gain.shape != (x.shape[1],):
            raise ValueError(f"RMSNorm gain shape {gain.shape} does not match {(x.shape[1],)}")
        gain_ptr = ctypes.c_void_p(gain.ctypes.data)
    out = np.empty(x.shape, dtype=np.int64)
    rc = lib.bonsai_rmsnorm_i64(
        x.ctypes.data,
        ctypes.c_int64(x.shape[0]),
        ctypes.c_int64(x.shape[1]),
        ctypes.c_int64(int(frac)),
        ctypes.c_int64(int(eps)),
        gain_ptr,
        out.ctypes.data,
    )
    if rc == 4:
        return None
    if rc != 0:
        raise RuntimeError(f"bonsai_rmsnorm_i64 failed with code {rc}")
    return out


def _kv_for_native(arr: np.ndarray) -> np.ndarray:
    """Return a (Hkv, L, hd) int64 view whose per-head (L, hd) block is contiguous, WITHOUT copying when the
    input is a KV-cache buffer slice: only the inter-head stride may exceed L*hd (the cap*hd buffer stride),
    which the native kernel takes as a parameter. Falls back to a contiguous copy for any other layout."""
    a = np.asarray(arr)
    if a.dtype != np.int64:
        a = np.ascontiguousarray(a, dtype=np.int64)
    if a.ndim != 3:
        return a
    it = a.itemsize
    _Hkv, L, hd = a.shape
    if (a.strides[2] == it and a.strides[1] == hd * it
            and a.strides[0] % it == 0 and a.strides[0] >= L * hd * it):
        return a
    return np.ascontiguousarray(a)


def silu_native(x_fp: np.ndarray, frac: int) -> np.ndarray | None:
    """Native element-wise fixed-point SiLU, or None when unavailable. Byte-identical to
    reference_bonsai.fixed_point_silu; the caller falls back to the NumPy oracle on None."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_silu_i64"):
        return None
    x = np.ascontiguousarray(np.asarray(x_fp, dtype=np.int64))
    out = np.empty(x.shape, dtype=np.int64)
    rc = lib.bonsai_silu_i64(
        x.ctypes.data,
        ctypes.c_int64(x.size),
        ctypes.c_int64(int(frac)),
        out.ctypes.data,
    )
    if rc != 0:
        raise RuntimeError(f"bonsai_silu_i64 failed with code {rc}")
    return out


def gdn_decode_native(
    state_fp: np.ndarray,
    q_fp: np.ndarray,
    k_fp: np.ndarray,
    v_fp: np.ndarray,
    beta_fp: np.ndarray,
    decay_fp: np.ndarray,
    frac: int,
    state_frac: int,
    outer_shift: int,
    inv_sqrt_fp: int,
) -> np.ndarray | None:
    """Mutate one M=1 Gated DeltaNet state and return its exact output."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_gdn_decode_i64"):
        return None
    state = np.asarray(state_fp)
    if state.dtype != np.int64 or not state.flags.c_contiguous or state.ndim != 3:
        raise ValueError("GDN state must be a writable contiguous (heads, dim, dim) int64 array")
    heads, dim, dim2 = map(int, state.shape)
    if dim != dim2 or not state.flags.writeable:
        raise ValueError("GDN state must be writable with square per-head matrices")

    def matrix(value, name):
        a = np.ascontiguousarray(np.asarray(value, dtype=np.int64))
        if a.shape != (heads, dim):
            raise ValueError(f"GDN {name} shape {a.shape} != {(heads, dim)}")
        return a

    q = matrix(q_fp, "q")
    k = matrix(k_fp, "k")
    v = matrix(v_fp, "v")
    beta = np.ascontiguousarray(np.asarray(beta_fp, dtype=np.int64).reshape(-1))
    decay = np.ascontiguousarray(np.asarray(decay_fp, dtype=np.int64).reshape(-1))
    if beta.shape != (heads,) or decay.shape != (heads,):
        raise ValueError(f"GDN beta/decay shapes {beta.shape}/{decay.shape} != {(heads,)}")
    out = np.empty((heads, dim), dtype=np.int64)
    rc = lib.bonsai_gdn_decode_i64(
        state.ctypes.data, q.ctypes.data, k.ctypes.data, v.ctypes.data,
        beta.ctypes.data, decay.ctypes.data,
        ctypes.c_int64(heads), ctypes.c_int64(dim), ctypes.c_int64(int(frac)),
        ctypes.c_int64(int(state_frac)), ctypes.c_int64(int(outer_shift)),
        ctypes.c_int64(int(inv_sqrt_fp)), out.ctypes.data,
    )
    if rc != 0:
        raise RuntimeError(f"bonsai_gdn_decode_i64 failed with code {rc}")
    return out


def gdn_prefill_native(
    state_fp: np.ndarray,
    q_fp: np.ndarray,
    k_fp: np.ndarray,
    v_fp: np.ndarray,
    beta_fp: np.ndarray,
    decay_fp: np.ndarray,
    frac: int,
    state_frac: int,
    outer_shift: int,
    inv_sqrt_fp: int,
) -> np.ndarray | None:
    """Run a sequential-token GDN prefill while parallelizing independent heads."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_gdn_prefill_i64"):
        return None
    state = np.asarray(state_fp)
    if state.dtype != np.int64 or not state.flags.c_contiguous or state.ndim != 3:
        raise ValueError("GDN state must be a writable contiguous (heads, dim, dim) int64 array")
    heads, dim, dim2 = map(int, state.shape)
    if dim != dim2 or not state.flags.writeable:
        raise ValueError("GDN state must be writable with square per-head matrices")
    q = np.ascontiguousarray(np.asarray(q_fp, dtype=np.int64))
    k = np.ascontiguousarray(np.asarray(k_fp, dtype=np.int64))
    v = np.ascontiguousarray(np.asarray(v_fp, dtype=np.int64))
    if q.ndim != 3 or q.shape[1:] != (heads, dim):
        raise ValueError(f"GDN q shape {q.shape} must be (tokens, {heads}, {dim})")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"GDN q/k/v shapes differ: {q.shape}/{k.shape}/{v.shape}")
    tokens = int(q.shape[0])
    beta = np.ascontiguousarray(np.asarray(beta_fp, dtype=np.int64))
    decay = np.ascontiguousarray(np.asarray(decay_fp, dtype=np.int64))
    if beta.shape != (tokens, heads) or decay.shape != (tokens, heads):
        raise ValueError(
            f"GDN beta/decay shapes {beta.shape}/{decay.shape} != {(tokens, heads)}"
        )
    out = np.empty(q.shape, dtype=np.int64)
    rc = lib.bonsai_gdn_prefill_i64(
        state.ctypes.data, q.ctypes.data, k.ctypes.data, v.ctypes.data,
        beta.ctypes.data, decay.ctypes.data,
        ctypes.c_int64(tokens), ctypes.c_int64(heads), ctypes.c_int64(dim),
        ctypes.c_int64(int(frac)), ctypes.c_int64(int(state_frac)),
        ctypes.c_int64(int(outer_shift)), ctypes.c_int64(int(inv_sqrt_fp)),
        out.ctypes.data,
    )
    if rc != 0:
        raise RuntimeError(f"bonsai_gdn_prefill_i64 failed with code {rc}")
    return out


def attention_decode_native(q_fp: np.ndarray, k_fp: np.ndarray, v_fp: np.ndarray,
                            frac: int, inv_sqrt_fp: int) -> np.ndarray | None:
    """Native M=1 cached-decode attention. q:(H,hd), k/v:(Hkv,L,hd) int64 fixed-point (post q/k-norm+RoPE
    for q/k; cached for k/v). Returns (H,hd) int64, or None when unavailable / the workspace is too large /
    a head would overflow the int64 attention bound (the caller then falls back to the NumPy path, which
    fails loud, preserving attention's no-silent-wrap contract)."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_attention_decode_i64"):
        return None
    q = np.ascontiguousarray(q_fp, dtype=np.int64)            # (H, hd) is contiguous already (no-op)
    k = _kv_for_native(k_fp)
    v = _kv_for_native(v_fp)
    if q.ndim != 2 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(f"attention shapes: q{q.shape} k{k.shape} v{v.shape}")
    H, hd = q.shape
    Hkv, L, hd_k = k.shape
    if hd_k != hd or v.shape != (Hkv, L, hd):
        raise ValueError(f"attention shape mismatch: q{q.shape} k{k.shape} v{v.shape}")
    it = k.itemsize
    k_kv_stride = k.strides[0] // it                          # cap*hd for a KV-cache buffer slice (no copy)
    v_kv_stride = v.strides[0] // it
    scratch = _attn_scratch(H * L)
    if scratch is None:
        return None
    out = np.empty((H, hd), dtype=np.int64)
    rc = lib.bonsai_attention_decode_i64(
        q.ctypes.data, k.ctypes.data, v.ctypes.data,
        ctypes.c_int64(H), ctypes.c_int64(Hkv), ctypes.c_int64(hd), ctypes.c_int64(L),
        ctypes.c_int64(int(k_kv_stride)), ctypes.c_int64(int(v_kv_stride)),
        ctypes.c_int64(int(frac)), ctypes.c_int64(int(inv_sqrt_fp)),
        out.ctypes.data, scratch.ctypes.data, ctypes.c_size_t(scratch.size),
    )
    if rc == 2 or rc == 3:
        return None
    if rc != 0:
        raise RuntimeError(f"bonsai_attention_decode_i64 failed with code {rc}")
    return out


def attention_prefill_native(q_fp: np.ndarray, k_fp: np.ndarray, v_fp: np.ndarray,
                             start: int, frac: int, inv_sqrt_fp: int) -> np.ndarray | None:
    """Native M=N causal PREFILL attention. q:(H,M,hd) post q-norm+RoPE; k/v:(Hkv,L,hd) RoPE'd K / raw V,
    L == start+M. Returns (H,M,hd) int64 byte-identical to the NumPy causal path, or None when unavailable /
    a head would overflow the int64 bound (the caller then uses the loud NumPy path — no silent wrap)."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_attention_prefill_i64"):
        return None
    q = np.ascontiguousarray(q_fp, dtype=np.int64)
    k = np.ascontiguousarray(k_fp, dtype=np.int64)        # contiguous copy: prefill is one call/layer, not hot
    v = np.ascontiguousarray(v_fp, dtype=np.int64)
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(f"prefill attn shapes: q{q.shape} k{k.shape} v{v.shape}")
    H, M, hd = q.shape
    Hkv, L, hd_k = k.shape
    if hd_k != hd or v.shape != (Hkv, L, hd) or L != start + M:
        raise ValueError(f"prefill attn shape/length mismatch: q{q.shape} k{k.shape} v{v.shape} start={start}")
    out = np.empty((H, M, hd), dtype=np.int64)
    rc = lib.bonsai_attention_prefill_i64(
        q.ctypes.data, k.ctypes.data, v.ctypes.data,
        ctypes.c_int64(H), ctypes.c_int64(Hkv), ctypes.c_int64(hd),
        ctypes.c_int64(M), ctypes.c_int64(L), ctypes.c_int64(int(start)),
        ctypes.c_int64(int(frac)), ctypes.c_int64(int(inv_sqrt_fp)),
        out.ctypes.data,
    )
    if rc == 2:
        return None
    if rc != 0:
        raise RuntimeError(f"bonsai_attention_prefill_i64 failed with code {rc}")
    return out


def attention_decode_batched_native(q_fp: np.ndarray, k_list, v_list, lengths,
                                    frac: int, inv_sqrt_fp: int) -> np.ndarray | None:
    """Native BATCHED M=1 decode attention: B independent decode attentions in ONE call. q:(B,H,hd);
    k_list/v_list: B cache arrays each (Hkv, L_b, hd) int64 (cache-buffer views — the buffer stride is passed,
    no copy); lengths: B ints L_b. Returns (B,H,hd) int64 byte-identical to B separate
    attention_decode_native(q[b], k_list[b], v_list[b]) calls, or None on overflow / unavailable (caller then
    uses the per-sequence NumPy/M=1 path — no silent wrap)."""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "bonsai_attention_decode_batched_i64"):
        return None
    B = len(k_list)
    if B == 0 or len(v_list) != B or len(lengths) != B:
        raise ValueError("batched attn: k_list/v_list/lengths must have the same B > 0")
    q = np.ascontiguousarray(q_fp, dtype=np.int64)
    if q.ndim != 3 or q.shape[0] != B:
        raise ValueError(f"batched attn q shape {q.shape} (want (B={B}, H, hd))")
    H, hd = int(q.shape[1]), int(q.shape[2])
    k_addr = np.empty(B, dtype=np.uintp)
    v_addr = np.empty(B, dtype=np.uintp)
    klen = np.empty(B, dtype=np.int64)
    kstr = np.empty(B, dtype=np.int64)
    vstr = np.empty(B, dtype=np.int64)
    keep = []                                                # hold refs so addresses stay valid through the call
    Hkv = None
    for b in range(B):
        k = _kv_for_native(k_list[b])
        v = _kv_for_native(v_list[b])
        keep.append((k, v))
        hkv_b, Lb, hd_k = k.shape
        if Hkv is None:
            Hkv = int(hkv_b)
        if hd_k != hd or k.shape != v.shape or int(hkv_b) != Hkv or Lb != int(lengths[b]):
            raise ValueError(f"batched attn shape mismatch at b={b}: k{k.shape} v{v.shape} len={lengths[b]}")
        k_addr[b] = k.ctypes.data
        v_addr[b] = v.ctypes.data
        klen[b] = Lb
        kstr[b] = k.strides[0] // k.itemsize
        vstr[b] = v.strides[0] // v.itemsize
    out = np.empty((B, H, hd), dtype=np.int64)
    rc = lib.bonsai_attention_decode_batched_i64(
        q.ctypes.data, k_addr.ctypes.data, v_addr.ctypes.data, klen.ctypes.data,
        kstr.ctypes.data, vstr.ctypes.data,
        ctypes.c_int64(B), ctypes.c_int64(H), ctypes.c_int64(int(Hkv)), ctypes.c_int64(hd),
        ctypes.c_int64(int(frac)), ctypes.c_int64(int(inv_sqrt_fp)),
        out.ctypes.data,
    )
    del keep
    if rc == 2:
        return None
    if rc != 0:
        raise RuntimeError(f"bonsai_attention_decode_batched_i64 failed with code {rc}")
    return out
