from __future__ import annotations

import numpy as np
import pytest

from trinote.cli.trace_bonsai35_cli import _token_ids
from trinote.cli.quality_gate_bonsai_cli import _model_for_artifact
from trinote.infer_int.reference_bonsai35 import (
    _Qwen35Cache,
    BonsaiQwen35ReferenceModel,
    random_bonsai35_artifact,
)
from trinote.infer_int.trace_bonsai35 import (
    CACHE_COMMITMENT_FORMAT,
    TRACE_FORMAT,
    assert_trace_equal,
    canonical_cache_digest,
    canonical_cache_record,
    tensor_digest,
    trace_cached_greedy,
    trace_prefill,
)


def test_tensor_digest_commits_shape_dtype_and_logical_values():
    a = np.arange(12, dtype=np.int64).reshape(3, 4)
    assert tensor_digest(a) == tensor_digest(np.asfortranarray(a))
    assert tensor_digest(a) != tensor_digest(a.reshape(2, 6))
    assert tensor_digest(a) != tensor_digest(a.astype(np.int32))
    assert tensor_digest(a) == tensor_digest(a.astype(">i8"))
    with pytest.raises(TypeError):
        tensor_digest(np.asarray([object()]))


def test_qwen35_trace_is_repeatable_and_cache_sensitive():
    artifact = random_bonsai35_artifact(seed=31)
    a = trace_prefill(artifact, [1, 7, 3])
    b = trace_prefill(artifact, [1, 7, 3])
    assert_trace_equal(a, b)
    assert a["format"] == TRACE_FORMAT
    assert len(a["layers"]) == artifact["config"]["nLayers"]
    assert a["layers"][0]["kind"] == "recurrent"
    assert "state" in a["layers"][0]["cache"]
    assert a["layers"][3]["kind"] == "attention"
    assert "k" in a["layers"][3]["cache"]

    changed = trace_prefill(artifact, [1, 7, 4])
    assert changed["layers"][-1]["output"] != a["layers"][-1]["output"]


def test_qwen35_trace_native_matches_oracle_if_available():
    artifact = random_bonsai35_artifact(seed=32)
    oracle = trace_prefill(artifact, [2, 5, 9], native=False)
    native = trace_prefill(artifact, [2, 5, 9], native=True)
    assert_trace_equal(native, oracle)


def test_cache_commitment_is_independent_of_oracle_and_resident_layout():
    artifact = random_bonsai35_artifact(seed=320)
    model = BonsaiQwen35ReferenceModel(artifact)
    cache = _Qwen35Cache(len(artifact["layers"]))
    residual = model._run_layers([2, 5, 9], cache)
    kinds = [layer["kind"] for layer in artifact["layers"]]

    def oracle_tensor(layer, name):
        return getattr(cache, name)[layer]

    oracle = canonical_cache_record(
        kinds,
        position=cache.t,
        tensor_for=oracle_tensor,
        last_residual=residual,
    )

    # Model the resident ABI: recurrent arrays are flat allocations and KV is
    # context-strided with an unused tail.  The exporter presents only the
    # same logical shapes committed by the oracle.
    resident: dict[tuple[int, str], np.ndarray] = {}
    for layer, kind in enumerate(kinds):
        for name in (("state", "conv") if kind == "recurrent" else ("k", "v")):
            value = np.asarray(oracle_tensor(layer, name))
            if kind == "attention":
                padded = np.zeros(
                    (value.shape[0], artifact["config"]["context_len"], value.shape[2]),
                    dtype=np.int64,
                )
                padded[:, :cache.t] = value
                resident[layer, name] = padded
            else:
                resident[layer, name] = value.reshape(-1).copy()

    def resident_tensor(layer, name):
        value = np.asarray(oracle_tensor(layer, name))
        stored = resident[layer, name]
        if kinds[layer] == "attention":
            return np.ascontiguousarray(stored[:, :cache.t])
        return stored.reshape(value.shape)

    native = canonical_cache_record(
        kinds,
        position=cache.t,
        tensor_for=resident_tensor,
        last_residual=residual[-1].copy(),
    )
    assert native == oracle
    assert native["format"] == CACHE_COMMITMENT_FORMAT
    assert canonical_cache_digest(native) == canonical_cache_digest(oracle)


def test_cache_commitment_canonicalizes_strided_multi_head_kv_prefix():
    # A capacity-backed [Hkv, capacity, hd] prefix is not C-contiguous when
    # Hkv > 1 and position < capacity.  It is nevertheless the canonical
    # logical KV tensor and must hash identically to a compact exporter.
    capacity = np.arange(4 * 16 * 8, dtype=np.int64).reshape(4, 16, 8)
    strided = capacity[:, :3]
    assert not strided.flags.c_contiguous
    compact = np.ascontiguousarray(strided)

    def record(value):
        return canonical_cache_record(
            ["attention"],
            position=3,
            tensor_for=lambda _layer, _name: value,
            last_residual=np.asarray([1, 2, 3], dtype=np.int64),
        )

    assert record(strided) == record(compact)


def test_qwen35_cached_trace_commits_each_step():
    artifact = random_bonsai35_artifact(seed=33)
    trace = trace_cached_greedy(artifact, [3, 1], 3)
    assert len(trace["outputIds"]) == 3
    assert [row["token"] for row in trace["steps"]] == trace["outputIds"]
    assert "postDecodeLayers" in trace["steps"][0]
    assert "postDecodeLayers" in trace["steps"][-1]


def test_trace_comparison_reports_the_exact_path():
    expected = {"a": [{"b": "one"}]}
    actual = {"a": [{"b": "two"}]}
    with pytest.raises(AssertionError, match=r"\$ \.a\[0\]\.b"):
        assert_trace_equal(actual, expected)


def test_trace_cli_token_parser_is_strict():
    assert _token_ids("1, 2,3") == [1, 2, 3]
    with pytest.raises(Exception):
        _token_ids("")
    with pytest.raises(Exception):
        _token_ids("1,-2")


def test_quality_gate_selects_qwen35_graph():
    model, architecture = _model_for_artifact(random_bonsai35_artifact(seed=34))
    assert architecture == "qwen35"
    assert isinstance(model, BonsaiQwen35ReferenceModel)
    with pytest.raises(ValueError, match="unsupported"):
        _model_for_artifact({"config": {"architecture": "not-a-model"}})
