from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from trinote.infer_int.artifact_io_bonsai import load_artifact_bonsai
from trinote.infer_int.q1_native import Bonsai35NativeExecutor, _B35ExecStats
from trinote.infer_int.reference_bonsai35 import (
    BonsaiQwen35ReferenceModel,
    random_bonsai35_artifact,
)


_REAL_OPT_IN = os.environ.get("TRINOTE_RUN_BONSAI35_REAL_EXECUTOR", "") == "1"


def _real_artifact():
    path = Path(os.environ.get(
        "BONSAI_INTEGER_27B_ARTIFACT",
        str(Path.home() / ".local/trinote/models/"
            "Bonsai-27B-Q1_0-int-qwen35.safetensors"),
    ))
    assert path.is_file(), path
    artifact, _info = load_artifact_bonsai(path)
    return artifact


def test_executor_stats_abi_contains_prompt_tile_and_exact_rms_counters():
    names = tuple(name for name, _ctype in _B35ExecStats._fields_)
    assert names[7:] == (
        "layer_major_prefills",
        "layer_major_rows",
        "prefill_tiles_40",
        "prefill_tiles_48",
        "prefill_tiles_136",
        "fused_residual_rms_calls",
        "parallel_rms_calls",
        "last_team_size",
        "selected_isa",
        "selected_lut_bits",
        "cache_width_bits",
        "prefill_tile_40",
        "prefill_tile_48",
        "prefill_tile_136",
    )


def test_oracle_selected_layer_intermediates_are_named_and_exact():
    artifact = random_bonsai35_artifact(seed=711)
    model = BonsaiQwen35ReferenceModel(artifact)
    trace = model.debug_layer_intermediates([7, 3, 11], 0, native_primitives=False)
    boundary_names = ("n1", "branch", "residual", "n2", "ffn", "output")
    internal_names = (
        "qkv", "z", "alphaRaw", "betaRaw", "conv", "q", "k", "decay",
        "beta", "pred", "delta", "state", "gated",
    )
    assert tuple(trace) == boundary_names + internal_names
    assert all(
        trace[name].shape == (3, artifact["config"]["dModel"])
        for name in boundary_names
    )
    assert all(value.dtype == np.int64 and value.flags.c_contiguous for value in trace.values())
    assert np.array_equal(trace["output"], trace["residual"] + trace["ffn"])
    assert trace["q"].shape[0] == trace["pred"].shape[0] == 1
    assert trace["state"].shape == (
        artifact["config"]["ssmTimeStepRank"],
        artifact["config"]["ssmStateSize"],
        artifact["config"]["ssmStateSize"],
    )


@pytest.mark.skipif(
    not _REAL_OPT_IN,
    reason="set TRINOTE_RUN_BONSAI35_REAL_EXECUTOR=1 for the external 4 GiB artifact",
)
@pytest.mark.parametrize("layer_index", [0, 3])
def test_real_layer_major_named_trace_matches_m1_decode(layer_index):
    artifact = _real_artifact()
    prompt = [12675, 11]

    tiled = Bonsai35NativeExecutor(artifact)
    tiled.debug_trace_layer(layer_index)
    tiled_hidden = tiled.prefill(prompt)
    tiled_trace = tiled.export_debug_trace()
    tiled_stats = tiled.stats()

    sequential = Bonsai35NativeExecutor(artifact)
    sequential.debug_trace_layer(layer_index)
    first_hidden = sequential.prefill(prompt[:1])
    first_trace = sequential.export_debug_trace()
    second_hidden = sequential.decode(prompt[1])
    second_trace = sequential.export_debug_trace()
    sequential_stats = sequential.stats()

    assert np.array_equal(tiled_hidden, np.concatenate((first_hidden, second_hidden), axis=0))
    boundary_names = ("n1", "branch", "residual", "n2", "ffn", "output")
    for name in boundary_names:
        expected = np.concatenate((first_trace[name], second_trace[name]), axis=0)
        assert np.array_equal(tiled_trace[name], expected), name
    for name in tiled_trace:
        if name not in boundary_names:
            assert np.array_equal(tiled_trace[name], second_trace[name]), name
    assert tiled.cache_fingerprints() == sequential.cache_fingerprints()
    assert tiled_stats["layer_major_prefills"] == 1
    assert tiled_stats["layer_major_rows"] == 2
    assert tiled_stats["prefill_tiles_40"] > 0
    assert tiled_stats["prefill_tiles_48"] > 0
    assert tiled_stats["prefill_tiles_136"] > 0
    assert tiled_stats["fused_residual_rms_calls"] > 0
    assert sequential_stats["parallel_rms_calls"] > 0


@pytest.mark.skipif(
    not _REAL_OPT_IN,
    reason="set TRINOTE_RUN_BONSAI35_REAL_EXECUTOR=1 for the external 4 GiB artifact",
)
@pytest.mark.parametrize("layer_index", [0, 3])
@pytest.mark.parametrize("prompt", [(12675,), (12675, 11)])
def test_real_named_trace_matches_independent_oracle(monkeypatch, layer_index, prompt):
    artifact = _real_artifact()
    monkeypatch.setenv("TRINOTE_BONSAI35_MODEL_EXECUTOR", "0")
    oracle = BonsaiQwen35ReferenceModel(artifact)
    assert oracle.enable_native()
    expected = oracle.debug_layer_intermediates(
        prompt, layer_index, native_primitives=True
    )

    executor = Bonsai35NativeExecutor(artifact)
    executor.debug_trace_layer(layer_index)
    executor.prefill(prompt)
    actual = executor.export_debug_trace()
    for name in expected:
        assert np.array_equal(actual[name], expected[name]), name


@pytest.mark.skipif(
    not _REAL_OPT_IN,
    reason="set TRINOTE_RUN_BONSAI35_REAL_EXECUTOR=1 for the external 4 GiB artifact",
)
def test_real_layer_major_uint64_policy_matches_m1(monkeypatch):
    monkeypatch.setenv("TRINOTE_BONSAI35_Q1_LUT32", "0")
    artifact = _real_artifact()
    tiled = Bonsai35NativeExecutor(artifact)
    tiled_hidden = tiled.prefill([12675, 11])
    sequential = Bonsai35NativeExecutor(artifact)
    expected = np.concatenate(
        (sequential.prefill([12675]), sequential.decode(11)), axis=0
    )
    assert np.array_equal(tiled_hidden, expected)
    assert tiled.cache_fingerprints() == sequential.cache_fingerprints()
    stats = tiled.stats()
    assert stats["selected_lut_bits"] == 64
    assert stats["lut64_groups"] == stats["q1_groups"]
    assert stats["lut32_hits"] == 0
    assert (
        stats["prefill_tile_40"],
        stats["prefill_tile_48"],
        stats["prefill_tile_136"],
    ) == (1, 1, 1)


@pytest.mark.skipif(
    not _REAL_OPT_IN,
    reason="set TRINOTE_RUN_BONSAI35_REAL_EXECUTOR=1 for the external 4 GiB artifact",
)
def test_real_layer_major_forced_lut_replay_is_exact():
    artifact = _real_artifact()
    expected = Bonsai35NativeExecutor(artifact)
    expected_hidden = expected.prefill([12675, 11])

    replay = Bonsai35NativeExecutor(artifact)
    replay.force_lut_fallback(True)
    actual_hidden = replay.prefill([12675, 11])
    assert np.array_equal(actual_hidden, expected_hidden)
    assert replay.cache_fingerprints() == expected.cache_fingerprints()
    stats = replay.stats()
    assert stats["lut32_fallbacks"] == stats["q1_groups"]
    assert stats["lut32_hits"] == 0


@pytest.mark.skipif(
    not _REAL_OPT_IN,
    reason="set TRINOTE_RUN_BONSAI35_REAL_EXECUTOR=1 for the external 4 GiB artifact",
)
def test_real_prefill_tile_override_is_reported(monkeypatch):
    monkeypatch.setenv("TRINOTE_BONSAI35_PREFILL_TILE", "3")
    artifact = _real_artifact()
    executor = Bonsai35NativeExecutor(artifact)
    prompt = [12675, 11, 42, 7]  # crosses the explicit three-row tile boundary
    hidden = executor.prefill(prompt)
    stats = executor.stats()
    assert (
        stats["prefill_tile_40"],
        stats["prefill_tile_48"],
        stats["prefill_tile_136"],
    ) == (3, 3, 3)
    assert stats["layer_major_rows"] == 4
    assert min(
        stats["prefill_tiles_40"],
        stats["prefill_tiles_48"],
        stats["prefill_tiles_136"],
    ) > 0
    sequential = Bonsai35NativeExecutor(artifact)
    expected = np.concatenate(
        (
            sequential.prefill([12675]),
            sequential.decode(11),
            sequential.decode(42),
            sequential.decode(7),
        ),
        axis=0,
    )
    assert np.array_equal(hidden, expected)
    assert executor.cache_fingerprints() == sequential.cache_fingerprints()
