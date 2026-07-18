from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from trinote.infer_int.prompt_cache_bonsai35 import (
    build_prompt_state,
    generate_from_prompt_state,
    load_prompt_state,
    prompt_cache_key,
    save_prompt_state,
)
from trinote.infer_int.reference_bonsai35 import (
    BonsaiQwen35ReferenceModel,
    random_bonsai35_artifact,
)


def _greedy(row, _position, _history):
    return int(np.asarray(row).argmax())


def test_prompt_cache_roundtrip_continues_byte_identically(tmp_path):
    artifact = random_bonsai35_artifact(seed=41)
    digest = "ab" * 32
    prompt = [2, 7, 3]
    expected_model = BonsaiQwen35ReferenceModel(artifact)
    expected = expected_model.generate_greedy_tokens_cached(prompt, 4)

    producer = BonsaiQwen35ReferenceModel(artifact)
    state = build_prompt_state(producer, prompt, digest)
    path = save_prompt_state(state, artifact, tmp_path / "prefix.safetensors")
    loaded = load_prompt_state(path, artifact, digest)
    actual = generate_from_prompt_state(producer, loaded, 4, _greedy)
    assert actual == expected
    assert loaded.cache.t == len(prompt) + 4
    assert loaded.input_ids == tuple(prompt + expected)

    one_shot = load_prompt_state(path, artifact, digest)
    assert generate_from_prompt_state(
        producer, one_shot, 4, _greedy, keep_reusable=False
    ) == expected
    assert one_shot.cache.t == len(prompt) + 3
    assert len(one_shot.input_ids) == one_shot.cache.t


def test_prompt_cache_key_binds_artifact_and_exact_ids():
    key = prompt_cache_key("aa" * 32, [1, 2, 3])
    assert key != prompt_cache_key("bb" * 32, [1, 2, 3])
    assert key != prompt_cache_key("aa" * 32, [1, 3, 2])


def test_prompt_cache_rejects_wrong_artifact_and_tampered_tensor(tmp_path):
    artifact = random_bonsai35_artifact(seed=42)
    digest = "cd" * 32
    state = build_prompt_state(BonsaiQwen35ReferenceModel(artifact), [1, 4], digest)
    path = save_prompt_state(state, artifact, tmp_path / "prefix.safetensors")
    with pytest.raises(ValueError, match="artifact digest"):
        load_prompt_state(path, artifact, "ef" * 32)

    tensors = {}
    with safe_open(str(path), framework="numpy") as f:
        meta = f.metadata()
        for name in f.keys():
            tensors[name] = f.get_tensor(name)
    tensors["last_x"] = tensors["last_x"].copy()
    tensors["last_x"][0, 0] += 1
    bad = tmp_path / "tampered.safetensors"
    save_file(tensors, str(bad), metadata=meta)
    with pytest.raises(ValueError, match="commitment"):
        load_prompt_state(bad, artifact, digest)


def test_prompt_cache_rejects_empty_prompt():
    artifact = random_bonsai35_artifact(seed=43)
    with pytest.raises(ValueError, match="empty"):
        build_prompt_state(BonsaiQwen35ReferenceModel(artifact), [], "aa" * 32)
