"""Focused allocation, growth, and lifecycle gates for the reusable KV cache."""

import numpy as np
import pytest

from nmc import cohere2 as c2


def _rows(start, count, *, heads=2, width=3, dtype=np.int64):
    values = np.arange(start, start + heads * count * width, dtype=dtype)
    return values.reshape(heads, count, width)


def test_append_preserves_content_and_reuses_capacity():
    cache = c2.KVCache(2, initial_capacity=4)
    k0, v0 = _rows(0, 1), _rows(100, 1)
    cache.append(0, k0, v0)
    pointer = cache.k[0].__array_interface__["data"][0]
    first_view = cache.k[0]

    k1, v1 = _rows(10, 2), _rows(110, 2)
    cache.append(0, k1, v1)

    assert cache.length(0) == 3
    assert cache.capacity(0) == 4
    assert cache.k[0].__array_interface__["data"][0] == pointer
    assert np.shares_memory(first_view, cache.k[0])
    assert np.array_equal(cache.k[0], np.concatenate((k0, k1), axis=1))
    assert np.array_equal(cache.v[0], np.concatenate((v0, v1), axis=1))
    assert cache.length(1) == 0 and cache.k[1] is None and cache.v[1] is None


def test_growth_is_geometric_and_keeps_existing_rows():
    cache = c2.KVCache(1, initial_capacity=2)
    first_k, first_v = _rows(0, 2), _rows(100, 2)
    cache.append(0, first_k, first_v)
    first_pointer = cache.k[0].__array_interface__["data"][0]

    second_k, second_v = _rows(20, 1), _rows(120, 1)
    cache.append(0, second_k, second_v)
    assert cache.capacity() == 3
    assert cache.k[0].__array_interface__["data"][0] != first_pointer

    third_k, third_v = _rows(30, 2), _rows(130, 2)
    cache.append(0, third_k, third_v)
    assert cache.length() == 5
    assert cache.capacity() == 6
    assert np.array_equal(cache.k[0], np.concatenate((first_k, second_k, third_k), axis=1))
    assert np.array_equal(cache.v[0], np.concatenate((first_v, second_v, third_v), axis=1))


def test_reset_reuses_storage_and_release_drops_it():
    cache = c2.KVCache(1, initial_capacity=8)
    cache.append(0, _rows(0, 3), _rows(100, 3))
    pointer = cache.k[0].__array_interface__["data"][0]

    cache.reset()
    assert cache.length() == 0
    assert cache.capacity() == 8
    assert cache.k[0].shape == (2, 0, 3)

    replacement_k, replacement_v = _rows(50, 2), _rows(150, 2)
    cache.append(0, replacement_k, replacement_v)
    assert cache.k[0].__array_interface__["data"][0] == pointer
    assert np.array_equal(cache.k[0], replacement_k)
    assert np.array_equal(cache.v[0], replacement_v)

    cache.reset(release=True)
    assert cache.length() == cache.capacity() == 0
    assert cache.k[0] is None and cache.v[0] is None


def test_max_length_failure_is_atomic():
    cache = c2.KVCache(1, max_length=3)
    cache.append(0, _rows(0, 2), _rows(100, 2))
    before_k, before_v = cache.k[0].copy(), cache.v[0].copy()

    with pytest.raises(ValueError, match="exceeds max_length 3"):
        cache.append(0, _rows(20, 2), _rows(120, 2))

    assert cache.length() == 2
    assert np.array_equal(cache.k[0], before_k)
    assert np.array_equal(cache.v[0], before_v)


@pytest.mark.parametrize(
    "k_new,v_new,message",
    [
        (np.zeros((2, 3)), np.zeros((2, 3)), "must both have shape"),
        (np.zeros((2, 1, 3)), np.zeros((2, 2, 3)), "shapes differ"),
    ],
)
def test_append_rejects_invalid_shapes(k_new, v_new, message):
    cache = c2.KVCache(1)
    with pytest.raises(ValueError, match=message):
        cache.append(0, k_new, v_new)
    assert cache.length() == 0


def test_append_rejects_outer_dimension_change():
    cache = c2.KVCache(1)
    cache.append(0, _rows(0, 1), _rows(100, 1))
    with pytest.raises(ValueError, match="do not match cached dimensions"):
        cache.append(0, _rows(10, 1, width=4), _rows(110, 1, width=4))
