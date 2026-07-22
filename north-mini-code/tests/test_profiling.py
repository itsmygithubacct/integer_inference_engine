"""CPU-only gates for structured cold/warm inference telemetry."""
import pytest

from nmc.profiling import InferenceProfiler


def test_profile_separates_cold_and_warm_and_preserves_native_snapshot():
    profile = InferenceProfiler(enabled=True)
    with profile.phase("registration.weight", bucket="cold"):
        pass
    profile.mark_warm()
    with profile.phase("attention"):
        pass
    profile.record("projection.q", wall_ns=17, calls=2)
    native = {"h2d_calls": 1, "h2d_bytes": 2048, "allocation_calls": 0}
    snapshot = profile.snapshot(native)

    assert snapshot["schema_version"] == 1
    assert snapshot["current_bucket"] == "warm"
    assert snapshot["phases"]["cold"]["registration.weight"]["calls"] == 1
    assert snapshot["phases"]["warm"]["attention"]["calls"] == 1
    assert snapshot["phases"]["warm"]["projection.q"] == {"calls": 2, "wall_ns": 17, "errors": 0}
    assert snapshot["native_cuda"] == native
    native["h2d_calls"] = 99
    assert snapshot["native_cuda"]["h2d_calls"] == 1       # handoff snapshots are immutable copies


def test_profile_records_failure_without_swallowing_it():
    profile = InferenceProfiler(enabled=True)
    with pytest.raises(RuntimeError, match="boom"):
        with profile.phase("moe"):
            raise RuntimeError("boom")
    row = profile.snapshot()["phases"]["cold"]["moe"]
    assert row["calls"] == row["errors"] == 1


def test_disabled_profile_is_semantically_inert():
    profile = InferenceProfiler(enabled=False)
    with profile.phase("attention"):
        pass
    profile.record("routing", wall_ns=1)
    assert profile.snapshot()["phases"] == {"cold": {}, "warm": {}}


def test_profile_validates_manual_records():
    profile = InferenceProfiler(enabled=True)
    with pytest.raises(ValueError):
        profile.record("", wall_ns=1)
    with pytest.raises(ValueError):
        profile.record("attention", wall_ns=-1)
    with pytest.raises(ValueError):
        profile.record("attention", bucket="later")
