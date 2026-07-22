"""CPU-only tests for the auditable real-model retained-layer gate."""
from __future__ import annotations

import json
import stat
from types import SimpleNamespace

import numpy as np
import pytest

from tools import gate_resident_layers as gate


def _trial_evidence(*, rate=10.0, allocations=(3, 0, 0), workspace=4096):
    return {
        "warm_tokens_per_second": rate,
        "transitions": [
            {
                "native_cuda": {"allocation_calls": value},
                "request_workspace_bytes_before": workspace,
                "request_workspace_bytes_after": workspace,
            }
            for value in allocations
        ],
    }


def _gpu(*, peak=20000, total=24576, apps=None, errors=None):
    return {
        "peak_memory_used_mib": peak,
        "device": {"memory_total_mib": total},
        "initial_compute_apps": [] if apps is None else apps,
        "sample_count": 10,
        "sampler_errors": [] if errors is None else errors,
    }


def test_atomic_report_is_strict_private_json(tmp_path):
    path = tmp_path / "gate.json"
    path.write_text("old")
    gate.atomic_write_json(path, {"status": "passed", "value": 3})
    assert json.loads(path.read_text()) == {"status": "passed", "value": 3}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".gate.json.*.tmp"))
    with pytest.raises(ValueError):
        gate.atomic_write_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text())["status"] == "passed"


def test_array_evidence_and_mismatch_are_exact_and_bounded():
    left = np.array([[1, -2, 3]], dtype=np.int64)
    same = gate._array_comparison(left, left.copy())
    assert same["exact"] and same["mismatch_count"] == 0
    right = np.array([[1, -9, 4]], dtype=np.int64)
    different = gate._array_comparison(left, right)
    assert not different["exact"]
    assert different["mismatch_count"] == 2
    assert different["first_mismatch"] == {"flat_index": 1, "expected": -2, "actual": -9}
    assert different["expected"]["sha256"] != different["actual"]["sha256"]


def test_counter_delta_fails_on_reset_or_schema_change():
    assert gate._counter_delta({"a": 2}, {"a": 5}) == {"a": 3}
    with pytest.raises(RuntimeError, match="keys changed"):
        gate._counter_delta({"a": 2}, {"b": 5})
    with pytest.raises(RuntimeError, match="backwards"):
        gate._counter_delta({"a": 5}, {"a": 2})


def test_source_identity_binds_deployment_snapshot(monkeypatch, tmp_path):
    commit = "1" * 40
    snapshot = tmp_path / "source-snapshot.json"
    snapshot.write_text(json.dumps({
        "format": "trinote-integer-speed-source-snapshot/v1",
        "baseCommit": commit,
        "snapshotDigest": "2" * 64,
        "entryCount": 17,
        "contentBytes": 1234,
    }))

    def fake_run(command, **_kwargs):
        if command[1:3] == ["rev-parse", "HEAD"]:
            return commit + "\n"
        return ""

    monkeypatch.setattr(gate, "_run_text", fake_run)
    monkeypatch.setenv("TRINOTE_SOURCE_SNAPSHOT", str(snapshot))
    identity = gate._source_identity(tmp_path)
    assert identity["git_commit"] == commit
    assert identity["snapshot"]["snapshot_digest"] == "2" * 64
    assert identity["snapshot"]["entry_count"] == 17


def test_gpu_csv_parsers_preserve_auditable_device_identity():
    rows = gate._parse_gpu_rows("0, GPU-abc, NVIDIA RTX 4090, 21, 24564, 590.48\n")
    assert rows == [{
        "index": 0, "uuid": "GPU-abc", "name": "NVIDIA RTX 4090",
        "memory_used_mib": 21, "memory_total_mib": 24564,
        "driver_version": "590.48",
    }]
    assert gate._parse_compute_rows("GPU-abc, 123, 2048\n") == [
        {"uuid": "GPU-abc", "pid": 123, "memory_used_mib": 2048}
    ]
    with pytest.raises(RuntimeError, match="unexpected"):
        gate._parse_gpu_rows("0, too-short\n")


def test_gate_verdict_requires_every_promotion_condition():
    result = gate.evaluate_gate(
        parity={"exact": True},
        baseline=_trial_evidence(rate=10.0),
        candidate=_trial_evidence(rate=9.6),
        gpu=_gpu(), max_throughput_regression=0.05, max_memory_fraction=0.90,
    )
    assert result["passed"]
    assert all(result["checks"].values())
    assert result["measurements"]["warm_candidate_transition_count"] == 2


@pytest.mark.parametrize(
    "change,check",
    [
        ({"parity": {"exact": False}}, "exact_hidden_logit_token_parity"),
        ({"candidate": _trial_evidence(allocations=(3, 1, 0))}, "warm_allocation_stability"),
        ({"candidate": _trial_evidence(rate=9.4)}, "warm_throughput_no_regression"),
        ({"gpu": _gpu(peak=23000, total=24576)}, "combined_peak_below_limit"),
        ({"gpu": _gpu(apps=[{"pid": 4}])}, "isolated_gpu_at_start"),
        ({"gpu": _gpu(errors=["sample failed"])}, "gpu_sampling_complete"),
    ],
)
def test_gate_verdict_fails_closed(change, check):
    values = {
        "parity": {"exact": True},
        "baseline": _trial_evidence(rate=10.0),
        "candidate": _trial_evidence(rate=9.6),
        "gpu": _gpu(),
    }
    values.update(change)
    result = gate.evaluate_gate(
        **values, max_throughput_regression=0.05, max_memory_fraction=0.90,
    )
    assert not result["passed"]
    assert not result["checks"][check]


def test_parity_evidence_compares_hidden_logits_and_tokens():
    def trial(token, hidden, logits):
        state = gate.StepState(np.array(hidden), np.array(logits), token, 1)
        return gate.Trial("x", [state], [], {}, [token], 1, 2)

    exact = gate._parity_evidence(trial(2, [[1]], [3, 4]), trial(2, [[1]], [3, 4]))
    assert exact["exact"]
    wrong = gate._parity_evidence(trial(2, [[1]], [3, 4]), trial(1, [[9]], [3, 5]))
    assert not wrong["exact"]
    assert not wrong["steps"][0]["token_exact"]
    assert not wrong["steps"][0]["hidden"]["exact"]
    assert not wrong["steps"][0]["logits"]["exact"]


def test_runtime_gate_refuses_fallback_and_wrong_architecture(monkeypatch):
    monkeypatch.setattr(gate.qk_cuda, "resident_layer_available", lambda: True)
    engine = SimpleNamespace(
        bname="cpu", resident=False, resident_attention=True, DENSE=1, NL=49,
    )
    with pytest.raises(RuntimeError, match="cuda-resident"):
        gate._require_runtime(engine)
    engine.bname, engine.resident, engine.DENSE, engine.NL = "cuda-resident", True, 2, 49
    with pytest.raises(RuntimeError, match="1 dense . 48 MoE"):
        gate._require_runtime(engine)


def test_cli_requires_enough_warm_tokens_and_canonical_expected_hash(tmp_path):
    with pytest.raises(SystemExit):
        gate.parse_args(["model.gguf", "--output", str(tmp_path / "x"), "--new-tokens", "3"])
    with pytest.raises(SystemExit):
        gate.parse_args([
            "model.gguf", "--output", str(tmp_path / "x"),
            "--expected-model-sha256", "A" * 64,
        ])


def test_main_publishes_atomic_failure_before_gpu_or_model_load(tmp_path):
    output = tmp_path / "failure.json"
    rc = gate.main([str(tmp_path / "missing.gguf"), "--output", str(output)])
    evidence = json.loads(output.read_text())
    assert rc == 1
    assert evidence["status"] == "error"
    assert evidence["error"]["type"] == "FileNotFoundError"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
