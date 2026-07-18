from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


BONSAI_ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    path = BONSAI_ROOT / "tools" / "compare_bonsai35_benchmarks.py"
    spec = importlib.util.spec_from_file_location("compare_bonsai35_benchmarks", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _summary(median: float) -> dict:
    return {
        "n": 5,
        "median": median,
        "p10": median * 0.98,
        "p90": median * 1.02,
        "mean": median,
        "coefficient_of_variation": 0.01,
        "min": median * 0.97,
        "max": median * 1.03,
    }


def _benchmark(producer: str) -> dict:
    output_hash = "c" * 64
    cache_hash = "d" * 64
    output_ids = list(range(128))
    iterations = [{
        "input_ids": [12675],
        "output_ids": output_ids,
        "output_token_count": 128,
        "commitments": {
            "output_ids_sha256": output_hash,
            "cache_state_sha256": cache_hash,
            "layer_trace_sha256": cache_hash,
        },
    } for _ in range(5)]
    thread_environment = {
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "8",
        "OMP_DYNAMIC": "FALSE",
        "OMP_WAIT_POLICY": "PASSIVE",
        "OMP_PLACES": "cores",
        "OMP_PROC_BIND": "close",
        "OMP_MAX_ACTIVE_LEVELS": "1",
        "GOMP_SPINCOUNT": "0",
        "KMP_BLOCKTIME": "0",
        "TRINOTE_BONSAI35_Q1_CHUNK": "8",
        "TRINOTE_Q1_LUT32": "0" if producer == "legacy-native" else "",
    }
    metrics = {
        "prompt_prefill_s": _summary(30.0 if producer == "legacy-native" else 10.0),
        "time_to_first_output_token_compute_s": _summary(
            36.0 if producer == "legacy-native" else 12.0
        ),
        "steady_decode_token_3_32_median_s": _summary(
            4.0 if producer == "legacy-native" else 1.0
        ),
        "steady_decode_token_33_128_median_s": _summary(
            5.0 if producer == "legacy-native" else 1.0
        ),
        "generation_compute_s": _summary(
            640.0 if producer == "legacy-native" else 128.0
        ),
    }
    variation_names = (
        "prompt_prefill_s",
        "steady_decode_token_3_32_median_s",
        "steady_decode_token_33_128_median_s",
        "generation_compute_s",
    )
    return {
        "format": "trinote-bonsai35-benchmark/1",
        "mode": "model",
        "condition": "second-turn",
        "accepted": True,
        "rejection_reasons": [],
        "control": {
            "repetitions": 5,
            "busy_override": False,
            "core_mode": "physical",
            "requested_affinity": list(range(8)),
            "effective_affinity": list(range(8)),
            "thread_environment": thread_environment,
            "prefill_q1_chunk": 8,
        },
        "configuration": {
            "producer": producer,
            "raw_ids": "12675",
            "prompt": "Hi",
            "chat": False,
            "max_new": 128,
            "sampler": "greedy",
            "seed": 0,
            "ignore_eos": True,
        },
        "workers": [{
            "producer": producer,
            "artifact": {"sha256": "a" * 64, "size_bytes": 4_226_001_568},
            "environment": {
                "source": {
                    "revision": "1" * 40,
                    "dirty": True,
                    "porcelain_entry_count": 91,
                    "porcelain_sha256": "2" * 64,
                    "files": {
                        relative: {"sha256": "4" * 64, "size_bytes": 1234}
                        for relative in _load_tool().REQUIRED_SOURCE_FILES
                    },
                },
                "kernel": {
                    "present": True,
                    "sha256": "b" * 64,
                    "elf_build_id": "3" * 40,
                    "compiler_comment": "GCC synthetic",
                },
            },
            "iterations": iterations,
        }],
        "aggregate": {
            "iteration_count": 5,
            "metrics": metrics,
            "variation_exit_gate": {
                "minimum_samples": 5,
                "eligible": True,
                "passed": True,
                "sample_counts": {name: 5 for name in variation_names},
                "values": {name: 0.01 for name in variation_names},
            },
            "commitment_consistency_gate": {
                "applicable": True,
                "eligible": True,
                "iteration_count": 5,
                "complete_record_count": 5,
                "output_ids_sha256_values": [output_hash],
                "cache_state_sha256_values": [cache_hash],
                "output_token_count_values": [128],
                "output_ids_sha256_identical": True,
                "cache_state_sha256_identical": True,
                "output_token_count_identical": True,
                "passed": True,
            },
        },
        "hardware_counters": {
            "requested": True,
            "available": False,
            "reason": "perf executable is not installed",
            "events": "cycles,instructions",
            "runs": [None],
        },
    }


def _set_all_output_hashes(run: dict, value: str) -> None:
    for iteration in run["workers"][0]["iterations"]:
        iteration["commitments"]["output_ids_sha256"] = value
    run["aggregate"]["commitment_consistency_gate"][
        "output_ids_sha256_values"
    ] = [value]


def test_comparison_passes_and_preserves_raw_metrics_and_pmu_reason():
    tool = _load_tool()
    legacy, native = _benchmark("legacy-native"), _benchmark("native")
    report = tool.compare_runs(
        legacy, native, minimum_steady_speedup=4.0, minimum_prompt_speedup=3.0
    )
    assert report["status"] == "pass"
    assert report["metrics"]["prefill"]["speedup"] == 3.0
    assert report["metrics"]["ttft"]["speedup"] == 3.0
    assert report["metrics"]["token_3_32"]["speedup"] == 4.0
    assert report["metrics"]["token_33_128"]["speedup"] == 5.0
    assert report["metrics"]["prefill"]["legacy_summary"] == (
        legacy["aggregate"]["metrics"]["prompt_prefill_s"]
    )
    assert report["commitments"]["cross_producer_equal"] is True
    assert report["hardware_counters"]["legacy"]["reason"] == (
        "perf executable is not installed"
    )


def _wrong_role(legacy, native):
    native["configuration"]["producer"] = "legacy-native"


def _rejected(legacy, native):
    legacy["accepted"] = False


def _busy_override(legacy, native):
    legacy["control"]["busy_override"] = True


def _too_few_repetitions(legacy, native):
    legacy["control"]["repetitions"] = 4


def _failed_variation(legacy, native):
    legacy["aggregate"]["variation_exit_gate"]["passed"] = False


def _failed_within_run_commitment(legacy, native):
    legacy["aggregate"]["commitment_consistency_gate"]["passed"] = False


def _artifact_mismatch(legacy, native):
    native["workers"][0]["artifact"]["sha256"] = "e" * 64


def _source_mismatch(legacy, native):
    native["workers"][0]["environment"]["source"]["porcelain_sha256"] = "e" * 64


def _source_file_mismatch(legacy, native):
    files = native["workers"][0]["environment"]["source"]["files"]
    files["tools/bonsai_q1_kernel.c"]["sha256"] = "e" * 64


def _kernel_mismatch(legacy, native):
    native["workers"][0]["environment"]["kernel"]["sha256"] = "e" * 64


def _thread_mismatch(legacy, native):
    native["control"]["thread_environment"]["OMP_NUM_THREADS"] = "7"


def _workload_mismatch(legacy, native):
    native["configuration"]["max_new"] = 127


def _cross_commitment_mismatch(legacy, native):
    _set_all_output_hashes(native, "e" * 64)


def _missing_steady_metric(legacy, native):
    del native["aggregate"]["metrics"]["steady_decode_token_33_128_median_s"]


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (_wrong_role, "producer_role"),
        (_rejected, "acceptance"),
        (_busy_override, "acceptance"),
        (_too_few_repetitions, "repetitions"),
        (_failed_variation, "variation"),
        (_failed_within_run_commitment, "commitment_consistency"),
        (_artifact_mismatch, "artifact_identity_mismatch"),
        (_source_mismatch, "source_identity_mismatch"),
        (_source_file_mismatch, "source_identity_mismatch"),
        (_kernel_mismatch, "kernel_identity_mismatch"),
        (_thread_mismatch, "thread_identity_mismatch"),
        (_workload_mismatch, "workload_identity_mismatch"),
        (_cross_commitment_mismatch, "cross_commitments"),
        (_missing_steady_metric, "metric_missing"),
    ],
)
def test_comparison_rejects_major_provenance_and_consistency_failures(mutation, code):
    tool = _load_tool()
    legacy, native = _benchmark("legacy-native"), _benchmark("native")
    mutation(legacy, native)
    with pytest.raises(tool.ComparisonError) as error:
        tool.compare_runs(legacy, native)
    assert error.value.code == code


def test_comparison_rejects_steady_and_requested_prompt_thresholds():
    tool = _load_tool()
    legacy, native = _benchmark("legacy-native"), _benchmark("native")
    native["aggregate"]["metrics"]["steady_decode_token_33_128_median_s"] = (
        _summary(2.0)
    )
    with pytest.raises(tool.ComparisonError) as steady:
        tool.compare_runs(legacy, native)
    assert steady.value.code == "steady_threshold"
    assert steady.value.details["metrics"]["token_33_128"]["speedup"] == 2.5
    assert not steady.value.details["thresholds"]["steady"]["passed"]

    legacy, native = _benchmark("legacy-native"), _benchmark("native")
    native["aggregate"]["metrics"]["prompt_prefill_s"] = _summary(15.0)
    with pytest.raises(tool.ComparisonError) as prompt:
        tool.compare_runs(legacy, native, minimum_prompt_speedup=3.0)
    assert prompt.value.code == "prompt_threshold"


def test_cli_writes_exact_atomic_comparison_with_input_hashes(tmp_path, capsys):
    tool = _load_tool()
    legacy_path = tmp_path / "legacy.json"
    native_path = tmp_path / "native.json"
    output_path = tmp_path / "comparison.json"
    legacy_bytes = (json.dumps(_benchmark("legacy-native"), sort_keys=True) + "\n").encode()
    native_bytes = (json.dumps(_benchmark("native"), sort_keys=True) + "\n").encode()
    legacy_path.write_bytes(legacy_bytes)
    native_path.write_bytes(native_bytes)

    assert tool.main([
        "--legacy", str(legacy_path),
        "--native", str(native_path),
        "--json-out", str(output_path),
        "--require-prompt-speedup",
    ]) == 0
    stdout = capsys.readouterr().out.encode()
    assert output_path.read_bytes() == stdout
    result = json.loads(stdout)
    assert result["inputs"]["legacy"]["sha256"] == hashlib.sha256(legacy_bytes).hexdigest()
    assert result["inputs"]["native"]["sha256"] == hashlib.sha256(native_bytes).hexdigest()
    assert result["metrics"]["token_33_128"]["native_summary"]["median"] == 1.0
    assert not list(tmp_path.glob(".comparison.json.*.tmp"))


def test_cli_atomically_emits_machine_readable_rejection(tmp_path, capsys):
    tool = _load_tool()
    legacy = _benchmark("legacy-native")
    native = _benchmark("native")
    native["accepted"] = False
    legacy_path = tmp_path / "legacy.json"
    native_path = tmp_path / "native.json"
    output_path = tmp_path / "comparison.json"
    legacy_path.write_text(json.dumps(legacy))
    native_path.write_text(json.dumps(native))

    assert tool.main([
        "--legacy", str(legacy_path),
        "--native", str(native_path),
        "--json-out", str(output_path),
    ]) == 2
    stdout = capsys.readouterr().out.encode()
    assert output_path.read_bytes() == stdout
    result = json.loads(stdout)
    assert result["status"] == "fail"
    assert result["error"]["code"] == "acceptance"
    assert result["hardware_counters"]["native"]["reason"] == (
        "perf executable is not installed"
    )
