"""CPU-only parser and serialization tests for the real CUDA acceptance tool."""
from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "verify_bonsai35_gpu_real.py"


def _module():
    spec = importlib.util.spec_from_file_location("verify_bonsai35_gpu_real", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_real_parser_defaults_bind_release(monkeypatch, tmp_path):
    module = _module()
    artifact = tmp_path / "release.safetensors"
    monkeypatch.setenv("BONSAI_INTEGER_27B_ARTIFACT", str(artifact))
    args = module.parse_args([])
    report = module._base_report(args)
    assert args.artifact == artifact
    assert args.generated_tokens == 128
    assert args.throughput_context is None
    assert report["mode"] == "full_cpu_oracle_parity"
    assert report["request"] == {
        "prompt_name": "rawHi",
        "prompt_ids": [12675],
        "generated_tokens": 128,
        "consumed_tokens": 128,
    }
    assert report["artifact"]["required_sha256"] == module.RELEASE_ARTIFACT_SHA256


def test_gpu_real_parser_selects_populated_4k_mode():
    module = _module()
    args = module.parse_args(["--throughput-context", "4096"])
    report = module._base_report(args)
    assert args.throughput_context == 4096
    assert module.target_consumed_tokens(args) == 4096
    assert report["mode"] == "gpu_populated_context_throughput"
    assert report["request"]["generated_tokens"] == 4096
    assert report["request"]["consumed_tokens"] == 4096


def test_gpu_real_parser_keeps_modes_separate():
    module = _module()
    with pytest.raises(SystemExit):
        module.parse_args([
            "--throughput-context", "4096", "--generated-tokens", "64"
        ])


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_gpu_real_parser_rejects_nonpositive_counts(value):
    module = _module()
    with pytest.raises(SystemExit):
        module.parse_args(["--generated-tokens", value])


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_gpu_real_parser_rejects_bad_throughput_context(value):
    module = _module()
    with pytest.raises(SystemExit):
        module.parse_args(["--throughput-context", value])


def test_gpu_real_array_parity_hashes_shape_dtype_and_mismatch():
    module = _module()
    cpu = np.arange(12, dtype=np.int64).reshape(3, 4)
    same = module.array_parity_record(cpu, cpu.copy())
    assert same["equal"]
    assert same["cpu"] == same["gpu"]
    assert same["cpu"]["shape"] == [3, 4]
    assert same["cpu"]["dtype"] == "int64"
    changed = cpu.copy(); changed[2, 3] += 1
    different = module.array_parity_record(cpu, changed)
    assert not different["equal"]
    assert different["cpu"]["sha256"] != different["gpu"]["sha256"]


def test_gpu_real_atomic_json_output_is_exact_and_leaves_no_temp(tmp_path):
    module = _module()
    target = tmp_path / "nested" / "result.json"
    value = {"z": [3, 2, 1], "a": "café"}
    payload = module.canonical_json_bytes(value)
    module.atomic_write(target, payload)
    assert target.read_bytes() == payload
    assert json.loads(target.read_text()) == value
    assert list(target.parent.iterdir()) == [target]


def test_gpu_real_runtime_identity_binds_exact_cuda_bytes(monkeypatch, tmp_path):
    module = _module()
    library = tmp_path / "libbonsai_q1_gpu.so"
    library.write_bytes(b"exact-cuda-elf-bytes")

    def command_output(command):
        if command[-2:] == ["rev-parse", "HEAD"]:
            return "ab" * 20
        if command[-2:] == ["status", "--porcelain=v1"]:
            return " M tools/bonsai_q1_gpu.cu"
        if command[:2] == ["readelf", "-n"]:
            return "Build ID: CAFE1234"
        if command[:3] == ["readelf", "-p", ".comment"]:
            return "nvcc test compiler"
        return None

    monkeypatch.setattr(module, "_command_output", command_output)
    identity = module.build_runtime_identity(library)
    assert identity["source"]["revision"] == "ab" * 20
    assert identity["source"]["dirty"] is True
    assert identity["source"]["porcelain_entry_count"] == 1
    assert len(identity["source"]["acceptance_tool_sha256"]) == 64
    assert identity["cuda_kernel"] == {
        "path": str(library.resolve()),
        "present": True,
        "sha256": module.sha256_file(library),
        "size_bytes": len(b"exact-cuda-elf-bytes"),
        "elf_build_id": "cafe1234",
        "compiler_comment": "nvcc test compiler",
    }


def test_gpu_real_timing_windows_and_sequence_hash_are_stable():
    module = _module()
    samples = [float(i) for i in range(1, 130)]
    assert module.median_window(samples, 2, 32) == 17.5
    assert module.median_window(samples, 32, 128) == 80.5
    assert module.median_window(samples, 200, 300) is None
    assert module.int64_sequence_sha256([1, 2, 3]) == module.int64_sequence_sha256(
        np.asarray([1, 2, 3], dtype=np.int64)
    )


def test_gpu_real_throughput_summary_and_accounting_gates():
    module = _module()
    target = 40
    generated = [11] + list(range(1, target))
    timings = [0.2] * 8 + [0.05] * 32
    stats = {
        "graph_launches": target,
        "position": target,
        "graph_ready": True,
        "poisoned": False,
        "input_mode": "token_id",
        "token_input_submissions": target,
        "embedded_input_submissions": 0,
        "model_input_host_bytes": target * 8,
    }
    memory = {
        "after_create": {"used_bytes": 7_100_000_000},
        "after_first_graph": {"used_bytes": 7_200_000_000},
        "at_target": {"used_bytes": 7_150_000_000},
    }
    summary = module.build_throughput_summary(
        target_context=target,
        generated_ids=generated,
        step_seconds=timings,
        stats=stats,
        memory_samples=memory,
        proof_peak_used_bytes=7_748_780_032,
        ceiling_bytes=8_053_063_680,
    )
    assert summary["timing"]["last_32_decode_seconds"] == [0.05] * 32
    assert summary["timing"]["last_32_median_tokens_per_second"] == 20.0
    assert summary["memory"]["observed_peak_used_bytes"] == 7_200_000_000
    assert summary["memory"]["live_at_target_used_bytes"] == 7_150_000_000
    assert summary["consumed_ids"]["count"] == target
    assert summary["consumed_ids"]["first_8"][:2] == [12675, 11]
    assert all(module.throughput_acceptance(summary).values())

    drifted = dict(summary)
    drifted["stats"] = {**stats, "graph_launches": target - 1, "poisoned": True}
    gates = module.throughput_acceptance(drifted)
    assert not gates["one_graph_submission_per_consumed_token"]
    assert not gates["context_not_poisoned"]

    host_embedded = dict(summary)
    host_embedded["stats"] = {
        **stats,
        "input_mode": "embedded_row",
        "token_input_submissions": 0,
        "embedded_input_submissions": target,
        "model_input_host_bytes": target * 5120 * 8,
    }
    gates = module.throughput_acceptance(host_embedded)
    assert not gates["token_id_input_mode"]
    assert not gates["device_embedding_only"]
    assert not gates["model_input_host_bytes_exact"]

    slow = dict(summary)
    slow["timing"] = {
        **summary["timing"],
        "last_32_median_tokens_per_second": 9.99,
    }
    assert not module.throughput_acceptance(slow)[
        "last_32_at_least_10_tokens_per_second"
    ]


def test_cpu_oracle_trace_is_pure_by_default():
    from trinote.infer_int.gpu_bonsai35 import cpu_oracle_trace

    signature = inspect.signature(cpu_oracle_trace)
    assert signature.parameters["accelerated_native"].default is False
