from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np


BONSAI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BONSAI_ROOT.parent


def _load_tool(name: str):
    path = BONSAI_ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_benchmark_summary_reports_quantiles_and_variation():
    bench = _load_tool("bench_bonsai35")
    summary = bench._metric_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["n"] == 5
    assert summary["median"] == 3.0
    assert summary["p10"] == 1.4
    assert summary["p90"] == 4.6
    assert summary["coefficient_of_variation"] > 0
    one_run = bench._aggregate([{
        "iterations": [{"timing": {"prompt_prefill_s": 1.0, "generation_compute_s": 2.0}}]
    }])
    gate = one_run["variation_exit_gate"]
    assert gate["minimum_samples"] == 5
    assert not gate["eligible"]
    assert not gate["passed"]


def test_benchmark_git_identity_commits_dirty_porcelain(monkeypatch):
    bench = _load_tool("bench_bonsai35")
    porcelain = " M tools/bench_bonsai35.py\n?? tests/new test.py"

    def command_output(command, timeout=10.0):
        del timeout
        if command[-2:] == ["rev-parse", "HEAD"]:
            return "a" * 40
        assert command[-2:] == ["status", "--porcelain=v1"]
        return porcelain

    monkeypatch.setattr(bench, "_command_output", command_output)
    identity = bench._git_identity(Path("/checkout"))
    assert identity == {
        "revision": "a" * 40,
        "dirty": True,
        "porcelain_entry_count": 2,
        "porcelain_sha256": hashlib.sha256(porcelain.encode("utf-8")).hexdigest(),
    }


def test_benchmark_resource_record_has_current_and_peak_rss(tmp_path):
    bench = _load_tool("bench_bonsai35")
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmRSS:\t12345 kB\n")
    assert bench._current_rss_kib(status) == 12345

    before_usage = SimpleNamespace(
        ru_utime=1.0, ru_stime=2.0, ru_maxrss=40000, ru_minflt=10,
        ru_majflt=2, ru_nvcsw=3, ru_nivcsw=4,
    )
    after_usage = SimpleNamespace(
        ru_utime=2.0, ru_stime=2.5, ru_maxrss=50000, ru_minflt=14,
        ru_majflt=3, ru_nvcsw=8, ru_nivcsw=6,
    )
    record = bench._usage_delta(
        (before_usage, 10.0, 7.0, 12000),
        (after_usage, 12.0, 9.0, 15000),
    )
    assert record["peak_rss_kib"] == 50000
    assert record["current_rss_kib_before"] == 12000
    assert record["current_rss_kib_after"] == 15000


def test_benchmark_aggregate_commitment_consistency_gate():
    bench = _load_tool("bench_bonsai35")

    def iteration(output_hash="o" * 64, cache_hash="c" * 64, count=128):
        return {
            "timing": {"prompt_prefill_s": 1.0, "generation_compute_s": 2.0},
            "commitments": {
                "output_ids_sha256": output_hash,
                "cache_state_sha256": cache_hash,
            },
            "output_token_count": count,
        }

    passed = bench._aggregate([{"iterations": [iteration() for _ in range(5)]}])[
        "commitment_consistency_gate"
    ]
    assert passed["applicable"] and passed["eligible"] and passed["passed"]
    assert passed["complete_record_count"] == 5
    assert passed["output_ids_sha256_values"] == ["o" * 64]
    assert passed["cache_state_sha256_values"] == ["c" * 64]
    assert passed["output_token_count_values"] == [128]

    failed = bench._aggregate([{
        "iterations": [
            iteration(),
            iteration(output_hash="x" * 64),
            iteration(cache_hash="y" * 64),
            iteration(count=127),
        ]
    }])["commitment_consistency_gate"]
    assert failed["eligible"] and not failed["passed"]
    assert not failed["output_ids_sha256_identical"]
    assert not failed["cache_state_sha256_identical"]
    assert not failed["output_token_count_identical"]

    q1 = bench._aggregate([{"iterations": [{"timing": {"median_s": 0.1}}]}])[
        "commitment_consistency_gate"
    ]
    assert not q1["applicable"] and not q1["eligible"] and q1["passed"] is None


def test_benchmark_perf_parser_preserves_unavailable_counters():
    bench = _load_tool("bench_bonsai35")
    parsed = bench.parse_perf_stat(
        "1000;;cycles;1;100.00;;\n"
        "250;;instructions;1;100.00;;\n"
        "<not supported>;;cache-misses;0;100.00;;\n"
    )
    assert parsed["counters"] == {"cycles": 1000.0, "instructions": 250.0}
    assert parsed["unavailable"] == {"cache-misses": "<not supported>"}


def test_benchmark_thread_policy_is_explicit_and_overrideable(monkeypatch):
    bench = _load_tool("bench_bonsai35")
    for key in bench.THREAD_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(bench, "_physical_cpu_map", lambda: {(0, i): [i, i + 8] for i in range(8)})
    env = bench.configure_thread_environment(4, 2)
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"
    assert env["OMP_NUM_THREADS"] == "4"
    assert env["OMP_WAIT_POLICY"] == "PASSIVE"
    assert env["GOMP_SPINCOUNT"] == "0"
    assert env["TRINOTE_BONSAI35_Q1_CHUNK"] == "2"

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "3")
    assert bench.configure_thread_environment(2, 4)["OPENBLAS_NUM_THREADS"] == "3"


def test_benchmark_worker_receives_selected_threads_and_prefill_chunk():
    bench = _load_tool("bench_bonsai35")
    args = bench.build_parser().parse_args([
        "--threads", "3", "--prefill-q1-chunk", "1",
    ])
    command = bench._worker_args(args, repetitions=5, warmups=1)

    assert command[command.index("--threads") + 1] == "3"
    assert command[command.index("--prefill-q1-chunk") + 1] == "1"


def test_legacy_native_profile_forces_and_reports_every_replay_toggle(monkeypatch):
    bench = _load_tool("bench_bonsai35")
    for key in bench.LEGACY_NATIVE_ENVIRONMENT:
        monkeypatch.setenv(key, "ambient-value")

    report = bench.configure_producer_environment("legacy-native")

    assert report["profile"]["name"] == "bonsai35-pre-fusion-python-native-primitives"
    assert report["previous"] == {
        key: "ambient-value" for key in bench.LEGACY_NATIVE_ENVIRONMENT
    }
    assert report["forced"] == bench.LEGACY_NATIVE_ENVIRONMENT
    assert report["effective"] == bench.LEGACY_NATIVE_ENVIRONMENT
    assert {key: os.environ[key] for key in bench.LEGACY_NATIVE_ENVIRONMENT} == (
        bench.LEGACY_NATIVE_ENVIRONMENT
    )
    assert report["forced"]["TRINOTE_BONSAI35_MODEL_EXECUTOR"] == "0"
    assert report["forced"]["TRINOTE_BONSAI35_Q1_FUSED"] == "0"
    assert report["forced"]["TRINOTE_BONSAI35_Q1_LUT32"] == "0"
    assert report["forced"]["TRINOTE_BONSAI35_NATIVE_GDN"] == "0"
    # These were already native, separately-dispatched primitives in the
    # historical snapshot; the baseline pins them on rather than pretending
    # that the much slower all-NumPy graph was measured.
    assert report["forced"]["TRINOTE_NATIVE_RMSNORM"] == "1"
    assert report["forced"]["TRINOTE_NATIVE_SILU"] == "1"
    assert report["forced"]["TRINOTE_NATIVE_ATTN"] == "1"


def test_nonlegacy_producers_do_not_mutate_runtime_environment(monkeypatch):
    bench = _load_tool("bench_bonsai35")
    monkeypatch.setenv("TRINOTE_BONSAI35_MODEL_EXECUTOR", "custom")
    report = bench.configure_producer_environment("native")
    assert report["forced"] == report["previous"] == report["effective"] == {}
    assert report["profile"] is None
    assert os.environ["TRINOTE_BONSAI35_MODEL_EXECUTOR"] == "custom"


def test_integer_27b_launcher_sets_pools_before_python_and_cli_threads_win(tmp_path):
    model = tmp_path / "model.gguf"
    artifact = tmp_path / "artifact.safetensors"
    model.touch()
    artifact.touch()
    env = os.environ.copy()
    for key in (
        "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS",
        "OMP_DYNAMIC", "OMP_WAIT_POLICY", "OMP_PLACES", "OMP_PROC_BIND", "OMP_MAX_ACTIVE_LEVELS",
        "GOMP_SPINCOUNT", "KMP_BLOCKTIME",
    ):
        env.pop(key, None)
    env.update({
        "BONSAI_DRYRUN": "1", "BONSAI_27B_GGUF": str(model),
        "BONSAI_INTEGER_27B_ARTIFACT": str(artifact), "BONSAI_INTEGER_27B_THREADS": "7",
    })
    proc = subprocess.run(
        [str(REPO_ROOT / "bonsai-integer-27b-cli"), "repl", "--threads", "3"],
        env=env, capture_output=True, text=True, check=True,
    )
    output = proc.stdout
    assert "OPENBLAS_NUM_THREADS=1" in output
    assert "MKL_NUM_THREADS=1" in output
    assert "NUMEXPR_NUM_THREADS=1" in output
    assert "OMP_NUM_THREADS=3" in output
    assert "OMP_WAIT_POLICY=PASSIVE" in output
    assert "GOMP_SPINCOUNT=0" in output


def test_idle_diagnostic_samples_a_sleeping_process_without_signalling_it():
    idle = _load_tool("diagnose_bonsai_idle")
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        time.sleep(0.05)
        sample = idle.sample_idle_cpu(proc.pid, 0.15)
        snapshot = idle.process_thread_snapshot(proc.pid)
        assert sample["cpu_percent_of_one_core"] < 10.0
        assert snapshot["thread_count"] >= 1
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_qwen35_output_is_identical_with_one_or_two_blas_threads():
    code = """
import hashlib
from trinote.infer_int.reference_bonsai35 import BonsaiQwen35ReferenceModel, random_bonsai35_artifact
m = BonsaiQwen35ReferenceModel(random_bonsai35_artifact(seed=91))
m.enable_native()
out = m.forward([1, 7], last_only=True)
print(hashlib.sha256(out.astype('<i8', copy=False).tobytes()).hexdigest())
"""
    digests = []
    for threads in ("1", "2"):
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": str(BONSAI_ROOT / "src"), "OPENBLAS_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads, "NUMEXPR_NUM_THREADS": threads,
            "OMP_NUM_THREADS": "1", "OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": "0",
        })
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                              text=True, check=True, timeout=60)
        digests.append(proc.stdout.strip())
    assert digests[0] == digests[1]


def test_model_iteration_uses_fresh_prefill_semantics_without_pretouch(monkeypatch):
    """The timed prefill resets semantics; the harness must not pre-reset."""

    bench = _load_tool("bench_bonsai35")

    events = []

    class Resident:
        def __init__(self):
            self._position = 19

        def position(self):
            events.append("position")
            return self._position

        def stats(self):
            events.append("stats")
            return {
                "decode_calls": 0, "prefill_calls": 0, "team_entries": 0,
                "q1_groups": 0, "lut32_hits": 0, "lut32_fallbacks": 0,
                "lut64_groups": 0, "layer_major_prefills": 0,
                "layer_major_rows": 0, "prefill_tiles_40": 0,
                "prefill_tiles_48": 0, "prefill_tiles_136": 0,
                "last_team_size": 1,
                "selected_isa": "portable", "selected_lut_bits": 64,
                "cache_width_bits": 64, "prefill_tile_40": 1,
                "prefill_tile_48": 1, "prefill_tile_136": 1,
            }

        def prefill_argmax(self, token_ids):
            # Match the resident C ABI: prefill resets semantic state inside
            # the measured call while retaining the handle/allocations.
            events.append("prefill")
            self._position = len(token_ids)
            return 11

        def export_cache_tensor(self, layer_index, name):
            raise AssertionError("the zero-layer fixture has no cache tensors")

        def export_last_residual(self):
            return np.zeros((1, 2), dtype=np.int64)

    resident = Resident()

    class Model:
        cfg = {"frac": 16, "rmsEpsilonFp2": 1}
        artifact = {"layers": []}
        _model_executor = resident

    class Reader:
        kv = {"tokenizer.ggml.eos_token_id": -1}

    class Args:
        producer = "native"
        raw_ids = "12675"
        prompt = "Hi"
        chat = False
        gguf = "unused.gguf"
        bin_dir = "unused-bin"
        sampler = "greedy"
        seed = 0
        max_new = 1
        ignore_eos = True

    # Avoid platform GPU/resource probes; they are unrelated to cache state.
    def usage_snapshot():
        events.append("usage")
        return None, 0.0, 0.0

    monkeypatch.setattr(bench, "_usage_snapshot", usage_snapshot)
    monkeypatch.setattr(bench, "_usage_delta", lambda before, after: {})

    first = bench._model_iteration(Model(), Reader(), Args())
    first_events = list(events)
    events.clear()
    resident._position = 37
    second = bench._model_iteration(Model(), Reader(), Args())

    assert first["input_ids"] == second["input_ids"] == [12675]
    assert first["output_ids"] == second["output_ids"] == [11]
    assert first["commitments"] == second["commitments"]
    assert first["commitments"]["cache_position"] == 1
    # No out-of-band reset/pre-touch occurs before the initial counter and
    # resource snapshots; the fresh-state operation is the timed prefill.
    assert first_events[:3] == ["stats", "usage", "prefill"]
    assert events[:3] == ["stats", "usage", "prefill"]

    kernel = (BONSAI_ROOT / "tools" / "bonsai_q1_kernel.c").read_text()
    for symbol in (
        "bonsai35_model_prefill", "bonsai35_model_prefill_logits",
        "bonsai35_model_prefill_argmax",
    ):
        start = kernel.index(f"int {symbol}(")
        body = kernel[start:kernel.index("\n}", start)]
        assert "bonsai35_model_run(m, tokens, count, 1," in body


def test_legacy_iteration_uses_canonical_python_cache_commitment(monkeypatch):
    bench = _load_tool("bench_bonsai35")

    class Runtime:
        fused = False
        lut32_mode = "0"

    class Model:
        cfg = {"frac": 16, "rmsEpsilonFp2": 1}
        artifact = {
            "layers": [],
            "final_norm_gain_fp": np.ones(2, dtype=np.int64) << 16,
        }
        _native = True
        _native_runtime = Runtime()
        _model_executor = None

        def _run_layers(self, token_ids, cache):
            cache.t += len(token_ids)
            return np.asarray([[1 << 16, 0]], dtype=np.int64)

        def _output_argmax(self, _last):
            return np.asarray([11], dtype=np.int64)

    class Reader:
        kv = {"tokenizer.ggml.eos_token_id": -1}

    class Args:
        producer = "legacy-native"
        raw_ids = "12675"
        prompt = "Hi"
        chat = False
        gguf = "unused.gguf"
        bin_dir = "unused-bin"
        sampler = "greedy"
        seed = 0
        max_new = 1
        ignore_eos = True

    monkeypatch.setattr(bench, "_usage_snapshot", lambda: (None, 0.0, 0.0))
    monkeypatch.setattr(bench, "_usage_delta", lambda _before, _after: {})

    result = bench._model_iteration(Model(), Reader(), Args())
    assert result["runtime"] == "python-native-primitives-legacy-baseline"
    assert result["output_ids"] == [11]
    assert result["commitments"]["cache_position"] == 1
    assert result["commitments"]["cache_state_format"] == (
        "trinote-bonsai35-cache-commitment/1"
    )
    assert len(result["commitments"]["cache_state_sha256"]) == 64


def test_legacy_native_disables_only_post_snapshot_native_gdn(monkeypatch):
    from trinote.infer_int import reference_bonsai35

    monkeypatch.delenv("TRINOTE_BONSAI35_NATIVE_GDN", raising=False)
    assert reference_bonsai35._native_gdn_enabled()
    monkeypatch.setenv("TRINOTE_BONSAI35_NATIVE_GDN", "0")
    assert not reference_bonsai35._native_gdn_enabled()
    monkeypatch.setenv("TRINOTE_BONSAI35_NATIVE_GDN", "1")
    assert reference_bonsai35._native_gdn_enabled()
