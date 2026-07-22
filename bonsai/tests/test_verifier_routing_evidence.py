from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import struct
from types import SimpleNamespace

import pytest

from trinote.cli import receipt_bundle_cli as bundle_cli
from trinote.cli import run_bonsai_cli as run_cli
from trinote.cli.run_bonsai_cli import _generate_native_turn, _validate_args
from trinote.cli.run_evidence import ReceiptRunEvidence, SCHEMA as RUN_SCHEMA
from trinote.cli.live_session import LiveNativeSession
from trinote.cli.thread_bootstrap import THREAD_ENV, maybe_reexec_with_threads
from trinote.cli.verifier_policy import (
    SCHEMA as POLICY_SCHEMA,
    route_verification,
    validate_verifier_policy,
)
from trinote.infer_int.verify import _full_verification_strategy


def test_release_golden_fixture_has_exact_committed_19x64_ids():
    from pathlib import Path

    path = Path(__file__).parent / "fixtures/bonsai35_19x64_golden.json"
    fixture = json.loads(path.read_text("utf-8"))
    assert fixture["schema"] == "trinote-bonsai35-golden/v1"
    assert len(fixture["inputIds"]) == 19
    assert len(fixture["outputIds"]) == 64

    def ids_sha(values):
        raw = b"".join(struct.pack("<q", int(value)) for value in values)
        return hashlib.sha256(raw).hexdigest()

    assert ids_sha(fixture["inputIds"]) == fixture["commitments"]["inputIdsInt64LeSha256"]
    assert ids_sha(fixture["outputIds"]) == fixture["commitments"]["outputIdsInt64LeSha256"]


def test_release_golden_tool_requires_device_sampler_and_zero_full_logits_d2h():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "tools/verify_bonsai35_golden.py"
    ).read_text("utf-8")
    assert "executor.generate_device(" in source
    assert "executor.generate(" not in source
    for gate in (
        '"deviceLogitsInputMode"',
        '"singleNativePrefill"',
        '"deviceOnlyConsumedPositions"',
        '"deviceSamplerCallsExact"',
        '"zeroFullLogitsD2H"',
        '"samplerTrafficBelowOneLogitsRow"',
    ):
        assert gate in source


def test_release_golden_execution_exception_writes_atomic_failure_and_closes_gpu(
    tmp_path, monkeypatch,
):
    import trinote.infer_int.artifact_io_bonsai as artifact_io
    import trinote.infer_int.gpu_bonsai35 as gpu_bonsai35
    import trinote.infer_int.sampler as sampler_module
    from pathlib import Path

    tool = Path(__file__).resolve().parents[1] / "tools/verify_bonsai35_golden.py"
    spec = importlib.util.spec_from_file_location("verify_bonsai35_golden_failure", tool)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    artifact_path = tmp_path / "artifact.safetensors"
    gguf_path = tmp_path / "model.gguf"
    artifact_path.write_bytes(b"artifact")
    gguf_path.write_bytes(b"gguf")
    input_ids = list(range(19))
    output_ids = list(range(100, 164))
    fixture = {
        "schema": "trinote-bonsai35-golden/v1",
        "inputIds": input_ids,
        "outputIds": output_ids,
        "commitments": {
            "inputIdsInt64LeSha256": module.ids_sha(input_ids),
            "outputIdsInt64LeSha256": module.ids_sha(output_ids),
            "visibleBytesSha256": "00" * 32,
        },
        "release": {
            "artifactSha256": module.sha_file(artifact_path),
            "ggufSha256": module.sha_file(gguf_path),
        },
    }
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

    class Report:
        def as_dict(self):
            return {"resident": True}

    class Executor:
        def __init__(self):
            self.closed = False

        def generate_device(self, *_args, **_kwargs):
            raise RuntimeError("synthetic graph failure")

        def close(self):
            self.closed = True

    executor = Executor()

    class Factory:
        @staticmethod
        def try_create_reported(_artifact):
            return executor, Report()

    monkeypatch.setattr(
        artifact_io, "load_artifact_bonsai",
        lambda _path: ({"config": {"vocab": 256}}, {}),
    )
    monkeypatch.setattr(gpu_bonsai35, "Bonsai35GpuExecutor", Factory)
    monkeypatch.setattr(sampler_module, "resolve_sampler", lambda *_args, **_kwargs: object())
    output_path = tmp_path / "golden.json"
    rc = module.main([
        "--artifact", str(artifact_path),
        "--gguf", str(gguf_path),
        "--fixture", str(fixture_path),
        "--json-out", str(output_path),
    ])

    assert rc == 4
    record = json.loads(output_path.read_text("utf-8"))
    assert record["status"] == "fail" and record["stage"] == "gpu-execution"
    assert "synthetic graph failure" in record["error"]
    assert record["cleanup"] == {"gpuClosed": True}
    assert not list(tmp_path.glob(".*.tmp"))


def test_verifier_policy_routes_by_committed_token_counts_in_rule_order():
    policy = {
        "schema": POLICY_SCHEMA,
        "artifactSha256": "ab" * 32,
        "threads": 8,
        "rules": [
            {
                "maxInputTokens": 19,
                "maxOutputTokens": 20,
                "engine": "oracle",
                "strategy": "teacher-forced",
            },
            {
                "maxOutputTokens": 64,
                "engine": "native",
                "strategy": "teacher-forced",
            },
        ],
        "default": {"engine": "native", "strategy": "cached-replay"},
    }
    assert route_verification(policy, input_tokens=19, output_tokens=20) == {
        "engine": "oracle", "strategy": "teacher-forced"
    }
    assert route_verification(policy, input_tokens=40, output_tokens=20) == {
        "engine": "native", "strategy": "teacher-forced"
    }
    assert route_verification(policy, input_tokens=19, output_tokens=65) == {
        "engine": "native", "strategy": "cached-replay"
    }


def test_verifier_policy_rejects_unknown_or_weakened_routes():
    with pytest.raises(ValueError, match="schema"):
        validate_verifier_policy({"schema": "wrong", "rules": [], "default": {}})
    with pytest.raises(ValueError, match="strategy"):
        validate_verifier_policy({
            "schema": POLICY_SCHEMA,
            "artifactSha256": "ab" * 32,
            "threads": 8,
            "rules": [],
            "default": {"engine": "native", "strategy": "sampled"},
        })


def test_benchmark_policy_selects_one_thread_budget_and_oracle_constraint():
    from pathlib import Path

    tool = Path(__file__).resolve().parents[1] / "tools/bench_bonsai35_verifier.py"
    spec = importlib.util.spec_from_file_location("bench_bonsai35_verifier", tool)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    results = []
    for threads, base in ((2, 3.0), (8, 1.0)):
        for count in (8, 20):
            for engine, strategy, offset in (
                ("oracle", "teacher-forced", 0.4),
                ("oracle", "cached-replay", 0.2),
                ("native", "cached-replay", 0.0),
            ):
                results.append({
                    "verified": True, "threads": threads,
                    "requestedThreads": threads, "actualThreads": threads,
                    "threadsMatched": True,
                    "inputTokens": 19, "outputTokens": count,
                    "requestedEngine": engine, "requestedStrategy": strategy,
                    "medianVerifySeconds": base + offset,
                })
    policy = module._policy(results, artifact_sha="ab" * 32)
    assert policy["threads"] == 8
    assert all(rule["engine"] == "native" for rule in policy["rules"])
    oracle = module._policy(results, artifact_sha="ab" * 32, engine_filter="oracle")
    assert oracle["threads"] == 8 and oracle["engineConstraint"] == "oracle"
    assert all(rule["engine"] == "oracle" for rule in oracle["rules"])
    assert policy["requireMeasuredPoint"] is True
    assert policy["measuredPoints"] == [
        {"inputTokens": 19, "outputTokens": 8},
        {"inputTokens": 19, "outputTokens": 20},
    ]
    assert route_verification(policy, input_tokens=19, output_tokens=8)["engine"] == "native"
    with pytest.raises(ValueError, match="outside.*measured matrix"):
        route_verification(policy, input_tokens=19, output_tokens=9)
    with pytest.raises(ValueError, match="outside.*measured matrix"):
        route_verification(policy, input_tokens=20, output_tokens=8)


def test_measured_policy_requires_exact_first_match_route_and_thread_attestation():
    base = {
        "schema": POLICY_SCHEMA,
        "artifactSha256": "ab" * 32,
        "threads": 8,
        "requireMeasuredPoint": True,
        "measuredPoints": [{"inputTokens": 19, "outputTokens": 20}],
        "default": {"engine": "native", "strategy": "cached-replay"},
    }
    with pytest.raises(ValueError, match="exact input/output bounds"):
        validate_verifier_policy({**base, "rules": []})
    with pytest.raises(ValueError, match="exact input/output bounds"):
        validate_verifier_policy({
            **base,
            "rules": [
                {
                    "maxInputTokens": 100,
                    "maxOutputTokens": 100,
                    "engine": "oracle",
                    "strategy": "teacher-forced",
                    "measuredThreads": 8,
                },
                {
                    "minInputTokens": 19, "maxInputTokens": 19,
                    "minOutputTokens": 20, "maxOutputTokens": 20,
                    "engine": "native", "strategy": "cached-replay",
                    "measuredThreads": 8,
                },
            ],
        })
    with pytest.raises(ValueError, match="measuredThreads"):
        validate_verifier_policy({
            **base,
            "rules": [{
                "minInputTokens": 19, "maxInputTokens": 19,
                "minOutputTokens": 20, "maxOutputTokens": 20,
                "engine": "native", "strategy": "cached-replay",
                "measuredThreads": 4,
            }],
        })


@pytest.mark.parametrize(
    "changes",
    [
        {"verifier_engine": "oracle"},
        {"oracle": True},
        {"strategy": "teacher-forced"},
    ],
)
def test_measured_bundle_policy_rejects_route_overrides(monkeypatch, changes):
    args = SimpleNamespace(verifier_engine="auto", oracle=False, strategy="auto")
    for key, value in changes.items():
        setattr(args, key, value)
    monkeypatch.setattr(bundle_cli, "_committed_token_counts", lambda _path: (19, 20))
    policy = {
        "schema": POLICY_SCHEMA,
        "artifactSha256": "ab" * 32,
        "threads": 8,
        "requireMeasuredPoint": True,
        "measuredPoints": [{"inputTokens": 19, "outputTokens": 20}],
        "rules": [{
            "minInputTokens": 19, "maxInputTokens": 19,
            "minOutputTokens": 20, "maxOutputTokens": 20,
            "engine": "native", "strategy": "cached-replay", "measuredThreads": 8,
        }],
        "default": {"engine": "native", "strategy": "cached-replay"},
    }
    with pytest.raises(ValueError, match="authoritative"):
        bundle_cli._resolve_verifier_route(args, policy, "unused")


def test_benchmark_policy_rejects_requested_only_thread_claims():
    from pathlib import Path

    tool = Path(__file__).resolve().parents[1] / "tools/bench_bonsai35_verifier.py"
    spec = importlib.util.spec_from_file_location("bench_bonsai35_thread_claim", tool)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    claimed = [{
        "verified": True,
        "threads": 64,
        "requestedThreads": 64,
        "actualThreads": 32,
        "threadsMatched": False,
        "inputTokens": 19,
        "outputTokens": 20,
        "requestedEngine": "oracle",
        "requestedStrategy": "teacher-forced",
        "medianVerifySeconds": 1.0,
    }]
    with pytest.raises(RuntimeError, match="no requested verifier benchmark cell"):
        module._policy(claimed, artifact_sha="ab" * 32)


def test_benchmark_publishes_policies_only_after_the_full_matrix_passes():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "tools/bench_bonsai35_verifier.py"
    ).read_text("utf-8")
    assert 'matrix_passed = report["status"] == "pass"' in source
    assert "args.policy_out and matrix_passed" in source
    assert "args.oracle_policy_out and matrix_passed" in source


def test_benchmark_native_cells_require_the_resident_model_executor():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "tools/bench_bonsai35_verifier.py"
    ).read_text("utf-8")
    assert 'getattr(model, "_model_executor", None) is None' in source
    assert '"could not be created"' in source


def test_benchmark_cell_verdict_requires_full_coverage_and_requested_strategy():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "tools/bench_bonsai35_verifier.py"
    ).read_text("utf-8")
    assert 'int(result.get("checked", -1)) == len(output_ids)' in source
    assert 'result.get("strategy") == expected_result_strategy' in source
    assert '"strategyMatched"' in source


def test_bundle_native_q35_route_requires_resident_model_executor(monkeypatch):
    import trinote.infer_int.artifact_io_bonsai as artifact_io
    import trinote.infer_int.reference_bonsai35 as reference_bonsai35

    class PrimitiveOnlyQwen35:
        def __init__(self, artifact):
            self.artifact = artifact
            self._model_executor = None

        def enable_native(self):
            return True

        def enable_fast(self, **_kwargs):
            raise AssertionError("q35 primitive fallback is not a sign cache")

    artifact = {"config": {"architecture": "qwen35"}}
    monkeypatch.setattr(
        artifact_io, "load_artifact_bonsai",
        lambda _: (artifact, {"digest": "ab" * 32}),
    )
    monkeypatch.setattr(
        reference_bonsai35, "BonsaiQwen35ReferenceModel", PrimitiveOnlyQwen35,
    )

    _model, _digest, engine = bundle_cli._load_model(
        "fake.safetensors", fast=True, require_native=False,
    )
    assert engine == "native-primitives"
    with pytest.raises(RuntimeError, match="native verifier engine was required"):
        bundle_cli._load_model("fake.safetensors", fast=True, require_native=True)


def test_bundle_native_q35_route_accepts_resident_model_executor(monkeypatch):
    import trinote.infer_int.artifact_io_bonsai as artifact_io
    import trinote.infer_int.reference_bonsai35 as reference_bonsai35

    class ResidentQwen35:
        def __init__(self, artifact):
            self.artifact = artifact
            self._model_executor = object()

        def enable_native(self):
            return True

    artifact = {"config": {"architecture": "qwen35"}}
    monkeypatch.setattr(
        artifact_io, "load_artifact_bonsai",
        lambda _: (artifact, {"digest": "cd" * 32}),
    )
    monkeypatch.setattr(
        reference_bonsai35, "BonsaiQwen35ReferenceModel", ResidentQwen35,
    )

    _model, digest, engine = bundle_cli._load_model(
        "fake.safetensors", fast=True, require_native=True,
    )
    assert digest == "cd" * 32
    assert engine == "native"


class _CachedModel:
    receipt_verify_cached_threshold = 8

    def generate_cached(self):  # pragma: no cover - presence is the capability signal
        raise AssertionError


def test_exact_verifier_strategy_override_is_deterministic():
    model = _CachedModel()
    assert _full_verification_strategy(model, 19, 7) == "teacher-forced"
    assert _full_verification_strategy(model, 19, 8) == "cached-replay"
    model.receipt_verify_strategy = "teacher-forced"
    assert _full_verification_strategy(model, 19, 64) == "teacher-forced"
    model.receipt_verify_strategy = "cached-replay"
    assert _full_verification_strategy(model, 19, 1) == "cached-replay"


def _valid_cli_args(**changes):
    values = dict(
        repl=False, prompt="hello", max_new=1, think=False, no_think=False,
        chat=False, sampler="greedy", temp=1.0, top_k=1, top_p=1.0,
        min_p=0.0, no_repeat_ngram=0, receipt=False, engine="native",
        onchain=False, json=False, ctx_size=None, n_gpu_layers=None,
        threads=0, fast_required=False, fast=False, require_gpu=False,
        gpu=False, prompt_cache=False, verify_mode="fast-local",
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_require_gpu_implies_opt_in_and_rejects_fallback_only_modes(capsys):
    args = _valid_cli_args(require_gpu=True)
    assert _validate_args(args) == 0
    assert args.gpu is True

    assert _validate_args(_valid_cli_args(require_gpu=True, prompt_cache=True)) == 2
    assert "cannot be combined" in capsys.readouterr().err
    assert _validate_args(_valid_cli_args(require_gpu=True, engine="prismml.cpp")) == 2


def test_receipt_policy_rejects_explicit_strategy_override(capsys):
    args = _valid_cli_args(
        receipt=True,
        verify_mode="fresh-oracle",
        receipt_verify_policy="policy.json",
        receipt_verify_strategy="teacher-forced",
    )
    assert _validate_args(args) == 2
    assert "authoritative" in capsys.readouterr().err


def test_require_gpu_refuses_runtime_guard_without_cpu_replay():
    class Ref:
        cfg = {"frac": 16}

    class GuardedExecutor:
        def generate(self, *_args, **_kwargs):
            return [7], False

    args = SimpleNamespace(
        max_new=1, require_gpu=True, prompt_cache=False, prompt_cache_dir=None
    )
    streamed = []
    with pytest.raises(RuntimeError, match="--require-gpu"):
        _generate_native_turn(
            Ref(), [1, 2], args=args,
            cfg=SimpleNamespace(), eos=None, on_token=streamed.append,
            artifact_arch="qwen35", artifact_digest="ab" * 32,
            gpu_executor=GuardedExecutor(), live_session=None,
        )
    assert streamed == []


def test_gpu_dispatch_prefers_exact_device_sampler_and_scalar_path():
    class Ref:
        cfg = {"frac": 16}

    class DeviceExecutor:
        def generate_device(self, input_ids, n_new, cfg, *, eos, on_token):
            assert input_ids == [1, 2] and n_new == 2 and cfg is sampler
            assert eos is None
            on_token(7); on_token(8)
            return [7, 8], True

        def generate(self, *_args, **_kwargs):
            raise AssertionError("host-logits sampler path was selected")

    sampler = SimpleNamespace()
    streamed = []
    args = SimpleNamespace(
        max_new=2, require_gpu=False, prompt_cache=False, prompt_cache_dir=None
    )
    output = _generate_native_turn(
        Ref(), [1, 2],
        args=args,
        cfg=sampler, eos=None, on_token=streamed.append,
        artifact_arch="qwen35", artifact_digest="ab" * 32,
        gpu_executor=DeviceExecutor(), live_session=None,
    )
    assert output == streamed == [7, 8]
    assert args._last_generation_path == {
        "actualProducer": "resident-cuda",
        "gpuAttempted": True,
        "gpuFallback": False,
    }


def test_generation_path_records_live_and_non_live_gpu_fallback(monkeypatch):
    class Ref:
        cfg = {"frac": 16}

    class GuardedExecutor:
        def generate_device(self, _ids, _count, _cfg, *, eos, on_token):
            assert eos is None
            on_token(7)
            return [7], False

    def cpu_replay(_ref, _ids, _count, *, sampler, eos, on_token):
        assert eos is None and sampler is cfg
        on_token(7)
        on_token(8)
        return [7, 8]

    monkeypatch.setattr(run_cli, "generate_bonsai_tokens", cpu_replay)
    cfg = SimpleNamespace()
    args = SimpleNamespace(
        max_new=2, require_gpu=False, prompt_cache=False, prompt_cache_dir=None,
    )
    streamed = []
    output = _generate_native_turn(
        Ref(), [1, 2], args=args, cfg=cfg, eos=None, on_token=streamed.append,
        artifact_arch="qwen35", artifact_digest="ab" * 32,
        gpu_executor=GuardedExecutor(), live_session=None,
    )
    assert output == streamed == [7, 8]
    assert args._last_generation_path == {
        "actualProducer": "cpu", "gpuAttempted": True, "gpuFallback": True,
    }

    class LiveFallback:
        gpu = object()

        def generate(self, *_args, on_token, on_gpu_fallback, **_kwargs):
            on_token(9)
            on_gpu_fallback()
            return SimpleNamespace(output_ids=[9], gpu_fallback=True)

    live_args = SimpleNamespace(max_new=1, require_gpu=False)
    live_streamed = []
    live_output = _generate_native_turn(
        Ref(), [1, 2], args=live_args, cfg=cfg, eos=None,
        on_token=live_streamed.append, artifact_arch="qwen35",
        artifact_digest="ab" * 32, gpu_executor=object(),
        live_session=LiveFallback(),
    )
    assert live_output == live_streamed == [9]
    assert live_args._last_generation_path == {
        "actualProducer": "cpu", "gpuAttempted": True, "gpuFallback": True,
    }


def test_run_evidence_includes_generation_path_per_turn():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "src/trinote/cli/run_bonsai_cli.py"
    ).read_text("utf-8")
    assert 'generation_path = dict(getattr(args, "_last_generation_path", {}))' in source
    assert source.count("**generation_path") >= 2


def test_thread_bootstrap_reexecs_real_argv_and_preserves_list_calls(monkeypatch):
    class Reexec(Exception):
        pass

    captured = {}

    def fake_exec(executable, command, environment):
        captured.update(executable=executable, command=command, environment=environment)
        raise Reexec

    monkeypatch.setattr("trinote.cli.thread_bootstrap.os.execve", fake_exec)
    monkeypatch.setattr("trinote.cli.thread_bootstrap.sys.argv", ["cli", "--cpu-threads", "7"])
    for name in (*THREAD_ENV, "OMP_DYNAMIC", "TRINOTE_CLI_THREAD_BOOTSTRAP"):
        monkeypatch.delenv(name, raising=False)

    maybe_reexec_with_threads(7, real_argv=False, module_name="trinote.cli.run_bonsai_cli")
    assert captured == {}
    with pytest.raises(Reexec):
        maybe_reexec_with_threads(7, real_argv=True, module_name="trinote.cli.run_bonsai_cli")
    assert captured["command"][-2:] == ["--cpu-threads", "7"]
    assert all(captured["environment"][name] == "7" for name in THREAD_ENV)
    assert captured["environment"]["OMP_DYNAMIC"] == "FALSE"
    assert captured["environment"]["TRINOTE_CLI_THREAD_BOOTSTRAP"].endswith(":7")
    for name, value in captured["environment"].items():
        if name in (*THREAD_ENV, "OMP_DYNAMIC", "TRINOTE_CLI_THREAD_BOOTSTRAP"):
            monkeypatch.setenv(name, value)
    captured.clear()
    maybe_reexec_with_threads(7, real_argv=True, module_name="trinote.cli.run_bonsai_cli")
    assert captured == {}
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    with pytest.raises(RuntimeError, match="changed after CLI bootstrap"):
        maybe_reexec_with_threads(7, real_argv=True, module_name="trinote.cli.run_bonsai_cli")


def test_receipt_policy_thread_budget_reaches_preimport_bootstrap(tmp_path, monkeypatch):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({
        "schema": POLICY_SCHEMA,
        "artifactSha256": "ab" * 32,
        "threads": 6,
        "rules": [],
        "default": {"engine": "oracle", "strategy": "teacher-forced"},
    }), encoding="utf-8")
    captured = {}

    class Reexec(Exception):
        pass

    def bootstrap(count, *, real_argv, module_name):
        captured.update(count=count, real_argv=real_argv, module_name=module_name)
        raise Reexec

    monkeypatch.setattr(bundle_cli, "maybe_reexec_with_threads", bootstrap)
    args = SimpleNamespace(
        bundle=[], oracle=False, verifier_engine="auto",
        strategy_policy=str(policy_path), threads=0, _real_argv=True,
    )
    with pytest.raises(Reexec):
        bundle_cli._cmd_verify(args)
    assert captured == {
        "count": 6,
        "real_argv": True,
        "module_name": "trinote.cli.receipt_bundle_cli",
    }


def test_live_gpu_device_sampler_reuses_resident_prefix_without_logits_d2h():
    class DeviceGpu:
        def __init__(self):
            self.position = 0
            self.prefills = 0
            self.samples = iter((
                ((1, 2), 2, 7),
                ((1, 2, 7), 3, 8),
                ((1, 2, 7, 8, 3), 5, 9),
            ))
            self.decoded = []

        def reset(self):
            self.position = 0
            return True

        def _prefill_device(self, ids):
            self.prefills += 1
            self.position += len(ids)
            return True

        def sample_device(self, _cfg, history, position):
            expected_history, expected_position, token = next(self.samples)
            assert tuple(history) == expected_history
            assert position == expected_position == self.position
            return token

        def decode_token_device(self, token):
            self.decoded.append(int(token))
            self.position += 1
            return True

    gpu = DeviceGpu()
    session = LiveNativeSession(
        SimpleNamespace(cfg={"frac": 16}),
        architecture="qwen35", artifact_digest="ab", gpu_executor=gpu,
    )
    cfg = SimpleNamespace()
    first = session.generate([1, 2], 2, lambda *_: (_ for _ in ()).throw(AssertionError),
                             sampler_cfg=cfg)
    assert first.output_ids == [7, 8]
    assert gpu.prefills == 1 and gpu.position == 4
    second = session.generate([1, 2, 7, 8, 3], 1,
                              lambda *_: (_ for _ in ()).throw(AssertionError),
                              sampler_cfg=cfg)
    assert second.output_ids == [9]
    assert second.reused_tokens == 4
    assert gpu.prefills == 1 and gpu.position == 6
    assert gpu.decoded == [7, 8, 3, 9]


def test_receipt_run_evidence_is_atomic_and_finalized(tmp_path):
    path = tmp_path / "run.json"
    evidence = ReceiptRunEvidence(path, operation="test", options={"gpuRequired": True})
    evidence.update("model", artifactDigest="ab" * 32)
    evidence.add_phase("verification", 0.125, strategy="cached-replay")
    evidence.update("cleanup", gpuClosed=True)
    evidence.finish("pass", exit_code=0)

    record = json.loads(path.read_text("utf-8"))
    assert record["schema"] == RUN_SCHEMA
    assert record["status"] == "pass"
    assert record["phases"][0]["strategy"] == "cached-replay"
    assert record["cleanup"]["gpuClosed"] is True
    assert record["resources"]["maxRssKiB"] > 0
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob("*.tmp"))


def test_receipt_run_evidence_redacts_nested_paths_prompts_and_keys(tmp_path):
    path = tmp_path / "run.json"
    evidence = ReceiptRunEvidence(
        path,
        operation="test",
        options={"strategyPolicy": "/srv/private/policy.json", "prompt": "secret words"},
    )
    evidence.add_phase(
        "failure", 0.0,
        details={"modelKeyPath": "/srv/private/model.key", "message": "open /srv/secret/key.json"},
    )
    evidence.finish("failed", exit_code=1, error="missing /srv/private/model.key")
    rendered = path.read_text("utf-8")
    record = json.loads(rendered)
    assert "secret words" not in rendered
    assert "/srv/private" not in rendered
    assert record["options"]["prompt"] == "[redacted]"
    assert record["phases"][0]["details"]["modelKeyPath"] == "[redacted]"


def test_receipt_run_evidence_redacts_common_nested_credentials_and_chat_content(tmp_path):
    path = tmp_path / "sensitive.json"
    evidence = ReceiptRunEvidence(
        path,
        operation="test",
        options={
            "apiKey": "sk-this-is-a-long-test-credential",
            "nested": {"password": "do-not-write", "messages": [
                {"role": "user", "content": "private conversation"},
            ]},
        },
    )
    evidence.finish(
        "failed",
        exit_code=1,
        error=(
            'api_key=super-secret-value password=hunter-example '
            'passphrase="correct \\"horse\\" battery staple" token=plain-token-value\n'
            'auth_token=auth-value session_token=session-value id_token=id-value\n'
            'secret=generic-secret authorization=Basic basic-credential\n'
            'authToken=camel-auth-secret clientSecret=camel-client-secret\n'
            'OPENAI_API_KEY=env-openai-secret AWS_SECRET_ACCESS_KEY=env-aws-secret\n'
            'passphrase="first-line-secret\nsecond-line-secret"'
        ),
    )
    rendered = path.read_text("utf-8")
    assert "long-test-credential" not in rendered
    assert "do-not-write" not in rendered
    assert "private conversation" not in rendered
    assert "super-secret-value" not in rendered
    assert "hunter-example" not in rendered
    assert "horse battery staple" not in rendered
    assert "plain-token-value" not in rendered
    for secret in (
        "auth-value", "session-value", "id-value", "generic-secret", "basic-credential",
        "camel-auth-secret", "camel-client-secret", "env-openai-secret", "env-aws-secret",
        "second-line-secret",
    ):
        assert secret not in rendered
    record = json.loads(rendered)
    assert record["options"]["apiKey"] == "[redacted]"
    assert record["options"]["nested"]["messages"] == "[redacted]"


@pytest.mark.parametrize(
    ("value", "secret"),
    [
        ("authToken=camel-auth-secret", "camel-auth-secret"),
        ("clientSecret=camel-client-secret", "camel-client-secret"),
        ("OPENAI_API_KEY=env-openai-secret", "env-openai-secret"),
        ("AWS_SECRET_ACCESS_KEY=env-aws-secret", "env-aws-secret"),
        ("credentials=credential-secret-value", "credential-secret-value"),
        ("credential=singular-credential-value", "singular-credential-value"),
        ("_OPENAI_API_KEY=underscore-openai-secret", "underscore-openai-secret"),
        ("__AUTH_TOKEN=underscore-auth-secret", "underscore-auth-secret"),
        ("GITHUB_PAT=github-pat-secret", "github-pat-secret"),
        ("SENTRY_DSN=https://user:prefixed-dsn-value@sentry.example/1", "prefixed-dsn-value"),
        ("READ_REPLICA_DATABASE_URL=postgresql://user:replica-value@db/service", "replica-value"),
        ('payload="{\\"password\\":\\"escaped-json-value\\"}"', "escaped-json-value"),
        (
            'payload="{\\"messages\\":[{\\"role\\":\\"user\\",'
            '\\"content\\":\\"private roadmap\\"}]}"',
            "private roadmap",
        ),
        ("os.environ['OPENAI_API_KEY']='python-env-value'", "python-env-value"),
        ('passphrase="first-line-secret\nsecond-line-secret"', "second-line-secret"),
    ],
)
def test_safe_text_redacts_assignment_aliases_and_multiline_tails(value, secret):
    from trinote.cli.run_evidence import _safe_text

    rendered = _safe_text(value)
    assert secret not in rendered
    assert rendered.endswith("[redacted]")


def test_oracle_worker_count_reports_the_pool_created_at_import():
    from trinote.infer_int.reference_bonsai import oracle_q1_worker_count

    assert oracle_q1_worker_count() >= 1
