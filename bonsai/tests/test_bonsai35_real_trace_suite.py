from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_bonsai35_real_trace_suite.py"
DEFAULT_INPUTS = (
    Path.home() / ".local" / "trinote" / "results"
    / "bonsai35-real-trace-suite-v1.inputs.json"
)
_OPT_IN = os.environ.get("TRINOTE_RUN_BONSAI35_REAL_TRACE_SUITE", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _runner_module():
    spec = importlib.util.spec_from_file_location("bonsai35_real_trace_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_inputs() -> dict:
    required = {
        "rawHi": {"ids": [12675], "expectedNextGreedyToken": 11},
        "chatHi": {"ids": [1, 2, 3], "expectedNextGreedyToken": 4},
        "prompt32Unicode": {"ids": list(range(32)), "text": "café 東京 🌱"},
        "prompt128": {"ids": list(range(128)), "text": "long diagnostic"},
    }
    return {
        "format": "trinote-bonsai35-real-trace-inputs/1",
        "artifactSha256": "ab" * 32,
        "inputs": required,
        "traces": {
            "prefill": ["rawHi", "chatHi", "prompt32Unicode", "prompt128"],
            "cachedGreedy": {"input": "chatHi", "newTokens": 32},
        },
    }


def test_real_trace_runner_validates_and_plans_all_required_cases(tmp_path):
    module = _runner_module()
    inputs = tmp_path / "suite.inputs.json"
    inputs.write_text(json.dumps(_valid_inputs()))
    loaded = module.load_suite_inputs(inputs)
    jobs = module.trace_jobs(loaded)
    assert [job["key"] for job in jobs] == [
        "prefill/rawHi",
        "prefill/chatHi",
        "prefill/prompt32Unicode",
        "prefill/prompt128",
        "cached-greedy/chatHi/32",
    ]
    assert module.default_expected_dir(inputs).name == "suite.expected"

    json_out = tmp_path / "reports" / "plan.json"
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--mode", "plan", "--inputs", str(inputs),
         "--json-out", str(json_out)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert json_out.read_text() == proc.stdout
    assert report["mode"] == "plan"
    assert len(report["jobs"]) == 5
    assert report["artifactSha256"] == "ab" * 32


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data["inputs"]["rawHi"].update(ids=[1]), "12675"),
        (lambda data: data["inputs"]["prompt32Unicode"].update(ids=[1] * 31), "32"),
        (lambda data: data["inputs"]["prompt128"].update(ids=[1] * 127), "128"),
        (lambda data: data["traces"]["cachedGreedy"].update(newTokens=31), "32 tokens"),
    ],
)
def test_real_trace_runner_rejects_incomplete_release_manifests(tmp_path, mutation, message):
    module = _runner_module()
    data = _valid_inputs()
    mutation(data)
    path = tmp_path / "bad.inputs.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        module.load_suite_inputs(path)


def test_real_trace_runner_refuses_to_mint_native_expectations(tmp_path):
    module = _runner_module()
    with pytest.raises(ValueError, match="must be minted by the NumPy oracle"):
        module.write_expected_suite(
            artifact_path=tmp_path / "unused.safetensors",
            inputs_path=tmp_path / "unused.inputs.json",
            expected_dir=tmp_path / "expected",
            producer="native",
            force=False,
        )


def test_resident_cached_trace_checks_logits_final_residual_caches_and_teams():
    module = _runner_module()
    from trinote.infer_int.trace_bonsai35 import tensor_digest

    prompt = [5, 7]
    logits = np.asarray([[0, 1, 9, 3]], dtype=np.int64)
    residual = np.asarray([[11, 12]], dtype=np.int64)
    state = np.asarray([[[13]]], dtype=np.int64)
    conv = np.asarray([[14, 15]], dtype=np.int64)
    k = np.asarray([[[16], [17], [18], [19]]], dtype=np.int64)
    v = np.asarray([[[20], [21], [22], [23]]], dtype=np.int64)
    artifact = {"layers": [{"kind": "recurrent"}, {"kind": "attention"}]}
    final_layers = [
        {
            "layer": 0, "kind": "recurrent", "output": tensor_digest(residual),
            "cache": {"state": tensor_digest(state), "conv": tensor_digest(conv)},
        },
        {
            "layer": 1, "kind": "attention", "output": tensor_digest(residual),
            "cache": {"k": tensor_digest(k), "v": tensor_digest(v)},
        },
    ]
    expected = {
        "format": "trinote-bonsai35-trace/1",
        "inputIds": prompt,
        "outputIds": [2] * 32,
        "steps": [
            {"step": index, "token": 2, "logits": tensor_digest(logits),
             "layers": final_layers}
            for index in range(32)
        ],
    }

    class Executor:
        def __init__(self):
            self.position_value = 0
            self.prefills = self.decodes = self.teams = 0

        def stats(self):
            return {
                "prefill_calls": self.prefills,
                "decode_calls": self.decodes,
                "team_entries": self.teams,
                "selected_isa": 1,
            }

        def prefill_logits(self, ids):
            self.position_value = len(ids)
            self.prefills += 1
            self.teams += 1
            return logits.copy()

        def decode_logits(self, token):
            assert token == 2
            self.position_value += 1
            self.decodes += 1
            self.teams += 1
            return logits.copy()

        def decode(self, token):
            assert token == 2
            self.position_value += 1
            self.decodes += 1
            self.teams += 1
            return residual.copy()

        def export_last_residual(self):
            return residual.copy()

        def export_cache_tensor(self, layer, name):
            return {(0, "state"): state, (0, "conv"): conv,
                    (1, "k"): k, (1, "v"): v}[layer, name].copy()

        def position(self):
            return self.position_value

    report = module.verify_resident_cached_trace(artifact, expected, Executor())
    assert report["status"] == "pass"
    assert all(report["acceptance"].values())
    assert report["actualPosition"] == len(prompt) + 32
    assert report["residentCounterDelta"] == {
        "prefill_calls": 1, "decode_calls": 32, "team_entries": 33,
    }
    assert len(report["preconsumeLogits"]) == 32
    assert len(report["finalCaches"]) == 2

    expected["steps"][7]["logits"] = "00" * 32
    failed = module.verify_resident_cached_trace(artifact, expected, Executor())
    assert failed["status"] == "fail"
    assert not failed["acceptance"]["all_preconsume_logits_equal"]
    assert module.report_exit_code(failed) == 1
    assert module.report_exit_code(report) == 0


@pytest.mark.skipif(
    not _OPT_IN,
    reason="set TRINOTE_RUN_BONSAI35_REAL_TRACE_SUITE=1 for the external 4 GiB suite",
)
@pytest.mark.parametrize("producer", ["oracle", "native"])
def test_real_bonsai35_trace_suite_matches_expected_directory(producer):
    inputs = Path(os.environ.get("BONSAI35_REAL_TRACE_INPUTS", DEFAULT_INPUTS))
    expected = Path(
        os.environ.get(
            "BONSAI35_REAL_TRACE_EXPECTED_DIR",
            str(inputs.with_name(inputs.name.removesuffix(".inputs.json") + ".expected")),
        )
    )
    artifact = Path(
        os.environ.get(
            "BONSAI_INTEGER_27B_ARTIFACT",
            str(
                Path.home()
                / ".local/trinote/models/Bonsai-27B-Q1_0-int-qwen35.safetensors"
            ),
        )
    )
    assert inputs.is_file(), inputs
    assert (expected / "suite-manifest.json").is_file(), expected
    assert artifact.is_file(), artifact
    proc = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--mode",
            "verify",
            "--producer",
            producer,
            "--artifact",
            str(artifact),
            "--inputs",
            str(inputs),
            "--expected-dir",
            str(expected),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=7200,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["producer"] == producer
    assert len(report["verified"]) == 5


@pytest.mark.skipif(
    not _OPT_IN,
    reason="set TRINOTE_RUN_BONSAI35_REAL_TRACE_SUITE=1 for the external 4 GiB suite",
)
def test_real_bonsai35_resident_cached32_matches_oracle_expected_directory():
    inputs = Path(os.environ.get("BONSAI35_REAL_TRACE_INPUTS", DEFAULT_INPUTS))
    expected = Path(
        os.environ.get(
            "BONSAI35_REAL_TRACE_EXPECTED_DIR",
            str(inputs.with_name(inputs.name.removesuffix(".inputs.json") + ".expected")),
        )
    )
    artifact = Path(
        os.environ.get(
            "BONSAI_INTEGER_27B_ARTIFACT",
            str(Path.home() / ".local/trinote/models/Bonsai-27B-Q1_0-int-qwen35.safetensors"),
        )
    )
    proc = subprocess.run(
        [
            sys.executable, str(RUNNER), "--mode", "verify", "--producer", "resident",
            "--artifact", str(artifact), "--inputs", str(inputs),
            "--expected-dir", str(expected),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=7200,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["producer"] == "resident"
    assert report["verified"] == ["cached-greedy/chatHi/32"]
    assert report["status"] == "pass"
    assert all(report["acceptance"].values())
