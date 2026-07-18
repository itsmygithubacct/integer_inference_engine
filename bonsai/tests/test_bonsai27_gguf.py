"""Offline wiring tests for the optional Linux Bonsai-27B GGUF backend."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from trinote.cli.quality_gate_bonsai_cli import _run_teacher_harness
from trinote.cli.run_bonsai_cli import _run_prismml, main
from trinote.infer_int.gguf_tokenizer_v2 import (
    _persistent_tokenizer,
    llama_complete,
    llama_tokenize,
)
from trinote.infer_int.sampler import resolve_sampler


ROOT = Path(__file__).resolve().parents[2]


def test_llama_tokenize_uses_ids_and_preserves_literal_unicode(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="[17, 23, 248045]\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert llama_tokenize("café 東京 🌱\\n", "/model.gguf", bin_dir="/runtime") == [17, 23, 248045]
    assert seen["cmd"][-3:] == ["--ids", "--log-disable", "--no-escape"]
    assert seen["cmd"][seen["cmd"].index("-p") + 1] == "café 東京 🌱\\n"
    assert seen["kwargs"]["text"] is True


def test_quality_gate_offloads_both_prism_runs(monkeypatch):
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        if "prismml_teacher_forced" in str(cmd[0]):
            return SimpleNamespace(returncode=0, stdout='{"rows": []}\n', stderr="")
        return SimpleNamespace(returncode=0, stdout="continuation", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert llama_complete(
        "prompt", "/model.gguf", 2, bin_dir="/runtime", n_gpu_layers=99
    ) == "continuation"
    assert _run_teacher_harness(
        "/runtime/prismml_teacher_forced", gguf="/model.gguf", full_ids=[1, 2],
        start=1, ctx_size=128, threads=4, top_k=10, n_gpu_layers=99,
    ) == {"rows": []}
    for command in commands:
        flag = "--gpu-layers" if "teacher_forced" in str(command[0]) else "-ngl"
        assert command[command.index(flag) + 1] == "99"


def test_persistent_tokenizer_reuses_one_vocab_process(tmp_path, monkeypatch):
    server = tmp_path / "tokenizer-server"
    count = tmp_path / "starts"
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"fixture")
    server.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "open(os.environ['TOKENIZER_STARTS'], 'a').write('1')\n"
        "for line in sys.stdin:\n"
        "    line=line.strip()\n"
        "    if line == 'Q': break\n"
        "    print(json.dumps([len(bytes.fromhex(line))]), flush=True)\n"
    )
    server.chmod(0o755)
    monkeypatch.setenv("BONSAI_TOKENIZER_SERVER", str(server))
    monkeypatch.setenv("TOKENIZER_STARTS", str(count))
    _persistent_tokenizer.cache_clear()
    assert llama_tokenize("Hi", gguf, bin_dir=tmp_path) == [2]
    assert llama_tokenize("東京", gguf, bin_dir=tmp_path) == [6]
    helper = _persistent_tokenizer(str(server.resolve()), str(gguf.resolve()))
    helper.close()
    _persistent_tokenizer.cache_clear()
    assert count.read_text() == "1"


def test_bonsai27_sampler_preset_matches_model_card():
    cfg = resolve_sampler("bonsai27-rec", seed=9)
    assert (cfg.mode, cfg.temperature, cfg.top_k, cfg.top_p, cfg.min_p, cfg.seed) == (
        "top_p", 0.7, 20, 0.95, 0.0, 9
    )


def test_prismml_command_includes_bonsai27_cuda_controls(monkeypatch):
    seen = {}

    def fake_run(cmd):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "trinote.cli.run_bonsai_cli._GGUFReader",
        lambda _path: SimpleNamespace(kv={
            "general.architecture": "qwen35",
            "qwen35.context_length": 262_144,
        }),
    )
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    args = SimpleNamespace(
        bin_dir="/runtime/bin",
        gguf="/models/Bonsai-27B-Q1_0.gguf",
        max_new=24,
        prompt="hello",
        ctx_size=4096,
        n_gpu_layers=99,
        flash_attn=True,
        rep_penalty=1.0,
    )

    assert _run_prismml(args, resolve_sampler("bonsai27-rec", seed=3)) == 0
    cmd = seen["cmd"]
    for pair in (
        ["-c", "4096"], ["-ngl", "99"], ["-fa", "on"], ["--temp", "0.7"],
        ["--top-k", "20"], ["--top-p", "0.95"], ["--min-p", "0.0"], ["--seed", "3"],
    ):
        index = cmd.index(pair[0])
        assert cmd[index:index + 2] == pair
    assert cmd[cmd.index("-t") + 1] == "7"
    assert cmd[cmd.index("-p") + 1] == "hello"
    assert "--single-turn" in cmd


def test_prismml_default_context_uses_model_hardware_auto_fit(monkeypatch):
    seen = {}

    def fake_run(cmd):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "trinote.cli.run_bonsai_cli._GGUFReader",
        lambda _path: SimpleNamespace(kv={
            "general.architecture": "qwen35",
            "qwen35.context_length": 262_144,
        }),
    )
    args = SimpleNamespace(
        bin_dir="/runtime/bin", gguf="/models/Bonsai-27B-Q1_0.gguf",
        max_new=24, prompt=None, ctx_size=None, n_gpu_layers=99,
        flash_attn=True, rep_penalty=1.0,
    )
    assert _run_prismml(args, resolve_sampler("bonsai27-rec", seed=3)) == 0
    cmd = seen["cmd"]
    assert cmd[cmd.index("-c") + 1] == "0"
    assert cmd[cmd.index("--fit") + 1] == "on"
    assert "--conversation" in cmd


@pytest.mark.parametrize(
    "mode_args, expected",
    [
        (["--receipt"], "--receipt requires --engine native"),
        (["--no-receipt", "--onchain"], "--onchain requires --engine native"),
        (["--no-receipt", "--json"], "--json reproduction mode requires --engine native"),
    ],
)
def test_gguf_verification_modes_fail_closed(mode_args, expected, capsys):
    argv = ["--engine", "prismml.cpp", "-p", "hello", *mode_args]
    assert main(argv) == 2
    assert expected in capsys.readouterr().err


def test_linux_launcher_and_installers_expose_the_pins():
    env = {**os.environ, "BONSAI_DRYRUN": "1"}
    env.pop("BONSAI_CONTEXT_SIZE", None)
    env.pop("BONSAI_27B_CTX_SIZE", None)
    launch = subprocess.run(
        [ROOT / "bonsai-27b-cli", "hello"], env=env, text=True, capture_output=True, check=True
    ).stdout
    assert "Bonsai-27B-Q1_0.gguf" in launch
    assert "--context-size" not in launch
    assert "--n-gpu-layers 99" in launch

    override = subprocess.run(
        [ROOT / "bonsai-27b-cli", "hello"],
        env={**env, "BONSAI_CONTEXT_SIZE": "8192"},
        text=True, capture_output=True, check=True,
    ).stdout
    assert "--context-size 8192" in override

    runtime = subprocess.run(
        [ROOT / "bonsai" / "scripts" / "install_bonsai_27b_gguf.sh", "--dry-run"],
        text=True, capture_output=True, check=True,
    ).stdout
    assert "prism-b9591-62061f9" in runtime
    assert "67c64046abcf73bf489e27c9ebe7525f5b77c58db9490d1d711efe6e17bf2975" in runtime

    model = subprocess.run(
        [ROOT / "bonsai" / "scripts" / "fetch_bonsai_27b_gguf.sh", "--dry-run"],
        text=True, capture_output=True, check=True,
    ).stdout
    assert "0cf7e3d21581b169b4df1de8bf01316000e2fbb7" in model
    assert "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0" in model
