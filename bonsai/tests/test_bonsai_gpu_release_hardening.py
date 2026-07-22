"""Static/offline gates for optional CUDA publication and large-model fetches."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "bonsai" / "tools" / "build_bonsai_q1_gpu.sh"
FETCH = ROOT / "bonsai" / "scripts" / "fetch_bonsai_27b_gguf.sh"
GPU_WORKFLOW = ROOT / ".github" / "workflows" / "gpu-ci.yml"


def test_fetcher_exposes_bounded_watchdog_and_segmented_resume():
    run = subprocess.run(
        [FETCH, "--dry-run", "--segments", "7"],
        text=True, capture_output=True, check=True,
    )
    assert "7 segment(s)" in run.stdout
    assert "retry-all-errors=8" in run.stdout
    assert "low-speed=1024B/s for 90s" in run.stdout

    source = FETCH.read_text()
    for contract in (
        "--retry-all-errors", "--retry-max-time", "--speed-limit", "--speed-time",
        "sha256sum", "EXPECTED_SIZE", 'mv -f "$PART" "$DEST"',
    ):
        assert contract in source


def test_fetcher_rejects_invalid_segment_count_before_network():
    run = subprocess.run(
        [FETCH, "--dry-run", "--segments", "0"],
        text=True, capture_output=True,
    )
    assert run.returncode == 2
    assert "segments must be a positive integer" in run.stderr


def test_gpu_build_rejects_invalid_arch_before_compilation(tmp_path):
    run = subprocess.run(
        [BUILD], env={**os.environ, "CUDA_ARCH": "native", "BONSAI_BIN_DIR": str(tmp_path)},
        text=True, capture_output=True,
    )
    assert run.returncode == 2
    assert "CUDA_ARCH must look like sm_86" in run.stderr
    assert not (tmp_path / "libbonsai_q1_gpu.so").exists()


def test_gpu_build_failed_symbol_probe_cannot_replace_known_good_library(tmp_path):
    bin_dir = tmp_path / "bin"
    tools = tmp_path / "tools"
    bin_dir.mkdir()
    tools.mkdir()
    output = bin_dir / "libbonsai_q1_gpu.so"
    output.write_bytes(b"known-good")

    nvcc = tools / "nvcc-fake"
    nvcc.write_text("#!/usr/bin/env bash\nprintf 'candidate' > \"${@: -1}\"\n")
    nvcc.chmod(0o755)
    nm = tools / "nm"
    nm.write_text("#!/usr/bin/env bash\nexit 0\n")
    nm.chmod(0o755)

    run = subprocess.run(
        [BUILD],
        env={
            **os.environ,
            "PATH": f"{tools}:{os.environ['PATH']}",
            "NVCC": str(nvcc),
            "CUDA_ARCH": "sm_86",
            "BONSAI_BIN_DIR": str(bin_dir),
            "BONSAI_SKIP_DEVICE_PROBE": "1",
        },
        text=True, capture_output=True,
    )
    assert run.returncode == 1
    assert "required dynamic symbol is missing" in run.stderr
    assert output.read_bytes() == b"known-good"
    assert not list(bin_dir.glob("libbonsai_q1_gpu.so.tmp.*"))


def test_self_hosted_gpu_workflow_uses_ephemeral_kernels_and_pinned_actions():
    source = GPU_WORKFLOW.read_text("utf-8")
    assert source.count("BONSAI_BIN_DIR: ${{ runner.temp }}/") == 2
    assert source.count("persist-credentials: false") == 2
    assert "actions/checkout@v" not in source
    assert "actions/upload-artifact@v" not in source
    assert "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683" in source
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in source
