#!/usr/bin/env python3
"""Release-bound Bonsai-27B CUDA correctness and populated-cache acceptance.

The command writes exactly one machine-readable JSON document to stdout.  All
human progress goes to stderr.  It deliberately refuses artifacts other than
the committed Bonsai-27B integer release and exits nonzero when CUDA residency,
decode, cache/state parity, token parity, or graph-submission accounting fails.
With no mode flag it performs the 128-token pure-NumPy-oracle parity soak.
``--throughput-context 4096`` instead fills the real cache on GPU and measures
the final 32 decode steps without running the CPU oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FORMAT = "trinote-bonsai35-gpu-real-verification/1"
RELEASE_ARTIFACT_SHA256 = "7eab414ceff3fff1489053d415d0c6adb1e646e552d091cc1a898d0456adf3fb"
RAW_HI_TOKEN_ID = 12675
RAW_HI_EXPECTED_NEXT_GREEDY_ID = 11
DEFAULT_GENERATED_TOKENS = 128
MIN_THROUGHPUT_TOKENS_PER_SECOND = 10.0

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_ARTIFACT = 3
EXIT_PREFLIGHT = 4
EXIT_GPU_RUNTIME = 5
EXIT_PARITY = 6
EXIT_JSON_OUT = 7


class VerificationFailure(RuntimeError):
    def __init__(self, stage: str, message: str, exit_code: int):
        super().__init__(message)
        self.stage = stage
        self.exit_code = int(exit_code)


def _notary_home() -> Path:
    value = os.environ.get("BONSAI_NOTARY_HOME")
    return Path(value).expanduser() if value else Path.home() / ".local" / "trinote"


def default_artifact() -> Path:
    explicit = os.environ.get("BONSAI_INTEGER_27B_ARTIFACT")
    if explicit:
        return Path(explicit).expanduser()
    models = Path(os.environ.get("BONSAI_MODELS_DIR", _notary_home() / "models"))
    return models.expanduser() / "Bonsai-27B-Q1_0-int-qwen35.safetensors"


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=default_artifact(),
                        help="release integer artifact (digest is fixed by this tool)")
    parser.add_argument("--generated-tokens", type=positive_int,
                        default=DEFAULT_GENERATED_TOKENS,
                        help="greedy outputs in full CPU-parity mode (default: 128)")
    parser.add_argument(
        "--throughput-context",
        type=positive_int,
        help=(
            "GPU-only populated-cache mode: consume exactly this many tokens; "
            "use 4096 for the release performance gate"
        ),
    )
    parser.add_argument("--json-out", type=Path,
                        help="also atomically write the exact stdout JSON document")
    parser.add_argument("--verbose", action="store_true",
                        help="print a traceback to stderr for unexpected failures")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.throughput_context is not None
        and args.generated_tokens != DEFAULT_GENERATED_TOKENS
    ):
        parser.error("--generated-tokens is only valid in full CPU-parity mode")
    return args


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    """Durably replace ``path`` without exposing a partial JSON document."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (proc.stdout or proc.stderr).strip()
    return output or None


def build_runtime_identity(cuda_library: Path) -> dict[str, Any]:
    """Bind a result to the checkout, acceptance tool, and loaded CUDA ELF.

    The CUDA library path is the exact path selected by ``gpu_native``.  Its
    content digest is the load-bearing kernel identity; ELF build/compiler
    notes are retained as human-auditable build metadata.  A dirty checkout is
    surfaced rather than being silently represented by its HEAD revision.
    """

    status = _command_output(["git", "-C", str(ROOT), "status", "--porcelain=v1"])
    status_bytes = (status or "").encode("utf-8")
    revision = _command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    tool_path = Path(__file__).resolve()
    library_path = Path(cuda_library).expanduser().resolve()
    kernel: dict[str, Any] = {
        "path": str(library_path),
        "present": library_path.is_file(),
    }
    if library_path.is_file():
        notes = _command_output(["readelf", "-n", str(library_path)]) or ""
        build_id = re.search(r"Build ID:\s*([0-9a-fA-F]+)", notes)
        kernel.update({
            "sha256": sha256_file(library_path),
            "size_bytes": library_path.stat().st_size,
            "elf_build_id": build_id.group(1).lower() if build_id else None,
            "compiler_comment": _command_output(
                ["readelf", "-p", ".comment", str(library_path)]
            ),
        })
    return {
        "source": {
            "revision": revision,
            "dirty": bool(status),
            "porcelain_entry_count": len(status.splitlines()) if status else 0,
            "porcelain_sha256": hashlib.sha256(status_bytes).hexdigest(),
            "acceptance_tool_path": str(tool_path),
            "acceptance_tool_sha256": sha256_file(tool_path),
        },
        "cuda_kernel": kernel,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def array_parity_record(cpu: np.ndarray, gpu: np.ndarray) -> dict[str, Any]:
    cpu_array = np.asarray(cpu)
    gpu_array = np.asarray(gpu)
    return {
        "equal": bool(np.array_equal(cpu_array, gpu_array)),
        "cpu": {
            "dtype": str(cpu_array.dtype),
            "shape": list(cpu_array.shape),
            "sha256": array_sha256(cpu_array),
        },
        "gpu": {
            "dtype": str(gpu_array.dtype),
            "shape": list(gpu_array.shape),
            "sha256": array_sha256(gpu_array),
        },
    }


def int64_sequence_sha256(values) -> str:
    array = np.ascontiguousarray(np.asarray(list(values), dtype="<i8"))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def median_window(samples: list[float], start: int, stop: int) -> float | None:
    window = samples[start:min(stop, len(samples))]
    return float(statistics.median(window)) if window else None


def target_consumed_tokens(args: argparse.Namespace) -> int:
    return int(
        args.throughput_context
        if args.throughput_context is not None
        else args.generated_tokens
    )


def build_throughput_summary(
    *,
    target_context: int,
    generated_ids: list[int],
    step_seconds: list[float],
    stats: dict[str, Any],
    memory_samples: dict[str, dict[str, int] | None],
    proof_peak_used_bytes: int,
    ceiling_bytes: int,
) -> dict[str, Any]:
    """Build the stable GPU-only result block without requiring CUDA imports."""
    target = int(target_context)
    if target <= 0 or len(generated_ids) != target or len(step_seconds) != target:
        raise ValueError("throughput summary needs one generated ID and timing per consumed token")
    last_32 = [float(value) for value in step_seconds[-32:]]
    median_seconds = float(statistics.median(last_32))
    consumed_ids = [RAW_HI_TOKEN_ID, *generated_ids[:-1]]
    used_values = [
        int(sample["used_bytes"])
        for sample in memory_samples.values()
        if sample is not None and "used_bytes" in sample
    ]
    return {
        "target_context": target,
        "generated_ids": {
            "count": len(generated_ids),
            "first_8": generated_ids[:8],
            "last_8": generated_ids[-8:],
            "sha256_int64_le": int64_sequence_sha256(generated_ids),
        },
        "consumed_ids": {
            "count": len(consumed_ids),
            "first_8": consumed_ids[:8],
            "last_8": consumed_ids[-8:],
            "sha256_int64_le": int64_sequence_sha256(consumed_ids),
        },
        "stats": dict(stats),
        "timing": {
            "last_32_decode_seconds": last_32,
            "last_32_median_seconds": median_seconds,
            "last_32_median_tokens_per_second": 1.0 / median_seconds,
        },
        "memory": {
            "samples": memory_samples,
            "observed_peak_used_bytes": max(used_values) if used_values else None,
            "live_at_target_used_bytes": (
                None
                if memory_samples.get("at_target") is None
                else int(memory_samples["at_target"]["used_bytes"])
            ),
            "proof_peak_used_bytes": int(proof_peak_used_bytes),
            "ceiling_bytes": int(ceiling_bytes),
        },
    }


def throughput_acceptance(summary: dict[str, Any]) -> dict[str, bool]:
    target = int(summary["target_context"])
    stats = summary["stats"]
    median_tps = float(summary["timing"]["last_32_median_tokens_per_second"])
    memory = summary["memory"]
    return {
        "context_position_exact": int(stats.get("position", -1)) == target,
        "one_graph_submission_per_consumed_token": (
            int(stats.get("graph_launches", -1)) == target
        ),
        "graph_ready": stats.get("graph_ready") is True,
        "context_not_poisoned": stats.get("poisoned") is False,
        "token_id_input_mode": stats.get("input_mode") == "token_id",
        "device_embedding_only": (
            int(stats.get("token_input_submissions", -1)) == target
            and int(stats.get("embedded_input_submissions", -1)) == 0
        ),
        "model_input_host_bytes_exact": (
            int(stats.get("model_input_host_bytes", -1)) == target * 8
        ),
        "generated_id_count_exact": int(summary["generated_ids"]["count"]) == target,
        "consumed_id_count_exact": int(summary["consumed_ids"]["count"]) == target,
        "raw_hi_next_token_11": (
            bool(summary["generated_ids"]["first_8"])
            and int(summary["generated_ids"]["first_8"][0])
            == RAW_HI_EXPECTED_NEXT_GREEDY_ID
        ),
        "device_under_7_5_gib_ceiling": (
            int(memory["proof_peak_used_bytes"]) <= int(memory["ceiling_bytes"])
        ),
        "live_memory_queries_available": all(
            sample is not None for sample in memory["samples"].values()
        ),
        "last_32_at_least_10_tokens_per_second": (
            median_tps >= MIN_THROUGHPUT_TOKENS_PER_SECOND
        ),
    }


def _base_report(args: argparse.Namespace) -> dict[str, Any]:
    throughput = args.throughput_context is not None
    target = target_consumed_tokens(args)
    return {
        "format": FORMAT,
        "mode": "gpu_populated_context_throughput" if throughput else "full_cpu_oracle_parity",
        "status": "fail",
        "artifact": {
            "path": str(Path(args.artifact).expanduser().resolve()),
            "required_sha256": RELEASE_ARTIFACT_SHA256,
        },
        "request": {
            "prompt_name": "rawHi",
            "prompt_ids": [RAW_HI_TOKEN_ID],
            "generated_tokens": target,
            # One prompt step plus N-1 generated-token decode steps produces N
            # greedy outputs, hence the default is both 128 generated and 128
            # consumed graph submissions.
            "consumed_tokens": target,
        },
    }


def run_verification(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    from trinote.infer_int.artifact_io_bonsai import load_artifact_bonsai
    from trinote.infer_int.gpu_bonsai35 import (
        Bonsai35GpuExecutor,
        cpu_oracle_trace,
    )
    from trinote.infer_int import gpu_native

    gpu_memory_info = gpu_native.gpu_memory_info

    report = _base_report(args)
    report["runtime_identity"] = build_runtime_identity(gpu_native._LIB)
    throughput_mode = args.throughput_context is not None
    target_consumed = target_consumed_tokens(args)
    executor = None
    exit_code = EXIT_ERROR
    try:
        artifact_path = Path(args.artifact).expanduser().resolve()
        print(f"[gpu-real] loading {artifact_path}", file=sys.stderr, flush=True)
        load_started = time.perf_counter()
        try:
            artifact, info = load_artifact_bonsai(artifact_path)
        except Exception as exc:
            raise VerificationFailure(
                "artifact_load", f"{type(exc).__name__}: {exc}", EXIT_ARTIFACT
            ) from exc
        artifact_digest = str(info["digest"])
        report["artifact"].update({
            "actual_sha256": artifact_digest,
            "load_seconds": time.perf_counter() - load_started,
        })
        if artifact_digest != RELEASE_ARTIFACT_SHA256:
            raise VerificationFailure(
                "artifact_binding",
                f"artifact SHA-256 {artifact_digest} does not match release {RELEASE_ARTIFACT_SHA256}",
                EXIT_ARTIFACT,
            )
        cfg = artifact.get("config", {})
        if str(cfg.get("architecture")) != "qwen35" or len(artifact.get("layers", ())) != 64:
            raise VerificationFailure(
                "artifact_binding", "release artifact is not the 64-layer Qwen3.5 graph", EXIT_ARTIFACT
            )
        context = min(int(cfg["context_len"]), int(artifact["cos_fp"].shape[0]))
        if target_consumed > context:
            raise VerificationFailure(
                "request", f"{target_consumed} consumed tokens exceed context {context}", EXIT_ARTIFACT
            )
        report["artifact"]["config"] = {
            "architecture": str(cfg["architecture"]),
            "layers": len(artifact["layers"]),
            "context_len": context,
            "d_model": int(cfg["dModel"]),
            "vocab": int(cfg["vocab"]),
        }

        print("[gpu-real] proving residency and constructing CUDA graph", file=sys.stderr, flush=True)
        create_started = time.perf_counter()
        executor, feasibility = Bonsai35GpuExecutor.try_create_reported(
            artifact,
            # Full parity exports every post-layer residual.  The populated-
            # context throughput gate intentionally measures the production
            # graph without those diagnostic copy nodes.
            capture_trace=not throughput_mode,
        )
        report["gpu_feasibility"] = feasibility.as_dict()
        report["timing"] = {"executor_create_seconds": time.perf_counter() - create_started}
        if executor is None:
            raise VerificationFailure("gpu_preflight", feasibility.reason, EXIT_PREFLIGHT)
        memory_after_create = gpu_memory_info()

        consumed_ids = [RAW_HI_TOKEN_ID]
        generated_ids: list[int] = []
        step_seconds: list[float] = []
        decode_started = time.perf_counter()
        started = time.perf_counter()
        logits = executor.decode_token(RAW_HI_TOKEN_ID)
        step_seconds.append(time.perf_counter() - started)
        if logits is None:
            report["gpu_failure"] = {
                "stats": executor.stats(),
                "memory": gpu_memory_info(),
            }
            raise VerificationFailure("gpu_decode", "prompt decode failed/poisoned", EXIT_GPU_RUNTIME)
        memory_after_first_graph = gpu_memory_info() if throughput_mode else None

        progress_interval = 256 if throughput_mode else 16
        for output_index in range(target_consumed):
            token_id = int(np.argmax(logits))
            generated_ids.append(token_id)
            if output_index + 1 == target_consumed:
                break
            consumed_ids.append(token_id)
            started = time.perf_counter()
            logits = executor.decode_token(token_id)
            step_seconds.append(time.perf_counter() - started)
            if logits is None:
                report["gpu_failure"] = {
                    "stats": executor.stats(),
                    "memory": gpu_memory_info(),
                }
                raise VerificationFailure(
                    "gpu_decode",
                    f"decode failed/poisoned after {len(consumed_ids)} consumed tokens",
                    EXIT_GPU_RUNTIME,
                )
            if len(consumed_ids) % progress_interval == 0:
                print(
                    f"[gpu-real] GPU consumed {len(consumed_ids)}/{target_consumed}",
                    file=sys.stderr,
                    flush=True,
                )
        gpu_decode_seconds = time.perf_counter() - decode_started
        if len(consumed_ids) != target_consumed:
            raise VerificationFailure("gpu_decode", "internal consumed-token count drift", EXIT_GPU_RUNTIME)

        stats = executor.stats()
        graph_metadata = executor.graph_metadata()
        report["cuda_graph"] = graph_metadata
        print(
            "[gpu-real] captured graph "
            f"nodes={graph_metadata['graph_nodes']} "
            f"kernels={graph_metadata['kernel_nodes']} "
            f"memcpy={graph_metadata['memcpy_nodes']} "
            f"projection_grouping={'on' if graph_metadata['projection_grouping_enabled'] else 'off'} "
            f"trace={'on' if graph_metadata['trace_enabled'] else 'off'}",
            file=sys.stderr,
            flush=True,
        )
        memory_at_target = gpu_memory_info()
        if throughput_mode:
            throughput = build_throughput_summary(
                target_context=target_consumed,
                generated_ids=generated_ids,
                step_seconds=step_seconds,
                stats=stats,
                memory_samples={
                    "after_create": memory_after_create,
                    "after_first_graph": memory_after_first_graph,
                    "at_target": memory_at_target,
                },
                proof_peak_used_bytes=feasibility.peak_used_bytes,
                ceiling_bytes=feasibility.ceiling_bytes,
            )
            report["throughput"] = throughput
            report["timing"].update({
                "gpu_decode_total_seconds": gpu_decode_seconds,
                **throughput["timing"],
            })
            report["acceptance"] = throughput_acceptance(throughput)
            report["acceptance"]["diagnostic_trace_disabled"] = (
                graph_metadata["trace_enabled"] is False
                and graph_metadata["trace_copy_nodes_per_launch"] == 0
            )
            report["acceptance"]["projection_grouping_enabled"] = (
                graph_metadata["projection_grouping_enabled"] is True
                and graph_metadata["projection_kernel_nodes_saved_per_launch"] > 0
            )
            accepted = all(report["acceptance"].values())
            report["status"] = "pass" if accepted else "fail"
            # No CPU replay or multi-GiB cache export is part of this mode.
            executor.close()
            executor = None
            if not accepted:
                failed = [
                    name for name, passed in report["acceptance"].items() if not passed
                ]
                raise VerificationFailure(
                    "throughput_acceptance",
                    "failed gates: " + ", ".join(failed),
                    EXIT_PARITY,
                )
            exit_code = EXIT_OK
            return report, exit_code

        snapshot_started = time.perf_counter()
        gpu_snapshot = executor.debug_snapshot()
        snapshot_seconds = time.perf_counter() - snapshot_started
        if gpu_snapshot is None:
            raise VerificationFailure("gpu_export", "GPU cache/state export failed", EXIT_GPU_RUNTIME)
        gpu_logits = np.asarray(logits).copy()
        report["gpu"] = {
            "stats": stats,
            "generated_ids": {
                "count": len(generated_ids),
                "first_8": generated_ids[:8],
                "last_8": generated_ids[-8:],
                "sha256_int64_le": int64_sequence_sha256(generated_ids),
            },
        }
        # Snapshot/state/logits are now owned by the host.  Release the 7+ GiB
        # graph before the potentially long CPU replay rather than monopolize
        # the card while no further CUDA work is required.
        executor.close()
        executor = None
        report["gpu"]["memory_after_graph_release"] = gpu_memory_info()
        report["timing"].update({
            "gpu_decode_total_seconds": gpu_decode_seconds,
            "gpu_snapshot_export_seconds": snapshot_seconds,
            "gpu_first_graph_capture_seconds": step_seconds[0],
            "gpu_tokens_3_32_median_seconds": median_window(step_seconds, 2, 32),
            "gpu_tokens_33_128_median_seconds": median_window(step_seconds, 32, 128),
            "gpu_steady_median_seconds": median_window(step_seconds, 2, len(step_seconds)),
        })
        steady = report["timing"]["gpu_steady_median_seconds"]
        report["timing"]["gpu_steady_median_tokens_per_second"] = (
            None if steady is None else 1.0 / steady
        )

        print("[gpu-real] replaying all consumed tokens on the CPU oracle", file=sys.stderr, flush=True)
        oracle_started = time.perf_counter()
        cpu = cpu_oracle_trace(
            artifact,
            consumed_ids,
            capture_argmax=True,
            accelerated_native=False,
        )
        report["cpu_oracle"] = {"implementation": "canonical_numpy", "accelerated_native": False}
        report["timing"]["cpu_oracle_seconds"] = time.perf_counter() - oracle_started

        parity_started = time.perf_counter()
        parity = {
            key: array_parity_record(cpu[key], gpu_snapshot[key])
            for key in ("trace", "state", "conv", "k", "v")
        }
        parity["logits"] = array_parity_record(cpu["logits"], gpu_logits)
        cpu_argmax = np.asarray(cpu["argmax_ids"], dtype=np.int64).tolist()
        token_equal = cpu_argmax == generated_ids
        parity["generated_ids"] = {
            "equal": token_equal,
            "cpu_sha256_int64_le": int64_sequence_sha256(cpu_argmax),
            "gpu_sha256_int64_le": int64_sequence_sha256(generated_ids),
            "cpu_first_8": cpu_argmax[:8],
            "cpu_last_8": cpu_argmax[-8:],
        }
        report["parity"] = parity
        report["timing"]["parity_compare_seconds"] = time.perf_counter() - parity_started

        graph_ok = (
            int(stats.get("graph_launches", -1)) == len(consumed_ids)
            and int(stats.get("position", -1)) == len(consumed_ids)
            and stats.get("graph_ready") is True
            and stats.get("poisoned") is False
        )
        token_input_ok = (
            stats.get("input_mode") == "token_id"
            and int(stats.get("token_input_submissions", -1)) == len(consumed_ids)
            and int(stats.get("embedded_input_submissions", -1)) == 0
            and int(stats.get("model_input_host_bytes", -1)) == len(consumed_ids) * 8
        )
        first_token_ok = bool(generated_ids) and generated_ids[0] == RAW_HI_EXPECTED_NEXT_GREEDY_ID
        arrays_ok = all(parity[key]["equal"] for key in ("trace", "state", "conv", "k", "v", "logits"))
        report["acceptance"] = {
            "artifact_bound": True,
            "array_parity": arrays_ok,
            "generated_token_parity": token_equal,
            "raw_hi_next_token_11": first_token_ok,
            "one_graph_submission_per_consumed_token": graph_ok,
            "device_resident_token_embedding": token_input_ok,
            "device_under_7_5_gib_ceiling": bool(feasibility.feasible),
        }
        accepted = all(report["acceptance"].values())
        report["status"] = "pass" if accepted else "fail"
        if not accepted:
            failed = [name for name, passed in report["acceptance"].items() if not passed]
            raise VerificationFailure(
                "acceptance", "failed gates: " + ", ".join(failed), EXIT_PARITY
            )
        exit_code = EXIT_OK
    except VerificationFailure as exc:
        report["status"] = "fail"
        report["failure"] = {"stage": exc.stage, "message": str(exc)}
        exit_code = exc.exit_code
    except Exception as exc:  # keep stdout machine-readable even for unexpected failures
        report["status"] = "fail"
        report["failure"] = {
            "stage": "unexpected",
            "message": f"{type(exc).__name__}: {exc}",
        }
        if args.verbose:
            traceback.print_exc(file=sys.stderr)
        exit_code = EXIT_ERROR
    finally:
        if executor is not None:
            executor.close()
        try:
            report["post_cleanup_gpu_memory"] = gpu_memory_info()
        except Exception as exc:
            report["post_cleanup_gpu_memory_error"] = f"{type(exc).__name__}: {exc}"
    return report, exit_code


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report, exit_code = run_verification(args)
    payload = canonical_json_bytes(report)
    if args.json_out is not None:
        try:
            atomic_write(args.json_out, payload)
        except Exception as exc:
            report["status"] = "fail"
            report["failure"] = {
                "stage": "json_out",
                "message": f"cannot atomically write {args.json_out}: {type(exc).__name__}: {exc}",
            }
            payload = canonical_json_bytes(report)
            exit_code = EXIT_JSON_OUT
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
