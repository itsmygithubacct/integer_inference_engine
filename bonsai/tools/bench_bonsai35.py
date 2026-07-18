#!/usr/bin/env python3
"""Controlled Bonsai-27B/Qwen3.5 integer benchmark.

The controller deliberately imports only the Python standard library.  It
sets thread-pool policy, checks host contamination, pins an affinity set, and
then starts one or more workers.  NumPy and the native kernel are imported
only inside a worker, after that policy is in the environment.

Raw results are always written outside the checkout by default, under
``$BONSAI_BENCHMARKS_DIR/results/bonsai35`` (normally
``~/.local/trinote/benchmarks/results/bonsai35``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORMAT = "trinote-bonsai35-benchmark/1"
SOURCE_IDENTITY_FILES = (
    "tools/bench_bonsai35.py",
    "tools/bonsai_q1_kernel.c",
    "src/trinote/infer_int/q1_native.py",
    "src/trinote/infer_int/reference_bonsai35.py",
    "src/trinote/infer_int/reference_bonsai.py",
    "src/trinote/infer_int/sampler.py",
    "src/trinote/infer_int/trace_bonsai35.py",
    "src/trinote/infer_int/artifact_io_bonsai.py",
    "src/trinote/determinism/fixedpoint.py",
)
PERF_EVENTS = (
    "cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,"
    "LLC-loads,LLC-load-misses,branches,branch-misses,"
    "stalled-cycles-frontend,stalled-cycles-backend,context-switches,cpu-migrations,"
    "page-faults,minor-faults,major-faults"
)
THREAD_ENV_KEYS = (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_WAIT_POLICY", "OMP_PLACES",
    "OMP_PROC_BIND", "OMP_MAX_ACTIVE_LEVELS", "GOMP_SPINCOUNT", "KMP_BLOCKTIME",
    "TRINOTE_BONSAI35_Q1_CHUNK", "TRINOTE_Q1_LUT32",
)

# The frozen lane reconstructs the architecture measured in the optimization
# plan's 15.3--16.1 second diagnostic profile.  It intentionally does not try
# to recreate the contaminated host load or the old source/binary itself.
LEGACY_NATIVE_PRODUCER = "legacy-native"
LEGACY_NATIVE_ENVIRONMENT = {
    # Keep Python in charge of the graph and semantic cache.
    "TRINOTE_BONSAI35_MODEL_EXECUTOR": "0",
    # Reproduce two native entries per projection group: prepare, then
    # prepared-multi apply.  The later prepare+apply ABI must not be selected.
    "TRINOTE_BONSAI35_Q1_FUSED": "0",
    # The captured profile used the uint64 activation LUT and int64 scales.
    "TRINOTE_BONSAI35_Q1_LUT32": "0",
    "TRINOTE_Q1_LUT32": "0",
    "TRINOTE_Q1_SCALE_CACHE": "0",
    # The post-profile native GDN primitive must fall back to the original
    # Python/NumPy recurrent update.
    "TRINOTE_BONSAI35_NATIVE_GDN": "0",
    # These separately dispatched primitives were already present in the
    # captured profile.  Force them on so an ambient debug env cannot turn the
    # lane into a different, slower pseudo-baseline.
    "TRINOTE_NATIVE_RMSNORM": "1",
    "TRINOTE_NATIVE_SILU": "1",
    "TRINOTE_NATIVE_ATTN": "1",
    "TRINOTE_Q1_PREPARED_MULTI": "1",
    # The later AVX2 dispatcher only accelerates the guarded narrow-LUT path,
    # which is disabled above.  Portable is nevertheless forced to make the
    # replay contract explicit and future-proof.
    "TRINOTE_Q1_ISA": "portable",
    # This is a CPU comparison lane even if the calling shell opted into a
    # generic Bonsai GPU producer.
    "TRINOTE_GPU": "0",
    "TRINOTE_GPU_FULL": "0",
    "TRINOTE_GPU_RESIDENT_BATCH": "0",
    "BONSAI_VERIFY_GPU": "0",
}
PRODUCER_ENV_KEYS = tuple(LEGACY_NATIVE_ENVIRONMENT)

LEGACY_NATIVE_PROFILE = {
    "name": "bonsai35-pre-fusion-python-native-primitives",
    "historical_diagnostic_wall_s": {"minimum": 15.3, "maximum": 16.1},
    "historical_profile_characteristics": {
        "python_graph_orchestration": True,
        "native_packed_q1": True,
        "layer_q1_groups_per_hidden_step": 256,
        "layer_q1_weight_projections_per_hidden_step": 496,
        "output_q1_argmax_calls_per_generated_token": 1,
        "separate_q1_prepare_and_apply_entries": True,
        "native_rmsnorm_silu_and_decode_attention": True,
        "native_gdn": False,
        "resident_model_executor": False,
        "fused_q1_prepare_apply": False,
        "guarded_int32_activation_lut": False,
    },
    "reconstruction_limits": [
        "uses the current source, native library, compiler, Python, and NumPy identities recorded in this JSON",
        "uses controlled pinning/passive thread pools instead of recreating the historically contaminated host",
        "the 15.3--16.1 second range is diagnostic context, not an acceptance band for this matched run",
    ],
}


def configure_producer_environment(producer: str) -> dict[str, Any]:
    """Force and report the execution controls for a benchmark producer."""
    forced = LEGACY_NATIVE_ENVIRONMENT if producer == LEGACY_NATIVE_PRODUCER else {}
    before = {key: os.environ.get(key) for key in forced}
    for key, value in forced.items():
        os.environ[key] = value
    effective = {key: os.environ.get(key) for key in forced}
    return {
        "producer": producer,
        "forced": dict(forced),
        "previous": before,
        "effective": effective,
        "profile": dict(LEGACY_NATIVE_PROFILE) if forced else None,
    }


def _notary_home() -> Path:
    value = os.environ.get("BONSAI_NOTARY_HOME")
    return Path(value).expanduser() if value else Path.home() / ".local" / "trinote"


def _default_paths() -> tuple[str, str, str, str]:
    home = _notary_home()
    models = Path(os.environ.get("BONSAI_MODELS_DIR", home / "models")).expanduser()
    release = "prism-b9591-62061f9"
    bin_dir = Path(os.environ.get(
        "BONSAI_27B_BIN_DIR", home / "vendor" / "llama.cpp-bonsai27" / release / "bin"
    )).expanduser()
    results = Path(os.environ.get("BONSAI_BENCHMARKS_DIR", home / "benchmarks")).expanduser() / "results"
    return (
        str(models / "Bonsai-27B-Q1_0.gguf"),
        str(models / "Bonsai-27B-Q1_0-int-qwen35.safetensors"),
        str(bin_dir),
        str(results / "bonsai35"),
    )


def _physical_cpu_map() -> dict[tuple[int, int], list[int]]:
    mapping: dict[tuple[int, int], list[int]] = {}
    allowed = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
    cpu_root = Path("/sys/devices/system/cpu")
    for cpu_dir in cpu_root.glob("cpu[0-9]*"):
        try:
            cpu = int(cpu_dir.name[3:])
            if cpu not in allowed:
                continue
            topo = cpu_dir / "topology"
            package = int((topo / "physical_package_id").read_text().strip())
            core = int((topo / "core_id").read_text().strip())
        except (OSError, ValueError):
            continue
        mapping.setdefault((package, core), []).append(cpu)
    if not mapping:
        for cpu in sorted(allowed):
            mapping[(0, cpu)] = [cpu]
    for siblings in mapping.values():
        siblings.sort()
    return mapping


def _affinity_for_mode(mode: str) -> list[int]:
    topology = _physical_cpu_map()
    if mode == "physical":
        return sorted(siblings[0] for siblings in topology.values())
    return sorted(cpu for siblings in topology.values() for cpu in siblings)


def configure_thread_environment(threads: int, prefill_chunk: int) -> dict[str, str]:
    physical = max(1, len(_physical_cpu_map()))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    if threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(threads)
    else:
        os.environ.setdefault("OMP_NUM_THREADS", str(physical))
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("OMP_PLACES", "cores")
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
    os.environ.setdefault("GOMP_SPINCOUNT", "0")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    if prefill_chunk > 0:
        os.environ["TRINOTE_BONSAI35_Q1_CHUNK"] = str(prefill_chunk)
    return {key: os.environ.get(key, "") for key in THREAD_ENV_KEYS}


def _read_proc_cpu() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    values = [int(x) for x in fields]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def sample_host_cpu(seconds: float) -> float:
    total0, idle0 = _read_proc_cpu()
    time.sleep(max(0.05, seconds))
    total1, idle1 = _read_proc_cpu()
    elapsed = total1 - total0
    return 0.0 if elapsed <= 0 else 100.0 * (elapsed - (idle1 - idle0)) / elapsed


def _ancestor_pids() -> set[int]:
    out: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in out:
        out.add(pid)
        try:
            fields = (Path("/proc") / str(pid) / "stat").read_text().split()
            pid = int(fields[3])
        except (OSError, ValueError, IndexError):
            break
    return out


def find_other_native_engines() -> list[dict[str, Any]]:
    ancestors = _ancestor_pids()
    found: list[dict[str, Any]] = []
    for entry in Path("/proc").glob("[0-9]*"):
        pid = int(entry.name)
        if pid in ancestors:
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except (OSError, PermissionError):
            continue
        if ("trinote.cli.run_bonsai_cli" in cmd and "--engine native" in cmd) or "bonsai-integer-27b-cli" in cmd:
            found.append({"pid": pid, "command": cmd[:500]})
    return found


def _sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _command_output(command: list[str], timeout: float = 10.0) -> str | None:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (proc.stdout or proc.stderr).strip()
    return text if text else None


def _git_identity(root: Path) -> dict[str, Any]:
    revision = _command_output(["git", "-C", str(root), "rev-parse", "HEAD"])
    dirty_text = _command_output(
        ["git", "-C", str(root), "status", "--porcelain=v1"]
    )
    porcelain = dirty_text or ""
    return {
        "revision": revision,
        "dirty": bool(porcelain),
        "porcelain_entry_count": len(porcelain.splitlines()),
        "porcelain_sha256": hashlib.sha256(porcelain.encode("utf-8")).hexdigest(),
    }


def _source_file_identity(root: Path) -> dict[str, dict[str, Any]]:
    """Bind dirty runtime sources by content, not only porcelain path status."""

    result: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_IDENTITY_FILES:
        path = root / relative
        result[relative] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def _cpu_identity() -> dict[str, Any]:
    model = None
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    topology = _physical_cpu_map()
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    return {
        "model": model,
        "logical_cpus": os.cpu_count(),
        "physical_cores_in_affinity": len(topology),
        "affinity": allowed,
        "numa_nodes": len(list(Path("/sys/devices/system/node").glob("node[0-9]*"))),
    }


def _gpu_identity() -> list[dict[str, Any]]:
    query = "index,name,uuid,driver_version,memory.total"
    text = _command_output(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    if not text:
        return []
    keys = query.split(",")
    return [dict(zip(keys, (part.strip() for part in line.split(",")))) for line in text.splitlines()]


def _gpu_process_memory_mib() -> float | None:
    query = "pid,used_gpu_memory"
    text = _command_output([
        "nvidia-smi", f"--query-compute-apps={query}", "--format=csv,noheader,nounits"
    ])
    if not text:
        return None
    total = 0.0
    seen = False
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(os.getpid()):
            try:
                total += float(parts[1])
                seen = True
            except ValueError:
                pass
    return total if seen else 0.0


def _kernel_identity(repo_root: Path) -> dict[str, Any]:
    home = _notary_home()
    configured = os.environ.get("BONSAI_BIN_DIR")
    candidates = [Path(configured) / "libbonsai_q1_kernel.so"] if configured else []
    candidates += [home / "bin" / "libbonsai_q1_kernel.so", repo_root / "tools" / "libbonsai_q1_kernel.so"]
    path = next((p for p in candidates if p.exists()), candidates[0] if candidates else Path(""))
    if not path.exists():
        return {"path": str(path), "present": False}
    notes = _command_output(["readelf", "-n", str(path)]) or ""
    match = re.search(r"Build ID:\s*([0-9a-fA-F]+)", notes)
    comments = _command_output(["readelf", "-p", ".comment", str(path)])
    return {
        "path": str(path.resolve()), "present": True, "sha256": _sha256(path),
        "elf_build_id": match.group(1).lower() if match else None,
        "compiler_comment": comments,
    }


def environment_identity(repo_root: Path) -> dict[str, Any]:
    try:
        import numpy as np
        numpy_version = np.__version__
    except ImportError:
        numpy_version = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "platform": platform.platform(),
        "python": platform.python_version(), "numpy": numpy_version,
        "source": {
            **_git_identity(repo_root),
            "files": _source_file_identity(repo_root),
        },
        "kernel": _kernel_identity(repo_root),
        "cpu": _cpu_identity(), "gpus": _gpu_identity(),
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
        "producer_environment": {key: os.environ.get(key) for key in PRODUCER_ENV_KEYS},
    }


def _current_rss_kib(status_path: Path | None = None) -> int | None:
    """Return this process's current resident set, distinct from ru_maxrss."""

    path = status_path if status_path is not None else Path("/proc/self/status")
    try:
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) >= 2:
                    return int(fields[1])
    except (OSError, ValueError):
        pass
    return None


def _usage_snapshot() -> tuple[resource.struct_rusage, float, float, int | None]:
    return (
        resource.getrusage(resource.RUSAGE_SELF),
        time.monotonic(),
        _gpu_process_memory_mib() or 0.0,
        _current_rss_kib(),
    )


def _usage_delta(before, after) -> dict[str, Any]:
    r0, wall0, gpu0, rss0 = before
    r1, wall1, gpu1, rss1 = after
    wall = max(0.0, wall1 - wall0)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    return {
        "wall_s": wall, "user_cpu_s": r1.ru_utime - r0.ru_utime,
        "system_cpu_s": r1.ru_stime - r0.ru_stime,
        "process_cpu_percent_of_one_core": 0.0 if wall <= 0 else cpu / wall * 100.0,
        "peak_rss_kib": int(r1.ru_maxrss),
        "current_rss_kib_before": rss0,
        "current_rss_kib_after": rss1,
        "minor_page_faults": int(r1.ru_minflt - r0.ru_minflt),
        "major_page_faults": int(r1.ru_majflt - r0.ru_majflt),
        "voluntary_context_switches": int(r1.ru_nvcsw - r0.ru_nvcsw),
        "involuntary_context_switches": int(r1.ru_nivcsw - r0.ru_nivcsw),
        "device_memory_mib_before": gpu0, "device_memory_mib_after": gpu1,
    }


def _metric_summary(values: list[float | None]) -> dict[str, Any] | None:
    clean = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not clean:
        return None
    n = len(clean)

    def quantile(q: float) -> float:
        if n == 1:
            return clean[0]
        pos = (n - 1) * q
        low = int(math.floor(pos))
        high = int(math.ceil(pos))
        return clean[low] + (clean[high] - clean[low]) * (pos - low)

    mean = statistics.fmean(clean)
    stdev = statistics.pstdev(clean) if n > 1 else 0.0
    return {
        "n": n, "median": statistics.median(clean), "p10": quantile(0.10), "p90": quantile(0.90),
        "mean": mean, "coefficient_of_variation": 0.0 if mean == 0 else stdev / abs(mean),
        "min": clean[0], "max": clean[-1],
    }


def _model_iteration(model, gguf_reader, args) -> dict[str, Any]:
    import numpy as np
    from trinote.cli.run_bonsai_cli import _qwen3_chat_prompt
    from trinote.infer_int.gguf_tokenizer_v2 import llama_tokenize
    from trinote.infer_int.reference_bonsai import _rmsnorm
    from trinote.infer_int.reference_bonsai35 import _Qwen35Cache
    from trinote.infer_int.sampler import resolve_sampler, sample_token
    from trinote.infer_int.trace_bonsai35 import (
        CACHE_COMMITMENT_FORMAT,
        canonical_cache_digest,
        canonical_cache_record,
    )

    tokenizer_start = time.monotonic()
    if args.raw_ids:
        input_ids = [int(part) for part in args.raw_ids.split(",") if part.strip()]
    else:
        text = _qwen3_chat_prompt(args.prompt, gguf_reader.kv) if args.chat else args.prompt
        input_ids = llama_tokenize(text, args.gguf, bin_dir=args.bin_dir)
    tokenizer_s = time.monotonic() - tokenizer_start
    if not input_ids:
        raise ValueError("benchmark input contains no token ids")

    cfg = resolve_sampler(args.sampler, temperature=0.7, top_k=20, top_p=0.95,
                          min_p=0.0, seed=args.seed, rep_penalty=0, no_repeat_ngram=0)
    frac = int(model.cfg["frac"])
    eps = int(model.cfg.get("rmsEpsilonFp2", 1))
    resident = model._model_executor if args.producer == "native" else None
    if args.producer == "native" and resident is None:
        raise RuntimeError(
            "native model benchmark requires the resident Bonsai-27B executor ABI"
        )
    if args.producer == LEGACY_NATIVE_PRODUCER:
        if not model._native:
            raise RuntimeError("legacy-native benchmark requires the packed native Q1 library")
        if model._model_executor is not None:
            raise RuntimeError("legacy-native benchmark must not create the resident model executor")
        runtime = model._native_runtime
        if runtime is None or runtime.fused or runtime.lut32_mode not in {
            "0", "false", "no", "off",
        }:
            raise RuntimeError(
                "legacy-native benchmark did not select separate uint64 prepare/apply primitives"
            )
    # Every resident prefill ABI is a fresh semantic transaction: the C entry
    # point calls bonsai35_model_run(..., reset_first=1), so it clears state,
    # convolution history, and position inside the measured prefill call.  Do
    # not pre-reset here: doing so would touch hundreds of MiB immediately
    # before the timed reset and bias TTFT.  The process, descriptors, packed
    # weights, arenas, and native handle still remain warm across iterations.
    cache = None if resident is not None else _Qwen35Cache(len(model.artifact["layers"]))
    resident_stats0 = resident.stats() if resident is not None else None
    usage0 = _usage_snapshot()
    compute_start = time.monotonic()
    prefill_start = time.monotonic()
    if resident is None:
        x = model._run_layers(input_ids, cache)
        resident_token = resident_row = None
    elif cfg.mode == "greedy":
        resident_token = resident.prefill_argmax(input_ids)
        resident_row = x = None
    else:
        resident_row = resident.prefill_logits(input_ids)[0]
        resident_token = x = None
    prefill_s = time.monotonic() - prefill_start
    history = list(input_ids)
    output_ids: list[int] = []
    graph_times: list[float] = []
    norm_times: list[float] = []
    projection_times: list[float] = []
    sampling_times: list[float] = []
    token_times: list[float] = []
    eos = int(gguf_reader.kv.get("tokenizer.ggml.eos_token_id", -1))

    for step in range(args.max_new):
        token_start = time.monotonic()
        graph_s = 0.0
        if resident is not None and step:
            graph_start = time.monotonic()
            if cfg.mode == "greedy":
                resident_token = resident.decode_argmax(output_ids[-1])
            else:
                resident_row = resident.decode_logits(output_ids[-1])[0]
            graph_s = time.monotonic() - graph_start
        elif resident is None and step:
            graph_start = time.monotonic()
            x = model._run_layers([output_ids[-1]], cache)
            graph_s = time.monotonic() - graph_start
        if resident is not None:
            # Final norm + output projection/argmax are part of the single
            # resident ABI/team call and therefore belong to prefill/graph_s.
            norm_s = projection_s = 0.0
            if cfg.mode == "greedy":
                token = int(resident_token)
                sampling_s = 0.0
            else:
                sampling_start = time.monotonic()
                token = int(sample_token(
                    resident_row, cfg, len(history), frac, history_ids=history
                ))
                sampling_s = time.monotonic() - sampling_start
        else:
            norm_start = time.monotonic()
            last = _rmsnorm(x[-1:], frac, model.artifact["final_norm_gain_fp"],
                            native=model._native, eps=eps)
            norm_s = time.monotonic() - norm_start
            projection_start = time.monotonic()
            if cfg.mode == "greedy":
                token = int(model._output_argmax(last)[0])
                projection_s = time.monotonic() - projection_start
                sampling_s = 0.0
            else:
                row = model._output_linear(last)[0]
                projection_s = time.monotonic() - projection_start
                sampling_start = time.monotonic()
                token = int(sample_token(row, cfg, len(history), frac, history_ids=history))
                sampling_s = time.monotonic() - sampling_start
        output_ids.append(token)
        history.append(token)
        graph_times.append(graph_s)
        norm_times.append(norm_s)
        projection_times.append(projection_s)
        sampling_times.append(sampling_s)
        token_times.append(time.monotonic() - token_start)
        if not args.ignore_eos and eos >= 0 and token == eos:
            break

    compute_s = time.monotonic() - compute_start
    usage1 = _usage_snapshot()
    if resident is not None:
        cache_position = resident.position()
        resident_stats1 = resident.stats()
        resident_counters = {
            key: resident_stats1[key] - resident_stats0[key]
            for key in (
                "decode_calls", "prefill_calls", "team_entries", "q1_groups",
                "lut32_hits", "lut32_fallbacks", "lut64_groups",
                "layer_major_prefills", "layer_major_rows",
                "prefill_tiles_40", "prefill_tiles_48", "prefill_tiles_136",
            )
        }
        resident_counters.update({
            key: resident_stats1[key]
            for key in (
                "last_team_size", "selected_isa", "selected_lut_bits",
                "cache_width_bits", "prefill_tile_40", "prefill_tile_48",
                "prefill_tile_136",
            )
        })
        runtime_name = "resident-model-executor"
        tensor_for = resident.export_cache_tensor
        last_residual = resident.export_last_residual()
    else:
        cache_position = int(cache.t)
        resident_counters = None
        runtime_name = (
            "python-native-primitives-legacy-baseline"
            if args.producer == LEGACY_NATIVE_PRODUCER
            else "python-oracle-graph"
        )

        def tensor_for(layer_index, name):
            value = getattr(cache, name)[layer_index]
            if value is None:
                raise RuntimeError(
                    f"oracle cache layer {layer_index} {name} was not populated"
                )
            return value

        last_residual = x
    cache_record = canonical_cache_record(
        [str(layer["kind"]) for layer in model.artifact["layers"]],
        position=cache_position,
        tensor_for=tensor_for,
        last_residual=last_residual,
    )
    cache_digest = canonical_cache_digest(cache_record)
    output_payload = np.asarray(output_ids, dtype="<i8").tobytes()
    first_output_stage = token_times[0] if token_times else None
    first_projection_sampling = (
        projection_times[0] + sampling_times[0] if projection_times else None
    )
    ttft = prefill_s + first_output_stage if first_output_stage is not None else None

    def median_slice(start: int, stop: int) -> float | None:
        values = token_times[start:stop]
        return statistics.median(values) if values else None

    return {
        "input_ids": input_ids, "output_ids": output_ids,
        "commitments": {
            "output_ids_sha256": hashlib.sha256(output_payload).hexdigest(),
            # Backward-compatible key, now defined by a producer-independent
            # canonical state/conv/K/V/position/final-residual record.
            "layer_trace_sha256": cache_digest,
            "cache_state_sha256": cache_digest,
            "cache_state_format": CACHE_COMMITMENT_FORMAT,
            "cache_position": cache_position,
        },
        "runtime": runtime_name,
        "resident_executor_counters": resident_counters,
        "input_token_count": len(input_ids), "output_token_count": len(output_ids),
        "timing": {
            "tokenizer_subprocess_s": tokenizer_s, "prompt_prefill_s": prefill_s,
            "prompt_prefill_tokens_per_s": len(input_ids) / prefill_s if prefill_s else None,
            "time_to_first_output_token_compute_s": ttft,
            "time_to_first_output_token_end_to_end_s": tokenizer_s + ttft if ttft is not None else None,
            "first_output_projection_sampling_s": first_projection_sampling,
            "first_decode_token_after_prefill_s": token_times[1] if len(token_times) > 1 else None,
            "steady_decode_token_3_32_median_s": median_slice(2, 32),
            "steady_decode_token_33_128_median_s": median_slice(32, 128),
            "steady_decode_token_3_32_tokens_per_s": (
                1.0 / median_slice(2, 32) if median_slice(2, 32) else None
            ),
            "steady_decode_token_33_128_tokens_per_s": (
                1.0 / median_slice(32, 128) if median_slice(32, 128) else None
            ),
            "generation_compute_s": compute_s,
            "output_projection_s_total": sum(projection_times),
            "sampling_s_total": sum(sampling_times),
            "final_norm_s_total": sum(norm_times),
            "decode_graph_s_total": sum(graph_times),
            "per_output_token_s": token_times,
            "per_output_projection_s": projection_times,
            "per_sampling_s": sampling_times,
        },
        "resources": _usage_delta(usage0, usage1),
    }


def _q1_iteration(args) -> dict[str, Any]:
    import numpy as np
    from trinote.infer_int.q1_native import (
        q1_linear_prepared_native, q1_native_available, q1_prepare_native,
    )

    if not q1_native_available():
        raise RuntimeError("native Q1 kernel is unavailable")
    if args.q1_in_features % 128:
        raise ValueError("--q1-in-features must be divisible by 128")
    n_blocks = args.q1_in_features // 128
    x = np.arange(args.q1_tokens * args.q1_in_features, dtype=np.int64).reshape(
        args.q1_tokens, args.q1_in_features
    )
    x = (x % 4097) - 2048
    bits = np.arange(args.q1_out_features * n_blocks * 16, dtype=np.uint8).reshape(
        args.q1_out_features, n_blocks, 16
    )
    scales = np.full((args.q1_out_features, n_blocks), 257, dtype=np.int32)
    usage0 = _usage_snapshot()
    durations: list[float] = []
    output_digest = None
    prepare_calls = apply_calls = 0
    prepared = None
    if args.mode == "q1-apply":
        prepared = q1_prepare_native(x, n_blocks, lut32=args.q1_lut32)
        if prepared is None:
            raise RuntimeError("Q1 preparation failed or requested LUT32 was outside the exact envelope")
        prepare_calls += 1
    for _ in range(args.q1_iterations):
        started = time.monotonic()
        if args.mode == "q1-prepare":
            prepared = q1_prepare_native(x, n_blocks, lut32=args.q1_lut32)
            prepare_calls += 1
            if prepared is None:
                raise RuntimeError("Q1 preparation failed or requested LUT32 was outside the exact envelope")
            payload = prepared.lut
        else:
            payload = q1_linear_prepared_native(prepared, bits, scales, 16)
            apply_calls += 1
            if payload is None:
                raise RuntimeError("Q1 prepared apply is unavailable")
        durations.append(time.monotonic() - started)
        output_digest = hashlib.sha256(np.ascontiguousarray(payload).view(np.uint8)).hexdigest()
    usage1 = _usage_snapshot()
    lut_itemsize = 4 if args.q1_lut32 else 8
    lut_bytes = args.q1_tokens * n_blocks * 16 * 256 * lut_itemsize
    return {
        "shape": {"tokens": args.q1_tokens, "in_features": args.q1_in_features,
                  "out_features": args.q1_out_features, "n_blocks": n_blocks},
        "variant": {"lut32": args.q1_lut32, "scale_dtype": "int32"},
        "timing": {"per_call_s": durations, "median_s": statistics.median(durations)},
        "logical_counters": {
            "prepare_calls": prepare_calls, "apply_calls": apply_calls,
            "estimated_openmp_regions": prepare_calls + apply_calls,
            "lut_bytes_per_prepare": lut_bytes,
            "barrier_time_s": None,
            "barrier_time_status": "requires an OMPT-instrumented OpenMP runtime",
        },
        "output_sha256": output_digest, "resources": _usage_delta(usage0, usage1),
    }


def _load_model_worker(args, repo_root: Path):
    from trinote.infer_int.artifact_io_bonsai import load_artifact_bonsai
    from trinote.infer_int.import_gguf_v2 import _GGUFReader
    from trinote.infer_int.reference_bonsai35 import BonsaiQwen35ReferenceModel

    gguf_start = time.monotonic()
    gguf_reader = _GGUFReader(args.gguf)
    gguf_metadata_load_s = time.monotonic() - gguf_start
    load_start = time.monotonic()
    artifact, info = load_artifact_bonsai(args.artifact)
    if artifact.get("config", {}).get("architecture") != "qwen35":
        raise ValueError("benchmark requires a Qwen3.5 artifact")
    model = BonsaiQwen35ReferenceModel(artifact)
    load_s = time.monotonic() - load_start
    native_start = time.monotonic()
    uses_native = args.producer in {"native", LEGACY_NATIVE_PRODUCER}
    if uses_native and not model.enable_native():
        raise RuntimeError("native packed-Q1 kernel is required for this benchmark")
    native_enable_s = time.monotonic() - native_start
    if args.producer == LEGACY_NATIVE_PRODUCER and model._model_executor is not None:
        raise RuntimeError("legacy-native replay unexpectedly created a resident executor")
    selected_isa = None
    if uses_native:
        from trinote.infer_int.q1_native import q1_selected_isa
        selected_isa = q1_selected_isa()
    identity_start = time.monotonic()
    digest = info["digest"]
    identity_s = time.monotonic() - identity_start
    return model, gguf_reader, {
        "gguf_metadata_load_s": gguf_metadata_load_s,
        "artifact_load_s": load_s, "native_enable_s": native_enable_s,
        "producer": args.producer,
        "producer_runtime": {
            "native_packed_q1": bool(model._native),
            "resident_model_executor": model._model_executor is not None,
            "q1_fused_prepare_apply": (
                bool(model._native_runtime.fused)
                if model._native_runtime is not None else None
            ),
            "q1_lut32_mode": (
                model._native_runtime.lut32_mode
                if model._native_runtime is not None else None
            ),
            "selected_q1_isa": selected_isa,
        },
        "artifact_identity_hash_s": identity_s,
        "artifact": {"path": str(Path(args.artifact).resolve()), "sha256": digest,
                     "size_bytes": Path(args.artifact).stat().st_size},
        "gguf": {"path": str(Path(args.gguf).resolve()), "size_bytes": Path(args.gguf).stat().st_size},
        "environment": environment_identity(repo_root),
    }


def _worker(args, producer_environment: dict[str, Any]) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    result: dict[str, Any] = {
        "format": FORMAT, "worker_pid": os.getpid(), "mode": args.mode,
        "initial_affinity": (
            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
        "producer_environment_control": producer_environment,
    }
    try:
        if args.mode == "model":
            model, gguf_reader, identity = _load_model_worker(args, repo_root)
            result.update(identity)
            for _ in range(args.worker_warmups):
                _model_iteration(model, gguf_reader, args)
            result["iterations"] = [
                _model_iteration(model, gguf_reader, args) for _ in range(args.worker_repetitions)
            ]
        else:
            result["environment"] = environment_identity(repo_root)
            for _ in range(args.worker_warmups):
                _q1_iteration(args)
            result["iterations"] = [_q1_iteration(args) for _ in range(args.worker_repetitions)]
    except Exception as exc:  # worker must give the controller machine-readable failure context
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(json.dumps(result, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


def parse_perf_stat(text: str) -> dict[str, Any]:
    counters: dict[str, Any] = {}
    unavailable: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(";")
        if len(fields) < 3:
            continue
        value, _unit, event = fields[:3]
        event = event.strip()
        if not event:
            continue
        if value.strip().startswith("<"):
            unavailable[event] = value.strip()
            continue
        try:
            counters[event] = float(value.strip().replace(",", ""))
        except ValueError:
            unavailable[event] = value.strip()
    return {"counters": counters, "unavailable": unavailable}


def _perf_available(events: str) -> tuple[bool, str | None]:
    perf = shutil.which("perf")
    if not perf:
        return False, "perf executable is not installed"
    probe = subprocess.run([perf, "stat", "-e", events, "--", "true"],
                           capture_output=True, text=True, check=False)
    if probe.returncode:
        return False, (probe.stderr or probe.stdout).strip()[:1000]
    return True, None


def _worker_args(args, repetitions: int, warmups: int) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--mode", args.mode, "--worker-repetitions", str(repetitions),
        "--worker-warmups", str(warmups), "--gguf", args.gguf,
        "--producer", args.producer,
        "--artifact", args.artifact, "--bin-dir", args.bin_dir,
        "--prompt", args.prompt, "--max-new", str(args.max_new),
        "--sampler", args.sampler, "--seed", str(args.seed),
        "--threads", str(args.threads),
        "--prefill-q1-chunk", str(args.prefill_q1_chunk),
        "--q1-in-features", str(args.q1_in_features),
        "--q1-out-features", str(args.q1_out_features),
        "--q1-tokens", str(args.q1_tokens), "--q1-iterations", str(args.q1_iterations),
    ]
    if args.raw_ids:
        command += ["--raw-ids", args.raw_ids]
    if args.chat:
        command.append("--chat")
    if args.ignore_eos:
        command.append("--ignore-eos")
    if args.q1_lut32:
        command.append("--q1-lut32")
    return command


def _run_worker(command: list[str], perf_events: str | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    perf_result = None
    if perf_events:
        with tempfile.NamedTemporaryFile(prefix="bonsai35-perf-", suffix=".txt", delete=False) as tmp:
            perf_path = Path(tmp.name)
        command = [shutil.which("perf") or "perf", "stat", "-x", ";", "--no-big-num",
                   "-e", perf_events, "-o", str(perf_path), "--", *command]
    else:
        perf_path = None
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if perf_path is not None:
        try:
            perf_result = parse_perf_stat(perf_path.read_text(errors="replace"))
            perf_path.unlink(missing_ok=True)
        except OSError as exc:
            perf_result = {"error": str(exc)}
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"benchmark worker emitted no JSON (exit {proc.returncode}): {proc.stderr[-2000:]}")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"benchmark worker emitted invalid JSON: {lines[-1][-1000:]}") from exc
    if proc.returncode or payload.get("error"):
        raise RuntimeError(f"benchmark worker failed: {payload.get('error')} stderr={proc.stderr[-1000:]}")
    if proc.stderr:
        payload["worker_stderr"] = proc.stderr[-4000:]
    return payload, perf_result


def _commitment_consistency_gate(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    """Prove measured model repetitions committed the same result and length."""

    applicable = any(
        "commitments" in iteration or "output_token_count" in iteration
        for iteration in iterations
    )
    records: list[tuple[str, str, int]] = []
    for iteration in iterations:
        commitments = iteration.get("commitments")
        if not isinstance(commitments, dict):
            continue
        output_hash = commitments.get("output_ids_sha256")
        cache_hash = commitments.get("cache_state_sha256")
        output_count = iteration.get("output_token_count")
        if (
            isinstance(output_hash, str)
            and isinstance(cache_hash, str)
            and type(output_count) is int
        ):
            records.append((output_hash, cache_hash, output_count))

    eligible = applicable and len(records) == len(iterations) and bool(records)
    output_hashes = sorted({record[0] for record in records})
    cache_hashes = sorted({record[1] for record in records})
    output_counts = sorted({record[2] for record in records})
    output_hash_identical = eligible and len(output_hashes) == 1
    cache_hash_identical = eligible and len(cache_hashes) == 1
    output_count_identical = eligible and len(output_counts) == 1
    return {
        "applicable": applicable,
        "eligible": eligible,
        "iteration_count": len(iterations),
        "complete_record_count": len(records),
        "output_ids_sha256_values": output_hashes,
        "cache_state_sha256_values": cache_hashes,
        "output_token_count_values": output_counts,
        "output_ids_sha256_identical": output_hash_identical,
        "cache_state_sha256_identical": cache_hash_identical,
        "output_token_count_identical": output_count_identical,
        "passed": (
            output_hash_identical
            and cache_hash_identical
            and output_count_identical
        ) if applicable else None,
    }


def _aggregate(workers: list[dict[str, Any]]) -> dict[str, Any]:
    iterations = [iteration for worker in workers for iteration in worker.get("iterations", [])]
    metric_paths = {
        "gguf_metadata_load_s": [worker.get("gguf_metadata_load_s") for worker in workers],
        "artifact_load_s": [worker.get("artifact_load_s") for worker in workers],
        "native_enable_s": [worker.get("native_enable_s") for worker in workers],
        "artifact_identity_hash_s": [worker.get("artifact_identity_hash_s") for worker in workers],
    }
    if iterations and "tokenizer_subprocess_s" in iterations[0].get("timing", {}):
        for key in (
            "tokenizer_subprocess_s", "prompt_prefill_s", "prompt_prefill_tokens_per_s",
            "time_to_first_output_token_compute_s", "time_to_first_output_token_end_to_end_s",
            "first_output_projection_sampling_s",
            "first_decode_token_after_prefill_s", "steady_decode_token_3_32_median_s",
            "steady_decode_token_33_128_median_s", "steady_decode_token_3_32_tokens_per_s",
            "steady_decode_token_33_128_tokens_per_s", "generation_compute_s",
            "output_projection_s_total", "sampling_s_total", "final_norm_s_total", "decode_graph_s_total",
        ):
            metric_paths[key] = [iteration["timing"].get(key) for iteration in iterations]
    elif iterations and "median_s" in iterations[0].get("timing", {}):
        metric_paths["q1_call_median_s"] = [iteration["timing"].get("median_s") for iteration in iterations]
    summaries = {key: summary for key, values in metric_paths.items()
                 if (summary := _metric_summary(values)) is not None}
    variation = {
        key: value["coefficient_of_variation"] for key, value in summaries.items()
        if key in {"prompt_prefill_s", "steady_decode_token_3_32_median_s",
                   "steady_decode_token_33_128_median_s", "generation_compute_s"}
    }
    variation_sample_counts = {
        key: summaries[key]["n"] for key in variation
    }
    required_samples = 5
    variation_eligible = bool(variation) and all(
        count >= required_samples for count in variation_sample_counts.values()
    )
    return {
        "iteration_count": len(iterations), "metrics": summaries,
        "commitment_consistency_gate": _commitment_consistency_gate(iterations),
        "variation_exit_gate": {
            "threshold": 0.05,
            "minimum_samples": required_samples,
            "sample_counts": variation_sample_counts,
            "eligible": variation_eligible,
            "values": variation,
            "passed": variation_eligible and all(v < 0.05 for v in variation.values()),
        },
    }


def _write_result(payload: dict[str, Any], output: str, output_dir: str) -> Path:
    if output:
        path = Path(output).expanduser()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = Path(output_dir).expanduser() / f"bonsai35-{payload.get('mode', 'benchmark')}-{stamp}-{os.getpid()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path.resolve()


def _controller(args) -> int:
    producer_environment = configure_producer_environment(args.producer)
    thread_env = configure_thread_environment(args.threads, args.prefill_q1_chunk)
    affinity = _affinity_for_mode(args.core_mode)
    affinity_error = None
    if not args.no_pin and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, affinity)
        except OSError as exc:
            affinity_error = str(exc)
    background_cpu = sample_host_cpu(args.background_sample_seconds)
    other_engines = find_other_native_engines()
    rejection_reasons = []
    if background_cpu > args.max_background_cpu_percent:
        rejection_reasons.append(
            f"background CPU {background_cpu:.2f}% exceeds {args.max_background_cpu_percent:.2f}%"
        )
    if other_engines:
        rejection_reasons.append(f"found {len(other_engines)} other native inference process(es)")
    if affinity_error:
        rejection_reasons.append(f"could not pin benchmark affinity: {affinity_error}")
    rejected = bool(rejection_reasons) and not args.allow_busy

    payload: dict[str, Any] = {
        "format": FORMAT, "mode": args.mode, "condition": args.condition,
        "accepted": not rejected, "rejection_reasons": rejection_reasons,
        "control": {
            "repetitions": args.repetitions, "background_cpu_percent": background_cpu,
            "max_background_cpu_percent": args.max_background_cpu_percent,
            "other_native_engines": other_engines, "core_mode": args.core_mode,
            "requested_affinity": affinity, "effective_affinity": (
                sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
            ),
            "thread_environment": thread_env, "prefill_q1_chunk": args.prefill_q1_chunk,
            "producer_environment": producer_environment,
            "busy_override": args.allow_busy,
        },
        "configuration": {
            "gguf": args.gguf, "artifact": args.artifact, "prompt": args.prompt,
            "raw_ids": args.raw_ids, "chat": args.chat, "max_new": args.max_new,
            "sampler": args.sampler, "seed": args.seed, "ignore_eos": args.ignore_eos,
            "producer": args.producer,
            "q1_shape": [args.q1_tokens, args.q1_out_features, args.q1_in_features],
            "q1_iterations": args.q1_iterations, "q1_lut32": args.q1_lut32,
        },
    }
    if rejected:
        path = _write_result(payload, args.output, args.output_dir)
        payload["result_path"] = str(path)
        print(json.dumps(payload, sort_keys=True))
        return 2

    perf_events = None
    perf_status = {"requested": args.perf, "available": False, "reason": None}
    if args.perf:
        available, reason = _perf_available(args.perf_events)
        perf_status.update({"available": available, "reason": reason})
        if available:
            perf_events = args.perf_events
    workers: list[dict[str, Any]] = []
    perf_runs: list[dict[str, Any] | None] = []
    if args.condition == "cold-process":
        schedules = [(1, 0)] * args.repetitions
    elif args.condition == "warm-process":
        schedules = [(1, 1)] * args.repetitions
    else:
        schedules = [(args.repetitions, 1)]
    for repetitions, warmups in schedules:
        worker, counters = _run_worker(_worker_args(args, repetitions, warmups), perf_events)
        workers.append(worker)
        perf_runs.append(counters)
    payload.update({
        "workers": workers, "aggregate": _aggregate(workers),
        "hardware_counters": {**perf_status, "events": args.perf_events, "runs": perf_runs},
    })
    path = _write_result(payload, args.output, args.output_dir)
    payload["result_path"] = str(path)
    print(json.dumps(payload, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    gguf, artifact, bin_dir, output_dir = _default_paths()
    parser = argparse.ArgumentParser(
        description="Controlled JSON benchmark for native Bonsai-27B/Qwen3.5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--mode", choices=("model", "q1-prepare", "q1-apply"), default="model")
    parser.add_argument(
        "--producer", choices=("oracle", "native", LEGACY_NATIVE_PRODUCER), default="native",
        help=("model mode: canonical NumPy oracle, optimized resident native producer, "
              "or controlled pre-fusion Python/native-primitives baseline"),
    )
    parser.add_argument("--condition", choices=("cold-process", "warm-process", "second-turn"),
                        default="second-turn",
                        help=("process/allocation warmth only; every measured model iteration starts "
                              "from an empty semantic model cache"))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--gguf", default=gguf)
    parser.add_argument("--artifact", default=artifact)
    parser.add_argument("--bin-dir", default=bin_dir)
    parser.add_argument("--prompt", default="Hi")
    parser.add_argument("--raw-ids", default="",
                        help="comma-separated exact token IDs; bypasses tokenizer and --prompt")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--sampler", choices=("greedy", "bonsai27-rec"), default="greedy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--threads", type=int, default=0, help="0 selects physical-core count")
    parser.add_argument("--prefill-q1-chunk", type=int, default=8)
    parser.add_argument("--core-mode", choices=("physical", "smt"), default="physical")
    parser.add_argument("--no-pin", action="store_true")
    parser.add_argument("--background-sample-seconds", type=float, default=1.0)
    parser.add_argument("--max-background-cpu-percent", type=float, default=20.0)
    parser.add_argument("--allow-busy", action="store_true",
                        help="record contamination but run anyway (result remains visibly annotated)")
    parser.add_argument("--perf", action="store_true", help="collect perf stat counters when permitted")
    parser.add_argument("--perf-events", default=PERF_EVENTS)
    parser.add_argument("--output", default="")
    parser.add_argument("--output-dir", default=output_dir)
    parser.add_argument("--q1-in-features", type=int, default=5120)
    parser.add_argument("--q1-out-features", type=int, default=5120)
    parser.add_argument("--q1-tokens", type=int, default=1)
    parser.add_argument("--q1-iterations", type=int, default=5)
    parser.add_argument("--q1-lut32", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-repetitions", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--worker-warmups", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("repetitions", "max_new", "prefill_q1_chunk", "q1_in_features",
                 "q1_out_features", "q1_tokens", "q1_iterations", "worker_repetitions"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.threads < 0:
        raise SystemExit("--threads must be non-negative")
    if args.worker:
        producer_environment = configure_producer_environment(args.producer)
        configure_thread_environment(args.threads, args.prefill_q1_chunk)
        return _worker(args, producer_environment)
    return _controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
