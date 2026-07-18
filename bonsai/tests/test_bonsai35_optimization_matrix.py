from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from trinote.infer_int.artifact_io_bonsai import save_artifact_bonsai
from trinote.infer_int.reference_bonsai35 import random_bonsai35_artifact


THREADS = (1, 2, 4, 8, 12, 15)
TILES = (1, 2, 4, 8, 16, 32)
LUT_WIDTHS = ("uint64", "int32")
ISAS = ("portable", "avx2")

# Keep the prompt one row beyond both the initial 16-row KV capacity and the
# largest Q1 tile.  This makes every tile setting observable and exercises KV
# growth at 16 -> 17 and 32 -> 33 in every matrix worker.
_PROMPT = tuple((i * 17 + 3) % 127 for i in range(33))


_CHILD = r"""
import json
import os
import sys

import numpy as np

from trinote.infer_int.artifact_io_bonsai import load_artifact_bonsai
from trinote.infer_int.bonsai_runtime import generate_bonsai_tokens
from trinote.infer_int.prompt_cache_bonsai35 import (
    build_prompt_state,
    generate_from_prompt_state,
)
from trinote.infer_int.q1_native import q1_native_stats, q1_selected_isa
from trinote.infer_int.reference_bonsai import _rmsnorm, q1_rows_fp
from trinote.infer_int.reference_bonsai35 import (
    BonsaiQwen35ReferenceModel,
    _Qwen35Cache,
    _ffn,
    _full_attention,
    _project,
    _recurrent_attention,
)
from trinote.infer_int.sampler import SamplerConfig, resolve_sampler, sample_token
from trinote.infer_int.trace_bonsai35 import tensor_digest


PROMPT = json.loads(sys.argv[2])
PREFIX_LEN = 17


def cache_row(cache, layer_index, kind):
    if kind == "recurrent":
        return {
            "state": tensor_digest(cache.state[layer_index]),
            "conv": tensor_digest(cache.conv[layer_index]),
        }
    return {
        "k": tensor_digest(cache.k[layer_index]),
        "v": tensor_digest(cache.v[layer_index]),
        "length": int(cache.lengths[layer_index]),
    }


def run_traced(model, token_ids, cache):
    # The model loop with checkpoints, while retaining its validated runtime.
    # trace_bonsai35.trace_prefill intentionally accepts just a native boolean.
    # The optimization matrix must instead pass model._native_runtime so LUT/ISA
    # controls exercise the actual optimized Qwen3.5 dispatch.
    ids = np.asarray(token_ids, dtype=np.int64)
    artifact = model.artifact
    cfg = model.cfg
    frac = int(cfg["frac"])
    eps = int(cfg.get("rmsEpsilonFp2", 1))
    start = int(cache.t)
    x = q1_rows_fp(
        artifact["embed_bits"], artifact["embed_scale_fp"], ids, frac
    )
    rows = []
    for li, layer in enumerate(artifact["layers"]):
        n1 = _rmsnorm(
            x, frac, layer["n1_gain_fp"], native=model._native, eps=eps
        )
        if layer["kind"] == "recurrent":
            branch = _recurrent_attention(
                n1,
                layer,
                artifact,
                cache,
                li,
                native=model._native,
                runtime=model._native_runtime,
            )
        else:
            branch = _full_attention(
                n1,
                layer,
                artifact,
                cache,
                li,
                start,
                native=model._native,
                runtime=model._native_runtime,
            )
        residual = x + branch
        n2 = _rmsnorm(
            residual, frac, layer["n2_gain_fp"], native=model._native, eps=eps
        )
        ffn = _ffn(
            n2,
            layer,
            frac,
            native=model._native,
            runtime=model._native_runtime,
        )
        x = residual + ffn
        rows.append({
            "layer": li,
            "kind": layer["kind"],
            "n1Last": tensor_digest(n1[-1:]),
            "branchLast": tensor_digest(branch[-1:]),
            "residualLast": tensor_digest(residual[-1:]),
            "n2Last": tensor_digest(n2[-1:]),
            "ffnLast": tensor_digest(ffn[-1:]),
            "outputLast": tensor_digest(x[-1:]),
            "cache": cache_row(cache, li, layer["kind"]),
        })
    cache.t = start + int(ids.size)
    final = _rmsnorm(
        x[-1:],
        frac,
        artifact["final_norm_gain_fp"],
        native=model._native,
        eps=eps,
    )
    logits = _project(
        final,
        artifact,
        "output",
        frac,
        native=model._native,
        runtime=model._native_runtime,
    )
    return x, {
        "layers": rows,
        "finalNorm": tensor_digest(final),
        "logits": tensor_digest(logits),
        "argmax": int(logits[0].argmax()),
    }


def cold_and_reused(model):
    n_layers = len(model.artifact["layers"])
    cold_cache = _Qwen35Cache(n_layers)
    _cold_x, cold = run_traced(model, PROMPT, cold_cache)

    reused_cache = _Qwen35Cache(n_layers)
    run_traced(model, PROMPT[:PREFIX_LEN], reused_cache)
    _reused_x, reused = run_traced(model, PROMPT[PREFIX_LEN:], reused_cache)

    # A split-prefix execution only returns suffix rows, so every last-row
    # checkpoint and every final cache must nevertheless equal cold prefill.
    if reused != cold:
        raise AssertionError("cold and reused Qwen3.5 caches/traces diverged")
    return cold


def sampler_pick(cfg, frac):
    return lambda row, pos, hist: sample_token(
        row, cfg, position=pos, frac_bits=frac, history_ids=hist
    )


def generation_modes(model):
    prompt = PROMPT[:5]
    frac = int(model.cfg["frac"])
    configs = {
        "greedy": SamplerConfig(mode="greedy"),
        "sampled": resolve_sampler("bonsai27-rec", seed=0xB05A127),
    }
    result = {}
    for name, cfg in configs.items():
        cold = generate_bonsai_tokens(model, prompt, 4, sampler=cfg)
        state = build_prompt_state(model, prompt, "a5" * 32)
        reused = generate_from_prompt_state(
            model, state, 4, sampler_pick(cfg, frac), keep_reusable=True
        )
        if reused != cold:
            raise AssertionError(f"{name} cold/reused generation diverged")
        result[name] = cold
    return result


artifact, _ = load_artifact_bonsai(sys.argv[1])
oracle_model = BonsaiQwen35ReferenceModel(artifact)
oracle = {
    "trace": cold_and_reused(oracle_model),
    "generation": generation_modes(oracle_model),
}

results = []
for tile in (1, 2, 4, 8, 16, 32):
    os.environ["TRINOTE_BONSAI35_Q1_CHUNK"] = str(tile)
    for lut_width in ("uint64", "int32"):
        os.environ["TRINOTE_BONSAI35_Q1_LUT32"] = (
            "0" if lut_width == "uint64" else "1"
        )
        model = BonsaiQwen35ReferenceModel(artifact)
        if not model.enable_native():
            raise RuntimeError("native Qwen3.5 runtime is unavailable")
        selected_isa = q1_selected_isa()
        requested_isa = os.environ["TRINOTE_Q1_ISA"]
        if selected_isa != requested_isa:
            raise AssertionError(
                f"requested ISA {requested_isa!r}, selected {selected_isa!r}"
            )
        q1_native_stats(reset=True)
        actual = {
            "trace": cold_and_reused(model),
            "generation": generation_modes(model),
        }
        if actual != oracle:
            raise AssertionError(
                f"native/oracle mismatch tile={tile} lut={lut_width}"
            )
        stats = q1_native_stats()
        if lut_width == "uint64":
            if stats["u64_calls"] <= 0 or stats["lut32_hits"] != 0:
                raise AssertionError(
                    f"uint64 configuration did not use uint64 LUTs: {stats}"
                )
        elif stats["lut32_hits"] <= 0:
            raise AssertionError(
                f"int32 configuration did not use guarded int32 LUTs: {stats}"
            )
        results.append({
            "tile": tile,
            "lutWidth": lut_width,
            "selectedIsa": selected_isa,
            "stats": stats,
        })

print(json.dumps({"oracle": oracle, "results": results}, sort_keys=True))
"""


def _host_has_avx2() -> bool:
    from trinote.infer_int import q1_native

    lib = q1_native._load_lib()
    if lib is None or not hasattr(lib, "bonsai_q1_runtime_has_avx2"):
        return False
    return bool(lib.bonsai_q1_runtime_has_avx2())


def _subprocess_matrix(path: Path, *, threads: int, isa: str) -> dict:
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": str(threads),
        "OMP_DYNAMIC": "FALSE",
        "OMP_WAIT_POLICY": "PASSIVE",
        "OMP_PLACES": "threads",
        "OMP_PROC_BIND": "spread",
        "GOMP_SPINCOUNT": "0",
        "KMP_BLOCKTIME": "0",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TRINOTE_Q1_ISA": isa,
    })
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(path), json.dumps(_PROMPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_qwen35_full_native_optimization_matrix(tmp_path):
    path = tmp_path / "tiny-qwen35.safetensors"
    save_artifact_bonsai(
        random_bonsai35_artifact(seed=61, seq_len=48),
        path,
        provenance={"kind": "optimization-matrix-test"},
    )

    isas = ISAS if _host_has_avx2() else ("portable",)
    expected_oracle = None
    seen = set()
    for threads, isa in itertools.product(THREADS, isas):
        payload = _subprocess_matrix(path, threads=threads, isa=isa)
        if expected_oracle is None:
            expected_oracle = payload["oracle"]
        else:
            assert payload["oracle"] == expected_oracle
        assert len(payload["results"]) == len(TILES) * len(LUT_WIDTHS)
        seen.update(
            (threads, row["tile"], row["lutWidth"], row["selectedIsa"])
            for row in payload["results"]
        )

    assert seen == set(itertools.product(THREADS, TILES, LUT_WIDTHS, isas))


@pytest.mark.skipif(_host_has_avx2(), reason="host supports AVX2")
def test_qwen35_forced_avx2_fails_closed_when_unsupported(tmp_path):
    """The CPU dispatcher must never silently label a portable run AVX2."""
    path = tmp_path / "tiny-qwen35.safetensors"
    save_artifact_bonsai(random_bonsai35_artifact(seed=62), path)
    env = os.environ.copy()
    env["TRINOTE_Q1_ISA"] = "avx2"
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(path), json.dumps(_PROMPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode != 0
    assert "avx2" in proc.stderr.lower()
