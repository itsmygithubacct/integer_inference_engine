from pathlib import Path
import json

import numpy as np
import pytest

from trinote.config_bonsai import ATLAS_NOTARIZED_BONSAI_8B as CFG
from trinote.infer_int.artifact_io_bonsai import (
    artifact_digest_bonsai,
    load_artifact_bonsai,
    read_artifact_info_bonsai,
    save_artifact_bonsai,
)
from trinote.infer_int.import_bonsai_gguf import _GGML_Q1_0
from trinote.infer_int.import_gguf_v2 import _GGUFReader
from trinote.infer_int.bonsai_runtime import emit_and_verify_bonsai_receipt
from trinote.infer_int.reference_bonsai import BonsaiReferenceModel, random_bonsai_artifact
from trinote.infer_int.sampler import SamplerConfig
from trinote.model.rope_v2 import build_yarn_rope_tables
from trinote.receipts import LocalLedger, build_receipt, emit_receipt, keygen
from trinote.receipts.verify import verify_receipt

_ROOT = Path(__file__).resolve().parents[1]


def _small_bonsai(seed=0):
    cfg = {
        "dModel": 128, "nHeads": 4, "nHeadsKv": 2, "headDim": 32,
        "dFfn": 256, "vocab": 256, "nLayers": 2, "fpFracBits": 16,
        "ropeBase": 1_000_000, "ropeScalingType": "none",
    }
    return random_bonsai_artifact(cfg, seq_len=32, seed=seed)


def _synthetic_bonsai_provenance(source="synthetic-bonsai-q1_0"):
    return {
        "kind": "imported-weights",
        "source": source,
        "license": "test-only",
        "ggufFile": "synthetic-bonsai-q1_0.gguf",
        "ggufSha256": "12" * 32,
        "importer": "trinote.infer_int.import_bonsai_gguf",
        "quant": "GGUF Q1_0 g128",
    }


def test_bonsai_config_matches_hf_card_shape():
    assert CFG.architecture == "qwen3"
    assert CFG.d_model == 4096 and CFG.n_layers == 36
    assert CFG.n_heads == 32 and CFG.n_heads_kv == 8 and CFG.head_dim == 128
    assert CFG.d_ffn == 12_288 and CFG.context_len == 65_536
    assert CFG.vocab_size == 151_669 and CFG.quant == "q1_0-g128"
    assert 8_180_000_000 < CFG.param_count() < 8_190_000_000


def test_bonsai_forward_and_roundtrip(tmp_path):
    art = _small_bonsai(seed=1)
    ref = BonsaiReferenceModel(art)
    ids = [1, 2, 3, 4]
    logits = ref.forward(ids)
    assert logits.shape == (len(ids), art["config"]["vocab"])
    p = tmp_path / "bonsai.safetensors"
    save_artifact_bonsai(art, p, provenance={"kind": "test"})
    loaded, info = load_artifact_bonsai(p)
    assert info["format"] == "trinote-artifact-bonsai-qwen3/1"
    assert np.array_equal(ref.forward(ids), BonsaiReferenceModel(loaded).forward(ids))


def test_bonsai_artifact_metadata_binds_identity(tmp_path):
    art = _small_bonsai(seed=3)
    provenance = _synthetic_bonsai_provenance()
    p = tmp_path / "bonsai.safetensors"

    digest = save_artifact_bonsai(art, p, provenance=provenance)
    assert digest == artifact_digest_bonsai(art, provenance=provenance)

    loaded, loaded_info = load_artifact_bonsai(p)
    read_info = read_artifact_info_bonsai(p)
    assert loaded_info["digest"] == digest
    assert loaded_info["provenance"] == provenance
    assert read_info["digest"] == digest
    assert read_info["provenance"] == provenance
    assert read_info["config"]["architecture"] == "qwen3"
    assert BonsaiReferenceModel(loaded).cfg["architecture"] == "qwen3"

    changed_provenance = _synthetic_bonsai_provenance(source="different-source")
    assert artifact_digest_bonsai(art, provenance=changed_provenance) != digest
    assert artifact_digest_bonsai(art) != digest


def test_bonsai_yarn_tables_match_prismml_llamacpp_constants():
    cos, sin = build_yarn_rope_tables(
        2,
        128,
        base=1_000_000,
        freq_scale=0.25,
        n_ctx_orig=16_384,
        beta_fast=32.0,
        beta_slow=1.0,
    )

    assert cos.shape == (2, 64)
    assert sin.shape == (2, 64)
    assert (int(cos[0, 0]), int(sin[0, 0])) == (74_621, 0)
    assert (int(cos[1, 0]), int(sin[1, 0])) == (40_318, 62_792)
    assert (int(cos[1, 10]), int(sin[1, 10])) == (74_124, 8_598)
    assert (int(cos[1, 19]), int(sin[1, 19])) == (74_611, 1_235)


def test_bonsai_chat_prompt_validates_metadata():
    from trinote.cli.run_bonsai_cli import _qwen3_chat_prompt

    kv = {"tokenizer.chat_template": (
        "{%- for message in messages %}"
        "{{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}"
        "{%- endif %}"
    )}
    prompt = _qwen3_chat_prompt("Hello", kv)
    assert prompt.startswith("<|im_start|>user\nHello<|im_end|>\n")
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")

    try:
        _qwen3_chat_prompt("Hello", {"tokenizer.chat_template": "{{ add_generation_prompt }}"})
    except ValueError as exc:
        assert "unsupported tokenizer.chat_template" in str(exc)
    else:
        raise AssertionError("unsupported Bonsai chat template metadata should fail closed")


def test_bonsai_receipt_reexecutes():
    ref = BonsaiReferenceModel(_small_bonsai(seed=2))
    ids = [5, 6, 7]
    full = ref.generate_greedy(ids, 2)
    out = full[len(ids):]
    mk = keygen(label="bonsai", secret_hex="11" * 32)
    ck = keygen(label="counterparty", secret_hex="22" * 32)
    mh = "ab" * 32
    bundle = build_receipt(model_hash=mh, input_ids=ids, output_ids=out,
                           sampler={"mode": "greedy"}, model_key=mk, counterparty_key=ck,
                           artifact_digest=mh)
    v = verify_receipt(bundle, model=ref, model_digest=mh, model_key=mk, counterparty_key=ck)
    assert v["ok"] and v["reexec"]["ok"]


def test_bonsai_runtime_runner_emits_bound_receipt(tmp_path):
    art = _small_bonsai(seed=5)
    artifact_path = tmp_path / "bonsai.safetensors"
    model_hash = save_artifact_bonsai(
        art,
        artifact_path,
        provenance=_synthetic_bonsai_provenance(),
    )
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps({"modelHash": model_hash}))

    loaded, info = load_artifact_bonsai(artifact_path)
    ref = BonsaiReferenceModel(loaded)
    input_ids = [9, 10, 11]
    output_ids = ref.generate_greedy(input_ids, 1)[len(input_ids):]
    bundle, verified, emitted = emit_and_verify_bonsai_receipt(
        ref,
        input_ids=input_ids,
        output_ids=output_ids,
        model_digest=info["digest"],
        sampler=SamplerConfig(mode="greedy"),
        identity_path=identity_path,
        ledger_path=tmp_path / "bonsai.ledger.jsonl",
        broadcast_to_log=False,
        ts="2026-06-18T00:00:00+00:00",
    )

    assert verified["ok"]
    assert verified["verificationMode"] == "fast-local"
    assert verified["artifactBindingOk"] is True
    assert verified["modelHashMatch"] is True
    assert emitted["ledgerEntry"]["modelHash"] == model_hash
    assert emitted["ledgerEntry"]["receiptHash"] == bundle["receipt"]["receiptHash"]
    assert emitted["onchain"]["status"] == "disabled"

    bad_identity = tmp_path / "bad-identity.json"
    bad_identity.write_text(json.dumps({"modelHash": "cd" * 32}))
    try:
        emit_and_verify_bonsai_receipt(
            ref,
            input_ids=input_ids,
            output_ids=output_ids,
            model_digest=info["digest"],
            sampler=SamplerConfig(mode="greedy"),
            identity_path=bad_identity,
            ledger_path=tmp_path / "bad.ledger.jsonl",
            broadcast_to_log=False,
        )
    except ValueError as exc:
        assert "identity modelHash" in str(exc)
    else:
        raise AssertionError("mismatched Bonsai identity should fail closed")


def test_bonsai_runtime_can_verify_with_fresh_oracle(tmp_path):
    art = _small_bonsai(seed=6)
    artifact_path = tmp_path / "bonsai.safetensors"
    model_hash = save_artifact_bonsai(
        art,
        artifact_path,
        provenance=_synthetic_bonsai_provenance(),
    )
    loaded, info = load_artifact_bonsai(artifact_path)
    ref = BonsaiReferenceModel(loaded)
    verifier = BonsaiReferenceModel(loaded)
    input_ids = [9, 10, 11]
    output_ids = ref.generate_greedy(input_ids, 1)[len(input_ids):]
    bundle, verified, emitted = emit_and_verify_bonsai_receipt(
        ref,
        input_ids=input_ids,
        output_ids=output_ids,
        model_digest=info["digest"],
        sampler=SamplerConfig(mode="greedy"),
        verifier_model=verifier,
        verifier_mode="fresh-oracle",
        ledger_path=tmp_path / "bonsai.ledger.jsonl",
        broadcast_to_log=False,
        ts="2026-06-18T00:00:00+00:00",
    )

    assert info["digest"] == model_hash
    assert verified["ok"]
    assert verified["verificationMode"] == "fresh-oracle"
    assert emitted["ledgerEntry"]["receiptHash"] == bundle["receipt"]["receiptHash"]


def test_bonsai_receipt_runner_contract_emits_and_verifies(tmp_path):
    art = _small_bonsai(seed=4)
    artifact_path = tmp_path / "bonsai.safetensors"
    model_hash = save_artifact_bonsai(
        art,
        artifact_path,
        provenance=_synthetic_bonsai_provenance(),
    )
    loaded, info = load_artifact_bonsai(artifact_path)
    assert info["digest"] == model_hash

    ref = BonsaiReferenceModel(loaded)
    input_ids = [5, 6, 7]
    output_ids = ref.generate_greedy(input_ids, 2)[len(input_ids):]
    mk = keygen(label="bonsai", secret_hex="11" * 32)
    ck = keygen(label="counterparty", secret_hex="22" * 32)
    bundle = build_receipt(
        model_hash=model_hash,
        input_ids=input_ids,
        output_ids=output_ids,
        sampler={"mode": "greedy"},
        model_key=mk,
        counterparty_key=ck,
        model_label="ATLAS-Notarized-Bonsai-8B",
        artifact_digest=info["digest"],
    )

    verified = verify_receipt(
        bundle,
        model=ref,
        model_digest=info["digest"],
        model_key=mk,
        counterparty_key=ck,
    )
    assert verified["ok"]
    assert verified["artifactBindingOk"] is True
    assert verified["modelHashMatch"] is True
    assert verified["reexec"]["ok"] is True

    # PROBABILISTIC AUDIT tier: sample_k>0 routes verify_receipt -> verify_sampled, re-checking only k
    # positions and labelling the result greedy-sampled / sampled=True (a partial-coverage screen, NOT a
    # full verification). Wired through verify_bundle/CLI (--sample-positions).
    sampled = verify_receipt(bundle, model=ref, model_digest=info["digest"], model_key=mk,
                             counterparty_key=ck, sample_k=1, sample_seed=7)
    assert sampled["reexec"]["strategy"] == "greedy-sampled"
    assert sampled["reexec"]["checked"] == 1 and sampled["reexec"]["of"] == len(output_ids)
    assert sampled["reexec"]["sampled"] is True
    assert sampled["reexec"]["ok"] is True

    wrong_artifact = verify_receipt(
        bundle,
        model=ref,
        model_digest="cd" * 32,
        model_key=mk,
        counterparty_key=ck,
    )
    assert wrong_artifact["reexec"]["ok"] is True
    assert wrong_artifact["modelHashMatch"] is False
    assert wrong_artifact["ok"] is False

    ledger = LocalLedger(tmp_path / "bonsai.ledger.jsonl")
    emitted = emit_receipt(
        bundle["receipt"],
        ledger=ledger,
        ts="2026-06-18T00:00:00+00:00",
        broadcast_to_log=False,
    )
    assert emitted["ledgerEntry"]["modelHash"] == model_hash
    assert emitted["ledgerEntry"]["receiptHash"] == bundle["receipt"]["receiptHash"]
    assert emitted["chainArtifact"]["modelHash"] == model_hash
    assert emitted["chainArtifact"]["receiptHash"] == bundle["receipt"]["receiptHash"]
    assert emitted["chainArtifact"]["samplerMode"] == "greedy"
    assert emitted["chainArtifact"]["seed"] == 0
    assert emitted["onchain"]["status"] == "disabled"
    vc = ledger.verify_chain()
    assert vc["ok"] is True and vc["brokenAt"] is None and vc["count"] == 1 and vc["head"]


def test_min_p_and_qwen3_rec_sampler():
    """min-p is a deterministic, receipt-bound mode; qwen3-rec is the default preset. Crucially, adding min-p
    must NOT change any non-min-p committed block (existing receiptHashes stay byte-identical)."""
    import numpy as np
    from trinote.infer_int.sampler import resolve_sampler, sample_token, sampler_config_from_block
    from trinote.receipts.receipt import sampler_to_block

    # qwen3-rec preset expands to the Qwen3 vendor recommendation
    c = resolve_sampler("qwen3-rec", seed=0)
    assert c.mode == "top_p" and c.top_k == 20
    assert abs(c.temperature - 0.6) < 1e-9 and abs(c.top_p - 0.95) < 1e-9

    # min-p: deterministic draw (same logits/seed/position -> same token) + receipt block round-trips
    row = (np.arange(40, dtype=np.int64) - 20) * 1500
    mc = resolve_sampler("min_p", min_p=0.1, seed=0)
    assert sample_token(row, mc, position=4, frac_bits=16) == sample_token(row, mc, position=4, frac_bits=16)
    blk = sampler_to_block(mc, 16)
    assert blk["mode"] == "min_p" and "minPFp" in blk
    assert sampler_to_block(sampler_config_from_block(blk), 16) == blk        # round-trip stable

    # BLOCK COMPAT: a non-min-p block must NOT gain a minPFp key (so prior receiptHashes are unchanged)
    for name in ("greedy", "qwen3-rec", "top_p"):
        assert "minPFp" not in sampler_to_block(resolve_sampler(name, temperature=0.7, top_p=0.9), 16)


def test_qwen3_rec_reproducible():
    """The default sampler (qwen3-rec, seeded nucleus) is BYTE-EXACTLY reproducible at a fixed seed: an
    autoregressive multi-token draw over the same logits yields the identical token sequence on every run —
    yet the seed genuinely drives the draw (it is not accidentally argmax)."""
    import numpy as np
    from trinote.infer_int.sampler import resolve_sampler, sample_token

    # a flat distribution so the nucleus keeps many tokens and the seeded draw really varies
    flat = np.zeros(200, dtype=np.int64)

    def draw_seq(seed):
        cfg = resolve_sampler("qwen3-rec", seed=seed)
        return [sample_token(flat, cfg, position=p, frac_bits=16) for p in range(16)]

    s0a, s0b = draw_seq(0), draw_seq(0)
    assert s0a == s0b                       # reproducible at the fixed default seed 0
    assert draw_seq(0) == s0a               # and stable across a third run
    assert draw_seq(999) == draw_seq(999)   # reproducible at any other fixed seed
    assert draw_seq(999) != s0a             # the seed genuinely changes the draw (truly stochastic, not argmax)


def test_native_prefill_attention_byte_exact():
    """Native M=N causal prefill attention (deep-dive L5) is byte-identical to the NumPy causal path, across
    shapes incl. GQA / M=1 / start>0. Skips if the native kernel isn't built."""
    import numpy as np
    from trinote.infer_int.q1_native import attention_prefill_native, q1_native_available
    if not q1_native_available():
        pytest.skip("native kernel not built")
    from trinote.determinism.fixedpoint import fixed_point_matmul, fixed_point_softmax
    from trinote.infer_int.reference_bonsai import _NEG_INF_SHIFT
    frac = 16
    rng = np.random.default_rng(11)

    def ref(qh, K, V, start, inv):
        H, M, hd = qh.shape
        Hkv, L, _ = K.shape
        rep = H // Hkv
        neg = -(1 << (frac + _NEG_INF_SHIFT))
        mask = np.arange(L)[None, :] > (start + np.arange(M))[:, None]
        out = np.empty((H, M, hd), np.int64)
        for h in range(H):
            kv = h // rep
            s = fixed_point_matmul(qh[h], K[kv].T, frac)
            s = (s * inv) >> frac
            s = np.where(mask, neg, s)
            out[h] = fixed_point_matmul(fixed_point_softmax(s, frac), V[kv], frac)
        return out

    for (H, Hkv, hd, M, start) in [(4, 2, 8, 5, 3), (32, 8, 128, 12, 0), (8, 4, 16, 1, 7), (6, 3, 8, 20, 0)]:
        L = start + M
        q = rng.integers(-(1 << 18), 1 << 18, (H, M, hd), dtype=np.int64)
        K = rng.integers(-(1 << 18), 1 << 18, (Hkv, L, hd), dtype=np.int64)
        V = rng.integers(-(1 << 18), 1 << 18, (Hkv, L, hd), dtype=np.int64)
        inv = round((1.0 / np.sqrt(hd)) * (1 << frac))
        nat = attention_prefill_native(q, K, V, start, frac, inv)
        assert nat is not None and np.array_equal(ref(q, K, V, start, inv), nat), (H, Hkv, hd, M, start)


def test_generate_batched_byte_exact():
    """Request-batching (generate_batched, deep-dive L11) is byte-identical to per-sequence standalone decode
    — for greedy AND seeded min_p — across ragged-length prompts. (Throughput is a separate concern; this
    guards the correctness invariant that batching only changes scheduling, never a committed token.)"""
    import numpy as np
    from trinote.infer_int.reference_bonsai import BonsaiReferenceModel
    from trinote.infer_int.sampler import resolve_sampler, sample_token
    ref = BonsaiReferenceModel(_small_bonsai(seed=5))
    frac = int(ref.cfg["frac"])
    prompts = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11], [2, 4]]      # ragged lengths -> differing positions
    N = 6
    # greedy: generate_batched (argmax) == per-sequence generate_greedy_tokens_cached
    assert ref.generate_batched(prompts, N) == [ref.generate_greedy_tokens_cached(p, N) for p in prompts]
    # seeded min_p (per-sequence draw keyed by seed+position+history) == per-sequence generate_cached
    cfg = resolve_sampler("min_p", seed=0)
    pk = lambda row, pos, hist: sample_token(row, cfg, position=pos, frac_bits=frac, history_ids=hist)
    assert ref.generate_batched(prompts, N, [pk] * 4) == [ref.generate_cached(p, N, pk) for p in prompts]


def test_batched_attention_kernel_byte_exact():
    """The native batched decode-attention kernel == B separate M=1 native calls, across GQA / B / ragged
    cache lengths. Skips if the native kernel isn't built."""
    import numpy as np
    from trinote.infer_int.q1_native import (
        attention_decode_native, attention_decode_batched_native, q1_native_available)
    if not q1_native_available():
        pytest.skip("native kernel not built")
    frac = 16
    rng = np.random.default_rng(17)
    for (B, H, Hkv, hd) in [(4, 8, 2, 16), (8, 32, 8, 128), (1, 4, 1, 8), (6, 6, 3, 16)]:
        Ls = [int(rng.integers(1, 25)) for _ in range(B)]
        inv = round((1.0 / np.sqrt(hd)) * (1 << frac))
        q = rng.integers(-(1 << 18), 1 << 18, (B, H, hd), dtype=np.int64)
        ks = [rng.integers(-(1 << 18), 1 << 18, (Hkv, L, hd), dtype=np.int64) for L in Ls]
        vs = [rng.integers(-(1 << 18), 1 << 18, (Hkv, L, hd), dtype=np.int64) for L in Ls]
        nb = attention_decode_batched_native(q, ks, vs, Ls, frac, inv)
        ref = np.stack([attention_decode_native(q[b], ks[b], vs[b], frac, inv) for b in range(B)])
        assert nb is not None and np.array_equal(nb, ref), (B, H, Hkv, hd, Ls)


def test_bonsai_gguf_header_if_present():
    gguf = _ROOT / "models" / "Bonsai-8B-Q1_0.gguf"
    if not gguf.exists():
        pytest.skip("models/Bonsai-8B-Q1_0.gguf not present (gitignored/large)")
    from trinote.cli.run_bonsai_cli import _qwen3_chat_prompt
    r = _GGUFReader(gguf)
    assert r.kv["general.architecture"] == "qwen3"
    assert int(r.kv["qwen3.embedding_length"]) == 4096
    assert int(r.kv["qwen3.block_count"]) == 36
    assert int(r.kv["qwen3.attention.head_count"]) == 32
    assert int(r.kv["qwen3.attention.head_count_kv"]) == 8
    assert int(r.kv["qwen3.context_length"]) == 65_536
    assert r.tensors["token_embd.weight"].ggml_type == _GGML_Q1_0
    assert r.tensors["output.weight"].ggml_type == _GGML_Q1_0
    assert _qwen3_chat_prompt("Hello", r.kv).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_bonsai_token_bytes_support_incremental_utf8_streaming():
    import codecs
    from trinote.infer_int.gguf_tokenizer_v2 import decode, token_bytes

    tokens = ["Ã", "©", "A", "<|im_start|>"]
    assert decode([0, 1, 2, 3], tokens, skip_special_from=3) == "éA"
    assert token_bytes(3, tokens, skip_special_from=3) == b""

    inc = codecs.getincrementaldecoder("utf-8")(errors="replace")
    assert inc.decode(token_bytes(0, tokens), final=False) == ""
    assert inc.decode(token_bytes(1, tokens), final=False) == "é"
    assert inc.decode(token_bytes(2, tokens), final=False) == "A"


# BONSAI-SPEED-IMPLEMENTATION.md: native speed APIs must stay byte-identical to the oracle.

def test_bonsai_fixed_point_silu_matches_two_logit_softmax_oracle():
    from trinote.determinism.fixedpoint import fixed_point_softmax
    from trinote.infer_int.reference_bonsai import fixed_point_silu

    frac = 16
    rng = np.random.default_rng(42)
    x = np.concatenate([
        rng.integers(-(1 << 20), 1 << 20, size=4096, dtype=np.int64),
        np.array([
            -(1 << 40), -(1 << 30), -(1 << 20), -1, 0, 1,
            1 << 20, 1 << 30, 1 << 40,
        ], dtype=np.int64),
    ]).reshape(1, -1)
    pairs = np.stack([np.zeros_like(x), x], axis=-1).reshape(-1, 2)
    sig = fixed_point_softmax(pairs, frac)[:, 1].reshape(x.shape)
    oracle = (x * sig) >> frac
    assert np.array_equal(fixed_point_silu(x, frac), oracle)


def test_bonsai_prefill_logits_match_forward_last_only():
    ref = BonsaiReferenceModel(_small_bonsai(seed=7))
    for ids in ([5], [5, 12, 7, 1], [5, 12, 7, 1, 99, 42, 8, 3]):
        cached = ref.prefill_logits(ids)
        oracle = ref.forward(ids, last_only=True)
        assert cached.shape == (1, ref.cfg["vocab"])
        assert np.array_equal(cached, oracle), f"Bonsai cached prefill diverged at len {len(ids)}"
        assert np.array_equal(oracle, ref.forward(ids)[-1:])


def test_bonsai_kv_cache_is_bit_identical():
    ref = BonsaiReferenceModel(_small_bonsai(seed=8))
    ids = [5, 12, 7, 1, 99]
    assert ref.generate_greedy_cached(ids, 8) == ref.generate_greedy(ids, 8)


def test_bonsai_kv_cache_streaming_callback_matches():
    ref = BonsaiReferenceModel(_small_bonsai(seed=9))
    ids = [1, 2, 3, 4]
    streamed = []
    out = ref.generate_cached(
        ids,
        6,
        pick=lambda row, pos, hist: int(np.asarray(row).argmax()),
        on_token=streamed.append,
    )
    assert out == streamed
    assert (list(ids) + out) == ref.generate_greedy(ids, 6)


def test_bonsai_eos_is_committed_but_not_streamed():
    ref = BonsaiReferenceModel(_small_bonsai(seed=19))
    streamed = []
    out = ref.generate_cached(
        [1, 2, 3],
        4,
        pick=lambda _row, _pos, _hist: 9,
        eos=9,
        on_token=streamed.append,
    )
    assert out == [9]
    assert streamed == []


def test_bonsai_kv_cache_context_overflow_falls_back_exactly():
    art = _small_bonsai(seed=10)
    art["config"]["context_len"] = 8
    ref = BonsaiReferenceModel(art)
    ids = [0, 1, 2, 3, 4]
    n_new = 4
    assert len(ids) + n_new > ref.cfg["context_len"]
    assert ref.generate_greedy_cached(ids, n_new) == ref.generate_greedy(ids, n_new)


def test_bonsai_q1_sign_cache_matches_oracle():
    from trinote.infer_int.reference_bonsai import (
        _unpack_q1_signs,
        q1_linear_ref,
        q1_linear_signs_ref,
    )

    rng = np.random.default_rng(0)
    frac = 16
    out_f, in_f = 9, 256
    n_blocks = in_f // 128
    scale_fp = rng.integers(1 << 8, 1 << 12, size=(out_f, n_blocks), dtype=np.int64)
    cases = [
        rng.integers(0, 256, size=(out_f, n_blocks, 16), dtype=np.uint8),
        np.zeros((out_f, n_blocks, 16), dtype=np.uint8),
        np.full((out_f, n_blocks, 16), 255, dtype=np.uint8),
        np.tile(np.array([0xAA, 0x55], dtype=np.uint8), (out_f, n_blocks, 8)),
    ]
    activations = [
        rng.integers(-(1 << 14), 1 << 14, size=(3, in_f), dtype=np.int64),
        np.full((2, in_f), 1 << 20, dtype=np.int64),
    ]
    for bits in cases:
        signs = _unpack_q1_signs(bits)
        assert signs.dtype == np.int8
        for x_fp in activations:
            oracle = q1_linear_ref(x_fp, bits, scale_fp, frac, out_chunk=5)
            fast = q1_linear_signs_ref(x_fp, signs, scale_fp, frac, out_chunk=5)
            assert np.array_equal(fast, oracle)


def test_bonsai_q1_block_fast_path_matches_oracle():
    from trinote.infer_int.reference_bonsai import _q1_bl_fast, _q1_bl_ref, _unpack_q1_signs

    ref = BonsaiReferenceModel(_small_bonsai(seed=11))
    layer = ref.artifact["layers"][0]
    frac = int(ref.cfg["frac"])
    rng = np.random.default_rng(1)
    x_fp = rng.integers(-(1 << 14), 1 << 14, size=(3, ref.cfg["dModel"]), dtype=np.int64)

    for name in ("wq", "wk", "wv", "wo", "w1", "wu"):
        layer[f"{name}_signs_i8"] = _unpack_q1_signs(layer[f"{name}_bits"])
        assert np.array_equal(_q1_bl_fast(x_fp, layer, name, frac), _q1_bl_ref(x_fp, layer, name, frac))

    down_in = rng.integers(-(1 << 14), 1 << 14, size=(3, ref.cfg["dFfn"]), dtype=np.int64)
    layer["w2_signs_i8"] = _unpack_q1_signs(layer["w2_bits"])
    assert np.array_equal(_q1_bl_fast(down_in, layer, "w2", frac), _q1_bl_ref(down_in, layer, "w2", frac))


def test_bonsai_fast_forward_matches_oracle_logits():
    ref = BonsaiReferenceModel(_small_bonsai(seed=12))
    ids = [5, 12, 7, 1, 99, 42]
    oracle = ref.forward(ids)
    assert ref.enable_fast(check_ram=False) is True
    assert np.array_equal(ref.forward_fast(ids), oracle)
    assert np.array_equal(ref.teacher_forced_logits(ids), oracle)

    layer = ref.artifact["layers"][0]
    for name in ("wq", "wk", "wv", "wo", "w1", "wu", "w2"):
        signs = layer[f"{name}_signs_i8"]
        assert signs.dtype == np.int8
        assert signs.shape == layer[f"{name}_bits"].shape[:2] + (128,)
    assert ref.artifact["output_signs_i8"].dtype == np.int8


def test_bonsai_fast_kv_cache_is_bit_identical():
    ref = BonsaiReferenceModel(_small_bonsai(seed=13))
    ids = [5, 12, 7, 1, 99]
    oracle = ref.generate_greedy(ids, 8)
    cached_slow = ref.generate_greedy_cached(ids, 8)
    assert ref.enable_fast(check_ram=False) is True
    cached_fast = ref.generate_greedy_cached(ids, 8)
    assert oracle == cached_slow == cached_fast


def test_bonsai_greedy_runtime_uses_argmax_fast_path(monkeypatch):
    import trinote.infer_int.bonsai_runtime as br

    ref = BonsaiReferenceModel(_small_bonsai(seed=22))
    ids = [5, 12, 7, 1]
    calls = {"argmax": 0, "generic": 0}
    expected = ref.generate_greedy_cached(ids, 3)[len(ids):]

    def fake_greedy(input_ids, n_new, *, eos=None, on_token=None):
        calls["argmax"] += 1
        return list(expected)

    def fake_generic(*_args, **_kwargs):
        calls["generic"] += 1
        return []

    monkeypatch.setattr(ref, "generate_greedy_tokens_cached", fake_greedy)
    monkeypatch.setattr(ref, "generate_cached", fake_generic)
    out = br.generate_bonsai_tokens(ref, ids, 3, sampler=SamplerConfig(mode="greedy"))
    assert out == expected
    assert calls == {"argmax": 1, "generic": 0}

    br.generate_bonsai_tokens(ref, ids, 3, sampler=SamplerConfig(mode="temp", seed=1))
    assert calls == {"argmax": 1, "generic": 1}


def test_bonsai_cached_decode_skips_unused_final_advance(monkeypatch):
    ref = BonsaiReferenceModel(_small_bonsai(seed=29))
    ids = [5, 12, 7, 1]
    calls = {"run_layers": 0}
    original = ref._run_layers

    def counted(*args, **kwargs):
        calls["run_layers"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ref, "_run_layers", counted)
    out = ref.generate_greedy_tokens_cached(ids, 4)
    assert out == ref.generate_greedy(ids, 4)[len(ids):]
    assert calls["run_layers"] == 4  # one prompt prefill plus one advance for each non-final token


def test_bonsai_fast_receipt_reexecutes():
    ref = BonsaiReferenceModel(_small_bonsai(seed=14))
    assert ref.enable_fast(check_ram=False) is True
    ids = [5, 6, 7]
    full = ref.generate_greedy_cached(ids, 2)
    out = full[len(ids):]
    mk = keygen(label="bonsai", secret_hex="11" * 32)
    ck = keygen(label="counterparty", secret_hex="22" * 32)
    mh = "ab" * 32
    bundle = build_receipt(
        model_hash=mh,
        input_ids=ids,
        output_ids=out,
        sampler={"mode": "greedy"},
        model_key=mk,
        counterparty_key=ck,
        artifact_digest=mh,
    )
    v = verify_receipt(bundle, model=ref, model_digest=mh, model_key=mk, counterparty_key=ck)
    assert v["ok"] and v["reexec"]["ok"]


def test_bonsai_long_receipt_uses_cached_replay_verifier():
    ref = BonsaiReferenceModel(_small_bonsai(seed=20))
    ref.receipt_verify_cached_threshold = 3
    ids = [5, 6, 7]
    out = ref.generate_greedy_cached(ids, 5)[len(ids):]
    mk = keygen(label="bonsai", secret_hex="11" * 32)
    ck = keygen(label="counterparty", secret_hex="22" * 32)
    mh = "ab" * 32
    bundle = build_receipt(
        model_hash=mh,
        input_ids=ids,
        output_ids=out,
        sampler={"mode": "greedy"},
        model_key=mk,
        counterparty_key=ck,
        artifact_digest=mh,
    )
    v = verify_receipt(bundle, model=ref, model_digest=mh, model_key=mk, counterparty_key=ck)
    assert v["ok"] and v["reexec"]["ok"]
    assert v["reexec"]["strategy"] == "greedy-cached-replay"

    bad = list(out)
    bad[0] = (bad[0] + 1) % ref.cfg["vocab"]
    bad_bundle = build_receipt(
        model_hash=mh,
        input_ids=ids,
        output_ids=bad,
        sampler={"mode": "greedy"},
        model_key=mk,
        counterparty_key=ck,
        artifact_digest=mh,
    )
    bad_v = verify_receipt(bad_bundle, model=ref, model_digest=mh, model_key=mk, counterparty_key=ck)
    assert bad_v["reexec"]["strategy"] == "greedy-cached-replay"
    assert bad_v["reexec"]["ok"] is False
    assert bad_v["ok"] is False


def test_bonsai_native_receipt_reexecutes_with_fresh_slow_model_if_present():
    from trinote.infer_int.q1_native import q1_native_available

    if not q1_native_available():
        pytest.skip("native Q1 kernel (libbonsai_q1_kernel.so) not available")
    art = _small_bonsai(seed=17)
    producer = BonsaiReferenceModel(art)
    assert producer.enable_native() is True
    ids = [5, 6, 7]
    full = producer.generate_greedy_cached(ids, 2)
    out = full[len(ids):]
    mk = keygen(label="bonsai", secret_hex="11" * 32)
    ck = keygen(label="counterparty", secret_hex="22" * 32)
    mh = "ab" * 32
    bundle = build_receipt(
        model_hash=mh,
        input_ids=ids,
        output_ids=out,
        sampler={"mode": "greedy"},
        model_key=mk,
        counterparty_key=ck,
        artifact_digest=mh,
    )
    verifier = BonsaiReferenceModel(art)
    v = verify_receipt(bundle, model=verifier, model_digest=mh, model_key=mk, counterparty_key=ck)
    assert v["ok"] and v["reexec"]["ok"]


def test_bonsai_native_q1_kernel_matches_oracle_if_present():
    from trinote.infer_int.q1_native import q1_linear_native, q1_native_available
    from trinote.infer_int.reference_bonsai import q1_linear_ref

    if not q1_native_available():
        pytest.skip("native Q1 kernel (libbonsai_q1_kernel.so) not available")
    ref = BonsaiReferenceModel(_small_bonsai(seed=15))
    layer = ref.artifact["layers"][0]
    frac = int(ref.cfg["frac"])
    rng = np.random.default_rng(15)
    for name, width in (("wq", ref.cfg["dModel"]), ("wk", ref.cfg["dModel"]),
                        ("wv", ref.cfg["dModel"]), ("wo", ref.cfg["dModel"]),
                        ("w1", ref.cfg["dModel"]), ("wu", ref.cfg["dModel"]),
                        ("w2", ref.cfg["dFfn"])):
        x_fp = rng.integers(-(1 << 14), 1 << 14, size=(2, width), dtype=np.int64)
        native = q1_linear_native(x_fp, layer[f"{name}_bits"], layer[f"{name}_scale_fp"], frac)
        oracle = q1_linear_ref(x_fp, layer[f"{name}_bits"], layer[f"{name}_scale_fp"], frac)
        assert native is not None
        assert np.array_equal(native, oracle)


def test_bonsai_native_q1_kernel_reuses_workspace_if_present():
    import trinote.infer_int.q1_native as qn
    from trinote.infer_int.reference_bonsai import q1_linear_ref

    if not qn.q1_native_available() or not hasattr(qn._load_lib(), "bonsai_q1_linear_i64_workspace"):
        pytest.skip("native Q1 workspace kernel (bonsai_q1_linear_i64_workspace) not available")
    if hasattr(qn._TLS, "q1_workspace"):
        delattr(qn._TLS, "q1_workspace")
    ref = BonsaiReferenceModel(_small_bonsai(seed=21))
    layer = ref.artifact["layers"][0]
    frac = int(ref.cfg["frac"])
    rng = np.random.default_rng(21)
    x_fp = rng.integers(-(1 << 14), 1 << 14, size=(1, ref.cfg["dModel"]), dtype=np.int64)

    native = qn.q1_linear_native(x_fp, layer["wq_bits"], layer["wq_scale_fp"], frac)
    oracle = q1_linear_ref(x_fp, layer["wq_bits"], layer["wq_scale_fp"], frac)
    totals, lut = qn._TLS.q1_workspace
    first_ids = (id(totals), id(lut))
    native2 = qn.q1_linear_native(x_fp, layer["wq_bits"], layer["wq_scale_fp"], frac)
    totals2, lut2 = qn._TLS.q1_workspace

    assert native is not None and native2 is not None
    assert np.array_equal(native, oracle)
    assert np.array_equal(native2, oracle)
    assert (id(totals2), id(lut2)) == first_ids
    assert totals.size >= layer["wq_scale_fp"].shape[1]
    assert lut.size >= layer["wq_scale_fp"].shape[1] * 16 * 256


def test_bonsai_native_q1_prepared_kernel_matches_oracle_if_present():
    import trinote.infer_int.q1_native as qn
    from trinote.infer_int.reference_bonsai import q1_linear_ref

    if not qn.q1_native_available() or not hasattr(qn._load_lib(), "bonsai_q1_prepare_i64"):
        pytest.skip("native Q1 prepared kernel (bonsai_q1_prepare_i64) not available")
    ref = BonsaiReferenceModel(_small_bonsai(seed=25))
    layer = ref.artifact["layers"][0]
    frac = int(ref.cfg["frac"])
    rng = np.random.default_rng(25)
    x_fp = rng.integers(-(1 << 14), 1 << 14, size=(2, ref.cfg["dModel"]), dtype=np.int64)
    prep = qn.q1_prepare_native(x_fp, int(layer["wq_scale_fp"].shape[1]))
    assert prep is not None

    names = ("wq", "wk", "wv", "w1", "wu")
    grouped = None
    if hasattr(qn._load_lib(), "bonsai_q1_linear_i64_prepared_multi"):
        weights = tuple((layer[f"{name}_bits"], layer[f"{name}_scale_fp"]) for name in names)
        grouped = qn.q1_linear_prepared_many_native(prep, weights, frac)
        assert grouped is not None

    for i, name in enumerate(names):
        native = qn.q1_linear_prepared_native(prep, layer[f"{name}_bits"], layer[f"{name}_scale_fp"], frac)
        oracle = q1_linear_ref(x_fp, layer[f"{name}_bits"], layer[f"{name}_scale_fp"], frac)
        assert native is not None
        assert np.array_equal(native, oracle)
        if grouped is not None:
            assert np.array_equal(grouped[i], oracle)


def test_bonsai_prepared_multi_env_default(monkeypatch):
    import trinote.infer_int.reference_bonsai as rb

    monkeypatch.delenv("TRINOTE_Q1_PREPARED_MULTI", raising=False)
    assert rb._prepared_multi_enabled() is True

    for value in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("TRINOTE_Q1_PREPARED_MULTI", value)
        assert rb._prepared_multi_enabled() is False

    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv("TRINOTE_Q1_PREPARED_MULTI", value)
        assert rb._prepared_multi_enabled() is True

    monkeypatch.delenv("TRINOTE_NATIVE_RMSNORM", raising=False)
    assert rb._native_rmsnorm_enabled() is True

    for value in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("TRINOTE_NATIVE_RMSNORM", value)
        assert rb._native_rmsnorm_enabled() is False

    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv("TRINOTE_NATIVE_RMSNORM", value)
        assert rb._native_rmsnorm_enabled() is True


def test_bonsai_native_rmsnorm_matches_oracle_if_present():
    import trinote.infer_int.q1_native as qn
    from trinote.determinism.fixedpoint import fixed_point_rmsnorm

    if not qn.q1_native_available() or not hasattr(qn._load_lib(), "bonsai_rmsnorm_i64"):
        pytest.skip("native RMSNorm kernel (bonsai_rmsnorm_i64) not available")
    frac = 16
    rng = np.random.default_rng(28)
    x = rng.integers(-(1 << 20), 1 << 20, size=(5, 17), dtype=np.int64)
    gain = rng.integers(1 << 14, 1 << 17, size=17, dtype=np.int64)

    native = qn.rmsnorm_native(x, frac, gain_q=gain)
    oracle = fixed_point_rmsnorm(x, frac, gain_q=gain)
    assert native is not None
    assert np.array_equal(native, oracle)

    too_wide = np.full((1, 5), np.iinfo(np.int64).max, dtype=np.int64)
    assert qn.rmsnorm_native(too_wide, frac) is None


def test_bonsai_native_q1_argmax_matches_oracle_if_present():
    import trinote.infer_int.q1_native as qn

    if not qn.q1_native_available() or not hasattr(qn._load_lib(), "bonsai_q1_argmax_i64_workspace"):
        pytest.skip("native Q1 argmax kernel (bonsai_q1_argmax_i64_workspace) not available")
    ref = BonsaiReferenceModel(_small_bonsai(seed=23))
    frac = int(ref.cfg["frac"])
    rng = np.random.default_rng(23)
    for x_fp in (
        rng.integers(-(1 << 14), 1 << 14, size=(1, ref.cfg["dModel"]), dtype=np.int64),
        rng.integers(-(1 << 14), 1 << 14, size=(3, ref.cfg["dModel"]), dtype=np.int64),
    ):
        ids = qn.q1_argmax_native(x_fp, ref.artifact["output_bits"], ref.artifact["output_scale_fp"], frac)
        logits = qn.q1_linear_native(x_fp, ref.artifact["output_bits"], ref.artifact["output_scale_fp"], frac)
        assert ids is not None and logits is not None
        assert np.array_equal(ids, logits.argmax(axis=1))


def test_bonsai_native_q1_kernel_matches_oracle_at_overflow_boundary_if_present():
    from trinote.infer_int.q1_native import q1_linear_native, q1_native_available
    from trinote.infer_int.reference_bonsai import q1_linear_ref

    if not q1_native_available():
        pytest.skip("native Q1 kernel (libbonsai_q1_kernel.so) not available")
    frac = 1
    x_fp = np.zeros((1, 128), dtype=np.int64)
    x_fp[0, 0] = np.int64(1 << 62)
    bits = np.zeros((3, 1, 16), dtype=np.uint8)
    bits[:, 0, 0] = 0x01
    scale_fp = np.array([[3], [-3], [4]], dtype=np.int64)

    native = q1_linear_native(x_fp, bits, scale_fp, frac)
    oracle = q1_linear_ref(x_fp, bits, scale_fp, frac)
    assert native is not None
    assert np.array_equal(native, oracle)


def test_bonsai_native_q1_kernel_rejects_overflowed_dimensions_if_present():
    import ctypes
    import trinote.infer_int.q1_native as qn

    if not qn.q1_native_available():
        pytest.skip("native Q1 kernel (libbonsai_q1_kernel.so) not available")
    lib = qn._load_lib()
    one_i64 = np.zeros(1, dtype=np.int64)
    one_u8 = np.zeros(1, dtype=np.uint8)
    rc = lib.bonsai_q1_linear_i64(
        one_i64.ctypes.data,
        one_u8.ctypes.data,
        one_i64.ctypes.data,
        ctypes.c_int64(1 << 62),
        ctypes.c_int64(1),
        ctypes.c_int64(1 << 62),
        ctypes.c_int64(16),
        one_i64.ctypes.data,
    )
    assert rc == 1


def test_bonsai_native_forward_matches_oracle_if_present():
    from trinote.infer_int.q1_native import q1_native_available

    if not q1_native_available():
        pytest.skip("native Q1 kernel (libbonsai_q1_kernel.so) not available")
    ref = BonsaiReferenceModel(_small_bonsai(seed=16))
    ids = [5, 12, 7, 1, 99]
    oracle = ref.forward(ids)
    assert ref.enable_native() is True
    assert np.array_equal(ref.forward_fast(ids), oracle)
    assert ref.generate_greedy_cached(ids, 5) == ref.generate_greedy(ids, 5)


def test_bonsai_native_argmax_greedy_decode_matches_full_logits_if_present():
    from trinote.infer_int.q1_native import q1_native_available

    if not q1_native_available():
        pytest.skip("native Q1 kernel (libbonsai_q1_kernel.so) not available")
    ref = BonsaiReferenceModel(_small_bonsai(seed=24))
    ids = [5, 12, 7, 1, 99]
    oracle = ref.generate_greedy(ids, 6)
    assert ref.enable_native() is True
    fast = list(ids) + ref.generate_greedy_tokens_cached(ids, 6)
    assert fast == oracle


def test_bonsai_native_kernel_failure_falls_back_to_oracle(monkeypatch):
    import trinote.infer_int.reference_bonsai as rb

    ref = BonsaiReferenceModel(_small_bonsai(seed=18))
    ids = [5, 12, 7, 1]
    oracle = ref.forward(ids)

    def fail_native(*_args, **_kwargs):
        raise RuntimeError("forced native failure")

    monkeypatch.setattr(rb, "q1_linear_native", fail_native)
    ref._native = True
    ref._fast = True
    assert np.array_equal(ref.forward_fast(ids), oracle)


def test_bonsai_identity_binds_real_model_if_present():
    """The shipped identity modelHash must equal the real safetensors artifact's digest."""
    model = _ROOT / "artifacts" / "model" / "atlas-notarized-bonsai-8b.safetensors"
    identity = _ROOT / "artifacts" / "atlas-notarized-bonsai-8b.identity.json"
    if not model.exists():
        pytest.skip("artifacts/model/atlas-notarized-bonsai-8b.safetensors not present (gitignored/large)")
    if not identity.exists():
        pytest.skip("artifacts/atlas-notarized-bonsai-8b.identity.json not present")
    info = read_artifact_info_bonsai(model)
    minted = json.loads(identity.read_text(encoding="utf-8"))
    assert info["digest"] == minted["modelHash"]


def test_bonsai_seeded_receipt_reexecutes():
    """A SEEDED (non-greedy) receipt is receipt-bound and re-executes/verifies OK."""
    from trinote.infer_int.bonsai_runtime import generate_bonsai_tokens

    ref = BonsaiReferenceModel(_small_bonsai(seed=26))
    ids = [5, 6, 7]
    sampler = SamplerConfig(mode="temp", temperature=0.8, seed=1234)
    out = generate_bonsai_tokens(ref, ids, 4, sampler=sampler)
    mk = keygen(label="bonsai", secret_hex="11" * 32)
    ck = keygen(label="counterparty", secret_hex="22" * 32)
    mh = "ab" * 32
    bundle = build_receipt(model_hash=mh, input_ids=ids, output_ids=out,
                           sampler=sampler, model_key=mk, counterparty_key=ck,
                           artifact_digest=mh)
    assert bundle["receipt"]["receiptBound"] is True
    v = verify_receipt(bundle, model=ref, model_digest=mh, model_key=mk, counterparty_key=ck)
    assert v["ok"] and v["reexec"]["ok"]
    assert v["reexec"]["checked"] == len(out)


def test_bonsai_tampered_output_fails_verification():
    """Flipping a single committed output token must fail re-execution verification."""
    ref = BonsaiReferenceModel(_small_bonsai(seed=27))
    ids = [5, 6, 7]
    out = ref.generate_greedy(ids, 4)[len(ids):]
    mk = keygen(label="bonsai", secret_hex="11" * 32)
    ck = keygen(label="counterparty", secret_hex="22" * 32)
    mh = "ab" * 32
    tampered = list(out)
    tampered[0] = (tampered[0] + 1) % ref.cfg["vocab"]
    assert tampered != out
    bundle = build_receipt(model_hash=mh, input_ids=ids, output_ids=tampered,
                           sampler={"mode": "greedy"}, model_key=mk, counterparty_key=ck,
                           artifact_digest=mh)
    v = verify_receipt(bundle, model=ref, model_digest=mh, model_key=mk, counterparty_key=ck)
    assert v["reexec"]["ok"] is False
    assert v["ok"] is False


def test_bonsai_log_broadcast_backend_round_trip(tmp_path):
    """LogBroadcastBackend appends the chain artifact to JSONL; the txid recomputes from it."""
    from trinote.receipts import LogBroadcastBackend, chain_artifact, commit

    receipt = {
        "modelHash": "ab" * 32,
        "receiptHash": "cd" * 32,
        "trace": {"sampler": {"mode": "greedy", "seed": 0}},
    }
    artifact = chain_artifact(receipt)
    log_path = tmp_path / "broadcast.log.jsonl"
    backend = LogBroadcastBackend(log_path)
    result = backend.broadcast(artifact, ts="2026-06-18T00:00:00+00:00")

    lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["artifact"] == artifact
    assert record["network"] == "log" and record["broadcast"] is False
    expected_txid = "log:" + commit(artifact)[:32]
    assert record["txid"] == expected_txid == result["txid"]


def test_bonsai_receipt_v2_is_float_free_and_scale_bound():
    """receipt/v2 commits a fully-integer sampler block (no IEEE floats in canonical bytes), the committed
    fpFracBits is load-bearing at verify (a mismatched scale fails closed), and v1 stays reproducible."""
    import re
    from trinote.receipts.canonical import canonical_bytes

    mk = keygen(secret_hex="11" * 32)
    ck = keygen(secret_hex="22" * 32)
    cfg = SamplerConfig(mode="temp", temperature=0.8, top_p=0.95, top_k=40, seed=7)

    # v2 (default): integer sampler block, no float tokens anywhere in the committed receipt bytes.
    r = build_receipt(model_hash="ab" * 32, input_ids=[1, 2, 3], output_ids=[4, 5],
                      sampler=cfg, model_key=mk, counterparty_key=ck, fp_frac_bits=16)["receipt"]
    assert r["schema"] == "trinote.receipt/v2"
    blk = r["trace"]["sampler"]
    assert {"invTempFp", "topPFp", "fpFracBits"} <= blk.keys()
    assert "temperature" not in blk and "topP" not in blk
    assert blk["fpFracBits"] == 16
    assert not re.findall(rb"-?\d+\.\d+", canonical_bytes(r)), "v2 committed bytes must contain no floats"

    # fpFracBits is binding: a self-consistent v2 receipt at frac=12 must FAIL against a frac=16 engine.
    class _FakeModel:
        cfg = {"frac": 16, "vocab": 0}
    mismatched = build_receipt(model_hash="ab" * 32, input_ids=[1, 2], output_ids=[3],
                               sampler=SamplerConfig(mode="greedy"), model_key=mk, counterparty_key=ck,
                               fp_frac_bits=12)
    vr = verify_receipt(mismatched, model=_FakeModel())
    assert vr["fpFracBitsMatch"] is False
    assert vr["reexec"]["strategy"] == "fpfracbits-mismatch"
    assert vr["ok"] is False

    # Hardening guards fail closed.
    with pytest.raises(ValueError):
        build_receipt(model_hash="ab" * 32, input_ids=[1], output_ids=[2],
                      sampler=cfg, model_key=mk, counterparty_key=ck, schema_version="nope")
    with pytest.raises(ValueError):           # no floats permitted in a committed v2 trace
        build_receipt(model_hash="ab" * 32, input_ids=[1], output_ids=[2],
                      sampler=cfg, model_key=mk, counterparty_key=ck, trace={"attributableShards": [0.5]})

    # v1 stays reproducible (legacy float block) so historical receipts re-derive byte-for-byte.
    r1 = build_receipt(model_hash="ab" * 32, input_ids=[1, 2, 3], output_ids=[4, 5],
                       sampler=cfg, model_key=mk, counterparty_key=ck, schema_version="v1")["receipt"]
    assert r1["schema"] == "trinote.receipt/v1"
    assert r1["trace"]["sampler"]["temperature"] == 0.8 and r1["trace"]["sampler"]["topP"] == 0.95


def test_bonsai_receipt_asymmetric_signature_third_party_verifiable():
    """An EC (secp256k1) receipt is verifiable by a third party holding ONLY the public key — no secret —
    which the symmetric HMAC vouch cannot offer. Signatures are deterministic; tampering is caught; and a
    wrong expected pubkey (identity mismatch) fails."""
    from trinote.receipts import ECKey, ec_keygen, verify_ec
    from trinote.receipts.canonical import canonical_bytes
    from trinote.receipts.verify import verify_receipt

    mk = ec_keygen(label="model", secret_hex="aa" * 32)
    ck = ec_keygen(label="counterparty", secret_hex="bb" * 32)
    cfg = SamplerConfig(mode="greedy")
    b = build_receipt(model_hash="cd" * 32, input_ids=[1, 2, 3], output_ids=[4, 5],
                      sampler=cfg, model_key=mk, counterparty_key=ck, model_label="ref-deploy")
    r = b["receipt"]

    # the receipt carries the PUBLIC key, never a secret; the sig is the secp256k1 scheme
    assert r["sigModel"].startswith("secp256k1-ecdsa@v1:")
    assert r["sigModelPubKey"] == mk.public_hex and "secret" not in r and len(mk.public_hex) == 66

    # THIRD-PARTY verification with no secret and no key object — just the committed public key, pinned to identity
    v = verify_receipt(b, model_pubkey=mk.public_hex, counterparty_pubkey=ck.public_hex)
    assert v["sigModelOk"] is True and v["sigCounterpartyOk"] is True and v["signatureOk"] is True

    # deterministic signatures (RFC 6979): re-signing the same payload is byte-identical
    msg = canonical_bytes({"x": 1})
    assert mk.sign(msg) == mk.sign(msg)

    # tamper the signature → fails; wrong expected identity → fails
    bad = dict(r); bad["sigModel"] = r["sigModel"][:-2] + ("00" if r["sigModel"][-2:] != "00" else "11")
    assert verify_receipt({"receipt": bad, "preimage": b["preimage"]},
                          model_pubkey=mk.public_hex)["sigModelOk"] is False
    assert verify_ec(msg, mk.sign(msg), expected_pubkey_hex=ck.public_hex) is False  # identity mismatch
    assert verify_ec(msg, mk.sign(msg), expected_pubkey_hex=mk.public_hex) is True


def test_bonsai_runtime_emits_asymmetric_receipt_verifiable_without_secret(tmp_path):
    """End-to-end: the runner signs with EC keys and the receipt re-executes AND its signatures verify from
    the public key alone — the 'reference deployment' path the theory-fidelity review recommended."""
    from trinote.receipts import ec_keygen
    import trinote.infer_int.bonsai_runtime as br
    ref = BonsaiReferenceModel(_small_bonsai())
    mh = "ab" * 32
    ids = [1, 2, 3, 4]
    out = br.generate_bonsai_tokens(ref, ids, 3, sampler=SamplerConfig(mode="greedy"))
    mk = ec_keygen(label="model", secret_hex="cc" * 32)
    ck = ec_keygen(label="counterparty", secret_hex="dd" * 32)
    bundle, verified, emitted = br.emit_and_verify_bonsai_receipt(
        ref, input_ids=ids, output_ids=out, model_digest=mh,
        sampler=SamplerConfig(mode="greedy"), model_key=mk, counterparty_key=ck,
        ledger_path=tmp_path / "ledger.jsonl", broadcast_log=tmp_path / "bcast.log")
    assert bundle["receipt"]["sigModel"].startswith("secp256k1-ecdsa@v1:")
    assert verified["ok"] is True
    assert verified["signatureOk"] is True            # verified from the committed public key, no secret
    assert verified["sigModelPubKey"] == mk.public_hex


# ---------------------------------------------------------------------------
# Native-coverage enforcement: every native parity test below is gated on `q1_native_available()` and
# SKIPS when the .so can't load — so a broken/absent kernel yields a fully-green suite that exercised NONE
# of the byte-exact native paths. In CI (and any gate that sets TRINOTE_REQUIRE_NATIVE=1) this test turns
# that silent skip into a hard FAILURE, asserting the kernel loads and exposes every key symbol.
# ---------------------------------------------------------------------------
_REQUIRE_NATIVE_SYMBOLS = (
    "bonsai_q1_linear_i64", "bonsai_q1_linear_i64_workspace", "bonsai_q1_prepare_i64",
    "bonsai_q1_linear_i64_prepared_multi", "bonsai_rmsnorm_i64", "bonsai_attention_decode_i64",
    "bonsai_silu_i64", "bonsai_q1_argmax_i64_workspace", "bonsai_q1_linear_i64_workspace_scale32",
    "bonsai_q1_linear_i64_workspace_lut32",
)


def test_native_kernel_required_when_flag_set():
    """TRINOTE_REQUIRE_NATIVE=1 (CI) makes the native kernel MANDATORY: the suite fails instead of silently
    skipping the 30+ native byte-exact parity / overflow-guard / lut32 / attention tests on a missing .so."""
    import os
    if os.environ.get("TRINOTE_REQUIRE_NATIVE", "").lower() not in ("1", "true", "yes", "on"):
        pytest.skip("TRINOTE_REQUIRE_NATIVE not set (local dev); CI sets it to enforce native coverage")
    import trinote.infer_int.q1_native as qn
    assert qn.q1_native_available(), (
        "TRINOTE_REQUIRE_NATIVE=1 but the native kernel failed to load — build it with "
        "tools/build_bonsai_q1_kernel.sh")
    lib = qn._load_lib()
    missing = [s for s in _REQUIRE_NATIVE_SYMBOLS if not hasattr(lib, s)]
    assert not missing, f"native kernel is missing required symbols: {missing}"


# ---------------------------------------------------------------------------
# Hardening tests for the performance pass (close audit-identified coverage gaps):
#   * native fast path on the trust-critical VERIFY side of a receipt
#   * direct adversarial coverage of the staged-doubling activation LUT (uint64 wrap)
#   * the native RMSNorm int128 sum-of-squares envelope boundary (rc==4 fallback)
#   * OpenMP thread-count invariance of every native kernel (incl. the argmax merge)
# ---------------------------------------------------------------------------


def test_bonsai_native_verifier_reexecutes_receipt_if_present(monkeypatch):
    """The trustless property is that a VERIFIER can re-run the committed model. The existing native
    receipt test re-executes on the Python oracle; this one enables the native fast path ON THE VERIFIER,
    confirms it reproduces the committed IDs and verifies, counts that the native kernel actually executed
    during verification, and shows a tampered output still fails re-execution."""
    import trinote.infer_int.reference_bonsai as rb
    from trinote.infer_int.q1_native import q1_native_available

    if not q1_native_available():
        pytest.skip("native Q1 kernel (libbonsai_q1_kernel.so) not available")
    art = _small_bonsai(seed=31)
    producer = BonsaiReferenceModel(art)
    assert producer.enable_native() is True
    ids = [5, 6, 7]
    out = producer.generate_greedy_cached(ids, 3)[len(ids):]
    mk = keygen(label="bonsai", secret_hex="11" * 32)
    ck = keygen(label="counterparty", secret_hex="22" * 32)
    mh = "ab" * 32
    bundle = build_receipt(model_hash=mh, input_ids=ids, output_ids=out, sampler={"mode": "greedy"},
                           model_key=mk, counterparty_key=ck, artifact_digest=mh)

    calls = {"n": 0}
    original_native = rb._q1_bl_native

    def _counting_native(*a, **k):
        calls["n"] += 1
        return original_native(*a, **k)

    monkeypatch.setattr(rb, "_q1_bl_native", _counting_native)
    verifier = BonsaiReferenceModel(art)
    assert verifier.enable_native() is True
    v = verify_receipt(bundle, model=verifier, model_digest=mh, model_key=mk, counterparty_key=ck)
    assert v["ok"] and v["reexec"]["ok"]
    assert calls["n"] > 0  # the native kernel really ran on the trust-critical verify side

    bad = list(out)
    bad[0] = (bad[0] + 1) % int(producer.cfg["vocab"])
    bad_bundle = build_receipt(model_hash=mh, input_ids=ids, output_ids=bad, sampler={"mode": "greedy"},
                               model_key=mk, counterparty_key=ck, artifact_digest=mh)
    bad_v = verify_receipt(bad_bundle, model=verifier, model_digest=mh, model_key=mk, counterparty_key=ck)
    assert bad_v["reexec"]["ok"] is False and bad_v["ok"] is False


def test_bonsai_native_q1_lut_recurrence_matches_oracle_adversarial_if_present():
    """Direct adversarial check of the staged-doubling activation LUT (audit: only indirect coverage).
    The oracle computes the Q1 dot WITHOUT a LUT (einsum over signs), so byte-exact agreement on inputs
    that (a) drive a wide spread of the 256 per-lane mask patterns and (b) force uint64 subset-sum and
    signed_sum*scale wrap proves the recurrence reproduces the naive 256-mask table's modular values."""
    from trinote.infer_int.q1_native import q1_linear_native, q1_native_available
    from trinote.infer_int.reference_bonsai import q1_linear_ref

    if not q1_native_available():
        pytest.skip("native Q1 kernel (libbonsai_q1_kernel.so) not available")
    rng = np.random.default_rng(4242)
    frac = 16
    n_blocks = 3
    in_features = n_blocks * 128
    out_features = 512  # plenty of weight rows to spread the 256 per-byte-lane mask patterns
    bits = rng.integers(0, 256, size=(out_features, n_blocks, 16), dtype=np.uint8)
    # scales incl. large + negative magnitudes to drive signed_sum*scale across the uint64 boundary
    scale_fp = rng.integers(-(1 << 40), 1 << 40, size=(out_features, n_blocks), dtype=np.int64)
    cases = (
        rng.integers(-(1 << 13), 1 << 13, size=(2, in_features), dtype=np.int64),   # ordinary magnitude
        rng.integers(-(1 << 62), 1 << 62, size=(3, in_features), dtype=np.int64),   # subset sums wrap u64
        np.full((1, in_features), np.int64(1 << 62), dtype=np.int64),               # extreme uniform
    )
    for x_fp in cases:
        native = q1_linear_native(x_fp, bits, scale_fp, frac)
        oracle = q1_linear_ref(x_fp, bits, scale_fp, frac)
        assert native is not None
        assert np.array_equal(native, oracle)


def test_bonsai_native_rmsnorm_envelope_boundary_if_present():
    """Exercise the native RMSNorm int128 sum-of-squares envelope (audit: only the scalar-int64 fallback
    was covered). A large row whose ssq still fits int128 must equal the big-int oracle exactly; a row
    whose ssq overflows int128 (incl. one row of a batch) must fall back (None) while the Python big-int
    oracle still produces a finite committed result."""
    import trinote.infer_int.q1_native as qn
    from trinote.determinism.fixedpoint import fixed_point_rmsnorm

    if not qn.q1_native_available() or not hasattr(qn._load_lib(), "bonsai_rmsnorm_i64"):
        pytest.skip("native RMSNorm kernel (bonsai_rmsnorm_i64) not available")
    frac = 16
    big = np.int64(1 << 62)
    # in-envelope: ssq ~ 2 * 2**124 < 2**128, so the int128 fast path must match the big-int oracle
    in_env = np.array([[big, -big, big // 3]], dtype=np.int64)
    assert np.array_equal(qn.rmsnorm_native(in_env, frac), fixed_point_rmsnorm(in_env, frac))

    # over-envelope: 8 near-2**63 columns push ssq past int128 -> native falls back, oracle stays finite
    over = np.full((1, 8), np.int64((1 << 63) - 1), dtype=np.int64)
    assert qn.rmsnorm_native(over, frac) is None
    oracle_over = fixed_point_rmsnorm(over, frac)
    assert oracle_over.shape == over.shape
    assert np.all(np.isfinite(oracle_over.astype(np.float64)))

    # mixed batch: a single over-envelope row makes the whole native call fall back (rc==4)
    batch = np.array([[1 << 20, -(1 << 20), 1 << 19, 0, 0, 0],
                      [(1 << 63) - 1] * 6], dtype=np.int64)
    assert qn.rmsnorm_native(batch, frac) is None
    assert fixed_point_rmsnorm(batch, frac).shape == batch.shape


def _omp_thread_control():
    """Return (set_threads, restore) bound to whichever OpenMP runtime symbol is reachable in this build,
    or (None, None) if the thread count cannot be controlled from Python."""
    import ctypes
    import trinote.infer_int.q1_native as qn

    handles = []
    lib = qn._load_lib()
    if lib is not None:
        handles.append(lib)
    for name in ("libgomp.so.1", "libomp.so", "libomp.so.1", "libiomp5.so"):
        try:
            handles.append(ctypes.CDLL(name))
        except OSError:
            pass
    try:
        handles.append(ctypes.CDLL(None))
    except OSError:
        pass
    for h in handles:
        setter = getattr(h, "omp_set_num_threads", None)
        getter = getattr(h, "omp_get_max_threads", None)
        if setter is not None and getter is not None:
            setter.argtypes = [ctypes.c_int]
            setter.restype = None
            getter.argtypes = []
            getter.restype = ctypes.c_int
            original = int(getter())
            return setter, (lambda: setter(original))
    return None, None


def test_bonsai_native_kernels_are_thread_count_invariant_if_present():
    """The native kernels parallelize over INDEPENDENT output elements (no nondeterministic partial-
    reduction combiner), so outputs must be byte-identical regardless of OpenMP thread count. Covers Q1
    linear, RMSNorm, and the argmax cross-thread merge — including a forced all-ties case that must still
    resolve to the lowest index under every thread count (audit: threading determinism untested)."""
    import trinote.infer_int.q1_native as qn

    if not qn.q1_native_available():
        pytest.skip("native Q1 kernel (libbonsai_q1_kernel.so) not available")
    set_threads, restore = _omp_thread_control()
    if set_threads is None:
        pytest.skip("cannot control OpenMP thread count from this build")
    try:
        ref = BonsaiReferenceModel(_small_bonsai(seed=33))
        layer = ref.artifact["layers"][0]
        frac = int(ref.cfg["frac"])
        vocab = int(ref.cfg["vocab"])
        rng = np.random.default_rng(33)
        x_fp = rng.integers(-(1 << 14), 1 << 14, size=(7, ref.cfg["dModel"]), dtype=np.int64)
        rms_x = rng.integers(-(1 << 20), 1 << 20, size=(11, 29), dtype=np.int64)
        rms_gain = rng.integers(1 << 14, 1 << 17, size=29, dtype=np.int64)
        out_bits = ref.artifact["output_bits"]
        out_scale = ref.artifact["output_scale_fp"]
        n_out_blocks = int(out_scale.shape[1])
        # every output row identical -> all logits tie -> lowest index (0) must win under any thread count
        tie_bits = np.zeros((vocab, n_out_blocks, 16), dtype=np.uint8)
        tie_scale = np.ones((vocab, n_out_blocks), dtype=np.int64)

        base = None
        for n in (1, 2, 4, 8):
            set_threads(n)
            lin = qn.q1_linear_native(x_fp, layer["w1_bits"], layer["w1_scale_fp"], frac)
            rms = qn.rmsnorm_native(rms_x, frac, gain_q=rms_gain)
            amx = qn.q1_argmax_native(x_fp, out_bits, out_scale, frac)
            tie = qn.q1_argmax_native(x_fp, tie_bits, tie_scale, frac)
            assert lin is not None and rms is not None and amx is not None and tie is not None
            assert np.array_equal(tie, np.zeros(x_fp.shape[0], dtype=np.int64))
            if base is None:
                base = (lin, rms, amx)
            else:
                assert np.array_equal(lin, base[0])
                assert np.array_equal(rms, base[1])
                assert np.array_equal(amx, base[2])
    finally:
        restore()


def test_bonsai_native_prepared_kernels_thread_count_invariant_if_present():
    """R6: the DEFAULT-on prepared / prepared-multi Q1 kernels (the QKV + gate/up decode path) use a distinct
    OpenMP structure (outer parallel + inner collapse(2)) the workspace test above does not exercise. Sweep
    thread counts and assert byte-identical output — each (token, out_feature) element is owned by exactly one
    thread, so the exact-integer sums are order- (hence thread-count-) invariant."""
    import trinote.infer_int.q1_native as qn

    if not qn.q1_native_available() or not hasattr(qn._load_lib(), "bonsai_q1_linear_i64_prepared_multi"):
        pytest.skip("native prepared-multi kernel (bonsai_q1_linear_i64_prepared_multi) not available")
    set_threads, restore = _omp_thread_control()
    if set_threads is None:
        pytest.skip("cannot control OpenMP thread count from this build")
    try:
        ref = BonsaiReferenceModel(_small_bonsai(seed=37))
        layer = ref.artifact["layers"][0]
        frac = int(ref.cfg["frac"])
        n_blocks = int(layer["wq_scale_fp"].shape[1])
        rng = np.random.default_rng(37)
        x_fp = rng.integers(-(1 << 14), 1 << 14, size=(5, n_blocks * 128), dtype=np.int64)
        weights = [(layer["wq_bits"], layer["wq_scale_fp"]),
                   (layer["wk_bits"], layer["wk_scale_fp"]),
                   (layer["wv_bits"], layer["wv_scale_fp"])]
        base_many = base_one = None
        for n in (1, 2, 4, 8):
            set_threads(n)
            prep = qn.q1_prepare_native(x_fp, n_blocks)
            assert prep is not None
            many = qn.q1_linear_prepared_many_native(prep, weights, frac)
            one = qn.q1_linear_prepared_native(prep, layer["wq_bits"], layer["wq_scale_fp"], frac)
            assert many is not None and one is not None
            if base_many is None:
                base_many, base_one = many, one
            else:
                for got, want in zip(many, base_many):
                    assert np.array_equal(got, want)
                assert np.array_equal(one, base_one)
    finally:
        restore()


# ---------------------------------------------------------------------------
# Narrow int32 Q1 scale cache (PERFORMANCE-DETERMINISM-REVIEW.md Recommendation 7): a native-only
# reproducer that halves Q1 scale-array bandwidth. The committed int64 artifact is unchanged; these
# tests prove byte-exact parity with the int64 native kernel and the LUT-free oracle, the env opt-in,
# and the all-or-nothing range guard.
# ---------------------------------------------------------------------------


def test_bonsai_native_scale_cache_matches_oracle_if_present():
    """The int32 scale cache must be byte-identical to the int64 native kernel and the LUT-free oracle,
    including under uint64 wrap, for the layer projections, the output projection, and argmax."""
    import trinote.infer_int.q1_native as qn
    from trinote.infer_int.reference_bonsai import q1_linear_ref

    if not qn.q1_native_available() or not hasattr(qn._load_lib(), "bonsai_q1_linear_i64_workspace_scale32"):
        pytest.skip("native int32 scale-cache kernel (bonsai_q1_linear_i64_workspace_scale32) not available")
    ref = BonsaiReferenceModel(_small_bonsai(seed=41))
    assert ref.enable_scale_cache() is True
    layer = ref.artifact["layers"][0]
    frac = int(ref.cfg["frac"])
    rng = np.random.default_rng(41)
    for name, width in (("wq", ref.cfg["dModel"]), ("wo", ref.cfg["dModel"]),
                        ("w1", ref.cfg["dModel"]), ("w2", ref.cfg["dFfn"])):
        s64 = layer[f"{name}_scale_fp"]
        s32 = layer[f"{name}_scale_fp_i32"]
        assert s32.dtype == np.int32 and np.array_equal(s32.astype(np.int64), s64)
        for x_fp in (
            rng.integers(-(1 << 14), 1 << 14, size=(3, width), dtype=np.int64),   # ordinary
            rng.integers(-(1 << 62), 1 << 62, size=(2, width), dtype=np.int64),   # forces uint64 wrap
        ):
            oracle = q1_linear_ref(x_fp, layer[f"{name}_bits"], s64, frac)
            narrow = qn.q1_linear_native(x_fp, layer[f"{name}_bits"], s32, frac)
            wide = qn.q1_linear_native(x_fp, layer[f"{name}_bits"], s64, frac)
            assert narrow is not None
            assert np.array_equal(narrow, oracle)
            assert np.array_equal(wide, oracle)

    out_bits = ref.artifact["output_bits"]
    s64o = ref.artifact["output_scale_fp"]
    s32o = ref.artifact["output_scale_fp_i32"]
    x_fp = rng.integers(-(1 << 14), 1 << 14, size=(4, ref.cfg["dModel"]), dtype=np.int64)
    ids = qn.q1_argmax_native(x_fp, out_bits, s32o, frac)
    logits = qn.q1_linear_native(x_fp, out_bits, s64o, frac)
    assert ids is not None and logits is not None
    assert np.array_equal(ids, logits.argmax(axis=1))


def test_bonsai_scale_cache_forward_and_receipt_match_oracle_if_present():
    """End-to-end: full logits, greedy decode, and a receipt produced AND verified entirely on the int32
    scale-cache native path must match the pure-Python oracle byte-for-byte (committed artifact unchanged)."""
    import trinote.infer_int.q1_native as qn

    if not qn.q1_native_available() or not hasattr(qn._load_lib(), "bonsai_q1_linear_i64_workspace_scale32"):
        pytest.skip("native int32 scale-cache kernel (bonsai_q1_linear_i64_workspace_scale32) not available")
    ids = [5, 12, 7, 1, 99]
    oracle_model = BonsaiReferenceModel(_small_bonsai(seed=42))  # deterministic artifact (seeded)
    oracle_logits = oracle_model.forward(ids)
    oracle_greedy = oracle_model.generate_greedy(ids, 5)

    cached = BonsaiReferenceModel(_small_bonsai(seed=42))
    assert cached.enable_native() is True
    assert cached.enable_scale_cache() is True
    assert np.array_equal(cached.forward_fast(ids), oracle_logits)
    assert cached.generate_greedy_cached(ids, 5) == oracle_greedy

    out = cached.generate_greedy_cached(ids, 3)[len(ids):]
    mk = keygen(label="bonsai", secret_hex="11" * 32)
    ck = keygen(label="counterparty", secret_hex="22" * 32)
    mh = "ab" * 32
    bundle = build_receipt(model_hash=mh, input_ids=ids, output_ids=out, sampler={"mode": "greedy"},
                           model_key=mk, counterparty_key=ck, artifact_digest=mh)
    verifier = BonsaiReferenceModel(_small_bonsai(seed=42))
    assert verifier.enable_native() is True
    assert verifier.enable_scale_cache() is True
    v = verify_receipt(bundle, model=verifier, model_digest=mh, model_key=mk, counterparty_key=ck)
    assert v["ok"] and v["reexec"]["ok"]


def test_bonsai_scale_cache_env_default_and_range_guard(monkeypatch):
    """The scale cache is opt-in (default OFF), enabled by TRINOTE_Q1_SCALE_CACHE, and is all-or-nothing:
    any scale outside int32 leaves every weight on the committed int64 representation."""
    import trinote.infer_int.reference_bonsai as rb
    from trinote.infer_int.q1_native import q1_native_available

    monkeypatch.delenv("TRINOTE_Q1_SCALE_CACHE", raising=False)
    assert rb._scale_cache_enabled() is False
    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv("TRINOTE_Q1_SCALE_CACHE", value)
        assert rb._scale_cache_enabled() is True
    for value in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("TRINOTE_Q1_SCALE_CACHE", value)
        assert rb._scale_cache_enabled() is False

    ref = BonsaiReferenceModel(_small_bonsai(seed=43))
    assert ref.enable_scale_cache() is True
    assert ref.artifact["layers"][0]["wq_scale_fp_i32"].dtype == np.int32
    assert ref.artifact["output_scale_fp_i32"].dtype == np.int32

    bad = _small_bonsai(seed=44)
    bad["layers"][0]["wq_scale_fp"][0, 0] = np.int64(1 << 40)  # exceeds int32
    bad_model = BonsaiReferenceModel(bad)
    assert bad_model.enable_scale_cache() is False
    assert "wq_scale_fp_i32" not in bad["layers"][0]
    assert "output_scale_fp_i32" not in bad

    monkeypatch.setenv("TRINOTE_Q1_SCALE_CACHE", "1")
    if q1_native_available():
        auto = BonsaiReferenceModel(_small_bonsai(seed=45))
        assert auto.enable_native() is True            # env opt-in -> enable_native builds the cache
        assert getattr(auto, "_scale_cache", False) is True
        assert auto.artifact["output_scale_fp_i32"].dtype == np.int32


def test_bonsai_native_prepared_multi_mixed_scale_dtype_is_safe_if_present():
    """A prepared-multi batch that MIXES an int32 scale cache with int64 scales must not hand an int32
    array to the int64 kernel (which would read 4-byte entries as 8-byte scales -> silent corruption);
    the wrapper canonicalizes to int64, so the result still equals the oracle. enable_scale_cache is
    all-or-nothing so this mix is unreachable in production, but the wrapper must stay safe for any caller."""
    import trinote.infer_int.q1_native as qn
    from trinote.infer_int.reference_bonsai import q1_linear_ref

    if not qn.q1_native_available() or not hasattr(qn._load_lib(), "bonsai_q1_linear_i64_prepared_multi"):
        pytest.skip("native prepared-multi kernel (bonsai_q1_linear_i64_prepared_multi) not available")
    ref = BonsaiReferenceModel(_small_bonsai(seed=46))
    assert ref.enable_scale_cache() is True
    layer = ref.artifact["layers"][0]
    frac = int(ref.cfg["frac"])
    rng = np.random.default_rng(46)
    x_fp = rng.integers(-(1 << 14), 1 << 14, size=(3, ref.cfg["dModel"]), dtype=np.int64)
    prep = qn.q1_prepare_native(x_fp, int(layer["wq_scale_fp"].shape[1]))
    assert prep is not None
    names = ("wq", "wk", "wv")
    mixed = [
        (layer["wq_bits"], layer["wq_scale_fp_i32"]),   # int32 cache
        (layer["wk_bits"], layer["wk_scale_fp"]),       # int64
        (layer["wv_bits"], layer["wv_scale_fp_i32"]),   # int32 cache
    ]
    got = qn.q1_linear_prepared_many_native(prep, mixed, frac)
    assert got is not None
    for i, name in enumerate(names):
        oracle = q1_linear_ref(x_fp, layer[f"{name}_bits"], layer[f"{name}_scale_fp"], frac)
        assert np.array_equal(got[i], oracle)


# ---------------------------------------------------------------------------
# Native M=1 cached-decode attention (PERFORMANCE-DETERMINISM-REVIEW.md Recommendation 4): a native
# kernel for the single-query decode step, byte-identical to the NumPy fixed-point path. These cover
# direct parity across context lengths, the fail-loud overflow fallback, thread invariance, the env
# opt-out, and end-to-end ID equivalence. (End-to-end is also implicitly covered by the existing
# greedy-cached-vs-uncached-oracle tests, which now run native attention by default.)
# ---------------------------------------------------------------------------


def _numpy_decode_attention(q, k, v, frac, inv_sqrt_fp):
    """Reference M=1 attention: exact replica of the NumPy per-head loop (no mask). q:(H,hd) k,v:(Hkv,L,hd)."""
    from trinote.determinism.fixedpoint import fixed_point_matmul, fixed_point_softmax
    H, hd = q.shape
    Hkv = k.shape[0]
    rep = H // Hkv
    out = np.empty((H, hd), dtype=np.int64)
    for h in range(H):
        kv = h // rep
        scores = fixed_point_matmul(q[h:h + 1], k[kv].T, frac)
        scores = (scores * inv_sqrt_fp) >> frac
        probs = fixed_point_softmax(scores, frac)
        out[h] = fixed_point_matmul(probs, v[kv], frac)[0]
    return out


def test_bonsai_native_attention_matches_oracle_if_present():
    """Native M=1 attention must be byte-identical to the NumPy fixed-point per-head path (the integer
    softmax, the two floor shifts, and both matmuls) across context lengths and GQA shapes."""
    import trinote.infer_int.q1_native as qn

    if not qn.q1_native_available() or not hasattr(qn._load_lib(), "bonsai_attention_decode_i64"):
        pytest.skip("native attention kernel (bonsai_attention_decode_i64) not available")
    frac = 16
    rng = np.random.default_rng(51)
    for H, Hkv, hd in ((4, 2, 32), (8, 8, 16), (6, 3, 8)):
        inv_sqrt_fp = round((1.0 / np.sqrt(hd)) * (1 << frac))
        for L in (1, 2, 7, 64, 257):
            q = rng.integers(-(1 << 16), 1 << 16, size=(H, hd), dtype=np.int64)
            k = rng.integers(-(1 << 16), 1 << 16, size=(Hkv, L, hd), dtype=np.int64)
            v = rng.integers(-(1 << 16), 1 << 16, size=(Hkv, L, hd), dtype=np.int64)
            native = qn.attention_decode_native(q, k, v, frac, inv_sqrt_fp)
            oracle = _numpy_decode_attention(q, k, v, frac, inv_sqrt_fp)
            assert native is not None
            assert np.array_equal(native, oracle)


def test_bonsai_native_attention_fail_loud_overflow_if_present():
    """Attention must NOT silently wrap. When a head would overflow the int64 matmul bound, the native
    kernel returns None (so the caller falls back to the NumPy path, which raises) — matching the
    fixed_point_matmul fail-loud contract rather than the Q1 wrap policy."""
    import trinote.infer_int.q1_native as qn
    from trinote.determinism.fixedpoint import fixed_point_matmul

    if not qn.q1_native_available() or not hasattr(qn._load_lib(), "bonsai_attention_decode_i64"):
        pytest.skip("native attention kernel (bonsai_attention_decode_i64) not available")
    frac = 16
    H, Hkv, hd, L = 2, 1, 8, 4
    inv_sqrt_fp = round((1.0 / np.sqrt(hd)) * (1 << frac))
    big = np.int64(1 << 40)  # max|q|*max|k|*hd = 2^40*2^40*8 = 2^83 > 2^63
    q = np.full((H, hd), big, dtype=np.int64)
    k = np.full((Hkv, L, hd), big, dtype=np.int64)
    v = np.zeros((Hkv, L, hd), dtype=np.int64)
    assert qn.attention_decode_native(q, k, v, frac, inv_sqrt_fp) is None
    # the NumPy path it falls back to does raise (fail loud), confirming the contract
    with pytest.raises(OverflowError):
        fixed_point_matmul(q[0:1], k[0].T, frac)

    # Regression: the bound check must not itself wrap mod 2^128. With max|q|*max|k|*hd = 2^61*2^61*128
    # = 2^130, a naive 128-bit triple product wraps to 0 (<= INT64_MAX) and would silently defeat the
    # guard; the division-form check must still detect the overflow and return None (oracle raises).
    big2 = np.int64(1 << 61)
    H2, Hkv2, hd2, L2 = 1, 1, 128, 2
    inv2 = round((1.0 / np.sqrt(hd2)) * (1 << frac))
    q2 = np.full((H2, hd2), big2, dtype=np.int64)
    k2 = np.full((Hkv2, L2, hd2), big2, dtype=np.int64)
    v2 = np.zeros((Hkv2, L2, hd2), dtype=np.int64)
    assert qn.attention_decode_native(q2, k2, v2, frac, inv2) is None
    with pytest.raises(OverflowError):
        fixed_point_matmul(q2[0:1], k2[0].T, frac)


def test_bonsai_native_attention_thread_count_invariant_if_present():
    """Native attention parallelizes over independent heads, so the output must be byte-identical across
    OpenMP thread counts."""
    import trinote.infer_int.q1_native as qn

    if not qn.q1_native_available() or not hasattr(qn._load_lib(), "bonsai_attention_decode_i64"):
        pytest.skip("native attention kernel (bonsai_attention_decode_i64) not available")
    set_threads, restore = _omp_thread_control()
    if set_threads is None:
        pytest.skip("cannot control OpenMP thread count from this build")
    try:
        frac = 16
        H, Hkv, hd, L = 8, 2, 16, 128
        inv_sqrt_fp = round((1.0 / np.sqrt(hd)) * (1 << frac))
        rng = np.random.default_rng(52)
        q = rng.integers(-(1 << 16), 1 << 16, size=(H, hd), dtype=np.int64)
        k = rng.integers(-(1 << 16), 1 << 16, size=(Hkv, L, hd), dtype=np.int64)
        v = rng.integers(-(1 << 16), 1 << 16, size=(Hkv, L, hd), dtype=np.int64)
        base = None
        for n in (1, 2, 4, 8):
            set_threads(n)
            got = qn.attention_decode_native(q, k, v, frac, inv_sqrt_fp)
            assert got is not None
            if base is None:
                base = got
            else:
                assert np.array_equal(got, base)
    finally:
        restore()


def test_bonsai_native_attention_env_default_and_end_to_end_if_present(monkeypatch):
    """TRINOTE_NATIVE_ATTN defaults on with a 0/off opt-out, and toggling it does not change output IDs
    (native attention is byte-identical to the NumPy path)."""
    import trinote.infer_int.reference_bonsai as rb
    from trinote.infer_int.q1_native import q1_native_available

    monkeypatch.delenv("TRINOTE_NATIVE_ATTN", raising=False)
    assert rb._native_attn_enabled() is True
    for value in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("TRINOTE_NATIVE_ATTN", value)
        assert rb._native_attn_enabled() is False
    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv("TRINOTE_NATIVE_ATTN", value)
        assert rb._native_attn_enabled() is True

    if not q1_native_available():
        pytest.skip("native Q1 kernel (libbonsai_q1_kernel.so) not available")
    ids = [5, 12, 7, 1, 99]
    oracle = BonsaiReferenceModel(_small_bonsai(seed=53)).generate_greedy(ids, 6)

    monkeypatch.setenv("TRINOTE_NATIVE_ATTN", "1")
    on = BonsaiReferenceModel(_small_bonsai(seed=53))
    assert on.enable_native() is True
    ids_on = on.generate_greedy_cached(ids, 6)

    monkeypatch.setenv("TRINOTE_NATIVE_ATTN", "0")
    off = BonsaiReferenceModel(_small_bonsai(seed=53))
    assert off.enable_native() is True
    ids_off = off.generate_greedy_cached(ids, 6)

    assert ids_on == oracle
    assert ids_off == oracle


# ---------------------------------------------------------------------------
# Native fixed-point SiLU (optimization-scopes/NATIVE-SILU.md): a native element-wise kernel byte-identical
# to fixed_point_silu, reusing the integer-softmax helpers. Default-on under the native path
# (TRINOTE_NATIVE_SILU=0 opts out); forward() stays on the NumPy oracle (native=False) so it remains a
# faithful comparison oracle.
# ---------------------------------------------------------------------------


def test_bonsai_native_silu_matches_oracle_if_present():
    """Native SiLU must be byte-identical to the NumPy fixed_point_silu (sigmoid d_clip form, the (x*sig)
    wrap, floor shifts) across magnitudes including the int64 extremes."""
    import trinote.infer_int.q1_native as qn
    from trinote.infer_int.reference_bonsai import fixed_point_silu

    if not qn.q1_native_available() or not hasattr(qn._load_lib(), "bonsai_silu_i64"):
        pytest.skip("native SiLU kernel (bonsai_silu_i64) not available")
    rng = np.random.default_rng(61)
    # Sweep frac: 16 (committed) AND 29 (the envelope edge where the kernel d_clip previously omitted the
    # (1<<62)//log2e cap the oracle applies, so native and oracle diverged — review-3 MEDIUM).
    for frac in (16, 29):
        for shape in ((4, 256), (1, 33), (7, 129), (3, 12288)):
            for lo, hi in ((-(1 << 14), 1 << 14), (-(1 << 20), 1 << 20), (-(1 << 40), 1 << 40)):
                x = rng.integers(lo, hi, size=shape, dtype=np.int64)
                native = qn.silu_native(x, frac)
                oracle = fixed_point_silu(x, frac, native=False)
                assert native is not None
                assert np.array_equal(native, oracle), f"frac={frac}"
        edges = np.array([[np.iinfo(np.int64).max, np.iinfo(np.int64).min, 0, 1, -1,
                           1 << 16, -(1 << 16), 1 << 45]], dtype=np.int64)
        assert np.array_equal(qn.silu_native(edges, frac), fixed_point_silu(edges, frac, native=False)), \
            f"frac={frac} edges"


def test_bonsai_native_silu_thread_count_invariant_if_present():
    """SiLU partitions independent output elements, so the result is byte-identical across thread counts."""
    import trinote.infer_int.q1_native as qn

    if not qn.q1_native_available() or not hasattr(qn._load_lib(), "bonsai_silu_i64"):
        pytest.skip("native SiLU kernel (bonsai_silu_i64) not available")
    set_threads, restore = _omp_thread_control()
    if set_threads is None:
        pytest.skip("cannot control OpenMP thread count from this build")
    try:
        rng = np.random.default_rng(62)
        x = rng.integers(-(1 << 20), 1 << 20, size=(5, 4096), dtype=np.int64)
        base = None
        for n in (1, 2, 4, 8):
            set_threads(n)
            got = qn.silu_native(x, 16)
            assert got is not None
            if base is None:
                base = got
            else:
                assert np.array_equal(got, base)
    finally:
        restore()


def test_bonsai_native_silu_env_default_and_end_to_end_if_present(monkeypatch):
    """TRINOTE_NATIVE_SILU defaults on with a 0/off opt-out, and toggling it does not change output IDs."""
    import trinote.infer_int.reference_bonsai as rb
    from trinote.infer_int.q1_native import q1_native_available

    monkeypatch.delenv("TRINOTE_NATIVE_SILU", raising=False)
    assert rb._native_silu_enabled() is True
    for value in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("TRINOTE_NATIVE_SILU", value)
        assert rb._native_silu_enabled() is False
    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv("TRINOTE_NATIVE_SILU", value)
        assert rb._native_silu_enabled() is True

    if not q1_native_available():
        pytest.skip("native Q1 kernel (libbonsai_q1_kernel.so) not available")
    ids = [5, 12, 7, 1, 99]
    oracle = BonsaiReferenceModel(_small_bonsai(seed=61)).generate_greedy(ids, 6)
    monkeypatch.setenv("TRINOTE_NATIVE_SILU", "1")
    on = BonsaiReferenceModel(_small_bonsai(seed=61))
    assert on.enable_native() is True
    assert on.generate_greedy_cached(ids, 6) == oracle
    monkeypatch.setenv("TRINOTE_NATIVE_SILU", "0")
    off = BonsaiReferenceModel(_small_bonsai(seed=61))
    assert off.enable_native() is True
    assert off.generate_greedy_cached(ids, 6) == oracle


# ---------------------------------------------------------------------------
# int32 activation-LUT-entry kernels (optimization-scopes/INT32-LUT-ENTRY.md): opt-in (TRINOTE_Q1_LUT32)
# kernels that halve the Q1 gather data, byte-identical to the uint64-LUT path for in-envelope blocks with a
# per-block range guard that falls back to the int64 LUT. Cover parity (linear/prepared/multi/argmax),
# range-guard fallback, thread invariance, and the env opt-in + end-to-end ID equivalence.
# ---------------------------------------------------------------------------


def test_bonsai_native_lut32_matches_oracle_if_present():
    """The int32-LUT-entry kernels must be byte-identical to the uint64-LUT path and the LUT-free oracle for
    in-envelope inputs, across the single/prepared/prepared-multi/argmax variants."""
    import trinote.infer_int.q1_native as qn
    from trinote.infer_int.reference_bonsai import q1_linear_ref

    if not qn.q1_native_available() or not hasattr(qn._load_lib(), "bonsai_q1_linear_i64_workspace_lut32"):
        pytest.skip("native int32-LUT kernels not available")
    ref = BonsaiReferenceModel(_small_bonsai(seed=71))
    layer = ref.artifact["layers"][0]
    frac = int(ref.cfg["frac"])
    rng = np.random.default_rng(71)
    # single (workspace) lut32
    for name, width in (("wq", ref.cfg["dModel"]), ("wo", ref.cfg["dModel"]),
                        ("w1", ref.cfg["dModel"]), ("w2", ref.cfg["dFfn"])):
        x = rng.integers(-(1 << 14), 1 << 14, size=(3, width), dtype=np.int64)
        nat = qn.q1_linear_native(x, layer[f"{name}_bits"], layer[f"{name}_scale_fp"], frac, lut32=True)
        oracle = q1_linear_ref(x, layer[f"{name}_bits"], layer[f"{name}_scale_fp"], frac)
        assert nat is not None and np.array_equal(nat, oracle)
    # prepared + prepared-multi lut32
    x = rng.integers(-(1 << 14), 1 << 14, size=(2, ref.cfg["dModel"]), dtype=np.int64)
    prep = qn.q1_prepare_native(x, int(layer["wq_scale_fp"].shape[1]), lut32=True)
    assert prep is not None and prep.lut.dtype == np.int32
    names = ("wq", "wk", "wv")
    many = qn.q1_linear_prepared_many_native(
        prep, [(layer[f"{n}_bits"], layer[f"{n}_scale_fp"]) for n in names], frac)
    assert many is not None
    for i, name in enumerate(names):
        oracle = q1_linear_ref(x, layer[f"{name}_bits"], layer[f"{name}_scale_fp"], frac)
        single = qn.q1_linear_prepared_native(prep, layer[f"{name}_bits"], layer[f"{name}_scale_fp"], frac)
        assert single is not None and np.array_equal(single, oracle)
        assert np.array_equal(many[i], oracle)
    # argmax lut32 (the vocab-head rider)
    xo = rng.integers(-(1 << 14), 1 << 14, size=(4, ref.cfg["dModel"]), dtype=np.int64)
    ids = qn.q1_argmax_native(xo, ref.artifact["output_bits"], ref.artifact["output_scale_fp"], frac, lut32=True)
    logits = q1_linear_ref(xo, ref.artifact["output_bits"], ref.artifact["output_scale_fp"], frac)
    assert ids is not None and np.array_equal(ids, logits.argmax(axis=1))


def test_bonsai_native_lut32_range_guard_falls_back_if_present():
    """A block whose activations exceed the int32 LUT envelope must make the int32 prepare signal fallback
    (None / rc 5), and the lut32 linear must transparently fall back to the uint64 LUT, still == oracle."""
    import trinote.infer_int.q1_native as qn
    from trinote.infer_int.reference_bonsai import q1_linear_ref

    if not qn.q1_native_available() or not hasattr(qn._load_lib(), "bonsai_q1_linear_i64_workspace_lut32"):
        pytest.skip("native int32-LUT kernels not available")
    ref = BonsaiReferenceModel(_small_bonsai(seed=72))
    layer = ref.artifact["layers"][0]
    frac = int(ref.cfg["frac"])
    rng = np.random.default_rng(72)
    x = rng.integers(-(1 << 14), 1 << 14, size=(1, ref.cfg["dModel"]), dtype=np.int64)
    x[0, 0] = np.int64(1 << 40)   # one activation past int32 -> its lane abs-sum exceeds INT32_MAX
    assert qn.q1_prepare_native(x, int(layer["wq_scale_fp"].shape[1]), lut32=True) is None  # rc 5 -> fallback
    nat = qn.q1_linear_native(x, layer["wq_bits"], layer["wq_scale_fp"], frac, lut32=True)   # falls back
    oracle = q1_linear_ref(x, layer["wq_bits"], layer["wq_scale_fp"], frac)
    assert nat is not None and np.array_equal(nat, oracle)


def test_bonsai_native_lut32_thread_count_invariant_if_present():
    import trinote.infer_int.q1_native as qn

    if not qn.q1_native_available() or not hasattr(qn._load_lib(), "bonsai_q1_linear_i64_workspace_lut32"):
        pytest.skip("native int32-LUT kernels not available")
    set_threads, restore = _omp_thread_control()
    if set_threads is None:
        pytest.skip("cannot control OpenMP thread count from this build")
    try:
        ref = BonsaiReferenceModel(_small_bonsai(seed=73))
        layer = ref.artifact["layers"][0]
        frac = int(ref.cfg["frac"])
        rng = np.random.default_rng(73)
        x = rng.integers(-(1 << 14), 1 << 14, size=(7, ref.cfg["dModel"]), dtype=np.int64)
        base = None
        for n in (1, 2, 4, 8):
            set_threads(n)
            got = qn.q1_linear_native(x, layer["w1_bits"], layer["w1_scale_fp"], frac, lut32=True)
            assert got is not None
            if base is None:
                base = got
            else:
                assert np.array_equal(got, base)
    finally:
        restore()


def test_bonsai_lut32_env_default_and_end_to_end_if_present(monkeypatch):
    """TRINOTE_Q1_LUT32 is opt-in (default OFF), and enabling it does not change output IDs."""
    import trinote.infer_int.reference_bonsai as rb
    from trinote.infer_int.q1_native import q1_native_available

    monkeypatch.delenv("TRINOTE_Q1_LUT32", raising=False)
    assert rb._lut32_enabled() is False
    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv("TRINOTE_Q1_LUT32", value)
        assert rb._lut32_enabled() is True
    for value in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("TRINOTE_Q1_LUT32", value)
        assert rb._lut32_enabled() is False

    if not q1_native_available():
        pytest.skip("native Q1 kernel (libbonsai_q1_kernel.so) not available")
    ids = [5, 12, 7, 1, 99]
    oracle = BonsaiReferenceModel(_small_bonsai(seed=74)).generate_greedy(ids, 6)
    monkeypatch.setenv("TRINOTE_Q1_LUT32", "1")
    on = BonsaiReferenceModel(_small_bonsai(seed=74))
    assert on.enable_native() is True
    assert on.generate_greedy_cached(ids, 6) == oracle

# ---------------------------------------------------------------------------
# Hardening guards added in the low/info + round-2 batch. Each is written to FAIL on the pre-fix code:
#   R5 RoPE fail-loud overflow · #5 native/oracle RMSNorm gain-policy parity · #1 sampler temp-scale guard ·
#   #2 sigmoid frac assert · #7 scale dtype guard · R19 typed GGUF config-KV errors.
# ---------------------------------------------------------------------------
def test_rope_fixed_point_fails_loud_on_overflow():
    """R5: the RoPE multiply-add now fails loud instead of silently wrapping int64 (RoPE has no native peer,
    so a wrap would be a deterministic-but-wrong committed logit). A normal magnitude is unaffected."""
    from trinote.model.rope_v2 import apply_rope_fixed_neox
    from trinote.model.rope import apply_rope_fixed
    frac = 16
    cos = np.full((2, 2), 1 << 15, dtype=np.int64)
    sin = np.full((2, 2), 1 << 15, dtype=np.int64)
    big = np.full((1, 2, 4), 1 << 60, dtype=np.int64)
    small = np.full((1, 2, 4), 1 << 10, dtype=np.int64)
    for fn in (apply_rope_fixed_neox, apply_rope_fixed):
        with pytest.raises(OverflowError):
            fn(big, cos, sin, frac)
        out = fn(small, cos, sin, frac)        # in-envelope: no raise
        assert out.shape == small.shape


def test_native_and_oracle_rmsnorm_agree_on_overflow_branch_if_present():
    """#5: a gain that trips the oracle's coarse fail-loud envelope (max|out|*max|gain| > INT64_MAX) but
    whose per-element shifted products would fit int64 must make BOTH paths refuse — the oracle raises and the
    native kernel falls back (None). Pre-fix, native returned a value here while the oracle raised (a silent
    divergence). In-range gains stay byte-identical."""
    import trinote.infer_int.q1_native as qn
    from trinote.determinism.fixedpoint import fixed_point_rmsnorm

    if not qn.q1_native_available() or not hasattr(qn._load_lib(), "bonsai_rmsnorm_i64"):
        pytest.skip("native RMSNorm kernel (bonsai_rmsnorm_i64) not available")
    frac = 16
    rng = np.random.default_rng(5)
    x = rng.integers(-(1 << 20), 1 << 20, size=(4, 29), dtype=np.int64)
    # max|normalized| ~ 2^17 here; gain 2^52 makes the COARSE product (~2^69) trip INT64_MAX while the
    # per-element shifted product (~2^53) still fits int64 — exactly the gap where pre-fix native diverged.
    gain = np.full(29, np.int64(1) << 52, dtype=np.int64)
    with pytest.raises(OverflowError):
        fixed_point_rmsnorm(x, frac, gain_q=gain)
    assert qn.rmsnorm_native(x, frac, gain_q=gain) is None    # native refuses too (rc 4 -> oracle fallback)
    # in-range gain: native byte-identical to the oracle
    g_ok = rng.integers(1 << 14, 1 << 17, size=29, dtype=np.int64)
    assert np.array_equal(qn.rmsnorm_native(x, frac, gain_q=g_ok), fixed_point_rmsnorm(x, frac, gain_q=g_ok))


def test_sampler_temp_guard_catches_int64_min_logit():
    """#1: a most-negative logit must not slip past the temperature-overflow guard (the old np.abs(row).max()
    wrapped abs(INT64_MIN) back to a negative value, under-counting the peak)."""
    from trinote.infer_int.sampler import _apply_temp_fp
    row = np.array([np.iinfo(np.int64).min, 0, 5], dtype=np.int64)
    with pytest.raises(OverflowError):
        _apply_temp_fp(row, 1 << 16, 16)


def test_fixed_point_sigmoid_asserts_frac_envelope():
    """#2: fixed_point_sigmoid now enforces the same [1,29] frac envelope as softmax / the native kernel."""
    from trinote.determinism.fixedpoint import fixed_point_sigmoid
    assert fixed_point_sigmoid(np.array([0, 1, -1], dtype=np.int64), 16).shape == (3,)
    with pytest.raises(AssertionError):
        fixed_point_sigmoid(np.array([0], dtype=np.int64), 30)


def test_contiguous_q1_weight_rejects_non_integer_scale():
    """#7: a float (or otherwise non-integer) scale would be silently truncated; reject it loudly."""
    from trinote.infer_int.q1_native import _contiguous_q1_weight
    bits = np.zeros((1, 1, 16), dtype=np.uint8)
    with pytest.raises(TypeError):
        _contiguous_q1_weight(bits, np.zeros((1, 1), dtype=np.float64))
    # an integer scale is accepted
    b, s, out_f, n_blocks = _contiguous_q1_weight(bits, np.zeros((1, 1), dtype=np.int64))
    assert out_f == 1 and n_blocks == 1


def test_gguf_typed_config_kv_helpers_fail_loud():
    """R19: required GGUF config KVs report a clear error (missing / wrong type / non-integral) instead of a
    bare int()/float() exception."""
    from trinote.infer_int.import_bonsai_gguf import _req_int, _req_int_from_float
    assert _req_int({"k": 7}, "k") == 7
    assert _req_int_from_float({"k": 1000000.0}, "k") == 1000000
    with pytest.raises(ValueError):
        _req_int({}, "missing")
    with pytest.raises(ValueError):
        _req_int({"k": "not-an-int"}, "k")
    with pytest.raises(ValueError):
        _req_int_from_float({"k": 1.5}, "k")          # non-integral must not silently truncate


# ---------------------------------------------------------------------------
# Local receipt bundle (BSV off) is reproducible: pack it, then VERIFY BY IN-MODEL REPLAY — re-execution
# must reproduce the committed token ids byte-exactly (reexecOk + artifactBoundOk). A wrong model digest
# must break the artifact binding. This is the end-to-end "reproducible thing" the bundle exists to provide.
# ---------------------------------------------------------------------------
def test_local_bundle_replay_reproduces_byte_exact():
    import trinote.infer_int.bonsai_runtime as br
    from trinote.bundle import pack_bundle, verify_bundle
    import tempfile, pathlib

    ref = BonsaiReferenceModel(_small_bonsai(seed=51))
    mh = "ab" * 32
    ids = [1, 2, 3, 4]
    out = br.generate_bonsai_tokens(ref, ids, 5, sampler=SamplerConfig(mode="greedy"))
    with tempfile.TemporaryDirectory() as td:
        tdp = pathlib.Path(td)
        bundle, verified, emitted = br.emit_and_verify_bonsai_receipt(
            ref, input_ids=ids, output_ids=out, model_digest=mh,
            sampler=SamplerConfig(mode="greedy"),
            ledger_path=tdp / "ledger.jsonl", broadcast_log=tdp / "bcast.log")
        assert verified["ok"]
        transcript = {"prompt": "count r's in strawberry", "output": "Answer: 3", "modelLabel": "small-bonsai"}
        res = pack_bundle(bundle=bundle, onchain=None, out_dir=tdp / "b.tar.gz",
                          ledger_entry=emitted["ledgerEntry"], transcript=transcript, as_tar=True)
        # offline only (no model) — internal consistency
        assert verify_bundle(res["path"])["ok"]
        # REPLAY: load the model + its committed digest and RE-EXECUTE → must reproduce the committed ids
        rep = verify_bundle(res["path"], reexec=True, model=ref, model_digest=mh)
        rx = rep["reexec"]
        assert rep["ok"] and rx["reexecOk"] and rx["artifactBoundOk"]
        assert rx["raw"]["reexec"]["strategy"] in ("greedy-full", "greedy-cached-replay")
        # a WRONG artifact digest must break the binding (and therefore the layer) — the #9 guarantee
        bad = verify_bundle(res["path"], reexec=True, model=ref, model_digest="00" * 32)
        assert not bad["ok"] and bad["reexec"]["artifactBoundOk"] is False
