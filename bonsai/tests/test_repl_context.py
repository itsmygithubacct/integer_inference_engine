from __future__ import annotations

import io
import os
import select
import termios
from types import SimpleNamespace

import numpy as np
import pytest

from trinote.cli.context_window import resolve_context_window
from trinote.cli.conversation import ContextOverflow, Conversation
from trinote.cli.live_session import LiveNativeSession
from trinote.cli.repl import TerminalNoise, TerminalRepl, parse_command
from trinote.cli.run_bonsai_cli import _context_size_arg, _handle_session_command
from trinote.infer_int.reference_bonsai import BonsaiReferenceModel, random_bonsai_artifact
from trinote.infer_int.reference_bonsai35 import (
    BonsaiQwen35ReferenceModel,
    random_bonsai35_artifact,
)


def _artifact(architecture: str, context: int, *, attention_layers: int = 1) -> dict:
    return {
        "config": {
            "architecture": architecture,
            "context_len": context,
            "n_heads_kv": 4,
            "head_dim": 128,
        },
        "cos_fp": SimpleNamespace(shape=(context, 64)),
        "layers": [
            {"kind": "attention"} for _ in range(attention_layers)
        ],
    }


def test_native_auto_context_prefers_original_rope_window_for_qwen3():
    profile = resolve_context_window(
        {
            "general.architecture": "qwen3",
            "general.name": "Bonsai-8B",
            "qwen3.context_length": 65_536,
            "qwen3.rope.scaling.original_context_length": 16_384,
        },
        artifact=_artifact("qwen3", 65_536, attention_layers=36),
        available_bytes=256 << 30,
        total_bytes=256 << 30,
    )
    assert profile.effective == 16_384
    assert profile.hard_max == 65_536
    assert "original RoPE window" in profile.reason


def test_native_auto_context_obeys_qwen35_artifact_cap():
    profile = resolve_context_window(
        {
            "general.architecture": "qwen35",
            "general.name": "Bonsai-27B",
            "qwen35.context_length": 262_144,
        },
        artifact=_artifact("qwen35", 4096, attention_layers=16),
        available_bytes=256 << 30,
        total_bytes=256 << 30,
    )
    assert profile.effective == 4096
    assert profile.artifact_max == 4096
    assert "artifact cap" in profile.reason


def test_explicit_context_cannot_exceed_native_artifact():
    with pytest.raises(ValueError, match="artifact maximum 4096"):
        resolve_context_window(
            {"general.architecture": "qwen35", "qwen35.context_length": 262_144},
            artifact=_artifact("qwen35", 4096),
            requested=8192,
        )


def test_prism_auto_delegates_hardware_fit_and_cli_accepts_auto():
    profile = resolve_context_window(
        {"general.architecture": "qwen35", "qwen35.context_length": 262_144},
        backend="prismml.cpp",
    )
    assert profile.effective == 0
    assert profile.automatic is True
    assert _context_size_arg("auto") == 0
    assert _context_size_arg("8192") == 8192


def _byte_tokens(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def test_conversation_replays_exact_outputs_and_evicts_whole_turns():
    conversation = Conversation(
        _byte_tokens,
        architecture="qwen35",
        context_size=4096,
        max_new=32,
        eos_id=999,
        chat=True,
        thinking=False,
        system_prompt="Answer precisely.",
    )
    first = conversation.prepare("first question")
    first_turn = conversation.commit(first, [70, 71], "FG")
    second = conversation.prepare("second question")
    assert tuple(second.input_ids[:len(conversation._system_ids)]) == conversation._system_ids
    assert first_turn.output_ids == (70, 71)
    assert list(first_turn.output_ids) in [
        list(second.input_ids[i:i + 2]) for i in range(len(second.input_ids) - 1)
    ]
    second_turn = conversation.commit(second, [72, 73], "HI")

    full = conversation.prepare("third question")
    first_size = (
        len(first_turn.user_segment_ids) + len(first_turn.output_ids) + len(first_turn.close_ids)
    )
    conversation.set_context_size(conversation.max_new + len(full.input_ids) - first_size)
    trimmed = conversation.prepare("third question")
    assert trimmed.evicted == 1
    assert trimmed.retained_turns == (second_turn,)


def test_conversation_rejects_a_single_turn_larger_than_budget():
    conversation = Conversation(
        _byte_tokens,
        architecture="qwen35",
        context_size=64,
        max_new=32,
        eos_id=None,
        chat=True,
    )
    with pytest.raises(ContextOverflow, match="only 32 fit"):
        conversation.prepare("x" * 100)


def test_repl_commands_and_mouse_escape_rejection():
    command = parse_command('/system "be terse"')
    assert command is not None
    assert command.name == "system"
    assert command.args == ("be terse",)
    assert parse_command(":quit").name == "exit"
    with pytest.raises(TerminalNoise):
        TerminalRepl.sanitize("question\x1b[M !!")


def test_session_context_auto_restores_model_default_and_system_unquotes(tmp_path):
    conversation = Conversation(
        _byte_tokens,
        architecture="qwen35",
        context_size=8192,
        max_new=32,
        eos_id=None,
        context_automatic=False,
    )
    explicit = SimpleNamespace(
        effective=8192, automatic=False, reason="explicit override",
        hard_max=65_536, source_max=262_144, artifact_max=65_536,
    )
    automatic = SimpleNamespace(
        effective=16_384, automatic=True, reason="original RoPE window 16384",
        hard_max=65_536, source_max=262_144, artifact_max=65_536,
    )
    terminal = TerminalRepl(stdin=io.StringIO(), stderr=io.StringIO())

    handled, prompt, leave = _handle_session_command(
        "/context auto",
        terminal=terminal,
        conversation=conversation,
        context_profile=explicit,
        auto_context_profile=automatic,
        last_run={},
        ref=object(),
        model_digest="ab",
        bundle_dir=tmp_path,
    )
    assert (handled, prompt, leave) == (True, None, False)
    assert conversation.context_size == 16_384
    assert conversation.context_automatic is True

    _handle_session_command(
        '/system "be terse"',
        terminal=terminal,
        conversation=conversation,
        context_profile=explicit,
        auto_context_profile=automatic,
        last_run={},
        ref=object(),
        model_digest="ab",
        bundle_dir=tmp_path,
    )
    assert conversation.system_prompt == "be terse"


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY support required")
def test_terminal_quarantine_disables_echo_and_flushes_typeahead():
    master_fd, slave_fd = os.openpty()
    stdin = os.fdopen(os.dup(slave_fd), "r", buffering=1)
    terminal = TerminalRepl(stdin=stdin, stderr=io.StringIO())
    before = termios.tcgetattr(slave_fd)
    try:
        with terminal.quarantine_input():
            quiet = termios.tcgetattr(slave_fd)
            assert not (quiet[3] & termios.ECHO)
            assert quiet[3] & termios.ISIG
            os.write(master_fd, b"\x1b[Mmouse-junk\n")
        restored = termios.tcgetattr(slave_fd)
        assert bool(restored[3] & termios.ECHO) == bool(before[3] & termios.ECHO)
        ready, _, _ = select.select([slave_fd], [], [], 0)
        assert ready == []
    finally:
        stdin.close()
        os.close(master_fd)
        os.close(slave_fd)


class _FakeNativeExecutor:
    def __init__(self) -> None:
        self._history: list[int] = []
        self.prefills = 0
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1
        self._history.clear()

    def prefill_logits(self, ids) -> np.ndarray:
        self.prefills += 1
        self._history = [int(v) for v in ids]
        return np.zeros((1, 32), dtype=np.int64)

    def decode_logits(self, token: int) -> np.ndarray:
        self._history.append(int(token))
        return np.zeros((1, 32), dtype=np.int64)

    def decode(self, token: int) -> np.ndarray:
        self._history.append(int(token))
        return np.zeros((1, 1), dtype=np.int64)


def test_qwen35_live_session_reuses_exact_prefix_and_rebuilds_on_retry():
    executor = _FakeNativeExecutor()
    model = SimpleNamespace(cfg={"frac": 16}, _model_executor=executor)
    session = LiveNativeSession(model, architecture="qwen35", artifact_digest="ab")
    picks = iter((5, 6, 7))
    pick = lambda _row, _position, _history: next(picks)

    first = session.generate([1, 2], 2, pick)
    assert first.output_ids == [5, 6]
    assert executor.prefills == 1
    assert executor._history == [1, 2, 5, 6]

    second_input = [1, 2, 5, 6, 3, 4]
    second = session.generate(second_input, 1, pick)
    assert second.reused_tokens == 4
    assert executor.prefills == 1
    assert executor._history == second_input + [7]

    retry_picks = iter((8,))
    retried = session.generate([1, 9], 1, lambda *_: next(retry_picks))
    assert retried.reused_tokens == 0
    assert executor.prefills == 2
    assert executor._history == [1, 9, 8]


def test_live_session_invalidation_discards_cancelled_prefix():
    executor = _FakeNativeExecutor()
    model = SimpleNamespace(cfg={"frac": 16}, _model_executor=executor)
    session = LiveNativeSession(model, architecture="qwen35", artifact_digest="ab")

    with pytest.raises(KeyboardInterrupt):
        session.generate(
            [1, 2], 2, lambda *_: 5,
            on_token=lambda _token: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
    session.invalidate()
    assert executor._history == []
    assert executor.resets == 1


def test_qwen3_live_cache_matches_fresh_canonical_generation_across_turns():
    cfg = {
        "dModel": 128, "nHeads": 4, "nHeadsKv": 2, "headDim": 32,
        "dFfn": 256, "vocab": 256, "nLayers": 2, "fpFracBits": 16,
        "ropeBase": 1_000_000, "ropeScalingType": "none",
    }
    artifact = random_bonsai_artifact(cfg, seq_len=48, seed=1201)
    model = BonsaiReferenceModel(artifact)
    session = LiveNativeSession(model, architecture="qwen3", artifact_digest="ab")
    greedy = lambda row, _position, _history: int(np.asarray(row).argmax())

    first_input = [1, 2, 3]
    expected_first = BonsaiReferenceModel(artifact).generate_cached(
        first_input, 3, greedy
    )
    first = session.generate(first_input, 3, greedy)
    assert first.output_ids == expected_first

    second_input = first_input + expected_first + [4, 5]
    expected_second = BonsaiReferenceModel(artifact).generate_cached(
        second_input, 2, greedy
    )
    second = session.generate(second_input, 2, greedy)
    assert second.output_ids == expected_second
    assert second.reused_tokens == len(first_input) + len(expected_first)


def test_qwen35_python_live_cache_matches_fresh_generation_across_turns():
    artifact = random_bonsai35_artifact(seq_len=48, seed=1202)
    model = BonsaiQwen35ReferenceModel(artifact)
    session = LiveNativeSession(model, architecture="qwen35", artifact_digest="cd")
    greedy = lambda row, _position, _history: int(np.asarray(row).argmax())

    first_input = [1, 2, 3]
    expected_first = BonsaiQwen35ReferenceModel(artifact).generate_cached(
        first_input, 3, greedy
    )
    first = session.generate(first_input, 3, greedy)
    assert first.output_ids == expected_first

    second_input = first_input + expected_first + [4, 5]
    expected_second = BonsaiQwen35ReferenceModel(artifact).generate_cached(
        second_input, 2, greedy
    )
    second = session.generate(second_input, 2, greedy)
    assert second.output_ids == expected_second
    assert second.reused_tokens == len(first_input) + len(expected_first)
