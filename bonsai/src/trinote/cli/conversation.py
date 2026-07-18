"""Structured, token-budgeted conversation state for the Bonsai REPL."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


class ContextOverflow(ValueError):
    """The current user turn cannot fit even after old turns are evicted."""


@dataclass(frozen=True)
class ChatTurn:
    user: str
    user_segment_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    output_text: str
    close_ids: tuple[int, ...]
    thinking: bool


@dataclass(frozen=True)
class PreparedTurn:
    user: str
    user_segment_ids: tuple[int, ...]
    input_ids: tuple[int, ...]
    retained_turns: tuple[ChatTurn, ...]
    evicted: int
    thinking: bool


class Conversation:
    """Build exact append-only Qwen chat tokens and evict whole old turns."""

    def __init__(
        self,
        tokenize: Callable[[str], list[int]],
        *,
        architecture: str,
        context_size: int,
        max_new: int,
        eos_id: int | None,
        chat: bool = True,
        thinking: bool = False,
        system_prompt: str | None = None,
        context_automatic: bool = True,
    ) -> None:
        self._tokenize = tokenize
        self.architecture = str(architecture)
        self.context_size = int(context_size)
        self.max_new = int(max_new)
        self.eos_id = int(eos_id) if eos_id is not None else None
        self.chat = bool(chat)
        self.thinking = bool(thinking) if self.architecture == "qwen35" else False
        self.system_prompt = system_prompt or ""
        self.context_automatic = bool(context_automatic)
        self.turns: list[ChatTurn] = []
        self.evicted_total = 0
        self._system_ids = self._encode_system(self.system_prompt)
        self._newline_ids: tuple[int, ...] | None = None
        self._close_ids: tuple[int, ...] | None = None

    @property
    def input_budget(self) -> int:
        return max(0, self.context_size - self.max_new)

    def _encode(self, text: str) -> tuple[int, ...]:
        return tuple(int(v) for v in self._tokenize(text))

    def _encode_system(self, text: str) -> tuple[int, ...]:
        if not text:
            return ()
        if self.chat:
            return self._encode(f"<|im_start|>system\n{text}<|im_end|>\n")
        return self._encode(f"System: {text}\n\n")

    def _assistant_prefix(self, thinking: bool) -> str:
        if not self.chat:
            return "Assistant:"
        if self.architecture == "qwen35":
            return "<think>\n" if thinking else "<think>\n\n</think>\n\n"
        # The installed Qwen3 template uses its hard non-thinking prefix.
        return "<think>\n\n</think>\n\n"

    def _user_segment(self, user: str, thinking: bool) -> tuple[int, ...]:
        if self.chat:
            text = (
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n{self._assistant_prefix(thinking)}"
            )
        else:
            text = f"User: {user}\n{self._assistant_prefix(thinking)}"
        return self._encode(text)

    def _closure(self, output_ids: tuple[int, ...]) -> tuple[int, ...]:
        if self.chat:
            if output_ids and self.eos_id is not None and output_ids[-1] == self.eos_id:
                if self._newline_ids is None:
                    self._newline_ids = self._encode("\n")
                return self._newline_ids
            if self._close_ids is None:
                self._close_ids = self._encode("<|im_end|>\n")
            return self._close_ids
        if self._newline_ids is None:
            self._newline_ids = self._encode("\n\n")
        return self._newline_ids

    def _assemble(self, turns: list[ChatTurn] | tuple[ChatTurn, ...], current: tuple[int, ...]) -> tuple[int, ...]:
        ids = list(self._system_ids)
        for turn in turns:
            ids.extend(turn.user_segment_ids)
            ids.extend(turn.output_ids)
            ids.extend(turn.close_ids)
        ids.extend(current)
        return tuple(ids)

    def prepare(self, user: str) -> PreparedTurn:
        user = str(user)
        segment = self._user_segment(user, self.thinking)
        retained = list(self.turns)
        evicted = 0
        ids = self._assemble(retained, segment)
        while len(ids) > self.input_budget and retained:
            retained.pop(0)
            evicted += 1
            ids = self._assemble(retained, segment)
        if len(ids) > self.input_budget:
            raise ContextOverflow(
                f"turn needs {len(ids)} input tokens but only {self.input_budget} fit "
                f"({self.context_size} context - {self.max_new} reserved output)"
            )
        return PreparedTurn(
            user=user,
            user_segment_ids=segment,
            input_ids=ids,
            retained_turns=tuple(retained),
            evicted=evicted,
            thinking=self.thinking,
        )

    def commit(self, prepared: PreparedTurn, output_ids: list[int], output_text: str) -> ChatTurn:
        output = tuple(int(v) for v in output_ids)
        turn = ChatTurn(
            user=prepared.user,
            user_segment_ids=prepared.user_segment_ids,
            output_ids=output,
            output_text=str(output_text),
            close_ids=self._closure(output),
            thinking=prepared.thinking,
        )
        self.turns = list(prepared.retained_turns) + [turn]
        self.evicted_total += prepared.evicted
        return turn

    def clear(self) -> None:
        self.turns.clear()

    def retry(self) -> str | None:
        if not self.turns:
            return None
        return self.turns.pop().user

    def set_system(self, text: str) -> None:
        self.system_prompt = str(text)
        self._system_ids = self._encode_system(self.system_prompt)
        self.clear()

    def set_context_size(self, size: int, *, automatic: bool | None = None) -> None:
        size = int(size)
        if size <= self.max_new:
            raise ValueError(
                f"context {size} must exceed the {self.max_new}-token output reserve"
            )
        self.context_size = size
        if automatic is not None:
            self.context_automatic = bool(automatic)

    def retained_tokens(self) -> int:
        return len(self._assemble(self.turns, ()))

    def messages(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        if self.system_prompt:
            rows.append({"role": "system", "content": self.system_prompt})
        for turn in self.turns:
            rows.append({"role": "user", "content": turn.user})
            rows.append({"role": "assistant", "content": turn.output_text})
        return rows
