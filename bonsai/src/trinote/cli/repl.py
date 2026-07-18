"""Terminal-safe input and command parsing for the Bonsai REPL."""
from __future__ import annotations

import contextlib
import re
import shlex
import sys
from dataclasses import dataclass


_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class TerminalNoise(ValueError):
    """A terminal escape/control sequence leaked into an entered line."""


@dataclass(frozen=True)
class ReplCommand:
    name: str
    args: tuple[str, ...]
    raw_args: str = ""


def parse_command(line: str) -> ReplCommand | None:
    stripped = line.strip()
    if not stripped or stripped[0] not in {"/", ":"}:
        return None
    body = stripped[1:]
    name, _, raw = body.partition(" ")
    aliases = {"h": "help", "?": "help", "quit": "exit"}
    name = aliases.get(name.lower(), name.lower())
    try:
        args = tuple(shlex.split(raw)) if raw else ()
    except ValueError as exc:
        raise ValueError(f"cannot parse command: {exc}") from exc
    return ReplCommand(name=name, args=args, raw_args=raw.strip())


class TerminalRepl:
    """Read edited lines and quarantine type-ahead while inference runs."""

    COMMANDS = (
        "/help", "/clear", "/context", "/system", "/think", "/retry",
        "/history", "/paste", "/bundle", "/verify", "/exit",
    )

    def __init__(self, *, stdin=None, stderr=None) -> None:
        self.stdin = stdin or sys.stdin
        self.stderr = stderr or sys.stderr
        self._readline = None
        if self.stdin is sys.stdin:
            try:
                import readline
                self._readline = readline
                readline.parse_and_bind("set editing-mode emacs")
                readline.parse_and_bind("set enable-bracketed-paste on")
                readline.set_completer(self._complete)
                readline.set_completer_delims(" \t\n")
                readline.parse_and_bind("tab: complete")
            except (ImportError, OSError):
                self._readline = None

    def _complete(self, text: str, state: int) -> str | None:
        matches = [item for item in self.COMMANDS if item.startswith(text)]
        return matches[state] if state < len(matches) else None

    @staticmethod
    def sanitize(line: str) -> str:
        if "\x1b" in line or _CONTROL.search(line):
            raise TerminalNoise("ignored terminal escape/control input")
        return line

    def _input(self, prompt: str) -> str:
        if self.stdin is sys.stdin:
            return input(prompt)
        self.stderr.write(prompt)
        self.stderr.flush()
        line = self.stdin.readline()
        if line == "":
            raise EOFError
        return line.rstrip("\r\n")

    def read(self, prompt: str = "bonsai> ") -> str | None:
        """Return a clean line, ``None`` on EOF, and keep Ctrl-C at the prompt local."""
        try:
            line = self._input(prompt)
        except EOFError:
            return None
        except KeyboardInterrupt:
            self.stderr.write("\n")
            self.stderr.flush()
            return ""
        line = self.sanitize(line)
        # A trailing backslash gives a dependency-free multiline input mode.
        parts = []
        while line.endswith("\\") and not line.endswith("\\\\"):
            parts.append(line[:-1])
            try:
                line = self.sanitize(self._input("... "))
            except EOFError:
                break
        parts.append(line)
        return "\n".join(parts)

    def read_paste(self) -> str:
        self.stderr.write("[repl] paste mode; enter a single '.' line to submit\n")
        self.stderr.flush()
        rows: list[str] = []
        while True:
            try:
                row = self._input("... ")
            except EOFError:
                break
            if row == ".":
                break
            rows.append(self.sanitize(row))
        return "\n".join(rows)

    @contextlib.contextmanager
    def quarantine_input(self):
        """Disable echo and discard queued mouse/type-ahead bytes around work."""
        if not getattr(self.stdin, "isatty", lambda: False)():
            yield
            return
        try:
            import termios
            fd = self.stdin.fileno()
            old = termios.tcgetattr(fd)
            quiet = list(old)
            quiet[3] &= ~termios.ECHO
            termios.tcflush(fd, termios.TCIFLUSH)
            termios.tcsetattr(fd, termios.TCSANOW, quiet)
        except (ImportError, OSError, ValueError):
            yield
            return
        try:
            yield
        finally:
            try:
                termios.tcflush(fd, termios.TCIFLUSH)
                termios.tcsetattr(fd, termios.TCSANOW, old)
            except OSError:
                pass
