"""Engine settings from a TOML config file.

Precedence: an explicit **CLI flag  >  the config file  >  the built-in argparse default**. The config
lives at ``$BONSAI_CONFIG`` (else ``$BONSAI_NOTARY_HOME/bonsai.toml``). Keys mirror the
``run_bonsai_cli`` argument dests (dashes → underscores); ``[sections]`` are cosmetic and flattened on
load. Unknown or out-of-range keys are ignored **with a warning**, so a typo can never silently change
behaviour. Only generation/sampler settings are configurable here — invocation modes (gpu / fast /
receipt) stay launcher- and flag-controlled.
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from .notary_paths import config_path

# Settings the config may set (mirror run_bonsai_cli dests). Anything else is ignored.
CONFIG_KEYS = frozenset({
    "sampler", "max_new", "engine", "chat", "verify_mode",
    "temp", "top_k", "top_p", "min_p", "seed", "rep_penalty", "no_repeat_ngram",
})

# Choice-constrained keys — a value outside the set is dropped (argparse does NOT re-validate defaults).
_CHOICES = {
    "sampler": {"min_p", "qwen3-rec", "greedy", "temp", "top_k", "top_p"},
    "engine": {"native", "prismml.cpp"},
    "verify_mode": {"fast-local", "fresh-oracle"},
}


def load_config(path: str | Path | None = None) -> dict:
    """Return validated ``{dest: value}`` overrides from the TOML config (``{}`` if absent/invalid)."""
    p = Path(path).expanduser() if path is not None else config_path()
    if not p.exists():
        return {}
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError, ValueError) as exc:
        print(f"[bonsai] WARNING: ignoring invalid config {p}: {exc}", file=sys.stderr)
        return {}

    # Flatten any [sections] into one flat dict (section names are cosmetic).
    flat: dict = {}
    for key, val in data.items():
        if isinstance(val, dict):
            flat.update(val)
        else:
            flat[key] = val

    out: dict = {}
    unknown, bad = [], []
    for key, val in flat.items():
        if key not in CONFIG_KEYS:
            unknown.append(key)
        elif key in _CHOICES and val not in _CHOICES[key]:
            bad.append(f"{key}={val!r}")
        else:
            out[key] = val
    if unknown:
        print(f"[bonsai] WARNING: ignoring unknown config key(s) in {p}: {', '.join(sorted(unknown))}",
              file=sys.stderr)
    if bad:
        print(f"[bonsai] WARNING: ignoring out-of-range config value(s) in {p}: {', '.join(sorted(bad))}",
              file=sys.stderr)
    return out
