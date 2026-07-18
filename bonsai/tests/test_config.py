"""Tests for trinote.config — TOML engine settings with CLI > config > default precedence + validation."""
import textwrap

from trinote.config import load_config


def test_flatten_sections_and_known_keys(tmp_path):
    p = tmp_path / "bonsai.toml"
    p.write_text(textwrap.dedent("""
        [inference]
        sampler = "min_p"
        max_new = 32
        [sampling]
        top_k = 7
        rep_penalty = 1.2
    """))
    assert load_config(p) == {"sampler": "min_p", "max_new": 32, "top_k": 7, "rep_penalty": 1.2}


def test_unknown_key_dropped(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('sampler = "greedy"\nbogus_key = 1\n')
    assert load_config(p) == {"sampler": "greedy"}


def test_invalid_choice_dropped(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('sampler = "nope"\nengine = "native"\n')
    assert load_config(p) == {"engine": "native"}   # bad sampler dropped, valid engine kept


def test_missing_file_is_empty():
    assert load_config("/nonexistent/does/not/exist.toml") == {}


def test_invalid_toml_is_empty(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("this = = not valid [[[")
    assert load_config(p) == {}


def test_repl_context_and_system_settings_are_normalized(tmp_path):
    p = tmp_path / "bonsai.toml"
    p.write_text(textwrap.dedent('''
        [repl]
        context_size = "auto"
        system_prompt = "Answer precisely."
        no_think = false
    '''))
    assert load_config(p) == {
        "ctx_size": 0,
        "system_prompt": "Answer precisely.",
        "no_think": False,
    }


def test_invalid_context_setting_is_dropped(tmp_path):
    p = tmp_path / "bonsai.toml"
    p.write_text('context_size = -1\nchat = true\n')
    assert load_config(p) == {"chat": True}
