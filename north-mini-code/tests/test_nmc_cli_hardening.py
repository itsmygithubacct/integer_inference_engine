"""CLI gates that must fail before a bad receipt can be reported as a successful run."""
from types import SimpleNamespace

import pytest

from tools import nmc_cli


def _fake_engine():
    return SimpleNamespace(
        bname="test",
        fused=False,
        cfg=SimpleNamespace(d_model=8, n_experts=2, n_used=1),
        NL=1,
    )


def test_broadcast_requires_receipts_before_model_load(monkeypatch):
    monkeypatch.setattr(nmc_cli, "find_blob", lambda *_: pytest.fail("model lookup must not run"))
    with pytest.raises(SystemExit) as exc:
        nmc_cli.main(["prompt", "--broadcast"])
    assert exc.value.code == 2


@pytest.mark.parametrize("mode", ["oneshot", "repl"])
def test_failed_receipt_returns_nonzero_in_every_interactive_mode(monkeypatch, mode):
    monkeypatch.setattr(nmc_cli, "find_blob", lambda *_: "fake.gguf")
    monkeypatch.setattr(nmc_cli, "Engine", lambda *_: _fake_engine())
    failed = {"offline_ok": False, "verify_bundle": {"ok": False}}
    monkeypatch.setattr(nmc_cli, "run_one", lambda *a, **kw: (None, failed))
    if mode == "repl":
        monkeypatch.setattr("builtins.input", lambda *_: "prompt")
        rc = nmc_cli.main(["repl", "--receipts"])
    else:
        rc = nmc_cli.main(["prompt", "--receipts"])
    assert rc == 1
