"""
The maintenance scripts carry on past a store that fails, so the rest still get
done — but they used to exit 0 either way, so a shell or a scheduled run read a
partial run as a clean one. ``recompute_fronts.py`` is the repair path for the
front layers, which is where that mattered.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

#: script -> the per-var_key function its main() loops over
_WORKERS = {
    "recompute_fronts": "run",
    "rechunk_store": "run",
    "repair_axis_drift": "repair",
}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worker(fail_on: set[str]):
    def work(var_key, **_kwargs):
        if var_key in fail_on:
            raise RuntimeError("boom")
        return 1

    return work


@pytest.mark.parametrize("name", sorted(_WORKERS))
def test_a_failed_var_key_makes_the_run_exit_non_zero(name, monkeypatch, capsys):
    script = _load(name)
    monkeypatch.setattr(script, _WORKERS[name], _worker({"bad"}))
    if hasattr(script, "clear_staging"):
        monkeypatch.setattr(script, "clear_staging", lambda _k: 0)

    assert script.main(["bad", "good"]) == 1
    assert "1 var_key(s) failed: bad" in capsys.readouterr().out


@pytest.mark.parametrize("name", sorted(_WORKERS))
def test_a_clean_run_exits_zero(name, monkeypatch):
    script = _load(name)
    monkeypatch.setattr(script, _WORKERS[name], _worker(set()))

    assert script.main(["good", "also_good"]) == 0
