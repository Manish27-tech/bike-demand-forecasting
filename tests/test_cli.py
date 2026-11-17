"""Tests for the stage runner."""

from __future__ import annotations

import pytest

from bikedemand import cli


def test_stage_names_are_stable():
    assert list(cli.STAGES) == [
        "download",
        "clean",
        "eda",
        "train",
        "diagnostics",
        "report",
    ]


def test_all_runs_every_stage_in_order(monkeypatch):
    calls: list[str] = []

    for name in cli.STAGES:
        monkeypatch.setitem(cli.STAGES, name, lambda name=name, force=False: calls.append(name))

    assert cli.main(["all"]) == 0
    assert calls == list(cli.STAGES)


def test_single_stage_runs_once(monkeypatch):
    calls: list[str] = []
    monkeypatch.setitem(cli.STAGES, "clean", lambda: calls.append("clean"))
    assert cli.main(["clean"]) == 0
    assert calls == ["clean"]


def test_download_accepts_force(monkeypatch):
    received: list[bool] = []
    monkeypatch.setitem(cli.STAGES, "download", lambda force=False: received.append(force))
    assert cli.main(["download", "--force"]) == 0
    assert received == [True]


def test_stage_failure_returns_non_zero(monkeypatch):
    def explode() -> None:
        raise RuntimeError("no data")

    monkeypatch.setitem(cli.STAGES, "eda", explode)
    assert cli.main(["eda"]) == 1


def test_unknown_stage_is_rejected(capsys):
    with pytest.raises(SystemExit):
        cli.main(["frobnicate"])
    assert "invalid choice" in capsys.readouterr().err
