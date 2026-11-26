"""Risk reporting respects the configuration stored with each completed run."""

from pathlib import Path

import pandas as pd
import pytest

from run import risk_report


def _saved_run(tmp_path: Path, *, var: bool, greeks: bool) -> Path:
    db = tmp_path / "daedalus.duckdb"
    db.touch()
    (tmp_path / "market_structure.yaml").write_text(
        f"risk_var:\n  enabled: {str(var).lower()}\n"
        f"risk_greeks:\n  enabled: {str(greeks).lower()}\n",
        encoding="utf-8",
    )
    return db


def test_synthetic_scenario_without_risk_exits_successfully(tmp_path, monkeypatch, capsys):
    db = tmp_path / "daedalus.duckdb"
    db.touch()
    # The default market_structure.yaml has no risk_var or risk_greeks sections.
    (tmp_path / "market_structure.yaml").write_text("days: 42\nmarkets: {}\n")

    def no_db_reads(_db):
        pytest.fail("Risk-disabled scenarios must not query DuckDB")

    monkeypatch.setattr(risk_report, "load_risk_report_data", no_db_reads)
    assert risk_report.main(["--db", str(db)]) == 0
    message = capsys.readouterr().out
    assert "VaR: not enabled" in message
    assert "Greeks: not enabled" in message
    assert "nothing to generate" in message
    assert not (tmp_path / "risk_report").exists()


def test_enabled_var_with_missing_observations_is_not_silently_skipped(tmp_path, monkeypatch):
    db = _saved_run(tmp_path, var=True, greeks=False)

    def missing_var(_db):
        raise ValueError("No EoD VaR observations in this run")

    monkeypatch.setattr(risk_report, "load_risk_report_data", missing_var)
    with pytest.raises(ValueError, match="No EoD VaR observations"):
        risk_report.generate_report(db_path=db, output_dir=tmp_path / "risk_report")


def test_greeks_only_can_generate_charts_without_var(tmp_path, monkeypatch, capsys):
    db = _saved_run(tmp_path, var=False, greeks=True)
    values = pd.DataFrame(
        [
            {"day": 0, "agent_id": 10, "group_name": "OptionMM", "delta": -7.0, "gamma": 2.0},
            {"day": 0, "agent_id": 11, "group_name": "Noise", "delta": 5.0, "gamma": 0.1},
        ]
    )
    positions = pd.DataFrame({"agent_id": [10, 11], "options_any_eod": [1, 0]})
    monkeypatch.setattr(
        risk_report,
        "load_risk_report_data",
        lambda *_: pytest.fail("VaR loader must not run for Greeks-only report"),
    )
    monkeypatch.setattr(risk_report, "load_wig20_greeks", lambda *_: values)
    monkeypatch.setattr(risk_report, "load_derivative_exposure", lambda *_: positions)

    def make_plots(greeks, *, out, agent_ids):
        assert agent_ids == [10]
        assert set(greeks.agent_id) == {10}
        path = out / "greeks_wig20_optionmm_agent_10.png"
        path.write_bytes(b"mock chart")
        return [path]

    monkeypatch.setattr(risk_report, "greeks_evolution_plots", make_plots)
    output = tmp_path / "risk_report"
    results = risk_report.generate_report(db_path=db, output_dir=output)
    assert len(results) == 1 and results[0].is_file()
    assert set(pd.read_csv(output / "greeks_wig20.csv").agent_id) == {10}
    assert "VaR: not enabled" in capsys.readouterr().out
    assert not (output / "kupiec_1d_apl.csv").exists()


def test_no_greeks_observations_skips_gracefully(tmp_path, monkeypatch, capsys):
    db = _saved_run(tmp_path, var=False, greeks=True)
    values = pd.DataFrame(columns=["day", "agent_id", "delta", "gamma"])
    values.attrs["diagnostic"] = "No agent option Greeks were recorded"
    monkeypatch.setattr(risk_report, "load_wig20_greeks", lambda *_: values)
    results = risk_report.generate_report(db_path=db, output_dir=tmp_path / "risk_report")
    assert results == []
    assert "No agent option Greeks were recorded" in capsys.readouterr().out
    assert not (tmp_path / "risk_report").exists()


def test_missing_run_snapshot_preserves_existing_diagnostics(tmp_path, monkeypatch):
    db = tmp_path / "daedalus.duckdb"
    db.touch()

    def missing_var(_db):
        raise ValueError("No EoD VaR observations in this run")

    monkeypatch.setattr(risk_report, "load_risk_report_data", missing_var)
    with pytest.raises(ValueError, match="No EoD VaR observations"):
        risk_report.generate_report(db_path=db, output_dir=tmp_path / "risk_report")


def test_invalid_saved_config_fails_explicitly(tmp_path):
    db = tmp_path / "daedalus.duckdb"
    db.touch()
    (tmp_path / "market_structure.yaml").write_text("- not\n- a\n- config\n")
    with pytest.raises(ValueError, match="Invalid saved run configuration"):
        risk_report.generate_report(db_path=db, output_dir=tmp_path / "risk_report")
