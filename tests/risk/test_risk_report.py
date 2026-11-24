"""Offline risk reporting, cross-agent aggregation and Kupiec regression tests."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from marketsim.plot.risk_report import add_var_percent, agent_summary, group_summary, select_var
from marketsim.risk.var_backtesting import kupiec_report, kupiec_uc, one_day_apl
from run.risk_report import generate_report, latest_run_db


def fixtures() -> tuple[pd.DataFrame, pd.DataFrame]:
    risks = []
    equity = []
    for agent_id, group, scale in [(10, "MM", 1.0), (11, "MM", 2.0), (12, "Momentum", 3.0)]:
        for day in range(5):
            equity.append(dict(day=day, agent_id=agent_id, portfolio_value=1000 - 10 * day))
            for method, multiplier in [
                ("parametric", 0.9),
                ("historical", 1.1),
                ("montecarlo", 1.0),
            ]:
                for level in (0.95, 0.99):
                    for horizon in (1, 5):
                        risks.append(
                            dict(
                                method=method,
                                day=day,
                                agent_id=agent_id,
                                group_name=group,
                                confidence_level=level,
                                horizon_days=horizon,
                                var_amount=scale * multiplier * 20,
                                portfolio_value=1000 - 10 * day,
                            )
                        )
    return pd.DataFrame(risks), pd.DataFrame(equity)


def test_kupiec_zero_and_all_exceptions_are_finite():
    for x in (0, 100):
        stat, p = kupiec_uc(n=100, exceptions=x, confidence=0.99)
        assert np.isfinite(stat) and stat >= 0
        assert 0 <= p <= 1
    statistic, p = kupiec_uc(n=100, exceptions=1, confidence=0.99)
    assert statistic == pytest.approx(0)
    assert p == pytest.approx(1)


def test_kupiec_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        kupiec_uc(n=0, exceptions=0, confidence=0.99)
    with pytest.raises(ValueError):
        kupiec_uc(n=10, exceptions=11, confidence=0.99)
    with pytest.raises(ValueError):
        kupiec_uc(n=10, exceptions=0, confidence=1.0)


def test_one_day_apl_is_forward_shifted_and_excludes_nonconsecutive_gaps():
    risk, equity = fixtures()
    results = one_day_apl(risk, equity)
    assert set(results.day) == {0, 1, 2, 3}
    assert set(results.horizon_days) == {1}
    assert (results.actual_pnl == -10).all()
    assert (results.loss == 10).all()
    # Missing t+1 is never silently bridged to t+2.
    incomplete = equity.loc[equity.day != 2].copy()
    results = one_day_apl(risk, incomplete)
    assert set(results.day) == {0, 3}


def test_agent_group_statistics_use_agents_not_daily_observations():
    risk, _ = fixtures()
    selected = select_var(add_var_percent(risk), horizon=1, confidence=0.99)
    agents = agent_summary(selected)
    groups = group_summary(agents)
    mm = groups.loc[(groups.method == "montecarlo") & (groups.group_name == "MM")].iloc[0]
    assert mm.agents == 2
    assert len(agents.loc[agents.method == "montecarlo"]) == 3
    assert mm.p10_var_pct <= mm.median_var_pct <= mm.p90_var_pct


def test_nonpositive_equity_is_excluded_from_percent_not_reported_as_zero():
    risk, _ = fixtures()
    risk.loc[risk.agent_id == 10, "portfolio_value"] = 0
    normalized = add_var_percent(risk)
    assert normalized.loc[normalized.agent_id == 10, "var_pct_equity"].isna().all()
    assert normalized.loc[normalized.agent_id == 11, "var_pct_equity"].notna().all()


def test_short_kupiec_samples_warn_instead_of_claiming_validation():
    risk, equity = fixtures()
    report = kupiec_report(one_day_apl(risk, equity))
    assert (report.observations == 4).all()
    assert (report.assessment == "LOW_POWER").all()
    assert set(report.confidence_level) == {0.95, 0.99}
    assert (report.exceptions == 0).all()


def test_postrun_generates_charts_and_csv_without_touching_database(tmp_path, monkeypatch):
    risk, equity = fixtures()
    monkeypatch.setattr("run.risk_report.load_risk_report_data", lambda unused: (risk, equity))
    outputs = generate_report(db_path=Path("unused.duckdb"), output_dir=tmp_path)
    assert len(outputs) == 5  # top three agents, group comparison and ranking
    assert all(path.is_file() and path.stat().st_size > 500 for path in outputs)
    for name in (
        "var_agents_summary.csv",
        "var_groups_summary.csv",
        "kupiec_observations_apl.csv",
        "kupiec_1d_apl.csv",
    ):
        assert (tmp_path / name).exists()
    kupiec = pd.read_csv(tmp_path / "kupiec_1d_apl.csv")
    assert (kupiec.observations == 4).all()


def test_latest_db_scans_run_directories_only(tmp_path):
    for run in ("run_20261120_010101", "run_20261124_020202"):
        path = tmp_path / "output" / run / "daedalus.duckdb"
        path.parent.mkdir(parents=True)
        path.touch()
    assert "20261124" in str(latest_run_db(root=tmp_path))


def test_cli_reports_locked_duckdb_without_traceback(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "daedalus.duckdb"
    db_path.touch()

    def locked(_db_path):
        raise duckdb.IOException(
            'IO Error: Cannot open file "daedalus.duckdb": The process cannot '
            "access the file because it is being used by another process. "
            "File is already open in DBeaver\\dbeaver.exe (PID 30840)"
        )

    monkeypatch.setattr("run.risk_report.load_risk_report_data", locked)
    from run.risk_report import main

    with pytest.raises(SystemExit) as result:
        main(["--db", str(db_path)])
    assert result.value.code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "simulation database is locked" in captured.err
    assert str(db_path) in captured.err
    assert "DBeaver" in captured.err
    assert "No simulation rerun is necessary" in captured.err


def test_cli_unrelated_duckdb_io_error_is_not_called_lock(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "daedalus.duckdb"
    db_path.touch()

    def io_failure(_db_path):
        raise duckdb.IOException("IO Error: corrupted file")

    monkeypatch.setattr("run.risk_report.load_risk_report_data", io_failure)
    from run.risk_report import main

    with pytest.raises(SystemExit) as result:
        main(["--db", str(db_path)])
    assert result.value.code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "DuckDB I/O error" in captured.err
    assert "corrupted file" in captured.err
    assert "locked" not in captured.err
