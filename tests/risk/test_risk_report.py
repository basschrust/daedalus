"""Offline risk reporting, cross-agent aggregation, Greeks and Kupiec regression tests."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from marketsim.database.risk_reporting_queries import (
    load_derivative_exposure,
    load_wig20_greeks,
)
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


def test_postrun_generates_charts_greeks_and_csv_without_touching_database(tmp_path, monkeypatch):
    risk, equity = fixtures()
    monkeypatch.setattr("run.risk_report.load_risk_report_data", lambda unused: (risk, equity))
    monkeypatch.setattr(
        "run.risk_report.load_derivative_exposure",
        lambda unused: pd.DataFrame(
            [
                dict(
                    agent_id=10,
                    options_last_eod=1,
                    futures_last_eod=0,
                    options_any_eod=2,
                    futures_any_eod=1,
                    last_eod_day=4,
                ),
                dict(
                    agent_id=11,
                    options_last_eod=0,
                    futures_last_eod=0,
                    options_any_eod=0,
                    futures_any_eod=0,
                    last_eod_day=4,
                ),
                dict(
                    agent_id=12,
                    options_last_eod=1,
                    futures_last_eod=1,
                    options_any_eod=1,
                    futures_any_eod=1,
                    last_eod_day=4,
                ),
            ]
        ),
    )
    monkeypatch.setattr(
        "run.risk_report.load_wig20_greeks",
        lambda unused: pd.DataFrame(
            [
                dict(
                    day=0,
                    agent_id=10,
                    underlying_symbol="WIG20",
                    delta=1.2,
                    gamma=0.2,
                    group_name="MM",
                    name="A10",
                ),
                dict(
                    day=1,
                    agent_id=10,
                    underlying_symbol="WIG20",
                    delta=1.0,
                    gamma=0.1,
                    group_name="MM",
                    name="A10",
                ),
                dict(
                    day=0,
                    agent_id=12,
                    underlying_symbol="WIG20",
                    delta=-2.0,
                    gamma=-0.2,
                    group_name="Momentum",
                    name="A12",
                ),
            ]
        ),
    )
    outputs = generate_report(db_path=Path("unused.duckdb"), output_dir=tmp_path)
    assert len(outputs) == 7  # top three agents, group, ranking, 2 Greeks
    assert all(path.is_file() and path.stat().st_size > 500 for path in outputs)
    for name in (
        "var_agents_summary.csv",
        "var_groups_summary.csv",
        "kupiec_observations_apl.csv",
        "kupiec_1d_apl.csv",
        "greeks_wig20.csv",
    ):
        assert (tmp_path / name).exists()
    kupiec = pd.read_csv(tmp_path / "kupiec_1d_apl.csv")
    assert (kupiec.observations == 4).all()
    summary = pd.read_csv(tmp_path / "var_agents_summary.csv")
    row = summary.loc[(summary.agent_id == 10) & (summary.method == "montecarlo")].iloc[0]
    assert row.options_last_eod == 1
    assert row.options_any_eod == 2
    assert row.futures_any_eod == 1
    assert row.last_eod_day == 4
    greeks = pd.read_csv(tmp_path / "greeks_wig20.csv")
    assert set(greeks.agent_id) == {10, 12}


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


def test_agent_ranking_uses_full_period_median_not_last_day():
    risk, _ = fixtures()
    selected = select_var(add_var_percent(risk), horizon=1, confidence=0.99)
    summary = agent_summary(selected)
    item = summary.loc[(summary.agent_id == 10) & (summary.method == "montecarlo")].iloc[0]
    observed = selected.loc[
        (selected.agent_id == 10) & (selected.method == "montecarlo")
    ].sort_values("day")
    assert item.days == 5
    assert item.median_var_pct == pytest.approx(observed.var_pct_equity.median())
    assert item.median_var_pct != pytest.approx(observed.var_pct_equity.iloc[-1])


def test_derivative_exposure_reads_option_and_futures_eod_positions(tmp_path):
    db = tmp_path / "daedalus.duckdb"
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE agents (agent_id INTEGER)")
        conn.execute("INSERT INTO agents VALUES (10), (11), (12), (13)")
        conn.execute("CREATE TABLE securities (asset_id INTEGER, instrument_class VARCHAR)")
        conn.execute("INSERT INTO securities VALUES (1, 'stock'), (2, 'option'), (3, 'option')")
        conn.execute("CREATE TABLE futures (contract_id INTEGER)")
        conn.execute("INSERT INTO futures VALUES (9)")
        conn.execute(
            "CREATE TABLE eod_agent_positions "
            "(day INTEGER, agent_id INTEGER, asset_id INTEGER, position INTEGER)"
        )
        conn.execute(
            "INSERT INTO eod_agent_positions VALUES "
            "(0, 10, 2, 3), (0, 10, 3, -2), (1, 10, 1, 5), "
            "(2, 10, 2, 0), (2, 11, 2, -4), (2, 12, 1, 15)"
        )
        conn.execute(
            "CREATE TABLE eod_agent_futures_position "
            "(day INTEGER, agent_id INTEGER, asset_id INTEGER, position INTEGER)"
        )
        conn.execute(
            "INSERT INTO eod_agent_futures_position VALUES "
            "(0, 10, 9, 7), (2, 10, 9, 0), (2, 12, 9, -2)"
        )
        conn.execute(
            "CREATE TABLE eod_agent_portfolio_value "
            "(day INTEGER, agent_id INTEGER, portfolio_value DOUBLE)"
        )
        conn.execute(
            "INSERT INTO eod_agent_portfolio_value VALUES "
            "(0, 10, 1000), (1, 10, 999), (2, 10, 1001), "
            "(2, 11, 950), (2, 12, 1000), (2, 13, 1000)"
        )
    positions = load_derivative_exposure(db).set_index("agent_id")
    assert len(positions) == 4
    assert set(positions.last_eod_day) == {2}
    assert positions.loc[10, "options_any_eod"] == 2
    assert positions.loc[10, "futures_any_eod"] == 1
    assert positions.loc[10, "options_last_eod"] == 0
    assert positions.loc[10, "futures_last_eod"] == 0
    assert positions.loc[11, "options_last_eod"] == 1
    assert positions.loc[12, "futures_last_eod"] == 1
    assert positions.loc[13, ["options_any_eod", "futures_any_eod"]].sum() == 0


def test_derivative_exposure_errors_on_missing_tables(tmp_path):
    db = tmp_path / "daedalus.duckdb"
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE agents(agent_id INTEGER)")
    with pytest.raises(ValueError, match="derivative-position tables"):
        load_derivative_exposure(db)


def test_missing_derivative_inventory_is_not_silently_labelled_flat(tmp_path, monkeypatch):
    risk, equity = fixtures()
    monkeypatch.setattr("run.risk_report.load_risk_report_data", lambda unused: (risk, equity))
    monkeypatch.setattr(
        "run.risk_report.load_derivative_exposure",
        lambda unused: pd.DataFrame(
            [
                dict(
                    agent_id=10,
                    options_last_eod=0,
                    futures_last_eod=0,
                    options_any_eod=0,
                    futures_any_eod=0,
                    last_eod_day=4,
                )
            ]
        ),
    )
    monkeypatch.setattr(
        "run.risk_report.load_wig20_greeks",
        lambda unused: pd.DataFrame(
            columns=["day", "agent_id", "underlying_symbol", "delta", "gamma"]
        ),
    )
    with pytest.raises(ValueError, match="Derivative inventory unavailable"):
        generate_report(db_path=Path("unused.duckdb"), output_dir=tmp_path)


def test_wig20_greeks_uses_actual_agent_option_tables(tmp_path):
    db = tmp_path / "daedalus.duckdb"
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE agents (agent_id INTEGER, group_name VARCHAR, name VARCHAR)")
        conn.execute("INSERT INTO agents VALUES (10, 'MM', 'Agent 10'), (11, 'Noise', 'Agent 11')")
        conn.execute("CREATE TABLE indices (index_id INTEGER, name VARCHAR, short_name VARCHAR)")
        conn.execute(
            "INSERT INTO indices VALUES (7, 'WIG Twenty', 'WIG20'), (8, 'Other', 'mWIG40')"
        )
        conn.execute(
            "CREATE TABLE eod_agent_option_greeks ("
            "day INTEGER, agent_id INTEGER, asset_id INTEGER, underlying_id INTEGER, "
            "position INTEGER, delta DOUBLE, delta_equivalent DOUBLE, multiplier DOUBLE, "
            "calibration_id VARCHAR, gamma_equivalent DOUBLE)"
        )
        conn.execute(
            "INSERT INTO eod_agent_option_greeks VALUES "
            "(0, 10, 20, 7, 2, 0.4, 8, 10, 'c0', 0.3), "
            "(0, 10, 21, 7, -1, -0.2, 2, 10, 'c0', -0.1), "
            "(1, 11, 30, 8, 3, 0.9, 27, 10, 'c1', 0.8)"
        )
        conn.execute(
            "CREATE TABLE eod_option_greeks ("
            "day INTEGER, asset_id INTEGER, underlying_id INTEGER, "
            "gamma DOUBLE, multiplier DOUBLE, calibration_id VARCHAR)"
        )
        conn.execute(
            "INSERT INTO eod_option_greeks VALUES "
            "(0, 20, 7, 0.015, 10, 'c0'), "
            "(0, 21, 7, 0.01, 10, 'c0'), "
            "(1, 20, 7, 0.014, 10, 'c1'), "
            "(1, 21, 7, 0.01, 10, 'c1'), "
            "(1, 30, 8, 0.01, 10, 'c1')"
        )
    data = load_wig20_greeks(db)
    assert list(data.columns) == [
        "day",
        "agent_id",
        "underlying_symbol",
        "delta",
        "gamma",
        "group_name",
        "name",
    ]
    assert set(data.underlying_symbol) == {"WIG20"}
    agent_10 = data.loc[data.agent_id == 10].set_index("day")
    assert agent_10.loc[0, "delta"] == pytest.approx(10)
    assert agent_10.loc[0, "gamma"] == pytest.approx(0.2)
    # No WIG20 options at day 1, but the market Greeks were recorded.
    assert agent_10.loc[1, "delta"] == pytest.approx(0)
    assert agent_10.loc[1, "gamma"] == pytest.approx(0)
    agent_11 = data.loc[data.agent_id == 11].set_index("day")
    assert agent_11.loc[1, "delta"] == pytest.approx(0)
    assert agent_11.loc[1, "gamma"] == pytest.approx(0)


def test_older_schema_can_recover_gamma_from_per_option_table(tmp_path):
    db = tmp_path / "daedalus.duckdb"
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE agents (agent_id INTEGER, group_name VARCHAR, name VARCHAR)")
        conn.execute("INSERT INTO agents VALUES (10, 'MM', 'A10')")
        conn.execute("CREATE TABLE indices (index_id INTEGER, name VARCHAR, short_name VARCHAR)")
        conn.execute("INSERT INTO indices VALUES (7, 'WIG20', 'WIG20')")
        conn.execute(
            "CREATE TABLE eod_agent_option_greeks ("
            "day INTEGER, agent_id INTEGER, asset_id INTEGER, underlying_id INTEGER, "
            "position INTEGER, delta DOUBLE, delta_equivalent DOUBLE, multiplier DOUBLE)"
        )
        conn.execute("INSERT INTO eod_agent_option_greeks VALUES (0, 10, 20, 7, -3, 0.4, -12, 10)")
        conn.execute(
            "CREATE TABLE eod_option_greeks ("
            "day INTEGER, asset_id INTEGER, underlying_id INTEGER, "
            "gamma DOUBLE, multiplier DOUBLE)"
        )
        conn.execute("INSERT INTO eod_option_greeks VALUES (0, 20, 7, 0.025, 10)")
    data = load_wig20_greeks(db)
    assert data.loc[0, "delta"] == pytest.approx(-12)
    assert data.loc[0, "gamma"] == pytest.approx(-0.75)


def test_missing_greeks_table_does_not_block_var_reporting(tmp_path):
    db = tmp_path / "daedalus.duckdb"
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE agents(agent_id INTEGER)")
    result = load_wig20_greeks(db)
    assert result.empty
    assert "eod_agent_option_greeks" in result.attrs["diagnostic"]


def test_report_keeps_var_plots_when_greeks_were_disabled(tmp_path, monkeypatch, capsys):
    risk, equity = fixtures()
    monkeypatch.setattr("run.risk_report.load_risk_report_data", lambda unused: (risk, equity))
    monkeypatch.setattr(
        "run.risk_report.load_derivative_exposure",
        lambda unused: pd.DataFrame(
            [
                dict(
                    agent_id=agent_id,
                    options_last_eod=int(agent_id == 10),
                    futures_last_eod=0,
                    options_any_eod=int(agent_id == 10),
                    futures_any_eod=0,
                    last_eod_day=4,
                )
                for agent_id in (10, 11, 12)
            ]
        ),
    )

    def no_greeks(unused):
        result = pd.DataFrame(columns=["day", "agent_id", "underlying_symbol", "delta", "gamma"])
        result.attrs["diagnostic"] = "EoD Greeks disabled; no recorded option sensitivities"
        return result

    monkeypatch.setattr("run.risk_report.load_wig20_greeks", no_greeks)
    paths = generate_report(db_path=Path("unused.duckdb"), output_dir=tmp_path)
    assert len(paths) == 5  # Existing VaR plots; no Greeks can be fabricated.
    assert (tmp_path / "greeks_wig20.csv").exists()
    assert "EoD Greeks disabled" in capsys.readouterr().out
