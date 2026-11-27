"""Offline WIG20 EoD Greek plotting after a local simulation has closed."""

import pandas as pd

from run.simulation_main import plot_wig20_greeks_after_run


def test_skips_when_there_is_no_local_duckdb(tmp_path, capsys):
    assert plot_wig20_greeks_after_run(output_dir=tmp_path) == []
    assert "no local DuckDB" in capsys.readouterr().out


def test_generates_wig20_plot_and_csv_after_simulation(tmp_path, monkeypatch):
    (tmp_path / "daedalus.duckdb").touch()
    greeks = pd.DataFrame(
        [
            {
                "day": 0,
                "agent_id": 9,
                "underlying_symbol": "WIG20",
                "delta": -17.5,
                "gamma": 2.4,
                "group_name": "option_market_makers",
                "name": "maker 9",
            }
        ]
    )
    monkeypatch.setattr(
        "run.simulation_main.load_wig20_greeks",
        lambda unused: greeks,
    )
    observed = {}

    def fake_plots(values, *, out, agent_ids):
        observed["ids"] = agent_ids
        observed["dir"] = out
        return [out / "greeks_wig20_option_market_makers_agent_9.png"]

    monkeypatch.setattr("run.simulation_main.greeks_evolution_plots", fake_plots)
    images = plot_wig20_greeks_after_run(output_dir=tmp_path)
    assert len(images) == 1
    assert observed["ids"] == [9]
    assert observed["dir"] == tmp_path / "risk_report"
    written = pd.read_csv(tmp_path / "risk_report" / "greeks_wig20.csv")
    assert written.loc[0, "gamma"] == 2.4
