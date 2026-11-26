"""A generated WIG20 scenario should work without handwritten preparation commands."""

from pathlib import Path

import duckdb
import pandas as pd
import pytest
import yaml

from marketsim.database.prepared_snapshot_restore import restore_historical_csv
from run import simulation_main as cli


def _scenario() -> dict:
    return {
        "markets": {
            "ale.wa": {"symbol": "ALE.WA", "instrument_class": "stock"},
            "wig20": {"symbol": "WIG20", "instrument_class": "index"},
        },
        "historical_prices": {
            "enabled": True,
            "path": "market_data/wig20_auto.ohlcv.csv",
            "cutoff_date": "2026-11-18",
            "lookback_sessions": 252,
        },
        "start_date": "2026-11-19",
        "risk_var": {"enabled": True},
    }


def _cached_history(root: Path, *, scenario: dict) -> Path:
    cache = root / "input" / "market_data" / "market_cache.duckdb"
    cache.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(cache)) as conn:
        conn.execute(
            "CREATE TABLE prepared_snapshots ("
            "dataset VARCHAR, cutoff_date DATE, scenario_yaml VARCHAR, rows_count BIGINT)"
        )
        conn.execute(
            "CREATE TABLE historical_ohlcv ("
            "dataset VARCHAR, symbol VARCHAR, session_date DATE, open VARCHAR, "
            "high VARCHAR, low VARCHAR, close VARCHAR, volume VARCHAR)"
        )
        conn.execute(
            "INSERT INTO prepared_snapshots VALUES (?, ?, ?, ?)",
            ["wig20", "2026-11-18", yaml.safe_dump(scenario), 2],
        )
        conn.execute(
            "INSERT INTO historical_ohlcv VALUES "
            "('wig20','ALE.WA','2026-11-18','67.95','68.01','67.90','68.00','100000'),"
            "('wig20','WIG20','2026-11-18','4200','4230','4195','4226.28','1000000')"
        )
    return cache


def test_restores_exact_older_market_snapshot_without_fabrication(tmp_path):
    scenario = _scenario()
    cache = _cached_history(tmp_path, scenario=scenario)
    output = tmp_path / "input" / "market_data" / "wig20_auto.ohlcv.csv"
    assert restore_historical_csv(
        cache_path=cache, output_path=output, dataset="wig20", scenario=scenario
    )
    data = pd.read_csv(output)
    assert set(data.symbol) == {"ALE.WA", "WIG20"}
    assert data.loc[data.symbol == "WIG20", "close"].iloc[0] == pytest.approx(4226.28)
    assert data.loc[data.symbol == "ALE.WA", "volume"].iloc[0] == 100000
    # The cache is immutable: it can restore a dated scenario even weeks later.
    with duckdb.connect(str(cache), read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM historical_ohlcv").fetchone()[0] == 2


def test_never_mix_market_universes_from_different_yaml(tmp_path):
    scenario = _scenario()
    cache = _cached_history(tmp_path, scenario=scenario)
    changed = _scenario()
    changed["markets"]["ale.wa"]["symbol"] = "PZU.WA"
    output = tmp_path / "input" / "market_data" / "wig20_auto.ohlcv.csv"
    assert not restore_historical_csv(
        cache_path=cache, output_path=output, dataset="wig20", scenario=changed
    )
    assert not output.exists()


def test_existing_history_is_used_without_preparation(tmp_path, monkeypatch):
    data_dir = tmp_path / "input" / "market_data"
    data_dir.mkdir(parents=True)
    (data_dir / "wig20_auto.ohlcv.csv").write_text("symbol,date\n", encoding="utf-8")
    monkeypatch.setattr(cli, "INPUT_DIR", tmp_path / "input")
    monkeypatch.setattr(cli, "prepare_dataset", lambda *a, **k: pytest.fail("No fetch needed"))
    assert not cli.ensure_historical_input(config_file="wig20_auto.yaml", scenario=_scenario())


def test_missing_history_uses_matching_cache(tmp_path, monkeypatch, capsys):
    scenario = _scenario()
    _cached_history(tmp_path, scenario=scenario)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "INPUT_DIR", tmp_path / "input")
    monkeypatch.setattr(cli, "prepare_dataset", lambda *a, **k: pytest.fail("No fetch needed"))
    assert not cli.ensure_historical_input(config_file="wig20_auto.yaml", scenario=scenario)
    assert (tmp_path / "input" / "market_data" / "wig20_auto.ohlcv.csv").is_file()
    assert "Restored matching WIG20 history" in capsys.readouterr().out


def test_local_ikaros_is_automatically_used_if_cache_missing(tmp_path, monkeypatch):
    scenario = _scenario()
    source = tmp_path / "ikaros.duckdb"
    source.touch()
    output = tmp_path / "input" / "market_data" / "wig20_auto.ohlcv.csv"
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "INPUT_DIR", tmp_path / "input")
    monkeypatch.setattr(cli, "_local_ikaros_sources", lambda: [source])
    monkeypatch.setattr(cli, "_has_snowflake_settings", lambda: False)
    calls = []

    def prepare(name, *, source: str, ikaros_db: Path):
        calls.append((name, source, ikaros_db))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("symbol,date,open,high,low,close,volume\n", encoding="utf-8")

    monkeypatch.setattr(cli, "prepare_dataset", prepare)
    assert cli.ensure_historical_input(config_file="wig20_auto.yaml", scenario=scenario)
    assert calls == [("wig20", "duckdb", source)]


def test_no_source_has_short_message_and_no_empty_run(tmp_path, monkeypatch, capsys):
    scenario = _scenario()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "INPUT_DIR", tmp_path / "input")
    monkeypatch.setattr(cli, "load_dotenv", lambda *args: None)
    monkeypatch.setattr(cli, "load_config", lambda *args: scenario)
    monkeypatch.setattr(cli, "_local_ikaros_sources", lambda: [])
    monkeypatch.setattr(cli, "_has_snowflake_settings", lambda: False)
    monkeypatch.setattr(
        cli, "create_run_directory", lambda: pytest.fail("Do not start an empty run")
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(["wig20_auto.yaml"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "A real WIG20 simulation needs historical Ikaros data" in captured.err
    assert "input/market_data/" in captured.err
    assert "No empty simulation was started" in captured.err
    assert not (tmp_path / "output").exists()


def test_non_wig20_missing_csv_has_plain_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "INPUT_DIR", tmp_path / "input")
    with pytest.raises(cli.ScenarioInputError, match="Required historical prices are missing"):
        cli.ensure_historical_input(config_file="my_scenario.yaml", scenario=_scenario())
