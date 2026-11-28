"""The existing risk report command invokes historical VaR only when enabled."""

from pathlib import Path

from run import risk_report


def test_uses_saved_run_configuration_and_existing_report_dir(
    *, tmp_path: Path, monkeypatch
) -> None:
    db = tmp_path / "daedalus.duckdb"
    db.touch()
    (tmp_path / "market_structure.yaml").write_text(
        "risk_backtesting:\n  enabled: true\n  symbol: WIG20\n"
        "historical_prices:\n  enabled: true\n  path: market_data/example.csv\n",
        encoding="utf-8",
    )
    seen: dict = {}

    def fake_prices(*, config: dict, config_dir: Path) -> dict:
        seen["config_dir"] = config_dir
        return {"WIG20": "fake historical price frame"}

    def fake_backtest(**kwargs) -> None:
        seen.update(kwargs)

    monkeypatch.setattr(risk_report, "load_historical_prices", fake_prices)
    monkeypatch.setattr(risk_report, "write_post_run_backtests", fake_backtest)
    target = tmp_path / "custom_report"
    risk_report._historical_reporting_if_enabled(db_path=db, output_dir=target)
    assert seen["db_path"] == db
    assert seen["output_dir"] == target
    assert seen["settings"] == {"enabled": True, "symbol": "WIG20"}
    assert seen["historical"] == {"WIG20": "fake historical price frame"}


def test_disabled_backtest_does_not_read_market_history(*, tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "daedalus.duckdb"
    (tmp_path / "market_structure.yaml").write_text("days: 20\n", encoding="utf-8")

    def unexpected(**kwargs) -> None:
        raise AssertionError("Disabled historical backtest must not access history")

    monkeypatch.setattr(risk_report, "load_historical_prices", unexpected)
    risk_report._historical_reporting_if_enabled(db_path=db, output_dir=tmp_path / "risk_report")
