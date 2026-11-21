"""Regressions for the per-dataset market-data cache."""

from datetime import date
from pathlib import Path

import duckdb
import yaml

from marketsim.scenarios.prepared_cache import (
    PreparedDatasetCache,
    required_cutoff,
    selection_signature,
)


def test_required_cutoff_weekend_and_exchange_holiday():
    assert required_cutoff(today=date(2026, 11, 23)) == date(2026, 11, 20)
    assert required_cutoff(today=date(2026, 11, 21)) == date(2026, 11, 20)
    assert required_cutoff(
        today=date(2026, 11, 12), non_trading_dates={date(2026, 11, 11)}
    ) == date(2026, 11, 10)


def test_agents_change_does_not_invalidate_market_data():
    settings = {"days": 15, "lookback_sessions": 252, "futures_count": 3, "options_count": 20}
    profile1 = {"agents": {"noise_wide": {"number": 8}}, "non_trading_dates": []}
    profile2 = {"agents": {"noise_wide": {"number": 12}}, "non_trading_dates": []}
    assert selection_signature("wig20", settings, profile1) == selection_signature(
        "wig20", settings, profile2
    )
    changed = {**settings, "options_count": 5}
    assert selection_signature("wig20", settings, profile1) != selection_signature(
        "wig20", changed, profile1
    )


def _create_existing_scenario(root: Path):
    input_dir = root / "input"
    data = input_dir / "market_data"
    data.mkdir(parents=True)
    (input_dir / "gpw_50d.yaml").write_text(
        yaml.safe_dump({"non_trading_dates": [], "agents": {"noise_wide": {"number": 8}}}),
        encoding="utf-8",
    )
    settings = {
        "days": 15,
        "sim_time": 25,
        "seed": 67,
        "lookback_sessions": 252,
        "futures_count": 1,
        "options_count": 1,
    }
    (input_dir / "wig20.yaml").write_text(yaml.safe_dump(settings), encoding="utf-8")
    scenario = {
        "days": 15,
        "sim_time": 25,
        "seed": 67,
        "markets": {
            "ale.wa": {"instrument_class": "stock"},
            "wig20": {"instrument_class": "index"},
            "future": {"instrument_class": "future"},
            "option": {"instrument_class": "option"},
        },
        "historical_prices": {
            "enabled": True,
            "path": "market_data/wig20_auto.ohlcv.csv",
            "cutoff_date": "2026-11-20",
            "lookback_sessions": 252,
        },
        "start_date": "2026-11-21",
        "agents": {"noise_wide": {"number": 8}},
    }
    (input_dir / "wig20_auto.yaml").write_text(yaml.safe_dump(scenario), encoding="utf-8")
    (data / "wig20_auto.ohlcv.csv").write_text(
        "symbol,date,open,high,low,close,volume\n"
        "ALE.WA,2026-11-20,10.02,10.03,10.01,10.02,100\n"
        "WIG20,2026-11-20,4000,4001,3999,4000,1000\n",
        encoding="utf-8",
    )
    return settings


def test_bootstrap_cache_without_snowflake_and_profile_update(tmp_path):
    settings = _create_existing_scenario(tmp_path)
    cache = PreparedDatasetCache(root=tmp_path)
    assert cache.use_if_fresh(
        dataset="wig20", settings=settings, yaml_name="wig20_auto.yaml", today=date(2026, 11, 21)
    )
    assert cache.db_path.is_file()
    with duckdb.connect(str(cache.db_path)) as con:
        assert con.execute("SELECT COUNT(*) FROM historical_ohlcv").fetchone()[0] == 2
    profile = tmp_path / "input" / "gpw_50d.yaml"
    profile.write_text(yaml.safe_dump({"agents": {"noise_wide": {"number": 12}}}), encoding="utf-8")
    assert cache.use_if_fresh(
        dataset="wig20", settings=settings, yaml_name="wig20_auto.yaml", today=date(2026, 11, 23)
    )
    scenario = yaml.safe_load((tmp_path / "input" / "wig20_auto.yaml").read_text())
    assert scenario["agents"]["noise_wide"]["number"] == 12
    assert not cache.use_if_fresh(
        dataset="wig20", settings=settings, yaml_name="wig20_auto.yaml", today=date(2026, 11, 24)
    )


def test_snapshot_restores_csv_and_yaml(tmp_path):
    settings = _create_existing_scenario(tmp_path)
    cache = PreparedDatasetCache(root=tmp_path)
    assert cache.use_if_fresh(
        dataset="wig20", settings=settings, yaml_name="wig20_auto.yaml", today=date(2026, 11, 21)
    )
    csv_path = tmp_path / "input" / "market_data" / "wig20_auto.ohlcv.csv"
    csv_path.unlink()
    (tmp_path / "input" / "wig20_auto.yaml").unlink()
    assert cache.use_if_fresh(
        dataset="wig20", settings=settings, yaml_name="wig20_auto.yaml", today=date(2026, 11, 21)
    )
    assert csv_path.exists()
    assert yaml.safe_load((tmp_path / "input" / "wig20_auto.yaml").read_text())["markets"]
