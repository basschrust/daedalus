"""Regression checks for distributed WIG20 futures liquidity configuration."""

from pathlib import Path

import yaml

PROFILE = Path(__file__).resolve().parents[2] / "marketsim/input/gpw_50d.yaml"


def test_no_dedicated_futures_liquidity_group():
    agents = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))["agents"]
    assert "futures_liquidity" not in agents


def test_existing_noise_groups_participate_in_futures():
    agents = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))["agents"]
    for name in ("noise_tight", "noise_medium", "noise_wide"):
        group = agents[name]
        assert group["agent_class"] == "NoiseAgent"
        assert {"stock", "future"} <= set(group["instrument_classes"])
        settings = group["config"]
        assert settings["future_mean_contracts"] > 1.5
        assert settings["future_max_order_contracts"] >= settings["future_mean_contracts"]
        assert settings["future_max_daily_contracts"] > 10
        assert settings["future_max_position_contracts"] > 20


def test_market_makers_provide_futures_liquidity():
    agents = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))["agents"]
    group = agents["market_makers"]
    assert group["agent_class"] == "MMZOHAgent"
    assert "future" in group["instrument_classes"]
    assert group["number"] >= 2
    assert 0 < group["config"]["volume_fraction"] <= 1
