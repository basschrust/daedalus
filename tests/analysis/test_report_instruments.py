"""Regression tests for market realism coverage across instrument classes."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from marketsim.analysis.report import MarketRealismReport


def _history(n=100):
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(67).normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-01", periods=n),
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.arange(n) * 10 + 1000,
        }
    ).set_index("date")


def test_report_includes_futures_index_and_diagnoses_missing_options(tmp_path):
    history = _history()
    dates = pd.bdate_range("2026-08-01", periods=20)
    stock = SimpleNamespace(instrument_class="stock", symbol="AAA.WA", historical_ohlcv=history)
    future = SimpleNamespace(instrument_class="future", name="FW20Z26", historical_ohlcv=history)
    index = SimpleNamespace(instrument_class="index", name="WIG20", historical_ohlcv=history)
    option = SimpleNamespace(instrument_class="option", name="OW20X26", historical_ohlcv=None)
    markets = {i: m for i, m in enumerate((stock, future, index, option))}
    sim = SimpleNamespace(markets=markets, historical_ohlcv_by_symbol={})
    report = MarketRealismReport()
    observations = history.tail(20).reset_index()
    for market in (stock, future, index, option):
        report.observations[report._symbol(market)] = [
            {**r, "date": d}
            for r, d in zip(observations.drop(columns="date").to_dict("records"), dates)
        ]
    report.generate(sim, tmp_path)
    out = tmp_path / "market_realism"
    score = pd.read_csv(out / "market_realism_scorecard.csv")
    assert set(score.symbol) == {"AAA.WA", "FW20Z26", "WIG20"}
    assert pd.isna(score.loc[score.symbol == "WIG20", "volume_ratio"].iloc[0])
    assert score.loc[score.symbol == "FW20Z26", "volume_basis"].iloc[0] == "contracts"
    controls = pd.read_csv(out / "market_realism_historical_control.csv")
    assert set(controls.symbol) == {"AAA.WA", "FW20Z26", "WIG20"}
    diag = pd.read_csv(out / "market_realism_diagnostics.csv")
    assert diag.loc[diag.symbol == "OW20X26", "status"].iloc[0] == "missing_historical_data"
