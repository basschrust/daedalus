from datetime import date

import pandas as pd
import pytest

from marketsim.scenarios import wig20
from marketsim.scenarios.wig20 import (
    build_markets,
    fetch_index_close,
)


def test_builds_twenty_stocks_then_index():
    symbols = [f"S{i}.WA" for i in range(20)]

    members = pd.DataFrame(
        {
            "symbol": symbols,
            "weight": [0.05] * 20,
        }
    )

    history = {symbol: pd.DataFrame({"close": [100.0]}) for symbol in symbols}

    markets = build_markets(members, history, 3000.0)

    assert len(markets) == 21
    assert list(markets)[-1] == "wig20"

    assert markets["wig20"]["components"][0]["base_price"] == 100.0

    assert markets["wig20"]["base_index_level"] == 3000.0


def test_invalid_index_level():
    with pytest.raises(ValueError):
        build_markets(pd.DataFrame(), {}, 0)


def test_index_close_uses_index_prices(monkeypatch):
    observed = {}

    def fake_query(conn, sql, params):
        observed["sql"] = sql
        observed["params"] = params

        return pd.DataFrame({"index_close": [3125.5]})

    monkeypatch.setattr(wig20, "query", fake_query)

    result = fetch_index_close(None, date(2026, 10, 28))

    assert result == 3125.5
    assert "INDEX_PRICES" in observed["sql"]
    assert observed["params"] == ("WIG20", date(2026, 10, 28))


def test_index_close_missing_fails(monkeypatch):
    monkeypatch.setattr(
        wig20,
        "query",
        lambda *args: pd.DataFrame(columns=["index_close"]),
    )

    with pytest.raises(
        ValueError,
        match="INDEX_PRICES",
    ):
        fetch_index_close(None, date(2026, 10, 28))
