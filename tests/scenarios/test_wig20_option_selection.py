"""Prevent sparse WIG20 option histories or expiry filtering from removing the market."""

from datetime import date

import pandas as pd
import pytest

from marketsim.database.ikaros import ikaros_queries
from marketsim.scenarios import wig20_options

CUTOFF = date(2026, 11, 20)
DATES = [
    date(2026, 11, 23),
    date(2026, 11, 24),
    date(2026, 11, 25),
    date(2026, 11, 26),
    date(2026, 11, 27),
]


def _history(last_day="2026-11-19"):
    dates = pd.bdate_range(end=last_day, periods=5)
    return pd.DataFrame(
        {
            "date": dates,
            "open": [15] * 5,
            "high": [16] * 5,
            "low": [14] * 5,
            "close": [15] * 5,
            "volume": [2] * 5,
        }
    )


def _contracts():
    return pd.DataFrame(
        [
            {
                "contract_id": 1,
                "symbol": "SHORT_CALL",
                "strike": "4200",
                "option_side": "CALL",
                "expiry_date": date(2026, 11, 24),
                "multiplier": 10,
            },
            {
                "contract_id": 2,
                "symbol": "LONG_CALL",
                "strike": "4225",
                "option_side": "CALL",
                "expiry_date": date(2026, 12, 18),
                "multiplier": 10,
            },
            {
                "contract_id": 3,
                "symbol": "SHORT_PUT",
                "strike": "4250",
                "option_side": "PUT",
                "expiry_date": date(2026, 11, 25),
                "multiplier": 10,
            },
        ]
    )


def test_prefers_full_horizon_but_keeps_shorter_contracts(monkeypatch):
    monkeypatch.setattr(
        ikaros_queries,
        "fetch_wig20_option_contracts",
        lambda *_: _contracts(),
    )
    monkeypatch.setattr(wig20_options, "futures_history", lambda *_: _history())
    chosen = wig20_options.select_options(
        object(), CUTOFF, DATES, "4225", count=3, lookback_sessions=252
    )
    assert [c["symbol"] for c in chosen] == ["LONG_CALL", "SHORT_PUT", "SHORT_CALL"]
    assert chosen[0]["history"]["date"].max().date() == date(2026, 11, 19)


def test_no_eligible_options_is_an_explicit_error(monkeypatch):
    monkeypatch.setattr(
        ikaros_queries,
        "fetch_wig20_option_contracts",
        lambda *_: pd.DataFrame(),
    )
    with pytest.raises(ValueError, match="no WIG20 option contracts"):
        wig20_options.select_options(object(), CUTOFF, DATES, "4225", count=3)


def test_zero_requested_options_stays_supported(monkeypatch):
    monkeypatch.setattr(
        ikaros_queries,
        "fetch_wig20_option_contracts",
        lambda *_: pytest.fail("Must not fetch if count=0"),
    )
    assert wig20_options.select_options(object(), CUTOFF, DATES, "4225", count=0) == []
