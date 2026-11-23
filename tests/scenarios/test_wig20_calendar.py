"""WIG20 historical-session alignment regressions (no external database)."""

from datetime import date

import pandas as pd
import pytest

from marketsim.scenarios import wig20
from marketsim.scenarios.wig20_data import align_stock_history, index_session_history

HOLIDAY = date(2026, 11, 11)
WINDOW = (date(2026, 11, 10), date(2026, 11, 12), date(2026, 11, 13))


def _frame(*values):
    return pd.DataFrame({"date": list(values), "close": [100] * len(values)})


def test_index_reference_excludes_holiday_and_weekends():
    result = index_session_history(
        _frame(date(2026, 11, 9), *WINDOW, HOLIDAY, date(2026, 11, 14)),
        cutoff=date(2026, 11, 13),
        sessions=3,
        non_trading_dates={HOLIDAY},
    )
    assert result["date"].tolist() == list(WINDOW)


def test_index_duplicate_in_required_window_is_fatal():
    with pytest.raises(ValueError, match="duplicate valid sessions: 2026-11-12"):
        index_session_history(
            _frame(*WINDOW, date(2026, 11, 12)),
            cutoff=date(2026, 11, 13),
            sessions=3,
            non_trading_dates={HOLIDAY},
        )


def test_older_index_duplicates_outside_window_are_irrelevant():
    result = index_session_history(
        _frame(date(2026, 11, 6), date(2026, 11, 6), *WINDOW),
        cutoff=date(2026, 11, 13),
        sessions=3,
        non_trading_dates={HOLIDAY},
    )
    assert result["date"].tolist() == list(WINDOW)


def test_stock_excludes_holiday_but_preserves_all_trading_sessions():
    stock, ignored = align_stock_history(
        _frame(HOLIDAY, *reversed(WINDOW)),
        symbol="ALE.WA",
        required_sessions=WINDOW,
        non_trading_dates={HOLIDAY},
    )
    assert stock["date"].tolist() == list(WINDOW)
    assert ignored == (HOLIDAY,)


def test_stock_reports_missing_real_session_without_filling():
    with pytest.raises(ValueError, match="PZU.WA: missing 1 required sessions: 2026-11-12"):
        align_stock_history(
            _frame(WINDOW[0], WINDOW[2], HOLIDAY),
            symbol="PZU.WA",
            required_sessions=WINDOW,
            non_trading_dates={HOLIDAY},
        )


def test_stock_duplicate_valid_session_is_fatal():
    with pytest.raises(ValueError, match="duplicate valid sessions: 2026-11-12"):
        align_stock_history(
            _frame(*WINDOW, WINDOW[1]),
            symbol="ALE.WA",
            required_sessions=WINDOW,
            non_trading_dates={HOLIDAY},
        )


def test_unmatched_weekday_rejects_suspect_index_or_stock_calendar():
    expected = (WINDOW[0], WINDOW[2])
    with pytest.raises(ValueError, match="unexpected weekdays absent from WIG20 index"):
        align_stock_history(
            _frame(*expected, WINDOW[1]),
            symbol="ALE.WA",
            required_sessions=expected,
            non_trading_dates={HOLIDAY},
        )


def test_resolve_skips_holiday_candidates_and_reports_symbols(monkeypatch):
    cutoff = date(2026, 11, 20)
    expected = (date(2026, 11, 19), cutoff)
    requested = []

    monkeypatch.setattr(
        wig20,
        "candidate_dates",
        lambda conn, maximum: [HOLIDAY, cutoff],
    )

    def mock_constituents(conn, day):
        requested.append(day)
        return pd.DataFrame({"symbol": ["ALE.WA", "PZU.WA"]})

    monkeypatch.setattr(wig20, "constituents", mock_constituents)
    monkeypatch.setattr(
        wig20,
        "_index_window",
        lambda conn, day, sessions, holidays: _frame(*expected),
    )

    def mock_stock(conn, symbol, day, sessions, *, expected_sessions, non_trading_dates):
        if symbol == "PZU.WA":
            raise ValueError("PZU.WA: missing 2 required sessions: 2026-11-19, 2026-11-20")
        return _frame(*expected), ()

    monkeypatch.setattr(wig20, "stock_history", mock_stock)

    with pytest.raises(ValueError, match="PZU.WA: missing 2 required sessions") as error:
        wig20.resolve(None, sessions=2, non_trading_dates={HOLIDAY})
    assert "Stock trading calendars do not match" not in str(error.value)
    assert requested == [cutoff]
