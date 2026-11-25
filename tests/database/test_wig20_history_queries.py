"""WIG20 source query must select a date interval, not LIMIT N rows."""

from datetime import date

import pandas as pd

from marketsim.database import wig20_history_queries


def test_stock_window_uses_date_interval(monkeypatch):
    calls = []
    expected = pd.DataFrame({"date": [date(2026, 11, 20)]})

    def fake_query(conn, sql, params):
        calls.append((sql, params))
        return expected

    monkeypatch.setattr(wig20_history_queries, "_query", fake_query)
    result = wig20_history_queries.fetch_stock_window(
        object(),
        symbol="PZU.WA",
        first_session=date(2025, 11, 20),
        cutoff=date(2026, 11, 20),
    )
    assert result is expected
    sql, params = calls[0]
    assert 'TO_DATE("DATE") >= %s' in sql
    assert 'TO_DATE("DATE") <= %s' in sql
    assert "LIMIT" not in sql.upper()
    assert params == ("PZU.WA", date(2025, 11, 20), date(2026, 11, 20))
