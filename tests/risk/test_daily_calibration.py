"""No-lookahead rolling calibration tests using the local DuckDB backend."""

from datetime import date
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest

from marketsim.database.risk.risk_queries import (
    ensure_daily_calibration_tables,
    publish_daily_calibration,
)
from marketsim.risk.daily_calibration import DailyRiskCalibration


def _simulator(conn):
    dates = pd.to_datetime(["2026-11-10", "2026-11-11", "2026-11-12", "2026-11-13"])
    pzu = SimpleNamespace(instrument_class="stock", short_name="PZU.WA", last_traded_price=106.0)
    wig20 = SimpleNamespace(instrument_class="index", short_name="WIG20", last_traded_price=2030.0)
    option = SimpleNamespace(instrument_class="option", short_name="OW20", last_traded_price=5.0)
    return SimpleNamespace(
        repository=SimpleNamespace(
            connection=conn,
            ensure_risk_calibration_tables=lambda: ensure_daily_calibration_tables(
                SimpleNamespace(connection=conn)
            ),
            publish_risk_calibration=lambda calibration_id, instruments, pairs, **details: (
                publish_daily_calibration(
                    SimpleNamespace(connection=conn), calibration_id, instruments, pairs, **details
                )
            ),
        ),
        markets={1: pzu, 2: wig20, 3: option},
        historical_ohlcv_by_symbol={
            "PZU.WA": pd.DataFrame({"close": [100, 101, 103, 104]}, index=dates),
            "WIG20": pd.DataFrame({"close": [2000, 2010, 2005, 2020]}, index=dates),
        },
        simulation_dates=[date(2026, 11, 16), date(2026, 11, 17)],
    )


def test_calibration_uses_historical_then_simulated_closes():
    with duckdb.connect(":memory:") as conn:
        sim = _simulator(conn)
        model = DailyRiskCalibration(sim, lookback_sessions=4)
        historical_id = model.latest_calibration_id
        assert historical_id.startswith("SIM-20261113-")

        initial_rows = conn.execute(
            "SELECT symbol, mean_daily_return, daily_sigma "
            "FROM risk_l2_instrument_statistics WHERE calibration_id = ? ORDER BY symbol",
            [historical_id],
        ).fetchall()
        assert [row[0] for row in initial_rows] == ["PZU.WA", "WIG20"]
        pzu_returns = np.diff(np.log([100, 101, 103, 104]))
        assert initial_rows[0][1] == pytest.approx(pzu_returns.mean())
        assert initial_rows[0][2] == pytest.approx(pzu_returns.std(ddof=1))

        day0_id = model.append_eod(sim, 0)
        assert day0_id.startswith("SIM-20261116-")
        assert day0_id != historical_id

        # 4-session rolling window now includes Nov 16; Nov 10 has fallen out.
        rows = conn.execute(
            "SELECT mean_daily_return, daily_sigma FROM risk_l2_instrument_statistics "
            "WHERE calibration_id = ? AND symbol = 'PZU.WA'",
            [day0_id],
        ).fetchall()
        expected = np.diff(np.log([101, 103, 104, 106]))
        assert rows[0][0] == pytest.approx(expected.mean())
        assert rows[0][1] == pytest.approx(expected.std(ddof=1))
        assert (
            conn.execute(
                "SELECT count(*) FROM risk_l2_pair_statistics WHERE calibration_id = ?",
                [day0_id],
            ).fetchone()[0]
            == 1
        )

        sim.markets[1].last_traded_price = 108.0
        sim.markets[2].last_traded_price = 2035.0
        day1_id = model.append_eod(sim, 1)
        assert len({historical_id, day0_id, day1_id}) == 3
        assert conn.execute(
            "SELECT day, source, observations FROM risk_calibration_history ORDER BY day"
        ).fetchall() == [
            (-1, "IKAROS", 3),
            (0, "SIMULATED", 3),
            (1, "SIMULATED", 3),
        ]
        # No historical calibration is overwritten as days advance.
        assert (
            conn.execute(
                "SELECT count(*) FROM risk_l2_instrument_statistics WHERE calibration_id = ?",
                [historical_id],
            ).fetchone()[0]
            == 2
        )


def test_missing_historical_factor_fails_instead_of_using_a_default():
    with duckdb.connect(":memory:") as conn:
        sim = _simulator(conn)
        del sim.historical_ohlcv_by_symbol["WIG20"]
        with pytest.raises(ValueError, match="Missing Ikaros historical OHLCV for WIG20"):
            DailyRiskCalibration(sim, lookback_sessions=4)


def test_invalid_simulated_close_rejected_without_advancing():
    with duckdb.connect(":memory:") as conn:
        sim = _simulator(conn)
        model = DailyRiskCalibration(sim, lookback_sessions=4)
        sim.markets[1].last_traded_price = float("nan")
        with pytest.raises(ValueError, match="must be finite and positive"):
            model.append_eod(sim, 0)
        assert model.next_day == 0
        assert date(2026, 11, 16) not in model.series["PZU.WA"]


def test_calibration_has_no_future_data_at_initialization():
    with duckdb.connect(":memory:") as conn:
        sim = _simulator(conn)
        sim.historical_ohlcv_by_symbol["PZU.WA"].loc[pd.Timestamp("2026-11-16")] = 105
        with pytest.raises(ValueError, match="overlaps simulation"):
            DailyRiskCalibration(sim, lookback_sessions=4)


def test_historical_calendar_gap_is_not_treated_as_a_daily_return():
    with duckdb.connect(":memory:") as conn:
        sim = _simulator(conn)
        sim.historical_ohlcv_by_symbol["WIG20"] = sim.historical_ohlcv_by_symbol["WIG20"].drop(
            pd.Timestamp("2026-11-12")
        )
        with pytest.raises(ValueError, match="trading calendars differ"):
            DailyRiskCalibration(sim, lookback_sessions=4)


def test_bad_second_factor_does_not_partially_advance_first_factor():
    with duckdb.connect(":memory:") as conn:
        sim = _simulator(conn)
        model = DailyRiskCalibration(sim, lookback_sessions=4)
        sim.markets[2].last_traded_price = float("nan")
        with pytest.raises(ValueError, match="WIG20: risk-calibration close"):
            model.append_eod(sim, 0)
        assert all(date(2026, 11, 16) not in history for history in model.series.values())
        assert model.next_day == 0
