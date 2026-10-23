
from datetime import date

import pytest

from marketsim.simulator.simulation_calendar import SimulationCalendar


@pytest.fixture
def calendar():
    """Five consecutive trading days, skipping the weekend."""
    return SimulationCalendar([
        date(2026, 10, 19),
        date(2026, 10, 20),
        date(2026, 10, 21),
        date(2026, 10, 22),
        date(2026, 10, 23),
        date(2026, 10, 26),
    ])


@pytest.mark.parametrize(
    "day, expected_date",
    [
        (0, date(2026, 10, 19)),
        (2, date(2026, 10, 21)),
        (5, date(2026, 10, 26)),
    ],
)
def test_day_to_date(calendar, day, expected_date):
    assert calendar.to_date(day) == expected_date


@pytest.mark.parametrize(
    "trading_date, expected_day",
    [
        (date(2026, 10, 19), 0),
        (date(2026, 10, 22), 3),
        (date(2026, 10, 26), 5),
    ],
)
def test_date_to_day(calendar, trading_date, expected_day):
    assert calendar.to_day(trading_date) == expected_day


def test_roundtrip_mapping(calendar):
    for day in range(len(calendar)):
        trading_date = calendar.to_date(day)
        assert calendar.to_day(trading_date) == day


def test_calendar_length(calendar):
    assert len(calendar) == 6


def test_single_day_calendar():
    trading_date = date(2026, 10, 23)
    calendar = SimulationCalendar([trading_date])

    assert len(calendar) == 1
    assert calendar.to_date(0) == trading_date
    assert calendar.to_day(trading_date) == 0


def test_empty_calendar_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        SimulationCalendar([])


def test_duplicate_dates_rejected():
    with pytest.raises(ValueError, match="unique and chronological"):
        SimulationCalendar([
            date(2026, 10, 19),
            date(2026, 10, 19),
        ])


def test_unsorted_dates_rejected():
    with pytest.raises(ValueError, match="unique and chronological"):
        SimulationCalendar([
            date(2026, 10, 20),
            date(2026, 10, 19),
        ])


def test_missing_trading_date(calendar):
    with pytest.raises(KeyError):
        calendar.to_day(date(2026, 10, 24))


def test_missing_calendar_date(calendar):
    with pytest.raises(KeyError):
        calendar.to_day(date(2027, 1, 1))


def test_out_of_range_day(calendar):
    with pytest.raises(IndexError):
        calendar.to_date(len(calendar))


def test_weekend_is_skipped(calendar):
    friday = date(2026, 10, 23)
    monday = date(2026, 10, 26)

    assert calendar.to_day(monday) == calendar.to_day(friday) + 1
    assert len(calendar) == 6


def test_negative_day_rejected(calendar):
    with pytest.raises(IndexError):
        calendar.to_date(-1)
