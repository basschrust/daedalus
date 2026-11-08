from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from marketsim.market import Price
from marketsim.market.instruments.future import Future


@pytest.mark.parametrize(
    ("instrument_class", "expected"),
    [
        ("index", Price(3100)),
        ("stock", Price(125)),
    ],
)
def test_future_theoretical_value(instrument_class, expected):
    underlying = SimpleNamespace(
        instrument_class=instrument_class,
        last_traded_price=expected,
        calculate_value=MagicMock(return_value=expected),
    )

    # Test the valuation method without initializing
    # the full market infrastructure.
    future = Future.__new__(Future)
    future.underlying = underlying

    assert future.calculate_theoretical_value() == expected

    if instrument_class == "index":
        underlying.calculate_value.assert_called_once_with()
    else:
        underlying.calculate_value.assert_not_called()


def test_future_theoretical_value_ignores_own_market_price():
    underlying = SimpleNamespace(
        instrument_class="stock",
        last_traded_price=Price(125),
    )

    future = Future.__new__(Future)
    future.underlying = underlying
    future.last_traded_price = Price(140)

    assert future.calculate_theoretical_value() == Price(125)
