from decimal import Decimal

import pytest

from marketsim.market import Price


@pytest.mark.parametrize(
    "base_currency, day, expected",
    [
        ("PLN", 0, "5200.00"),
        ("PLN", 1, "5276.80"),
        ("USD", 0, str(Decimal("5200") / Decimal("3.80"))),
        ("USD", 1, str(Decimal("5276.80") / Decimal("3.90"))),
        ("EUR", 0, str(Decimal("5200") / Decimal("4.25"))),
        ("EUR", 1, str(Decimal("5276.80") / Decimal("4.20"))),
    ],
)
def test_multicurrency_portfolio_valuation(
    multicurrency_agent,
    fx_provider,
    base_currency,
    day,
    expected,
):
    agent = multicurrency_agent
    agent.base_currency = base_currency
    agent.fx_rate_provider = fx_provider

    prices = {
        0: ("100", "50", "80"),
        1: ("105", "52", "78"),
    }

    for market, price in zip(
        agent.markets.values(),
        prices[day],
    ):
        market.last_traded_price = Price(price)

    actual = agent.calculate_portfolio_value(day=day)

    assert actual == Price(expected)
