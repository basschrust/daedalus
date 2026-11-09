"""Regression tests for option strike types at expiration."""

from unittest.mock import MagicMock, patch

import pytest

from marketsim.market.instruments.option import Option
from marketsim.market.price import Price


@pytest.mark.parametrize(
    ("option_side", "underlying_price", "expected_payoff"),
    [
        ("CALL", "2600", "100"),  # In the money
        ("CALL", "2400", "0"),  # Out of the money
        ("PUT", "2400", "100"),  # In the money
        ("PUT", "2600", "0"),  # Out of the money
    ],
)
def test_option_with_string_strike_at_expiration(
    option_side,
    underlying_price,
    expected_payoff,
):
    """A YAML string strike must work at zero time to expiration."""

    underlying = MagicMock()
    underlying.last_traded_price = Price(underlying_price)
    underlying.short_name = "WIG20"

    repository = MagicMock()

    with patch(
        "marketsim.market.instruments.option.Security.__init__",
        return_value=None,
    ):
        option = Option(
            derivatives_config={
                "strike": "2500.00",
                "expiration": 8,
                "option_side": option_side,
                "option_type": "European",
            },
            underlying=underlying,
            repository=repository,
            name="WIG20_TEST_OPTION",
        )

    assert isinstance(option.strike, Price)
    assert option.strike == Price("2500.00")

    payoff = option.get_theoretical_price(as_of_day=8)

    assert payoff == Price(expected_payoff)
