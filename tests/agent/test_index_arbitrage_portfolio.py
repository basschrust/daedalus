from types import SimpleNamespace

import pytest

from marketsim.agent.helpers.index_arbitrage_portfolio import (
    index_hedge_quantities,
)


@pytest.fixture
def future():
    index = SimpleNamespace(
        base_index_level=3000,
        components={
            11: {
                "weight": 0.6,
                "base_price": 100,
            },
            12: {
                "weight": 0.4,
                "base_price": 200,
            },
        },
    )

    return SimpleNamespace(
        asset_id=101,
        multiplier=20,
        underlying=index,
    )


def test_short_futures_long_stocks(future):
    result = index_hedge_quantities(future, contracts=-2)

    # Coefficients: 18 and 6.
    # Two contracts, multiplier 20.
    assert result == {
        11: 720,
        12: 240,
    }


def test_long_futures_short_stocks(future):
    result = index_hedge_quantities(future, contracts=2)

    assert result == {
        11: -720,
        12: -240,
    }


def test_flat_futures_flat_basket(future):
    assert index_hedge_quantities(future, contracts=0) == {
        11: 0,
        12: 0,
    }


def test_hedge_scales_with_contracts(future):
    one = index_hedge_quantities(future, contracts=1)
    five = index_hedge_quantities(future, contracts=5)

    assert five == {asset_id: 5 * quantity for asset_id, quantity in one.items()}


def test_weights_do_not_need_to_sum_to_one(future):
    for component in future.underlying.components.values():
        component["weight"] *= 10

    assert index_hedge_quantities(future, contracts=-2) == {
        11: 720,
        12: 240,
    }


def test_invalid_multiplier(future):
    future.multiplier = 0

    with pytest.raises(ValueError):
        index_hedge_quantities(future, contracts=1)


def test_invalid_base_price(future):
    future.underlying.components[11]["base_price"] = 0

    with pytest.raises(ValueError):
        index_hedge_quantities(future, contracts=1)


def test_legacy_index_coefficients(future):
    future.underlying.base_index_level = None

    result = index_hedge_quantities(future, contracts=-2)

    assert result == {
        11: 24,
        12: 16,
    }
