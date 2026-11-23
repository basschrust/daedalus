"""Unified Black-Scholes valuation for market prices and VaR scenarios."""

import numpy as np
import pytest

from marketsim.market.price import Price
from marketsim.market.valuation_libs.BlackScholes import BSCall, BSPut


@pytest.mark.parametrize("model", [BSCall, BSPut])
def test_vector_prices_and_deltas_match_scalar(model):
    spots = np.array([75.0, 100.0, 125.0, 180.0])
    batch = model(S=spots, K=100, r=0.04, volatility=0.2, Time=0.75)
    assert isinstance(batch["price"], np.ndarray)
    for i, spot in enumerate(spots):
        one = model(S=float(spot), K=100, r=0.04, volatility=0.2, Time=0.75)
        for field in ("price", "delta", "gamma", "vega", "theta", "rho"):
            assert batch[field][i] == pytest.approx(one[field], abs=1e-10)


@pytest.mark.parametrize("model,expected", [(BSCall, [0, 0, 20]), (BSPut, [20, 0, 0])])
def test_expiry_prices_are_vectorized_and_scalar_preserves_price(model, expected):
    spots = np.array([80.0, 100.0, 120.0])
    prices = model(S=spots, K=100, r=0, volatility=0, Time=0)["price"]
    np.testing.assert_allclose(prices, expected)
    assert isinstance(model(S=120, K=100, r=0, volatility=0, Time=0)["price"], Price)


def test_expired_put_delta_has_correct_negative_sign():
    assert BSPut(S=80, K=100, r=0, volatility=0.2, Time=0)["delta"] == -1
    assert BSPut(S=100, K=100, r=0, volatility=0.2, Time=0)["delta"] == -0.5


@pytest.mark.parametrize("model", [BSCall, BSPut])
def test_zero_volatility_matches_deterministic_discounted_payoff(model):
    spot = np.array([80.0, 100.0, 120.0])
    rate = 0.05
    years = 2
    out = model(S=spot, K=100, r=rate, volatility=0, Time=years)
    strike_pv = 100 * np.exp(-rate * years)
    expected = (
        np.maximum(spot - strike_pv, 0) if model is BSCall else np.maximum(strike_pv - spot, 0)
    )
    np.testing.assert_allclose(out["price"], expected, atol=1e-12)
    assert np.isfinite(out["delta"]).all()


def test_nonzero_dividends_satisfy_put_call_parity():
    spot = np.array([100.0, 110.0])
    kwargs = dict(S=spot, K=100, r=0.04, volatility=0.25, Time=1.2, d=0.02)
    call = BSCall(**kwargs)["price"]
    put = BSPut(**kwargs)["price"]
    np.testing.assert_allclose(call - put, spot * np.exp(-0.02 * 1.2) - 100 * np.exp(-0.04 * 1.2))
