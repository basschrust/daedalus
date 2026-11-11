from types import SimpleNamespace

import numpy as np
import pytest

from marketsim.market.parametric_var import ParametricVaR


def test_two_asset_covariance_and_short_positions():
    regime = SimpleNamespace(
        symbols=("AAA", "BBB"),
        covariance=np.array(
            [
                [0.04**2, 0.04 * 0.02 * 0.5],
                [0.04 * 0.02 * 0.5, 0.02**2],
            ]
        ),
    )
    risk = ParametricVaR("test", regime)

    def market(asset_id, symbol):
        return SimpleNamespace(
            asset_id=asset_id,
            short_name=symbol,
            instrument_class="stock",
            last_traded_price=100,
            currency="PLN",
        )

    agent = SimpleNamespace(
        agent_id=7,
        position={1: 10, 2: -5},
        futures_position={},
        markets={1: market(1, "AAA"), 2: market(2, "BBB")},
        base_currency="PLN",
        portfolio_value=900,
        fx_rate_provider=SimpleNamespace(convert=lambda amount, **kwargs: amount),
    )
    row = risk.row_for_agent(agent, 2)
    exposure = np.array([1000, -500])
    assert row["var_amount"] == pytest.approx(
        risk.z * np.sqrt(exposure @ regime.covariance @ exposure)
    )
    assert row["covered_exposure"] == 1500
    assert row["excluded_exposure"] == 0
    assert row["horizon_days"] == 1
    agent.position = {1: 0, 2: 0}
    assert risk.row_for_agent(agent, 2)["var_amount"] == 0


def test_derivatives_are_explicitly_excluded():
    risk = ParametricVaR("test", SimpleNamespace(symbols=("AAA",), covariance=np.array([[0.01]])))
    option = SimpleNamespace(
        instrument_class="option",
        short_name="OPT",
        last_traded_price=5,
        multiplier=10,
        currency="PLN",
    )
    future = SimpleNamespace(
        instrument_class="future",
        short_name="FUT",
        last_traded_price=4000,
        multiplier=20,
        currency="PLN",
    )
    agent = SimpleNamespace(
        agent_id=1,
        position={1: 2},
        futures_position={2: -3},
        markets={1: option, 2: future},
        base_currency="PLN",
        portfolio_value=1000,
        fx_rate_provider=SimpleNamespace(convert=lambda amount, **kwargs: amount),
    )
    row = risk.row_for_agent(agent, 0)
    assert row["var_amount"] == 0
    assert row["excluded_exposure"] == 2 * 5 * 10 + 3 * 4000 * 20
