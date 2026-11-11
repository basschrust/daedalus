from types import SimpleNamespace

import pandas as pd

from marketsim.agent.mm_instrument_limits import InstrumentLimitMixin


class Dummy(InstrumentLimitMixin):
    position = {1: 0, 2: 0, 3: 0}
    futures_position = {}


def market(asset_id, volume, kind="stock"):
    return SimpleNamespace(
        asset_id=asset_id,
        name=str(asset_id),
        instrument_class=kind,
        is_tradable=True,
        historical_ohlcv=pd.DataFrame({"volume": [volume, volume]}),
    )


def test_independent_limits_and_daily_reset():
    agent = Dummy()
    agent.initialize_instrument_limits(
        {1: market(1, 100000), 2: market(2, 1000), 3: market(3, 100, "future")},
        {"q_max": 1000000, "inventory_fraction": 0.10, "daily_volume_fraction": 0.20},
    )
    assert agent.q_max_by_asset == {1: 10000, 2: 100, 3: 10}
    assert agent.daily_volume_limit_by_asset == {1: 20000, 2: 200, 3: 20}
    assert agent.limited_quote_sizes(1, 1000, 4) == (1000, 1000)
    assert agent.limited_quote_sizes(3, 1000, 4) == (2, 2)
    agent.daily_executed_volume_by_asset[3] = 20
    assert agent.limited_quote_sizes(3, 1000, 4) == (0, 0)
