import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from marketsim.agent.market_maker_zoh import MMZOHAgent
from marketsim.market import Price
from marketsim.market.fourheap.constants import BUY, SELL


def expected_mm_volume(
    daily_volume: float,
    fraction: float = 0.10,
    sim_time: int = 40,
    rebalance_period: int = 3,
    levels: int = 5,
) -> int:
    rebalance_count = math.ceil(sim_time / rebalance_period)

    return max(
        1,
        round(daily_volume * fraction / (rebalance_count * 2 * levels)),
    )


def make_market(asset_id, symbol, volumes):
    """Lightweight market with historical OHLCV and order capture."""
    if volumes is None:
        history = None
    else:
        history = pd.DataFrame({"volume": volumes})

    return SimpleNamespace(
        asset_id=asset_id,
        name=symbol,
        instrument_class="stock",
        historical_ohlcv=history,
        last_traded_price=Price(100),
        order_book=SimpleNamespace(
            get_best_bid=MagicMock(return_value=Price(99)),
            get_best_ask=MagicMock(return_value=Price(101)),
        ),
        withdraw_all=MagicMock(),
        add_orders=MagicMock(),
    )


def make_mm(markets, tmp_path, **config):
    repository = MagicMock()

    configuration = {
        "K": 5,
        "volume_fraction": 0.10,
        "volume": 500,
        "rebalance_period": 3,
        "xi": 0.1,
        "omega": 0.1,
        "q_max": 1000000,
        **config,
    }

    return MMZOHAgent(
        markets=markets,
        repository=repository,
        configuration=configuration,
        output_dir=tmp_path,
        sim_time=40,
    )


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [
        (0.05, 71),
        (0.10, 143),
        (0.20, 286),
        (1.00, 1429),
    ],
)
def test_configurable_volume_fraction(tmp_path, fraction, expected):
    market = make_market(1, "PKO.WA", [200_000])

    mm = make_mm(
        [market],
        tmp_path,
        volume_fraction=fraction,
    )

    assert mm.volume_by_asset[1] == expected


def test_different_assets_get_different_quantities(tmp_path):
    markets = [
        make_market(1, "PKO.WA", [2_000_000] * 5),
        make_market(2, "PKN.WA", [1_000_000] * 5),
        make_market(3, "KGH.WA", [500_000] * 5),
    ]

    mm = make_mm(markets, tmp_path)

    assert mm.volume_by_asset == {
        1: 1429,
        2: 714,
        3: 357,
    }


@pytest.mark.parametrize(
    "fraction",
    [-1, -0.1, 0, 1.01, 2],
)
def test_invalid_volume_fraction_is_rejected(tmp_path, fraction):
    market = make_market(1, "PKO.WA", [200_000])

    with pytest.raises(
        ValueError,
        match="volume_fraction",
    ):
        make_mm(
            [market],
            tmp_path,
            volume_fraction=fraction,
        )


@pytest.mark.parametrize(
    "volumes",
    [
        None,
        [],
    ],
)
def test_missing_history_uses_fixed_volume(tmp_path, volumes):
    market = make_market(1, "PKO.WA", volumes)

    mm = make_mm(
        [market],
        tmp_path,
        volume=750,
    )

    assert mm.volume_by_asset[1] == 750


@pytest.mark.parametrize(
    "volumes",
    [
        [0, 0, 0],
        [-100, -200],
        [100, float("nan")],
        [float("inf")],
    ],
)
def test_invalid_historical_volume_is_rejected(tmp_path, volumes):
    market = make_market(1, "PKO.WA", volumes)

    with pytest.raises(
        ValueError,
        match="Invalid historical volume",
    ):
        make_mm([market], tmp_path)


def test_small_volume_has_minimum_order_quantity(tmp_path):
    market = make_market(1, "PKO.WA", [10, 20])

    mm = make_mm([market], tmp_path)

    assert mm.volume_by_asset[1] == 1


def test_mm_uses_calibrated_quantity_in_orders(tmp_path):
    market = make_market(
        asset_id=1,
        symbol="PKO.WA",
        volumes=[200_000] * 5,
    )

    mm = make_mm([market], tmp_path)

    # Position is initially neutral.
    mm.take_action(current_time=0)

    market.withdraw_all.assert_called_once_with(agent_id=mm.agent_id)
    market.add_orders.assert_called_once()

    orders = market.add_orders.call_args.args[0]

    # Five levels, two sides.
    assert len(orders) == 10

    assert all(order.quantity == 143 for order in orders)
    assert all(order.asset_id == 1 for order in orders)

    assert sum(o.order_type == BUY for o in orders) == 5
    assert sum(o.order_type == SELL for o in orders) == 5


def test_multiasset_orders_use_correct_quantities(tmp_path):
    pko = make_market(1, "PKO.WA", [200_000])
    kgh = make_market(2, "KGH.WA", [80_000])

    mm = make_mm([pko, kgh], tmp_path)
    mm.take_action(current_time=0)

    pko_orders = pko.add_orders.call_args.args[0]
    kgh_orders = kgh.add_orders.call_args.args[0]

    assert len(pko_orders) == 10
    assert len(kgh_orders) == 10

    assert {o.quantity for o in pko_orders} == {143}
    assert {o.quantity for o in kgh_orders} == {57}

    assert {o.asset_id for o in pko_orders} == {1}
    assert {o.asset_id for o in kgh_orders} == {2}
