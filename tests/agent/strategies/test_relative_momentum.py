from datetime import date

import pandas as pd
import pytest

from marketsim.agent.strategies.relative_momentum import (
    MomentumSettings,
    equal_weight_targets,
    rank_momentum,
)


def bars(closes, *, volume=100_000, start="2025-01-01"):
    return pd.DataFrame(
        {
            "date": pd.bdate_range(start, periods=len(closes)),
            "close": closes,
            "volume": [volume] * len(closes),
        }
    )


def test_ranks_52_week_range_and_equal_weights():
    settings = MomentumSettings(
        lookback_sessions=3, turnover_sessions=2, min_median_turnover=0, portfolio_size=2
    )
    histories = {"A": bars([10, 12, 15]), "B": bars([20, 25, 21]), "C": bars([10, 10, 10])}
    ranking = rank_momentum(histories, as_of=date(2025, 1, 3), settings=settings)
    assert ranking.symbol.tolist() == ["A", "B"]
    assert ranking.score.iloc[0] == 1.0
    assert equal_weight_targets(ranking) == {"A": 0.5, "B": 0.5}


def test_does_not_see_future_bars():
    settings = MomentumSettings(lookback_sessions=3, turnover_sessions=2, min_median_turnover=0)
    history = bars([10, 12, 15, 1000])
    ranking = rank_momentum({"A": history}, as_of=date(2025, 1, 3), settings=settings)
    assert ranking.last_close.iloc[0] == 15


def test_insufficient_history_and_liquidity():
    settings = MomentumSettings(
        lookback_sessions=4, turnover_sessions=2, min_median_turnover=1_000_000
    )
    assert rank_momentum(
        {"A": bars([10, 11, 12])}, as_of=date(2025, 1, 10), settings=settings
    ).empty
    assert rank_momentum(
        {"A": bars([10, 11, 12, 13], volume=1)}, as_of=date(2025, 1, 10), settings=settings
    ).empty


def test_duplicate_dates_rejected():
    history = bars([10, 11, 12])
    history.loc[1, "date"] = history.loc[0, "date"]
    with pytest.raises(ValueError, match="duplicate"):
        rank_momentum(
            {"A": history},
            as_of=date(2025, 1, 10),
            settings=MomentumSettings(
                lookback_sessions=2, turnover_sessions=2, min_median_turnover=0
            ),
        )
