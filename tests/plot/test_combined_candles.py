import pandas as pd

from marketsim.plot.daedalus_plot_extensions import combine_candles


def test_combine_candles_with_expired_future():
    historical = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [100],
        },
        index=pd.to_datetime(["2026-01-02"]),
    )

    simulated = pd.DataFrame(
        {
            "day": [0, 1],
            "open": [101.0, 102.0],
            "high": [103.0, 104.0],
            "low": [100.0, 101.0],
            "close": [102.0, 103.0],
            "volume": [50, 60],
        }
    )

    simulation_dates = pd.bdate_range(
        "2026-01-05",
        periods=5,
    )

    combined, first_simulated = combine_candles(
        historical,
        simulated,
        simulation_dates,
    )

    assert len(combined) == 3
    assert first_simulated == pd.Timestamp("2026-01-05")
    assert combined.index.max() == pd.Timestamp("2026-01-06")
