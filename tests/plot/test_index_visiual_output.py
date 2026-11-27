from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from marketsim.market.instruments.index import Index


def test_index_summary_uses_zero_padded_filename(tmp_path: Path) -> None:
    index = object.__new__(Index)
    index.asset_id = 7
    index.index_id = 7
    index.instrument_class = "index"
    index.short_name = "WIG20"
    index.name = "WIG20"
    index.output_dir = tmp_path
    index.logger = SimpleNamespace(info=lambda *_: None)
    index.historical_ohlcv = pd.DataFrame(
        data={"close": [3000.0]},
        index=pd.to_datetime(["2026-11-20"]),
    )
    index.simulation_dates = [pd.Timestamp("2026-11-23")]
    index.repository = SimpleNamespace(
        get_eod_index_values=lambda *, index_id: pd.DataFrame(
            [{"day": 0, "open": 3000, "high": 3100, "low": 2900, "close": 3050}]
        )
    )

    with patch("marketsim.market.instruments.index.plot_combined_candles") as plot:
        index.show_summary()

    assert plot.call_args.kwargs["output_file"] == str(
        tmp_path / "candle_plots" / "0007_index_WIG20.png"
    )
