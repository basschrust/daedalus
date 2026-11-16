from __future__ import annotations

import math

import pandas as pd


def historical_volume(market):
    history = getattr(market, "historical_ohlcv", None)
    if not isinstance(history, pd.DataFrame) or history.empty:
        return None
    if "volume" not in history:
        raise ValueError(f"{market.name}: historical volume column missing")
    values = pd.to_numeric(history["volume"], errors="coerce")
    if (
        values.isna().any()
        or (values < 0).any()
        or not all(math.isfinite(float(x)) for x in values)
    ):
        raise ValueError(f"{market.name}: invalid historical volume")
    return float(values.mean())
