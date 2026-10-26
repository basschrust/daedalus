from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class HistoricalMarketData:
    symbol: str
    as_of: date
    reference_price: float
    daily_mean_return: float
    daily_volatility: float
    median_daily_volume: float
    observations: int
