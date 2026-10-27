from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from marketsim.market.instruments.security import Security
from marketsim.market.price import Price


class Stock(Security):
    def __init__(
        self,
        *,
        repository,
        symbol: str | None = None,
        exchange: str | None = None,
        historical_ohlcv: pd.DataFrame | None = None,
        simulation_dates: list[date] | None = None,
        reference_price: Price | None = None,
        name: str | None = None,
        short_name: str | None = None,
        market_type: str = "continuous",
        output_dir: Path | str = "tmp",
        currency: str = "PLN",
    ) -> None:
        self.symbol = symbol
        self.exchange = exchange
        self.historical_ohlcv = historical_ohlcv
        self.simulation_dates = simulation_dates

        super().__init__(
            repository=repository,
            instrument_class="stock",
            reference_price=reference_price,
            name=name,
            short_name=short_name,
            market_type=market_type,
            output_dir=output_dir,
            currency=currency,
            historical_ohlcv=historical_ohlcv,
            simulation_dates=simulation_dates,
        )
