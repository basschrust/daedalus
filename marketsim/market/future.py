from __future__ import annotations

import pandas as pd

from typing_extensions import TYPE_CHECKING
from pathlib import Path

from marketsim.input import config
from marketsim.market.valuation_libs.BlackScholes import BSCall, BSPut
from marketsim.plot.candle import plot_candlestick_derivative
from marketsim.market import Security, Index
from marketsim.market import Price
# from marketsim.database.connectors.duckdb_storage import Repository

if TYPE_CHECKING:
    from database.connectors.duckdb_storage import Repository
    from marketsim.fourheap import MatchedOrder

class Future(Security):
    def __init__(self, *, underlying: Index, expiration: int, multiplier: float,
                 repository: Repository, market_type: str = "continuous",
                 name: str| None = None, short_name: str|None=None, output_dir: Path|str="tmp"):
        self.instrument_class = "future"
        self.underlying = underlying
        self.expiration = expiration
        self.multiplier = multiplier
        super().__init__(market_type=market_type, repository=repository, instrument_class="future",
                 name=name, short_name=short_name, output_dir=output_dir)

    def save(self):
        self.repository.save_future(self)

    def fill_theoretical_value(self):
        for t in self.traded_prices:
            self.traded_prices[t]["theoretical"] = self.underlying.traded_prices[t]["close"]

    def plot_history(self, traded_prices: dict):
        # fill the theoretical - simply index value,
        # TODO: add connrection on dividends and financing rate
        self.fill_theoretical_value()

        traded_prices_float = {t: {v: float(price_item) for v, price_item in item.items()}
                               for t, item in traded_prices.items()}
        df_candlestick = pd.DataFrame.from_dict(traded_prices_float,
                                                orient="index"
                                                )
        df_candlestick.index.name = "time"
        self.logger.info(f"Future candlestick to plot: {df_candlestick.head()}")

        candlestick_filename = f"{self.output_dir}/candlestick_{str(self)}.png"
        plot_candlestick_derivative(df=df_candlestick, output_file=candlestick_filename, title=self.name)


    def sod(self):
        if self.status == "active":
            super().sod()

            if self.eod_status == "open":
                theoretical_price = self.underlying.last_traded_price
                # TODO: adjust for dividends and financing
                self.traded_prices = {0: {"open": Price(theoretical_price),
                                          "low": Price(theoretical_price),
                                          "high": Price(theoretical_price),
                                          "close": Price(theoretical_price),
                                          "theoretical": Price(theoretical_price),
                                          "volume": 0, }}

                self.logger.info(f"Futures SoD completed for day: {self.current_day}")

