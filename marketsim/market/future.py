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
                 margin_rate: float,
                 repository: Repository, market_type: str = "continuous",
                 name: str| None = None, short_name: str|None=None, output_dir: Path|str="tmp"):
        self.instrument_class = "future"
        self.underlying = underlying
        self.expiration = expiration
        self.multiplier = multiplier
        self.margin_rate = margin_rate

        super().__init__(market_type=market_type, repository=repository, instrument_class="future",
                 name=name, short_name=short_name, output_dir=output_dir, reference_price=underlying.last_traded_price)

    def save(self):
        self.repository.save_future(self)

    def record_trade(self, matched_order: MatchedOrder) -> None:
        self.last_traded_price = matched_order.price
        self.record_volume_and_price(matched_order=matched_order)

        agent_id = matched_order.order.agent_id
        self.agents[agent_id].record_futures_trade(
            matched_order=matched_order,
        )

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

    def eod(self):
        if self.eod_status == "closed":
            return
        elif self.eod_status == "open":
            self.exercise()
            super().eod()

            self.fill_theoretical_value()
            self.traded_prices_df = (
                pd.DataFrame.from_dict(self.traded_prices, orient="index")
                .rename_axis("time_tick")
                .reset_index()
                [["time_tick", "open", "high", "low", "close", "volume", "theoretical"]]
            )

            self.traded_prices_df["asset_id"] = self.asset_id
            self.traded_prices_df["day"] = self.current_day - 1

            self.logger.info(f"Traded_prices_df: {self.traded_prices_df.head()}")
            self.repository.save_traded_prices(self.traded_prices_df)

            # eod_prices
            eod_prices_df = pd.DataFrame([{
                "open": self.traded_prices_df.loc[
                    self.traded_prices_df["time_tick"] == 0, "open"
                ].iloc[0],
                "high": self.traded_prices_df["high"].max(),
                "low": self.traded_prices_df["low"].min(),
                "close": self.traded_prices_df.loc[
                    self.traded_prices_df["time_tick"].idxmax(), "close"
                ],
                "volume": self.traded_prices_df["volume"].sum(),
                "asset_id": self.asset_id,
                "day": self.current_day - 1,
                "theoretical": self.underlying.last_traded_price,
            }])

            self.repository.save_eod_prices(eod_prices_df=eod_prices_df)

            if self.status != "active":
                self.cleanup()

            self.logger.info(f"Option EoD completed for day: {self.current_day-1}")

    def exercise(self):
        self.logger.info(f"Checking Future Expiration: {self.name}")
        if self.expiration == self.current_day:
            # TODO: calculate final settlement
            self.status = "expired"
            # for all agents that had open positions make final settlement, give them cash back :)
            for agent_id, agent in self.agents.items():
                agent.settle(self)
        else:
            return 0

