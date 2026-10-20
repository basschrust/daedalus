from __future__ import annotations

import pandas as pd

from typing_extensions import TYPE_CHECKING
from pathlib import Path

from marketsim.input import config
from marketsim.market.valuation_libs.BlackScholes import BSCall, BSPut
from marketsim.plot.candle import plot_candlestick_derivative
from marketsim.market import Security
from marketsim.market import Price

if TYPE_CHECKING:
    from database.connectors.duckdb_storage import Repository
    from marketsim.fourheap import MatchedOrder


class Option(Security):
    def __init__(self, *, derivatives_config: dict, underlying: Security,
                 market_type: str = "continuous", repository: Repository,
                 name: str| None = None, short_name: str|None=None, output_dir: Path|str="tmp") -> None:

        self.instrument_class = "option"
        self.underlying = underlying
        self.strike = derivatives_config["strike"]
        self.expiration = derivatives_config["expiration"] # TODO: prepare mapper for this
            # so that we can give relative time or precise dates or just take it from option series
            # for now we set mostly as 248 (number of trading days per year)
        self.option_side = derivatives_config["option_side"] # CALL/PUT
        self.option_type = derivatives_config.get("option_type", "European")
        self.r = 0 # the risk-free financing rate
        self.volatility = 0.157  # annualized volatility of the underlying security
        if short_name is None:
            self.short_name = f"{self.option_side} {self.underlying.short_name} {self.strike}"
        else:
            self.short_name = f"{short_name} {self.strike}"

        theoretical_price = self.get_theoretical_price(as_of_day=0)
        super().__init__(name=name, market_type=market_type, reference_price=Price(theoretical_price)
                         , instrument_class=self.instrument_class, repository=repository,
                         short_name=self.short_name, output_dir=output_dir)


    def calculate_greeks(self):
        pass

    def get_theoretical_price(self, as_of_day: int | None = None) -> Price:
        # TODO: is current_time needed as parameter?
        # as_of_day - day for which the option is valuated, could be current, could be future
        # returns theoretical price of the option, using BS formula
        as_of_day = as_of_day if as_of_day is not None else self.current_day
        if self.option_side == "CALL":
            call_option = BSCall(S=self.underlying.last_traded_price, K=self.strike,
                                 r=self.r, volatility=self.volatility, Time=(self.expiration-as_of_day)/248, d=0.0)
            # TODO: this gives us the option price along with its Greeks :)
            return call_option["price"]
        elif self.option_side == "PUT":
            put_option = BSPut(S=self.underlying.last_traded_price, K=self.strike,
                                 r=self.r, volatility=self.volatility, Time=(self.expiration-as_of_day)/248, d=0.0)
            # TODO: this gives us the option price along with its Greeks :)
            return put_option["price"]
        else:
            ValueError(f"Unknown option side: {self.option_side}")

    def fill_theoretical_price(self):
        # as_of_day - value of the option is calculated for that day
        for t, price_row in self.traded_prices.items():
            if "theoretical" in price_row:
                pass
            else:
                if self.option_side == "CALL":
                    call_option = BSCall(#S=price_row.get("close"),
                                         S=self.underlying.traded_prices.get(t, {}).get("close"),
                                         K=self.strike, r=self.r,
                                         volatility=self.volatility,
                                         Time=(self.expiration-self.current_day-1)/248, d=0.0)
                    price_row["theoretical"] = call_option.get("price", 100)
                elif self.option_side == "PUT":
                    put_option = BSPut(# S=price_row.get("close"),
                                        S=self.underlying.traded_prices.get(t,{}).get("close"),
                                       K=self.strike, r=self.r,
                                         volatility=self.volatility,
                                         Time=(self.expiration-self.current_day-1)/248, d=0.0)
                    # TODO: check if this is called as of D (today) or as of D-1 (yesterday) during EoD
                    price_row["theoretical"] = put_option.get("price", 100)

    def roll_traded_prices(self, current_time:int) -> None:
        yesterday = self.traded_prices[current_time - 1]
        self.traded_prices[current_time] = {"open": yesterday["close"],
                                            "low": yesterday["close"],
                                            "high": yesterday["close"],
                                            "close": yesterday["close"],
                                            "volume": 0,
                                            "theoretical" : yesterday["theoretical"] }
                                            # TODO we need the day number here, too to calculate it
                        #  "theoretical": self.get_theoretical_price(), }

    def record_volume_and_price(self, matched_order: MatchedOrder) -> None:
        # overriden in derivatives to include theoretical
        current_time = matched_order.time
        price = matched_order.price
        volume = matched_order.order.quantity
        if current_time in self.traded_prices:
            # update data
            if price > self.traded_prices[current_time]["high"]:
                self.traded_prices[current_time]["high"] = price
            elif price < self.traded_prices[current_time]["low"]:
                self.traded_prices[current_time]["low"] = price
            old_volume = self.traded_prices[current_time]["volume"]
            self.traded_prices[current_time]["volume"] = volume + old_volume
            self.traded_prices[current_time]["close"] = price
            self.traded_prices[current_time]["theoretical"] = self.get_theoretical_price()
        else:
            # enter as first day in this time tick
            self.traded_prices[current_time] = { "open": price,
                                                 "low": price,
                                                 "high": price,
                                                 "close": price,
                                                 "volume": volume,
                                                 "theoretical": self.get_theoretical_price(),}

    def plot_history(self, traded_prices: dict):
        # ensure that theoretical will be plotted, too:
        # self.fill_theoretical_price() # TODO: check if in historical analysis this takes proper Time

        traded_prices_float = {t: {v: float(price_item) for v, price_item in item.items()}
                               for t, item in traded_prices.items()}
        df_candlestick = pd.DataFrame.from_dict(traded_prices_float,
                                                orient="index"
                                                )
        df_candlestick.index.name = "time"
        self.logger.info(f"Option candlestick to plot: {df_candlestick.head()}")

        candlestick_filename = f"{self.output_dir}/candlestick_{str(self)}.png"
        plot_candlestick_derivative(df=df_candlestick, output_file=candlestick_filename, title=self.name)

    def sod(self):
        if self.status == "active":
            super().sod()

            if self.eod_status == "open":
                theoretical_price = self.get_theoretical_price()
                self.traded_prices = {0: {"open": Price(theoretical_price),
                                          "low": Price(theoretical_price),
                                          "high": Price(theoretical_price),
                                          "close": Price(theoretical_price),
                                          "theoretical": Price(theoretical_price),
                                          "volume": 0, }}

                self.logger.info(f"Option SoD completed for day: {self.current_day}")


    def eod(self):
        if self.eod_status == "closed":
            return
        elif self.eod_status == "open":
            self.exercise()
            super().eod()

            self.fill_theoretical_price()
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
                "theoretical": self.get_theoretical_price(as_of_day=self.current_day-1),
            }])

            self.repository.save_eod_prices(eod_prices_df=eod_prices_df)

            if self.status != "active":
                self.cleanup()

            self.logger.info(f"Option EoD completed for day: {self.current_day-1}")

    def exercise(self) -> Price:
        # exercise the option (first check if it's the expiry day)
        # return the premium
        premium = 0
        # check if this option should be exercised and if so, then
        # exercise the option - yet only European are served (as for American
        if self.option_type == "European":
            if self.expiration == self.current_day:
                self.logger.info(f"Day of European option expiry {self.name}")
                # ITM or OTM?
                premium = 0
                if self.option_side == "CALL":
                    if self.underlying.last_traded_price > self.strike:
                        premium = self.underlying.last_traded_price - self.strike
                    else:
                        premium = 0
                elif self.option_side == "PUT":
                    if self.underlying.last_traded_price < self.strike:
                        premium = self.strike - self.underlying.last_traded_price
                    else:
                        premium = 0
                premium = Price(premium)
                # for all holders transfer the cash and null out the positions
                for agent_id, agent in self.agents.items():
                    final_position = agent.position[self.asset_id]
                    if final_position != 0:
                        agent.update_position(quantity= -final_position,
                                              cash=premium*final_position,
                                              asset_id=self.asset_id)
                        agent.portfolio_value += premium * final_position
                        # TODO: this should be done by an agent method, but yet record_valuation
                        # TODO: is working only for intraday trades

                option_expiration = { "day": [self.current_day],
                                      "asset_id": [self.asset_id],
                                      "exercise_price": [self.underlying.last_traded_price],
                                      "premium": [premium],}
                option_expiration_df = pd.DataFrame.from_dict(option_expiration, orient="columns")
                self.repository.save_option_expiration(option_expiration_df=option_expiration_df)
                # move the security to non-tradable as expired
                # TODO: self.status = ""
                self.logger.info(f"Setting option {self.asset_id} as expired.")
                self.status = "expired"
        else:
            raise NotImplementedError(f"{self.option_type} not implemented")

        return premium
