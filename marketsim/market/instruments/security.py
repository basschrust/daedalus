from __future__ import annotations

import math
from abc import ABC
from collections import defaultdict
from datetime import date
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from marketsim.loggers.basic import setup_market_logger
from marketsim.market.fourheap import FourHeap
from marketsim.market.fourheap.event_queue import EventQueue
from marketsim.market.price import Price
from marketsim.plot.candle import plot_candlestick
from marketsim.plot.daedalus_plot_extensions import plot_combined_candles
from marketsim.plot.simple_plot import (
    plot_cash_transfers,
    plot_order_book,
    plot_volume_transfers,
)
from marketsim.utils.id_generator import id_generator

if TYPE_CHECKING:
    from database.connectors.duckdb_storage import Repository

    from marketsim.agent import Agent
    from marketsim.fourheap import MatchedOrder, Order
    from marketsim.market import Index


class Security(ABC):
    def __init__(
        self,
        *,
        reference_price: Price | None = None,
        name: str | None = None,
        repository: Repository,
        market_type: str = "discrete",
        instrument_class: str = "stock",
        short_name: str | None = None,
        output_dir: str | Path = "tmp",
        currency: str = "PLN",
        historical_ohlcv: pd.DataFrame | None = None,
        simulation_dates: list[date] | None = None,
    ) -> None:
        self.asset_id = id_generator.next()
        self.instrument_class = instrument_class
        self.is_tradable = True
        if short_name is None:
            self.short_name = f"{self.instrument_class} {self.asset_id}"
        else:
            self.short_name = short_name
        self.reference_price = Price(reference_price) if reference_price is not None else Price(100)
        self.last_traded_price = self.reference_price

        self.agent_groups = set()

        # TODO: check if this fundamental (externally provided value) is needed here
        # self.fundamental = fundamental

        self.order_book = self._create_order_book()
        self.event_queue = EventQueue()

        self.market_type = (
            market_type  # "discrete" or "continuous" # TODO: what if two phased? or more phased :)
        )
        self.agents = {}
        self.name = name

        self.logger = setup_market_logger(asset_id=self.asset_id, output_dir=output_dir)

        self.current_day = 0
        self.eod_status = "closed"  # open/closed  to make eod procedure idempotent
        self.status = "active"
        self.repository = repository
        self.output_dir = output_dir
        self.currency = currency
        self.indices_member = {}  # index_id: Index

        # if running with historical part:
        self.historical_ohlcv = historical_ohlcv
        self.simulation_dates = simulation_dates

        self.completed_daily_closes = []

        self.save()

        #### end of __init__

    def save(self):
        self.repository.save_security(self)

    def add_agents(self, agents: list[Agent] | None) -> None:
        for agent in agents:
            self.logger.info(f"Adding agent {str(agent)} to market {str(self)}")
            self.agents[agent.get_id()] = agent
            self.agent_groups.add(agent.group)

    def add_to_index(self, index: Index):
        self.indices_member[index.index_id] = Index

    def get_fundamental_value(self, current_time: int) -> float:
        return self.fundamental.get_value_at(current_time)

    def get_final_fundamental(self) -> float:
        return self.fundamental.get_final_fundamental()

    def withdraw_all(self, agent_id: int) -> None:
        self.order_book.withdraw_all(agent_id=agent_id)

    def clear_market(self, current_time: int) -> list[MatchedOrder]:
        newly_matched_orders = self.order_book.market_clear(
            current_time=current_time, trading_phase="continuous"
        )
        self.matched_orders += newly_matched_orders
        for matched_order in newly_matched_orders:
            inner_order = matched_order.order
            self.matched_orders_hashed[inner_order.order_id] = matched_order
            # the limit from the order is needed, not the executed price
        return newly_matched_orders

    def add_orders(self, orders: list[Order]) -> None:
        for order in orders:
            self.event_queue.schedule_activity(order)
            if order.order_type == 1:
                self.orders_by_agent_type[self.agents[order.agent_id].group]["count_buy"] += 1
                self.orders_by_agent_type[self.agents[order.agent_id].group]["volume_buy"] += (
                    order.quantity
                )
            elif order.order_type == -1:
                self.orders_by_agent_type[self.agents[order.agent_id].group]["count_sell"] += 1
                self.orders_by_agent_type[self.agents[order.agent_id].group]["volume_sell"] += (
                    order.quantity
                )

    def cancel_outdated_orders(self, current_time: int | None = None):
        if current_time is None:
            current_time = self.current_time
        self.order_book.cancel_outdated_orders(current_time=current_time)

    # TODO: move to sod()
    def roll_traded_prices(self, current_time: int) -> None:
        previous_tick_prices = self.traded_prices[current_time - 1]
        self.traded_prices[current_time] = {
            "open": previous_tick_prices["close"],
            "low": previous_tick_prices["close"],
            "high": previous_tick_prices["close"],
            "close": previous_tick_prices["close"],
            "volume": 0,
        }

    def step(self, current_time: int) -> list[MatchedOrder]:
        # step is operations happening during single time_tick
        # TODO Need to figure out how to handle ties for price and time - AK: maybe fractal time?
        self.logger.info(f"Starting step for time tick: {str(current_time)}")
        if self.status != "active":
            self.logger.info(f"Security {self.name} is inactive!")
            return []

        # first:cancel orders that are no longer valid
        self.cancel_outdated_orders(current_time=current_time)

        # second: rolling the traded_prices just for one tick... so not in sod()
        if current_time - 1 in self.traded_prices and current_time not in self.traded_prices:
            self.roll_traded_prices(current_time=current_time)

            for g1 in self.agent_groups:
                for g2 in self.agent_groups:
                    key = (current_time, g1, g2)
                    self.trade_stats[key] = {
                        "count_buy": 0,
                        "volume_buy": 0,
                        "cash_buy": 0,
                        "count_sell": 0,
                        "volume_sell": 0,
                        "cash_sell": 0,
                    }

        # taking the orders from queue to LOB:
        orders = self.event_queue.get_activities(current_time=current_time)
        self.buy_init_volume, self.sell_init_volume = 0, 0
        newly_matched_orders = []
        self.bid_ask_history.setdefault(
            current_time,
            [self.order_book.buy_unmatched.peek(), self.order_book.sell_unmatched.peek()],
        )
        self.logger.info(
            f"Starting step {current_time}. Current spread is: {self.order_book.buy_unmatched.peek()} {self.order_book.sell_unmatched.peek()}"
        )
        self.logger.info(
            f"Defined by orders: buy: {self.order_book.buy_unmatched.heap[0][1] if not self.order_book.buy_unmatched.is_empty() else '<None>'}"
            f", sell: {self.order_book.sell_unmatched.heap[0][1] if not self.order_book.sell_unmatched.is_empty() else '<None>'}"
        )
        self.logger.info(
            f"With volumes: buy: {self.order_book.buy_unmatched.peek_order()}"
            f", sell: {self.order_book.sell_unmatched.peek_order()}"
        )
        # plot the order book state here - first just print it:
        self.logger.info(f"The LOB buy orders: {self.order_book.buy_unmatched.heap}")
        self.logger.info(f"The LOB sell orders: {self.order_book.sell_unmatched.heap}")

        for order in orders:
            if order.quantity <= 0:
                continue
            if order.executed_price is not None or order.matched_with is not None:
                raise AssertionError(
                    f"Previously matched order re-entering book: "
                    f"day={self.current_day}, tick={current_time}, "
                    f"order={order}"
                )
            self.logger.info(f"Inserting order: {order}")
            self.order_book.insert(order)
            # if we are in continuous mode we should clear the market here, after entering each order
            # let's see what happens ...
            if self.market_type == "continuous":
                newly_matched_orders += self.clear_market(current_time=current_time)

        # after all orders have been inserted into LOB the cleraing procedure should start in the "fixing" phase
        newly_matched_orders_2 = self.clear_market(current_time=current_time)
        newly_matched_orders += newly_matched_orders_2
        if newly_matched_orders_2:
            # raise # currently we're in continuous only - so here no order should be matched
            # but it reaches this point :/
            self.logger.info(f"Should not reach this point, matched: {newly_matched_orders}")

        # Compute midprices. AK - in continuous mode it may need a change
        self.order_book.update_midprice(current_time=current_time)
        return newly_matched_orders

    def get_midprices(self) -> list:
        return self.order_book.midprices

    def reset(self) -> None:
        self.logger.info("Resetting market...")
        self.order_book = FourHeap(plus_one=True, market=self)
        self.event_queue = EventQueue()
        self.matched_orders = []

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
        else:
            # enter as first day in this time tick
            self.traded_prices[current_time] = {
                "open": price,
                "low": price,
                "high": price,
                "close": price,
                "volume": volume,
            }
        # TODO: update indices which this security is a component of

    def record_trade(self, matched_order: MatchedOrder) -> None:
        self.last_traded_price = matched_order.price

        # record for plots and summary:
        current_time = matched_order.time
        self.record_volume_and_price(matched_order=matched_order)

        # record for each agent:
        agent_id = matched_order.order.agent_id
        self.agents[agent_id].record_trade(matched_order=matched_order)
        # record it by type:
        cp_agent = self.matched_orders_hashed[matched_order.order.matched_with].order.agent_id
        cp_group = self.agents[cp_agent].group
        key = (current_time, self.agents[matched_order.order.agent_id].group, cp_group)

        if matched_order.order.order_type == 1:
            self.trades_by_agent_type[self.agents[matched_order.order.agent_id].group][
                "count_buy"
            ] += 1
            self.trades_by_agent_type[self.agents[matched_order.order.agent_id].group][
                "volume_buy"
            ] += matched_order.order.quantity
            # and the ext version - filling the waited/arrived value - why not use a pandas DF?
            self.trades_by_agent_type_ext[self.agents[matched_order.order.agent_id].group][
                "count_buy"
            ][matched_order.order.executed_mode] += 1
            self.trades_by_agent_type_ext[self.agents[matched_order.order.agent_id].group][
                "volume_buy"
            ][matched_order.order.executed_mode] += matched_order.order.quantity
            # and with information about counterparty group:
            self.trade_history_by_groups[self.agents[matched_order.order.agent_id].group][cp_group][
                "count_buy"
            ][matched_order.order.executed_mode] += 1
            self.trade_history_by_groups[self.agents[matched_order.order.agent_id].group][cp_group][
                "volume_buy"
            ][matched_order.order.executed_mode] += matched_order.order.quantity

            # use the power of Pandas DFs (later in conversion:) ):
            if key not in self.trade_stats:
                self.trade_stats[key] = {
                    "count_buy": 1,
                    "volume_buy": matched_order.order.quantity,
                    "cash_buy": matched_order.order.quantity * matched_order.price,
                    "count_sell": 0,
                    "volume_sell": 0,
                    "cash_sell": 0,
                }
            else:
                self.trade_stats[key]["count_buy"] += 1
                self.trade_stats[key]["volume_buy"] += matched_order.order.quantity
                self.trade_stats[key]["cash_buy"] += (
                    matched_order.order.quantity * matched_order.price
                )

        elif matched_order.order.order_type == -1:
            self.trades_by_agent_type[self.agents[matched_order.order.agent_id].group][
                "count_sell"
            ] += 1
            self.trades_by_agent_type[self.agents[matched_order.order.agent_id].group][
                "volume_sell"
            ] += matched_order.order.quantity
            self.trades_by_agent_type_ext[self.agents[matched_order.order.agent_id].group][
                "count_sell"
            ][matched_order.order.executed_mode] += 1
            self.trades_by_agent_type_ext[self.agents[matched_order.order.agent_id].group][
                "volume_sell"
            ][matched_order.order.executed_mode] += matched_order.order.quantity
            # split by ccp:
            self.trade_history_by_groups[self.agents[matched_order.order.agent_id].group][cp_group][
                "count_sell"
            ][matched_order.order.executed_mode] += 1
            self.trade_history_by_groups[self.agents[matched_order.order.agent_id].group][cp_group][
                "volume_sell"
            ][matched_order.order.executed_mode] += matched_order.order.quantity

            # use the power of Pandas DFs (later in conversion:) ):
            if key not in self.trade_stats:
                self.trade_stats[key] = {
                    "count_buy": 0,
                    "volume_buy": 0,
                    "cash_buy": 0,
                    "count_sell": 1,
                    "volume_sell": matched_order.order.quantity,
                    "cash_sell": matched_order.order.quantity * matched_order.price,
                }
            else:
                self.trade_stats[key]["count_sell"] += 1
                self.trade_stats[key]["volume_sell"] += matched_order.order.quantity
                self.trade_stats[key]["cash_sell"] += (
                    matched_order.order.quantity * matched_order.price
                )
        else:
            raise ValueError(f"Unknown order type {matched_order.order.order_type}")

    def __str__(self) -> str:
        return f"Security_{self.asset_id}_{self.instrument_class}_{self.short_name}"

    ## plotting and supporting functions     #######################

    def aggregate_order_queue(
        self, order_queue: dict, reverse: bool = False, cumulative: bool = False
    ):
        aggregated = defaultdict(int)

        for price, order_id in order_queue:
            buy_order = self.order_book.buy_unmatched.order_dict.get(order_id)
            sell_order = self.order_book.sell_unmatched.order_dict.get(order_id)

            quantity = (buy_order.quantity if buy_order is not None else 0) + (
                sell_order.quantity if sell_order is not None else 0
            )
            aggregated[abs(price)] += quantity

        items = sorted(aggregated.items(), reverse=reverse)

        if cumulative:
            volumes = list(accumulate(volume for _, volume in items))
            items = [(price, volume) for (price, _), volume in zip(items, volumes)]

        return dict(items)

    def plot_lob(self, current_time: int):
        if not self.is_tradable:
            self.logger.info(f"Not tradable instrument {self.name}, skipping LOB plotting.")
            return
        bids = self.aggregate_order_queue(order_queue=self.order_book.buy_unmatched.heap)
        asks = self.aggregate_order_queue(order_queue=self.order_book.sell_unmatched.heap)

        if not bids and not asks:
            self.logger.debug(f"Skipping empty LOB plot for {self.name}")
            return

        self.logger.info(f"Bids: {bids}")
        self.logger.info(f"Asks: {asks}")

        plot_order_book(
            bids=bids,
            asks=asks,
            output_file=f"{self.output_dir}/{str(self)}/LOB/LOB_{self.asset_id}_{current_time}.png",
            title=f"Order book at {current_time}",
        )

    def calculate_realized_volatility(
        self,
        window: int = 20,
    ) -> dict:
        """
        Calculate rolling realized volatility from closing prices.

        Returns:
            {time_tick: realized_volatility}
        """
        times = sorted(self.traded_prices)

        # Close price for each tick
        closes = {t: float(self.traded_prices[t]["close"]) for t in times}

        # Log returns
        returns = {}
        previous_price = None

        for t in times:
            price = closes[t]
            if previous_price is not None and previous_price > 0 and price > 0:
                returns[t] = math.log(price / previous_price)
            previous_price = price

        # Rolling realized volatility
        volatility = {}
        return_times = sorted(returns)

        for i, t in enumerate(return_times):
            window_returns = list(returns.values())[max(0, i - window + 1) : i + 1]

            volatility[t] = math.sqrt(sum(r * r for r in window_returns))

        return volatility

    def plot_trade_stats(self):
        self.trade_stats_df = pd.DataFrame(
            [
                {
                    "time_tick": timeTick,
                    "agent_group": agentGroup,
                    "cp_group": cpGroup,
                    **values,
                }
                for (timeTick, agentGroup, cpGroup), values in self.trade_stats.items()
            ]
        )

        self.logger.info(f"Volume transfers: {self.trade_stats_df.head(30)}")
        plot_volume_transfers(
            self.trade_stats_df,
            output_file_tpl=f"{self.output_dir}/{str(self)}/Transfers_vol_{str(self)}_",
        )

        # TODO: plot cash transfers
        plot_cash_transfers(
            self.trade_stats_df,
            output_file_tpl=f"{self.output_dir}/{str(self)}/Transfers_cash_{str(self)}_",
        )

    def plot_history(self, traded_prices: dict) -> None:
        traded_prices_float = {
            t: {
                field: float(value) if value is not None else float("nan")
                for field, value in prices.items()
            }
            for t, prices in traded_prices.items()
        }
        df_candlestick = pd.DataFrame.from_dict(traded_prices_float, orient="index")
        df_candlestick.index.name = "time"
        self.logger.info(df_candlestick.head())

        candlestick_filename = f"{self.output_dir}/candlestick_{str(self)}.png"
        plot_candlestick(df=df_candlestick, output_file=candlestick_filename, title=self.name)

    def show_summary(self):
        self.logger.info(f"\n\nMarket {str(self)} summary:")
        # self.logger.info(f"Orders matched: {len(self.matched_orders)}")
        self.logger.info(f"Last traded price: {self.last_traded_price}")

        # plot the security values history for all period - take data from DB:
        # TODO: for one day simulations plot the daily candlestick

        # Get simulated daily OHLCV from repository
        eod_prices_df = self.repository.get_eod_prices(asset_id=self.asset_id)

        candlestick_filename = Path(self.output_dir) / f"candlestick_{self}.png"

        if self.historical_ohlcv is not None and self.simulation_dates is not None:
            plot_combined_candles(
                historical=self.historical_ohlcv,
                simulated=eod_prices_df,
                simulation_dates=self.simulation_dates,
                output_file=str(candlestick_filename),
                title=self.name,
            )
        else:
            # Preserve existing behavior
            eod_prices = eod_prices_df.set_index("day").to_dict(orient="index")
            self.plot_history(traded_prices=eod_prices)

        # plotting by type - TODO: take overall data from DB:
        # plot_by_type(self.orders_by_agent_type,
        #              output_file=f"{config.output_dir}/{str(self)}/orders_by_type_{str(self)}.png",
        #              title=f"Orders by type in {self.name}")
        # plot_by_type(self.trades_by_agent_type,
        #              output_file=f"{config.output_dir}/{str(self)}/trades_by_type_{str(self)}.png",
        #              title=f"Trades by type in {self.name}")
        # plot_by_type(self.trades_by_agent_type_ext,
        #              output_file=f"{config.output_dir}/{str(self)}/trades_by_type_ext_{str(self)}.png",
        #              title=f"Trades by extended type in {self.name}", mode="extended")
        # plot_bid_ask(self.bid_ask_history,
        #              output_file=f"{config.output_dir}/{str(self)}/bid_ask_history_{str(self)}.png",
        #              title=f"Bid ask spread history {str(self)}")
        # calculate and plot realized volatility:
        # window = 50
        # volatility = self.calculate_realized_volatility(window=window)
        # plot_realized_volatility(volatility=volatility,
        #                          output_file=f"{config.output_dir}/{str(self)}/realized_volatility_{str(self)}.png",
        #                          title=f"Realized volatility {str(self)} with window {window}")

        # plot the history of trading between agent groups:
        # self.plot_trade_stats()

    ##################      SoD  and  EoD      ##########################################

    def sod(self):

        if self.status == "active":
            self.logger.info(
                f"Starting Start-of-Day procedure for {self.name}, day: {self.current_day}"
            )
            self.eod_status = "open"

            self.order_book = self._create_order_book()
            self.event_queue = EventQueue()
            self.matched_orders = []  # stores a list of all trades from the beginning of trading to the end of simulation
            self.matched_orders_hashed = {}  # {order_id: { "price": price, "quantity":quantity }}

            self.bid_ask_history = {}
            self.realized_volatility = {0: 0}
            self.orders_by_agent_type = {}
            self.trades_by_agent_type = {}
            self.trades_by_agent_type_ext = {}
            # by groups - including counterparty groups:
            self.trade_history_by_groups = {}

            self.traded_prices = {
                0: {
                    "open": self.last_traded_price,
                    "low": self.last_traded_price,
                    "high": self.last_traded_price,
                    "close": self.last_traded_price,
                    "volume": 0,
                }
            }

            # and let's use the power of DataFrames:
            self.trade_stats = {}
            self.trade_stats_df = pd.DataFrame()

            for agent_id, agent in self.agents.items():
                # TODO: we can relatively easily make those aggregations in the DB now
                # or at least in the agent's SoD procedure - not, these are attributes of the Security class
                self.orders_by_agent_type.setdefault(
                    agent.group,
                    {"count_buy": 0, "volume_buy": 0, "count_sell": 0, "volume_sell": 0},
                )
                self.trades_by_agent_type.setdefault(
                    agent.group,
                    {"count_buy": 0, "volume_buy": 0, "count_sell": 0, "volume_sell": 0},
                )
                self.trades_by_agent_type_ext.setdefault(
                    agent.group,
                    {
                        "count_buy": {"arrived": 0, "waited": 0},
                        "volume_buy": {"arrived": 0, "waited": 0},
                        "count_sell": {"arrived": 0, "waited": 0},
                        "volume_sell": {"arrived": 0, "waited": 0},
                    },
                )
                # this one is tricky as requires n-square combination
                # TODO: but also with already existing groups!
                # and then by time...

            for g1 in self.agent_groups:
                for g2 in self.agent_groups:
                    self.trade_history_by_groups.setdefault(g1, {}).setdefault(
                        g2,
                        {
                            "count_buy": {"arrived": 0, "waited": 0},
                            "volume_buy": {"arrived": 0, "waited": 0},
                            "count_sell": {"arrived": 0, "waited": 0},
                            "volume_sell": {"arrived": 0, "waited": 0},
                        },
                    )

    def eod(self):
        # End of Day process for the Market - create EoD DataFrames
        # omnipotent, so once called, makes summaries of his all structures
        # and makes a mark that it has been done
        if self.eod_status == "closed":
            return
        elif self.eod_status == "open":
            self.logger.info(f"Starting EoD procedure of day: {self.current_day}")

            self.eod_status = "closed"
            self.current_day += 1
            self.current_time = 0  # TODO: do we ever need it?
            # run the EoD procedure
            # make traded_price a DF to enable quick filtering and joining with agents' positions

            self.traded_prices_df = (
                pd.DataFrame.from_dict(self.traded_prices, orient="index")
                .rename_axis("time_tick")
                .reset_index()[["time_tick", "open", "high", "low", "close", "volume"]]
            )

            self.traded_prices_df["asset_id"] = self.asset_id
            self.traded_prices_df["day"] = self.current_day - 1  #

            if self.instrument_class == "stock":
                self.logger.info(f"Traded_prices_df: {self.traded_prices_df.head()}")

                self.logger.info("Saving traded prices for stock...")
                self.repository.save_traded_prices(self.traded_prices_df)

                # eod_prices
                eod_prices_df = pd.DataFrame(
                    [
                        {
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
                        }
                    ]
                )

                self.repository.save_eod_prices(eod_prices_df=eod_prices_df)

                self.completed_daily_closes.append(float(eod_prices_df["close"].iloc[0]))

            # orders and trades
            orders = []
            for order_id, order in self.order_book.buy_unmatched.order_dict.items():
                orders.append(order)

            for order_id, order in self.order_book.sell_unmatched.order_dict.items():
                orders.append(order)

            orders_df = pd.DataFrame(
                orders,
                columns=[
                    "day",
                    "price",
                    "order_type",
                    "quantity",
                    "agent_id",
                    "time",
                    "order_id",
                    "asset_id",
                    "executed_price",
                    "executed_mode",
                    "parent_id",
                    "matched_with",
                    "valid_until",
                ],
            )
            # TODO: if no orders made on given day we should make empty DF here
            # also, these are only the unmatched orders!
            # TODO: add missing columns, e.g. valid_until
            orders_df["day"] = self.current_day - 1
            self.repository.save_orders(orders_df=orders_df)

            matched_orders = []
            for order_id, matched_order in self.matched_orders_hashed.items():
                matched_orders.append(
                    {
                        "asset_id": matched_order.order.asset_id,
                        "agent_id": matched_order.order.agent_id,
                        "order_id": order_id,
                        "order_side": matched_order.order.order_type,
                        "matched_with": matched_order.order.matched_with,
                        "executed_mode": matched_order.order.executed_mode,
                        "executed_price": matched_order.price,
                        "executed_time": matched_order.time,
                        "executed_volume": matched_order.volume,
                        "cash": matched_order.cash,
                        "phase": matched_order.phase,
                        "day": self.current_day - 1,
                    }
                )
            matched_orders_df = pd.DataFrame(
                matched_orders,
                columns=[
                    "asset_id",
                    "agent_id",
                    "order_id",
                    "order_side",
                    "matched_with",
                    "executed_mode",
                    "executed_price",
                    "executed_time",
                    "executed_volume",
                    "cash",
                    "phase",
                    "day",
                ],
            )
            # adding conversion to powerful numbers to avoid exceeding range in DuckDB:
            matched_orders_df = matched_orders_df.astype(
                {
                    "asset_id": "int64",
                    "agent_id": "int64",
                    "order_id": "int64",
                    "order_side": "int64",
                    "matched_with": "int64",
                    "executed_mode": "object",
                    "executed_price": "float64",
                    "executed_time": "int64",
                    "executed_volume": "int64",
                    "cash": "float64",
                    "phase": "object",
                    "day": "int64",
                }
            )
            self.repository.save_trades(trades_df=matched_orders_df)

            if self.status != "active" and self.instrument_class == "stock":
                # other instruments cleaned up in their subclasses.
                self.cleanup()
            self.logger.info(f"EoD procedure of day: {self.current_day - 1} completed.")
        else:
            raise ValueError(f"Unknown eod status: {self.eod_status}")

    def cleanup(self):
        self.logger.info(f"Cleaning up the security {self.name} ...")
        self.order_book = None  # FourHeap(plus_one=True, market=self)
        self.matched_orders = None  # stores a list of all trades from the beginning of trading to the end of simulation
        self.matched_orders_hashed = None  # {order_id: { "price": price, "quantity":quantity }}

        self.bid_ask_history = None
        self.realized_volatility = None
        self.orders_by_agent_type = None
        self.trades_by_agent_type = None
        self.trades_by_agent_type_ext = None
        # by groups - including counterparty groups:
        self.trade_history_by_groups = None

        self.traded_prices = None

        self.trade_stats = None
        self.trade_stats_df = None

        self.orders_by_agent_type = None
        self.trades_by_agent_type = None
        self.trades_by_agent_type_ext = None
        self.trade_history_by_groups = None

    def _create_order_book(self) -> FourHeap:
        return FourHeap(
            plus_one=True,
            market_id=self.asset_id,
        )
