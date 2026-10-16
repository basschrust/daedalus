from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import math
from typing import TYPE_CHECKING
from collections import defaultdict
from loguru import logger
import pandas as pd
from pathlib import Path

from marketsim.loggers.basic import setup_logger, setup_agent_logger
from marketsim.input import config
from marketsim.utils.id_generator import id_generator
from marketsim.loggers.basic import terminal
from marketsim.plot.simple_plot import plot_agent_history_many_markets
from marketsim.market.price import Price

if TYPE_CHECKING:
    from marketsim.fourheap import MatchedOrder
    from marketsim.market import Security, Future
    from database.connectors.duckdb_storage import Repository



def validate_update(quantity: int, cash: Price) -> None:
    if not math.isfinite(cash):
        raise ValueError(f"cash must be finite (not NaN or ±inf) as here: {cash}")

    if quantity <= 0:
        if cash < 0:
            raise ValueError("Cash cannot be negative if quantity is negative!")

    if quantity >= 0:
        if cash > 0:
            raise ValueError("Cash cannot be positive if quantity is positive!")


class Agent(ABC):
    # An agent is an investor operating on single market (investing in single security against their cash)

    def __init__(self, *, markets: list[Security], repository: Repository, group: str | None = None,
                 name: str | None = None, configuration: dict|None, output_dir: Path|str="tmp"):
        self.agent_id = id_generator.next()

        self.group = group # previously group_name
        self.name = name if name is not None else f"{self.agent_id}_{self.group}"
        self.configuration = configuration

        self.markets = { market.asset_id: market for market in markets }  # converting to dict
        self.abandoned_markets = {}

        self.trade_history = {}  # dict of lists/dicts {time: [trades over that day, volume bought, volume sold]}

        self.portfolio_value = Price(0)
        self.portfolio_value_history = defaultdict(Price)

        self.position = { m_id: 0 for m_id  in self.markets }
        # for the MtM daily contracts:
        self.futures_position = { m_id: 0 for m_id, m
                                  in self.markets.items() if m.instrument_class == "future" }
        self.futures_value = {
            m_id: {
                "avg_initial_price": Price(0),
                "settlement_price": Price(0),
                "collateral": Price(0),
                "unsettled_pnl": Price(0),
            }
            for m_id in self.futures_position
        }
        # self.futures_value = {}  # asset_id -> current unsettled P&L


        self.position_history = defaultdict(dict)
        self.position_history[0] = { m_id:0  for m_id  in self.markets }  # valid one day! {time: {asset_id: number_of_shares}}
                # at the end of tick
        self.position_history_df = None
        self.cash = Price(0)
        self.cash_history = defaultdict(Price)

        self.repository = repository

        self.eod_status = "open" # open/closed  to make eod procedure idempotent
        self.current_day = 0

        self.output_dir = output_dir
        # setup_logger(subject="agent", subject_id=self.agent_id)
        self.logger = setup_agent_logger(agent_id=self.agent_id, output_dir=self.output_dir) #logger.bind(agent_id=self.agent_id)

        self.repository.save_agent(self)

    ####### init ends here  ###########

    @property
    def cash(self):
        return self._cash

    @cash.setter
    def cash(self, value):
        # print(f"Agent {id(self)} cash: {self._cash} -> {value}")
        # traceback.print_stack(limit=2)
        self._cash = value

    @abstractmethod
    def get_id(self) -> int:
        return self.agent_id

    @abstractmethod
    def take_action(self, current_time: int) -> None:
        # for each of its markets put the orders of the agent
        pass

    @abstractmethod
    def get_pos_value(self) -> float:
        pass

    def update_position(self, quantity: int, cash: Price, asset_id: int) -> None:
        validate_update(quantity=quantity, cash=cash)
        if asset_id not in self.position:
            raise ValueError(f"Position:  {self.position}. {asset_id} not in position")
        self.position[asset_id] += quantity
        self.cash += cash

    def update_futures_position(self, *,
                                quantity: int,
                                contract: Future,
                                ) -> None:
        # update the derivatives position of the agent
        if contract.asset_id not in self.futures_position:
            raise ValueError(f"Position:  {self.futures_position}. {contract.asset_id} not in position")
        current_position = self.futures_position[contract.asset_id]
        if current_position == 0:
            # initiating position on this contract
            collateral = contract.last_traded_price * contract.margin_rate * contract.multiplier
            self.futures_value[contract.asset_id] = {
                "avg_initial_price": contract.last_traded_price, # the current price
                "settlement_price": contract.last_traded_price,
                "collateral": collateral,
                "unsettled_pnl": Price(0),
            }
        elif current_position == - quantity:
            # squaring /closing the position
            freed_collateral = self.futures_value[contract.asset_id]["collateral"]
            self.cash += freed_collateral
            # TODO: depending on the price, there might be some unsettled_pnl to add to cash?
            self.futures_position[contract.asset_id] = 0
            self.futures_value[contract.asset_id] = {
                "avg_initial_price": contract.last_traded_price,  # the current price
                "settlement_price": contract.last_traded_price,
                "collateral": Price(0),
                "unsettled_pnl": Price(0),
            }
        else:
            existing_collateral = self.futures_value[contract.asset_id]["collateral"]
            target_collateral = contract.last_traded_price * contract.margin_rate * contract.multiplier
            # changing contract position - keeping it open with different size
            self.futures_position[contract.asset_id] += quantity
            self.cash -= target_collateral - existing_collateral

            total_pos = current_position + quantity
            old_price = self.futures_value[contract.asset_id]["avg_initial_price"]
            avg_initial_price = (old_price * current_position + contract.last_traded_price * quantity) / total_pos
            # TODO: calculate this
            self.futures_value[contract.asset_id] = {
                "avg_initial_price": avg_initial_price,  # the current price
                "settlement_price": contract.last_traded_price,
                "collateral": target_collateral,
                "unsettled_pnl": Price(0),
            }

    def reset(self) -> None:
        self.position = { m_id: 0 for m_id  in self.markets}
        self.cash = Price(0)

    def is_market_maker(self) -> bool:
        raise # this is utterly deprecated
        return False

    def record_valuation(self, current_time: int) -> None:
        # saving value of agents portfolio

        # valuation by last trade:
        self.portfolio_value = self.cash
        for asset_id, market in self.markets.items():
            self.portfolio_value += int(self.position[asset_id]) * market.last_traded_price

        # TODO: add futures - collateral + unsettled_pnl
        for contract_id, ft_val in self.futures_value.items():
            self.portfolio_value += ft_val["collateral"]
            self.portfolio_value += ft_val["unsettled_pnl"]

        # save valuation:
        self.portfolio_value_history[current_time] = self.portfolio_value
        self.position_history[current_time] = copy.deepcopy(self.position)
        self.cash_history[current_time] = self.cash

    def record_trade(self, matched_order: MatchedOrder) -> None:
        quantity = matched_order.order.order_type * matched_order.order.quantity
        cash = - Price(matched_order.price * matched_order.order.quantity * matched_order.order.order_type)

        self.update_position(quantity=quantity, cash=cash, asset_id=matched_order.order.asset_id)
        self.position_history[matched_order.time][matched_order.order.asset_id] = (
                self.position_history.get(matched_order.time, {}).get(matched_order.order.asset_id, 0) + quantity)

        if matched_order.time in self.trade_history:
            # just add info
            old = self.trade_history.get(matched_order.time, {}).get(matched_order.order.asset_id, {})
            if old:
                self.trade_history[matched_order.time][matched_order.order.asset_id] = {"trades": old["trades"]+1,
                                                               "volume": old["volume"] + abs(matched_order.order.quantity),}
            else:
                # first trade on this security on this date:
                self.trade_history[matched_order.time][matched_order.order.asset_id] =\
                            {"trades": 1, "volume": abs(matched_order.order.quantity),
                                                }
        else:
            # first trade this day
            self.trade_history[matched_order.time] = { matched_order.order.asset_id:
                                                {"trades": 1, "volume": abs(matched_order.order.quantity),
                                                 } }


    def record_futures_trade(self, matched_order: MatchedOrder) -> None:
        # TODO: fill this properly for trading futures
        quantity = matched_order.order.order_type * matched_order.order.quantity
        #cash = - Price(matched_order.price * matched_order.order.quantity * matched_order.order.order_type)
        # cash to collateral, both sides - so no matter if bought or sold
        contract_id = matched_order.order.asset_id
        contract = self.markets[contract_id]
        # make sure the key will exist now:

        self.update_futures_position(quantity=quantity, contract=contract)
        self.position_history[matched_order.time][contract_id] = (
                self.position_history.get(matched_order.time, {}).get(matched_order.order.asset_id, 0) + quantity)

        if matched_order.time in self.trade_history:
            # just add info
            old = self.trade_history.get(matched_order.time, {}).get(matched_order.order.asset_id, {})
            if old:
                self.trade_history[matched_order.time][matched_order.order.asset_id] = {"trades": old["trades"]+1,
                                                               "volume": old["volume"] + abs(matched_order.order.quantity),}
            else:
                # first trade on this security on this date:
                self.trade_history[matched_order.time][matched_order.order.asset_id] =\
                            {"trades": 1, "volume": abs(matched_order.order.quantity),
                                                }
        else:
            # first trade this day
            self.trade_history[matched_order.time] = { matched_order.order.asset_id:
                                                {"trades": 1, "volume": abs(matched_order.order.quantity),
                                                 } }

    def settle(self, contract):
        # final day settlement
        # "avg_initial_price": avg_initial_price,  # the current price
        # "settlement_price": contract.last_traded_price,
        # "collateral": target_collateral,
        # "unsettled_pnl": Price(0),
        ft_val = self.futures_value[contract.asset_id]
        self.cash += ft_val["collateral"] + ft_val["unsettled_pnl"]
        # what about settlement_price?
        self.futures_position[contract.asset_id] = 0
        self.futures_value.pop(contract.asset_id) # delete the contract from analysis
        self.abandoned_markets[contract.asset_id] = contract
        self.markets.pop(contract.asset_id)
        # TODO: porftolio value



    def sod(self):
        self.logger.info(f"Starting agent {self.agent_id} Start-of-Day procedure of day: {self.current_day}")

        self.portfolio_value_history = defaultdict(Price)
        self.cash_history = defaultdict(Price)

        # drop expired securities from portfolio
        expired_securities = { m_id: market for m_id, market in self.markets.items()
                               if market.status == "expired" }
        # or not "active" ?
        for m_id, market in expired_securities.items():
            self.markets.pop(m_id)
            self.abandoned_markets[m_id] = market

        self.logger.info(f"After removing expired securities agent will be active on the following markets:")
        self.logger.info(self.markets)

        self.position_history = defaultdict(dict)
        self.position_history[0] = { m_id: 0 for m_id in self.markets }

        self.eod_status = "open"
        self.logger.info(f"Start-of-Day agent {self.agent_id} procedure of day: {self.current_day} completed.")

    def eod(self) -> None:
        # End Of Day procedure of the agent
        # omnipotent, so once called by any market, makes summaries of his all structures
        # and makes a mark that it has been done
        if self.eod_status == "closed":
            return
        elif self.eod_status == "open":
            self.logger.info(f"Starting agent {self.agent_id} EoD procedure of day: {self.current_day}")

            self.eod_status = "closed"
            self.current_day += 1

            # make position history a DF to enable quick filtering
            # self.position_history
            self.position_history_df = (
                    pd.DataFrame.from_dict(self.position_history, orient="index")
                        .rename_axis("time_tick")
                        .reset_index()
                        .melt(
                            id_vars="time_tick",
                            var_name="asset_id",
                            value_name="position",
                        )
                    )

            self.position_history_df["agent_id"] = self.agent_id
            self.position_history_df["day"] = self.current_day - 1

            self.logger.info(f"Position_history_df: {self.position_history_df.head()}")
            self.repository.save_position_history(position_history_df=self.position_history_df)

            position_df = pd.DataFrame(
                self.position.items(),
                columns=["asset_id", "position"],
            )
            position_df["agent_id"] = self.agent_id
            position_df["day"] = self.current_day - 1
            self.repository.save_eod_position(eod_position_df=position_df)

            # portfolio_value_history:
            # by SQL or in Python?

            value_history_df = pd.DataFrame.from_dict(self.portfolio_value_history, orient="index", columns=["portfolio_value"])
            value_history_df["time_tick"] = value_history_df.index
            value_history_df["day"] = self.current_day -1
            value_history_df["agent_id"] = self.agent_id
            self.logger.info(f"Portfolio value history: {value_history_df.head()}")

            self.repository.save_portfolio_value_history(portfolio_value_history_df=value_history_df)

            # cash history
            cash_history_df = pd.DataFrame.from_dict(self.cash_history, orient="index", columns=["cash"])
            cash_history_df["time_tick"] = cash_history_df.index
            cash_history_df["day"] = self.current_day - 1
            cash_history_df["agent_id"] = self.agent_id

            self.logger.info(f"Cash history: {cash_history_df.head()}")

            self.repository.save_cash_history(cash_history_df=cash_history_df)

            # save eod_cash and eod_portfolio_value
            eod_cash_df = pd.DataFrame([{ "day": self.current_day-1,
                                                        "agent_id": self.agent_id,
                                                        "cash": self.cash,
                                                        }])
            self.logger.info(f"eod_cash_df: {eod_cash_df.head()}")
            self.repository.save_eod_cash(eod_cash_df=eod_cash_df)

            # calculate and save portfolio value
            self.portfolio_value = self.cash  # cash already includes today's option premiums
            for asset_id, market in self.markets.items():
                self.portfolio_value += self.position[asset_id] * market.last_traded_price

            eod_portfolio_value_df = pd.DataFrame([{ "day": self.current_day-1,
                                                          "agent_id": self.agent_id,
                                                          "portfolio_value": self.portfolio_value,
                                                                   }])
            self.logger.info(f"eod_portfolio_value_df: {eod_portfolio_value_df.head()}")
            self.repository.save_eod_portfolio_value(eod_portfolio_value_df=eod_portfolio_value_df)

            self.logger.info(f"EoD agent {self.agent_id} procedure of day: {self.current_day-1} completed.")
        else:
            raise ValueError(f"Unknown eod status: {self.eod_status}")

    def show_summary(self):
        agent_output_file = f"{self.output_dir}/agents_multimarket/{self.agent_id}_{str(self)}.png"

        positions_df = self.repository.get_eod_positions(agent_id=self.agent_id)
        positions_df.rename(columns={"day": "time_step"}, inplace=True)

        # TODO: for futures take from futures_position

        self.logger.info(f"Plotting eod positions for {self.agent_id} ...")
        # self.logger.info(f"{positions_df.head(100)}")
        eod_cash_df = self.repository.get_eod_cash(agent_id=self.agent_id)
        eod_cash_df.rename(columns={"day": "time_step"}, inplace=True)

        eod_portfolio_values_df = self.repository.get_eod_portfolio_values(agent_id=self.agent_id)
        eod_portfolio_values_df.rename(columns={"day": "time_step"}, inplace=True)

        labels_map = { asset_id: security.short_name for asset_id, security
                       in self.markets.items() }
        for m_id, market in self.abandoned_markets.items():
            labels_map[m_id] = market.short_name

        plot_agent_history_many_markets(position_history=positions_df,
                                        cash_history=eod_cash_df,
                                        value_history=eod_portfolio_values_df,
                                        output_file=agent_output_file,
                                        title=f"Agent {self.agent_id} {str(self)} summary",
                                        labels_map=labels_map,
                                        time_label="Simulation day")

    def __str__(self) -> str:
        return f"{self.agent_id}_{self.group}"
