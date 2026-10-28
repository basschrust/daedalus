from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from marketsim.loggers.basic import setup_agent_logger
from marketsim.market import FXRateProvider
from marketsim.market.knowledge.market_knowledge import MarketKnowledge
from marketsim.market.price import Price
from marketsim.plot.daedalus_plot_extensions import (
    plot_agent_history_multicurrency,
)
from marketsim.utils.id_generator import id_generator

if TYPE_CHECKING:
    from database.connectors.duckdb_storage import Repository

    from marketsim.fourheap import MatchedOrder
    from marketsim.market import Future, Security


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

    def __init__(
        self,
        *,
        markets: list[Security],
        repository: Repository,
        group: str | None = None,
        name: str | None = None,
        configuration: dict | None,
        output_dir: Path | str = "tmp",
        base_currency: str = "PLN",
        fx_rate_provider: FXRateProvider | None = None,
        market_knowledge: dict[int, MarketKnowledge] | None = None,
        sim_time: int | None = None,
    ):
        # TODO: looks stupid, but we'll check later:
        # if self.sim_time is None or self.sim_time <= 0:
        #     raise ValueError(
        #         "Agent sim_time must be initialized by Simulator"
        #     )
        self.sim_time = sim_time
        self.agent_id = id_generator.next()

        self.group = group  # previously group_name
        self.name = name if name is not None else f"{self.agent_id}_{self.group}"
        self.configuration = configuration

        self.markets = {market.asset_id: market for market in markets}  # converting to dict
        self.abandoned_markets = {}

        self.trade_history = {}  # dict of lists/dicts {time: [trades over that day, volume bought, volume sold]}

        self.portfolio_value = Price(0)
        self.portfolio_value_history = defaultdict(Price)

        self.position = {m_id: 0 for m_id in self.markets}  # TODO: stocks or all instruments?
        # for the MtM daily contracts:
        self.futures_position = {
            m_id: 0 for m_id, m in self.markets.items() if m.instrument_class == "future"
        }
        self.futures_value = {
            m_id: {
                "avg_initial_price": Price(0),
                "settlement_price": Price(0),
                "collateral": Price(0),
                "unsettled_pnl": Price(0),
            }
            for m_id in self.futures_position
        }

        # all assets:
        self.position_history = defaultdict(dict)
        self.position_history[0] = {
            m_id: 0 for m_id in self.markets
        }  # valid one day! {time: {asset_id: number_of_shares}}
        # at the end of tick
        self.position_history_df = None  # TODO: is needed as attribute - local var suffices?
        # futures (or derivatives):
        self.futures_position_history = defaultdict(dict)
        self.futures_position_history[0] = {
            m_id: 0 for m_id, m in self.markets.items() if m.instrument_class == "future"
        }
        self.futures_value_history = defaultdict(dict)

        # cash - multicurrency
        self.cash = {
            "USD": Price(0),
            "EUR": Price(0),
            "PLN": Price(0),
        }
        self.cash_history = defaultdict(dict)

        self.repository = repository

        self.eod_status = "open"  # open/closed  to make eod procedure idempotent
        # TODO: start with closed - no trading before sod() is finished !
        self.current_day = 0
        self.current_time = 0

        self.output_dir = output_dir
        self.base_currency = base_currency.upper()
        self.fx_rate_provider = fx_rate_provider

        self.market_knowledge = market_knowledge or {}

        self.logger = setup_agent_logger(agent_id=self.agent_id, output_dir=self.output_dir)

        self.repository.save_agent(self)

    ####### init ends here  ###########

    @property
    def cash(self) -> dict:
        # raise
        return self._cash

    @cash.setter
    def cash(self, value: dict) -> None:
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

    ##################################################################################
    ##              cash manipulation methods
    ##################################################################################

    def get_cash_balance(
        self,
        *,
        currency: str | None = None,
    ) -> Price:
        currency = currency or self.base_currency
        return self.cash.get(currency, Price(0))

    def get_cash_balance_all_currencies(self) -> dict:
        return self.cash

    def credit_cash(self, *, amount: Price, currency: str = "PLN") -> None:
        """Credit cash in the specified currency."""
        amount = Price(amount)

        if amount < 0:
            raise ValueError("Credit amount cannot be negative")

        self.cash[currency] = self.get_cash_balance(currency=currency) + amount

    def debit_cash(
        self,
        *,
        amount: Price,
        currency: str = "PLN",
        allow_negative: bool = True,
    ) -> None:
        """Debit cash in the specified currency."""
        amount = Price(amount)

        if amount < 0:
            raise ValueError("Debit amount cannot be negative")

        balance = self.get_cash_balance(currency=currency)

        if not allow_negative and balance < amount:
            raise ValueError(
                f"Insufficient {currency} cash: available={balance}, required={amount}"
            )

        self.cash[currency] = balance - amount

    def transfer_cash(
        self,
        *,
        recipient: "Agent",
        currency: str,
        amount: Price,
        allow_negative: bool = True,
    ) -> None:
        """Transfer cash between two agents."""
        amount = Price(amount)

        if amount < 0:
            raise ValueError("Transfer amount cannot be negative")

        self.debit_cash(
            currency=currency,
            amount=amount,
            allow_negative=allow_negative,
        )

        recipient.credit_cash(
            currency=currency,
            amount=amount,
        )

    #### position manipulation

    def update_position(
        self, *, quantity: int, cash: Price, asset_id: int, currency: str = "PLN"
    ) -> None:
        validate_update(quantity=quantity, cash=cash)
        if asset_id not in self.position:
            raise ValueError(f"Position:  {self.position}. {asset_id} not in position")
        self.position[asset_id] += quantity
        # self.cash += cash
        if cash > 0:
            self.credit_cash(currency=currency, amount=cash)
        else:
            self.debit_cash(currency=currency, amount=-cash)

    def update_futures_position(
        self,
        *,
        quantity: int,
        contract: Future,
    ) -> None:
        self.logger.info(
            f"Updating futures position agent: {self} contract: {contract} quantity: {quantity}"
        )
        # update the derivatives position of the agent
        if contract.asset_id not in self.futures_position:
            raise ValueError(
                f"Position:  {self.futures_position}. {contract.asset_id} not in position"
            )
        current_position = self.futures_position[contract.asset_id]
        new_position = current_position + quantity
        self.futures_position[contract.asset_id] = new_position
        required_collateral = (
            abs(new_position)
            * contract.last_traded_price
            * contract.multiplier
            * contract.margin_rate
        )
        old_collateral = self.futures_value[contract.asset_id]["collateral"]
        collateral_change = required_collateral - old_collateral
        # self.cash -= collateral_change
        if collateral_change > 0:
            self.debit_cash(currency=contract.currency, amount=collateral_change)
        else:
            self.credit_cash(currency=contract.currency, amount=-collateral_change)
        if current_position == 0:
            # initiating position on this contract
            # collateral = contract.last_traded_price * contract.margin_rate * contract.multiplier
            self.futures_value[contract.asset_id] = {
                "avg_initial_price": contract.last_traded_price,  # the current price
                "settlement_price": contract.last_traded_price,
                "collateral": required_collateral,
                "unsettled_pnl": Price(0),
            }
        elif current_position == -quantity:
            # squaring /closing the position
            # freed_collateral = self.futures_value[contract.asset_id]["collateral"]
            # self.cash += freed_collateral
            # TODO: depending on the price, there might be some unsettled_pnl to add to cash?
            self.futures_position[contract.asset_id] = 0
            self.futures_value[contract.asset_id] = {
                "avg_initial_price": contract.last_traded_price,  # the current price
                "settlement_price": contract.last_traded_price,
                "collateral": Price(0),
                "unsettled_pnl": Price(0),
            }
        else:
            # existing_collateral = self.futures_value[contract.asset_id]["collateral"]
            # target_collateral = contract.last_traded_price * contract.margin_rate * contract.multiplier
            # changing contract position - keeping it open with different size
            # self.futures_position[contract.asset_id] += quantity
            # self.cash -= target_collateral - existing_collateral

            total_pos = current_position + quantity
            old_price = self.futures_value[contract.asset_id]["avg_initial_price"]
            avg_initial_price = (
                old_price * current_position + contract.last_traded_price * quantity
            ) / total_pos
            # TODO: calculate this
            self.futures_value[contract.asset_id] = {
                "avg_initial_price": avg_initial_price,  # the current price
                "settlement_price": contract.last_traded_price,
                "collateral": required_collateral,
                "unsettled_pnl": Price(0),
            }
        self.futures_value_history[self.current_time] = copy.deepcopy(self.futures_value)

    def reset(self) -> None:
        self.position = {m_id: 0 for m_id in self.markets}
        # self.cash = Price(0) # TODO: is it ever needed?

    def is_market_maker(self) -> bool:
        raise  # this is utterly deprecated
        return False

    def record_valuation(self, current_time: int) -> None:
        self.current_time = current_time
        # saving value of agents portfolio

        # valuation by last trade:
        # self.portfolio_value = self.cash
        # TODO: add multiccy valuation
        # self.portfolio_value = self.get_cash_balance()
        # for asset_id, market in self.markets.items():
        #     self.portfolio_value += int(self.position[asset_id]) * market.last_traded_price

        self.portfolio_value = self.calculate_portfolio_value()

        self.portfolio_value_history[current_time] = self.portfolio_value
        self.position_history[current_time] = copy.deepcopy(self.position)
        self.futures_position_history[current_time] = copy.deepcopy(self.futures_position)
        self.cash_history[current_time] = copy.deepcopy(self.cash)

    def record_trade(self, matched_order: MatchedOrder) -> None:
        self.current_time = matched_order.time
        quantity = matched_order.order.order_type * matched_order.order.quantity
        cash = -Price(
            matched_order.price * matched_order.order.quantity * matched_order.order.order_type
        )

        # self.update_position(quantity=quantity, cash=cash, asset_id=matched_order.order.asset_id)
        asset_id = matched_order.order.asset_id
        self.update_position(
            quantity=quantity,
            cash=cash,
            asset_id=asset_id,
            currency=self.markets[asset_id].currency,
        )
        self.position_history[matched_order.time][matched_order.order.asset_id] = (
            self.position_history.get(matched_order.time, {}).get(matched_order.order.asset_id, 0)
            + quantity
        )

        if matched_order.time in self.trade_history:
            # just add info
            old = self.trade_history.get(matched_order.time, {}).get(
                matched_order.order.asset_id, {}
            )
            if old:
                self.trade_history[matched_order.time][matched_order.order.asset_id] = {
                    "trades": old["trades"] + 1,
                    "volume": old["volume"] + abs(matched_order.order.quantity),
                }
            else:
                # first trade on this security on this date:
                self.trade_history[matched_order.time][matched_order.order.asset_id] = {
                    "trades": 1,
                    "volume": abs(matched_order.order.quantity),
                }
        else:
            # first trade this day
            self.trade_history[matched_order.time] = {
                matched_order.order.asset_id: {
                    "trades": 1,
                    "volume": abs(matched_order.order.quantity),
                }
            }

    def record_futures_trade(self, matched_order: MatchedOrder) -> None:
        self.logger.info(f"Recording futures trade on futures {matched_order}")
        # TODO: fill this properly for trading futures
        quantity = matched_order.order.order_type * matched_order.order.quantity
        # cash = - Price(matched_order.price * matched_order.order.quantity * matched_order.order.order_type)
        # cash to collateral, both sides - so no matter if bought or sold
        contract_id = matched_order.order.asset_id
        contract = self.markets[contract_id]
        # make sure the key will exist now:

        self.update_futures_position(quantity=quantity, contract=contract)
        # TODO: the below history update encapsulate in the above function
        self.futures_position_history[matched_order.time][contract_id] = (
            self.futures_position_history.get(matched_order.time, {}).get(
                matched_order.order.asset_id, 0
            )
            + quantity
        )

        if matched_order.time in self.trade_history:
            # just add info
            old = self.trade_history.get(matched_order.time, {}).get(
                matched_order.order.asset_id, {}
            )
            if old:
                self.trade_history[matched_order.time][matched_order.order.asset_id] = {
                    "trades": old["trades"] + 1,
                    "volume": old["volume"] + abs(matched_order.order.quantity),
                }
            else:
                # first trade on this security on this date:
                self.trade_history[matched_order.time][matched_order.order.asset_id] = {
                    "trades": 1,
                    "volume": abs(matched_order.order.quantity),
                }
        else:
            # first trade this day
            self.trade_history[matched_order.time] = {
                matched_order.order.asset_id: {
                    "trades": 1,
                    "volume": abs(matched_order.order.quantity),
                }
            }

    ######### accounting and eod/sod valuations

    def calculate_portfolio_value(self, *, day: int | None = None) -> Price:
        # returns valuation of current portfolio as of given day.
        # but currently works reliably for the current day
        if day is None:
            day = self.current_day
        if self.fx_rate_provider is None:
            raise ValueError("FXRateProvider is required for portfolio valuation")

        total = Decimal("0")

        # Cash balances
        for currency, balance in self.cash.items():
            if balance == 0:
                continue
            # Convert nonzero balance
            total += self.fx_rate_provider.convert(
                Decimal(str(balance)),
                day=day,
                base=currency,
                quote=self.base_currency,
            )

        # Security positions
        for asset_id, quantity in self.position.items():
            # Zero positions do not contribute to portfolio value.
            if quantity == 0:
                continue

            market = self.markets[asset_id]

            # Futures are valued separately through collateral and unsettled P&L.
            if market.instrument_class == "future":
                continue

            native_value = Decimal(quantity) * Decimal(str(market.last_traded_price))

            total += self.fx_rate_provider.convert(
                native_value,
                day=day,
                base=market.currency,
                quote=self.base_currency,
            )

        # Futures: collateral + unsettled P&L
        for asset_id, values in self.futures_value.items():
            native_value = Decimal(str(values["collateral"])) + Decimal(
                str(values["unsettled_pnl"])
            )

            # Zero-value contracts contribute nothing to portfolio value.
            # No market lookup or FX conversion is necessary.
            if native_value == 0:
                continue

            market = self.markets[asset_id]

            total += self.fx_rate_provider.convert(
                native_value,
                day=day,
                base=market.currency,
                quote=self.base_currency,
            )

        return Price(total)

    def settle(self, contract: Future) -> None:
        # final day settlement
        # "avg_initial_price": avg_initial_price,  # the current price
        # "settlement_price": contract.last_traded_price,
        # "collateral": target_collateral,
        # "unsettled_pnl": Price(0),
        ft_val = self.futures_value[contract.asset_id]
        # self.cash += ft_val["collateral"] + ft_val["unsettled_pnl"]
        freed_cash = ft_val["collateral"] + ft_val["unsettled_pnl"]
        if freed_cash > 0:
            self.credit_cash(currency=contract.currency, amount=freed_cash)
        else:
            self.debit_cash(currency=contract.currency, amount=-freed_cash)
        # what about settlement_price?
        self.futures_position[contract.asset_id] = 0
        self.futures_value.pop(contract.asset_id)  # delete the contract from analysis
        self.abandoned_markets[contract.asset_id] = contract
        self.markets.pop(contract.asset_id)
        # TODO: porftolio value

    def settle_futures_mtm(self) -> None:
        """
        Settle daily variation margin for all active futures.

        Each contract is settled against its previous
        settlement reference price.
        """
        for asset_id, position in self.futures_position.items():
            if position == 0:
                continue

            contract = self.markets[asset_id]
            values = self.futures_value[asset_id]

            previous_price = values["settlement_price"]
            current_price = contract.last_traded_price

            vm = Price(position * contract.multiplier * (current_price - previous_price))

            # Daily cash settlement
            # self.cash += vm
            if vm > 0:
                self.credit_cash(amount=vm, currency=contract.currency)
            else:
                self.debit_cash(amount=-vm, currency=contract.currency)

            # Roll the reference price forward
            values["settlement_price"] = current_price

            # Variation margin has been paid
            values["unsettled_pnl"] = Price(0)

    def sod(self) -> None:
        self.logger.info(
            f"Starting agent {self.agent_id} Start-of-Day procedure of day: {self.current_day}"
        )
        self.current_time = 0

        self.portfolio_value_history = defaultdict(Price)
        self.cash_history = defaultdict(dict)

        # drop expired securities from portfolio
        expired_securities = {
            m_id: market for m_id, market in self.markets.items() if market.status == "expired"
        }
        # or not "active" ?
        for m_id, market in expired_securities.items():
            self.markets.pop(m_id)
            self.abandoned_markets[m_id] = market

        self.logger.info(
            "After removing expired securities agent will be active on the following markets:"
        )
        self.logger.info(self.markets)

        self.position_history = defaultdict(dict)
        self.position_history[0] = {m_id: 0 for m_id in self.markets}

        self.futures_position_history[0] = {
            m_id: 0 for m_id, market in self.markets.items() if market.instrument_class == "future"
        }
        self.futures_value_history[0] = {}

        self.eod_status = "open"
        self.logger.info(
            f"Start-of-Day agent {self.agent_id} procedure of day: {self.current_day} completed."
        )

    def eod(self) -> None:
        # End Of Day procedure of the agent
        # omnipotent, so once called by any market, makes summaries of his all structures
        # and makes a mark that it has been done
        if self.eod_status == "closed":
            return
        elif self.eod_status == "open":
            self.logger.info(
                f"Starting agent {self.agent_id} EoD procedure of day: {self.current_day}"
            )

            self.eod_status = "closed"
            self.current_day += 1

            # settle all mtm to know the collateral, cash and PnL
            self.settle_futures_mtm()

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

            # and futures history
            self.futures_position_history_df = (
                pd.DataFrame.from_dict(self.futures_position_history, orient="index")
                .rename_axis("time_tick")
                .reset_index()
                .melt(
                    id_vars="time_tick",
                    var_name="asset_id",
                    value_name="position",
                )
            )

            self.futures_position_history_df["agent_id"] = self.agent_id
            self.futures_position_history_df["day"] = self.current_day - 1

            self.logger.info(
                f"futures_position_history_df: {self.futures_position_history_df.head()}"
            )
            self.repository.save_futures_position_history(
                position_history_df=self.futures_position_history_df
            )

            futures_position_df = pd.DataFrame(
                self.futures_position.items(),
                columns=["asset_id", "position"],
            )
            futures_position_df["agent_id"] = self.agent_id
            futures_position_df["day"] = self.current_day - 1
            self.logger.info("")
            self.repository.save_eod_futures_position(eod_position_df=futures_position_df)

            # saving the values of futures
            rows = []
            for time_tick, contracts in self.futures_value_history.items():
                for asset_id, values in contracts.items():
                    rows.append(
                        {
                            "day": self.current_day - 1,
                            "time_tick": time_tick,
                            "agent_id": self.agent_id,
                            "asset_id": asset_id,
                            "position": self.futures_position_history.get(time_tick, {}).get(
                                asset_id, 0
                            ),
                            **values,
                        }
                    )
            if rows:
                # unbelievable - can this happen for agents not trading futures?
                futures_value_history_df = pd.DataFrame(rows)
                self.repository.save_futures_value_history(
                    futures_value_df=futures_value_history_df
                )

            eod_futures_value_df = pd.DataFrame(
                [
                    {
                        "day": self.current_day - 1,
                        "agent_id": self.agent_id,
                        "asset_id": asset_id,
                        "position": self.futures_position.get(asset_id, 0),
                        **values,
                    }
                    for asset_id, values in self.futures_value.items()
                ]
            )
            if not eod_futures_value_df.empty:
                self.logger.info(f"Save eod_futures_value_df: {eod_futures_value_df}")
                self.repository.save_eod_futures_value(eod_futures_value_df=eod_futures_value_df)

            eod_position_df = pd.DataFrame(
                self.position.items(),
                columns=["asset_id", "position"],
            )
            eod_position_df["agent_id"] = self.agent_id
            eod_position_df["day"] = self.current_day - 1
            self.repository.save_eod_position(eod_position_df=eod_position_df)

            # portfolio_value_history:
            # by SQL or in Python?

            value_history_df = pd.DataFrame.from_dict(
                self.portfolio_value_history, orient="index", columns=["portfolio_value"]
            )
            value_history_df["time_tick"] = value_history_df.index
            value_history_df["day"] = self.current_day - 1
            value_history_df["agent_id"] = self.agent_id
            self.logger.info(f"Portfolio value history: {value_history_df.head()}")

            self.repository.save_portfolio_value_history(
                portfolio_value_history_df=value_history_df
            )

            # cash history
            cash_history_df = pd.DataFrame.from_dict(
                self.cash_history, orient="index", columns=["cash", "currency"]
            )
            cash_history_df["time_tick"] = cash_history_df.index
            cash_history_df["day"] = self.current_day - 1
            cash_history_df["agent_id"] = self.agent_id

            self.logger.info(f"Cash history: {cash_history_df.head()}")

            self.repository.save_cash_history(cash_history_df=cash_history_df)

            # save eod_cash and eod_portfolio_value
            rows = []
            for ccy, amount in self.get_cash_balance_all_currencies().items():
                rows.append(
                    {
                        "day": self.current_day - 1,
                        "agent_id": self.agent_id,
                        "cash": amount,
                        "currency": ccy,
                    }
                )
            eod_cash_df = pd.DataFrame(rows)
            self.logger.info(f"eod_cash_df: {eod_cash_df.head()}")
            self.repository.save_eod_cash(eod_cash_df=eod_cash_df)

            # calculate and save portfolio value
            # self.portfolio_value = self.get_cash_balance() # cash  # cash already includes today's option premiums
            # for asset_id, market in self.markets.items():
            #     self.portfolio_value += self.position[asset_id] * market.last_traded_price
            self.portfolio_value = self.calculate_portfolio_value(day=self.current_day - 1)

            eod_portfolio_value_df = pd.DataFrame(
                [
                    {
                        "day": self.current_day - 1,
                        "agent_id": self.agent_id,
                        "portfolio_value": self.portfolio_value,
                    }
                ]
            )
            self.logger.info(f"eod_portfolio_value_df: {eod_portfolio_value_df.head()}")
            self.repository.save_eod_portfolio_value(eod_portfolio_value_df=eod_portfolio_value_df)

            self.logger.info(
                f"EoD agent {self.agent_id} procedure of day: {self.current_day - 1} completed."
            )
        else:
            raise ValueError(f"Unknown eod status: {self.eod_status}")

    def show_summary(self) -> None:
        agent_output_file = f"{self.output_dir}/agents_multimarket/{self.agent_id}_{str(self)}.png"

        positions_df = self.repository.get_eod_positions(agent_id=self.agent_id)
        positions_df.rename(columns={"day": "time_step"}, inplace=True)

        futures_positions_df = self.repository.get_eod_futures_positions(agent_id=self.agent_id)
        futures_positions_df.rename(columns={"day": "time_step"}, inplace=True)

        # overwriting futures_position as in positions_df they are zeros
        positions_df = (
            pd.concat([futures_positions_df, positions_df], ignore_index=True)
            .drop_duplicates(subset=["time_step", "asset_id"], keep="first")
            .sort_values(["time_step", "asset_id"])
            .reset_index(drop=True)
        )

        self.logger.info(f"Plotting eod positions for {self.agent_id} ...")
        # self.logger.info(f"{positions_df.head(100)}")
        eod_cash_df = self.repository.get_eod_cash(agent_id=self.agent_id)
        eod_cash_df.rename(columns={"day": "time_step"}, inplace=True)

        eod_portfolio_values_df = self.repository.get_eod_portfolio_values(agent_id=self.agent_id)
        # TODO: for futures contracts valuate using self.futures_value
        eod_portfolio_values_df.rename(columns={"day": "time_step"}, inplace=True)

        labels_map = {asset_id: security.short_name for asset_id, security in self.markets.items()}
        for m_id, market in self.abandoned_markets.items():
            labels_map[m_id] = market.short_name

        plot_agent_history_multicurrency(
            position_history=positions_df,
            cash_history=eod_cash_df,
            value_history=eod_portfolio_values_df,
            output_file=agent_output_file,
            title=f"Agent {self.agent_id} {str(self)} summary",
            labels_map=labels_map,
            time_label="Simulation day",
        )

        # plot_agent_history_many_markets(
        #     position_history=positions_df,
        #     cash_history=eod_cash_df,
        #     value_history=eod_portfolio_values_df,
        #     output_file=agent_output_file,
        #     title=f"Agent {self.agent_id} {str(self)} summary",
        #     labels_map=labels_map,
        #     time_label="Simulation day",
        # )

    def __str__(self) -> str:
        return f"{self.agent_id}_{self.group}"

    def get_market_knowledge(self, asset_id: int) -> MarketKnowledge:
        return self.market_knowledge.get(asset_id)

    def get_order_mean_volume(
        self,
        asset_id: int,
        fallback: float,
    ) -> float:
        """Calculate an agent's mean order size for one asset.

        volume_fraction: share of historical daily volume targeted
                         per agent across its expected daily orders.
        sim_time: ticks per simulation day.
        """
        cfg = self.configuration or {}
        fraction = cfg.get("volume_fraction")

        if fraction is None:
            return float(fallback)

        fraction = float(fraction)
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("volume_fraction must be in (0, 1]")

        market = self.markets[asset_id]
        history = getattr(market, "historical_ohlcv", None)

        if history is None or history.empty or "volume" not in history:
            return float(fallback)

        volumes = pd.to_numeric(history["volume"], errors="coerce")
        volumes = volumes.dropna()
        volumes = volumes[(volumes >= 0) & volumes.map(math.isfinite)]

        if volumes.empty:
            return float(fallback)

        daily_volume = float(volumes.mean())
        sim_time = int(self.sim_time)
        activity = float(cfg.get("lam", 1.0))

        if sim_time <= 0 or not 0 < activity <= 1:
            raise ValueError("Invalid sim_time or lam")

        expected_orders = sim_time * activity

        return max(
            1.0,
            daily_volume * fraction / expected_orders,
        )
