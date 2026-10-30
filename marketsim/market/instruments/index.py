from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd
from typing_extensions import TYPE_CHECKING

from marketsim.market.instruments.security import Security
from marketsim.market.price import Price
from marketsim.plot.daedalus_plot_extensions import (
    plot_combined_candles,
)

if TYPE_CHECKING:
    pass


class Index(Security):
    def __init__(
        self,
        *,
        repository,
        components: dict,
        name: str | None = None,
        short_name: str | None = None,
        output_dir: Path | str = "tmp",
        base_index_level=None,
        historical_ohlcv: pd.DataFrame | None = None,
        simulation_dates=None,
    ) -> None:
        self.instrument_class = "index"
        self.components = components

        self.base_index_level = (
            Decimal(str(base_index_level)) if base_index_level is not None else None
        )

        if self.base_index_level is not None:
            if not self.base_index_level.is_finite() or self.base_index_level <= 0:
                raise ValueError("Invalid base_index_level")

        self.value = self.calculate_value()

        super().__init__(
            name=name,
            reference_price=Price(self.value),
            instrument_class=self.instrument_class,
            repository=repository,
            short_name=short_name,
            output_dir=output_dir,
            historical_ohlcv=historical_ohlcv,
            simulation_dates=simulation_dates,
        )

        self.is_tradable = False

    def validate_components(self):
        for weight in self.components.values():
            if weight <= 0:
                self.logger(
                    f"Index components of {self.name} are broken - weights have to be positive, not: {weight}"
                )
                return False
        self.logger(f"Index components of {self.name} are properly configured")
        # it doesn't need to be 100% in total

    def calculate_value(self):
        weight_sum = sum(Decimal(str(c["weight"])) for c in self.components.values())

        weighted_return = sum(
            (
                Decimal(str(c["weight"]))
                * Decimal(str(c["security"].last_traded_price))
                / Decimal(str(c["base_price"]))
            )
            for c in self.components.values()
        )

        return Price(self.base_index_level * weighted_return / weight_sum)

    def save(self):
        # save the index definition to DB
        self.index_id = self.asset_id  # it looks strange, but this is the first place this value is
        # available and needed
        self.repository.save_index(index=self)

        # components
        components_table = []

        weight_sum = sum(
            Decimal(str(component["weight"])) for component in self.components.values()
        )

        for asset_id, component in self.components.items():
            weight = Decimal(str(component["weight"]))

            if self.base_index_level is not None:
                base_price = Decimal(str(component["base_price"]))

                valuation_coefficient = self.base_index_level * weight / (base_price * weight_sum)
            else:
                # Preserve legacy index valuation
                valuation_coefficient = weight

            components_table.append(
                {
                    "index_id": self.index_id,
                    "asset_id": component["security"].asset_id,
                    "weight": float(weight),
                    "valuation_coefficient": float(valuation_coefficient),
                }
            )

        components_df = pd.DataFrame(components_table)
        self.repository.save_index_components(components_df=components_df)

    def eod(self):
        if self.eod_status == "closed":
            return
        elif self.eod_status == "open":
            super().eod()
            self.repository.calculate_index_values(day=self.current_day - 1)
            self.repository.calculate_eod_index_values(day=self.current_day - 1)
            self.value = self.calculate_value()
            self.last_traded_price = Price(self.value)

    def show_summary(self):
        self.logger.info(f"\n\nIndex {str(self)} summary:")
        # self.logger.info(f"Orders matched: {len(self.matched_orders)}")
        # self.logger.info(f"Last index value: {self.last_traded_price}")

        # plot the security values history for all period - take data from DB:
        # TODO: for one day simulations plot the daily candlestick
        eod_prices_df = self.repository.get_eod_index_values(index_id=self.index_id)
        # The index is non-tradable; its EoD values have no trading volume.
        eod_prices_df = eod_prices_df.copy()
        eod_prices_df["volume"] = 0

        self.logger.info(f"EoD prices DF: {eod_prices_df}")
        eod_prices = eod_prices_df.set_index("day").to_dict(orient="index")
        self.logger.info(f"EOD prices: {eod_prices}")

        if self.historical_ohlcv is not None and self.simulation_dates is not None:
            history = self.historical_ohlcv

            if isinstance(history, tuple):
                if len(history) == 1 and isinstance(history[0], pd.DataFrame):
                    history = history[0]
                else:
                    raise TypeError(
                        f"Invalid historical_ohlcv for {self.name}: "
                        f"tuple of {[type(item).__name__ for item in history]}"
                    )

            if not isinstance(history, pd.DataFrame):
                raise TypeError(f"Expected DataFrame for {self.name}, got {type(history).__name__}")

            plot_combined_candles(
                historical=history,
                simulated=eod_prices_df,
                simulation_dates=self.simulation_dates,
                output_file=(Path(self.output_dir) / f"candlestick_{self}.png"),
                title=self.name,
            )
        else:
            self.plot_history(traded_prices=eod_prices)
