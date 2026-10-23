from __future__ import annotations

from pathlib import Path

import pandas as pd
from typing_extensions import TYPE_CHECKING

from marketsim.market.price import Price
from marketsim.market.security import Security

if TYPE_CHECKING:
    from database.connectors.duckdb_storage import Repository


class Index(Security):
    def __init__(
        self,
        *,
        repository: Repository,
        components: dict,
        name: str | None = None,
        short_name: str | None = None,
        output_dir: Path | str = "tmp",
    ) -> None:

        self.instrument_class = "index"
        # components & value calculation
        # { asset_id: { weight: weight[float] , security: Security }}
        self.components = components
        self.value = self.calculate_value()

        super().__init__(
            name=name,
            reference_price=Price(self.value),
            instrument_class=self.instrument_class,
            repository=repository,
            short_name=short_name,
            output_dir=output_dir,
        )

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
        value = 0
        for asset_id, component in self.components.items():
            value += component["weight"] * component["security"].last_traded_price

        return value

    def save(self):
        # save the index definition to DB
        self.index_id = self.asset_id  # it looks strange, but this is the first place this value is
        # available and needed
        self.repository.save_index(index=self)

        # components
        components_table = []
        for asset_id, component in self.components.items():
            components_table.append(
                {
                    "index_id": self.index_id,
                    "asset_id": component["security"].asset_id,
                    "weight": component["weight"],
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
        self.logger.info(f"EoD prices DF: {eod_prices_df}")
        eod_prices = eod_prices_df.set_index("day").to_dict(orient="index")
        self.logger.info(f"EOD prices: {eod_prices}")
        self.plot_history(traded_prices=eod_prices)
