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


class Index(Security):
    def __init__(self, *, repository: Repository, components: dict,
                 name: str| None = None, short_name: str|None=None, output_dir: Path|str="tmp") -> None:

        self.instrument_class = "index"
        # components & value calculation
        # { asset_id: { weight: weight[float] , security: Security }}
        self.components = components
        self.value = self.calculate_value()

        super().__init__(name=name, reference_price=Price(self.value)
                         , instrument_class=self.instrument_class, repository=repository,
                         short_name=short_name, output_dir=output_dir)

        self.index_id = self.asset_id

    def validate_components(self):
        weights_sum = 0.0
        for weight in self.components.values():
            weights_sum += weight.weight
        if weights_sum == 1:
            self.logger(f"Index components of {self.name} are properly configured")

    def calculate_value(self):
        value = 0
        for asset_id, component in self.components.items():
            value += component["weight"] * component["security"].last_traded_price

        return value

    def save(self):
        # save the index definition to DB
        # def
        self.repository.save_index(index=self)

        # components
        components_table = []
        for asset_id, component in self.components.items():
            components_table.append({  "index_id": self.index_id,
                                        "asset_id": asset_id,
                                        "weight": component["weight"],
                                       })

        components_df = pd.DataFrame(components_table)
        self.repository.save_index_components(components_df=components_df)


    def eod(self):
        if self.eod_status == "closed":
            return
        elif self.eod_status == "open":
            super().eod()
            index_values_df = pd.DataFrame()
            self.repository.save_eod_index_values(index_values_df=index_values_df)