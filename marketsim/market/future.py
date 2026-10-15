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

