from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from marketsim.market.price import Price
from marketsim.market.security import Security

if TYPE_CHECKING:
    from marketsim.database.repositories.interface import RepositoryInterface
    from marketsim.market.index import Index


class ETF(Security):
    def __init__(
        self,
        *,
        underlying: Index,
        repository: RepositoryInterface,
        tracking_ratio: float = 1.0,
        market_type: str = "continuous",
        name: str | None = None,
        short_name: str | None = None,
        output_dir: Path | str = "tmp",
    ) -> None:
        if underlying.instrument_class != "index":
            raise ValueError("ETF underlying must be an Index")
        if tracking_ratio <= 0:
            raise ValueError("tracking_ratio must be positive")

        self.underlying = underlying
        self.tracking_ratio = tracking_ratio
        initial_price = self.calculate_theoretical_price()

        super().__init__(
            repository=repository,
            instrument_class="etf",
            market_type=market_type,
            name=name,
            short_name=short_name,
            output_dir=output_dir,
            currency=underlying.currency,
            reference_price=initial_price,
        )

    def calculate_theoretical_price(self) -> Price:
        return Price(self.underlying.last_traded_price * self.tracking_ratio)

    def calculate_premium(self) -> float:
        theoretical = float(self.calculate_theoretical_price())
        if theoretical == 0:
            return 0.0

        market_price = float(self.last_traded_price)

        return (market_price / theoretical - 1.0) * 100
