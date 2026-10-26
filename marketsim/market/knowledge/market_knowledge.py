from .historical_market_data import HistoricalMarketData


class MarketKnowledge:
    def __init__(
        self,
        historical: HistoricalMarketData,
        confidence: float = 1.0,
    ):
        self.historical = historical
        self.confidence = confidence

    def expected_price(self, days_ahead: float = 1.0) -> float:
        return (
            self.historical.reference_price * (1 + self.historical.daily_mean_return) ** days_ahead
        )

    def expected_volatility(self) -> float:
        return self.historical.daily_volatility

    def expected_volume(self) -> float:
        return self.historical.median_daily_volume
