from __future__ import annotations

from pathlib import Path

import pandas as pd

from marketsim.analysis.market_realism_analyzer import MarketRealismAnalyzer
from marketsim.analysis.market_series import MarketSeries


def daily_ohlcv(traded_prices: pd.DataFrame, session_date) -> dict:
    """Aggregate existing intraday OHLCV into one daily observation."""
    df = traded_prices.sort_values("time_tick")

    return {
        "date": session_date,
        "open": float(df.iloc[0]["open"]),
        "high": float(df["high"].max()),
        "low": float(df["low"].min()),
        "close": float(df.iloc[-1]["close"]),
        "volume": float(df["volume"].sum()),
    }


class MarketRealismReport:
    def __init__(self):
        self.observations: dict[str, list[dict]] = {}

    def collect_day(self, simulator, day: int) -> None:
        """Call after simulator.eod(), before the next sod()."""
        if not simulator.simulation_dates:
            return

        for market in simulator.markets.values():
            if market.instrument_class != "stock":
                continue

            df = getattr(market, "traded_prices_df", None)

            if df is None or df.empty:
                continue

            symbol = getattr(market, "symbol", None)
            if not symbol:
                continue

            self.observations.setdefault(symbol, []).append(
                daily_ohlcv(df, simulator.simulation_dates[day])
            )

    def generate(self, simulator, output_dir: str | Path) -> pd.DataFrame:
        output_dir = Path(output_dir) / "market_realism"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        diagnostics = []
        scorecards = []

        for market in simulator.markets.values():
            if market.instrument_class != "stock":
                continue

            symbol = getattr(market, "symbol", None)
            historical = getattr(market, "historical_ohlcv", None)
            simulated = self.observations.get(symbol, [])

            if not symbol or historical is None or historical.empty:
                continue

            if len(simulated) < 4:
                diagnostics.append(
                    {
                        "symbol": symbol,
                        "status": "insufficient_simulated_history",
                        "observations": len(simulated),
                    }
                )
                continue

            try:
                historical_df = historical.tail(len(simulated)).copy()

                historical_series = MarketSeries(symbol, historical_df)
                simulated_series = MarketSeries(symbol, pd.DataFrame(simulated))

                analyzer = MarketRealismAnalyzer(
                    historical_series,
                    simulated_series,
                )

                results.append(analyzer.compare())
                scorecards.append(analyzer.scorecard())
                diagnostics.append(
                    {
                        "symbol": symbol,
                        "status": "ok",
                        "observations": len(simulated),
                    }
                )

            except (ValueError, TypeError, KeyError) as exc:
                diagnostics.append(
                    {
                        "symbol": symbol,
                        "status": "error",
                        "observations": len(simulated),
                        "details": str(exc),
                    }
                )

        report = (
            pd.concat(results, ignore_index=True)
            if results
            else pd.DataFrame(
                columns=[
                    "symbol",
                    "metric",
                    "historical",
                    "simulated",
                    "difference",
                ]
            )
        )

        report.to_csv(
            output_dir / "market_realism_metrics.csv",
            index=False,
        )

        pd.DataFrame(diagnostics).to_csv(
            output_dir / "market_realism_diagnostics.csv",
            index=False,
        )

        pd.DataFrame(
            scorecards,
            columns=[
                "symbol",
                "historical_days",
                "simulated_days",
                "volatility_ratio",
                "volume_ratio",
                "turnover_ratio",
                "volume_dispersion_ratio",
                "return_autocorrelation_difference",
                "volatility_clustering_difference",
                "price_volume_correlation_difference",
            ],
        ).to_csv(
            output_dir / "market_realism_scorecard.csv",
            index=False,
        )

        return report
