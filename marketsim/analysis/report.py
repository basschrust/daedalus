from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from marketsim.analysis.market_realism_analyzer import MarketRealismAnalyzer
from marketsim.analysis.market_series import MarketSeries

SCORECARD_COLUMNS = [
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
]
VOLUME_METRICS = {
    "mean_volume",
    "median_volume",
    "volume_std",
    "mean_turnover",
    "median_turnover",
    "zero_volume_fraction",
    "volume_abs_return_correlation",
}
VOLUME_SCORECARD = {
    "volume_ratio",
    "turnover_ratio",
    "volume_dispersion_ratio",
    "price_volume_correlation_difference",
}
RATIO_SCORECARD = {
    "volatility_ratio",
    "volume_ratio",
    "turnover_ratio",
    "volume_dispersion_ratio",
}


def daily_ohlcv(traded_prices: pd.DataFrame, session_date) -> dict:
    """Aggregate the recorded intraday OHLCV to a single session."""
    df = traded_prices.sort_values("time_tick")
    return {
        "date": session_date,
        "open": float(df.iloc[0]["open"]),
        "high": float(df["high"].max()),
        "low": float(df["low"].min()),
        "close": float(df.iloc[-1]["close"]),
        "volume": float(df["volume"].sum()),
    }


def _dated(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the historical index/column date representations."""
    result = frame.copy()
    if "date" not in result.columns:
        result = result.reset_index()
        result = result.rename(columns={result.columns[0]: "date"})
    return result


def _comparison(
    symbol: str, first: pd.DataFrame, second: pd.DataFrame, price_only: bool
) -> tuple[pd.DataFrame, dict]:
    analyzer = MarketRealismAnalyzer(
        MarketSeries(symbol, _dated(first)),
        MarketSeries(symbol, _dated(second)),
    )
    metrics = analyzer.compare()
    score = analyzer.scorecard()
    if price_only:
        metrics.loc[
            metrics["metric"].isin(VOLUME_METRICS), ["historical", "simulated", "difference"]
        ] = np.nan
        for key in VOLUME_SCORECARD:
            score[key] = np.nan
    return metrics, score


def _relative_score(symbol: str, simulated: dict, control: dict) -> dict:
    result = {"symbol": symbol}
    for metric in SCORECARD_COLUMNS[3:]:
        sim = simulated.get(metric, np.nan)
        hist = control.get(metric, np.nan)
        if pd.isna(sim) or pd.isna(hist):
            result[metric] = np.nan
            continue
        if metric in RATIO_SCORECARD:
            if sim <= 0 or hist <= 0:
                result[metric] = np.nan
                continue
            sim_deviation = abs(np.log(sim))
            hist_deviation = abs(np.log(hist))
        else:
            sim_deviation = abs(sim)
            hist_deviation = abs(hist)
        result[metric] = sim_deviation / hist_deviation if hist_deviation > 0 else np.nan
    return result


class MarketRealismReport:
    def __init__(self):
        self.observations: dict[str, list[dict]] = {}

    @staticmethod
    def _symbol(market) -> str | None:
        # Stocks expose .symbol; futures and indices currently use .name.
        return (
            getattr(market, "symbol", None)
            or getattr(market, "name", None)
            or getattr(market, "short_name", None)
        )

    def collect_day(self, simulator, day: int) -> None:
        """Collect EoD market observations after simulator.eod()."""
        if not simulator.simulation_dates:
            return
        session_date = simulator.simulation_dates[day]
        for market in simulator.markets.values():
            kind = getattr(market, "instrument_class", None)
            if kind not in {"stock", "future", "option", "index"}:
                continue
            symbol = self._symbol(market)
            if not symbol:
                continue

            if kind == "index":
                # An index is not a traded security. Use its calculated daily OHLC.
                frame = simulator.repository.get_eod_index_values(index_id=market.asset_id)
                if frame is None or frame.empty:
                    continue
                rows = frame.loc[frame["day"] == day]
                if rows.empty:
                    continue
                row = rows.iloc[-1]
                observation = {
                    "date": session_date,
                    "open": float(row["open"]),
                    "high": max(float(row["high"]), float(row["open"]), float(row["close"])),
                    "low": min(float(row["low"]), float(row["open"]), float(row["close"])),
                    "close": float(row["close"]),
                    "volume": 0.0,  # Placeholder only; index volume metrics are N/A.
                }
            else:
                frame = getattr(market, "traded_prices_df", None)
                if frame is None or frame.empty:
                    continue
                observation = daily_ohlcv(frame, session_date)
            self.observations.setdefault(symbol, []).append(observation)

    def generate(self, simulator, output_dir: str | Path) -> pd.DataFrame:
        output_dir = Path(output_dir) / "market_realism"
        output_dir.mkdir(parents=True, exist_ok=True)
        results, diagnostics, scorecards = [], [], []
        controls, relatives = [], []

        for market in simulator.markets.values():
            kind = getattr(market, "instrument_class", None)
            if kind not in {"stock", "future", "option", "index"}:
                continue
            symbol = self._symbol(market)
            if not symbol:
                continue
            simulated = self.observations.get(symbol, [])
            historical = getattr(market, "historical_ohlcv", None)
            if historical is None:
                # Options currently have no historical_ohlcv argument in their constructor.
                historical = simulator.historical_ohlcv_by_symbol.get(symbol)
            n = len(simulated)
            diagnostic = {"symbol": symbol, "instrument_class": kind, "observations": n}

            if historical is None or historical.empty:
                diagnostics.append(
                    {
                        **diagnostic,
                        "status": "missing_historical_data",
                        "details": "No historical OHLCV available for comparison",
                    }
                )
                continue
            if n < 4:
                diagnostics.append(
                    {
                        **diagnostic,
                        "status": "insufficient_simulated_history",
                        "details": "At least 4 simulated sessions required",
                    }
                )
                continue
            if len(historical) < n:
                diagnostics.append(
                    {
                        **diagnostic,
                        "status": "insufficient_historical_history",
                        "details": f"Need {n} historical sessions; got {len(historical)}",
                    }
                )
                continue

            try:
                history = _dated(historical).sort_values("date")
                recent = history.tail(n)
                simulated_frame = pd.DataFrame(simulated)
                price_only = kind == "index"
                metrics, score = _comparison(symbol, recent, simulated_frame, price_only)
                metrics.insert(1, "instrument_class", kind)
                score["instrument_class"] = kind
                score["volume_basis"] = (
                    "not_applicable"
                    if price_only
                    else "contracts"
                    if kind in {"future", "option"}
                    else "shares"
                )
                results.append(metrics)
                scorecards.append(score)
                diagnostic["status"] = "ok"

                if len(history) >= 2 * n:
                    older = history.iloc[-2 * n : -n]
                    control_metrics, control = _comparison(symbol, older, recent, price_only)
                    control["instrument_class"] = kind
                    controls.append(control)
                    relative = _relative_score(symbol, score, control)
                    relative["instrument_class"] = kind
                    relatives.append(relative)
                    diagnostic["control_status"] = "ok"
                else:
                    diagnostic["control_status"] = "insufficient_historical_history"
                    diagnostic["control_details"] = (
                        f"Need {2 * n} historical sessions for control; got {len(history)}"
                    )
                diagnostics.append(diagnostic)
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                diagnostics.append({**diagnostic, "status": "error", "details": str(exc)})

        metric_cols = [
            "symbol",
            "instrument_class",
            "metric",
            "historical",
            "simulated",
            "difference",
        ]
        report = (
            pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=metric_cols)
        )
        report.to_csv(output_dir / "market_realism_metrics.csv", index=False)
        pd.DataFrame(
            diagnostics,
            columns=[
                "symbol",
                "instrument_class",
                "status",
                "observations",
                "details",
                "control_status",
                "control_details",
            ],
        ).to_csv(output_dir / "market_realism_diagnostics.csv", index=False)
        pd.DataFrame(
            scorecards,
            columns=["symbol", "instrument_class", "volume_basis", *SCORECARD_COLUMNS[1:]],
        ).to_csv(output_dir / "market_realism_scorecard.csv", index=False)
        pd.DataFrame(
            controls, columns=["symbol", "instrument_class", *SCORECARD_COLUMNS[1:]]
        ).to_csv(output_dir / "market_realism_historical_control.csv", index=False)
        pd.DataFrame(
            relatives, columns=["symbol", "instrument_class", *SCORECARD_COLUMNS[3:]]
        ).to_csv(output_dir / "market_realism_relative_scorecard.csv", index=False)
        return report
