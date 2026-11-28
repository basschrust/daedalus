"""Post-run historical benchmark VaR + simulated agent APL backtesting.

No new command, database tables or cloud connections. Uses the existing
historical price panel and reads the completed simulation DuckDB after closure.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Mapping

import pandas as pd

from marketsim.database.risk.risk_reporting_queries import load_risk_report_data
from marketsim.risk.christoffersen import christoffersen_report
from marketsim.risk.historical_portfolios import build_historical_equity_snapshots
from marketsim.risk.historical_var_backtest import walk_forward_equity_var
from marketsim.risk.var_backtesting import kupiec_report, one_day_apl


def write_post_run_backtests(
    *,
    db_path: Path,
    output_dir: Path,
    historical: Mapping[str, pd.DataFrame],
    risk_config: Mapping[str, object],
    settings: Mapping[str, object],
) -> None:
    """Write read-only APL and synthetic historical WIG20 HPL diagnostics.

    The historical index book is an explicit *synthetic benchmark*, not a
    reconstruction of any market agent's old positions. Use the first
    historical close to determine the units and hold them constant thereafter.
    """
    if not settings.get("enabled", False):
        return
    directory = output_dir
    directory.mkdir(parents=True, exist_ok=True)
    if not db_path.is_file():
        raise FileNotFoundError(f"Post-run VaR database missing: {db_path}")

    # Simulated agent APL already exists in the DB. Do not mix with HPL.
    risk, equity = load_risk_report_data(db_path)
    apl = one_day_apl(risk, equity)
    apl_kupiec = kupiec_report(apl)
    apl_christoffersen = christoffersen_report(observations=apl)
    apl.to_csv(directory / "kupiec_observations_apl.csv", index=False)
    apl_kupiec.to_csv(directory / "kupiec_1d_apl.csv", index=False)
    apl_christoffersen.to_csv(directory / "christoffersen_1d_apl.csv", index=False)

    symbol = str(settings.get("symbol", "WIG20")).upper().strip()
    if symbol not in historical:
        raise ValueError(f"Historical benchmark {symbol} missing from Ikaros prices")
    history = historical[symbol]
    if "close" not in history or history.index.has_duplicates:
        raise ValueError(f"Historical benchmark {symbol} has missing/duplicate closes")
    ordered = history.sort_index()
    dates: list[date] = []
    closes: dict[date, dict[int, Decimal]] = {}
    for timestamp, value in ordered["close"].items():
        session = pd.Timestamp(timestamp).date()
        price = Decimal(str(value))
        if not price.is_finite() or price <= 0 or session in closes:
            raise ValueError(f"Bad historical benchmark close for {symbol}: {session}")
        dates.append(session)
        closes[session] = {0: price}

    units = settings.get("units", 20)
    if type(units) is not int or units <= 0:
        raise ValueError("risk_backtesting.units must be a positive integer")
    minimum = int(risk_config.get("historical_min_scenarios", 30))
    if len(dates) < minimum + 2:
        raise ValueError(f"Historical benchmark requires at least {minimum + 2} sessions")
    snapshots = build_historical_equity_snapshots(
        agent_id=-1,
        dates=dates,
        holdings={0: units},
        instrument_classes={0: "index"},
        closes=closes,
        initial_cash={"PLN": Decimal(0)},
        base_currency="PLN",
    )
    config = dict(risk_config)
    config["montecarlo_scenarios"] = int(settings.get("montecarlo_scenarios", 3000))
    forecasts, observations, hpl_kupiec = walk_forward_equity_var(
        snapshots=snapshots,
        closes=closes,
        symbols={0: symbol},
        instrument_classes={0: "index"},
        risk_config=config,
        lookback_sessions=int(settings.get("lookback_sessions", 125)),
        group_names={-1: "SYNTHETIC_INDEX_BENCHMARK"},
    )
    christoffersen = christoffersen_report(observations=observations)
    forecasts.to_csv(directory / "historical_var_forecasts_hpl.csv", index=False)
    observations.to_csv(directory / "historical_var_observations_hpl.csv", index=False)
    hpl_kupiec.to_csv(directory / "historical_kupiec_1d_hpl.csv", index=False)
    christoffersen.to_csv(directory / "historical_christoffersen_1d_hpl.csv", index=False)
    print(
        f"VaR backtesting: simulated APL {len(apl)} observations; "
        f"synthetic historical {symbol} HPL {len(observations)} observations. "
        f"Reports: {directory}"
    )
    print("Historical HPL is a synthetic fixed-index benchmark, not agent trading history.")
