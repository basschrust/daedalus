"""Read-only post-simulation risk reporting queries: local DuckDB only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

METHODS = ("parametric", "historical", "montecarlo")


def load_risk_report_data(db_path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load forecasts and EoD equity without initializing repositories or sources."""
    import duckdb

    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Simulation DuckDB not found: {db_path}")
    with duckdb.connect(str(db_path), read_only=True) as conn:
        present = {str(row[0]).lower() for row in conn.execute("SHOW TABLES").fetchall()}
        required = {"agents", "eod_agent_portfolio_value"}
        required.update(f"eod_agent_var_{method}" for method in METHODS)
        if missing := required - present:
            raise ValueError(f"Simulation missing risk tables: {sorted(missing)}")

        queries = [
            f"""SELECT '{method}' AS method, day, agent_id, confidence_level,
                       horizon_days, var_amount, portfolio_value, base_currency,
                       covered_exposure, excluded_exposure, calibration_id
                FROM eod_agent_var_{method}"""
            for method in METHODS
        ]
        risk = conn.execute(" UNION ALL ".join(queries)).fetchdf()
        agents = conn.execute("SELECT agent_id, group_name, name FROM agents").fetchdf()
        equity = conn.execute(
            "SELECT day, agent_id, portfolio_value FROM eod_agent_portfolio_value"
        ).fetchdf()
    if risk.empty:
        raise ValueError("No EoD VaR observations in this run")
    if agents.agent_id.duplicated().any():
        raise ValueError("Duplicate agent IDs in agents")
    if equity.duplicated(["day", "agent_id"]).any():
        raise ValueError("Duplicate EoD agent equity records")
    if risk.duplicated(["method", "day", "agent_id", "confidence_level", "horizon_days"]).any():
        raise ValueError("Duplicate VaR forecasts; refusing ambiguous reporting")
    risk = risk.merge(agents, on="agent_id", how="left", validate="many_to_one")
    risk["group_name"] = risk["group_name"].fillna("Unclassified").replace("", "Unclassified")
    risk["name"] = risk["name"].fillna("Unknown agent")
    return risk, equity
