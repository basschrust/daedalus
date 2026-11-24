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


def load_derivative_exposure(db_path: Path | str) -> pd.DataFrame:
    """Count distinct derivative series held at final/any simulation EoD.

    Only nonzero EoD positions count. Intraday positions closed before EoD
    are deliberately excluded, as are unexecuted derivative orders.
    Options are in eod_agent_positions/securities; futures are in their
    own position table and futures instrument registry.
    """
    import duckdb

    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Simulation DuckDB not found: {db_path}")
    with duckdb.connect(str(db_path), read_only=True) as conn:
        present = {str(row[0]).lower() for row in conn.execute("SHOW TABLES").fetchall()}
        required = {
            "agents",
            "securities",
            "futures",
            "eod_agent_positions",
            "eod_agent_futures_position",
            "eod_agent_portfolio_value",
        }
        if missing := required - present:
            raise ValueError(f"Simulation missing derivative-position tables: {sorted(missing)}")
        last_day = conn.execute("SELECT MAX(day) FROM eod_agent_portfolio_value").fetchone()[0]
        if last_day is None:
            raise ValueError("No simulation EoD portfolio values for derivative exposure")
        result = conn.execute(
            """
            WITH held AS (
                SELECT p.day, p.agent_id, p.asset_id, 'option' AS instrument
                FROM eod_agent_positions p
                JOIN securities s ON s.asset_id = p.asset_id
                WHERE LOWER(s.instrument_class) = 'option' AND p.position <> 0
                UNION ALL
                SELECT p.day, p.agent_id, p.asset_id, 'future' AS instrument
                FROM eod_agent_futures_position p
                JOIN futures f ON f.contract_id = p.asset_id
                WHERE p.position <> 0
            )
            SELECT a.agent_id,
                COUNT(DISTINCT CASE WHEN h.instrument = 'option'
                    AND h.day = ? THEN h.asset_id END) AS options_last_eod,
                COUNT(DISTINCT CASE WHEN h.instrument = 'future'
                    AND h.day = ? THEN h.asset_id END) AS futures_last_eod,
                COUNT(DISTINCT CASE WHEN h.instrument = 'option'
                    THEN h.asset_id END) AS options_any_eod,
                COUNT(DISTINCT CASE WHEN h.instrument = 'future'
                    THEN h.asset_id END) AS futures_any_eod
            FROM agents a
            LEFT JOIN held h ON h.agent_id = a.agent_id
            GROUP BY a.agent_id
            ORDER BY a.agent_id
            """,
            [last_day, last_day],
        ).fetchdf()
    result["last_eod_day"] = int(last_day)
    return result
