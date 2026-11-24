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


def _pick_first(columns: set[str], *candidates: str) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def load_wig20_greeks(db_path: Path | str) -> pd.DataFrame:
    """Load net EoD delta/gamma per agent/day for the WIG20 underlying.

    Supports either a wide table (delta/gamma columns) or a long table
    (greek_name + greek_value). If no underlying symbol column is present,
    all rows are assumed to belong to WIG20, which is the only configured
    Greeks underlying at this stage of the project.
    """
    import duckdb

    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Simulation DuckDB not found: {db_path}")
    with duckdb.connect(str(db_path), read_only=True) as conn:
        present = {str(row[0]).lower() for row in conn.execute("SHOW TABLES").fetchall()}
        required = {"agents", "eod_agent_greeks"}
        if missing := required - present:
            raise ValueError(f"Simulation missing Greeks tables: {sorted(missing)}")
        greeks = conn.execute("SELECT * FROM eod_agent_greeks").fetchdf()
        agents = conn.execute("SELECT agent_id, group_name, name FROM agents").fetchdf()
        index_map = pd.DataFrame()
        if "indices" in present:
            index_map = conn.execute(
                "SELECT index_id AS underlying_id, COALESCE(short_name, name) AS underlying_symbol FROM indices"
            ).fetchdf()
        security_map = pd.DataFrame()
        if "securities" in present:
            security_map = conn.execute(
                "SELECT asset_id AS underlying_id, COALESCE(short_name, name) AS underlying_symbol FROM securities"
            ).fetchdf()

    if greeks.empty:
        return pd.DataFrame(
            columns=["day", "agent_id", "underlying_symbol", "delta", "gamma", "group_name", "name"]
        )

    greeks.columns = [str(column).lower() for column in greeks.columns]
    columns = set(greeks.columns)
    if "agent_id" not in columns or "day" not in columns:
        raise ValueError("eod_agent_greeks must contain day and agent_id")

    if "delta" not in columns or "gamma" not in columns:
        greek_name_col = _pick_first(columns, "greek_name", "greek", "measure")
        greek_value_col = _pick_first(columns, "greek_value", "value", "amount")
        if greek_name_col is None or greek_value_col is None:
            raise ValueError(
                "eod_agent_greeks must expose delta/gamma directly or provide greek_name/greek_value"
            )
        index_cols = ["day", "agent_id"]
        for candidate in (
            "underlying_symbol",
            "underlying_short_name",
            "underlying_name",
            "underlying_id",
            "symbol",
            "short_name",
        ):
            if candidate in columns and candidate not in index_cols:
                index_cols.append(candidate)
                break
        greeks = greeks.pivot_table(
            index=index_cols,
            columns=greek_name_col,
            values=greek_value_col,
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
        greeks.columns = [str(column).lower() for column in greeks.columns]
        columns = set(greeks.columns)

    for greek in ("delta", "gamma"):
        if greek not in greeks.columns:
            greeks[greek] = 0.0
        greeks[greek] = pd.to_numeric(greeks[greek], errors="coerce").fillna(0.0)

    underlying_symbol_col = _pick_first(
        columns,
        "underlying_symbol",
        "underlying_short_name",
        "underlying_name",
        "symbol",
        "short_name",
    )
    if underlying_symbol_col is not None:
        greeks["underlying_symbol"] = greeks[underlying_symbol_col].astype(str)
    elif "underlying_id" in columns:
        greeks = greeks.merge(index_map, on="underlying_id", how="left")
        if not security_map.empty:
            greeks = greeks.merge(
                security_map.rename(columns={"underlying_symbol": "security_underlying_symbol"}),
                on="underlying_id",
                how="left",
            )
            greeks["underlying_symbol"] = greeks["underlying_symbol"].fillna(
                greeks["security_underlying_symbol"]
            )
            greeks = greeks.drop(columns=["security_underlying_symbol"])
    else:
        greeks["underlying_symbol"] = "WIG20"

    symbol = greeks["underlying_symbol"].fillna("").astype(str).str.upper()
    mask = symbol.str.replace(" ", "", regex=False).str.contains("WIG20", regex=False)
    if mask.any():
        greeks = greeks.loc[mask].copy()
    elif greeks["underlying_symbol"].nunique(dropna=True) <= 1:
        greeks = greeks.copy()
        greeks["underlying_symbol"] = "WIG20"
    else:
        greeks = greeks.iloc[0:0].copy()

    if greeks.empty:
        return pd.DataFrame(
            columns=["day", "agent_id", "underlying_symbol", "delta", "gamma", "group_name", "name"]
        )

    greeks["day"] = pd.to_numeric(greeks["day"], errors="raise").astype(int)
    greeks["agent_id"] = pd.to_numeric(greeks["agent_id"], errors="raise").astype(int)
    aggregated = (
        greeks.groupby(["day", "agent_id", "underlying_symbol"], as_index=False)[["delta", "gamma"]]
        .sum()
        .sort_values(["agent_id", "day"])
    )
    aggregated = aggregated.merge(agents, on="agent_id", how="left", validate="many_to_one")
    aggregated["group_name"] = (
        aggregated["group_name"].fillna("Unclassified").replace("", "Unclassified")
    )
    aggregated["name"] = aggregated["name"].fillna("Unknown agent")
    return aggregated
