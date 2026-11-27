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


def _empty_wig20_greeks(reason: str) -> pd.DataFrame:
    """Return a well-formed empty result and a reason for the caller to display."""
    result = pd.DataFrame(
        columns=["day", "agent_id", "underlying_symbol", "delta", "gamma", "group_name", "name"]
    )
    result.attrs["diagnostic"] = reason
    return result


def load_wig20_greeks(db_path: Path | str) -> pd.DataFrame:
    """Read WIG20 option-portfolio Delta/Gamma from Daedalus EoD Greeks tables.

    Agent rows already carry signed contract-multiplier-adjusted exposures.
    Empty Greeks tables are possible when risk_greeks was disabled; in that
    case the report can still generate its unrelated VaR output.
    """
    import duckdb
    import numpy as np

    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"Simulation DuckDB not found: {path}")

    with duckdb.connect(str(path), read_only=True) as conn:
        tables = {str(row[0]).lower() for row in conn.execute("SHOW TABLES").fetchall()}
        if "eod_agent_option_greeks" not in tables:
            return _empty_wig20_greeks(
                "No eod_agent_option_greeks table. The run may have had "
                "risk_greeks.enabled=false; no Greeks plots can be reconstructed."
            )
        if not {"agents", "indices"} <= tables:
            return _empty_wig20_greeks("Missing agents or indices needed to identify WIG20 Greeks")

        indices = conn.execute("SELECT index_id, name, short_name FROM indices").fetchdf()
        indices["symbol"] = indices["short_name"].fillna(indices["name"])
        wig20 = indices.loc[indices["symbol"].astype(str).str.upper().str.strip() == "WIG20"]
        if wig20.empty:
            return _empty_wig20_greeks("No WIG20 index registered in this simulation")
        wig20_ids = set(wig20["index_id"])

        rows = conn.execute("SELECT * FROM eod_agent_option_greeks").fetchdf()
        if rows.empty:
            return _empty_wig20_greeks("No agent option Greeks were recorded in this run")
        rows.columns = [str(name).lower() for name in rows.columns]
        required = {"day", "agent_id", "asset_id", "underlying_id", "position", "multiplier"}
        if required - set(rows.columns):
            raise ValueError(
                "Incomplete eod_agent_option_greeks columns: "
                f"{sorted(required - set(rows.columns))}"
            )
        rows = rows.loc[rows["underlying_id"].isin(wig20_ids)].copy()
        if rows.empty:
            return _empty_wig20_greeks("No WIG20 option Greeks recorded in this run")

        if "delta_equivalent" not in rows.columns:
            if "delta" not in rows.columns:
                raise ValueError("Agent option Greeks contain neither delta_equivalent nor delta")
            rows["delta_equivalent"] = rows["position"] * rows["multiplier"] * rows["delta"]
        elif rows["delta_equivalent"].isna().any():
            if "delta" not in rows.columns:
                raise ValueError("Missing option delta-equivalent observations")
            missing_delta = rows["delta_equivalent"].isna()
            rows.loc[missing_delta, "delta_equivalent"] = (
                rows.loc[missing_delta, "position"]
                * rows.loc[missing_delta, "multiplier"]
                * rows.loc[missing_delta, "delta"]
            )

        if "gamma_equivalent" not in rows.columns:
            rows["gamma_equivalent"] = float("nan")

        option_greek_days = None
        if "eod_option_greeks" in tables:
            raw = conn.execute("SELECT * FROM eod_option_greeks").fetchdf()
            raw.columns = [str(name).lower() for name in raw.columns]
            needed = {"day", "asset_id", "underlying_id", "gamma", "multiplier"}
            if needed <= set(raw.columns):
                raw = raw.loc[raw["underlying_id"].isin(wig20_ids)].copy()
                option_greek_days = sorted(raw["day"].unique().tolist())
                if rows["gamma_equivalent"].isna().any():
                    join_keys = ["day", "asset_id", "underlying_id"]
                    if "calibration_id" in rows and "calibration_id" in raw:
                        join_keys.append("calibration_id")
                    if raw.duplicated(join_keys).any():
                        raise ValueError("Duplicate option-Gamma snapshots for a calibration")
                    selected_raw = raw[join_keys + ["gamma"]]
                    rows = rows.merge(
                        selected_raw, on=join_keys, how="left", validate="many_to_one"
                    )
                    missing_gamma = rows["gamma_equivalent"].isna()
                    rows.loc[missing_gamma, "gamma_equivalent"] = (
                        rows.loc[missing_gamma, "position"]
                        * rows.loc[missing_gamma, "multiplier"]
                        * rows.loc[missing_gamma, "gamma"]
                    )
                    rows = rows.drop(columns=["gamma"])

        for column in ("delta_equivalent", "gamma_equivalent"):
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
            if not np.isfinite(rows[column].to_numpy(dtype=float)).all():
                return _empty_wig20_greeks(
                    f"Missing or non-finite {column} in recorded WIG20 Greeks. "
                    "Cannot plot a reliable exposure series."
                )

        agents = conn.execute("SELECT agent_id, group_name, name FROM agents").fetchdf()
        if agents["agent_id"].duplicated().any():
            raise ValueError("Duplicate agent IDs in agent registry")

    aggregate = rows.groupby(["day", "agent_id"], as_index=False).agg(
        delta=("delta_equivalent", "sum"), gamma=("gamma_equivalent", "sum")
    )
    # A Greek snapshot is available every day on which the option Greeks table
    # was populated. Fill missing agent/days with zero only on those verified days:
    # such agents held no WIG20 options at that EoD.
    if option_greek_days:
        frame = pd.MultiIndex.from_product(
            [sorted(agents["agent_id"].astype(int).tolist()), option_greek_days],
            names=["agent_id", "day"],
        ).to_frame(index=False)
        aggregate = frame.merge(
            aggregate, on=["agent_id", "day"], how="left", validate="one_to_one"
        )
        aggregate[["delta", "gamma"]] = aggregate[["delta", "gamma"]].fillna(0.0)

    aggregate["underlying_symbol"] = "WIG20"
    aggregate = aggregate.merge(agents, on="agent_id", how="left", validate="many_to_one")
    aggregate["group_name"] = (
        aggregate["group_name"].fillna("Unclassified").replace("", "Unclassified")
    )
    aggregate["name"] = aggregate["name"].fillna("Unknown agent")
    return (
        aggregate[["day", "agent_id", "underlying_symbol", "delta", "gamma", "group_name", "name"]]
        .sort_values(["agent_id", "day"])
        .reset_index(drop=True)
    )
