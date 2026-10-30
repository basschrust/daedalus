"""Snowflake persistence backend for Daedalus.

Install: pip install 'snowflake-connector-python[pandas]'
Credentials: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD.
Never commit passwords or private keys.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

from marketsim.database.queries.group_flows import GROUP_FLOW_INSERT_SQL
from marketsim.plot.execution_volumes import EXECUTION_VOLUME_SQL

if TYPE_CHECKING:
    from marketsim.agent import Agent
    from marketsim.market import Future, Index, Security

# Column names intentionally mirror the existing DuckDB tables.
# Snowflake's NUMBER(38,0) corresponds to integer values; FLOAT to DuckDB DOUBLE.
TABLES: dict[str, dict[str, str]] = {
    "securities": dict(
        asset_id="NUMBER(38,0)",
        instrument_class="VARCHAR",
        market_type="VARCHAR",
        name="VARCHAR",
        short_name="VARCHAR",
        eod_status="VARCHAR",
        reference_price="FLOAT",
        currency="VARCHAR",
    ),
    "indices": dict(index_id="NUMBER(38,0)", name="VARCHAR", short_name="VARCHAR"),
    "index_components": dict(
        index_id="NUMBER(38,0)",
        asset_id="NUMBER(38,0)",
        weight="FLOAT",
        valuation_coefficient="FLOAT",
    ),
    "futures": dict(
        contract_id="NUMBER(38,0)",
        underlying_type="VARCHAR",
        underlying_id="NUMBER(38,0)",
        expiration_day="NUMBER(38,0)",
        multiplier="FLOAT",
        margin_rate="FLOAT",
        name="VARCHAR",
        short_name="VARCHAR",
    ),
    "options": dict(
        asset_id="NUMBER(38,0)",
        underlying_id="NUMBER(38,0)",
        configuration="VARCHAR",
        expiration_day="NUMBER(38,0)",
        strike="FLOAT",
    ),
    "agents": dict(
        agent_id="NUMBER(38,0)", name="VARCHAR", group_name="VARCHAR", configuration="VARCHAR"
    ),
    "agent_position_history": dict(
        day="NUMBER(38,0)",
        time_tick="NUMBER(38,0)",
        agent_id="NUMBER(38,0)",
        asset_id="NUMBER(38,0)",
        position="NUMBER(38,0)",
    ),
    "agent_futures_position_history": dict(
        day="NUMBER(38,0)",
        time_tick="NUMBER(38,0)",
        agent_id="NUMBER(38,0)",
        asset_id="NUMBER(38,0)",
        position="NUMBER(38,0)",
    ),
    "agent_futures_value_history": dict(
        day="NUMBER(38,0)",
        time_tick="NUMBER(38,0)",
        agent_id="NUMBER(38,0)",
        asset_id="NUMBER(38,0)",
        position="NUMBER(38,0)",
        avg_initial_price="FLOAT",
        collateral="FLOAT",
        unsettled_pnl="FLOAT",
        settlement_price="FLOAT",
    ),
    "agent_eod_positions": dict(
        day="NUMBER(38,0)",
        agent_id="NUMBER(38,0)",
        asset_id="NUMBER(38,0)",
        position="NUMBER(38,0)",
    ),
    "agent_eod_futures_position": dict(
        day="NUMBER(38,0)",
        agent_id="NUMBER(38,0)",
        asset_id="NUMBER(38,0)",
        position="NUMBER(38,0)",
    ),
    "agent_eod_futures_value": dict(
        day="NUMBER(38,0)",
        agent_id="NUMBER(38,0)",
        asset_id="NUMBER(38,0)",
        position="NUMBER(38,0)",
        avg_initial_price="FLOAT",
        collateral="FLOAT",
        unsettled_pnl="FLOAT",
        settlement_price="FLOAT",
    ),
    "agent_portfolio_value_history": dict(
        day="NUMBER(38,0)",
        time_tick="NUMBER(38,0)",
        agent_id="NUMBER(38,0)",
        portfolio_value="FLOAT",
    ),
    "agent_eod_portfolio_value": dict(
        day="NUMBER(38,0)", agent_id="NUMBER(38,0)", portfolio_value="FLOAT"
    ),
    "agent_cash_history": dict(
        day="NUMBER(38,0)",
        time_tick="NUMBER(38,0)",
        agent_id="NUMBER(38,0)",
        cash="FLOAT",
        currency="VARCHAR",
    ),
    "agent_eod_cash": dict(
        day="NUMBER(38,0)", agent_id="NUMBER(38,0)", cash="FLOAT", currency="VARCHAR"
    ),
    "traded_prices": dict(
        day="NUMBER(38,0)",
        time_tick="NUMBER(38,0)",
        asset_id="NUMBER(38,0)",
        open="FLOAT",
        high="FLOAT",
        low="FLOAT",
        close="FLOAT",
        volume="FLOAT",
        theoretical="FLOAT",
    ),
    "eod_prices": dict(
        day="NUMBER(38,0)",
        asset_id="NUMBER(38,0)",
        open="FLOAT",
        high="FLOAT",
        low="FLOAT",
        close="FLOAT",
        volume="FLOAT",
        theoretical="FLOAT",
    ),
    "index_values": dict(
        day="NUMBER(38,0)",
        time_tick="NUMBER(38,0)",
        index_id="NUMBER(38,0)",
        open="FLOAT",
        high="FLOAT",
        low="FLOAT",
        close="FLOAT",
    ),
    "eod_index_values": dict(
        day="NUMBER(38,0)",
        index_id="NUMBER(38,0)",
        open="FLOAT",
        high="FLOAT",
        low="FLOAT",
        close="FLOAT",
    ),
    "future_rates": dict(
        contract_id="NUMBER(38,0)",
        day="NUMBER(38,0)",
        time_tick="NUMBER(38,0)",
        open="FLOAT",
        high="FLOAT",
        low="FLOAT",
        close="FLOAT",
        volume="FLOAT",
        lop="NUMBER(38,0)",
        theoretical="FLOAT",
    ),
    "eod_future_rates": dict(
        contract_id="NUMBER(38,0)",
        day="NUMBER(38,0)",
        open="FLOAT",
        high="FLOAT",
        low="FLOAT",
        close="FLOAT",
        volume="FLOAT",
        lop="NUMBER(38,0)",
        theoretical="FLOAT",
    ),
    "orders": dict(
        day="NUMBER(38,0)",
        price="FLOAT",
        order_type="NUMBER(38,0)",
        quantity="NUMBER(38,0)",
        agent_id="NUMBER(38,0)",
        time="NUMBER(38,0)",
        order_id="NUMBER(38,0)",
        asset_id="NUMBER(38,0)",
        executed_price="FLOAT",
        executed_mode="VARCHAR",
        parent_id="NUMBER(38,0)",
        matched_with="NUMBER(38,0)",
        valid_until="NUMBER(38,0)",
    ),
    "trades": dict(
        day="NUMBER(38,0)",
        asset_id="NUMBER(38,0)",
        agent_id="NUMBER(38,0)",
        matched_with="NUMBER(38,0)",
        executed_mode="VARCHAR",
        executed_time="NUMBER(38,0)",
        executed_price="FLOAT",
        order_id="NUMBER(38,0)",
        order_side="NUMBER(38,0)",
        executed_volume="NUMBER(38,0)",
        cash="FLOAT",
        phase="VARCHAR",
    ),
    "option_expiration": dict(
        day="NUMBER(38,0)", asset_id="NUMBER(38,0)", exercise_price="FLOAT", premium="FLOAT"
    ),
    "l1_group_flows": dict(
        day="NUMBER(38,0)",
        time_tick="NUMBER(38,0)",
        asset_id="NUMBER(38,0)",
        agent_id="NUMBER(38,0)",
        agent_group="VARCHAR",
        cp_group="VARCHAR",
        volume_buy="FLOAT",
        volume_sell="FLOAT",
        cash_buy="FLOAT",
        cash_sell="FLOAT",
    ),
}

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _identifier(value: str) -> str:
    if not _IDENT.fullmatch(value):
        raise ValueError(f"Invalid SQL identifier: {value!r}")
    return value.upper()


class SnowflakeRepository:
    """Matches marketsim.database.repositories.interface.RepositoryInterface."""

    def __init__(
        self,
        *,
        output_dir: Path | str = "tmp",  # factory compatibility; unused by cloud storage
        database: str = "DAEDALUS",
        schema: str = "SIMULATION",
        warehouse: str = "DAEDALUS_WH",
        account: str | None = None,
        user: str | None = None,
        password: str | None = None,
        authenticator: str = "snowflake",
        role: str | None = None,
    ) -> None:
        self.database = _identifier(database)

        # Example:
        # run_20261021_190000 -> SIMULATION_RUN_20261021_190000
        run_id = Path(output_dir).name

        self.schema = _identifier(f"{schema}_{run_id}")
        # del output_dir
        # self.database = _identifier(database)
        # self.schema = _identifier(schema)
        self.warehouse = _identifier(warehouse)
        kwargs: dict[str, Any] = dict(
            account=account or os.environ["SNOWFLAKE_ACCOUNT"],
            user=user or os.environ["SNOWFLAKE_USER"],
            authenticator=authenticator,
            warehouse=self.warehouse,
            database=self.database,
            schema=self.schema,
            enable_server_session_keep_alive_auto_detection=False,
        )
        if authenticator == "snowflake":
            secret = password or os.environ.get("SNOWFLAKE_PASSWORD")
            if not secret:
                raise ValueError("Set SNOWFLAKE_PASSWORD or pass password explicitly")
            kwargs["password"] = secret
        if role:
            kwargs["role"] = role
        # self.connection = snowflake.connector.connect(**kwargs)
        # self.prepare_tables()
        # Connect to the database without assuming the run schema exists.
        kwargs.pop("schema", None)

        self.connection = snowflake.connector.connect(**kwargs)

        try:
            self._execute(f'CREATE SCHEMA "{self.database}"."{self.schema}"')
            self._execute(f'USE SCHEMA "{self.database}"."{self.schema}"')
            self.prepare_tables()
        except Exception:
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SnowflakeRepository":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self.connection.cursor() as cur:
            cur.execute(sql, params)

    def prepare_tables(self) -> None:
        for table, columns in TABLES.items():
            ddl = ", ".join(f'"{name.upper()}" {sql_type}' for name, sql_type in columns.items())
            self._execute(f'CREATE TABLE IF NOT EXISTS "{_identifier(table)}" ({ddl})')

    def _insert_one(self, table: str, values: dict[str, Any]) -> None:
        known = TABLES[table]
        unknown = set(values) - set(known)
        if unknown:
            raise ValueError(f"Unknown columns for {table}: {unknown}")
        columns = list(values)
        cols = ", ".join(f'"{_identifier(c)}"' for c in columns)
        placeholders = ", ".join(["%s"] * len(columns))
        self._execute(
            f'INSERT INTO "{_identifier(table)}" ({cols}) VALUES ({placeholders})',
            tuple(values.values()),
        )

    def _insert_frame(self, table: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        expected = list(TABLES[table])
        # normalized = df.copy()
        normalized = df.reset_index(drop=True).copy()
        normalized.columns = [str(c).lower() for c in normalized.columns]
        if normalized.columns.duplicated().any():
            raise ValueError(f"Duplicate DataFrame columns for {table}")
        extra = set(normalized.columns) - set(expected)
        if extra:
            raise ValueError(f"Unexpected columns for {table}: {extra}")
        # Missing optional columns are loaded as NULL, matching DuckDB's explicit NULLs.
        normalized = normalized.reindex(columns=expected)

        # Convert DataFrame columns to Snowflake-compatible types.
        for column, sql_type in TABLES[table].items():
            if sql_type == "FLOAT":
                normalized[column] = pd.to_numeric(
                    normalized[column],
                    errors="raise",
                ).astype("float64")

            elif sql_type.startswith("NUMBER"):
                normalized[column] = pd.to_numeric(
                    normalized[column],
                    errors="raise",
                ).astype("Int64")

            elif sql_type == "VARCHAR":
                normalized[column] = normalized[column].astype("string")

        normalized.columns = [c.upper() for c in expected]
        # normalized = normalized.reindex(columns=expected)
        # normalized.columns = [c.upper() for c in expected]
        success, _chunks, _rows, _output = write_pandas(
            self.connection,
            normalized,
            table_name=table.upper(),
            database=self.database,
            schema=self.schema,
            quote_identifiers=False,
            auto_create_table=False,
            overwrite=False,
        )
        if not success:
            raise RuntimeError(f"Snowflake write_pandas failed for {table}")

    def _read(self, sql: str, params: tuple[Any, ...], columns: list[str]) -> pd.DataFrame:
        with self.connection.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return pd.DataFrame.from_records(rows, columns=columns)

    def save_security(self, security: Security) -> None:
        self._insert_one(
            "securities",
            dict(
                asset_id=security.asset_id,
                instrument_class=security.instrument_class,
                market_type=security.market_type,
                name=security.name,
                eod_status=security.eod_status,
                reference_price=float(security.reference_price),
                currency=security.currency,
            ),
        )

    def save_index(self, index: Index) -> None:
        self._insert_one(
            "indices", dict(index_id=index.index_id, name=index.name, short_name=index.short_name)
        )

    def save_index_components(self, components_df: pd.DataFrame) -> None:
        self._insert_frame("index_components", components_df)

    def save_future(self, future: Future) -> None:
        self._insert_one(
            "futures",
            dict(
                contract_id=future.asset_id,
                underlying_type="index",
                underlying_id=future.underlying.asset_id,
                expiration_day=future.expiration,
                multiplier=float(future.multiplier),
                name=future.name,
                short_name=future.short_name,
            ),
        )

    # def save_agent(self, agent: Agent) -> None:
    # self._insert_one("agents", dict(agent_id=agent.agent_id, name=agent.name, group_name=agent.group, configuration=agent.configuration))
    def save_agent(self, agent: Agent) -> None:
        self._insert_one(
            table="agents",
            values=dict(
                agent_id=agent.agent_id,
                name=agent.name,
                group_name=agent.group,
                configuration=json.dumps(agent.configuration, default=str),
            ),
        )

    def save_position_history(self, position_history_df: pd.DataFrame) -> None:
        self._insert_frame("agent_position_history", position_history_df)

    def save_futures_position_history(self, position_history_df: pd.DataFrame) -> None:
        self._insert_frame("agent_futures_position_history", position_history_df)

    def save_futures_value_history(self, futures_value_df: pd.DataFrame) -> None:
        self._insert_frame("agent_futures_value_history", futures_value_df)

    def save_eod_position(self, eod_position_df: pd.DataFrame) -> None:
        self._insert_frame("agent_eod_positions", eod_position_df)

    def save_eod_futures_position(self, eod_position_df: pd.DataFrame) -> None:
        self._insert_frame("agent_eod_futures_position", eod_position_df)

    def save_eod_futures_value(self, eod_futures_value_df: pd.DataFrame) -> None:
        self._insert_frame("agent_eod_futures_value", eod_futures_value_df)

    def save_portfolio_value_history(self, portfolio_value_history_df: pd.DataFrame) -> None:
        self._insert_frame("agent_portfolio_value_history", portfolio_value_history_df)

    def save_eod_portfolio_value(self, eod_portfolio_value_df: pd.DataFrame) -> None:
        self._insert_frame("agent_eod_portfolio_value", eod_portfolio_value_df)

    def save_cash_history(self, cash_history_df: pd.DataFrame) -> None:
        self._insert_frame("agent_cash_history", cash_history_df)

    def save_eod_cash(self, eod_cash_df: pd.DataFrame) -> None:
        self._insert_frame("agent_eod_cash", eod_cash_df)

    def save_traded_prices(self, traded_prices_df: pd.DataFrame) -> None:
        self._insert_frame("traded_prices", traded_prices_df)

    def save_eod_prices(self, eod_prices_df: pd.DataFrame) -> None:
        self._insert_frame("eod_prices", eod_prices_df)

    def save_orders(self, orders_df: pd.DataFrame) -> None:
        self._insert_frame("orders", orders_df)

    def save_trades(self, trades_df: pd.DataFrame) -> None:
        self._insert_frame("trades", trades_df)

    def save_option_expiration(self, option_expiration_df: pd.DataFrame) -> None:
        self._insert_frame("option_expiration", option_expiration_df)

    def get_eod_positions(self, agent_id: int) -> pd.DataFrame:
        return self._read(
            "SELECT DAY, ASSET_ID, POSITION FROM AGENT_EOD_POSITIONS WHERE AGENT_ID = %s",
            (agent_id,),
            ["day", "asset_id", "position"],
        )

    def get_eod_futures_positions(self, agent_id: int) -> pd.DataFrame:
        return self._read(
            "SELECT DAY, ASSET_ID, POSITION FROM AGENT_EOD_FUTURES_POSITION WHERE AGENT_ID = %s",
            (agent_id,),
            ["day", "asset_id", "position"],
        )

    def get_eod_prices(self, asset_id: int) -> pd.DataFrame:
        return self._read(
            "SELECT DAY, OPEN, HIGH, LOW, CLOSE, VOLUME, THEORETICAL FROM EOD_PRICES WHERE ASSET_ID = %s",
            (asset_id,),
            ["day", "open", "high", "low", "close", "volume", "theoretical"],
        )

    def get_eod_index_values(self, index_id: int) -> pd.DataFrame:
        return self._read(
            "SELECT DAY, OPEN, HIGH, LOW, CLOSE FROM EOD_INDEX_VALUES WHERE INDEX_ID = %s",
            (index_id,),
            ["day", "open", "high", "low", "close"],
        )

    def get_eod_future_rates(self, contract_id: int) -> pd.DataFrame:
        return self._read(
            "SELECT DAY, OPEN, HIGH, LOW, CLOSE, VOLUME, THEORETICAL, LOP FROM EOD_FUTURE_RATES WHERE CONTRACT_ID = %s",
            (contract_id,),
            ["day", "open", "high", "low", "close", "volume", "theoretical", "lop"],
        )

    def get_eod_cash(self, agent_id: int) -> pd.DataFrame:
        return self._read(
            "SELECT DAY, CURRENCY, CASH FROM AGENT_EOD_CASH WHERE AGENT_ID = %s",
            (agent_id,),
            ["day", "currency", "cash"],
        )

    def get_eod_portfolio_values(self, agent_id: int) -> pd.DataFrame:
        return self._read(
            "SELECT DAY, PORTFOLIO_VALUE FROM AGENT_EOD_PORTFOLIO_VALUE WHERE AGENT_ID = %s",
            (agent_id,),
            ["day", "portfolio_value"],
        )

    ####### getting L1 layer data

    def get_group_flows(self, agent_id: int) -> pd.DataFrame:
        columns = [
            "day",
            "time_tick",
            "asset_id",
            "agent_id",
            "agent_group",
            "cp_group",
            "volume_buy",
            "volume_sell",
            "cash_buy",
            "cash_sell",
        ]
        return self._read(
            "SELECT DAY, TIME_TICK, ASSET_ID, AGENT_ID, AGENT_GROUP, CP_GROUP, "
            "VOLUME_BUY, VOLUME_SELL, CASH_BUY, CASH_SELL "
            "FROM L1_GROUP_FLOWS WHERE AGENT_ID = %s "
            "ORDER BY DAY, TIME_TICK, ASSET_ID, CP_GROUP",
            (agent_id,),
            columns,
        )

    def get_all_group_flows(self) -> pd.DataFrame:
        return self._read(
            "SELECT DAY, TIME_TICK, ASSET_ID, AGENT_GROUP, CP_GROUP, "
            "SUM(VOLUME_BUY), SUM(VOLUME_SELL), SUM(CASH_BUY), SUM(CASH_SELL) "
            "FROM L1_GROUP_FLOWS "
            "GROUP BY DAY, TIME_TICK, ASSET_ID, AGENT_GROUP, CP_GROUP "
            "ORDER BY AGENT_GROUP, DAY, TIME_TICK, ASSET_ID, CP_GROUP",
            (),
            [
                "day",
                "time_tick",
                "asset_id",
                "agent_group",
                "cp_group",
                "volume_buy",
                "volume_sell",
                "cash_buy",
                "cash_sell",
            ],
        )

    def get_execution_volumes_by_group(self) -> pd.DataFrame:
        columns = ["asset_id", "agent_group", "order_side", "executed_mode", "volume"]
        return self._read(EXECUTION_VOLUME_SQL, (), columns)

    ######### calculations

    def calculate_index_values(self, day: int) -> None:
        self._execute(
            """
            INSERT INTO INDEX_VALUES (DAY, TIME_TICK, INDEX_ID, OPEN, HIGH, LOW, CLOSE)
            SELECT TP.DAY, TP.TIME_TICK, IC.INDEX_ID,
                   SUM(IC.VALUATION_COEFFICIENT * TP.OPEN),
                   SUM(IC.VALUATION_COEFFICIENT * TP.CLOSE),
                   SUM(IC.VALUATION_COEFFICIENT * TP.CLOSE),
                   SUM(IC.VALUATION_COEFFICIENT * TP.CLOSE)
            FROM INDEX_COMPONENTS IC
            JOIN TRADED_PRICES TP ON IC.ASSET_ID = TP.ASSET_ID
            WHERE TP.DAY = %s
            GROUP BY TP.DAY, TP.TIME_TICK, IC.INDEX_ID
        """,
            (day,),
        )

    def calculate_eod_index_values(self, day: int) -> None:
        self._execute(
            """
            INSERT INTO EOD_INDEX_VALUES (DAY, INDEX_ID, OPEN, HIGH, LOW, CLOSE)
            SELECT DAY, INDEX_ID,
                   MIN_BY(OPEN, TIME_TICK),
                   MAX(CLOSE), MIN(CLOSE),
                   MAX_BY(CLOSE, TIME_TICK)
            FROM INDEX_VALUES
            WHERE DAY = %s
            GROUP BY DAY, INDEX_ID
        """,
            (day,),
        )

    ###################################
    ###  L1 calculations

    def calculate_group_flows(self) -> None:
        """Rebuild L1 group flows, atomically."""
        self._execute("BEGIN TRANSACTION")
        try:
            self._execute("DELETE FROM L1_GROUP_FLOWS")
            self._execute(GROUP_FLOW_INSERT_SQL)
            self._execute("COMMIT")
        except Exception:
            self._execute("ROLLBACK")
            raise
