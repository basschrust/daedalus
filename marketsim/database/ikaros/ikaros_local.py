"""Stage a *read-only* Ikaros DuckDB snapshot into Daedalus's shared cache.

Ikaros never writes to Daedalus's cache and Daedalus never writes to Ikaros.
The database connection to Ikaros closes before the shared cache is opened.
DuckDB uses inter-process file locks: a concurrent Ikaros writer can make a
read-only connection impossible. In that case we fail cleanly; do not copy
an open database file, ignore its WAL, or claim that a partial import succeeded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import yaml

from marketsim.database.prepared_cache import required_cutoff

SOURCE_TABLES = (
    "PRICES",
    "INDEX_PRICES",
    "INDEX_CONSTITUENTS",
    "DERIVATIVE_CONTRACTS",
    "DERIVATIVE_PRICES",
)


@dataclass(frozen=True)
class LocalSnapshot:
    """A successfully committed source-data import, not a prepared scenario."""

    source: Path
    table_rows: dict[str, int]
    missing_tables: tuple[str, ...]


def _quoted_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def stage_local_ikaros(*, source_path: Path, cache_path: Path) -> LocalSnapshot:
    """Export Ikaros tables to temporary Parquet; import atomically to shared cache.

    Export preserves DECIMAL prices (pandas would coerce them to float). All
    source exports share a read transaction. Ikaros's connection is released
    BEFORE opening the Daedalus cache for writes. Source tables are namespaced
    as IKAROS_* so existing prepared_snapshots and simulation caches survive.
    """
    source_path = source_path.resolve()
    cache_path = cache_path.resolve()
    if source_path == cache_path:
        raise ValueError("Ikaros source and Daedalus cache must be different files")
    if not source_path.is_file():
        raise FileNotFoundError(f"Local Ikaros source not found: {source_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="ikaros_import_", dir=cache_path.parent) as temp:
        directory = Path(temp)
        exported: dict[str, Path] = {}
        try:
            with duckdb.connect(str(source_path), read_only=True) as source:
                source.execute("BEGIN TRANSACTION")
                available = {
                    str(item[0]).upper() for item in source.execute("SHOW TABLES").fetchall()
                }
                for name in SOURCE_TABLES:
                    if name not in available:
                        continue
                    target = directory / f"{name.lower()}.parquet"
                    source.execute(
                        f'COPY (SELECT * FROM "{name}") '
                        f"TO '{_quoted_path(target)}' (FORMAT PARQUET)"
                    )
                    exported[name] = target
                source.execute("COMMIT")
        except duckdb.Error as exc:
            raise RuntimeError(
                "Cannot read ikaros.duckdb. It may be held by a separate "
                "Ikaros writer process (DuckDB file lock). Stop or briefly "
                "pause ingestion, or have Ikaros publish a separate immutable "
                "snapshot for Daedalus. The existing cache was not changed. "
                f"Underlying error: {exc}"
            ) from exc

        if "PRICES" not in exported or "INDEX_PRICES" not in exported:
            missing = sorted({"PRICES", "INDEX_PRICES"} - exported.keys())
            raise ValueError(f"Ikaros lacks required price tables: {missing}")

        counts: dict[str, int] = {}
        # A single transaction keeps the previously prepared simulation intact
        # even if one source table is corrupt or a cache write fails.
        try:
            with duckdb.connect(str(cache_path)) as dest:
                dest.execute("BEGIN TRANSACTION")
                try:
                    for name in SOURCE_TABLES:
                        target = f"IKAROS_{name}"
                        if name not in exported:
                            dest.execute(f'DROP TABLE IF EXISTS "{target}"')
                            continue
                        dest.execute(
                            f'CREATE OR REPLACE TABLE "{target}" AS SELECT * FROM read_parquet(?)',
                            [str(exported[name])],
                        )
                        counts[name] = int(
                            dest.execute(f'SELECT COUNT(*) FROM "{target}"').fetchone()[0]
                        )
                    dest.execute(
                        """CREATE TABLE IF NOT EXISTS ikaros_import_manifest (
                            source_path VARCHAR, imported_at TIMESTAMP,
                            table_name VARCHAR, row_count BIGINT
                        )"""
                    )
                    dest.execute("DELETE FROM ikaros_import_manifest")
                    for table_name, row_count in counts.items():
                        dest.execute(
                            "INSERT INTO ikaros_import_manifest "
                            "VALUES (?, CURRENT_TIMESTAMP, ?, ?)",
                            [str(source_path), table_name, row_count],
                        )
                    dest.execute("COMMIT")
                except Exception:
                    dest.execute("ROLLBACK")
                    raise
        except duckdb.Error as exc:
            raise RuntimeError(
                "Cannot write shared market_cache.duckdb; another simulation or "
                "cache writer may be using it. Retry the import after that "
                f"process finishes. No partial import was committed: {exc}"
            ) from exc
    return LocalSnapshot(
        source=source_path,
        table_rows=counts,
        missing_tables=tuple(sorted(set(SOURCE_TABLES) - counts.keys())),
    )


# This adapter is used only *inside* marketsim/database. The existing scenario
# preparer is unchanged; its named Snowflake queries run against cached Ikaros
# tables after narrowly scoped SQL-dialect translation.
_TABLE_PATTERN = re.compile(
    r"\b(FROM|JOIN)\s+(PRICES|INDEX_PRICES|INDEX_CONSTITUENTS|"
    r"DERIVATIVE_CONTRACTS|DERIVATIVE_PRICES)\b",
    re.IGNORECASE,
)
_DATE_PATTERN = re.compile(r'\bTO_DATE\s*\(\s*("DATE")\s*\)', re.IGNORECASE)


def translate_cached_query(sql: str) -> str:
    """Translate the small set of established Ikaros queries to DuckDB SQL."""
    sql = _DATE_PATTERN.sub(r"CAST(\1 AS DATE)", sql)
    sql = _TABLE_PATTERN.sub(
        lambda match: f'{match.group(1)} "IKAROS_{match.group(2).upper()}"',
        sql,
    )
    return sql.replace("%s", "?")


class _CachedCursor:
    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor

    def __enter__(self) -> _CachedCursor:
        return self

    def __exit__(self, *_args: object) -> None:
        self._cursor.close()

    def execute(self, sql: str, params: tuple | list = ()) -> _CachedCursor:
        self._cursor.execute(translate_cached_query(sql), params)
        return self

    @property
    def description(self) -> Any:
        return self._cursor.description

    def fetchall(self) -> list[tuple]:
        return self._cursor.fetchall()


class CachedIkarosConnection:
    """Named-query compatible connection to the staged IKAROS_* tables."""

    def __init__(self, *, cache_path: Path) -> None:
        self._connection = duckdb.connect(str(cache_path), read_only=True)

    def __enter__(self) -> CachedIkarosConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def cursor(self) -> _CachedCursor:
        return _CachedCursor(self._connection.cursor())

    def close(self) -> None:
        self._connection.close()

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._connection


def _cached_tables(conn: duckdb.DuckDBPyConnection) -> set[str]:
    return {str(name).upper() for (name,) in conn.execute("SHOW TABLES").fetchall()}


def _columns(conn: duckdb.DuckDBPyConnection, name: str) -> set[str]:
    return {str(item[0]).upper() for item in conn.execute(f'DESCRIBE "IKAROS_{name}"').fetchall()}


def validate_wig20_snapshot(
    *, cache_path: Path, settings: dict, previous_yaml: Path | None = None
) -> tuple[date | None, list[str]]:
    """Preflight before updating a *prepared* WIG20 snapshot.

    Uses the latest stored index close, not an arbitrary max stock date.
    The old generated YAML may supply symbols for *diagnostics only*; it is
    never substituted for the missing authoritative constituent registry.
    """
    problems: list[str] = []
    with duckdb.connect(str(cache_path), read_only=True) as conn:
        tables = _cached_tables(conn)
        missing = [name for name in SOURCE_TABLES if f"IKAROS_{name}" not in tables]
        if missing:
            problems.append(f"Ikaros source tables absent: {', '.join(missing)}")
        if "IKAROS_INDEX_PRICES" not in tables or "IKAROS_PRICES" not in tables:
            return None, problems

        holidays = {date.fromisoformat(str(day)) for day in settings.get("non_trading_dates", [])}
        limit = required_cutoff(
            today=datetime.now(ZoneInfo("Europe/Warsaw")).date(),
            non_trading_dates=holidays,
        )
        result = conn.execute(
            'SELECT MAX(CAST("DATE" AS DATE)) FROM "IKAROS_INDEX_PRICES" '
            "WHERE UPPER(INDEX_SYMBOL) = 'WIG20' AND CLOSE IS NOT NULL "
            'AND CAST("DATE" AS DATE) <= ?',
            [limit],
        ).fetchone()
        cutoff = result[0] if result else None
        if cutoff is None:
            problems.append("WIG20 has no indexed close observations")
            return None, problems

        symbols: list[str] = []
        if "IKAROS_INDEX_CONSTITUENTS" in tables:
            required = {"INDEX_SYMBOL", "SECURITY_SYMBOL", "WEIGHT", "VALID_FROM", "VALID_TO"}
            missing_cols = required - _columns(conn, "INDEX_CONSTITUENTS")
            if missing_cols:
                problems.append(f"INDEX_CONSTITUENTS missing columns: {sorted(missing_cols)}")
            else:
                rows = conn.execute(
                    'SELECT SECURITY_SYMBOL FROM "IKAROS_INDEX_CONSTITUENTS" '
                    "WHERE INDEX_SYMBOL = 'WIG20' AND VALID_FROM <= ? "
                    "AND (VALID_TO IS NULL OR VALID_TO >= ?)",
                    [cutoff, cutoff],
                ).fetchall()
                symbols = [str(row[0]).upper() for row in rows]
                if len(symbols) != 20 or len(set(symbols)) != 20:
                    problems.append(f"WIG20 constituents at {cutoff}: {len(symbols)} (need 20)")
        elif previous_yaml and previous_yaml.is_file():
            previous = yaml.safe_load(previous_yaml.read_text(encoding="utf-8"))
            markets = previous.get("markets") or {}
            components = (markets.get("wig20") or {}).get("components") or []
            symbols = [
                str(markets.get(c.get("security"), {}).get("symbol", "")).upper()
                for c in components
            ]
            symbols = [symbol for symbol in symbols if symbol]
            problems.append(
                "Using previous WIG20 constituent list ONLY to diagnose prices; "
                "it is not accepted as current official index membership"
            )

        if symbols:
            latest_prices = conn.execute(
                'SELECT UPPER(SYMBOL), MAX(CAST("DATE" AS DATE)), COUNT(*) '
                'FROM "IKAROS_PRICES" WHERE CAST("DATE" AS DATE) <= ? '
                "AND UPPER(SYMBOL) IN "
                f"({', '.join('?' for _ in symbols)}) GROUP BY UPPER(SYMBOL)",
                [cutoff, *symbols],
            ).fetchall()
            found = {str(symbol): (end, count) for symbol, end, count in latest_prices}
            for symbol in sorted(set(symbols)):
                row = found.get(symbol)
                if row is None:
                    problems.append(f"{symbol}: no stored stock prices")
                elif row[0] != cutoff:
                    problems.append(f"{symbol}: last close {row[0]} (target {cutoff})")
                elif row[1] < int(settings.get("lookback_sessions", 252)):
                    problems.append(
                        f"{symbol}: {row[1]} stored sessions "
                        f"(need {settings.get('lookback_sessions', 252)})"
                    )

        if "IKAROS_DERIVATIVE_CONTRACTS" in tables:
            columns = _columns(conn, "DERIVATIVE_CONTRACTS")
            basics = {"CONTRACT_ID", "INSTRUMENT_TYPE", "UNDERLYING_SYMBOL", "EXPIRY_DATE"}
            if basics <= columns:
                futures = conn.execute(
                    'SELECT CONTRACT_ID FROM "IKAROS_DERIVATIVE_CONTRACTS" '
                    "WHERE UPPER(INSTRUMENT_TYPE)='FUTURE' "
                    "AND UPPER(UNDERLYING_SYMBOL)='WIG20' AND EXPIRY_DATE > ? "
                    "ORDER BY EXPIRY_DATE, CONTRACT_ID",
                    [cutoff],
                ).fetchall()
                future_ids = [str(record[0]) for record in futures]
                requested = int(settings.get("futures_count", 1))
                if len(future_ids) < requested:
                    problems.append(
                        f"WIG20 futures: {len(future_ids)} registered (requested {requested})"
                    )
                if "IKAROS_DERIVATIVE_PRICES" in tables and future_ids:
                    for contract_id in future_ids[:requested]:
                        record = conn.execute(
                            'SELECT MAX(CAST("DATE" AS DATE)) '
                            'FROM "IKAROS_DERIVATIVE_PRICES" WHERE CONTRACT_ID = ? '
                            'AND CAST("DATE" AS DATE) <= ?',
                            [contract_id, cutoff],
                        ).fetchone()
                        if record[0] != cutoff:
                            problems.append(
                                f"{contract_id}: last derivative close {record[0]} "
                                f"(target {cutoff})"
                            )
                if int(settings.get("options_count", 4)) > 0:
                    options = conn.execute(
                        'SELECT COUNT(*) FROM "IKAROS_DERIVATIVE_CONTRACTS" '
                        "WHERE UPPER(INSTRUMENT_TYPE) IN ('OPTION', 'INDEX_OPTION') "
                        "AND UPPER(UNDERLYING_SYMBOL)='WIG20' AND EXPIRY_DATE > ?",
                        [cutoff],
                    ).fetchone()[0]
                    if options == 0:
                        problems.append("No currently registered WIG20 option contracts")
            else:
                problems.append(f"DERIVATIVE_CONTRACTS missing columns: {sorted(basics - columns)}")
            missing_meta = {"SYMBOL", "MULTIPLIER"} - columns
            if missing_meta:
                problems.append(f"DERIVATIVE_CONTRACTS missing columns: {sorted(missing_meta)}")
            if int(settings.get("options_count", 4)) > 0:
                if not ({"STRIKE", "STRIKE_PRICE", "EXERCISE_PRICE"} & columns):
                    problems.append("DERIVATIVE_CONTRACTS lacks option strike metadata")
                if not ({"OPTION_SIDE", "CALL_PUT", "OPTION_TYPE", "PUT_CALL"} & columns):
                    problems.append("DERIVATIVE_CONTRACTS lacks call/put metadata")

        if "IKAROS_DERIVATIVE_PRICES" in tables:
            if int(settings.get("options_count", 4)) > 0 and (
                "IKAROS_DERIVATIVE_CONTRACTS" in tables
            ):
                # Warn when no options have the four real price observations
                # required by the existing select_options() implementation.
                enough = conn.execute(
                    "SELECT COUNT(*) FROM ("
                    'SELECT p.CONTRACT_ID FROM "IKAROS_DERIVATIVE_PRICES" p '
                    'JOIN "IKAROS_DERIVATIVE_CONTRACTS" c '
                    "ON c.CONTRACT_ID = p.CONTRACT_ID "
                    "WHERE UPPER(c.INSTRUMENT_TYPE) IN ('OPTION', 'INDEX_OPTION') "
                    "AND UPPER(c.UNDERLYING_SYMBOL) = 'WIG20' "
                    'AND CAST(p."DATE" AS DATE) <= ? '
                    'GROUP BY p.CONTRACT_ID HAVING COUNT(DISTINCT p."DATE") >= 4'
                    ") selected",
                    [cutoff],
                ).fetchone()[0]
                if enough == 0:
                    problems.append("No WIG20 option has four historical price sessions")
        return cutoff, problems
