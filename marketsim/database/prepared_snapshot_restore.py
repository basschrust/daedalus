"""Restore a generated scenario's market CSV from its matching DuckDB cache.

The cached market prices are actual historical observations. Never create
synthetic fills, mix histories from different prepared scenarios, or modify
Ikaros's source database.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import duckdb
import yaml

COLUMNS = ("symbol", "date", "open", "high", "low", "close", "volume")


def restore_historical_csv(
    *,
    cache_path: Path,
    output_path: Path,
    dataset: str,
    scenario: dict,
) -> bool:
    """Restore missing CSV if cache snapshot describes exactly this market setup.

    Reuse is independent of *freshness*: a user explicitly launching a
    dated, generated YAML must get the same observations as that saved run.
    A stale snapshot is not silently treated as current market data.
    """
    if not cache_path.is_file():
        return False

    with duckdb.connect(str(cache_path), read_only=True) as connection:
        tables = {str(row[0]).lower() for row in connection.execute("SHOW TABLES").fetchall()}
        if not {"prepared_snapshots", "historical_ohlcv"}.issubset(tables):
            return False

        entry = connection.execute(
            """SELECT cutoff_date, scenario_yaml, rows_count
               FROM prepared_snapshots WHERE dataset = ?""",
            [dataset],
        ).fetchone()
        if entry is None:
            return False

        cutoff, yaml_text, expected_rows = entry
        try:
            cached = yaml.safe_load(yaml_text)
            count = int(expected_rows)
        except (yaml.YAMLError, TypeError, ValueError):
            return False
        if not isinstance(cached, dict) or count < 1:
            return False

        # Agents and rendering settings may be edited independently. The
        # instruments, historical cutoff and simulated dates must match.
        for field in ("markets", "historical_prices", "start_date"):
            if cached.get(field) != scenario.get(field):
                return False

        settings = scenario.get("historical_prices") or {}
        if str(settings.get("cutoff_date")) != str(cutoff):
            return False

        rows = connection.execute(
            """SELECT symbol, CAST(session_date AS VARCHAR),
                      open, high, low, close, volume
               FROM historical_ohlcv WHERE dataset = ?
               ORDER BY symbol, session_date""",
            [dataset],
        ).fetchall()
        if len(rows) != count or not rows:
            return False
        if max(str(row[1]) for row in rows) != str(cutoff):
            return False
        if any(len(row) != 7 or any(item is None for item in row) for row in rows):
            return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=".wig20_restore_",
            suffix=".csv",
            dir=output_path.parent,
            delete=False,
        ) as stream:
            name = stream.name
            writer = csv.writer(stream)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        os.replace(name, output_path)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)
    return True
