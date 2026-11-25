"""Local, reusable Ikaros market snapshots for prepared GPW/WIG20 simulations.

The DuckDB stores *actual* observed OHLCV rows and an immutable copy of the
market structure; the generated YAML can be regenerated using today's agent
profile without downloading market data again. No synthetic prices are filled.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import yaml

from marketsim.scenarios.scenario_settings import build_scenario_settings

SOURCE_FIELDS = {
    "wig20": ("days", "lookback_sessions", "futures_count", "options_count"),
    "gpw": ("days", "lookback_sessions", "minimum_history_sessions", "margin_rate"),
}


def required_cutoff(*, today: date, non_trading_dates: set[date] | None = None) -> date:
    """Last completed Mon–Fri session before today's Warsaw calendar day.

    `non_trading_dates` should include GPW exchange holidays (not just weekends).
    """
    holidays = non_trading_dates or set()
    session = today - timedelta(days=1)
    while session.weekday() >= 5 or session in holidays:
        session -= timedelta(days=1)
    return session


def _digest(value: object) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def selection_signature(dataset: str, settings: dict, profile: dict) -> str:
    """Fingerprint fields that change the downloaded universe or expiry mapping.

    Agent tuning intentionally does not enter this fingerprint.
    """
    values = {key: settings.get(key) for key in SOURCE_FIELDS[dataset]}
    values["non_trading_dates"] = profile.get("non_trading_dates", [])
    return _digest(values)


def _profile_settings(input_dir: Path, settings: dict) -> dict:
    # New configuration: agents are already resolved through !include.
    if isinstance(settings.get("agents"), dict):
        return settings

    # Legacy GPW configuration: keep the existing behaviour.
    profile_name = settings.get("agents_profile", "gpw_50d.yaml")
    profile_path = (input_dir / profile_name).resolve()

    if not profile_path.is_relative_to(input_dir.resolve()):
        raise ValueError("agents_profile must be inside input/")

    with profile_path.open(encoding="utf-8") as stream:
        profile = yaml.safe_load(stream)

    if not isinstance(profile, dict):
        raise ValueError(f"Invalid scenario profile: {profile_path}")

    return profile


def _scenario_from_cached(
    dataset: str,
    snapshot: dict,
    profile: dict,
    settings: dict,
) -> dict:
    market_data = {key: snapshot[key] for key in ("markets", "historical_prices", "start_date")}

    return build_scenario_settings(
        settings=settings,
        market_data=market_data,
        profile=(None if isinstance(settings.get("agents"), dict) else profile),
    )


def _market_data_path(input_dir: Path, snapshot: dict) -> Path:
    setting = snapshot.get("historical_prices") or {}
    if not setting.get("enabled"):
        raise ValueError("Cached dataset must include historical_prices")
    raw = Path(str(setting.get("path", "")))
    if raw.is_absolute():
        raise ValueError("Cached market-data path must be relative to input/")
    resolved = (input_dir / raw).resolve()
    if not resolved.is_relative_to((input_dir / "market_data").resolve()):
        raise ValueError("Cached market data must be inside input/market_data/")
    return resolved


class PreparedDatasetCache:
    """A DuckDB-backed cache shared by --dataset wig20 and --dataset gpw."""

    def __init__(self, *, root: Path) -> None:
        self.input_dir = root / "input"
        self.data_dir = self.input_dir / "market_data"
        self.db_path = self.data_dir / "market_cache.duckdb"

    @staticmethod
    def _init_tables(conn: duckdb.DuckDBPyConnection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prepared_snapshots (
                dataset VARCHAR PRIMARY KEY,
                cutoff_date DATE NOT NULL,
                selection_hash VARCHAR NOT NULL,
                csv_sha VARCHAR NOT NULL,
                scenario_yaml VARCHAR NOT NULL,
                rows_count BIGINT NOT NULL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_ohlcv (
                dataset VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                session_date DATE NOT NULL,
                open VARCHAR, high VARCHAR, low VARCHAR, close VARCHAR, volume VARCHAR
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS greek_l2_cache (
                calibration_id VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                mean_daily_return DOUBLE,
                daily_sigma DOUBLE NOT NULL,
                PRIMARY KEY(calibration_id, symbol)
            )
        """)

    def _store(
        self,
        *,
        dataset: str,
        settings: dict,
        profile: dict,
        snapshot: dict,
        yaml_text: str,
        csv_path: Path,
    ) -> None:
        cutoff = date.fromisoformat(str(snapshot["historical_prices"]["cutoff_date"]))
        signature = selection_signature(dataset, settings, profile)
        csv_sha = _file_sha(csv_path)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(self.db_path)) as conn:
            self._init_tables(conn)
            # Parse the CSV before changing the previous, working snapshot.
            row = conn.execute(
                """
                SELECT COUNT(*), MAX(CAST(date AS DATE))
                FROM read_csv(?, all_varchar=true)
            """,
                [str(csv_path)],
            ).fetchone()
            if not row or not row[0] or row[1] != cutoff:
                raise ValueError(f"Market CSV has no rows through cutoff {cutoff}: {csv_path}")
            conn.execute("BEGIN TRANSACTION")
            try:
                conn.execute("DELETE FROM historical_ohlcv WHERE dataset = ?", [dataset])
                conn.execute(
                    """
                    INSERT INTO historical_ohlcv
                    SELECT ?, symbol, CAST(date AS DATE), open, high, low, close, volume
                    FROM read_csv(?, all_varchar=true)
                """,
                    [dataset, str(csv_path)],
                )
                conn.execute("DELETE FROM prepared_snapshots WHERE dataset = ?", [dataset])
                conn.execute(
                    """
                    INSERT INTO prepared_snapshots
                    (dataset, cutoff_date, selection_hash, csv_sha, scenario_yaml, rows_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    [dataset, cutoff, signature, csv_sha, yaml_text, row[0]],
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def _bootstrap(
        self, *, dataset: str, settings: dict, profile: dict, yaml_path: Path, minimum_cutoff: date
    ) -> bool:
        """Adopt already prepared, fresh YAML + CSV without Snowflake."""
        if not yaml_path.is_file():
            return False
        with yaml_path.open(encoding="utf-8") as stream:
            snapshot = yaml.safe_load(stream)
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("markets"), dict):
            return False
        history = snapshot.get("historical_prices") or {}
        try:
            cutoff = date.fromisoformat(str(history["cutoff_date"]))
        except (KeyError, ValueError):
            return False
        if cutoff < minimum_cutoff:
            return False
        if int(history.get("lookback_sessions", -1)) != int(settings.get("lookback_sessions", 252)):
            return False
        # A different universe cannot be recovered safely from a current CSV.
        if dataset == "wig20":
            markets = list(snapshot["markets"].values())
            futures = sum(m.get("instrument_class") == "future" for m in markets)
            options = sum(m.get("instrument_class") == "option" for m in markets)
            if futures != int(settings.get("futures_count", 1)):
                return False
            if options > int(settings.get("options_count", 4)):
                return False
        if int(snapshot.get("days", -1)) != int(settings.get("days", 50)):
            return False
        try:
            csv_path = _market_data_path(self.input_dir, snapshot)
            if not csv_path.is_file():
                return False
            self._store(
                dataset=dataset,
                settings=settings,
                profile=profile,
                snapshot=snapshot,
                yaml_text=yaml_path.read_text(encoding="utf-8"),
                csv_path=csv_path,
            )
        except (KeyError, ValueError, OSError):
            return False
        return True

    def use_if_fresh(
        self, *, dataset: str, settings: dict, yaml_name: str, today: date | None = None
    ) -> bool:
        """Reuse an existing snapshot, without opening any Snowflake connection.

        Files are restored from DuckDB as needed. If data is stale, returns False.
        Does not silently accept a stale snapshot even when Snowflake is unavailable.
        """
        profile = _profile_settings(self.input_dir, settings)
        if today is None:
            today = datetime.now(ZoneInfo("Europe/Warsaw")).date()
        holidays = {date.fromisoformat(str(d)) for d in profile.get("non_trading_dates", [])}
        holidays |= {date.fromisoformat(str(d)) for d in settings.get("non_trading_dates", [])}
        cutoff_needed = required_cutoff(today=today, non_trading_dates=holidays)
        yaml_path = self.input_dir / yaml_name
        # Bootstrap before entering the read transaction: opening another
        # writer connection to the same DuckDB would otherwise conflict.
        if self.db_path.is_file():
            with duckdb.connect(str(self.db_path)) as existing:
                self._init_tables(existing)
                present = (
                    existing.execute(
                        "SELECT 1 FROM prepared_snapshots WHERE dataset = ?", [dataset]
                    ).fetchone()
                    is not None
                )
        else:
            present = False
        if not present:
            if not self._bootstrap(
                dataset=dataset,
                settings=settings,
                profile=profile,
                yaml_path=yaml_path,
                minimum_cutoff=cutoff_needed,
            ):
                return False
        with duckdb.connect(str(self.db_path)) as conn:
            self._init_tables(conn)
            row = conn.execute(
                """
                SELECT cutoff_date, selection_hash, csv_sha, scenario_yaml, rows_count
                FROM prepared_snapshots WHERE dataset = ?
            """,
                [dataset],
            ).fetchone()
            if row is None:
                return False
            cutoff, signature, expected_csv_sha, yaml_text, expected_rows = row
            if cutoff < cutoff_needed:
                print(f"Cache stale: {dataset} cutoff={cutoff}, need {cutoff_needed}")
                return False
            if signature != selection_signature(dataset, settings, profile):
                print(f"Cache universe settings changed for {dataset}; refreshing")
                return False
            snapshot = yaml.safe_load(yaml_text)
            csv_path = _market_data_path(self.input_dir, snapshot)
            count, last_day = conn.execute(
                """
                SELECT COUNT(*), MAX(session_date) FROM historical_ohlcv WHERE dataset = ?
            """,
                [dataset],
            ).fetchone()
            if count != expected_rows or last_day != cutoff:
                print("Cache history incomplete; refreshing from Snowflake")
                return False
            if not csv_path.is_file() or _file_sha(csv_path) != expected_csv_sha:
                # DuckDB, not the easily deleted CSV, is the durable cache copy.
                rows = conn.execute(
                    """
                    SELECT symbol, CAST(session_date AS VARCHAR) AS date, open, high, low,
                           close, volume
                    FROM historical_ohlcv WHERE dataset = ?
                    ORDER BY symbol, session_date
                """,
                    [dataset],
                ).fetchdf()
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                rows.to_csv(csv_path, index=False)
                conn.execute(
                    "UPDATE prepared_snapshots SET csv_sha = ? WHERE dataset = ?",
                    [_file_sha(csv_path), dataset],
                )
            # Agent configuration is current, without needing to re-fetch prices.
            rebuilt = _scenario_from_cached(dataset, snapshot, profile, settings)
            yaml_path.write_text(yaml.safe_dump(rebuilt, sort_keys=False), encoding="utf-8")
            print(f"Using {dataset} DuckDB cache, cutoff={cutoff}, {count} OHLCV rows")
            return True

    def record_preparation(self, *, dataset: str, settings: dict, yaml_name: str) -> None:
        """Index the scenario produced by the existing Snowflake preparer."""
        profile = _profile_settings(self.input_dir, settings)
        yaml_path = self.input_dir / yaml_name
        yaml_text = yaml_path.read_text(encoding="utf-8")
        snapshot = yaml.safe_load(yaml_text)
        csv_path = _market_data_path(self.input_dir, snapshot)
        self._store(
            dataset=dataset,
            settings=settings,
            profile=profile,
            snapshot=snapshot,
            yaml_text=yaml_text,
            csv_path=csv_path,
        )

    def get_greek_calibration(self, *, calibration_id: str) -> list[tuple] | None:
        """Return immutable L2 statistics already downloaded for this calibration ID."""
        if not self.db_path.is_file():
            return None
        with duckdb.connect(str(self.db_path)) as conn:
            self._init_tables(conn)
            rows = conn.execute(
                """
                SELECT symbol, mean_daily_return, daily_sigma
                FROM greek_l2_cache WHERE calibration_id = ? ORDER BY symbol
            """,
                [calibration_id],
            ).fetchall()
        return rows or None

    def store_greek_calibration(self, *, calibration_id: str, rows: list[tuple]) -> None:
        """Persist a validated calibration for all future local simulation runs."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(self.db_path)) as conn:
            self._init_tables(conn)
            conn.execute("BEGIN TRANSACTION")
            try:
                existing = conn.execute(
                    """
                    SELECT symbol, mean_daily_return, daily_sigma
                    FROM greek_l2_cache WHERE calibration_id = ? ORDER BY symbol
                """,
                    [calibration_id],
                ).fetchall()
                normalized = sorted((str(sym).upper(), mean, sigma) for sym, mean, sigma in rows)
                if existing and existing != normalized:
                    raise ValueError(f"Immutable L2 calibration changed: {calibration_id}")
                if not existing:
                    conn.executemany(
                        """
                        INSERT INTO greek_l2_cache
                        (calibration_id, symbol, mean_daily_return, daily_sigma)
                        VALUES (?, ?, ?, ?)
                    """,
                        [(calibration_id, *record) for record in normalized],
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
