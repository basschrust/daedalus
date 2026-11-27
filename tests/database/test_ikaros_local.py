"""Local Ikaros -> shared market_cache ingestion regressions."""

from datetime import date
from decimal import Decimal

import duckdb
import pytest
import yaml

from marketsim.database.ikaros.ikaros_local import (
    CachedIkarosConnection,
    stage_local_ikaros,
    translate_cached_query,
    validate_wig20_snapshot,
)


def test_cached_query_translation():
    sql = (
        'SELECT TO_DATE("DATE") AS DATE, "CLOSE" FROM INDEX_PRICES '
        'WHERE "INDEX_SYMBOL" = %s AND TO_DATE("DATE") <= %s'
    )
    result = translate_cached_query(sql)
    assert 'CAST("DATE" AS DATE)' in result
    assert 'FROM "IKAROS_INDEX_PRICES"' in result
    assert result.count("?") == 2
    assert "%s" not in result


def _source_db(path):
    with duckdb.connect(str(path)) as con:
        con.execute(
            'CREATE TABLE PRICES (SYMBOL VARCHAR, "DATE" DATE, '
            '"OPEN" DECIMAL(24,8), "HIGH" DECIMAL(24,8), '
            '"LOW" DECIMAL(24,8), "CLOSE" DECIMAL(24,8), "VOLUME" BIGINT)'
        )
        con.execute(
            "INSERT INTO PRICES VALUES "
            "('PZU.WA', '2026-10-13', 70.01,70.02,70.00,70.01,100), "
            "('ALE.WA', '2026-11-20', 67.95,67.99,67.90,67.95,200)"
        )
        con.execute(
            'CREATE TABLE INDEX_PRICES (INDEX_SYMBOL VARCHAR, "DATE" DATE, '
            '"OPEN" DECIMAL(24,8), "HIGH" DECIMAL(24,8), '
            '"LOW" DECIMAL(24,8), "CLOSE" DECIMAL(24,8))'
        )
        con.execute("INSERT INTO INDEX_PRICES VALUES ('WIG20','2026-11-20',4200,4227,4199,4226.28)")
        con.execute(
            "CREATE TABLE DERIVATIVE_CONTRACTS (CONTRACT_ID VARCHAR, "
            "INSTRUMENT_TYPE VARCHAR, UNDERLYING_SYMBOL VARCHAR, EXPIRY_DATE DATE)"
        )
        con.execute(
            "INSERT INTO DERIVATIVE_CONTRACTS VALUES "
            "('FW20Z2620','FUTURE','WIG20','2026-12-18'), "
            "('OW20L264250','OPTION','WIG20','2026-12-18')"
        )
        con.execute(
            'CREATE TABLE DERIVATIVE_PRICES (CONTRACT_ID VARCHAR, "DATE" DATE, '
            '"OPEN" DECIMAL(24,8), "HIGH" DECIMAL(24,8), '
            '"LOW" DECIMAL(24,8), "CLOSE" DECIMAL(24,8), "VOLUME" BIGINT)'
        )
        con.execute(
            "INSERT INTO DERIVATIVE_PRICES VALUES ('FW20Z2620','2026-11-20',4200,4201,4199,4200,50)"
        )


def test_import_keeps_prepared_cache_and_preserves_decimal(tmp_path):
    source = tmp_path / "ikaros.duckdb"
    cache = tmp_path / "market_cache.duckdb"
    _source_db(source)
    with duckdb.connect(str(cache)) as con:
        con.execute("CREATE TABLE prepared_snapshots (label VARCHAR)")
        con.execute("INSERT INTO prepared_snapshots VALUES ('original')")

    result = stage_local_ikaros(source_path=source, cache_path=cache)
    assert result.table_rows["PRICES"] == 2
    assert "INDEX_CONSTITUENTS" in result.missing_tables
    with duckdb.connect(str(cache), read_only=True) as con:
        assert con.execute("SELECT * FROM prepared_snapshots").fetchall() == [("original",)]
        close = con.execute("SELECT CLOSE FROM IKAROS_PRICES WHERE SYMBOL = 'ALE.WA'").fetchone()[0]
        assert close == Decimal("67.95000000")
        assert con.execute("SELECT COUNT(*) FROM ikaros_import_manifest").fetchone()[0] == 4

    with CachedIkarosConnection(cache_path=cache) as local:
        with local.cursor() as cursor:
            cursor.execute('SELECT "CLOSE" FROM PRICES WHERE SYMBOL = %s', ("ALE.WA",))
            assert cursor.fetchall()[0][0] == Decimal("67.95000000")


def test_incomplete_wig20_does_not_replace_prepared_snapshot(tmp_path):
    source = tmp_path / "ikaros.duckdb"
    cache = tmp_path / "market_cache.duckdb"
    previous = tmp_path / "wig20_auto.yaml"
    _source_db(source)
    previous.write_text(
        yaml.safe_dump(
            {
                "markets": {
                    "wig20": {"components": [{"security": "pzu.wa"}]},
                    "pzu.wa": {"symbol": "PZU.WA"},
                }
            }
        ),
        encoding="utf-8",
    )
    stage_local_ikaros(source_path=source, cache_path=cache)
    cutoff, issues = validate_wig20_snapshot(
        cache_path=cache,
        settings={"futures_count": 3, "lookback_sessions": 252},
        previous_yaml=previous,
    )
    assert cutoff == date(2026, 11, 20)
    assert any("INDEX_CONSTITUENTS" in issue for issue in issues)
    assert any("PZU.WA: last close 2026-10-13" in issue for issue in issues)
    assert any("DERIVATIVE_CONTRACTS missing columns" in issue for issue in issues)


def test_missing_index_prices_does_not_modify_existing_cache(tmp_path):
    source = tmp_path / "ikaros.duckdb"
    cache = tmp_path / "market_cache.duckdb"
    with duckdb.connect(str(source)) as con:
        con.execute("CREATE TABLE PRICES (SYMBOL VARCHAR)")
    with duckdb.connect(str(cache)) as con:
        con.execute("CREATE TABLE prepared_snapshots (label VARCHAR)")
        con.execute("INSERT INTO prepared_snapshots VALUES ('original')")
    with pytest.raises(ValueError, match="INDEX_PRICES"):
        stage_local_ikaros(source_path=source, cache_path=cache)
    with duckdb.connect(str(cache)) as con:
        assert con.execute("SELECT * FROM prepared_snapshots").fetchall() == [("original",)]
        assert "IKAROS_PRICES" not in {
            str(row[0]).upper() for row in con.execute("SHOW TABLES").fetchall()
        }
