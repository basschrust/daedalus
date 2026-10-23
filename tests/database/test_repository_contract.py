"""Backend parity tests for Daedalus repositories.

DuckDB runs by default. Snowflake is opt-in via DAEDALUS_TEST_SNOWFLAKE=1.
Snowflake tests create and drop their own schema; use a dedicated test role/database.
"""

import os
from decimal import Decimal
from uuid import uuid4

import pandas as pd
import pytest

from marketsim.database.repositories.factory import create_repository


@pytest.fixture(params=["duckdb", "snowflake"])
def repository(request, tmp_path):
    driver = request.param
    output_dir = tmp_path / f"run_{uuid4().hex[:12]}"
    repo = None
    if driver == "snowflake":
        if os.getenv("DAEDALUS_TEST_SNOWFLAKE") != "1":
            pytest.skip("Set DAEDALUS_TEST_SNOWFLAKE=1 to enable Snowflake integration tests")
        required = ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD")
        missing = [key for key in required if not os.getenv(key)]
        if missing:
            pytest.skip(f"Missing Snowflake credentials: {missing}")
        config = {
            "database": os.getenv("SNOWFLAKE_TEST_DATABASE", "DAEDALUS"),
            "schema": os.getenv("SNOWFLAKE_TEST_SCHEMA_PREFIX", "PYTEST"),
            "warehouse": os.getenv("SNOWFLAKE_TEST_WAREHOUSE", "DAEDALUS_WH"),
        }
    else:
        config = {}

    try:
        repo = create_repository(driver=driver, output_dir=output_dir, **config)
        yield repo
    finally:
        if repo is not None:
            if driver == "snowflake":
                # Clean up ONLY the unique schema created for this test.
                # Ensure the role has CREATE/DROP SCHEMA privileges.
                try:
                    with repo.connection.cursor() as cur:
                        cur.execute("USE SCHEMA INFORMATION_SCHEMA")
                        cur.execute(f'DROP SCHEMA IF EXISTS "{repo.database}"."{repo.schema}"')
                finally:
                    repo.close()
            else:
                repo.connection.close()


def test_multicurrency_cash_roundtrip(repository):
    source = pd.DataFrame(
        [
            {"day": 0, "agent_id": 17, "currency": "PLN", "cash": Decimal("10000.25")},
            {"day": 0, "agent_id": 17, "currency": "USD", "cash": Decimal("-75.50")},
            {"day": 0, "agent_id": 17, "currency": "EUR", "cash": Decimal("0")},
            {"day": 1, "agent_id": 17, "currency": "PLN", "cash": Decimal("10125.25")},
            {"day": 0, "agent_id": 18, "currency": "USD", "cash": Decimal("999")},
        ]
    )
    repository.save_eod_cash(eod_cash_df=source)
    actual = repository.get_eod_cash(agent_id=17)
    assert set(actual.columns) == {"day", "currency", "cash"}
    assert len(actual) == 4
    actual = actual.sort_values(["day", "currency"]).reset_index(drop=True)
    expected = source.loc[source.agent_id == 17, ["day", "currency", "cash"]]
    expected = expected.sort_values(["day", "currency"]).reset_index(drop=True)
    assert actual["day"].astype(int).tolist() == expected["day"].tolist()
    assert actual["currency"].tolist() == expected["currency"].tolist()
    assert actual["cash"].astype(float).tolist() == pytest.approx(
        expected["cash"].astype(float).tolist()
    )


def test_eod_prices_with_nullable_theoretical(repository):
    source = pd.DataFrame(
        [
            {
                "day": 0,
                "asset_id": 41,
                "open": Decimal("101.25"),
                "high": 102.0,
                "low": 100.5,
                "close": 101.75,
                "volume": 200,
                "theoretical": None,
            },
            {
                "day": 1,
                "asset_id": 41,
                "open": 101.75,
                "high": 104.0,
                "low": 101.0,
                "close": 103.5,
                "volume": 150,
                "theoretical": 103.25,
            },
            {
                "day": 0,
                "asset_id": 42,
                "open": 5.0,
                "high": 6.0,
                "low": 4.0,
                "close": 5.5,
                "volume": 10,
                "theoretical": 5.25,
            },
        ]
    )
    repository.save_eod_prices(eod_prices_df=source)
    actual = repository.get_eod_prices(asset_id=41).sort_values("day").reset_index(drop=True)
    assert len(actual) == 2
    assert actual["day"].astype(int).tolist() == [0, 1]
    assert actual["close"].astype(float).tolist() == pytest.approx([101.75, 103.5])
    assert pd.isna(actual.loc[0, "theoretical"])
    assert float(actual.loc[1, "theoretical"]) == pytest.approx(103.25)


def test_portfolio_value_roundtrip_and_agent_filter(repository):
    source = pd.DataFrame(
        [
            {"day": 0, "agent_id": 17, "portfolio_value": 12345.75},
            {"day": 1, "agent_id": 17, "portfolio_value": -25.5},
            {"day": 0, "agent_id": 18, "portfolio_value": 9999.0},
        ]
    )
    repository.save_eod_portfolio_value(eod_portfolio_value_df=source)
    actual = repository.get_eod_portfolio_values(agent_id=17).sort_values("day")
    assert actual["day"].astype(int).tolist() == [0, 1]
    assert actual["portfolio_value"].astype(float).tolist() == pytest.approx([12345.75, -25.5])


def test_no_data_for_unknown_agent(repository):
    actual = repository.get_eod_cash(agent_id=999999)
    assert actual.empty
    assert set(actual.columns) == {"day", "currency", "cash"}
