def test_eod_tables_have_consistent_prefix(repository):
    names = {name for (name,) in repository.connection.execute("SHOW TABLES").fetchall()}
    assert not any(name.startswith("agent_eod_") for name in names)
    assert {"eod_agent_positions", "eod_agent_cash", "eod_instrument_liquidity"} <= names


def test_eod_liquidity_rows(repository):
    row = dict(
        day=0,
        asset_id=10,
        instrument_class="stock",
        historical_daily_volume=1000.0,
        executed_volume=25,
        submitted_buy_volume=100,
        submitted_sell_volume=80,
        mm_submitted_buy_volume=70,
        mm_submitted_sell_volume=60,
        mm_executed_volume=20,
        mm_inventory_limit=200,
        mm_daily_volume_limit=150,
        mm_blocked_quote_calls=2,
    )
    repository.save_eod_instrument_liquidity([row])
    result = repository.connection.execute(
        "SELECT executed_volume, mm_blocked_quote_calls FROM eod_instrument_liquidity"
    ).fetchone()
    assert result == (25, 2)
