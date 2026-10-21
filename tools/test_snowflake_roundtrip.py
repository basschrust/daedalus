import pandas as pd

from marketsim.database.connectors.snowflake_storage import SnowflakeRepository

with SnowflakeRepository(
    database="DAEDALUS",
    schema="SIMULATION",
    warehouse="DAEDALUS_WH",
) as repo:

    df = pd.DataFrame([
        {
            "day": 1,
            "agent_id": 999,
            "cash": 10000.0,
            "currency": "PLN",
        },
        {
            "day": 1,
            "agent_id": 999,
            "cash": 2500.0,
            "currency": "USD",
        },
    ])

    repo.save_eod_cash(df)

    result = repo.get_eod_cash(agent_id=999)
    print(result)
