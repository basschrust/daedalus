from marketsim.database.connectors.snowflake_storage import SnowflakeRepository

with SnowflakeRepository(
    database="DAEDALUS",
    schema="SIMULATION",
    warehouse="DAEDALUS_WH",
) as repo:
    print("Snowflake repository initialized successfully!")
