import os
from getpass import getpass

import snowflake.connector

conn = snowflake.connector.connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    user=os.environ["SNOWFLAKE_USER"],
    password=getpass("Snowflake password: "),
    warehouse="DAEDALUS_WH",
    database="DAEDALUS",
    schema="SIMULATION",
    enable_server_session_keep_alive_auto_detection=False,
)

with conn.cursor() as cur:
    cur.execute("USE DATABASE DAEDALUS")
    cur.execute("USE SCHEMA SIMULATION")

    cur.execute("""
        SELECT
            CURRENT_DATABASE(),
            CURRENT_SCHEMA(),
            CURRENT_WAREHOUSE()
    """)

    print(cur.fetchone())


try:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                CURRENT_VERSION(),
                CURRENT_USER(),
                CURRENT_DATABASE(),
                CURRENT_SCHEMA(),
                CURRENT_WAREHOUSE()
        """)
        print(cur.fetchone())


finally:
    conn.close()
