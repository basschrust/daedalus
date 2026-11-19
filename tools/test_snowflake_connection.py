"""Manual Snowflake connection diagnostic; SQL lives under marketsim.database."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marketsim.database.inspection import test_snowflake_connection

if __name__ == "__main__":
    test_snowflake_connection()
