import pandas as pd


def load_historical_ohlcv(
    connection,
    table: str,
    symbols: list[str],
    cutoff_date: str,
    lookback_days: int = 365,
) -> dict[str, pd.DataFrame]:

    query = f"""
        SELECT
            SYMBOL,
            DATE,
            OPEN,
            HIGH,
            LOW,
            CLOSE,
            VOLUME
        FROM {table}
        WHERE DATE <= DATEADD(day, 0, %s)
          AND DATE > DATEADD(day, -%s, %s)
        ORDER BY SYMBOL, DATE
    """

    df = pd.read_sql(
        query,
        connection,
        params=(cutoff_date, lookback_days, cutoff_date),
    )

    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])

    return {
        symbol: group.set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()
        for symbol, group in df.groupby("symbol")
        if symbol in symbols
    }
