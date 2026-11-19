import pandas as pd


def load_historical_ohlcv(
    connection,
    table: str,
    symbols: list[str],
    cutoff_date: str,
    lookback_days: int = 365,
) -> dict[str, pd.DataFrame]:

    from marketsim.database.ikaros_queries import fetch_legacy_historical_ohlcv

    df = fetch_legacy_historical_ohlcv(connection, table, cutoff_date, lookback_days)

    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])

    return {
        symbol: group.set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()
        for symbol, group in df.groupby("symbol")
        if symbol in symbols
    }
