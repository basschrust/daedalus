"""Prepare a Daedalus 10-day GPW baseline scenario from Ikaros Snowflake prices.

Run with: python prepare_gpw_scenario.py --date-column DATE --ticker-column TICKER
       --close-column CLOSE --volume-column VOLUME
Credentials: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD.
Read-only against DAEDALUS.MARKET_DATA.PRICES.
"""
from __future__ import annotations
import argparse
import os
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import yaml

TABLE = 'DAEDALUS.MARKET_DATA.PRICES'
SYMBOLS = {'PKO': 'PKO', 'PKN': 'ORLEN', 'KGH': 'KGHM'}

def safe_identifier(value: str) -> str:
    if not value.isidentifier() or not value.replace('_','').isalnum():
        raise ValueError(f'Unsafe SQL column identifier: {value!r}')
    return value.upper()

def fetch_prices(conn, *, ticker_col: str, date_col: str, close_col: str,
                 volume_col: str, start: date, end: date, symbols: tuple[str, ...]) -> pd.DataFrame:
    ticker_col, date_col, close_col, volume_col = map(safe_identifier, (ticker_col, date_col, close_col, volume_col))
    placeholders = ', '.join(['%s'] * len(symbols))
    sql = f'''SELECT {ticker_col} AS TICKER, {date_col} AS SESSION_DATE,
                     {close_col} AS CLOSE, {volume_col} AS VOLUME
              FROM {TABLE}
              WHERE {ticker_col} IN ({placeholders})
                AND {date_col} BETWEEN %s AND %s
              ORDER BY {ticker_col}, {date_col}'''
    with conn.cursor() as cur:
        cur.execute(sql, (*symbols, start, end))
        return pd.DataFrame(cur.fetchall(), columns=['ticker', 'session_date', 'close', 'volume'])

def calibrate(prices: pd.DataFrame, symbols: tuple[str, ...]) -> dict:
    result = {}
    for symbol in symbols:
        df = prices.loc[prices.ticker == symbol].copy()
        df['session_date'] = pd.to_datetime(df['session_date'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        df = df.dropna(subset=['session_date', 'close']).sort_values('session_date')
        df = df.drop_duplicates('session_date', keep='last')
        if len(df) < 30 or (df['close'] <= 0).any():
            raise ValueError(f'{symbol}: expected >=30 valid daily closing prices')
        returns = df['close'].pct_change().dropna()
        result[symbol] = {
            'reference_price': round(float(df['close'].iloc[-1]), 2),
            'daily_volatility': float(returns.std()),
            'daily_mean_return': float(returns.mean()),
            'median_daily_volume': float(df['volume'].median()),
            'observations': len(df),
            'last_session': str(df['session_date'].iloc[-1].date()),
        }
    return result

def build_scenario(stats: dict, *, seed: int = 67) -> dict:
    markets = {}
    for symbol, values in stats.items():
        markets[symbol.lower()] = {
            'name': SYMBOLS.get(symbol, symbol), 'short_name': symbol,
            'instrument_class': 'stock', 'market_type': 'continuous',
            'reference_price': values['reference_price'], 'currency': 'PLN',
        }
    return {
        'seed': seed, 'days': 10, 'sim_time': 390, 'lob_plot_interval': 10000,
        'markets': markets,
        'agents': {
            'noise': {'agent_class': 'NoiseAgent', 'number': 12,
                      'config': {'lam': 0.15, 'mean_volume': 20, 'mean_spread': 1.0,
                                 'initial_cash': {'PLN': 1000000}}},
            'market_makers': {'agent_class': 'MMZOHAgent', 'number': 2},
            'hbl': {'agent_class': 'HBLAgent', 'number': 3},
        },
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    symbols = ["PKO.WA", "PKN.WA", "KGH.WA"]
    # parser.add_argument('--ticker-column', default='TICKER')
    parser.add_argument("--ticker-column", default="SYMBOL")
    parser.add_argument('--date-column', default='DATE')
    parser.add_argument('--close-column', default='CLOSE')
    parser.add_argument('--volume-column', default='VOLUME')
    parser.add_argument('--symbols', nargs=3, default=symbols)
    parser.add_argument('--end', type=date.fromisoformat, default=date(2026, 10, 23))
    parser.add_argument('--output', type=Path, default=Path('marketsim/input/gpw_historical_10d.yaml'))
    args = parser.parse_args()
    import snowflake.connector
    connection = snowflake.connector.connect(
        account=os.environ['SNOWFLAKE_ACCOUNT'], user=os.environ['SNOWFLAKE_USER'],
        password=os.environ['SNOWFLAKE_PASSWORD'], database='DAEDALUS', schema='MARKET_DATA',
        warehouse=os.environ.get('SNOWFLAKE_WAREHOUSE', 'DAEDALUS_WH'),
    )

    try:
        prices = fetch_prices(connection, ticker_col=args.ticker_column, date_col=args.date_column,
                              close_col=args.close_column, volume_col=args.volume_column,
                              start=args.end - timedelta(days=364), end=args.end,
                              symbols=tuple(args.symbols))
    finally:
        connection.close()
    stats = calibrate(prices, tuple(args.symbols))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(build_scenario(stats), sort_keys=False), encoding='utf-8')
    args.output.with_suffix('.calibration.yaml').write_text(yaml.safe_dump(stats, sort_keys=False), encoding='utf-8')
    print(f'Wrote {args.output}; calibration: {stats}')

if __name__ == '__main__':
    main()
