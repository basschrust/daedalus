"""Select listed WIG20 options and their historical prices from Ikaros.

The initial selection excludes contracts expiring within the simulation horizon.
Expiry-matched historical benchmarking is intentionally deferred.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd


def _field(row: pd.Series, *names: str):
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    raise ValueError(f"Missing option metadata field; tried {names}")


def _side(value: object) -> str:
    value = str(value).strip().upper()
    if value in {"C", "CALL", "CE"}:
        return "CALL"
    if value in {"P", "PUT", "PE"}:
        return "PUT"
    raise ValueError(f"Unsupported option side {value!r}")


def select_options(
    conn,
    cutoff: date,
    simulation_dates: list[date],
    index_level: object,
    count: int = 4,
    lookback_sessions: int = 252,
) -> list[dict]:
    """Return options with metadata and OHLCV; prefer near-ATM liquid contracts.

    Requires DERIVATIVE_CONTRACTS columns CONTRACT_ID, SYMBOL,
    INSTRUMENT_TYPE, UNDERLYING_SYMBOL, EXPIRY_DATE, MULTIPLIER,
    and strike/side fields (recognized aliases below).
    """
    from marketsim.scenarios.wig20 import futures_history, query, to_price

    if count <= 0:
        return []
    if not simulation_dates:
        raise ValueError("Option selection requires simulation dates")

    contracts = query(
        conn,
        """
        SELECT * FROM DERIVATIVE_CONTRACTS
        WHERE UPPER(INSTRUMENT_TYPE) IN ('OPTION', 'INDEX_OPTION')
          AND UPPER(UNDERLYING_SYMBOL) = 'WIG20'
          AND EXPIRY_DATE > %s
        ORDER BY EXPIRY_DATE, CONTRACT_ID
    """,
        (simulation_dates[-1],),
    )
    if contracts.empty:
        print("Warning: no WIG20 options survive the simulation horizon")
        return []

    candidates = []
    for _, row in contracts.drop_duplicates(subset=["contract_id"]).iterrows():
        try:
            symbol = str(_field(row, "symbol", "contract_symbol"))
            strike = to_price(_field(row, "strike", "strike_price", "exercise_price"))
            side = _side(_field(row, "option_side", "call_put", "option_type", "put_call"))
            expiry = pd.Timestamp(_field(row, "expiry_date", "expiration_date")).date()
            multiplier = float(_field(row, "multiplier", "contract_multiplier"))
            if multiplier <= 0:
                raise ValueError("Invalid multiplier")
            history = futures_history(conn, row["contract_id"], cutoff, lookback_sessions)
            history = history.drop_duplicates("date").sort_values("date")
            # Only use positive-price observations; missing/non-trading sessions
            # are not filled with synthetic prices.
            if len(history) < 4 or history["date"].max().date() != cutoff:
                continue
            liquidity = float(history["volume"].tail(20).sum())
            candidates.append(
                {
                    "symbol": symbol,
                    "strike": strike,
                    "side": side,
                    "expiry_date": expiry,
                    "multiplier": multiplier,
                    "history": history,
                    "moneyness": abs(float(strike / Decimal(str(index_level))) - 1),
                    "liquidity": liquidity,
                }
            )
        except (ValueError, TypeError, KeyError) as exc:
            print(f"Skipping option {row.get('symbol', row.get('contract_id'))}: {exc}")

    # Select both calls and puts when available, then fill remaining slots.
    candidates.sort(key=lambda x: (x["moneyness"], -x["liquidity"], x["expiry_date"]))
    selected = []
    selected_symbols = set()
    for side in ("CALL", "PUT"):
        if len(selected) >= count:
            break
        match = next(
            (c for c in candidates if c["side"] == side and c["symbol"] not in selected_symbols),
            None,
        )
        if match is not None:
            selected.append(match)
            selected_symbols.add(match["symbol"])
    for candidate in candidates:
        if len(selected) >= count:
            break
        if candidate["symbol"] not in selected_symbols:
            selected.append(candidate)
            selected_symbols.add(candidate["symbol"])
    print(f"Selected {len(selected)} WIG20 options (requested {count})")
    return selected
