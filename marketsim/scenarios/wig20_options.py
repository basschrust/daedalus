"""Select listed WIG20 options and their observed historical prices from Ikaros.

Prefer contracts surviving the simulation horizon, but do not silently remove
all option markets when such contracts are unavailable. Shorter-lived contracts
retain their actual expiration dates and expire normally during simulation.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd

from marketsim.database.ikaros import ikaros_queries
from marketsim.scenarios.wig20_data import futures_history, to_price


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
    """Select liquid near-ATM options, preferring longer remaining maturity.

    Unlike a daily stock price feed, derivative trade history can legitimately
    have no row on the historical cutoff date. Do not require a trade exactly on
    `cutoff`; we only use observations actually stored in Ikaros.
    """
    if count <= 0:
        return []
    if not simulation_dates:
        raise ValueError("Option selection requires simulation dates")

    contracts = ikaros_queries.fetch_wig20_option_contracts(conn, cutoff)
    if contracts.empty:
        raise ValueError(
            "Ikaros returned no WIG20 option contracts active at the historical cutoff "
            f"{cutoff}; requested {count} options."
        )

    candidates: list[dict] = []
    for _, row in contracts.drop_duplicates(subset=["contract_id"]).iterrows():
        try:
            symbol = str(_field(row, "symbol", "contract_symbol"))
            strike = to_price(_field(row, "strike", "strike_price", "exercise_price"))
            side = _side(_field(row, "option_side", "call_put", "option_type", "put_call"))
            expiry = pd.Timestamp(_field(row, "expiry_date", "expiration_date")).date()
            if expiry <= cutoff:
                continue
            multiplier = float(_field(row, "multiplier", "contract_multiplier"))
            if multiplier <= 0:
                raise ValueError("Invalid multiplier")

            history = futures_history(conn, row["contract_id"], cutoff, lookback_sessions)
            history = history.drop_duplicates("date").sort_values("date")
            if len(history) < 4:
                print(
                    f"Skipping option {symbol}: only {len(history)} observed price sessions "
                    "(minimum four)"
                )
                continue
            # The query must not return future data, but there is no requirement
            # that the option traded on cutoff itself (thin markets often don't).
            last_observation = pd.Timestamp(history["date"].max()).date()
            if last_observation > cutoff:
                raise ValueError(f"History extends past cutoff: {last_observation}")
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

    if not candidates:
        raise ValueError(
            "No WIG20 options qualified: check Ikaros derivative contracts and "
            f"their price history at cutoff {cutoff} (minimum four observed sessions)."
        )

    horizon_end = simulation_dates[-1]

    def selection_order(candidate: dict) -> tuple:
        expiry = candidate["expiry_date"]
        if expiry > horizon_end:
            # A full-horizon contract: prioritize proximity to ATM and liquidity.
            return (0, candidate["moneyness"], -candidate["liquidity"], expiry.toordinal())
        # A short-lived contract: prefer those surviving more simulation days.
        return (1, -expiry.toordinal(), candidate["moneyness"], -candidate["liquidity"])

    candidates.sort(key=selection_order)
    selected: list[dict] = []
    selected_symbols: set[str] = set()
    # Preserve both option sides if they are available.
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
    full_horizon = sum(c["expiry_date"] > horizon_end for c in selected)
    print(
        f"Selected {len(selected)} WIG20 options (requested {count}); "
        f"{full_horizon} survive all {len(simulation_dates)} simulation sessions"
    )
    return selected
