"""Optional fundamental anchor for agent limit prices."""

from __future__ import annotations

import math


def fundamental_reference(agent, market, fallback: float) -> float:
    """Return a daily fundamental price when a regime is attached to the agent.

    Only stocks are eligible. Other instruments retain their existing pricing.
    An absent regime preserves legacy behaviour.
    """
    regime = getattr(agent, "fundamental_return_regime", None)
    if regime is None or market.instrument_class != "stock":
        return fallback
    symbol = str(market.symbol).upper()
    if symbol not in regime.symbols:
        return fallback
    initial = getattr(agent, "fundamental_initial_prices", {}).get(symbol)
    if initial is None:
        return fallback
    value = float(regime.value(agent.current_day, symbol, float(initial)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid fundamental price for {symbol}: {value}")
    return value
