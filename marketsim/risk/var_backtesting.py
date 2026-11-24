"""Kupiec one-day unconditional coverage using actual agent equity P&L.

Actual P&L (APL) includes discretionary trading, hedging and settlement;
it is not a frozen-position hypothetical P&L (HPL) backtest.
"""

from __future__ import annotations

from math import isfinite, log, log1p

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2

REPORT_COLUMNS = [
    "method",
    "agent_id",
    "group_name",
    "confidence_level",
    "observations",
    "exceptions",
    "exception_rate",
    "expected_exceptions",
    "lr_uc",
    "p_value",
    "assessment",
]


def kupiec_uc(*, n: int, exceptions: int, confidence: float) -> tuple[float, float]:
    """Likelihood-ratio unconditional coverage statistic and chi-square(1) p-value."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if isinstance(exceptions, bool) or not isinstance(exceptions, int) or not 0 <= exceptions <= n:
        raise ValueError("exceptions must be an integer between 0 and n")
    if not isfinite(confidence) or not 0.5 < confidence < 1:
        raise ValueError("confidence must be in (0.5, 1)")
    p = 1 - confidence
    observed = exceptions / n
    null_ll = exceptions * log(p) + (n - exceptions) * log1p(-p)
    fitted_ll = float(xlogy(exceptions, observed) + xlogy(n - exceptions, 1 - observed))
    stat = max(0.0, 2 * (fitted_ll - null_ll))
    return stat, float(chi2.sf(stat, 1))


def one_day_apl(risk: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    """Compare EoD day t forecast to realized equity difference t -> t+1."""
    one = risk.loc[risk["horizon_days"] == 1].copy()
    if one.empty:
        return pd.DataFrame(columns=[*risk.columns, "actual_pnl", "loss", "exception"])
    today = equity[["day", "agent_id", "portfolio_value"]].rename(
        columns={"portfolio_value": "equity_t"}
    )
    next_day = equity[["day", "agent_id", "portfolio_value"]].copy()
    next_day["day"] -= 1
    next_day = next_day.rename(columns={"portfolio_value": "equity_t_plus_1"})
    paired = one.merge(today, on=["day", "agent_id"], how="inner", validate="many_to_one")
    paired = paired.merge(next_day, on=["day", "agent_id"], how="inner", validate="many_to_one")
    for column in ("var_amount", "equity_t", "equity_t_plus_1"):
        if not np.isfinite(pd.to_numeric(paired[column]).to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite {column} in backtesting observations")
    paired["actual_pnl"] = paired["equity_t_plus_1"] - paired["equity_t"]
    paired["loss"] = -paired["actual_pnl"]
    paired["exception"] = paired["loss"] > paired["var_amount"]
    return paired


def kupiec_report(observations: pd.DataFrame) -> pd.DataFrame:
    """Independent result per agent/method/confidence; no pooling of agents."""
    if observations.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)
    data: list[dict] = []
    keys = ["method", "agent_id", "group_name", "confidence_level"]
    for (method, agent_id, group, confidence), frame in observations.groupby(keys, sort=True):
        n = len(frame)
        x = int(frame["exception"].sum())
        level = float(confidence)
        statistic, p_value = kupiec_uc(n=n, exceptions=x, confidence=level)
        expected = n * (1 - level)
        # Asymptotic chi-square calibration is weak for very rare exceptions.
        if min(expected, n - expected) < 5:
            assessment = "LOW_POWER"
        else:
            assessment = "REJECT_5PCT" if p_value < 0.05 else "NOT_REJECTED_5PCT"
        data.append(
            dict(
                method=method,
                agent_id=int(agent_id),
                group_name=group,
                confidence_level=level,
                observations=n,
                exceptions=x,
                exception_rate=x / n,
                expected_exceptions=expected,
                lr_uc=statistic,
                p_value=p_value,
                assessment=assessment,
            )
        )
    return pd.DataFrame(data, columns=REPORT_COLUMNS)
