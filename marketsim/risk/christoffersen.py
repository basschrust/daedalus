"""Christoffersen one-day exception clustering and conditional-coverage diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2

from marketsim.risk.var_backtesting import kupiec_uc

COLUMNS = [
    "method",
    "agent_id",
    "group_name",
    "confidence_level",
    "observations",
    "exceptions",
    "transitions_00",
    "transitions_01",
    "transitions_10",
    "transitions_11",
    "lr_ind",
    "p_value_ind",
    "lr_cc",
    "p_value_cc",
    "assessment",
    "sample_warning",
]


def _log_binomial(*, successes: int, failures: int, probability: float) -> float:
    """Log Bernoulli likelihood with 0*log(0) handled correctly."""
    return float(xlogy(successes, probability) + xlogy(failures, 1 - probability))


def christoffersen_report(*, observations: pd.DataFrame) -> pd.DataFrame:
    """Independence (df=1) and coverage (df=2) per independent daily exception series.

    This is based on one-day consecutive observations. Gaps are explicitly
    excluded from the transition sample rather than treated as adjacent days.
    """
    if observations.empty:
        return pd.DataFrame(columns=COLUMNS)
    required = {"method", "agent_id", "group_name", "confidence_level", "day", "exception"}
    if missing := required - set(observations.columns):
        raise ValueError(f"Missing Christoffersen observation columns: {sorted(missing)}")
    records = []
    keys = ["method", "agent_id", "group_name", "confidence_level"]
    for (method, agent_id, group, level), group_frame in observations.groupby(keys, sort=True):
        ordered = group_frame.sort_values("day")
        if ordered.day.duplicated().any():
            raise ValueError("Duplicate days in Christoffersen exception series")
        days = ordered.day.astype(int).tolist()
        flags = ordered.exception.tolist()
        if any(
            not isinstance(flag, (bool, int, np.bool_, np.integer))
            or flag not in (0, 1, False, True)
            for flag in flags
        ):
            raise ValueError("Invalid Christoffersen exception flag")
        n = len(flags)
        failures = sum(bool(x) for x in flags)
        n00 = n01 = n10 = n11 = 0
        for i in range(1, n):
            if days[i] != days[i - 1] + 1:
                continue
            previous, current = bool(flags[i - 1]), bool(flags[i])
            if not previous and not current:
                n00 += 1
            elif not previous and current:
                n01 += 1
            elif previous and not current:
                n10 += 1
            else:
                n11 += 1
        count0 = n00 + n01
        count1 = n10 + n11
        transitions = count0 + count1
        if transitions:
            pi = (n01 + n11) / transitions
            pi0 = n01 / count0 if count0 else 0.0
            pi1 = n11 / count1 if count1 else 0.0
            ll_iid = _log_binomial(successes=n01 + n11, failures=n00 + n10, probability=pi)
            ll_markov = _log_binomial(successes=n01, failures=n00, probability=pi0)
            ll_markov += _log_binomial(successes=n11, failures=n10, probability=pi1)
            lr_ind = max(0.0, 2 * (ll_markov - ll_iid))
            p_ind = float(chi2.sf(lr_ind, 1))
            lr_uc, _ = kupiec_uc(n=n, exceptions=failures, confidence=float(level))
            lr_cc = lr_uc + lr_ind
            p_cc = float(chi2.sf(lr_cc, 2))
        else:
            lr_ind = p_ind = lr_cc = p_cc = float("nan")
        expected = n * (1 - float(level))
        # Particularly unreliable asymptotics for sparse breach transitions.
        warning = "LOW_POWER" if expected < 5 or failures < 5 or transitions < 40 else ""
        assessment = (
            "LOW_POWER" if warning else ("REJECT_5PCT" if p_cc < 0.05 else "NOT_REJECTED_5PCT")
        )
        records.append(
            dict(
                method=method,
                agent_id=int(agent_id),
                group_name=group,
                confidence_level=float(level),
                observations=n,
                exceptions=failures,
                transitions_00=n00,
                transitions_01=n01,
                transitions_10=n10,
                transitions_11=n11,
                lr_ind=lr_ind,
                p_value_ind=p_ind,
                lr_cc=lr_cc,
                p_value_cc=p_cc,
                assessment=assessment,
                sample_warning=warning,
            )
        )
    return pd.DataFrame(records, columns=COLUMNS)
