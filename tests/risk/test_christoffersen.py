"""One-day exception transition diagnostics; no market data connection."""

import pandas as pd
import pytest

from marketsim.risk.christoffersen import christoffersen_report


def _observations(*, failures: list[bool], days: list[int] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": "historical",
            "agent_id": -1,
            "group_name": "BENCHMARK",
            "confidence_level": 0.95,
            "day": days if days is not None else list(range(len(failures))),
            "exception": failures,
        }
    )


def test_isolated_exceptions_are_not_counted_as_clusters() -> None:
    frame = _observations(failures=[False, True, False, False, True, False])
    report = christoffersen_report(observations=frame)
    row = report.iloc[0]
    assert (row.transitions_00, row.transitions_01, row.transitions_10, row.transitions_11) == (
        1,
        2,
        2,
        0,
    )
    assert row.lr_ind >= 0
    assert row.lr_cc >= row.lr_ind
    assert row.assessment == "LOW_POWER"


def test_missing_day_does_not_create_false_transition() -> None:
    frame = _observations(failures=[False, True, True], days=[0, 1, 3])
    row = christoffersen_report(observations=frame).iloc[0]
    assert row.transitions_01 == 1
    assert row.transitions_11 == 0


def test_duplicate_day_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        christoffersen_report(observations=_observations(failures=[False, True], days=[0, 0]))


def test_empty_observations_have_expected_columns() -> None:
    result = christoffersen_report(observations=pd.DataFrame())
    assert result.empty
    assert "p_value_cc" in result.columns
