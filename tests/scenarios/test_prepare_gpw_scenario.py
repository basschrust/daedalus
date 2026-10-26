import pandas as pd
import pytest

from tools.prepare_gpw_scenario import build_scenario, calibrate, safe_identifier


def test_identifier_rejects_injection():
    with pytest.raises(ValueError):
        safe_identifier("DATE; DROP TABLE PRICES")


def test_calibration_and_scenario():
    dates = pd.bdate_range("2026-08-01", periods=40)
    data = pd.DataFrame(
        [
            {"ticker": s, "session_date": d, "close": 100 + i, "volume": 1000 + i}
            for s in ("PKO", "PKN", "KGH")
            for i, d in enumerate(dates)
        ]
    )
    stats = calibrate(data, ("PKO", "PKN", "KGH"))
    scenario = build_scenario(stats)
    assert scenario["days"] == 10
    assert len(scenario["markets"]) == 3
    assert scenario["markets"]["pko"]["reference_price"] == 139.0
    assert stats["PKN"]["observations"] == 40


def test_missing_history_rejected():
    with pytest.raises(ValueError, match="expected >=30"):
        calibrate(pd.DataFrame(columns=["ticker", "session_date", "close", "volume"]), ("PKO",))
