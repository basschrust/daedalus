import numpy as np
import pytest

from marketsim.market.fundamental_return_regime import FundamentalReturnRegime


def regime(seed=17):
    return FundamentalReturnRegime(
        symbols=["B", "A"],
        mean_daily_returns={"A": 0.001, "B": -0.001},
        covariance_daily=np.array([[0.0004, 0.0001], [0.0001, 0.0009]]),
        seed=seed,
    )


def test_reproducible_and_order_independent():
    a = regime()
    b = FundamentalReturnRegime(
        symbols=["A", "B"],
        mean_daily_returns={"B": -0.001, "A": 0.001},
        covariance_daily=np.array([[0.0009, 0.0001], [0.0001, 0.0004]]),
        seed=17,
    )
    assert a.returns(9) == b.returns(9)
    assert a.returns(0) == b.returns(0)
    assert a.returns(9) == a.returns(9)


def test_cumulative_fundamental_value():
    r = regime()
    returns = r.returns(0)
    assert r.value(0, "A", 100) == pytest.approx(100 * np.exp(returns["A"]))


def test_reject_non_psd():
    with pytest.raises(ValueError, match="positive semidefinite"):
        FundamentalReturnRegime(
            symbols=["A", "B"],
            mean_daily_returns={"A": 0, "B": 0},
            covariance_daily=np.array([[1, 2], [2, 1]]),
        )


class FakeCursor:
    def __init__(self, responses):
        self.responses = iter(responses)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, params):
        assert params == ("cal-1",)
        self.current = next(self.responses)

    def fetchall(self):
        return self.current


class FakeConn:
    def __init__(self, responses):
        self.responses = responses

    def cursor(self):
        return FakeCursor(self.responses)


def test_loader():
    r = FundamentalReturnRegime.from_snowflake(
        FakeConn(
            [
                [("A", 0.001, 0.02), ("B", 0, 0.03)],
                [("A", "A", 0.0004), ("A", "B", 0.0001), ("B", "B", 0.0009)],
            ]
        ),
        "cal-1",
        symbols=["A", "B"],
    )
    assert r.covariance[0, 1] == pytest.approx(0.0001)
