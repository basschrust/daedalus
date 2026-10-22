"""Regression tests for Daedalus multicurrency agent cash plots.

Run from the project root:
    python -m pytest tests/plot/test_multicurrency_cash.py -v
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from marketsim.plot.simple_plot import plot_agent_history_many_markets


def _plot(tmp_path, cash_history, monkeypatch):
    """Capture per-currency line data and render the real PNG."""
    lines = {}
    original_plot = plt.Axes.plot

    def spy_plot(self, *args, **kwargs):
        label = kwargs.get("label")
        if label in {"PLN", "USD", "EUR"}:
            lines[label] = {
                "x": list(args[0]),
                "y": list(args[1]),
            }
        return original_plot(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(plt.Axes, "plot", spy_plot)
        target = tmp_path / "agent_multicurrency.png"
        plot_agent_history_many_markets(
            position_history=pd.DataFrame({
                "time_step": [0, 1, 2, 3],
                "asset_id": [1, 1, 1, 1],
                "position": [10, 12, 11, 10],
            }),
            cash_history=cash_history,
            value_history=pd.DataFrame({
                "time_step": [0, 1, 2, 3],
                "portfolio_value": [11000, 11200, 11100, 10900],
            }),
            output_file=str(target),
        )
    assert target.is_file() and target.stat().st_size > 0
    return lines


@pytest.fixture
def balances():
    return pd.DataFrame([
        {"time_step": 0, "currency": "PLN", "cash": 10000},
        {"time_step": 0, "currency": "USD", "cash": 0},
        {"time_step": 0, "currency": "EUR", "cash": 0},
        {"time_step": 1, "currency": "PLN", "cash": 10200},
        {"time_step": 1, "currency": "USD", "cash": 50},
        {"time_step": 1, "currency": "EUR", "cash": 0},
        {"time_step": 2, "currency": "PLN", "cash": 10100},
        {"time_step": 2, "currency": "USD", "cash": 75},
        {"time_step": 2, "currency": "EUR", "cash": 20},
    ])

################################################################
####  test functions
################################################################


def test_each_currency_has_its_own_line(tmp_path, monkeypatch, balances):
    lines = _plot(tmp_path, balances, monkeypatch)
    assert set(lines) == {"PLN", "USD", "EUR"}
    assert lines["PLN"] == {"x": [0, 1, 2], "y": [10000, 10200, 10100]}
    assert lines["USD"] == {"x": [0, 1, 2], "y": [0, 50, 75]}
    assert lines["EUR"] == {"x": [0, 1, 2], "y": [0, 0, 20]}


def test_unsorted_rows_are_plotted_in_time_order(tmp_path, monkeypatch, balances):
    shuffled = balances.sample(frac=1, random_state=42).reset_index(drop=True)
    lines = _plot(tmp_path, shuffled, monkeypatch)
    assert lines["PLN"]["x"] == [0, 1, 2]
    assert lines["PLN"]["y"] == [10000, 10200, 10100]
    assert lines["USD"]["x"] == [0, 1, 2]


def test_missing_currency_day_is_not_filled_with_zero(tmp_path, monkeypatch, balances):
    partial = balances.loc[~((balances.currency == "USD") & (balances.time_step == 1))]
    lines = _plot(tmp_path, partial, monkeypatch)
    assert lines["USD"] == {"x": [0, 2], "y": [0, 75]}
    assert lines["PLN"]["x"] == [0, 1, 2]


def test_negative_balances_are_preserved(tmp_path, monkeypatch, balances):
    modified = balances.copy()
    modified.loc[(modified.currency == "PLN") & (modified.time_step == 1), "cash"] = -250
    lines = _plot(tmp_path, modified, monkeypatch)
    assert lines["PLN"]["y"] == [10000, -250, 10100]


def test_empty_cash_history_produces_plot_without_currency_lines(tmp_path, monkeypatch):
    empty = pd.DataFrame(columns=["time_step", "currency", "cash"])
    lines = _plot(tmp_path, empty, monkeypatch)
    assert lines == {}


def test_missing_currency_column_is_rejected(tmp_path, monkeypatch, balances):
    with pytest.raises(ValueError, match="currency"):
        _plot(tmp_path, balances.drop(columns="currency"), monkeypatch)
