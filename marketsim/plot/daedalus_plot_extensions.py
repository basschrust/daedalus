"""Standalone plotting utilities for Daedalus.

Call these from existing summary methods; they do not query Snowflake.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


def combine_candles(
    historical: pd.DataFrame,
    simulated: pd.DataFrame,
    simulation_dates: Sequence[str | pd.Timestamp],
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return combined candles and boundary date (first simulated session).

    historical: date column or DatetimeIndex, OHLCV columns.
    simulated: day column (0..N-1) and OHLCV columns.
    simulation_dates: exact exchange trading dates, one per simulated day.
    """
    h = historical.copy()
    if "date" in h.columns:
        h = h.set_index("date")
    h.index = pd.DatetimeIndex(pd.to_datetime(h.index)).normalize()
    if h.index.has_duplicates or not h.index.is_monotonic_increasing:
        raise ValueError("Historical session dates must be unique and increasing")
    if h.empty:
        raise ValueError("Historical candles cannot be empty")
    s = simulated.copy()
    if "day" not in s.columns:
        raise ValueError("Simulated candles must contain day")
    if s["day"].duplicated().any():
        raise ValueError("Duplicate simulated days")
    days = pd.DatetimeIndex(pd.to_datetime(list(simulation_dates))).normalize()
    if len(days) == 0 or days.has_duplicates or not days.is_monotonic_increasing:
        raise ValueError("Simulation dates must be nonempty, unique and increasing")
    if days[0] <= h.index.max():
        raise ValueError("Simulation must begin after historical cutoff")
    expected = set(range(len(days)))
    if set(s["day"].astype(int)) != expected:
        raise ValueError("Simulated days must be 0..N-1 and match calendar")
    s.index = days.take(s["day"].astype(int).to_numpy())
    cols = ["open", "high", "low", "close", "volume"]
    for name, frame in [("historical", h), ("simulated", s)]:
        missing = set(cols) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
    out = pd.concat(
        [h[cols].assign(source="historical"), s[cols].assign(source="simulated")]
    ).sort_index()
    return out, days[0]


def plot_combined_candles(
    historical: pd.DataFrame,
    simulated: pd.DataFrame,
    simulation_dates: Sequence[str | pd.Timestamp],
    output_file: str | Path,
    title: str = "Historical and simulated prices",
) -> None:
    combined, first_simulated = combine_candles(historical, simulated, simulation_dates)
    fig, (ax, vol) = plt.subplots(
        2, 1, sharex=True, figsize=(16, 8), gridspec_kw={"height_ratios": [4, 1]}
    )
    for dt, row in combined.iterrows():
        x = mdates.date2num(dt)
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        ax.vlines(x, row["low"], row["high"], color=color, linewidth=0.8)
        body_height = abs(float(row["close"]) - float(row["open"]))
        ax.add_patch(
            Rectangle(
                (x - 0.35, min(row["open"], row["close"])),
                0.7,
                max(body_height, 0.001),
                facecolor=color,
                edgecolor=color,
                linewidth=0.5,
            )
        )
        vol.bar(x, row["volume"], width=0.7, color=color)
    # The boundary is halfway between last historical and first simulated session.
    cutoff = combined.loc[combined["source"] == "historical"].index.max()
    boundary = cutoff + (first_simulated - cutoff) / 2
    for axis in (ax, vol):
        axis.axvline(boundary, linestyle="--", color="darkorange", linewidth=1.5)
        axis.grid(alpha=0.2)
    ax.set_title(title)
    ax.set_ylabel("Price")
    vol.set_ylabel("Volume")
    vol.set_xlabel("Trading session date")
    vol.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    vol.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def active_cash_currencies(cash_history: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Select currencies whose recorded balance changed at least once."""
    required = {"currency", "time_step", "cash"}
    if not required.issubset(cash_history.columns):
        raise ValueError(
            f"Cash history missing columns: {sorted(required - set(cash_history.columns))}"
        )
    result = {}
    for currency, frame in cash_history.groupby("currency", sort=True):
        frame = frame.sort_values("time_step")
        values = pd.to_numeric(frame["cash"], errors="raise")
        if values.notna().sum() >= 2 and values.dropna().nunique() > 1:
            result[currency] = frame
    return result


def plot_agent_history_multicurrency(
    position_history: pd.DataFrame,
    cash_history: pd.DataFrame,
    value_history: pd.DataFrame,
    output_file: str | Path,
    title: str = "Agent Portfolio history",
    labels_map: dict | None = None,
    time_label: str = "Simulation day",
) -> None:
    labels_map = labels_map or {}
    assets = (
        sorted(position_history["asset_id"].dropna().unique())
        if "asset_id" in position_history
        else []
    )
    currencies = active_cash_currencies(cash_history)
    n = len(assets) + len(currencies) + 1
    fig, axes = plt.subplots(n, 1, figsize=(11, max(4, 2.6 * n)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    i = 0
    for asset_id in assets:
        frame = position_history.loc[position_history["asset_id"] == asset_id].sort_values(
            "time_step"
        )
        axes[i].plot(frame["time_step"], frame["position"])
        axes[i].set_ylabel(str(labels_map.get(asset_id, asset_id)))
        axes[i].grid(alpha=0.3)
        i += 1
    for currency, frame in currencies.items():
        axes[i].plot(frame["time_step"], frame["cash"], linewidth=1.8)
        axes[i].set_ylabel(f"Cash ({currency})")
        axes[i].grid(alpha=0.3)
        i += 1
    if not value_history.empty:
        axes[i].plot(value_history["time_step"], value_history["portfolio_value"])
    axes[i].set_ylabel("Portfolio value")
    axes[i].set_xlabel(time_label)
    axes[i].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
