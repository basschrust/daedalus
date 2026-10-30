"""Plotting utilities for Daedalus.

Historical and simulated candlesticks, plus multicurrency agent histories.
These functions operate on DataFrames and do not query Snowflake.
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
    """Combine historical and simulated daily OHLCV candles.

    Parameters
    ----------
    historical:
        Historical OHLCV with a DatetimeIndex or 'date' column.

    simulated:
        Simulated OHLCV with a 'day' column, numbered 0..N-1.

    simulation_dates:
        Actual trading dates corresponding to simulated days.

    Returns
    -------
    combined:
        DataFrame indexed by calendar date, with OHLCV and source.

    first_simulated:
        Calendar date of the first simulated session.
    """
    cols = ["open", "high", "low", "close", "volume"]

    h = historical.copy()

    if "date" in h.columns:
        h = h.set_index("date")

    h.index = pd.DatetimeIndex(pd.to_datetime(h.index)).normalize()

    if h.empty:
        raise ValueError("Historical candles cannot be empty")

    if h.index.has_duplicates or not h.index.is_monotonic_increasing:
        raise ValueError("Historical session dates must be unique and increasing")

    s = simulated.copy()

    if "day" not in s.columns:
        raise ValueError("Simulated candles must contain a 'day' column")

    if s["day"].duplicated().any():
        raise ValueError("Duplicate simulated days")

    dates = pd.DatetimeIndex(pd.to_datetime(list(simulation_dates))).normalize()

    if len(dates) == 0 or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Simulation dates must be nonempty, unique and increasing")

    if dates[0] <= h.index.max():
        raise ValueError("Simulation must begin after the historical cutoff")

    # Instruments may be active for only part of the simulation.
    # For example, a futures contract can expire before the final day.
    day_numbers = pd.to_numeric(s["day"], errors="raise")

    if s.empty:
        raise ValueError("No simulated candles available")

    if day_numbers.isna().any() or (day_numbers % 1 != 0).any():
        raise ValueError("Simulated days must be integers")

    day_numbers = day_numbers.astype(int)

    if (day_numbers < 0).any() or (day_numbers >= len(dates)).any():
        raise ValueError(
            "Simulated day outside the simulation calendar: "
            f"days={sorted(day_numbers.tolist())}, "
            f"calendar_length={len(dates)}"
        )

    # Map only the days for which the instrument has EoD records.
    s.index = dates.take(day_numbers.to_numpy())

    # expected_days = set(range(len(dates)))
    # actual_days = set(s["day"].astype(int))
    #
    # if actual_days != expected_days:
    #     raise ValueError("Simulated days must be 0..N-1 and match the calendar")
    #
    # s.index = dates.take(s["day"].astype(int).to_numpy())

    for name, frame in (
        ("historical", h),
        ("simulated", s),
    ):
        missing = set(cols) - set(frame.columns)

        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    combined = pd.concat(
        [
            h[cols].assign(source="historical"),
            s[cols].assign(source="simulated"),
        ]
    ).sort_index()

    return combined, dates[0]


def plot_combined_candles(
    historical: pd.DataFrame,
    simulated: pd.DataFrame,
    simulation_dates: Sequence[str | pd.Timestamp],
    output_file: str | Path,
    title: str = "Historical and simulated prices",
    theoretical: pd.Series | None = None,
) -> None:
    """Plot historical and simulated OHLCV on one calendar-date axis.

    Historical sessions have a light-grey background.
    A dashed vertical line marks the transition to simulation.
    """

    combined, first_simulated = combine_candles(
        historical=historical,
        simulated=simulated,
        simulation_dates=simulation_dates,
    )

    fig, (ax_price, ax_volume) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(16, 8),
        gridspec_kw={"height_ratios": [4, 1]},
    )

    historical_end = combined.loc[combined["source"] == "historical"].index.max()

    # Place the boundary between the last historical
    # and the first simulated trading session.
    boundary = historical_end + (first_simulated - historical_end) / 2

    boundary_x = mdates.date2num(boundary)

    # Shade the historical period.
    historical_start = combined.index.min()
    left_edge = mdates.date2num(historical_start) - 1

    for axis in (ax_price, ax_volume):
        axis.axvspan(
            left_edge,
            boundary_x,
            facecolor="#eeeeee",
            alpha=0.65,
            zorder=0,
        )

        axis.axvline(
            boundary_x,
            color="#d97706",
            linestyle="--",
            linewidth=1.6,
            zorder=5,
        )

        axis.grid(alpha=0.2)

    # Plot candles and volume.
    candle_width = 0.7

    for dt, row in combined.iterrows():
        x = mdates.date2num(dt)

        op = float(row["open"])
        hi = float(row["high"])
        lo = float(row["low"])
        cl = float(row["close"])
        volume = float(row["volume"])

        color = "#26a69a" if cl >= op else "#ef5350"

        # High-low wick
        ax_price.vlines(
            x,
            lo,
            hi,
            color=color,
            linewidth=0.8,
            zorder=3,
        )

        # Open-close body
        body_bottom = min(op, cl)
        body_height = max(abs(cl - op), 0.001)

        ax_price.add_patch(
            Rectangle(
                (
                    x - candle_width / 2,
                    body_bottom,
                ),
                candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.5,
                zorder=4,
            )
        )

        # Volume
        ax_volume.bar(
            x,
            volume,
            width=candle_width,
            color=color,
            zorder=3,
        )

    if theoretical is not None and not theoretical.empty:
        line = theoretical.copy()
        line.index = pd.to_datetime(line.index)
        line = pd.to_numeric(line, errors="raise")
        line = line.sort_index()

        ax_price.plot(
            line.index,
            line.to_numpy(dtype=float),
            color="black",
            linewidth=1.5,
            linestyle="-",
            label="Theoretical value",
            zorder=5,
        )

        ax_price.legend(loc="best")

    # Explicitly set limits: patches don't always autoscale.
    ax_price.set_xlim(
        left_edge,
        mdates.date2num(combined.index.max()) + 1,
    )

    price_min = float(combined["low"].min())
    price_max = float(combined["high"].max())

    if theoretical is not None and not theoretical.empty:
        price_min = min(price_min, float(line.min()))
        price_max = max(price_max, float(line.max()))

    price_padding = max(
        (price_max - price_min) * 0.05,
        0.01,
    )

    ax_price.set_ylim(
        price_min - price_padding,
        price_max + price_padding,
    )

    # Phase labels
    ax_price.text(
        boundary_x,
        0.98,
        f" Simulation starts: {first_simulated:%Y-%m-%d}",
        transform=ax_price.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=9,
        color="#92400e",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
            "pad": 3,
        },
        zorder=6,
    )

    ax_price.text(
        0.02,
        0.97,
        "HISTORICAL",
        transform=ax_price.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#777777",
    )

    ax_price.set_title(title)
    ax_price.set_ylabel("Price")

    ax_volume.set_ylabel("Volume")
    ax_volume.set_xlabel("Trading session date")

    # Calendar-date formatting
    locator = mdates.AutoDateLocator(
        minticks=5,
        maxticks=12,
    )

    ax_volume.xaxis.set_major_locator(locator)
    ax_volume.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = Path(output_file)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def active_cash_currencies(
    cash_history: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return currencies whose recorded cash balance changed."""

    required = {"currency", "time_step", "cash"}

    missing = required - set(cash_history.columns)

    if missing:
        raise ValueError(f"Cash history missing columns: {sorted(missing)}")

    result = {}

    for currency, frame in cash_history.groupby(
        "currency",
        sort=True,
    ):
        frame = frame.sort_values("time_step")

        values = pd.to_numeric(
            frame["cash"],
            errors="raise",
        )

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
    """Plot positions, active cash currencies and portfolio value.

    Each currency gets a separate subplot.
    Currencies with constant cash balances are omitted.
    """

    labels_map = labels_map or {}

    assets = (
        sorted(position_history["asset_id"].dropna().unique())
        if "asset_id" in position_history.columns
        else []
    )

    currencies = active_cash_currencies(cash_history)

    n = len(assets) + len(currencies) + 1

    fig, axes = plt.subplots(
        n,
        1,
        figsize=(11, max(4, 2.6 * n)),
        sharex=True,
        squeeze=False,
    )

    axes = axes[:, 0]

    i = 0

    # Asset positions
    for asset_id in assets:
        frame = position_history.loc[position_history["asset_id"] == asset_id].sort_values(
            "time_step"
        )

        axes[i].plot(
            frame["time_step"],
            frame["position"],
        )

        axes[i].set_ylabel(str(labels_map.get(asset_id, asset_id)))
        axes[i].grid(alpha=0.3)

        i += 1

    # Cash: one subplot per active currency
    for currency, frame in currencies.items():
        axes[i].plot(
            frame["time_step"],
            frame["cash"],
            linewidth=1.8,
            color="green",
        )

        axes[i].set_ylabel(f"Cash ({currency})")
        axes[i].grid(alpha=0.3)

        i += 1

    # Portfolio value
    if not value_history.empty:
        axes[i].plot(
            value_history["time_step"],
            value_history["portfolio_value"],
        )

    axes[i].set_ylabel("Portfolio value")
    axes[i].set_xlabel(time_label)
    axes[i].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    output_path = Path(output_file)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)
