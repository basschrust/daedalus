import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from matplotlib.patches import Rectangle


def plot_candlestick_old(df, output_file: str):
    fig, axlist = mpf.plot(
        df,
        type="candle",
        volume=True,
        style="yahoo",
        returnfig=True,
    )

    fig.savefig(output_file, dpi=150, bbox_inches="tight")


def plot_candlestick(
    df: pd.DataFrame,
    output_file: str | None = None,
    title: str = "Candlestick chart",
    x_label: str = "Simulation time [days]",
):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        Index = integer simulation time.
        Columns: open, high, low, close, volume.
    """
    fig, (ax_price, ax_volume) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    candle_width = 0.7

    for t, row in df.iterrows():
        op = float(row["open"])
        hi = float(row["high"])
        lo = float(row["low"])
        cl = float(row["close"])

        color = "#26a69a" if cl >= op else "#ef5350"

        # Wick
        ax_price.plot([t, t], [lo, hi], color=color, linewidth=1)

        # Candle body
        bottom = min(op, cl)
        height = abs(cl - op)

        # Prevent invisible body when open == close
        if height == 0:
            height = 0.001

        body = Rectangle(
            (t - candle_width / 2, bottom),
            candle_width,
            height,
            facecolor=color,
            edgecolor=color,
            linewidth=1,
        )

        ax_price.add_patch(body)

        # Volume
        ax_volume.bar(
            t,
            row.get("volume", 0),  # for indices it is 0
            width=candle_width,
            color=color,
            align="center",
        )

    ax_price.set_title(title)
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.3)

    ax_volume.set_ylabel("Volume")
    ax_volume.set_xlabel(x_label)
    ax_volume.grid(True, alpha=0.3)

    ax_price.set_xlim(df.index.min() - 1, df.index.max() + 1)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)

    plt.close(fig)


# plotting of a derivative - along with the theoretical price:
def plot_candlestick_derivative(
    df: pd.DataFrame, output_file: str | None = None, title: str = "Candlestick chart"
):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        Index = integer simulation time.
        Columns: open, high, low, close, volume, theoretical.
    """
    fig, (ax_price, ax_volume) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    candle_width = 0.7

    for t, row in df.iterrows():
        op = float(row["open"])
        hi = float(row["high"])
        lo = float(row["low"])
        cl = float(row["close"])

        color = "#26a69a" if cl >= op else "#ef5350"

        # Wick
        ax_price.plot([t, t], [lo, hi], color=color, linewidth=1)

        # Candle body
        bottom = min(op, cl)
        height = abs(cl - op)

        # Prevent invisible body when open == close
        if height == 0:
            height = 0.001

        body = Rectangle(
            (t - candle_width / 2, bottom),
            candle_width,
            height,
            facecolor=color,
            edgecolor=color,
            linewidth=1,
        )

        ax_price.add_patch(body)

        # Volume
        ax_volume.bar(
            t,
            row["volume"],
            width=candle_width,
            color=color,
            align="center",
        )

    # Theoretical value
    ax_price.plot(
        df.index,
        df["theoretical"],
        label="theoretical value",
        linewidth=1.5,
        linestyle="-",
        color="black",
    )

    ax_price.legend(loc="best")
    ax_price.set_title(title)
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.3)

    ax_volume.set_ylabel("Volume")
    ax_volume.set_xlabel("Simulation time")
    ax_volume.grid(True, alpha=0.3)

    ax_price.set_xlim(df.index.min() - 1, df.index.max() + 1)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)

    plt.close(fig)
