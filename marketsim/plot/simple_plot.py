import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def simple_plot_old(x: list, y: list, output_file: str) -> None:
    plt.plot(x, y)
    plt.savefig(output_file)

def simple_plot(
    x: list,
    y: list,
    output_file: str,
    ax=None,
    label: str = None,
    ylabel: str = "Portfolio value",
) -> None:

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    ax.plot(x, y, label=label)

    ax.set_xlabel("Simulation time")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    if label is not None:
        ax.legend()

    fig.tight_layout()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

def plot_agent_history_single_market(
    position_history: pd.DataFrame,
    value_history: dict,
    output_file: str,
) -> None:

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    # Position subplot
    ax1.plot(
        position_history.time_tick,
        position_history.position,
    )
    ax1.set_ylabel("Position")
    ax1.grid(True)

    # Portfolio value subplot
    ax2.plot(
        list(value_history.keys()),
        list(value_history.values()),
    )
    ax2.set_xlabel("Simulation time")
    ax2.set_ylabel("Portfolio value")
    ax2.grid(True)

    fig.tight_layout()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_agent_history_many_markets(
    position_history: pd.DataFrame,
    cash_history: pd.DataFrame, # dict,
    value_history: pd.DataFrame, #dict,
    output_file: str,
    title: str = "Agent Portfolio history",
    labels_map: dict|None=None, # map of short names: {asset_id: "stock TNT", 4: "CALL TNT 90", ...}
    time_label: str|None=None,
) -> None:

    time_label = time_label if time_label is None else "Simulation time"

    asset_ids = sorted(position_history["asset_id"].unique())
    n_assets = len(asset_ids)
    if labels_map is None:
        labels_map = { asset_id: f"Asset {asset_id}" for asset_id in asset_ids }
    print(f"Labels map: {labels_map}")

    fig, axes = plt.subplots(
        n_assets + 2,
        ncols=1,
        figsize=(10, 3 * (n_assets + 2)),
        sharex=True,
    )

    # Make sure axes is always iterable
    if n_assets + 2 == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14)

    # Position subplots
    for i, asset_id in enumerate(asset_ids):
        ax = axes[i]

        asset_history = position_history[
            position_history["asset_id"] == asset_id
        ].sort_values("time_step")

        ax.plot(
            asset_history["time_step"],
            asset_history["position"],
        )

        ax.set_ylabel(labels_map.get(asset_id))
        ax.grid(True)

    # Cash subplot — one line per currency
    ax_cash = axes[n_assets]

    if "currency" not in cash_history.columns:
        raise ValueError("Cash history must contain a 'currency' column")

    for currency, currency_df in cash_history.groupby("currency", sort=True):
        currency_df = currency_df.sort_values("time_step")

        ax_cash.plot(
            currency_df["time_step"],
            currency_df["cash"],
            label=currency,
            linewidth=1.8,
            marker=".",
        )

    ax_cash.set_ylabel("Cash")
    # case if no currencies provided:
    handles, labels = ax_cash.get_legend_handles_labels()
    if handles:
        ax_cash.legend(title="Currency")

    ax_cash.grid(True)

    # Total portfolio value subplot
    ax_value = axes[n_assets + 1]

    ax_value.plot(
        # list(value_history.keys()),
        # list(value_history.values()),
        value_history["time_step"],
        value_history["portfolio_value"]
    )

    ax_value.set_xlabel(time_label)
    ax_value.set_ylabel("Portfolio value")
    ax_value.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.985])

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_order_book(
    bids: dict,
    asks: dict,
    output_file: str,
    cumulative: bool = True,
    title: str = "Order Book",
) -> None:
    """
    bids and asks should be dictionaries:
        {price: volume}

    If cumulative=True, the values are first converted to cumulative depth.
    """

    def make_depth(book: dict, reverse: bool):
        items = sorted(book.items(), reverse=reverse)

        if cumulative:
            total = 0
            depth = []
            for price, volume in items: # TODO: these are yet price, order_id (!!!) - solved (?)
                total += volume
                depth.append((float(price), total))
            return depth

        return [(float(price), volume) for price, volume in items]

    bid_depth = make_depth(bids, reverse=True)
    ask_depth = make_depth(asks, reverse=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    if bid_depth:
        ax.bar(
            [p for p, _ in bid_depth],
            [v for _, v in bid_depth],
            width=0.01,
            color="royalblue",
            label="Bids",
        )

    if ask_depth:
        ax.bar(
            [p for p, _ in ask_depth],
            [v for _, v in ask_depth],
            width=0.01,
            color="darkred",
            label="Asks",
        )

    ax.set_xlabel("Price limit")
    ax.set_ylabel("Cumulative volume" if cumulative else "Volume")
    ax.set_title(title)
    ax.grid(axis="y")
    ax.legend()

    fig.tight_layout()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_by_type(
    orders_by_type: dict,
    output_file: str,
    title: str,
    mode: str = "simple",  # or extended
) -> None:
    """
    Plot order statistics by agent/type.

    Simple input:
        {
            "MM": {
                "count_buy": 6,
                "volume_buy": 67,
                "count_sell": 7,
                "volume_sell": 89,
            },
            ...
        }

    Extended input:
        {
            "group1": {
                "count_buy": {
                    "arrived": 60,
                    "waited": 40,
                },
                "volume_buy": {
                    "arrived": 20,
                    "waited": 0,
                },
                "count_sell": {
                    "arrived": 10,
                    "waited": 20,
                },
                "volume_sell": {
                    "arrived": 20,
                    "waited": 20,
                },
            },
            ...
        }
    """

    if mode not in ("simple", "extended"):
        raise ValueError(
            f"Unknown mode '{mode}'. Expected 'simple' or 'extended'."
        )

    order_types = list(orders_by_type.keys())

    fig, (ax_count, ax_volume) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    x = list(range(len(order_types)))
    width = 0.35

    if mode == "simple":

        buy_counts = [
            orders_by_type[order_type]["count_buy"]
            for order_type in order_types
        ]
        sell_counts = [
            orders_by_type[order_type]["count_sell"]
            for order_type in order_types
        ]

        buy_volumes = [
            orders_by_type[order_type]["volume_buy"]
            for order_type in order_types
        ]
        sell_volumes = [
            orders_by_type[order_type]["volume_sell"]
            for order_type in order_types
        ]

        # Order count
        ax_count.bar(
            [i - width / 2 for i in x],
            buy_counts,
            width=width,
            label="Buy",
            color="blue",
        )

        ax_count.bar(
            [i + width / 2 for i in x],
            sell_counts,
            width=width,
            label="Sell",
            color="darkred",
        )

        # Order volume
        ax_volume.bar(
            [i - width / 2 for i in x],
            buy_volumes,
            width=width,
            label="Buy",
            color="blue",
        )

        ax_volume.bar(
            [i + width / 2 for i in x],
            sell_volumes,
            width=width,
            label="Sell",
            color="darkred",
        )

    else:
        # Extract extended data
        buy_count_arrived = [
            orders_by_type[order_type]["count_buy"]["arrived"]
            for order_type in order_types
        ]
        buy_count_waited = [
            orders_by_type[order_type]["count_buy"]["waited"]
            for order_type in order_types
        ]

        sell_count_arrived = [
            orders_by_type[order_type]["count_sell"]["arrived"]
            for order_type in order_types
        ]
        sell_count_waited = [
            orders_by_type[order_type]["count_sell"]["waited"]
            for order_type in order_types
        ]

        buy_volume_arrived = [
            orders_by_type[order_type]["volume_buy"]["arrived"]
            for order_type in order_types
        ]
        buy_volume_waited = [
            orders_by_type[order_type]["volume_buy"]["waited"]
            for order_type in order_types
        ]

        sell_volume_arrived = [
            orders_by_type[order_type]["volume_sell"]["arrived"]
            for order_type in order_types
        ]
        sell_volume_waited = [
            orders_by_type[order_type]["volume_sell"]["waited"]
            for order_type in order_types
        ]

        # Order count - Buy
        ax_count.bar(
            [i - width / 2 for i in x],
            buy_count_arrived,
            width=width,
            label="Buy arrived",
            color="blue",
        )

        ax_count.bar(
            [i - width / 2 for i in x],
            buy_count_waited,
            width=width,
            bottom=buy_count_arrived,
            label="Buy waited",
            color="lightblue",
        )

        # Order count - Sell
        ax_count.bar(
            [i + width / 2 for i in x],
            sell_count_arrived,
            width=width,
            label="Sell arrived",
            color="darkred",
        )

        ax_count.bar(
            [i + width / 2 for i in x],
            sell_count_waited,
            width=width,
            bottom=sell_count_arrived,
            label="Sell waited",
            color="lightcoral",
        )

        # Order volume - Buy
        ax_volume.bar(
            [i - width / 2 for i in x],
            buy_volume_arrived,
            width=width,
            label="Buy arrived",
            color="blue",
        )

        ax_volume.bar(
            [i - width / 2 for i in x],
            buy_volume_waited,
            width=width,
            bottom=buy_volume_arrived,
            label="Buy waited",
            color="lightblue",
        )

        # Order volume - Sell
        ax_volume.bar(
            [i + width / 2 for i in x],
            sell_volume_arrived,
            width=width,
            label="Sell arrived",
            color="darkred",
        )

        ax_volume.bar(
            [i + width / 2 for i in x],
            sell_volume_waited,
            width=width,
            bottom=sell_volume_arrived,
            label="Sell waited",
            color="lightcoral",
        )

    # Common formatting
    ax_count.set_ylabel("Count")
    ax_count.set_title(title)
    ax_count.grid(axis="y")
    ax_count.legend()

    ax_volume.set_ylabel("Volume")
    ax_volume.set_xlabel("Type")
    ax_volume.grid(axis="y")
    ax_volume.legend()

    ax_volume.set_xticks(x)
    ax_volume.set_xticklabels(order_types)

    fig.tight_layout()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_bid_ask(
    bid_ask: dict[int, list[float]],
    output_file: str,
    title: str = "Bid ask",
    clip_value: float = 3.0,
) -> None:
    """
    Plot best bid and best ask over simulation time.

    Args:
        bid_ask: {
            time_tick: [best_bid, best_ask],
            ...
        }
        output_file: Output PNG filename.
        clip_value: Value used to clip +/-inf values.
    """
    times = sorted(bid_ask)
    spreads = []

    for time in times:
        bid, ask = bid_ask[time]

        spread = float(ask) - float(bid)

        if not math.isfinite(spread):
            spread = clip_value

        spreads.append(spread)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(times, spreads, label="Spread")

    ax.set_xlabel("Simulation time")
    ax.set_ylabel("Spread")
    ax.grid(True)
    ax.legend()
    ax.set_title(title)

    fig.tight_layout()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_realized_volatility(
    volatility: dict,
    output_file: str,
    title: str = "Realized volatility",
) -> None:
    if not volatility:
        return

    times = sorted(volatility)
    values = [volatility[t] for t in times]

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(times, values, label="Realized volatility")

    ax.set_xlabel("Simulation time")
    ax.set_ylabel("Realized volatility")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_agent_profitability_vs_volatility(
    agents: list,
    output_file: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    groups = defaultdict(list)

    for agent in agents:
        history = agent.portfolio_value_history

        if len(history) < 2:
            continue

        # Make sure observations are ordered by simulation time
        values = np.array(
            [float(v) for _, v in sorted(history.items())],
            dtype=float,
        )

        # Portfolio returns between consecutive simulation ticks
        returns = values[1:] / values[:-1] - 1

        volatility = np.std(returns)

        # Total profitability over the simulation
        profitability = values[-1] / values[0] - 1

        groups[agent.group].append(
            (volatility, profitability, agent.agent_id)
        )

    for group, points in groups.items():
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        ax.scatter(x, y, label=str(group), alpha=0.8)

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)

    ax.set_xlabel("Portfolio volatility")
    ax.set_ylabel("Profitability")
    ax.set_title("Agent Profitability vs Portfolio Volatility")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Group")

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


class Plotter:
    def line(self, x: list, y: list, output_file: str) -> None:
        plt.plot(x, y)
        plt.savefig(output_file)

    def candles(self, df: pd.DataFrame, output_file: str) -> None:
        pass

    def scatter(self, df: pd.DataFrame, output_file: str) -> None:
        pass

    def hist(self, df: pd.DataFrame, output_file: str) -> None:
        pass

# the magic subplots for capital movement between groups:
def plot_volume_transfers(df: pd.DataFrame, output_file_tpl: str):
    """
    Plot buy/sell volume history for each agent group.

    For each unique agentGroup, create one figure.
    The figure contains one subplot for each cpGroup
    associated with that agentGroup.

    X-axis: time_tick
    Y-axis: volume
    Lines: Volume_buy, Volume_sell
    """

    for agent_group in df["agent_group"].unique():

        agent_df = df[df["agent_group"] == agent_group]

        cp_groups = agent_df["cp_group"].unique()
        n_subplots = len(cp_groups)

        fig, axes = plt.subplots(
            n_subplots,
            ncols=1,
            figsize=(12, 4 * n_subplots),
            sharex=True,
        )

        # When there is only one subplot, matplotlib doesn't return a list
        if n_subplots == 1:
            axes = [axes]

        fig.suptitle(
            f"Volume history - Agent Group: {agent_group}",
            fontsize=14,
        )

        for ax, cp_group in zip(axes, cp_groups):
            cp_df = agent_df[agent_df["cp_group"] == cp_group]

            buy_total = cp_df["volume_buy"].sum()
            sell_total = cp_df["volume_sell"].sum()

            ax.plot(
                cp_df["time_tick"],
                cp_df["volume_buy"],
                label=f"volume_buy, total: {buy_total:g}",
            )
            ax.plot(
                cp_df["time_tick"],
                -cp_df["volume_sell"],
                label=f"volume_sell, total: {sell_total:g}",
            )
            # Display absolute values on Y-axis
            ax.yaxis.set_major_formatter(
                lambda x, pos: f"{abs(x):g}"
            )

            # Put X-axis through y=0
            ax.axhline(0, linewidth=1)

            ax.set_title(f"CP Group: {cp_group}")
            ax.set_ylabel("Volume")
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Time tick")

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        output_file = output_file_tpl + agent_group + ".png"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=150)
        plt.close(fig)

def plot_cash_transfers(df: pd.DataFrame, output_file_tpl: str):
    """
    Plot buy/sell cash transfer history for each agent group.

    For each unique agentGroup, create one figure.
    The figure contains one subplot for each cpGroup
    associated with that agentGroup.

    X-axis: time_tick
    Y-axis: cash
    Lines: cash_buy, cash_sell
    """

    for agent_group in df["agent_group"].unique():

        agent_df = df[df["agent_group"] == agent_group]
        cp_groups = agent_df["cp_group"].unique()
        n_subplots = len(cp_groups)

        fig, axes = plt.subplots(
            n_subplots,
            ncols=1,
            figsize=(12, 4 * n_subplots),
            sharex=True,
        )

        # When there is only one subplot, matplotlib doesn't return a list
        if n_subplots == 1:
            axes = [axes]

        fig.suptitle(
            f"Volume history - Agent Group: {agent_group}",
            fontsize=14,
        )

        for ax, cp_group in zip(axes, cp_groups):
            cp_df = agent_df[agent_df["cp_group"] == cp_group]

            buy_total = cp_df["cash_buy"].sum()
            sell_total = cp_df["cash_sell"].sum()

            ax.plot(
                cp_df["time_tick"],
                cp_df["cash_buy"],
                label=f"cash_buy, total: {buy_total:g}",
            )
            ax.plot(
                cp_df["time_tick"],
                -cp_df["cash_sell"],
                label=f"cash_sell, total: {sell_total:g}",
            )
            # Display absolute values on Y-axis
            ax.yaxis.set_major_formatter(
                lambda x, pos: f"{abs(x):g}"
            )

            # Put X-axis through y=0
            ax.axhline(0, linewidth=1)

            ax.set_title(f"CP Group: {cp_group}")
            ax.set_ylabel("Cash transferred")
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Time tick")

        plt.tight_layout(rect=[0, 0, 1, 0.98])

        output_file = output_file_tpl + agent_group + ".png"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=150)
        plt.close(fig)
