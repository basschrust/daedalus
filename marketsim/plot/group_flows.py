"""Group-level asset/cash flows: one multi-panel figure per agent group."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED = {
    "day",
    "time_tick",
    "asset_id",
    "agent_group",
    "cp_group",
    "volume_buy",
    "volume_sell",
    "cash_buy",
    "cash_sell",
}


def plot_group_flows(
    flows: pd.DataFrame,
    *,
    output_dir: str | Path,
    labels_map: dict[int, str] | None = None,
) -> list[Path]:
    """Write one figure per source group; rows=assets, columns=assets/cash.

    Data is aggregated over all agents of each source group. Within a panel,
    bars are stacked by counterparty group, separately above and below zero.
    The horizontal axis labels simulation days, not individual time ticks.
    """
    if flows.empty:
        return []
    missing = REQUIRED - set(flows.columns)
    if missing:
        raise ValueError(f"Missing group-flow columns: {sorted(missing)}")

    df = flows.copy()
    df["agent_group"] = df["agent_group"].fillna("UNGROUPED").astype(str)
    df["cp_group"] = df["cp_group"].fillna("UNGROUPED").astype(str)
    for column in ("day", "time_tick", "asset_id"):
        df[column] = pd.to_numeric(df[column], errors="raise").astype(int)
    for column in ("volume_buy", "volume_sell", "cash_buy", "cash_sell"):
        df[column] = pd.to_numeric(df[column], errors="raise")

    # Defensive aggregation: accepts both agent-granularity and group-granularity frames.
    keys = ["agent_group", "cp_group", "asset_id", "day", "time_tick"]
    df = df.groupby(keys, as_index=False, dropna=False)[
        ["volume_buy", "volume_sell", "cash_buy", "cash_sell"]
    ].sum()
    labels_map = labels_map or {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    for source_group, group_df in df.groupby("agent_group", sort=True):
        group_df = group_df.sort_values(["day", "time_tick", "asset_id", "cp_group"])
        moments = group_df[["day", "time_tick"]].drop_duplicates().reset_index(drop=True)
        moments["step"] = np.arange(len(moments))
        group_df = group_df.merge(moments, on=["day", "time_tick"], validate="many_to_one")
        assets = sorted(group_df["asset_id"].unique())
        counterparties = sorted(group_df["cp_group"].unique())
        fig, axes = plt.subplots(
            len(assets),
            2,
            sharex=True,
            squeeze=False,
            figsize=(15, max(4, 3.2 * len(assets))),
        )
        palette = plt.get_cmap("tab20")
        colors = {cp: palette(i % 20) for i, cp in enumerate(counterparties)}

        for row, asset_id in enumerate(assets):
            asset_df = group_df[group_df["asset_id"] == asset_id]
            for col, (positive, negative, ylabel) in enumerate(
                (
                    ("volume_buy", "volume_sell", "Asset units (+bought / -sold)"),
                    ("cash_sell", "cash_buy", "Cash (+received / -paid)"),
                )
            ):
                ax = axes[row, col]
                positive_base = np.zeros(len(moments), dtype=float)
                negative_base = np.zeros(len(moments), dtype=float)
                for cp in counterparties:
                    cp_df = asset_df[asset_df["cp_group"] == cp]
                    values = cp_df.groupby("step")[[positive, negative]].sum()
                    plus = np.zeros(len(moments), dtype=float)
                    minus = np.zeros(len(moments), dtype=float)
                    if not values.empty:
                        positions = values.index.to_numpy(dtype=int)
                        plus[positions] = values[positive].to_numpy(dtype=float)
                        minus[positions] = -values[negative].to_numpy(dtype=float)
                    ax.bar(
                        moments["step"],
                        plus,
                        bottom=positive_base,
                        width=0.85,
                        color=colors[cp],
                        label=cp,
                    )
                    ax.bar(
                        moments["step"], minus, bottom=negative_base, width=0.85, color=colors[cp]
                    )
                    positive_base += plus
                    negative_base += minus
                ax.axhline(0, color="black", linewidth=0.8)
                ax.grid(axis="y", alpha=0.2)
                ax.set_axisbelow(True)
                ax.set_ylabel(ylabel)
                ax.set_title(
                    f"{labels_map.get(int(asset_id), str(asset_id))} — {'Assets' if col == 0 else 'Cash'}"
                )

        # Exactly one x tick per observed day, positioned at the day's first trade.
        day_starts = moments.drop_duplicates("day", keep="first")
        for ax in axes[-1, :]:
            ax.set_xticks(day_starts["step"])
            ax.set_xticklabels(day_starts["day"].astype(str))
            ax.set_xlabel("Simulation day")
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            title="Counterparty group",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=min(5, len(counterparties)),
        )
        fig.suptitle(f"Agent group: {source_group} — asset and cash flows", y=1.01)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        # Group names may contain filesystem-unsafe characters.
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_group)
        path = out / f"{safe_name}_flows.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
