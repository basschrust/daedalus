
"""Group-level asset/cash flows: daily stacked bars per agent group."""

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

FLOW_COLUMNS = [
    "volume_buy",
    "volume_sell",
    "cash_buy",
    "cash_sell",
]


def plot_group_flows(
    flows: pd.DataFrame,
    *,
    output_dir: str | Path,
    labels_map: dict[int, str] | None = None,
) -> list[Path]:
    """Plot daily group flows, stacked by counterparty group.

    One figure per source group, with assets as rows and
    asset/cash flows as columns.

    Positive bars: assets bought or cash received.
    Negative bars: assets sold or cash paid.
    """
    if flows.empty:
        return []

    missing = REQUIRED - set(flows.columns)
    if missing:
        raise ValueError(
            f"Missing group-flow columns: {sorted(missing)}"
        )

    df = flows.copy()

    for column in ("agent_group", "cp_group"):
        df[column] = (
            df[column]
            .fillna("UNGROUPED")
            .astype(str)
        )

    for column in ("day", "asset_id"):
        df[column] = pd.to_numeric(
            df[column], errors="raise"
        ).astype(int)

    for column in FLOW_COLUMNS:
        df[column] = pd.to_numeric(
            df[column], errors="raise"
        )

    # Key optimization:
    # Aggregate all intraday ticks before creating bars.
    daily = (
        df.groupby(
            ["agent_group", "cp_group", "asset_id", "day"],
            as_index=False,
            sort=False,
        )[FLOW_COLUMNS]
        .sum()
    )

    labels_map = labels_map or {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = []

    for source_group, group_df in daily.groupby(
        "agent_group", sort=True
    ):
        days = np.sort(group_df["day"].unique())
        assets = sorted(group_df["asset_id"].unique())
        counterparties = sorted(group_df["cp_group"].unique())

        x = np.arange(len(days))
        palette = plt.get_cmap("tab20")
        colors = {
            cp: palette(i % 20)
            for i, cp in enumerate(counterparties)
        }

        fig, axes = plt.subplots(
            len(assets),
            2,
            sharex=True,
            squeeze=False,
            figsize=(15, max(4, 3.2 * len(assets))),
        )

        for row, asset_id in enumerate(assets):
            asset_df = group_df.loc[
                group_df["asset_id"] == asset_id
            ]

            for col, (positive, negative, ylabel) in enumerate(
                (
                    (
                        "volume_buy",
                        "volume_sell",
                        "Asset units (+bought / -sold)",
                    ),
                    (
                        "cash_sell",
                        "cash_buy",
                        "Cash (+received / -paid)",
                    ),
                )
            ):
                ax = axes[row, col]

                positive_base = np.zeros(len(days))
                negative_base = np.zeros(len(days))

                for cp in counterparties:
                    cp_df = asset_df.loc[
                        asset_df["cp_group"] == cp
                    ]

                    values = (
                        cp_df.groupby("day")[
                            [positive, negative]
                        ]
                        .sum()
                        .reindex(days, fill_value=0)
                    )

                    plus = values[positive].to_numpy(
                        dtype=float
                    )
                    minus = -values[negative].to_numpy(
                        dtype=float
                    )

                    ax.bar(
                        x,
                        plus,
                        bottom=positive_base,
                        width=0.8,
                        color=colors[cp],
                        label=cp,
                    )

                    ax.bar(
                        x,
                        minus,
                        bottom=negative_base,
                        width=0.8,
                        color=colors[cp],
                    )

                    positive_base += plus
                    negative_base += minus

                ax.axhline(
                    0,
                    color="black",
                    linewidth=0.8,
                )
                ax.grid(axis="y", alpha=0.2)
                ax.set_axisbelow(True)
                ax.set_ylabel(ylabel)

                asset_name = labels_map.get(
                    int(asset_id), str(asset_id)
                )

                flow_name = (
                    "Assets" if col == 0 else "Cash"
                )

                ax.set_title(
                    f"{asset_name} — {flow_name}"
                )

        for ax in axes[-1, :]:
            ax.set_xticks(x)
            ax.set_xticklabels(days.astype(str))
            ax.set_xlabel("Simulation day")

        handles, legend_labels = (
            axes[0, 0].get_legend_handles_labels()
        )

        if handles:
            fig.legend(
                handles,
                legend_labels,
                title="Counterparty group",
                loc="upper center",
                bbox_to_anchor=(0.5, 0.97),
                ncol=min(5, len(counterparties)),
            )

        fig.suptitle(
            f"Agent group: {source_group} — daily asset and cash flows",
            y=1.01,
        )

        fig.tight_layout(
            rect=(0, 0, 1, 0.94)
        )

        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in source_group
        )

        path = out / f"{safe_name}_flows.png"

        fig.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)
        paths.append(path)

    return paths
