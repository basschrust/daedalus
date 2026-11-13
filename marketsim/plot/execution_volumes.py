"""Execution volumes by asset, agent group, side and execution mode."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXECUTION_VOLUME_SQL = """
SELECT
    t.asset_id,
    COALESCE(a.group_name, 'UNGROUPED') AS agent_group,
    t.order_side,
    CASE WHEN LOWER(COALESCE(t.executed_mode, '')) IN ('waited', 'arrived')
         THEN LOWER(t.executed_mode) ELSE 'other' END AS executed_mode,
    SUM(t.executed_volume) AS volume
FROM trades t
LEFT JOIN agents a ON a.agent_id = t.agent_id
WHERE t.order_side IN (1, -1)
GROUP BY t.asset_id, COALESCE(a.group_name, 'UNGROUPED'),
         t.order_side, executed_mode
ORDER BY t.asset_id, agent_group, t.order_side, executed_mode
"""

REQUIRED_COLUMNS = {"asset_id", "agent_group", "order_side", "executed_mode", "volume"}


def plot_execution_volumes_by_asset(
    volumes: pd.DataFrame,
    *,
    output_dir: str | Path,
    labels_map: dict[int, str] | None = None,
    include_asset_ids: list[int] | None = None,
) -> list[Path]:
    """Write one stacked paired-bar chart per asset, across all simulation days.

    One bar per BUY/SELL for each agent group. Dark = waited, light = arrived.
    """
    if volumes.empty and not include_asset_ids:
        return []
    missing = REQUIRED_COLUMNS - set(volumes.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = volumes.copy()
    df["agent_group"] = df["agent_group"].fillna("UNGROUPED").astype(str)
    df["executed_mode"] = df["executed_mode"].fillna("other").astype(str).str.lower()
    df["asset_id"] = pd.to_numeric(df["asset_id"], errors="raise").astype(int)
    df["order_side"] = pd.to_numeric(df["order_side"], errors="raise").astype(int)
    df["volume"] = pd.to_numeric(df["volume"], errors="raise")
    invalid_modes = ~df["executed_mode"].isin(("waited", "arrived"))
    if invalid_modes.any():
        raise ValueError(
            "Unexpected execution modes in execution_volumes: "
            f"{df.loc[invalid_modes, 'executed_mode'].unique().tolist()}"
        )
    bad = df[~df["order_side"].isin((1, -1))]
    if not bad.empty:
        raise ValueError("Unexpected order_side or executed_mode values")
    if (df["volume"] < 0).any():
        raise ValueError("Volumes must be nonnegative")
    df = df.groupby(["asset_id", "agent_group", "order_side", "executed_mode"], as_index=False)[
        "volume"
    ].sum()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_map = labels_map or {}
    palette = {
        (1, "waited"): "#1556A6",
        (1, "arrived"): "#9AC8F5",
        (-1, "waited"): "#BA4B0B",
        (-1, "arrived"): "#F4B482",
    }
    result = []
    asset_ids = sorted(set(df["asset_id"].tolist()) | set(include_asset_ids or []))
    for asset_id in asset_ids:
        asset = df[df["asset_id"] == asset_id]
        if asset.empty:
            asset = pd.DataFrame(
                [
                    {
                        "asset_id": asset_id,
                        "agent_group": "No executions",
                        "order_side": 1,
                        "executed_mode": "other",
                        "volume": 0,
                    }
                ]
            )

        groups = sorted(asset["agent_group"].unique())
        pivot = asset.pivot_table(
            index="agent_group",
            columns=["order_side", "executed_mode"],
            values="volume",
            aggfunc="sum",
            fill_value=0,
        ).reindex(groups, fill_value=0)
        x = np.arange(len(groups), dtype=float)
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(8, 1.15 * len(groups) + 3), 5.5))
        for side, offset, side_name in ((1, -width / 2, "Buy"), (-1, width / 2, "Sell")):
            bottom = np.zeros(len(groups))
            for mode in ("waited", "arrived"):
                # TODO: yet are those 2 modes, but what if sth special is introduced? in fixing phase?
                values = (
                    pivot[(side, mode)].to_numpy(dtype=float)
                    if (side, mode) in pivot.columns
                    else np.zeros(len(groups))
                )
                ax.bar(
                    x + offset,
                    values,
                    width,
                    bottom=bottom,
                    color=palette[(side, mode)],
                    label=f"{side_name} — {mode}",
                )
                bottom += values
        ax.set_xticks(x, groups, rotation=25, ha="right")
        if asset["volume"].sum() == 0:
            ax.text(
                0.5,
                0.72,
                "No recorded executions",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_ylim(0, 1)
        ax.set_ylabel("Executed volume (asset units)")
        ax.set_xlabel("Agent group")
        ax.set_title(f"{labels_map.get(int(asset_id), str(asset_id))} — executed volume by group")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(ncol=2)
        fig.tight_layout()
        # calculating nice & safe file name:
        asset_name = str(labels_map.get(int(asset_id), f"asset_{asset_id}"))
        safe_name = re.sub(
            r"[^A-Za-z0-9._-]+",
            "_",
            asset_name,
        ).strip("._")
        path = output_dir / f"market_{asset_id}_{safe_name}_execution_volumes.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        result.append(path)
    return result
