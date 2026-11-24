"""Noninteractive VaR time-series, Greeks and cross-agent charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")

METHODS = ("parametric", "historical", "montecarlo")
LABELS = ("Parametric", "Historical", "Monte Carlo")


def add_var_percent(risk: pd.DataFrame) -> pd.DataFrame:
    """Normalize by positive equity; do not treat negative equity as risk-free."""
    data = risk.copy()
    for column in ("var_amount", "portfolio_value"):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite {column}")
    if (data["var_amount"] < 0).any():
        raise ValueError("Negative VaR amount")
    value = data["portfolio_value"].to_numpy(dtype=float)
    data["var_pct_equity"] = np.divide(
        100 * data["var_amount"].to_numpy(dtype=float),
        value,
        out=np.full(len(data), np.nan),
        where=value > 0,
    )
    return data


def select_var(risk: pd.DataFrame, *, horizon: int, confidence: float) -> pd.DataFrame:
    selected = risk.loc[
        (risk["horizon_days"] == horizon)
        & np.isclose(risk["confidence_level"].to_numpy(dtype=float), confidence, rtol=0, atol=1e-9)
    ].copy()
    if selected.empty:
        available = sorted(
            {(int(h), float(c)) for h, c in zip(risk.horizon_days, risk.confidence_level)}
        )
        raise ValueError(f"No {horizon}d {confidence:.1%} VaR available; combinations: {available}")
    return selected


def agent_summary(selected: pd.DataFrame) -> pd.DataFrame:
    """First aggregate days within each agent, preserving agent as unit of analysis."""
    usable = selected.dropna(subset=["var_pct_equity"])
    if usable.empty:
        raise ValueError("No positive agent equity for VaR normalization")
    return (
        usable.groupby(["method", "agent_id", "group_name"], as_index=False)
        .agg(days=("day", "nunique"), median_var_pct=("var_pct_equity", "median"))
        .sort_values(["method", "group_name", "agent_id"])
    )


def group_summary(agents: pd.DataFrame) -> pd.DataFrame:
    """Inter-agent 10th/median/90th quantiles of each agent's median VaR%."""
    return (
        agents.groupby(["method", "group_name"], as_index=False)
        .agg(
            agents=("agent_id", "nunique"),
            median_var_pct=("median_var_pct", "median"),
            p10_var_pct=("median_var_pct", lambda s: s.quantile(0.10)),
            p90_var_pct=("median_var_pct", lambda s: s.quantile(0.90)),
        )
        .sort_values(["method", "group_name"])
    )


def evolution_plots(
    selected: pd.DataFrame,
    *,
    out: Path,
    agent_ids: list[int],
    horizon: int,
    confidence: float,
) -> list[Path]:
    outputs: list[Path] = []
    for agent_id in agent_ids:
        rows = selected.loc[selected["agent_id"] == agent_id]
        if rows.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for method, label in zip(METHODS, LABELS):
            series = rows.loc[rows["method"] == method].sort_values("day")
            if not series.empty:
                ax.plot(
                    series["day"],
                    series["var_pct_equity"],
                    marker="o",
                    markersize=3,
                    linewidth=1.7,
                    label=label,
                )
        ax.set_title(
            f"Agent {agent_id} ({rows.iloc[0]['group_name']}) | "
            f"{horizon}d VaR, {confidence:.1%} confidence"
        )
        ax.set_xlabel("Simulation day (EoD)")
        ax.set_ylabel("VaR / positive equity (%)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        target = out / f"var_evolution_agent_{agent_id}.png"
        fig.savefig(target, dpi=140)
        plt.close(fig)
        outputs.append(target)
    return outputs


def group_plot(
    groups: pd.DataFrame,
    *,
    out: Path,
    horizon: int,
    confidence: float,
) -> Path:
    names = sorted(groups.group_name.unique())
    fig, axes = plt.subplots(3, 1, figsize=(max(10, 0.8 * len(names) + 4), 11), sharex=True)
    x = np.arange(len(names))
    for ax, method, label in zip(axes, METHODS, LABELS):
        frame = groups.loc[groups.method == method].set_index("group_name")
        chosen = [name for name in names if name in frame.index]
        if chosen:
            pos = np.asarray([names.index(name) for name in chosen])
            median = frame.loc[chosen, "median_var_pct"].to_numpy(dtype=float)
            low = frame.loc[chosen, "p10_var_pct"].to_numpy(dtype=float)
            high = frame.loc[chosen, "p90_var_pct"].to_numpy(dtype=float)
            err = np.vstack((np.maximum(0, median - low), np.maximum(0, high - median)))
            ax.errorbar(pos, median, yerr=err, fmt="o", capsize=3)
            for i, name in enumerate(chosen):
                ax.annotate(
                    f"n={int(frame.loc[name, 'agents'])}",
                    (pos[i], median[i]),
                    xytext=(3, 6),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.set_ylabel(f"{label}\nVaR / equity (%)")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x, names, rotation=45, ha="right")
    fig.suptitle(
        f"Agent groups | {horizon}d {confidence:.1%} VaR\n"
        "Median across agents; whiskers: 10th to 90th percentiles of agent medians"
    )
    fig.tight_layout()
    target = out / "var_groups_comparison.png"
    fig.savefig(target, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return target


def agent_ranking_plot(
    summary: pd.DataFrame,
    *,
    out: Path,
    horizon: int,
    confidence: float,
    max_agents: int,
) -> Path:
    """Rank by median daily VaR%, with final/any EoD derivative holdings."""
    pivot = summary.pivot(index="agent_id", columns="method", values="median_var_pct")
    metadata = summary.drop_duplicates("agent_id").set_index("agent_id")
    preferred = "montecarlo" if "montecarlo" in pivot else pivot.columns[0]
    pivot = pivot.sort_values(preferred, ascending=False).head(max_agents)
    fig, ax = plt.subplots(figsize=(max(11, 0.70 * len(pivot) + 3), 7.6))
    x = np.arange(len(pivot))
    for offset, method, label in zip((-0.26, 0, 0.26), METHODS, LABELS):
        if method in pivot:
            ax.bar(x + offset, pivot[method].to_numpy(dtype=float), width=0.25, label=label)

    labels = []
    for agent_id in pivot.index:
        item = metadata.loc[agent_id]
        labels.append(
            f"{int(agent_id)}\n{item['group_name']}\n"
            f"Last: O{int(item['options_last_eod'])}/F{int(item['futures_last_eod'])}\n"
            f"Any: O{int(item['options_any_eod'])}/F{int(item['futures_any_eod'])}"
        )
    ax.set_xticks(x, labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Median daily VaR / positive equity (%)")
    ax.set_title(
        f"Highest-risk agents | {horizon}d {confidence:.1%} VaR\n"
        "Ranked by median across simulation days (not final-day VaR)"
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.text(
        0.5,
        0.015,
        "O/F = distinct option/futures series with nonzero EoD positions; "
        "Last = final simulation EoD, Any = at least one EoD",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    target = out / "var_agents_ranking.png"
    fig.savefig(target, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return target


def greeks_evolution_plots(
    greeks: pd.DataFrame, *, out: Path, agent_ids: list[int], underlying: str = "WIG20"
) -> list[Path]:
    """Per-option-trader time series of net delta and gamma against WIG20."""
    outputs: list[Path] = []
    for agent_id in agent_ids:
        rows = greeks.loc[greeks["agent_id"] == agent_id].sort_values("day")
        if rows.empty:
            continue
        group = rows.iloc[0].get("group_name", "Unclassified")
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(rows["day"], rows["delta"], marker="o", markersize=3, linewidth=1.8)
        axes[0].axhline(0, linewidth=1.0, alpha=0.4)
        axes[0].set_ylabel("Option Delta (PLN / WIG20 point)")
        axes[0].grid(alpha=0.25)

        axes[1].plot(rows["day"], rows["gamma"], marker="o", markersize=3, linewidth=1.8)
        axes[1].axhline(0, linewidth=1.0, alpha=0.4)
        axes[1].set_ylabel("Option Gamma (PLN / point²)")
        axes[1].set_xlabel("Simulation day (EoD)")
        axes[1].grid(alpha=0.25)

        fig.suptitle(
            f"Agent {agent_id} ({group}) | {underlying} options Delta & Gamma through time\n"
            "Option-only exposures; futures hedges are excluded"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        target = out / f"greeks_wig20_agent_{agent_id}.png"
        fig.savefig(target, dpi=140, bbox_inches="tight")
        plt.close(fig)
        outputs.append(target)
    return outputs
