"""Offline VaR plots, Greeks and one-day Kupiec APL diagnostics from a simulation run.

Usage:
    python -m run.risk_report
    python -m run.risk_report --run-dir output/run_20261124_123456
    python -m run.risk_report --confidence 0.95 --horizon 5 --agents 1 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from marketsim.database.risk_reporting_queries import (
    load_derivative_exposure,
    load_risk_report_data,
    load_wig20_greeks,
)
from marketsim.plot.risk_report import (
    add_var_percent,
    agent_ranking_plot,
    agent_summary,
    evolution_plots,
    greeks_evolution_plots,
    group_plot,
    group_summary,
    select_var,
)
from marketsim.risk.var_backtesting import kupiec_report, one_day_apl

ROOT = Path(__file__).resolve().parents[1]


def latest_run_db(*, root: Path = ROOT) -> Path:
    """Use latest output/run_ directory containing a simulation DuckDB."""
    candidates = sorted((root / "output").glob("run_*/daedalus.duckdb"), reverse=True)
    if not candidates:
        raise FileNotFoundError("No output/run_*/daedalus.duckdb; use --run-dir")
    return candidates[0]


def generate_report(
    *,
    db_path: Path,
    output_dir: Path,
    horizon: int = 1,
    confidence: float = 0.99,
    top_agents: int = 6,
    agents: list[int] | None = None,
    max_ranked: int = 25,
) -> list[Path]:
    """Read simulation only; save figures/CSVs under output_dir."""
    if horizon < 1 or not 0.5 < confidence < 1 or top_agents < 1 or max_ranked < 1:
        raise ValueError("Invalid horizon, confidence, top-agents or max-ranked value")
    risk, equity = load_risk_report_data(db_path)
    exposure = load_derivative_exposure(db_path)
    greeks = load_wig20_greeks(db_path)
    risk = add_var_percent(risk)
    chosen = select_var(risk, horizon=horizon, confidence=confidence)
    summary = agent_summary(chosen)
    groups = group_summary(summary)
    summary = summary.merge(exposure, on="agent_id", how="left", validate="many_to_one")
    if summary["last_eod_day"].isna().any():
        unmatched = sorted(summary.loc[summary["last_eod_day"].isna(), "agent_id"].unique())
        raise ValueError(f"Derivative inventory unavailable for agent IDs: {unmatched}")
    derivative_columns = (
        "options_last_eod",
        "futures_last_eod",
        "options_any_eod",
        "futures_any_eod",
    )
    for column in derivative_columns:
        summary[column] = summary[column].fillna(0).astype(int)

    if agents is None:
        montecarlo = summary.loc[summary.method == "montecarlo"].sort_values(
            ["median_var_pct", "agent_id"], ascending=[False, True]
        )
        agents = montecarlo.agent_id.head(top_agents).astype(int).tolist()
        if not agents:
            agents = summary.agent_id.drop_duplicates().head(top_agents).astype(int).tolist()
    else:
        if unknown := set(agents) - set(chosen.agent_id):
            raise ValueError(f"Unknown agent IDs: {sorted(unknown)}")

    observations = one_day_apl(risk, equity)
    kupiec = kupiec_report(observations)

    option_agents = sorted(
        exposure.loc[exposure["options_any_eod"] > 0, "agent_id"].astype(int).tolist()
    )
    greeks = greeks.loc[greeks["agent_id"].isin(option_agents)].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "var_agents_summary.csv", index=False)
    groups.to_csv(output_dir / "var_groups_summary.csv", index=False)
    observations.to_csv(output_dir / "kupiec_observations_apl.csv", index=False)
    kupiec.to_csv(output_dir / "kupiec_1d_apl.csv", index=False)
    greeks.to_csv(output_dir / "greeks_wig20.csv", index=False)

    images = evolution_plots(
        chosen,
        out=output_dir,
        agent_ids=agents,
        horizon=horizon,
        confidence=confidence,
    )
    images.append(group_plot(groups, out=output_dir, horizon=horizon, confidence=confidence))
    images.append(
        agent_ranking_plot(
            summary,
            out=output_dir,
            horizon=horizon,
            confidence=confidence,
            max_agents=max_ranked,
        )
    )
    images.extend(greeks_evolution_plots(greeks, out=output_dir, agent_ids=option_agents))
    print(f"Read-only source: {db_path}")
    print(f"Report files: {output_dir} ({len(images)} PNGs + 5 CSVs)")
    print(f"VaR evolution agent IDs: {agents}")
    print("Agent ranking: median daily VaR/equity across the simulation, not the last day")
    print(
        "Derivatives: nonzero positions at final EoD "
        f"day {int(exposure['last_eod_day'].iloc[0])} vs any EoD (O=option, F=future)"
    )
    if option_agents:
        print(f"Greeks: WIG20 delta/gamma plots for option traders {option_agents}")
    else:
        print("Greeks: no agents held options at any EoD; no Greeks plots generated")
    if kupiec.empty:
        print("Kupiec: no eligible day-t -> day-(t+1) observations")
    else:
        low = int((kupiec.assessment == "LOW_POWER").sum())
        print(f"Kupiec one-day APL tests: {len(kupiec)}; LOW_POWER: {low}")
    print("CAUTION: Backtest uses realized equity P&L (APL), not frozen-book HPL.")
    print("A short run has very low power; no model validation is implied.")
    return images


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run-dir", type=Path, help="Existing output/run_... directory")
    source.add_argument("--db", type=Path, help="Explicit local daedalus.duckdb")
    parser.add_argument("--output-dir", type=Path, help="Default <run-dir>/risk_report")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--top-agents", type=int, default=6)
    parser.add_argument("--agents", type=int, nargs="+", help="Agents for time-series plots")
    parser.add_argument("--max-ranked", type=int, default=25)
    opts = parser.parse_args(argv)
    try:
        db = opts.db or (opts.run_dir / "daedalus.duckdb" if opts.run_dir else latest_run_db())
        generate_report(
            db_path=db,
            output_dir=opts.output_dir or db.parent / "risk_report",
            horizon=opts.horizon,
            confidence=opts.confidence,
            top_agents=opts.top_agents,
            agents=opts.agents,
            max_ranked=opts.max_ranked,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"Risk report failed: {exc}\n")
    except duckdb.IOException as exc:
        detail = str(exc)
        detail_lower = detail.lower()
        lock_signatures = (
            "being used by another process",
            "already open in",
            "used by another process",
            "file is locked",
            "database is locked",
        )
        if any(signature in detail_lower for signature in lock_signatures):
            parser.exit(
                2,
                f"Risk report failed: simulation database is locked: {db.resolve()}\n"
                "Close or disconnect that DuckDB connection in DBeaver (or the "
                "other process holding it), then rerun the report.\n"
                "No simulation rerun is necessary.\n",
            )
        parser.exit(2, f"Risk report failed: DuckDB I/O error reading {db}: {detail}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
