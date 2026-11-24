# Offline VaR reporting and Kupiec diagnostics

To generate the risk report run the following command from Daedalus project root:

```powershell
python -m run.risk_report
```

The final command selects the most recent existing `output/run_*/daedalus.duckdb` and creates `output/run_*/risk_report/`. It never invokes Ikaros, Snowflake, `prepare_dataset`, or the simulation runner. It reads DuckDB in read-only mode.

Select a particular run and an alternative VaR comparison:

```powershell
python -m run.risk_report --run-dir output/run_20261124_120000
python -m run.risk_report --horizon 5 --confidence 0.95 --agents 10 12
python -m run.risk_report --top-agents 10 --max-ranked 35
```

## Outputs

- `var_evolution_agent_<id>.png`: time series for parametric, historical, Monte Carlo VaR; one agent per figure, fixed selected horizon/confidence; defaults to six highest-median Monte Carlo risk agents.
- `var_agents_ranking.png`: selected highest-risk individual agents by median VaR as percentage of positive equity; all three methods compared.
- `var_groups_comparison.png`: one panel per method, group-wise median and between-agent 10th/90th percentiles, computed from **each agent's own time-series median**. Group labels read from the `agents` table.
- `var_agents_summary.csv` and `var_groups_summary.csv`: source values behind the comparison plots.
- `kupiec_observations_apl.csv`: day-t 1-day forecast, day-(t+1) realized equity change, loss, and exception boolean for each agent/method/confidence. No forecast/outcome pairing is made if an intervening EoD day is missing.
- `kupiec_1d_apl.csv`: exception count/rate, expected count, Kupiec unconditional-coverage likelihood-ratio statistic (LRuc), asymptotic chi-square(1) p-value, and assessment, **per agent/method/confidence**.

Only 1-day VaR is tested. The plots can show 1, 5, or 10-day VaR where present. Positive equity is required to plot VaR as a percentage; nonpositive equity is excluded from normalized comparisons, not set to zero. Kupiec results are still computed for valid nominal one-day VaR observations.

## Statistical limitations

The Kupiec test here uses **actual P&L (APL)**, approximated by the difference in consecutive EoD portfolio equity: `equity[t+1] - equity[t]`. It assumes no external capital contribution/withdrawal. That P&L includes the agent's trades, hedges, settlement and market moves. The current VaR is a frozen-portfolio market-risk forecast, so this is an *exploratory* APL diagnostic, **not** a valid substitute for frozen-position hypothetical P&L (HPL) regulatory backtesting. Avoid interpreting differences as purely model errors.

`LOW_POWER` is set whenever the number of expected exceptions is less than 5 (or the expected non-exception count is less than 5). At 99% confidence, that requires at least 500 usable independent trading-day pairs before the label can be lifted; actual testing power may require more. The LRuc and p-value are still shown for transparency. `NOT_REJECTED_5PCT` does *not* prove a model correct. VaR exceptions from agents trading the same simulated market are dependent; **do not pool agent-day observations as independent evidence**. The test checks unconditional coverage, not exception independence.

For a genuine HPL backtest we should eventually freeze EoD positions, price them with next-day observed market conditions, and account for option/futures cashflows; for risk beyond the currently modelled FX/dividend/volatility assumptions, extend the valuation and risk-factor model first.


## Highest-risk agents: time aggregation and derivative holdings

`var_agents_ranking.png` orders agents by **median daily Monte Carlo VaR /
positive portfolio equity** over the selected simulation period; it does **not**
rank by final-day VaR. Each of the three bars is the corresponding method's
median daily percentage, with days of nonpositive equity excluded.

Each agent label now includes `Last: O#/F#` and `Any: O#/F#` where `O` means
number of distinct **option series** with nonzero positions, and `F` means
number of distinct **futures series** with nonzero positions. `Last` is the
final simulation EoD; `Any` counts the distinct series held at **one or more
EoDs** (not the sum of daily positions). For instance:

- `Last: O0/F0` and `Any: O2/F1`: the agent had option and futures positions
  earlier but finished the simulation flat in derivatives.
- `Last: O1/F0` and `Any: O1/F0`: an option position remains on the final day.

This measures EoD inventory only, not intraday trading, contract quantity,
notional exposure, delta-equivalent exposure, or the presence of orders.
The four counts, plus `last_eod_day`, are also saved in
`var_agents_summary.csv`, repeated for each VaR method. The input DuckDB is
opened read-only and is never modified.
