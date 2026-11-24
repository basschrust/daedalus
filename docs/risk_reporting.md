# Offline risk reporting

Run `python -m run.risk_report` to read the latest local simulation DuckDB and
produce VaR time-series, agent comparisons and preliminary one-day Kupiec APL
diagnostics. It also produces one Delta/Gamma plot per agent who held options
on at least one simulation EoD, using the WIG20 underlying only.

The Greeks loader uses actual Daedalus tables:

- `eod_agent_option_greeks` for signed option exposures (`delta_equivalent`,
  `gamma_equivalent`), which already include contract positions and multipliers.
- `eod_option_greeks` for available EoD dates, and as a fallback for older
  snapshots without `gamma_equivalent`.
- `indices` to identify WIG20 by its registered symbol.

Each plot shows option-only Greeks, *not* the net delta of the entire agent
portfolio including futures or stock hedges. Flat option positions are shown
as zeros on days where per-option EoD Greeks were recorded. The report does
not treat absent Greek observations as zeros if the model never ran.

If the run did not calculate option Greeks (`risk_greeks.enabled: false`), the
VaR report still runs, prints a diagnostic, and does not invent Greek values.
Enable the EoD Greeks during a new simulation to plot real observations.
