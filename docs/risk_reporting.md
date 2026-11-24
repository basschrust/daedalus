# Risk reporting

The offline risk report now produces:

- VaR evolution plots for selected agents.
- Cross-agent and cross-group VaR comparisons.
- Preliminary one-day Kupiec APL diagnostics.
- WIG20 Greeks evolution plots (delta and gamma) for agents that held options at at least one EoD.

The Greeks charts are generated from `eod_agent_greeks` and filtered to the WIG20 underlying. They show net delta and net gamma through time for each option-trading agent.
