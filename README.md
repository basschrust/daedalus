# Daedalus — agent-based financial market simulator

Daedalus is a Python research sandbox for simulating financial markets with heterogeneous trading agents, limit order books, stocks, indices, futures, and options. It builds on PyMarketSim and supports both synthetic scenarios and historical market data from Ikaros.

## Quick start (Windows PowerShell)

You can run a synthetic simulation without a Snowflake account.

```powershell
git clone https://github.com/basschrust/daedalus.git
cd daedalus
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -m run.simulation_main
```

The last command runs `marketsim/input/market_structure.yaml`. Each run creates its own directory under `marketsim/output/run_YYYYMMDD_HHMMSS/` with its configuration and simulation outputs. The available plots and database artifacts depend on the scenario.

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` instead of `python` in the commands above.

## Try another scenario

Choose a YAML configuration from `marketsim/input/`:

```powershell
python -m run.simulation_main WT_3_markets_1000_2000_by_vol.yaml
```

To run your own scenario, create a YAML configuration in `marketsim/input/` and pass its filename as the positional argument. Existing YAML-based commands remain supported.

## Run real-market scenarios with Ikaros

The optional `--dataset` argument prepares a historical dataset from Ikaros in Snowflake, generates the simulation configuration, and runs it through the **same simulator** as the synthetic scenarios.

```powershell
# WIG20 constituents, index, and configured futures
python -m run.simulation_main --dataset wig20

# Full GPW scenario
python -m run.simulation_main --dataset gpw

# Prepare the WIG20 input without running the simulation
python -m run.simulation_main --dataset wig20 --prepare-only
```

These commands require access to the Ikaros Snowflake market-data tables and a configured `.env` file. The connection uses `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, and, for password authentication, `SNOWFLAKE_PASSWORD`. Optional settings include `SNOWFLAKE_WAREHOUSE` (default `DAEDALUS_WH`), `SNOWFLAKE_DATABASE` (default `DAEDALUS`), `SNOWFLAKE_SCHEMA` (default `MARKET_DATA`), `SNOWFLAKE_AUTHENTICATOR`, and `SNOWFLAKE_ROLE`. Never commit `.env` or credentials.

Dataset settings live in `marketsim/input/wig20.yaml` and `marketsim/input/gpw.yaml`. Preparation creates `wig20_auto.yaml` or `gpw_auto.yaml`, respectively, in `marketsim/input/`. You can also run a previously generated configuration directly, without fetching the dataset again:

```powershell
python -m run.simulation_main wig20_auto.yaml
```

## Market realism reports

When enabled in the scenario settings, Daedalus compares simulated stock prices and trading volumes with historical observations. Reports are saved under the run's `market_realism/` directory:

- `market_realism_metrics.csv` — detailed historical versus simulated statistics.
- `market_realism_scorecard.csv` — volatility, volume and turnover ratios, plus differences in correlations.
- `market_realism_diagnostics.csv` — instruments processed and validation errors.

For a custom YAML scenario, enable the report with:

```yaml
market_realism:
  enabled: true
```

The report requires historical OHLCV and sufficient simulated observations. This version generates reports after the simulation; regenerating them from a saved run without simulation is planned separately.

## Developer commands

```powershell
make test
make lint-check
make lint
make format
```

The main application entry point is `python -m run.simulation_main`. Dataset-specific modules in `marketsim/scenarios/` prepare market inputs; the simulator itself is shared.

## Core concepts

- **Limit order books:** order matching and execution statistics.
- **Trading agents:** noise, market making, momentum, and other configurable strategies.
- **Financial instruments:** stocks, indices, futures, and options.
- **Data backends:** DuckDB for local simulations and configurable Snowflake integration.
- **Configuration:** YAML files under `marketsim/input/`, including reusable agent profiles.

You can implement new trading policies by subclassing `marketsim.agent.agent.Agent` and referencing them from your simulation configuration.

## License

See the repository's license file for the applicable terms.
