# Deadalus - based on PyMarketSim

Daedalus is a research-oriented sandbox (based on PyMarketSim) for building and evaluating agent-based limit order book markets. The package provides reusable components for modeling fundamentals, simulating heterogeneous trading agents, and instrumenting the resulting market dynamics so that you can prototype new strategies or reinforcement-learning environments with minimal boilerplate.

## Key capabilities

- **Limit order book microstructure.** Matching is handled by a four-heap order book that respects price-time priority and exposes utilities for monitoring mid-prices and execution statistics.
- **Configurable market structure.** Place your desired market structure in the marketsim/input directory as a yaml file and run your own desired scenario
- **Agent library.** Combine zero-intelligence, market making, informed, and noise agents or author your own policies by extending the base agent interface.
- **Event-driven simulation loop.** A discrete event queue coordinates order arrivals and market clearing, letting you scale to multiple agents and assets while keeping control over the simulation clock.

## Installation

1. Create and activate a Python 3.10 virtual environment.
2. Install the dependencies and package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

This registers the `marketsim` package locally so you can import it from notebooks or scripts.

## Quick start

The snippet below runs a short simulation process with basic market structure (Market maker, noise traders, wash traders, momentum traders and spoofers). 
It demonstrates how to instantiate the core components and iterate the simulator.

```python
git clone https://github.com/basschrust/pymarketsim.git
cd pymarketsim

py -3.10 -m venv .venv
.\.venv\Scripts\activate

python -m pip install --upgrade pip

pip install -e .

python -m run.simulation_main
```
- Simulation results are stored in the directory pymarketsim/marketsim/output/ - each simulation gets its own new directory run_YYYYMMDD_hhmm where you can inspect portfolio history of each agent, candlestick plots of each security (including options and their theoretical value) and flows of assets and cash between your defined agent groups.
- To run one different market structures take a look into following directory: pymarketsim/marketsim/input and use one of the predefined structures or create your own, then run it as below:
```python
python -m run.simulation_main market_structure_WT_3_markets.yaml

or

python -m run.simulation_main pletora_of_agent_types_1.yaml
```


You can replace or augment simulation agents with your own implementations by subclassing `marketsim.agent.agent.Agent` and adding new section in the market structure defined in the yaml file used for your run.

## Working with agents and securities (markets)

- **Agents:** Agent policies live under `marketsim/agent`. They encapsulate order submission logic via a `take_action` method and maintain inventory through helper utilities like `update_position`.
- **Securities:** The `Security` class manages the event queue, order book, and matching process. At each step it ingests orders from agents, clears the book, and updates mid-prices so you can compute downstream metrics.

## Important note on reference price used in Deadalus

Unlike original PyMarketSim framework where fundamental value shared among market participants is used as a reference value, in Deadalus, during continuous trading, the last traded price is used a reference one.
It is planned to make it configurable in the future so each of the approaches can be simulated and tested.

## License

PyMarketSim is distributed under the MIT License. See `LICENSE.txt` for details.
