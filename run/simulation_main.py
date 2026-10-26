from __future__ import annotations

import inspect
import sys
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING

import yaml

from marketsim.input.config import create_run_directory, load_config
from marketsim.input.historical_prices import (
    load_historical_prices,
    load_simulation_dates,
)

from marketsim.simulator import Simulator

if TYPE_CHECKING:
    pass


def kwargs_for(func: Callable, config: dict) -> dict:
    params = inspect.signature(func).parameters
    return {k: v for k, v in config.items() if k in params}


def main():
    config_file = sys.argv[1] if len(sys.argv) > 1 else "market_structure.yaml"

    config = load_config(config_file)
    output_dir = create_run_directory()

    # Save the fully resolved configuration used for this run
    with open(output_dir / "market_structure.yaml", "w") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    config["output_dir"] = output_dir

    # run simulation...
    config_dir = Path("marketsim/input")

    historical_ohlcv_by_symbol = load_historical_prices(
        config=config,
        config_dir=config_dir,
    )

    simulation_dates = load_simulation_dates(config)

    sim_kwargs = kwargs_for(Simulator, config)

    # These are prepared Python objects, not raw YAML values.
    sim_kwargs["historical_ohlcv_by_symbol"] = historical_ohlcv_by_symbol
    sim_kwargs["simulation_dates"] = simulation_dates

    print(
        "Historical OHLCV:",
        {
            symbol: len(frame)
            for symbol, frame in historical_ohlcv_by_symbol.items()
        },
    )
    print("Simulation dates:", simulation_dates)

    sim = Simulator(**sim_kwargs)

    try:
        sim.run()
    finally:
        close = getattr(sim.repository, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
