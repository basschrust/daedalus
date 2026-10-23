from __future__ import annotations

import inspect
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

import yaml

from marketsim.input.config import create_run_directory, load_config

# from marketsim.input.config import CONFIG
# from marketsim.loggers.basic import StreamToLogger
# from marketsim.plot.simple_plot import simple_plot
from marketsim.simulator import Simulator

if TYPE_CHECKING:
    pass


def kwargs_for(func: Callable, config: dict) -> dict:
    params = inspect.signature(func).parameters
    return {k: v for k, v in config.items() if k in params}


def main():
    config_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "market_structure.yaml"
    )

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
    sim = Simulator(**kwargs_for(Simulator, config))

    try:
        sim.run()
    finally:
        close = getattr(sim.repository, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()





