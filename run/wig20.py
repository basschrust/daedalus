"""Prepare a WIG20 scenario from Ikaros Snowflake."""

import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv

from marketsim.scenarios.wig20 import connect, prepare
from run.simulation_main import kwargs_for
from marketsim.input.config import (
    create_run_directory,
    load_config,
)
from marketsim.market.historical_prices import (
    load_historical_prices,
    load_simulation_dates,
)
from marketsim.simulator import Simulator


def main():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-only",
        action="store_true",
    )
    args = parser.parse_args()

    config_path = root / "marketsim/input/wig20.yaml"

    with config_path.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)

    connection = connect()

    try:
        path, cutoff, index_level = prepare(connection, settings, root)
    finally:
        connection.close()

    print(f"Prepared {path} with 21 markets; cutoff={cutoff}; WIG20 close={index_level:.2f}")

    if args.prepare_only:
        return

    config = load_config("wig20_auto.yaml")

    output_dir = create_run_directory()

    with (output_dir / "market_structure.yaml").open(
            "w", encoding="utf-8"
    ) as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    config["output_dir"] = output_dir

    historical = load_historical_prices(
        config=config,
        config_dir=root / "marketsim" / "input",
    )

    simulation_dates = load_simulation_dates(config)

    sim_kwargs = kwargs_for(Simulator, config)
    sim_kwargs["historical_ohlcv_by_symbol"] = historical
    sim_kwargs["simulation_dates"] = simulation_dates

    sim = Simulator(**sim_kwargs)

    try:
        sim.run()
    finally:
        close = getattr(sim.repository, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
