"""Prepare and run the full GPW simulation."""

import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv

from marketsim.analysis.report import MarketRealismReport
from marketsim.input.config import (
    create_run_directory,
    load_config,
)
from marketsim.market.historical_prices import (
    load_historical_prices,
    load_simulation_dates,
)
from marketsim.scenarios.gpw import prepare
from marketsim.scenarios.wig20 import connect
from marketsim.simulator import Simulator
from run.simulation_main import kwargs_for


def main():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    settings_path = root / "marketsim/input/gpw.yaml"

    with settings_path.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)

    connection = connect()

    try:
        path, cutoff, market_count = prepare(connection, settings, root)
    finally:
        connection.close()

    print(f"Prepared {market_count} GPW markets; historical cutoff={cutoff}")
    print(f"Configuration: {path}")

    if args.prepare_only:
        return

    config = load_config("gpw_auto.yaml")
    output_dir = create_run_directory()

    with (output_dir / "market_structure.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    config["output_dir"] = output_dir

    historical = load_historical_prices(
        config=config,
        config_dir=root / "marketsim/input",
    )

    simulation_dates = load_simulation_dates(config)

    sim_kwargs = kwargs_for(Simulator, config)
    sim_kwargs["historical_ohlcv_by_symbol"] = historical
    sim_kwargs["simulation_dates"] = simulation_dates

    mrc_enabled = settings.get("market_realism", {}).get("enabled", True)

    mrc = MarketRealismReport() if mrc_enabled else None
    sim_kwargs["market_realism_report"] = mrc

    simulator = Simulator(**sim_kwargs)

    try:
        simulator.run()

        if mrc is not None:
            report = mrc.generate(simulator, output_dir)

            print(f"Market Realism Control: {report['symbol'].nunique()} stocks analyzed")
    finally:
        close = getattr(simulator.repository, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
