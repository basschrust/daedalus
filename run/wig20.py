"""Prepare a WIG20 scenario from Ikaros Snowflake."""

import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv
from snowflake.connector.errors import DatabaseError, OperationalError

from marketsim.analysis.report import MarketRealismReport
from marketsim.input.config import (
    create_run_directory,
    load_config,
)
from marketsim.market.historical_prices import (
    load_historical_prices,
    load_simulation_dates,
)
from marketsim.scenarios.wig20 import connect, prepare
from marketsim.simulator import Simulator
from run.simulation_main import kwargs_for


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

    try:
        connection = connect()
    except (DatabaseError, OperationalError) as exc:
        print(
            "\nERROR: Cannot connect to Snowflake.\n"
            "The server may be unreachable, or the network/DNS "
            "configuration may be incorrect.\n"
            f"Details: {exc}\n"
        )
        raise SystemExit(1) from None
    try:
        path, cutoff, index_level = prepare(connection, settings, root)
    finally:
        connection.close()

    print(f"Prepared {path} with 21 markets; cutoff={cutoff}; WIG20 close={index_level:.2f}")

    if args.prepare_only:
        return

    config = load_config("wig20_auto.yaml")

    output_dir = create_run_directory()

    with (output_dir / "market_structure.yaml").open("w", encoding="utf-8") as stream:
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
    mrc_enabled = settings.get("market_realism", {}).get("enabled", True)
    mrc = MarketRealismReport() if mrc_enabled else None
    sim_kwargs["market_realism_report"] = mrc

    sim = Simulator(**sim_kwargs)

    try:
        sim.run()
        if mrc is not None:
            report = mrc.generate(sim, output_dir)

            print(
                f"Market realism control complete: "
                f"{report['symbol'].nunique()} instruments analyzed."
            )
    finally:
        close = getattr(sim.repository, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
