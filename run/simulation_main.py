"""Single entry point for Daedalus simulations and prepared datasets."""

from __future__ import annotations

import argparse
import inspect
from collections.abc import Callable
from pathlib import Path

import yaml
from dotenv import load_dotenv

from input.config import create_run_directory, load_config
from marketsim.analysis.report import MarketRealismReport
from marketsim.database.ikaros_queries import connect
from marketsim.market.historical_prices import load_historical_prices, load_simulation_dates
from marketsim.risk.calibration_sync import sync_greeks_calibration
from marketsim.risk.daily_calibration import DailyCalibratedSimulator
from marketsim.scenarios.gpw import prepare as prepare_gpw
from marketsim.scenarios.prepared_cache import PreparedDatasetCache
from marketsim.scenarios.wig20 import prepare as prepare_wig20
from marketsim.simulator import Simulator

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "input"
DATASETS = {
    "wig20": ("wig20.yaml", "wig20_auto.yaml", prepare_wig20),
    "gpw": ("gpw.yaml", "gpw_auto.yaml", prepare_gpw),
}


def kwargs_for(func: Callable, config: dict) -> dict:
    params = inspect.signature(func).parameters
    return {key: value for key, value in config.items() if key in params}


def prepare_dataset(name: str, *, refresh_data: bool = False) -> tuple[str, bool]:
    """Prepare a dataset once; reuse fresh local OHLCV on subsequent runs."""
    settings_file, generated_file, prepare = DATASETS[name]
    settings = load_config(settings_file)

    cache = PreparedDatasetCache(root=ROOT)
    realism_enabled = settings.get("market_realism", {}).get("enabled", True)
    if not refresh_data and cache.use_if_fresh(
        dataset=name, settings=settings, yaml_name=generated_file
    ):
        return generated_file, realism_enabled

    # Only the cache miss/stale path needs to log into Snowflake.
    try:
        connection = connect()
    except Exception as exc:
        raise RuntimeError(f"Cannot connect to Ikaros Snowflake: {exc}") from exc

    try:
        prepared_path, cutoff, detail = prepare(connection, settings, ROOT)
    finally:
        connection.close()

    cache.record_preparation(dataset=name, settings=settings, yaml_name=generated_file)

    if name == "wig20":
        print(f"Prepared WIG20: {prepared_path}; cutoff={cutoff}; index close={detail:.2f}")
    else:
        print(f"Prepared GPW: {prepared_path}; cutoff={cutoff}; markets={detail}")

    return generated_file, realism_enabled


def run_simulation(config_file: str, *, realism_enabled: bool = False) -> None:
    config = load_config(config_file)
    output_dir = create_run_directory()

    # Preserve the resolved configuration for inspection and reproducibility.
    with (output_dir / "market_structure.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, default_flow_style=False)

    config["output_dir"] = output_dir
    historical = load_historical_prices(config=config, config_dir=INPUT_DIR)
    simulation_dates = load_simulation_dates(config)

    sim_kwargs = kwargs_for(Simulator, config)
    sim_kwargs["historical_ohlcv_by_symbol"] = historical
    sim_kwargs["simulation_dates"] = simulation_dates

    # Ordinary YAML configurations can opt in via market_realism.enabled.
    realism_enabled = realism_enabled or config.get("market_realism", {}).get("enabled", False)
    realism_report = MarketRealismReport() if realism_enabled else None
    sim_kwargs["market_realism_report"] = realism_report

    print("Historical OHLCV:", {symbol: len(frame) for symbol, frame in historical.items()})
    print("Simulation dates:", simulation_dates)
    print("Output directory:", output_dir)

    simulator = DailyCalibratedSimulator(risk_var=config.get("risk_var"), **sim_kwargs)
    try:
        sync_greeks_calibration(simulator, config)
        simulator.run()
        if realism_report is not None:
            report = realism_report.generate(simulator, output_dir)
            print(f"Market realism report: {report['symbol'].nunique()} instruments analyzed")
    finally:
        close = getattr(simulator.repository, "close", None)
        if callable(close):
            close()


def main(argv: list[str] | None = None) -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run Daedalus market simulations")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="YAML file in input (default: market_structure.yaml)",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        help="Prepare and simulate a market dataset from Ikaros (cached when fresh)",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare the selected dataset without running the simulation",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Ignore the local market-data cache and fetch again from Snowflake",
    )
    args = parser.parse_args(argv)

    if args.dataset and args.config:
        parser.error("Specify either a YAML configuration or --dataset, not both")
    if args.prepare_only and not args.dataset:
        parser.error("--prepare-only requires --dataset")
    if args.refresh_data and not args.dataset:
        parser.error("--refresh-data requires --dataset")

    if args.dataset:
        config_file, realism_enabled = prepare_dataset(args.dataset, refresh_data=args.refresh_data)
        if args.prepare_only:
            return
    else:
        config_file = args.config or "market_structure.yaml"
        realism_enabled = False

    run_simulation(config_file, realism_enabled=realism_enabled)


if __name__ == "__main__":
    main()
