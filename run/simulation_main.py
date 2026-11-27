"""Single entry point for Daedalus simulations and prepared datasets."""

from __future__ import annotations

import argparse
import inspect
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import yaml
from dotenv import load_dotenv

from input.config import (
    ScenarioConfigurationError,
    create_run_directory,
    load_config,
)
from marketsim.analysis.report import MarketRealismReport
from marketsim.database.ikaros.ikaros_local import (
    CachedIkarosConnection,
    stage_local_ikaros,
    validate_wig20_snapshot,
)
from marketsim.database.ikaros.ikaros_queries import connect
from marketsim.database.prepared_snapshot_restore import restore_historical_csv
from marketsim.database.risk.risk_reporting_queries import load_wig20_greeks
from marketsim.market.historical_prices import load_historical_prices, load_simulation_dates
from marketsim.plot.risk_report import greeks_evolution_plots
from marketsim.risk.calibration_sync import (
    GreeksCalibrationSetupError,
    sync_greeks_calibration,
)
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


def _promote_prepared_files(
    *,
    work_root: Path,
    generated_file: str,
    cache: PreparedDatasetCache,
    dataset: str,
    settings: dict,
) -> None:
    """Publish a prepared YAML/CSV pair and roll back on cache-store failure."""
    sources = [
        (
            work_root / "input" / "market_data" / f"{dataset}_auto.ohlcv.csv",
            INPUT_DIR / "market_data" / f"{dataset}_auto.ohlcv.csv",
        ),
        (work_root / "input" / generated_file, INPUT_DIR / generated_file),
    ]
    backups: list[tuple[Path, Path | None]] = []
    try:
        for index, (source, target) in enumerate(sources):
            if not source.is_file():
                raise FileNotFoundError(f"Preparatory output missing: {source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            backup = None
            if target.is_file():
                backup = work_root / f"backup_{index}"
                shutil.copy2(target, backup)
            backups.append((target, backup))
            temporary_target = target.with_name(target.name + ".incoming")
            try:
                shutil.copy2(source, temporary_target)
                os.replace(temporary_target, target)
            finally:
                temporary_target.unlink(missing_ok=True)
        cache.record_preparation(dataset=dataset, settings=settings, yaml_name=generated_file)
    except Exception:
        for target, backup in reversed(backups):
            if backup is None:
                target.unlink(missing_ok=True)
            else:
                os.replace(backup, target)
        raise


def prepare_dataset(
    name: str,
    *,
    refresh_data: bool = False,
    source: str = "snowflake",
    ikaros_db: Path | None = None,
) -> tuple[str, bool]:
    """Prepare a dataset once; reuse fresh local OHLCV on subsequent runs."""
    settings_file, generated_file, prepare = DATASETS[name]
    settings = load_config(settings_file)

    cache = PreparedDatasetCache(root=ROOT)
    realism_enabled = settings.get("market_realism", {}).get("enabled", True)
    if source == "duckdb":
        if name != "wig20":
            raise ValueError("Local Ikaros preparation is currently implemented for WIG20 only")
        snapshot = stage_local_ikaros(
            source_path=ikaros_db or (INPUT_DIR / "market_data" / "ikaros.duckdb"),
            cache_path=cache.db_path,
        )
        print(f"Imported read-only Ikaros source into {cache.db_path}: {snapshot.table_rows}")
        cutoff, issues = validate_wig20_snapshot(
            cache_path=cache.db_path,
            settings=settings,
            previous_yaml=INPUT_DIR / generated_file,
        )
        if issues:
            details = "\n  - ".join(issues)
            raise RuntimeError(
                f"Ikaros local snapshot through {cutoff} is not yet a complete "
                f"WIG20 input. Prepared simulation cache is untouched.\n  - {details}"
            )
        # Prepare into a temporary directory so any error leaves the existing
        # generated YAML/CSV pair intact. Ikaros's connection is already closed.
        with TemporaryDirectory(prefix="daedalus_local_prepare_", dir=ROOT) as temp:
            work_root = Path(temp)
            with CachedIkarosConnection(cache_path=cache.db_path) as connection:
                _, cutoff, detail = prepare(connection, settings, work_root)
            _promote_prepared_files(
                work_root=work_root,
                generated_file=generated_file,
                cache=cache,
                dataset=name,
                settings=settings,
            )
        print(f"Prepared WIG20 locally: {generated_file}; cutoff={cutoff}; index={detail:.2f}")
        return generated_file, realism_enabled
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


def plot_wig20_greeks_after_run(*, output_dir: Path) -> list[Path]:
    """Create WIG20 Delta/Gamma plots after the simulation DB is closed."""
    db = output_dir / "daedalus.duckdb"
    if not db.is_file():
        print("WIG20 Greeks plots: no local DuckDB for this run")
        return []

    try:
        values = load_wig20_greeks(db)
    except (ValueError, OSError, duckdb.Error) as exc:
        print(f"WIG20 Greeks plots skipped: {exc}")
        return []
    if values.empty:
        print(f"WIG20 Greeks plots: {values.attrs.get('diagnostic', 'no EoD option Greeks')}")
        return []

    report_dir = output_dir / "risk_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    values.to_csv(report_dir / "greeks_wig20.csv", index=False)
    agent_ids = sorted(values.agent_id.astype(int).unique().tolist())
    images = greeks_evolution_plots(values, out=report_dir, agent_ids=agent_ids)
    print(f"WIG20 Greeks plots: {len(images)} charts in {report_dir}")
    return images


class ScenarioInputError(Exception):
    """A normal, actionable setup problem rather than a simulator exception."""


def _historical_input_path(*, scenario: dict) -> Path | None:
    """Get an enabled scenario's historical CSV path without opening it."""
    settings = scenario.get("historical_prices") or {}
    if not settings.get("enabled", False):
        return None
    configured = settings.get("path")
    if not configured or not isinstance(configured, (str, Path)):
        raise ScenarioInputError(
            "Historical prices are enabled, but their CSV path is missing in the YAML."
        )
    path = Path(configured)
    return path if path.is_absolute() else INPUT_DIR / path


def _local_ikaros_sources() -> list[Path]:
    """Look only in conventional locations; never scan arbitrary user files."""
    candidates = (
        INPUT_DIR / "market_data" / "ikaros.duckdb",
        ROOT.parent / "Ikaros" / "input" / "market_data" / "ikaros.duckdb",
        ROOT.parent / "ikaros" / "input" / "market_data" / "ikaros.duckdb",
    )
    return [path for path in candidates if path.is_file()]


def _has_snowflake_settings() -> bool:
    """Do not attempt an interactive or unauthenticated external connection."""
    if not os.getenv("SNOWFLAKE_ACCOUNT") or not os.getenv("SNOWFLAKE_USER"):
        return False
    authenticator = os.getenv("SNOWFLAKE_AUTHENTICATOR", "snowflake")
    return authenticator not in {"snowflake", "username_password_mfa"} or bool(
        os.getenv("SNOWFLAKE_PASSWORD")
    )


def ensure_historical_input(*, config_file: str, scenario: dict) -> bool:
    """Prepare a missing generated WIG20 CSV; return True if YAML was rebuilt.

    No Ikaros input or Snowflake login is required when the file is present.
    A dated, matching prepared cache is accepted even when it is old: the
    explicit generated YAML is a reproducible snapshot, not a latest-data query.
    """
    path = _historical_input_path(scenario=scenario)
    if path is None or (path.is_file() and path.stat().st_size > 0):
        return False

    if Path(config_file).name != "wig20_auto.yaml":
        raise ScenarioInputError(
            f"Required historical prices are missing: {path.name}. "
            "Provide the file specified by historical_prices.path and try again."
        )

    print("WIG20 historical prices are not installed. Checking reusable market data...")
    cache = PreparedDatasetCache(root=ROOT)
    try:
        if restore_historical_csv(
            cache_path=cache.db_path,
            output_path=path,
            dataset="wig20",
            scenario=scenario,
        ):
            print("Restored matching WIG20 history from the local DuckDB cache.")
            return False
    except (OSError, ValueError, duckdb.Error):
        # If DuckDB is locked, local Ikaros or Snowflake may still be available.
        print("The local market cache is unavailable; checking other data sources.")

    failures: list[str] = []
    for source in _local_ikaros_sources():
        print(f"Preparing WIG20 from local Ikaros data: {source}")
        try:
            prepare_dataset("wig20", source="duckdb", ikaros_db=source)
        except (OSError, RuntimeError, ValueError, duckdb.Error) as exc:
            detail = str(exc).splitlines()[0][:140]
            failures.append(f"Local Ikaros data need an update: {detail}")
        else:
            if path.is_file() and path.stat().st_size > 0:
                return True

    if _has_snowflake_settings():
        print("Preparing WIG20 from configured Snowflake market data...")
        try:
            prepare_dataset("wig20", source="snowflake")
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            detail = str(exc).splitlines()[0][:140]
            failures.append(f"Snowflake preparation needs attention: {detail}")
        else:
            if path.is_file() and path.stat().st_size > 0:
                return True

    reason = "\n".join(failures)
    if reason:
        reason += "\n"
    raise ScenarioInputError(
        f"{reason}A real WIG20 simulation needs historical Ikaros data. "
        "Put a complete ikaros.duckdb in input/market_data/ or configure "
        "Snowflake in .env. Then run the same command again. "
        "No empty simulation was started."
    )


def _verify_simulation_config(*, config_file: str, config: dict) -> None:
    """Reject preparation-only YAMLs before any run directory is created."""
    if not isinstance(config.get("markets"), dict) or not config["markets"]:
        name = Path(config_file).name
        if name == "wig20.yaml":
            instruction = "Use: python -m run.simulation_main --dataset wig20"
        elif name == "gpw.yaml":
            instruction = "Use: python -m run.simulation_main --dataset gpw"
        else:
            instruction = "Choose a market scenario from input/, e.g. market_structure.yaml"
        raise ScenarioConfigurationError(
            f"'{name}' contains no markets, so it cannot be simulated directly. {instruction}"
        )


def check_scenarios() -> bool:
    """Check every input YAML and include without starting a simulation.

    This is a configuration preflight, not a guarantee of available market data
    or a substitute for actually executing each scenario.
    """
    filenames = sorted(path.name for path in INPUT_DIR.glob("*.yaml"))
    ready = 0
    needs_history: list[str] = []
    settings_only: list[str] = []
    invalid: list[tuple[str, str]] = []

    for name in filenames:
        try:
            config = load_config(name)
            if not isinstance(config.get("markets"), dict) or not config["markets"]:
                settings_only.append(name)
                continue
            history = _historical_input_path(scenario=config)
            if history is not None and (not history.is_file() or history.stat().st_size == 0):
                needs_history.append(name)
            ready += 1
        except (ScenarioConfigurationError, ScenarioInputError) as exc:
            invalid.append((name, str(exc)))

    print(f"Scenario check: {ready} runnable YAML configurations parsed successfully.")
    print(f"Preparation/settings YAMLs: {len(settings_only)} (not run directly).")
    if needs_history:
        print(
            f"Historical input needed: {len(needs_history)} scenarios "
            "(their YAML is valid, but their CSV is not installed)."
        )
        print("  " + ", ".join(needs_history))
    if invalid:
        print(f"Configuration problems: {len(invalid)}")
        for name, message in invalid:
            print(f"  {name}: {message}")
    else:
        print("All top-level YAML files and referenced templates loaded successfully.")
    print("No simulations, database writes, or external connections were started.")
    return not invalid


def run_simulation(config_file: str, *, realism_enabled: bool = False) -> None:
    config = load_config(config_file)
    _verify_simulation_config(config_file=config_file, config=config)
    if ensure_historical_input(config_file=config_file, scenario=config):
        # Preparation may refresh the generated instruments and dates.
        config = load_config(config_file)
        _verify_simulation_config(config_file=config_file, config=config)

    # Validate ordinary inputs before starting a run or creating an output directory.
    try:
        historical = load_historical_prices(config=config, config_dir=INPUT_DIR)
    except (OSError, ValueError) as exc:
        raise ScenarioInputError(
            f"Cannot load historical prices for '{Path(config_file).name}': "
            f"{str(exc).splitlines()[0] if str(exc).splitlines() else type(exc).__name__}. "
            "Check the CSV in input/market_data/. No simulation was started."
        ) from exc
    try:
        simulation_dates = load_simulation_dates(config)
    except ValueError as exc:
        raise ScenarioConfigurationError(
            f"Invalid simulation calendar in '{Path(config_file).name}': {exc}"
        ) from exc

    output_dir = create_run_directory()
    # Preserve the resolved configuration for inspection and reproducibility.
    with (output_dir / "market_structure.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, default_flow_style=False)

    config["output_dir"] = output_dir

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
    original_greeks_id = (config.get("risk_greeks") or {}).get("calibration_id")
    try:
        sync_greeks_calibration(simulator, config)
        if (config.get("risk_greeks") or {}).get("calibration_id") != original_greeks_id:
            # The effective local calibration has its own immutable ID. Keep
            # the copied run configuration reproducible; leave input/ untouched.
            persisted = {key: value for key, value in config.items() if key != "output_dir"}
            with (output_dir / "market_structure.yaml").open("w", encoding="utf-8") as stream:
                yaml.safe_dump(persisted, stream, sort_keys=False, default_flow_style=False)
        simulator.run()
        if realism_report is not None:
            report = realism_report.generate(simulator, output_dir)
            print(f"Market realism report: {report['symbol'].nunique()} instruments analyzed")
    finally:
        close = getattr(simulator.repository, "close", None)
        if callable(close):
            close()

    # Never open a second connection while DuckDB is still held by the simulator.
    if config.get("risk_greeks", {}).get("enabled", False):
        plot_wig20_greeks_after_run(output_dir=output_dir)
    elif any(m.instrument_class == "option" for m in simulator.markets.values()):
        print("Greeks plots skipped: risk_greeks is disabled for this scenario")


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
        "--check-scenarios",
        action="store_true",
        help="Check all scenario YAMLs and their included templates without simulating",
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
    parser.add_argument(
        "--source",
        choices=("snowflake", "duckdb"),
        default="snowflake",
        help="For --dataset, prepare from Snowflake or staged local Ikaros DuckDB",
    )
    parser.add_argument(
        "--ikaros-db",
        type=Path,
        help="Optional immutable Ikaros DuckDB snapshot (default: input/market_data/ikaros.duckdb)",
    )
    args = parser.parse_args(argv)

    if args.check_scenarios:
        if args.config or args.dataset or args.prepare_only or args.refresh_data:
            parser.error("--check-scenarios cannot be combined with a simulation or dataset")
        try:
            success = check_scenarios()
        except OSError as exc:
            parser.exit(2, f"Scenario check could not read input/: {exc}\n")
        if not success:
            parser.exit(2, "Scenario check found configuration problems.\n")
        return

    if args.dataset and args.config:
        parser.error("Specify either a YAML configuration or --dataset, not both")
    if args.prepare_only and not args.dataset:
        parser.error("--prepare-only requires --dataset")
    if args.refresh_data and not args.dataset:
        parser.error("--refresh-data requires --dataset")
    if args.source == "duckdb" and not args.dataset:
        parser.error("--source duckdb requires --dataset")
    if args.source == "duckdb" and args.refresh_data:
        parser.error("--source duckdb imports local data automatically; omit --refresh-data")
    if args.ikaros_db is not None and args.source != "duckdb":
        parser.error("--ikaros-db requires --source duckdb")

    if args.dataset:
        try:
            config_file, realism_enabled = prepare_dataset(
                args.dataset,
                refresh_data=args.refresh_data,
                source=args.source,
                ikaros_db=args.ikaros_db,
            )
        except ScenarioConfigurationError as exc:
            parser.exit(2, f"Daedalus setup: {exc}\n")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            if args.source != "duckdb":
                raise
            parser.exit(2, f"Local Ikaros preparation failed:\n{exc}\n")
        if args.prepare_only:
            return
    else:
        config_file = args.config or "market_structure.yaml"
        realism_enabled = False

    try:
        run_simulation(config_file, realism_enabled=realism_enabled)
    except (ScenarioConfigurationError, ScenarioInputError, GreeksCalibrationSetupError) as exc:
        parser.exit(2, f"Daedalus setup: {exc}\n")


if __name__ == "__main__":
    main()
