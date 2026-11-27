"""Greeks should reuse local observed risk calibration before any Snowflake call."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from marketsim.risk import calibration_sync
from run import simulation_main as cli


def _simulator(*, daily_id: str | None = "SIM-LOCAL") -> SimpleNamespace:
    underlying = SimpleNamespace(symbol="WIG20")
    option = SimpleNamespace(instrument_class="option", underlying=underlying)
    repository = Mock()
    repository.needs_local_greek_calibration.return_value = True
    repository.load_parametric_calibration.return_value = (
        [("ALE.WA", 0.0004, 0.0), ("WIG20", 0.0001, 0.014)],
        [],
    )
    simulator = SimpleNamespace(
        repository=repository,
        markets={1: option},
        daily_risk_calibration=(
            SimpleNamespace(latest_calibration_id=daily_id) if daily_id else None
        ),
        historical_ohlcv_by_symbol={},
        risk_greeks={"enabled": True, "calibration_id": "REMOTE-L2"},
    )
    return simulator


def test_local_var_calibration_is_used_when_remote_id_not_cached(monkeypatch):
    cache = Mock()
    cache.get_greek_calibration.return_value = None
    monkeypatch.setattr(calibration_sync, "PreparedDatasetCache", lambda **kwargs: cache)
    simulator = _simulator()
    config = {"risk_greeks": simulator.risk_greeks}

    calibration_sync.sync_greeks_calibration(simulator, config)

    assert config["risk_greeks"]["calibration_id"] == "SIM-LOCAL"
    simulator.repository.load_parametric_calibration.assert_called_once_with("SIM-LOCAL")
    simulator.repository.fetch_source_greek_calibration.assert_not_called()
    simulator.repository.import_greek_calibration.assert_not_called()
    cache.store_greek_calibration.assert_not_called()


def test_matching_local_L2_cache_is_kept_with_original_id(monkeypatch):
    cache = Mock()
    cache.get_greek_calibration.return_value = [("WIG20", 0.0002, 0.013)]
    monkeypatch.setattr(calibration_sync, "PreparedDatasetCache", lambda **kwargs: cache)
    simulator = _simulator()
    config = {"risk_greeks": simulator.risk_greeks}

    calibration_sync.sync_greeks_calibration(simulator, config)

    assert config["risk_greeks"]["calibration_id"] == "REMOTE-L2"
    simulator.repository.import_greek_calibration.assert_called_once_with(
        "REMOTE-L2", {"WIG20": (0.0002, 0.013)}
    )
    simulator.repository.load_parametric_calibration.assert_not_called()
    simulator.repository.fetch_source_greek_calibration.assert_not_called()


def test_no_local_history_produces_actionable_calibration_error(monkeypatch):
    cache = Mock()
    cache.get_greek_calibration.return_value = None
    monkeypatch.setattr(calibration_sync, "PreparedDatasetCache", lambda **kwargs: cache)
    simulator = _simulator(daily_id=None)
    simulator.repository.fetch_source_greek_calibration.side_effect = RuntimeError(
        'Snowflake support is not installed. Run: python -m pip install -e ".[snowflake]"'
    )

    with pytest.raises(calibration_sync.GreeksCalibrationSetupError, match="local historical"):
        calibration_sync.sync_greeks_calibration(simulator, {"risk_greeks": simulator.risk_greeks})


def test_explicit_local_mode_does_not_attempt_snowflake(monkeypatch):
    cache = Mock()
    cache.get_greek_calibration.return_value = None
    monkeypatch.setattr(calibration_sync, "PreparedDatasetCache", lambda **kwargs: cache)
    simulator = _simulator(daily_id=None)
    config = {"risk_greeks": {**simulator.risk_greeks, "source": "local"}}

    with pytest.raises(calibration_sync.GreeksCalibrationSetupError, match="No usable local"):
        calibration_sync.sync_greeks_calibration(simulator, config)

    simulator.repository.fetch_source_greek_calibration.assert_not_called()


def test_cli_hides_expected_calibration_traceback(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda *args: None)
    monkeypatch.setattr(
        cli,
        "run_simulation",
        Mock(
            side_effect=calibration_sync.GreeksCalibrationSetupError("Missing local Greeks input.")
        ),
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(["wig20_auto.yaml"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "Daedalus setup: Missing local Greeks input." in captured.err


def test_effective_local_calibration_id_is_saved_in_run_yaml(monkeypatch, tmp_path):
    config = {
        "markets": {"wig20": {"instrument_class": "index"}},
        "days": 1,
        "sim_time": 5,
        "risk_greeks": {"enabled": True, "calibration_id": "REMOTE-L2"},
    }
    monkeypatch.setattr(cli, "load_config", lambda *args: config)
    monkeypatch.setattr(cli, "ensure_historical_input", lambda **kwargs: False)
    monkeypatch.setattr(cli, "load_historical_prices", lambda **kwargs: {})
    monkeypatch.setattr(cli, "load_simulation_dates", lambda *args: None)
    monkeypatch.setattr(cli, "create_run_directory", lambda: tmp_path)

    simulator = Mock()
    simulator.repository.close.return_value = None
    simulator.markets = {}
    monkeypatch.setattr(cli, "DailyCalibratedSimulator", lambda **kwargs: simulator)

    def local_calibration(_simulator, configuration):
        configuration["risk_greeks"]["calibration_id"] = "SIM-LOCAL"

    monkeypatch.setattr(cli, "sync_greeks_calibration", local_calibration)
    monkeypatch.setattr(cli, "plot_wig20_greeks_after_run", lambda **kwargs: [])

    cli.run_simulation("wig20_auto.yaml")

    saved = yaml.safe_load((tmp_path / "market_structure.yaml").read_text())
    assert saved["risk_greeks"]["calibration_id"] == "SIM-LOCAL"
    assert saved.get("output_dir") is None
    simulator.run.assert_called_once()


def test_locked_shared_cache_falls_back_to_local_daily_calibration(monkeypatch):
    import duckdb

    cache = Mock()
    cache.get_greek_calibration.side_effect = duckdb.Error("shared cache locked")
    monkeypatch.setattr(calibration_sync, "PreparedDatasetCache", lambda **kwargs: cache)
    simulator = _simulator()
    calibration_sync.sync_greeks_calibration(simulator, {"risk_greeks": simulator.risk_greeks})
    assert simulator.risk_greeks["calibration_id"] == "SIM-LOCAL"
    simulator.repository.fetch_source_greek_calibration.assert_not_called()


def test_greeks_only_scenario_calibrates_from_observed_history(monkeypatch):
    import marketsim.risk.daily_calibration as daily_module

    cache = Mock()
    cache.get_greek_calibration.return_value = None
    monkeypatch.setattr(calibration_sync, "PreparedDatasetCache", lambda **kwargs: cache)
    simulator = _simulator(daily_id=None)
    simulator.historical_ohlcv_by_symbol = {"WIG20": object()}
    observed = {}

    def fake_daily(simulator_arg, *, lookback_sessions):
        observed["simulator"] = simulator_arg
        observed["lookback_sessions"] = lookback_sessions
        return SimpleNamespace(latest_calibration_id="SIM-LOCAL")

    monkeypatch.setattr(daily_module, "DailyRiskCalibration", fake_daily)
    config = {"risk_greeks": {**simulator.risk_greeks, "lookback_sessions": 125}}
    calibration_sync.sync_greeks_calibration(simulator, config)

    assert observed == {"simulator": simulator, "lookback_sessions": 125}
    assert config["risk_greeks"]["calibration_id"] == "SIM-LOCAL"
    simulator.repository.fetch_source_greek_calibration.assert_not_called()
