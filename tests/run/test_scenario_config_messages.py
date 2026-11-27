"""Configuration mistakes should produce concise CLI messages before a run starts."""

from pathlib import Path

import pytest

from input import config as scenario_config
from run import simulation_main as cli


def _input(tmp_path: Path, monkeypatch) -> Path:
    directory = tmp_path / "input"
    directory.mkdir()
    monkeypatch.setattr(scenario_config, "INPUT_DIR", directory)
    monkeypatch.setattr(cli, "INPUT_DIR", directory)
    monkeypatch.setattr(cli, "load_dotenv", lambda *args: None)
    monkeypatch.setattr(
        cli,
        "create_run_directory",
        lambda: pytest.fail("Expected configuration validation before creating output/"),
    )
    return directory


def test_unknown_yaml_has_friendly_message_and_no_empty_run(tmp_path, monkeypatch, capsys):
    directory = _input(tmp_path, monkeypatch)
    (directory / "market_structure.yaml").write_text(
        "sim_time: 5\nmarkets:\n  test: {name: Test}\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(["unknown_file.yaml"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "unknown_file.yaml" in captured.err
    assert "was not found in input/" in captured.err
    assert "market_structure.yaml" in captured.err
    assert not (tmp_path / "output").exists()


def test_typo_gets_suggested_filename(tmp_path, monkeypatch, capsys):
    directory = _input(tmp_path, monkeypatch)
    (directory / "derivatives_1.yaml").write_text("markets: {test: {name: Test}}", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        cli.main(["derivativs_1.yaml"])
    assert exc.value.code == 2
    assert "Did you mean derivatives_1.yaml?" in capsys.readouterr().err


def test_missing_include_gives_template_location(tmp_path, monkeypatch, capsys):
    directory = _input(tmp_path, monkeypatch)
    (directory / "scenario.yaml").write_text(
        "markets:\n  security_1: !include templates/markets/missing.yaml\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(["scenario.yaml"])
    assert exc.value.code == 2
    message = capsys.readouterr().err
    assert "Missing template 'templates/markets/missing.yaml'" in message
    assert "scenario.yaml" in message
    assert "Traceback" not in message


def test_invalid_yaml_reports_line_instead_of_traceback(tmp_path, monkeypatch, capsys):
    directory = _input(tmp_path, monkeypatch)
    (directory / "bad.yaml").write_text("markets:\n  a: [broken\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        cli.main(["bad.yaml"])
    assert exc.value.code == 2
    message = capsys.readouterr().err
    assert "Invalid YAML in 'bad.yaml'" in message
    assert "line" in message
    assert "Traceback" not in message


def test_settings_yaml_is_not_treated_as_simulation(tmp_path, monkeypatch, capsys):
    directory = _input(tmp_path, monkeypatch)
    (directory / "wig20.yaml").write_text("dataset: wig20\ndays: 20\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        cli.main(["wig20.yaml"])
    assert exc.value.code == 2
    message = capsys.readouterr().err
    assert "contains no markets" in message
    assert "--dataset wig20" in message
    assert "Traceback" not in message


def test_valid_template_is_resolved_relative_to_yaml(tmp_path, monkeypatch):
    directory = _input(tmp_path, monkeypatch)
    templates = directory / "templates" / "markets"
    templates.mkdir(parents=True)
    (templates / "test.yaml").write_text("name: Test market\n", encoding="utf-8")
    (directory / "scenario.yaml").write_text(
        "markets:\n  test: !include templates/markets/test.yaml\n", encoding="utf-8"
    )
    result = scenario_config.load_config("scenario.yaml")
    assert result["markets"]["test"]["name"] == "Test market"


def test_catalog_check_is_read_only_and_marks_external_data(tmp_path, monkeypatch, capsys):
    directory = _input(tmp_path, monkeypatch)
    (directory / "synthetic.yaml").write_text("markets: {test: {name: Test}}\n", encoding="utf-8")
    (directory / "wig20.yaml").write_text("dataset: wig20\n", encoding="utf-8")
    (directory / "historical.yaml").write_text(
        "markets: {test: {name: Test}}\n"
        "historical_prices:\n"
        "  enabled: true\n"
        "  path: market_data/missing.csv\n",
        encoding="utf-8",
    )
    cli.main(["--check-scenarios"])
    message = capsys.readouterr().out
    assert "2 runnable YAML configurations parsed successfully" in message
    assert "Preparation/settings YAMLs: 1" in message
    assert "Historical input needed: 1 scenarios" in message
    assert "No simulations, database writes, or external connections" in message
    assert not (tmp_path / "output").exists()


def test_old_scenario_catalog_loads_without_simulating():
    """Parsing all checked-in YAMLs catches broken !include paths and syntax."""
    files = sorted(scenario_config.INPUT_DIR.glob("*.yaml"))
    assert files, "No scenario definitions found under input/"
    for path in files:
        scenario_config.load_config(path.name)


def test_bad_historical_csv_is_reported_before_output(tmp_path, monkeypatch, capsys):
    directory = _input(tmp_path, monkeypatch)
    csv_path = directory / "market_data" / "existing.csv"
    csv_path.parent.mkdir()
    csv_path.write_text("unreadable CSV contents", encoding="utf-8")
    (directory / "scenario.yaml").write_text(
        "markets: {test: {name: Test}}\n"
        "historical_prices:\n"
        "  enabled: true\n"
        "  path: market_data/existing.csv\n",
        encoding="utf-8",
    )

    def malformed(**kwargs):
        raise ValueError("Historical OHLCV missing columns: ['close']")

    monkeypatch.setattr(cli, "load_historical_prices", malformed)
    with pytest.raises(SystemExit) as exc:
        cli.main(["scenario.yaml"])
    assert exc.value.code == 2
    captured = capsys.readouterr().err
    assert "Cannot load historical prices" in captured
    assert "missing columns" in captured
    assert "Traceback" not in captured
    assert not (tmp_path / "output").exists()
