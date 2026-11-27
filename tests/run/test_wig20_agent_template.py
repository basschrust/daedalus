"""WIG20 scenario groups must be direct, even when old YAML was generated."""

from pathlib import Path

import pytest
import yaml

from input import config as scenario_config


def test_wig20_template_contains_agent_groups_not_agents_wrapper():
    root = Path(__file__).resolve().parents[2]
    template = yaml.safe_load(
        (root / "input" / "templates" / "agents" / "wig20_agents.yaml").read_text(encoding="utf-8")
    )
    assert "agents" not in template
    assert len(template) >= 10
    assert all(isinstance(group, dict) and group.get("agent_class") for group in template.values())


def test_wig20_include_resolves_direct_groups(tmp_path, monkeypatch):
    inputs = tmp_path / "input"
    template_dir = inputs / "templates" / "agents"
    template_dir.mkdir(parents=True)
    source = (
        Path(__file__).resolve().parents[2] / "input" / "templates" / "agents" / "wig20_agents.yaml"
    )
    (template_dir / "wig20_agents.yaml").write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (inputs / "wig20.yaml").write_text(
        "agents: !include templates/agents/wig20_agents.yaml\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(scenario_config, "INPUT_DIR", inputs)
    config = scenario_config.load_config("wig20.yaml")
    assert config["agents"]["arbitrageurs"]["agent_class"] == "IndexFuturesArbitrageAgent"
    assert "agents" not in config["agents"]


def test_existing_generated_wig20_file_is_repaired_in_memory(tmp_path, monkeypatch, capsys):
    inputs = tmp_path / "input"
    inputs.mkdir()
    generated = inputs / "wig20_auto.yaml"
    old_yaml = (
        "markets: {wig20: {instrument_class: index}}\n"
        "agents:\n"
        "  agents:\n"
        "    arbitrageurs: {agent_class: IndexFuturesArbitrageAgent, number: 5}\n"
        "    noise: {agent_class: NoiseAgent, number: 2}\n"
    )
    generated.write_text(old_yaml, encoding="utf-8")
    monkeypatch.setattr(scenario_config, "INPUT_DIR", inputs)
    config = scenario_config.load_config("wig20_auto.yaml")
    assert config["agents"]["noise"]["agent_class"] == "NoiseAgent"
    assert "agents" not in config["agents"]
    assert "corrected legacy nested" in capsys.readouterr().out
    assert generated.read_text(encoding="utf-8") == old_yaml
    assert not (tmp_path / "output").exists()


def test_malformed_agent_group_is_reported_without_keyerror(tmp_path, monkeypatch):
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "broken.yaml").write_text(
        "markets: {stock: {instrument_class: stock}}\nagents: {noise: {number: 5}}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(scenario_config, "INPUT_DIR", inputs)
    with pytest.raises(
        scenario_config.ScenarioConfigurationError, match="group 'noise'.*agent_class"
    ):
        scenario_config.load_config("broken.yaml")
    assert not (tmp_path / "output").exists()
