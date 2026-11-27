"""Load Daedalus scenario YAML and reusable templates with readable diagnostics."""

from __future__ import annotations

import random
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG = "market_structure.yaml"
INPUT_DIR = Path(__file__).resolve().parent


class ScenarioConfigurationError(ValueError):
    """An invalid or unavailable scenario file (not a simulation engine failure)."""


class IncludeLoader(yaml.SafeLoader):
    """Safe YAML loader with includes resolved relative to their containing file."""


def available_configurations() -> list[str]:
    """List top-level input YAMLs, excluding reusable templates."""
    return sorted(path.name for path in INPUT_DIR.glob("*.yaml") if path.is_file())


def _short_path(path: Path) -> str:
    try:
        return str(path.relative_to(INPUT_DIR.resolve()))
    except ValueError:
        return str(path)


def _load_yaml(path: Path) -> object:
    try:
        with open(path, "r", encoding="utf-8") as stream:
            loader = IncludeLoader(stream)
            loader.name = str(path)
            try:
                return loader.get_single_data()
            finally:
                loader.dispose()
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        reason = getattr(exc, "problem", None) or "invalid YAML syntax"
        raise ScenarioConfigurationError(
            f"Invalid YAML in '{_short_path(path)}'{location}: {reason}."
        ) from exc
    except OSError as exc:
        raise ScenarioConfigurationError(
            f"Cannot read scenario file '{_short_path(path)}': {exc.strerror or exc}."
        ) from exc


def include_constructor(loader: IncludeLoader, node: yaml.Node) -> object:
    relative_path = loader.construct_scalar(node)
    parent = Path(loader.name).parent
    template = (parent / relative_path).resolve()
    if not template.is_relative_to(INPUT_DIR.resolve()):
        raise ScenarioConfigurationError(
            f"Template '{relative_path}' in '{_short_path(Path(loader.name))}' "
            "must be inside input/."
        )
    if not template.is_file():
        raise ScenarioConfigurationError(
            f"Missing template '{relative_path}' included by '{_short_path(Path(loader.name))}'."
        )
    return _load_yaml(template)


IncludeLoader.add_constructor("!include", include_constructor)


def load_config(filename: str = DEFAULT_CONFIG) -> dict:
    """Read an input scenario/settings file without triggering a simulation."""
    requested = Path(filename)
    if not requested.is_absolute() and requested.parts and requested.parts[0] == "input":
        requested = Path(*requested.parts[1:])
    path = (requested if requested.is_absolute() else INPUT_DIR / requested).resolve()
    if not path.is_relative_to(INPUT_DIR.resolve()):
        raise ScenarioConfigurationError("Scenario files must be located in input/.")
    if not path.is_file():
        available = available_configurations()
        similar = get_close_matches(requested.name, available, n=3, cutoff=0.45)
        if similar:
            hint = "Did you mean " + ", ".join(similar) + "?"
        else:
            examples = ("market_structure.yaml", "derivatives_1.yaml", "futures_1.yaml")
            suggestions = [name for name in examples if name in available]
            hint = (
                "Try " + ", ".join(suggestions) + "."
                if suggestions
                else "Add a YAML file to input/."
            )
        raise ScenarioConfigurationError(
            f"Scenario '{requested.name}' was not found in input/. {hint} "
            "See input/ for available scenarios."
        )

    config = _load_yaml(path)
    if not isinstance(config, dict):
        raise ScenarioConfigurationError(
            f"Scenario '{_short_path(path)}' must contain a YAML mapping, not a list or empty file."
        )

    seed = config.get("seed", 67)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ScenarioConfigurationError(
            f"Scenario '{_short_path(path)}': seed must be an integer from 0 to {2**32 - 1}."
        )
    random.seed(seed)
    np.random.seed(seed)
    return config


def create_run_directory() -> Path:
    output_dir = (
        Path(__file__).resolve().parents[1]
        / "output"
        / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
