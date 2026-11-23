from copy import deepcopy


def build_scenario_settings(
    *,
    profile: dict,
    settings: dict,
) -> dict:
    """Combine agent profile and scenario-specific settings.

    Scenario settings take precedence over profile defaults.
    """
    config = deepcopy(profile)

    defaults = {
        "days": 50,
        "sim_time": 60,
        "seed": 67,
    }

    for key, default in defaults.items():
        config[key] = int(settings.get(key, default))

    # These sections are owned by the scenario configuration.
    # Replace the whole section rather than merging obsolete defaults.
    for section in ("risk_var", "risk_greeks", "market_realism"):
        if section in settings:
            config[section] = deepcopy(settings[section])

    return config
