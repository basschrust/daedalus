from copy import deepcopy

GENERATED_KEYS = ("markets", "historical_prices", "start_date")


def build_scenario_settings(
    *,
    settings: dict,
    market_data: dict,
    profile: dict | None = None,
) -> dict:
    """Combine maintained settings with generated market data."""
    for key in GENERATED_KEYS:
        if key in settings:
            raise ValueError(f"{key} must come from market data")

    config = deepcopy(
        {key: value for key, value in (profile or {}).items() if key not in GENERATED_KEYS}
    )

    config.update(deepcopy(settings))

    for key in GENERATED_KEYS:
        config[key] = deepcopy(market_data[key])

    config.pop("dataset", None)
    config.pop("agents_profile", None)

    return config
