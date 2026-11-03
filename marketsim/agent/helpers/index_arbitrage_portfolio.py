from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal


def index_hedge_quantities(
    future,
    contracts: int,
) -> dict[int, int]:
    """Whole-share stock hedge for a signed futures position.

    Positive contracts -> short stock basket.
    Negative contracts -> long stock basket.

    Returns target stock positions, not orders.
    """
    index = future.underlying
    components = index.components

    weight_sum = sum(Decimal(str(c["weight"])) for c in components.values())

    if weight_sum <= 0:
        raise ValueError("Invalid index component weights")

    multiplier = Decimal(str(future.multiplier))

    if multiplier <= 0:
        raise ValueError("Invalid futures multiplier")

    result = {}

    for asset_id, component in components.items():
        weight = Decimal(str(component["weight"]))

        if index.base_index_level is None:
            coefficient = weight
        else:
            base_price = Decimal(str(component["base_price"]))

            if base_price <= 0:
                raise ValueError("Invalid component base price")

            coefficient = index.base_index_level * weight / (base_price * weight_sum)

        # Opposite exposure to the futures position.
        target = -Decimal(contracts) * multiplier * coefficient

        result[asset_id] = int(target.to_integral_value(rounding=ROUND_HALF_UP))

    return result
