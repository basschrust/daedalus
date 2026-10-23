from decimal import Decimal

import pytest

from marketsim.market import FXRateProvider


def test_fx_triangulation():
    fx = FXRateProvider()

    fx.set_rate(day=0, base="USD", quote="PLN", rate="3.80")
    fx.set_rate(day=0, base="EUR", quote="PLN", rate="4.30")

    assert fx.get_rate(day=0, base="USD", quote="EUR") == Decimal("3.80") / Decimal("4.30")

    assert fx.convert(
        Decimal("100"),
        day=0,
        base="USD",
        quote="EUR",
    ) == Decimal("100") * Decimal("3.80") / Decimal("4.30")


def test_fx_triangulation_missing_leg():
    fx = FXRateProvider()
    fx.set_rate(day=0, base="USD", quote="PLN", rate="3.80")

    with pytest.raises(KeyError):
        fx.get_rate(day=0, base="USD", quote="EUR")
