from decimal import Decimal

import pytest

from marketsim.market.fx_rate_provider import FXRateProvider


def test_direct_and_inverse_rates():
    fx = FXRateProvider()
    fx.set_rate(day=0, base='USD', quote='PLN', rate='4')
    assert fx.get_rate(day=0, base='USD', quote='PLN') == Decimal('4')
    assert fx.get_rate(day=0, base='PLN', quote='USD') == Decimal('0.25')


def test_conversion_and_identity():
    fx = FXRateProvider({(2, 'EUR', 'PLN'): '4.25'})
    assert fx.convert('100', day=2, base='EUR', quote='PLN') == Decimal('425.00')
    assert fx.convert('100', day=2, base='PLN', quote='PLN') == Decimal('100')


def test_missing_day_does_not_carry_forward():
    fx = FXRateProvider({(0, 'USD', 'PLN'): '4'})
    with pytest.raises(KeyError):
        fx.get_rate(day=1, base='USD', quote='PLN')


@pytest.mark.parametrize('rate', ['0', '-1', 'NaN', 'Infinity'])
def test_invalid_rates(rate):
    fx = FXRateProvider()
    with pytest.raises(ValueError):
        fx.set_rate(day=0, base='USD', quote='PLN', rate=rate)


def test_daily_revaluation():
    fx = FXRateProvider({(0, 'USD', 'PLN'): '4.00', (1, 'USD', 'PLN'): '3.80'})
    assert fx.convert('100', day=0, base='USD', quote='PLN') == Decimal('400.00')
    assert fx.convert('100', day=1, base='USD', quote='PLN') == Decimal('380.00')
