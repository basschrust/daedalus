from decimal import Decimal

import pytest

from marketsim.market import Price

############    test functions  ###############


def test_price_create():
    price = Price(98.00)

    assert price == Decimal("98.00")

def test_price_update():
    price = Price(98.00)

    price += Price(3.00)

    assert price == Price(101.00)


def test_price_multiplication():
    price = Price(98.00)
    price2 = 2 * price

    assert price2 == Price(196.00)


def test_price_is_rounded_to_tick():
    assert Price("100.004") == Price("100.00")
    assert Price("100.005") == Price("100.01")
    assert Price("100.006") == Price("100.01")


def test_price_multiplication_by_int():
    price = Price("100.00")

    result = price * 3

    assert result == Price("300.00")
    assert isinstance(result, Price)


def test_price_multiplication_by_float():
    price = Price("100.00")

    result = price * 1.5

    assert result == Price("150.00")
    assert isinstance(result, Price)


def test_float_multiplication_from_left_hand_side():
    price = Price("100.00")

    result = 1.5 * price

    assert result == Price("150.00")
    assert isinstance(result, Price)


def test_price_division_by_int():
    price = Price("100.00")

    result = price / 4

    assert result == Decimal("25.00")


def test_price_addition():
    price1 = Price("100.25")
    price2 = Price("50.50")

    result = price1 + price2

    assert result == Decimal("150.75")


def test_price_addition_with_decimal():
    price = Price("100.25")

    result = price + Decimal("50.50")

    assert result == Decimal("150.75")


def test_price_must_be_finite():
    with pytest.raises(ValueError, match="Price must be finite"):
        Price(float("inf"))

    with pytest.raises(ValueError, match="Price must be finite"):
        Price(float("-inf"))

    with pytest.raises(ValueError, match="Price must be finite"):
        Price("NaN")


def test_price_must_not_be_unreasonably_large():
    with pytest.raises(ValueError, match="Unreasonable price"):
        Price("100000000.01")

    with pytest.raises(ValueError, match="Unreasonable price"):
        Price("-100000000.01")