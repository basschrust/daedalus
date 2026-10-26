"""Unit tests for the proposed marketsim.market.etf.ETF class.

These tests target the ETF(Security) API discussed for Daedalus v1.
They do not require a running database or full simulation.
"""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from marketsim.market.instruments.etf import ETF
from marketsim.market.price import Price


@pytest.fixture
def underlying():
    return SimpleNamespace(
        instrument_class="index",
        last_traded_price=Price(2500),
        currency="PLN",
    )


@pytest.fixture
def etf_factory(underlying, tmp_path):
    """Construct ETF without writing to the repository in Security.__init__."""

    def create(**overrides):
        kwargs = {
            "underlying": underlying,
            "repository": SimpleNamespace(),
            "tracking_ratio": 0.1,
            "name": "Test ETF",
            "short_name": "ETF_TEST",
            "output_dir": tmp_path,
        }
        kwargs.update(overrides)
        with patch.object(ETF, "save"):
            return ETF(**kwargs)

    return create


def test_initialization(etf_factory, underlying):
    etf = etf_factory()
    assert etf.instrument_class == "etf"
    assert etf.underlying is underlying
    assert etf.tracking_ratio == 0.1
    assert etf.currency == "PLN"
    assert etf.market_type == "continuous"
    assert etf.short_name == "ETF_TEST"
    assert etf.reference_price == Price(250)
    assert etf.last_traded_price == Price(250)


@pytest.mark.parametrize(
    ("index_level", "ratio", "expected"),
    [
        (2500, 1.0, "2500.00"),
        (2500, 0.1, "250.00"),
        (1234.56, 0.25, "308.64"),
        (100.05, 0.1, "10.01"),
    ],
)
def test_theoretical_price(etf_factory, underlying, index_level, ratio, expected):
    underlying.last_traded_price = Price(index_level)
    etf = etf_factory(tracking_ratio=ratio)
    assert etf.calculate_theoretical_price() == Price(Decimal(expected))


def test_index_move_updates_theoretical_but_not_market_price(etf_factory, underlying):
    etf = etf_factory()
    etf.last_traded_price = Price(253)
    underlying.last_traded_price = Price(2600)
    assert etf.calculate_theoretical_price() == Price(260)
    assert etf.last_traded_price == Price(253)


@pytest.mark.parametrize(
    ("market_price", "expected_pct"),
    [(253, 1.2), (247, -1.2), (250, 0.0)],
)
def test_premium_and_discount(etf_factory, market_price, expected_pct):
    etf = etf_factory()
    etf.last_traded_price = Price(market_price)
    assert etf.calculate_premium() == pytest.approx(expected_pct)


@pytest.mark.parametrize("ratio", [0, -0.1, -1])
def test_non_positive_tracking_ratio_rejected(etf_factory, ratio):
    with pytest.raises(ValueError, match="tracking_ratio"):
        etf_factory(tracking_ratio=ratio)


def test_non_index_underlying_rejected(etf_factory, underlying):
    underlying.instrument_class = "stock"
    with pytest.raises(ValueError, match="Index"):
        etf_factory()


def test_theoretical_price_reads_current_index_level(etf_factory, underlying):
    etf = etf_factory()
    assert etf.calculate_theoretical_price() == Price(250)
    underlying.last_traded_price = Price(2510)
    assert etf.calculate_theoretical_price() == Price(251)


def test_no_premium_division_by_zero(etf_factory, underlying):
    etf = etf_factory()
    underlying.last_traded_price = Price(0)
    assert etf.calculate_premium() == 0.0
