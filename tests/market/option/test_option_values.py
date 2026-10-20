import pytest

from marketsim.market import Price, Security, Option

def test_option_underlying(
    index_genius,
    option_call_short,
    option_put_short,
):
    assert option_call_short.underlying is index_genius
    assert option_put_short.underlying is index_genius


@pytest.mark.parametrize(
    "option_type,strike,index_price,expected",
    [
        ("CALL", 75, 80, 5),
        ("CALL", 75, 70, 0),
        ("PUT", 75, 70, 5),
        ("PUT", 75, 80, 0),
        ("CALL", 75, 75, 0),
        ("PUT", 75, 75, 0),
    ],
)
def test_option_intrinsic_value(
    option_type, strike, index_price, expected
):
    if option_type == "CALL":
        intrinsic = max(index_price - strike, 0)
    else:
        intrinsic = max(strike - index_price, 0)

    assert intrinsic == expected


@pytest.mark.parametrize(
    "option_fixture,underlying_price,expected_premium",
    [
        ("option_call_short", 90, 15),
        ("option_call_short", 70, 0),
        ("option_put_short", 60, 15),
        ("option_put_short", 90, 0),
        ("option_call_short", 75, 0),
        ("option_put_short", 75, 0),
    ],
)
def test_option_expiration_payoff(
    request,
    index_genius,
    option_fixture,
    underlying_price,
    expected_premium,
):
    option = request.getfixturevalue(option_fixture)

    index_genius.last_traded_price = Price(underlying_price)
    option.current_day = option.expiration

    premium = option.exercise()

    assert premium == Price(expected_premium)
    assert option.status == "expired"
