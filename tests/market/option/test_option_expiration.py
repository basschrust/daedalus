import pytest

from marketsim.market.instruments.option import parse_expiration


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        ("0", 0),
        ("0D", 0),
        (20, 20),
        ("1W", 5),
        ("1M", 21),
        ("3M", 63),
        ("1Y", 248),
        ("2Y", 496),
    ],
)
def test_parse_expiration(value, expected):
    assert parse_expiration(value) == expected


@pytest.mark.parametrize(
    "value",
    ["unknown", "1Q", -1, "-1D", None, True],
)
def test_invalid_expiration(value):
    with pytest.raises(ValueError):
        parse_expiration(value)
