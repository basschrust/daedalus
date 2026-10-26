from dataclasses import FrozenInstanceError, asdict, replace
from datetime import date

import pytest

from marketsim.market.knowledge import HistoricalMarketData, MarketKnowledge


@pytest.fixture
def historical():
    return HistoricalMarketData(
        symbol="PKO.WA",
        as_of=date(2026, 10, 23),
        reference_price=109.54,
        daily_mean_return=0.0015490790089025827,
        daily_volatility=0.016532928071032425,
        median_daily_volume=2367521.5,
        observations=250,
    )


def test_historical_fields(historical):
    assert historical.symbol == "PKO.WA"
    assert historical.as_of == date(2026, 10, 23)
    assert historical.reference_price == 109.54
    assert historical.daily_mean_return == pytest.approx(0.0015490790089025827)
    assert historical.daily_volatility == pytest.approx(0.016532928071032425)
    assert historical.median_daily_volume == 2367521.5
    assert historical.observations == 250


@pytest.mark.parametrize("field,value", [
    ("symbol", "KGH.WA"),
    ("as_of", date(2026, 10, 22)),
    ("reference_price", 200.0),
    ("daily_mean_return", -0.01),
    ("daily_volatility", 0.04),
    ("median_daily_volume", 100_000.0),
    ("observations", 10),
])
def test_historical_immutable(historical, field, value):
    with pytest.raises(FrozenInstanceError):
        setattr(historical, field, value)


def test_historical_replace(historical):
    changed = replace(historical, reference_price=110.0)
    assert changed.reference_price == 110.0
    assert historical.reference_price == 109.54
    assert changed != historical


def test_historical_equality_and_serialization(historical):
    reconstructed = HistoricalMarketData(**asdict(historical))
    assert reconstructed == historical
    assert hash(reconstructed) == hash(historical)


def test_knowledge_defaults(historical):
    knowledge = MarketKnowledge(historical)
    assert knowledge.historical is historical
    assert knowledge.confidence == 1.0


@pytest.mark.parametrize("confidence", [0.0, 0.25, 0.75, 1.0])
def test_confidence_stored(historical, confidence):
    assert MarketKnowledge(historical, confidence).confidence == confidence


@pytest.mark.parametrize("horizon", [0, 0.5, 1, 2, 5, 10])
def test_expected_price_compounds(historical, horizon):
    knowledge = MarketKnowledge(historical)
    expected = historical.reference_price * (1 + historical.daily_mean_return) ** horizon
    assert knowledge.expected_price(horizon) == pytest.approx(expected)


def test_default_price_horizon_one_day(historical):
    knowledge = MarketKnowledge(historical)
    assert knowledge.expected_price() == pytest.approx(knowledge.expected_price(1))


@pytest.mark.parametrize("daily_return", [-0.03, 0.0, 0.04])
def test_expected_price_with_return_signs(historical, daily_return):
    snapshot = replace(historical, daily_mean_return=daily_return)
    assert MarketKnowledge(snapshot).expected_price(3) == pytest.approx(
        snapshot.reference_price * (1 + daily_return) ** 3
    )


def test_expected_price_zero_reference(historical):
    knowledge = MarketKnowledge(replace(historical, reference_price=0))
    assert knowledge.expected_price(5) == 0


def test_volatility_and_volume(historical):
    knowledge = MarketKnowledge(historical)
    assert knowledge.expected_volatility() == historical.daily_volatility
    assert knowledge.expected_volume() == historical.median_daily_volume


def test_security_specific_knowledge(historical):
    kgh = replace(
        historical, symbol="KGH.WA", reference_price=293.2,
        daily_volatility=0.03452413086969201, median_daily_volume=790501.5,
    )
    pko_knowledge = MarketKnowledge(historical)
    kgh_knowledge = MarketKnowledge(kgh)
    assert pko_knowledge.expected_price(0) == 109.54
    assert kgh_knowledge.expected_price(0) == 293.2
    assert pko_knowledge.expected_volatility() < kgh_knowledge.expected_volatility()
    assert pko_knowledge.expected_volume() > kgh_knowledge.expected_volume()


def test_shared_history_independent_agent_confidence(historical):
    informed = MarketKnowledge(historical, confidence=0.9)
    uninformed = MarketKnowledge(historical, confidence=0.1)
    informed.confidence = 0.5
    assert informed.historical is uninformed.historical
    assert uninformed.confidence == 0.1


def test_confidence_currently_does_not_affect_forecast(historical):
    confident = MarketKnowledge(historical, confidence=1.0)
    uncertain = MarketKnowledge(historical, confidence=0.0)
    assert confident.expected_price(5) == pytest.approx(uncertain.expected_price(5))


def test_replaced_snapshot_does_not_modify_existing_knowledge(historical):
    knowledge = MarketKnowledge(historical)
    newer = replace(historical, reference_price=120.0)
    assert knowledge.expected_price(0) == 109.54
    assert MarketKnowledge(newer).expected_price(0) == 120.0
