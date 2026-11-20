"""Method parity and daily-calibration regression tests (no cloud access)."""

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from marketsim.market.valuation_libs.BlackScholes import BSCall, BSPut
from marketsim.risk.eod_var import EodVaR


def calibration(count=80):
    # Independent-ish but empirically correlated stock and benchmark returns.
    x = np.arange(count, dtype=float)
    a = 101.0 * np.exp(0.001 * x + 0.015 * np.sin(x * 0.53))
    b = 4200.0 * np.exp(0.0006 * x + 0.012 * np.cos(x * 0.35))
    days = [date(2026, 4, 1) + timedelta(days=i) for i in range(count)]
    return SimpleNamespace(
        latest_calibration_id="SIM-20261118-daily",
        series={"AAA": dict(zip(days, a)), "WIG20": dict(zip(days, b))},
        lookback_sessions=count,
    )


def market(kind, symbol, price, **attrs):
    return SimpleNamespace(
        instrument_class=kind, short_name=symbol, last_traded_price=price, currency="PLN", **attrs
    )


def agent(markets=None, positions=None, futures_positions=None):
    return SimpleNamespace(
        agent_id=9,
        base_currency="PLN",
        portfolio_value=100000.0,
        position=positions or {},
        futures_position=futures_positions or {},
        markets=markets or {},
        fx_rate_provider=SimpleNamespace(convert=lambda amount, **kw: amount),
    )


def engine(config=None):
    return EodVaR(
        calibration(),
        config
        or {
            "confidence_levels": [0.95, 0.99],
            "horizons_days": [1, 5],
            "montecarlo_scenarios": 1500,
        },
        greeks_config={"short_rates": {"PLN": 0.04}, "dividend_yields": {"WIG20": 0.0}},
    )


def test_all_three_methods_emit_all_confidences_and_horizons():
    stock = market("stock", "AAA", 105)
    model = engine()
    rows = model.rows_for_agent(agent({1: stock}, {1: -50}), 2)
    for method in ("parametric", "historical", "montecarlo"):
        assert len(rows[method]) == 4
        assert {(r["confidence_level"], r["horizon_days"]) for r in rows[method]} == {
            (0.95, 1),
            (0.95, 5),
            (0.99, 1),
            (0.99, 5),
        }
        assert all(r["calibration_id"] == model.calibration_id for r in rows[method])
        assert all(np.isfinite(r["var_amount"]) and r["var_amount"] >= 0 for r in rows[method])
        assert all(r["covered_exposure"] == 5250 for r in rows[method])
        for horizon in [1, 5]:
            pair = [r for r in rows[method] if r["horizon_days"] == horizon]
            assert pair[1]["var_amount"] >= pair[0]["var_amount"]


def test_all_cash_has_zero_risk_in_all_methods():
    rows = engine().rows_for_agent(agent(), 0)
    assert all(r["var_amount"] == 0 for records in rows.values() for r in records)


def test_futures_and_european_option_revalue_against_index():
    index = market("index", "WIG20", 4200.0)
    future = market("future", "FUT", 4220.0, underlying=index, multiplier=20, expiration=12)
    option = market(
        "option",
        "OPT",
        120.0,
        underlying=index,
        multiplier=10,
        strike=4200.0,
        expiration=12,
        option_side="CALL",
        option_type="European",
    )
    rows = engine().rows_for_agent(agent({1: future, 2: option}, {2: 4}, {1: -3}), 1)
    assert all(r["covered_exposure"] > 200000 for records in rows.values() for r in records)
    assert all(r["excluded_exposure"] == 0 for records in rows.values() for r in records)
    assert all(np.isfinite(r["var_amount"]) for records in rows.values() for r in records)


def test_early_expiry_uses_path_at_expiration():
    index = market("index", "WIG20", 4200.0)
    option = market(
        "option",
        "OPT",
        110.0,
        underlying=index,
        multiplier=10,
        strike=4200.0,
        expiration=2,
        option_side="PUT",
        option_type="European",
    )
    future = market("future", "FUT", 4200.0, underlying=index, multiplier=20, expiration=2)
    model = engine(
        {"confidence_levels": [0.99], "horizons_days": [5], "montecarlo_scenarios": 1500}
    )
    portfolio, _, _ = model._portfolio(
        agent({1: future, 2: option}, {2: 2}, {1: 1}), day=0, horizon=5
    )
    path = {i: np.zeros((1, 2)) for i in range(1, 6)}
    path[1][0, model.index["WIG20"]] = 0.01
    path[2][0, model.index["WIG20"]] = 0.02
    path[3][0, model.index["WIG20"]] = 0.8
    path[4][0, model.index["WIG20"]] = -0.5
    path[5][0, model.index["WIG20"]] = 1.2
    calculated = model._pnl(portfolio, path, 5)
    assert calculated.shape == (1,)
    # Shock after expiry must have no effect on the contracts' value.
    path[3][0, model.index["WIG20"]] = -10
    path[5][0, model.index["WIG20"]] = -10
    assert model._pnl(portfolio, path, 5)[0] == pytest.approx(calculated[0])


def test_option_price_at_expiration_and_zero_volatility():
    for model, expected in ((BSCall, 10), (BSPut, 0)):
        assert model(S=110, K=100, r=0, volatility=0, Time=0)["price"] == pytest.approx(expected)
    assert BSCall(S=110, K=100, r=0, volatility=0, Time=1)["price"] == pytest.approx(10)


def test_option_delta_comes_from_black_scholes_model():
    model = engine()
    index = market("index", "WIG20", 4200.0)
    option = market(
        "option",
        "OPT",
        120,
        underlying=index,
        multiplier=10,
        strike=4300,
        expiration=12,
        option_side="PUT",
        option_type="European",
    )
    positions, _, _ = model._portfolio(agent({1: option}, {1: 4}), day=0, horizon=5)
    expected = BSPut(
        S=4200.0,
        K=4300.0,
        r=0.04,
        volatility=model.sigmas[model.index["WIG20"]],
        Time=12 / 252,
    )["delta"]
    assert positions[0].exposure == pytest.approx(4 * 10 * 4200 * expected)
    assert expected < 0


def test_invalid_configuration_and_missing_dividend_fail_explicitly():
    with pytest.raises(ValueError, match="confidence"):
        engine({"confidence_levels": [1.0], "montecarlo_scenarios": 2000})
    index = market("index", "WIG20", 4200.0)
    opt = market(
        "option",
        "OPT",
        50.0,
        underlying=index,
        multiplier=10,
        strike=4200,
        expiration=10,
        option_side="CALL",
        option_type="European",
    )
    model = EodVaR(
        calibration(), {"montecarlo_scenarios": 1500}, greeks_config={"short_rates": {"PLN": 0.04}}
    )
    with pytest.raises(ValueError, match="zero-dividend"):
        model.rows_for_agent(agent({1: opt}, {1: 1}), 0)


def test_calibration_refresh_changes_statistics_and_all_method_ids():
    from marketsim.risk.eod_var import write_eod_var

    c = calibration()
    captured = {}
    repo = SimpleNamespace(
        save_eod_agent_var_parametric=lambda rows: captured.setdefault("parametric", rows),
        save_eod_agent_var_historical=lambda rows: captured.setdefault("historical", rows),
        save_eod_agent_var_montecarlo=lambda rows: captured.setdefault("montecarlo", rows),
    )
    stock = market("stock", "AAA", 105)
    a = agent({1: stock}, {1: 20})
    sim = SimpleNamespace(
        agents={9: a}, repository=repo, risk_greeks={"short_rates": {"PLN": 0.04}}
    )
    config = {
        "confidence_levels": [0.95, 0.99],
        "horizons_days": [1, 5],
        "montecarlo_scenarios": 1500,
    }
    old_sigma = EodVaR(c, config).covariance[0, 0]
    c.latest_calibration_id = "SIM-new-day"
    last = max(c.series["AAA"])
    c.series["AAA"][last] *= 1.15
    write_eod_var(sim, 7, c, config)
    new_sigma = EodVaR(c, config).covariance[0, 0]
    assert new_sigma != pytest.approx(old_sigma)
    assert set(captured) == {"historical", "parametric", "montecarlo"}
    assert all(len(rows) == 4 for rows in captured.values())
    assert all(
        row["day"] == 7 and row["calibration_id"] == "SIM-new-day"
        for records in captured.values()
        for row in records
    )
    assert {
        (r["agent_id"], r["day"], r["confidence_level"], r["horizon_days"])
        for r in captured["parametric"]
    } == {
        (r["agent_id"], r["day"], r["confidence_level"], r["horizon_days"])
        for r in captured["historical"]
    }


def test_zero_market_premium_is_valid_for_option_risk():
    index = market("index", "WIG20", 4200.0)
    option = market(
        "option",
        "OTM",
        0.0,
        underlying=index,
        multiplier=10,
        strike=6000.0,
        expiration=10,
        option_side="CALL",
        option_type="European",
    )
    rows = engine().rows_for_agent(agent({1: option}, {1: 2}), 0)
    assert all(r["excluded_exposure"] == 0 for records in rows.values() for r in records)


@pytest.mark.parametrize("side", ["CALL", "PUT"])
@pytest.mark.parametrize("maturity,horizon", [(1, 1), (2, 5)])
def test_parametric_var_handles_scalar_price_at_or_after_option_expiry(side, maturity, horizon):
    """Black-Scholes returns Price at expiry, float before expiry."""
    index = market("index", "WIG20", 4200.0)
    option = market(
        "option",
        "EXPIRING_OPTION",
        115.0,
        underlying=index,
        multiplier=10,
        strike=4100.0 if side == "CALL" else 4300.0,
        expiration=maturity,
        option_side=side,
        option_type="European",
    )
    model = engine(
        {
            "confidence_levels": [0.95, 0.99],
            "horizons_days": [horizon],
            "montecarlo_scenarios": 1500,
        }
    )
    rows = model.rows_for_agent(agent({1: option}, {1: 2}), day=0)
    assert set(rows) == {"parametric", "historical", "montecarlo"}
    assert all(
        len(method_rows) == 2
        and all(np.isfinite(row["var_amount"]) and row["var_amount"] >= 0 for row in method_rows)
        for method_rows in rows.values()
    )


def test_rejects_fractional_horizon_instead_of_silent_truncation():
    with pytest.raises(ValueError, match="must contain integers"):
        engine({"horizons_days": [1.5], "montecarlo_scenarios": 1500})
