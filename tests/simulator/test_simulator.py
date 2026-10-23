
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from marketsim.simulator.simulator import Simulator


@pytest.fixture
def simulator(tmp_path):
    """Minimal simulator with a real temporary DuckDB repository."""
    sim = Simulator(
        sim_time=3,
        days=2,
        markets={},
        agents={},
        output_dir=tmp_path,
        lob_plot_interval=10,
    )
    yield sim
    sim.repository.connection.close()


def test_initialization(simulator):
    assert simulator.sim_time == 3
    assert simulator.days == 2
    assert simulator.current_time == 0
    assert simulator.day == 0
    assert simulator.markets == {}
    assert simulator.agents == {}


def test_fx_rates_initialized(tmp_path):
    sim = Simulator(
        sim_time=1,
        markets={},
        agents={},
        output_dir=tmp_path,
        fx_rates={
            0: {
                "USD/PLN": 3.80,
                "EUR/PLN": 4.30,
            },
        },
    )

    try:
        assert sim.fx_rate_provider.get_rate(
            day=0, base="USD", quote="PLN"
        ) == Decimal("3.8")

        assert sim.fx_rate_provider.get_rate(
            day=0, base="EUR", quote="USD"
        ) == Decimal("4.30") / Decimal("3.80")
    finally:
        sim.repository.connection.close()


def test_sod_resets_time_and_initializes_markets_first(simulator):
    events = []

    market = Mock()
    market.sod.side_effect = lambda: events.append("market_sod")

    agent = Mock()
    agent.sod.side_effect = lambda: events.append("agent_sod")

    simulator.markets = {1: market}
    simulator.agents = {2: agent}
    simulator.current_time = 99

    simulator.sod(day=0)

    assert simulator.current_time == 0
    assert events == ["market_sod", "agent_sod"]


def test_eod_processes_markets_before_agents(simulator):
    events = []

    market = Mock()
    market.eod.side_effect = lambda: events.append("market_eod")

    agent = Mock()
    agent.eod.side_effect = lambda: events.append("agent_eod")

    simulator.markets = {1: market}
    simulator.agents = {2: agent}

    simulator.eod(day=0)

    assert events == ["market_eod", "agent_eod"]


def test_step_executes_agents_before_markets_and_valuation(simulator):
    events = []

    agent = Mock()
    agent.take_action.side_effect = (
        lambda **kwargs: events.append("agent_action")
    )
    agent.record_valuation.side_effect = (
        lambda **kwargs: events.append("valuation")
    )

    market = Mock()
    market.status = "active"
    market.instrument_class = "stock"
    market.order_book = Mock()

    market.order_book.buy_matched.heap = []
    market.order_book.sell_matched.heap = []

    market.order_book.buy_unmatched.peek.return_value = None
    market.order_book.sell_unmatched.peek.return_value = None

    market.step.side_effect = (
        lambda **kwargs: events.append("market_step") or []
    )

    simulator.agents = {1: agent}
    simulator.markets = {2: market}

    simulator.step()

    assert events == [
        "agent_action",
        "market_step",
        "valuation",
    ]

    agent.take_action.assert_called_once_with(current_time=0)
    market.step.assert_called_once_with(current_time=0)
    agent.record_valuation.assert_called_once_with(current_time=0)
    assert simulator.current_time == 1


def test_step_skips_inactive_markets(simulator):
    market = Mock()
    market.status = "expired"

    simulator.markets = {1: market}

    simulator.step()

    market.step.assert_not_called()
    assert simulator.current_time == 1


def test_run_calls_lifecycle_correct_number_of_times(simulator):
    simulator.sod = Mock()
    simulator.step = Mock()
    simulator.eod = Mock()
    simulator.end_sim = Mock()
    simulator.show_progress_bar = Mock()

    simulator.run()

    assert simulator.sod.call_args_list == [
        call(day=0),
        call(day=1),
    ]

    assert simulator.step.call_count == 6

    assert simulator.eod.call_args_list == [
        call(day=0),
        call(day=1),
    ]

    simulator.end_sim.assert_called_once()


def test_add_agents_registers_agent_and_injects_fx(simulator):
    market = Mock()

    agent = Mock()
    agent.get_id.return_value = 42
    agent.configuration = {
        "base_currency": "usd",
        "initial_cash": {
            "USD": 10000,
            "PLN": 5000,
        },
    }
    agent.markets = {7: market}

    simulator.add_agents([agent])

    assert simulator.agents[42] is agent
    assert agent.fx_rate_provider is simulator.fx_rate_provider
    assert agent.base_currency == "USD"

    assert agent.credit_cash.call_count == 2
    assert {
        c.kwargs["currency"]
        for c in agent.credit_cash.call_args_list
    } == {"USD", "PLN"}

    market.add_agents.assert_called_once_with([agent])


def test_create_agents_resolves_selected_markets(simulator):
    market_a = Mock()
    market_b = Mock()

    simulator.markets = {
        10: market_a,
        20: market_b,
    }
    simulator.market_map = {
        "stock_a": 10,
        "stock_b": 20,
    }
    simulator.add_agent_group = Mock()

    simulator.create_agents(
        agent_group={
            "agent_class": "NoiseAgent",
            "number": 1,
            "markets": ["stock_b"],
        },
        group_name="test_agents",
    )

    kwargs = simulator.add_agent_group.call_args.kwargs

    assert kwargs["markets"] == [market_b]
    assert kwargs["group_name"] == "test_agents"


def test_create_agents_all_markets(simulator):
    market_a = Mock()
    market_b = Mock()

    simulator.markets = {10: market_a, 20: market_b}
    simulator.add_agent_group = Mock()

    simulator.create_agents(
        agent_group={
            "agent_class": "NoiseAgent",
            "number": 1,
            "markets": "ALL",
        },
        group_name="global_agents",
    )

    kwargs = simulator.add_agent_group.call_args.kwargs

    assert kwargs["markets"] == [market_a, market_b]
