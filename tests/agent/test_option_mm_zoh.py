from unittest.mock import patch

from marketsim.agent.option_market_maker_zoh import OptionMMZOHAgent


def test_option_mm_handles_empty_order_book(
    agent_option_mm: OptionMMZOHAgent,
):
    option_market = next(iter(agent_option_mm.option_markets.values()))

    # Register the agent, as Simulator.add_agents() does.
    if agent_option_mm.agent_id not in option_market.agents:
        option_market.add_agents([agent_option_mm])

    with (
        patch.object(
            option_market.order_book,
            "get_best_bid",
            return_value=float("-inf"),
        ),
        patch.object(
            option_market.order_book,
            "get_best_ask",
            return_value=float("inf"),
        ),
    ):
        agent_option_mm.take_action(current_time=0)
