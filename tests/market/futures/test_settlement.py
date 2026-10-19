import pytest


from marketsim.agent import Agent, NoiseAgent
from marketsim.market import Index, Future, Security


####################################################################
####    test functions
#############################################################



def test_futures_share_underlying(
    genius_index: Index,
    future_short: Future,
    future_long: Future,
):
    assert future_short.underlying is genius_index
    assert future_long.underlying is genius_index
    assert future_short.expiration < future_long.expiration


def test_futures_positions(
    futures_agents_with_positions: dict[str, Agent],
    future_short: Future,
    future_long: Future,
):
    agents = futures_agents_with_positions

    short_id = future_short.asset_id
    long_id = future_long.asset_id

    assert agents["buyer"].futures_position[short_id] == 2
    assert agents["seller"].futures_position[short_id] == -2

    assert agents["buyer"].futures_position[long_id] == 1
    assert agents["hedger"].futures_position[long_id] == -1
