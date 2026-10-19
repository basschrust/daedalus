import pytest

from marketsim.market import Security, Option, Price
from marketsim.database.connectors.duckdb_storage import Repository




################    test functions         ###################

def test_create_future(stock: Security, repository: Repository, derivatives_config1: dict) -> None:
    assert 1 == 1

###  trading



#### expiration / exercise ###



### reporting


