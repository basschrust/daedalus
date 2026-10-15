import pytest

from marketsim.market import Security, Option, Price, Index
from marketsim.database.connectors.duckdb_storage import Repository



################    test functions         ###################


def test_index_definition():
    assert 1 == 1

def test_weights_sum():
    assert 1 == 1

def test_weights_sum_fail():
    assert 1 == 1

def test_index_value():
    assert 1 == 1

def test_index_value_after_stock_change():
    assert 1 == 1
