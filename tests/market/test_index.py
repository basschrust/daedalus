from marketsim.market import Index, Price

################    test functions         ###################


def test_index_definition(index_genius: Index):
    assert index_genius is not None


def test_weights(index_genius: Index):
    for a_id, component in index_genius.components.items():
        assert component["weight"] > 0, "Weights must be strictly positive"


def test_weights_sum_fail():
    assert 1 == 1


def test_index_value():
    assert 1 == 1


def test_index_value_after_stock_change():
    assert 1 == 1


def test_genius_index(
    genius_index,
    stock_tnt,
    stock_mit,
):
    assert len(genius_index.components) == 2
    assert genius_index.last_traded_price == Price(77.97)
