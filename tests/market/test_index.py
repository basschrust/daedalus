from marketsim.market import Index, Price

################    test functions         ###################


def test_index_definition(index_genius: Index):
    assert index_genius is not None


def test_weights(index_genius: Index):
    for a_id, component in index_genius.components.items():
        assert component["weight"] > 0, "Weights must be strictly positive"


def test_genius_index(
    genius_index,
    stock_tnt,
    stock_mit,
):
    assert len(genius_index.components) == 2
    assert genius_index.last_traded_price == Price(77.97)


def test_legacy_index_components():
    components = [
        {"security": "stock_a", "weight": 0.6},
        {"security": "stock_b", "weight": 0.4},
    ]

    assert all("base_price" not in c for c in components)


def test_normalized_index_components():
    components = [
        {"security": "stock_a", "weight": 0.6, "base_price": "100.00"},
        {"security": "stock_b", "weight": 0.4, "base_price": "200.00"},
    ]

    assert all("base_price" in c for c in components)
