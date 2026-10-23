from marketsim.market import Security

############  test functions  ##################


def test_create_security(repository):
    security = Security(repository=repository)

    assert security is not None


#### test Start-of-Day


def test_sod(security: Security):
    security.sod()

    assert security.current_day == 0, "time should be reset"
    assert security.eod_status == "open"
    assert list(security.traded_prices.keys()) == [0], (
        "prices should be filled only for time_tick 0"
    )


def test_sod_after_sod(security_with_trades: Security):
    security_with_trades.sod()

    assert (
        security_with_trades.current_day == 0
    )  # TODO: when calling intraday the sod should have no effect
    assert security_with_trades.eod_status == "open"


def test_sod_intraday(security: Security):
    assert 1 == 1


def test_sod_on_abandoned_security(security_abandoned: Security):
    security_abandoned.sod()

    assert security_abandoned.current_day == 0, "Day still 0"
    assert security_abandoned.eod_status == "closed", (
        "Trading not allowed, SoD still holds security in closed state"
    )


### test End-of-Day  ###


def test_eod_no_sod(security: Security):
    security.current_time = 3
    security.current_day = 5
    security.eod()

    assert security.current_time == 3, "Security during  should have the current time unchanged a"
    assert security.current_day == 5, "Security not started day doesn't roll date"
    assert security.eod_status == "closed"


# @pytest.mark.skip(reason="fill orders first")
def test_eod_after_sod(security: Security):
    security.current_day = 3
    security.sod()
    # yet fails as no trades and not filled
    security.eod()

    assert security.current_day == 4, "Security should have the date rolled"
    assert security.current_time == 0, "EoD resets the current time"
    assert security.eod_status == "closed"
