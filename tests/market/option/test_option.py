import pytest

from marketsim.market import Security, Option, Price
from marketsim.database.connectors.duckdb_storage import Repository




################    test functions         ###################

def test_create_option(stock: Security, repository: Repository, derivatives_config1: dict) -> None:
    option1 = Option(underlying=stock, repository=repository, derivatives_config=derivatives_config1)

    assert option1.underlying == stock
    assert option1 is not None

def test_option_call_itm(stock_tnt: Security,
                                  repository: Repository,
                                  option_call_before_expiration_itm: Option) -> None:
    option = option_call_before_expiration_itm

    assert option is not None
    # TODO: check BS price?
    assert option.get_theoretical_price() > 0, "CALL ITM has positive price"

###  trading



#### expiration / exercise ###

# ITM

def test_exercise_option_call_itm_before(stock_tnt: Security,
                                  repository: Repository,
                                  option_call_before_expiration_itm: Option) -> None:
    option = option_call_before_expiration_itm
    premium = option.exercise()

    assert option is not None
    assert option.get_theoretical_price() > 0, "CALL ITM has positive price"
    assert premium == 0, "Premium before expiration should be 0"
    # TODO: agents cashflows check? - another test


def test_exercise_option_call_itm_exp_day(stock_tnt: Security,
                                  repository: Repository,
                                  option_call_at_expiration_itm: Option) -> None:
    option = option_call_at_expiration_itm
    premium = option.exercise()

    assert option is not None
    assert option.get_theoretical_price() > 0, "CALL ITM has positive price"
    assert premium == stock_tnt.last_traded_price - option.strike, "Premium at expiration should S-K"
    # TODO:


def test_exercise_option_put_itm_before(stock_tnt: Security,
                                  repository: Repository,
                                  option_put_before_expiration_itm: Option) -> None:
    option = option_put_before_expiration_itm
    premium = option.exercise()

    assert option is not None
    assert option.get_theoretical_price() > 0, "CALL ITM has positive price"
    assert premium == 0, "Premium before expiration should be 0"
    # TODO: agents cashflows check? - another test


def test_exercise_option_put_itm_exp_day(stock_tnt: Security,
                                  repository: Repository,
                                  option_put_at_expiration_itm: Option) -> None:
    option = option_put_at_expiration_itm
    stock_tnt.last_traded_price = Price(67)
    premium = option.exercise()

    assert option is not None
    assert premium == - stock_tnt.last_traded_price + option.strike, "Premium at expiration is K-S"

# ATM

def test_exercise_option_call_atm_before(stock_tnt: Security,
                                  repository: Repository,
                                  option_call_before_expiration_atm: Option) -> None:
    option = option_call_before_expiration_atm
    premium = option.exercise()

    assert option is not None
    assert option.get_theoretical_price() > 0, "CALL ATM has positive price"
    assert premium == 0, "Premium before expiration should be 0"


def test_exercise_option_call_atm_exp_day(stock_tnt: Security,
                                  repository: Repository,
                                  option_call_at_expiration_atm: Option) -> None:
    option = option_call_at_expiration_atm
    premium = option.exercise()

    assert option is not None
    assert premium == 0, "Premium of ATM at expiration should be 0"


def test_exercise_option_put_atm_before(stock_tnt: Security,
                                  repository: Repository,
                                  option_put_before_expiration_atm: Option) -> None:
    option = option_put_before_expiration_atm
    premium = option.exercise()

    assert option is not None
    assert option.get_theoretical_price() > 0, "CALL ITM has positive price"
    assert premium == 0, "Premium before expiration should be 0"
    # TODO: agents cashflows check? - another test


def test_exercise_option_put_atm_exp_day(stock_tnt: Security,
                                  repository: Repository,
                                  option_put_at_expiration_atm: Option) -> None:
    option = option_put_at_expiration_atm
    premium = option.exercise()

    assert option is not None
    assert premium == 0, "Premium of ATM at expiration is 0"


# OTM

def test_exercise_option_call_otm_before(stock_tnt: Security,
                                  repository: Repository,
                                  option_call_before_expiration_otm: Option) -> None:
    option = option_call_before_expiration_otm
    premium = option.exercise()

    assert option is not None
    assert option.get_theoretical_price() > 0, "CALL has positive price even if OTM"
    assert premium == 0, "Premium before expiration should be 0"
    # TODO: agents cashflows check? - another test


def test_exercise_option_call_otm_exp_day(stock_tnt: Security,
                                  repository: Repository,
                                  option_call_at_expiration_otm: Option) -> None:
    option = option_call_at_expiration_otm
    premium = option.exercise()

    assert option is not None
    assert premium == 0, "Premium of OTM at expiration should be 0"


def test_exercise_option_put_otm_before(stock_tnt: Security,
                                  repository: Repository,
                                  option_put_before_expiration_otm: Option) -> None:
    option = option_put_before_expiration_otm
    premium = option.exercise()

    assert option is not None
    assert option.get_theoretical_price() > 0, "CALL ITM has positive price"
    assert premium == 0, "Premium before expiration should be 0"
    # TODO: agents cashflows check? - another test


def test_exercise_option_put_otm_exp_day(stock_tnt: Security,
                                  repository: Repository,
                                  option_put_at_expiration_otm: Option) -> None:
    option = option_put_at_expiration_otm
    premium = option.exercise()

    assert option is not None
    assert premium == 0, "Premium of OTM at expiration should be 0"



### reporting


