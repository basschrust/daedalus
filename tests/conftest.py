from pathlib import Path

import pytest

from marketsim.agent import Agent, HBLAgent, MMZOHAgent, NoiseAgent
from marketsim.database.connectors.duckdb_storage import Repository
from marketsim.fourheap.order import Order
from marketsim.market import FXRateProvider, Index, Option, Price, Security
from marketsim.market import Price, Security, Option, Index, FXRateProvider
from marketsim.agent import Agent, MMZOHAgent, NoiseAgent, HBLAgent


BUY = 1
SELL = -1

tmp_path = Path("tmp")

@pytest.fixture
def repository(tmp_path: Path):
    repo = Repository(output_dir=tmp_path)
    yield repo
    repo.connection.close()


@pytest.fixture
def fx_rate_provider():
    fx = FXRateProvider()

    # Synthetic rates for deterministic tests
    fx.set_rate(day=0, base="USD", quote="PLN", rate="3.80")
    fx.set_rate(day=0, base="EUR", quote="PLN", rate="4.25")

    return fx

@pytest.fixture
def order_factory():
    def _create_order(
        price:int=100,
        order_type:int=BUY,
        quantity:int=10,
        agent_id:int=1,
        time:int=1,
        asset_id:int=1,
        valid_until:int|None=None,
    ):
        return Order(
            price=Price(price),
            order_type=order_type,
            quantity=quantity,
            agent_id=agent_id,
            time=time,
            asset_id=asset_id,
            valid_until=valid_until,
        )

    return _create_order


@pytest.fixture
def buy_order(order_factory):
    return order_factory()

@pytest.fixture
def buy_order_99(order_factory):
    return order_factory(price=99, order_type=BUY)

@pytest.fixture
def buy_order_101(order_factory):
    return order_factory(price=101, order_type=BUY)

@pytest.fixture
def sell_order(order_factory):
    return order_factory(order_type=SELL)

@pytest.fixture
def sell_order_99(order_factory):
    return order_factory(price=99, order_type=SELL)

@pytest.fixture
def sell_order_101(order_factory):
    return order_factory(price=101, order_type=SELL)


@pytest.fixture
def stock(repository: Repository):
    return Security(reference_price=Price(95.00), name="stock1", repository=repository,
                    market_type="continuous", instrument_class="stock")

@pytest.fixture
def stock_tnt(repository: Repository):
    return Security(reference_price=Price(167.00), name="stock TNT", repository=repository,
                    market_type="continuous", instrument_class="stock", short_name="TNT")

@pytest.fixture
def stock_mit(repository: Repository):
    return Security(reference_price=Price(18.61), name="stock MIT", repository=repository,
                    market_type="continuous", instrument_class="stock", short_name="MIT")


@pytest.fixture
def security(repository: Repository):
    return Security(reference_price=Price(101.00), name="stock1", repository=repository,
                    market_type="continuous", instrument_class="stock")

@pytest.fixture
def security_with_trades(repository: Repository):
    sec = Security(reference_price=Price(101.00), name="stock1", repository=repository,
                    market_type="continuous", instrument_class="stock")
    # TODO: add trade
    return sec

@pytest.fixture
def security_after_sod(repository: Repository):
    sec = Security(reference_price=Price(101.00), name="stock1", repository=repository,
                    market_type="continuous", instrument_class="stock")
    sec.sod()
    return sec

@pytest.fixture
def security_abandoned(repository: Repository, security: Security):
    # an expired option or other security that trading is not allowed any more
    security.status = "abandoned"
    return security

### derivatives:

@pytest.fixture
def option_config_call() -> dict:
    conf = {
        "strike": Price(90.00),
        "option_side": "CALL",
        "option_type": "European",
        "expiration": 10,
            }
    return conf

@pytest.fixture
def option_config_put() -> dict:
    conf = {
        "strike": Price(90.00),
        "option_side": "PUT",
        "option_type": "European",
        "expiration": 10,
            }
    return conf


@pytest.fixture
def option_tnt_call(repository: Repository, stock_tnt: Security, option_config_call: dict):
    option = Option(repository=repository, underlying=stock_tnt, derivatives_config=option_config_call)
    return option

@pytest.fixture
def option_tnt_put(repository: Repository, stock_tnt: Security, option_config_put: dict):
    option = Option(repository=repository, underlying=stock_tnt, derivatives_config=option_config_put)
    return option


# more customized options:

@pytest.fixture
def config_tnt_call(stock: Security):
    config1 = {
                "strike": Price(90),
                "expiration": 30,
                "option_side": "CALL",
                "option_type": "European",
                }
    return config1

@pytest.fixture
def config_tnt_put(stock: Security):
    config1 = {
                "strike": Price(90),
                "expiration": 30,
                "option_side": "PUT",
                "option_type": "European",
                }
    return config1



@pytest.fixture
def option_call_before_expiration_itm(stock_tnt: Security,
                                      config_tnt_call: dict,
                                      repository: Repository) -> Option:
    opt = Option(derivatives_config=config_tnt_call,
                 short_name="CALL before expiration ITM",
                 repository=repository,
                 underlying=stock_tnt)
    opt.sod()
    return opt

@pytest.fixture
def option_call_at_expiration_itm(stock_tnt: Security,
                                      config_tnt_call: dict,
                                      repository: Repository) -> Option:
    config_tnt_call["expiration"] = 0
    opt = Option(derivatives_config=config_tnt_call,
                 short_name="CALL at expiration ITM",
                 repository=repository,
                 underlying=stock_tnt)
    opt.sod()
    return opt


@pytest.fixture
def option_put_before_expiration_itm(stock_tnt: Security,
                                      config_tnt_put: dict,
                                      repository: Repository) -> Option:
    opt = Option(derivatives_config=config_tnt_put,
                 short_name="PUT before expiration ITM",
                 repository=repository,
                 underlying=stock_tnt)
    opt.sod()
    return opt

@pytest.fixture
def option_put_at_expiration_itm(stock_tnt: Security,
                                      config_tnt_put: dict,
                                      repository: Repository) -> Option:
    config_tnt_put["expiration"] = 0
    config_tnt_put["strike"] = Price(170.00)
    opt = Option(derivatives_config=config_tnt_put,
                 short_name="PUT at expiration ITM K: 170",
                 repository=repository,
                 underlying=stock_tnt)
    opt.sod()
    return opt


@pytest.fixture
def option_call_before_expiration_atm(stock_tnt: Security,
                                      config_tnt_call: dict,
                                      repository: Repository) -> Option:
    config_tnt_call["strike"] = Price(167)
    opt = Option(derivatives_config=config_tnt_call,
                 short_name="CALL before expiration ITM",
                 repository=repository,
                 underlying=stock_tnt)
    opt.sod()
    return opt

@pytest.fixture
def option_call_at_expiration_atm(stock_tnt: Security,
                                      config_tnt_call: dict,
                                      repository: Repository) -> Option:
    config_tnt_call["expiration"] = 0
    config_tnt_call["strike"] = Price(167)
    opt = Option(derivatives_config=config_tnt_call,
                 short_name="CALL at expiration ITM",
                 repository=repository,
                 underlying=stock_tnt)
    opt.sod()
    return opt


@pytest.fixture
def option_put_before_expiration_atm(stock_tnt: Security,
                                      config_tnt_put: dict,
                                      repository: Repository) -> Option:
    config_tnt_put["strike"] = Price(167)
    opt = Option(derivatives_config=config_tnt_put,
                 short_name="PUT before expiration ATM",
                 repository=repository,
                 underlying=stock_tnt)
    opt.sod()
    return opt

@pytest.fixture
def option_put_at_expiration_atm(stock_tnt: Security,
                                      config_tnt_put: dict,
                                      repository: Repository) -> Option:
    config_tnt_put["expiration"] = 0
    config_tnt_put["strike"] = Price(167)
    opt = Option(derivatives_config=config_tnt_put,
                 short_name="PUT at expiration ATM",
                 repository=repository,
                 underlying=stock_tnt)
    opt.sod()
    return opt


@pytest.fixture
def option_call_before_expiration_otm(stock_tnt: Security,
                                      config_tnt_call: dict,
                                      repository: Repository) -> Option:
    config_tnt_call["strike"] = Price(120)
    opt = Option(derivatives_config=config_tnt_call,
                 short_name="CALL before expiration OTM",
                 repository=repository,
                 underlying=stock_tnt)
    opt.sod()
    return opt

@pytest.fixture
def option_call_at_expiration_otm(stock_tnt: Security,
                                      config_tnt_call: dict,
                                      repository: Repository) -> Option:
    config_tnt_call["expiration"] = 0
    config_tnt_call["strike"] = Price(180)
    opt = Option(derivatives_config=config_tnt_call,
                 short_name="CALL at expiration OTM",
                 repository=repository,
                 underlying=stock_tnt)
    opt.sod()
    return opt


@pytest.fixture
def option_put_before_expiration_otm(stock_tnt: Security,
                                      config_tnt_put: dict,
                                      repository: Repository) -> Option:
    config_tnt_put["strike"] = Price(168)
    opt = Option(derivatives_config=config_tnt_put,
                 short_name="PUT before expiration OTM",
                 repository=repository,
                 underlying=stock_tnt)
    opt.sod()
    return opt

@pytest.fixture
def option_put_at_expiration_otm(stock_tnt: Security,
                                      config_tnt_put: dict,
                                      repository: Repository) -> Option:
    config_tnt_put["expiration"] = 0
    config_tnt_put["strike"] = Price(165)
    opt = Option(derivatives_config=config_tnt_put,
                 short_name="PUT at expiration OTM",
                 repository=repository,
                 underlying=stock_tnt)
    opt.sod()
    return opt







### agents:

@pytest.fixture
def agent_noise(stock_tnt: Security,
                stock_mit: Security,
                repository: Repository,
                fx_rate_provider: FXRateProvider):
    return NoiseAgent(
        markets=[stock_tnt, stock_mit],
        repository=repository,
        fx_rate_provider=fx_rate_provider,
    )

@pytest.fixture
def agent_hbl(security: Security, repository: Repository):
    return HBLAgent(markets=[security], repository=repository)

@pytest.fixture
def agent_mm(stock_tnt: Security,
             stock_mit: Security,
             repository: Repository,
             fx_rate_provider: FXRateProvider):
    return MMZOHAgent(markets=[stock_tnt, stock_mit],
                      repository=repository,
                      fx_rate_provider=fx_rate_provider)


## agents with open positions

@pytest.fixture
def agent_3k_mit_2k_tnt(agent_noise: Agent,
                        stock_mit: Security,
                        stock_tnt: Security):
    stock_tnt.add_agents([agent_noise])
    stock_mit.add_agents([agent_noise])
    agent_noise.sod()
    agent_noise.position[stock_tnt.asset_id] = 2000
    agent_noise.eod()
    agent_noise.sod()
    agent_noise.position[stock_mit.asset_id] = 3000
    agent_noise.eod()

    agent_noise.record_valuation(current_time=0)

    return agent_noise



### agents with option in portfolio:

@pytest.fixture
def agent_long_call_option_expiration_day(stock_tnt: Security,
                                          stock_mit: Security,
                                          option_call_at_expiration_itm: Option,
                                          repository: Repository,
                                          fx_rate_provider: FXRateProvider):
    agent =  NoiseAgent(markets=[stock_tnt, stock_mit, option_call_at_expiration_itm],
                        repository=repository,
                        fx_rate_provider=fx_rate_provider)
    option_call_at_expiration_itm.add_agents([agent])
    agent.position[option_call_at_expiration_itm.asset_id] = 10

    return agent


@pytest.fixture
def agent_long_put_option_expiration_day(stock_tnt: Security,
                                          stock_mit: Security,
                                          option_put_at_expiration_itm: Option,
                                          repository: Repository,
                                          fx_rate_provider: FXRateProvider):
    agent =  NoiseAgent(markets=[stock_tnt, stock_mit, option_put_at_expiration_itm],
                        repository=repository,
                        fx_rate_provider=fx_rate_provider)
    option_put_at_expiration_itm.add_agents([agent])
    agent.position[option_put_at_expiration_itm.asset_id] = 10

    return agent

@pytest.fixture
def agent_long_call_option_expiration_day_otm(stock_tnt: Security,
                                          stock_mit: Security,
                                          option_call_at_expiration_otm: Option,
                                          repository: Repository,
                                          fx_rate_provider: FXRateProvider):
    agent =  NoiseAgent(markets=[stock_tnt, stock_mit, option_call_at_expiration_otm],
                        repository=repository,
                        fx_rate_provider=fx_rate_provider)
    option_call_at_expiration_otm.add_agents([agent])
    agent.position[option_call_at_expiration_otm.asset_id] = 10

    return agent


@pytest.fixture
def agent_long_put_option_expiration_day_otm(stock_tnt: Security,
                                          stock_mit: Security,
                                          option_put_at_expiration_otm: Option,
                                          repository: Repository,
                                          fx_rate_provider: FXRateProvider):
    agent =  NoiseAgent(markets=[stock_tnt, stock_mit, option_put_at_expiration_otm],
                        repository=repository,
                        fx_rate_provider=fx_rate_provider)
    option_put_at_expiration_otm.add_agents([agent])
    agent.position[option_put_at_expiration_otm.asset_id] = 10

    return agent


# for sod testing after option expire:

@pytest.fixture
def agent_after_expiry_call(agent_long_call_option_expiration_day: Agent,
                              option_call_at_expiration_itm: Option):
    option_call_at_expiration_itm.eod()
    agent_long_call_option_expiration_day.eod()

    return agent_long_call_option_expiration_day

@pytest.fixture
def agent_after_expiry_put(agent_long_put_option_expiration_day: Agent,
                              option_put_at_expiration_itm: Option):
    option_put_at_expiration_itm.eod()
    agent_long_put_option_expiration_day.eod()

    return agent_long_put_option_expiration_day

@pytest.fixture
def agent_after_expiry_call_otm(agent_long_call_option_expiration_day_otm: Agent,
                              option_call_at_expiration_otm: Option):
    option_call_at_expiration_otm.eod()
    agent_long_call_option_expiration_day_otm.eod()

    return agent_long_call_option_expiration_day_otm

@pytest.fixture
def agent_after_expiry_put_otm(agent_long_put_option_expiration_day_otm: Agent,
                              option_put_at_expiration_otm: Option):
    option_put_at_expiration_otm.eod()
    agent_long_put_option_expiration_day_otm.eod()

    return agent_long_put_option_expiration_day_otm

##################
# index

@pytest.fixture
def index_genius(stock_tnt: Security, stock_mit: Security, repository: Repository):
    index = Index(repository=repository, short_name="GeNiUs", name="GeNiUS Index",
                  components={ stock_tnt.asset_id: { "weight": 0.4, "security": stock_tnt },
                               stock_mit.asset_id: { "weight": 0.6, "security": stock_mit }
                               })
    return index


