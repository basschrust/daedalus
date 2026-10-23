from __future__ import annotations

from loguru import logger
from pathlib import Path

from marketsim.database.repositories.factory import create_repository
from marketsim.loggers.basic import terminal
from marketsim.fundamental.mean_reverting import GaussianMeanReverting
from marketsim.market import Security, Option, Index, Future, FXRateProvider, Price
from marketsim.agent import Agent, WashTradingAgent, MomentumAgent, SpoofingAgent, NoiseAgent
from marketsim.agent import ZIAgentNotInformed, MMZOHAgent, HBLAgent, OptionMMZOHAgent
from marketsim.agent.washtrading import WashTradingPool


class Simulator:
    def __init__(self,
                 *,
                 sim_time: int,
                 days: int = 1,
                 lam: float = 0.1,
                 mean: float = 100.0,
                 r: float = .6,
                 shock_var=10,
                 markets: dict = {},
                 lob_plot_interval: int = 10,
                 agents: dict|None=None,
                 output_dir: Path|str = "tmp",
                 database: dict | None = None,
                 fx_rates: dict | None = None,
                 ):
        self.logger = logger.bind()
        self.logger.info("Initializing simulation with parameters in market_structure.yaml ...")
        self.output_dir = output_dir
        # self.repository = Repository(output_dir=self.output_dir)
        database = database or {"driver": "duckdb"}
        driver = database.get("driver", "duckdb")

        db_config = database.get(driver, {})

        self.repository = create_repository(
            driver=driver,
            output_dir=self.output_dir,
            **db_config,
        )

        self.logger.info(f"Database backend: {driver}")

        self.sim_time = sim_time # steps per day
        self.days = days
        self.lam = lam # lambda (activity factor)
        self.mean = mean
        self.r = r
        self.shock_var = shock_var # change probability of fundamental (market consensus) value

        self.current_time = 0 # TODO: needed in Market, here probably not?
        self.markets = {} # [] # each market serves single security, keys are asset_ids
        self.market_map = {} # { yaml_id: asset_id }
        self.asset_names_map = {} # { asset_id: {"name": name, "short_name": short_name, "ticker": ticker }

        self.agents = {} # boys are back in town! agents here instead of markets, as one agents serves many markets
        self.lob_plot_interval = lob_plot_interval
        self.last_progress = -1
        self.bar_length = 40
        self.day = 0 # day counter for SoD and EoD procedures

        self.fx_rate_provider = FXRateProvider()
        for day, rates in (fx_rates or {}).items():
            for pair, rate in rates.items():
                base, quote = pair.split("/")

                self.fx_rate_provider.set_rate(
                    day=int(day),
                    base=base,
                    quote=quote,
                    rate=rate,
                )

        for m_key, m_conf in markets.items():
            # TODO: do we need this fundamental at all?
            fundamental = GaussianMeanReverting(mean=self.mean, final_time=self.sim_time, r=self.r,
                                                shock_var=self.shock_var)

            # TODO: how to handle derivatives here?
            instrument_class = m_conf.get("instrument_class", "stock")
            if instrument_class == "option":
                # let's rock with first option here!
                # terminal.write(str(m_conf))
                # terminal.write(str(self.markets))
                derivatives_config = m_conf.get("derivatives_config")
                underlying = self.markets.get(self.market_map.get(m_conf.get("derivatives_config").get("underlying")))
                strikes = m_conf.get("derivatives_config").get("strikes", [])
                if strikes:
                    #series
                    for strike in strikes:
                        derivatives_config["strike"] = strike
                        market = Option(market_type=m_conf.get("market_type"), name=m_conf.get("name"),
                                   derivatives_config=derivatives_config
                                    , underlying=underlying, repository=self.repository,
                                        short_name=m_conf.get("short_name"), output_dir=self.output_dir)
                        self.market_map[m_key] = market.asset_id
                        self.markets[market.asset_id] = market
                        self.asset_names_map[market.asset_id] = {"name": market.name,
                                                                 # "short_name": market.short_name,
                                                                 # "ticker": market.ticker,
                                                                 }
                        for group_name, agent_group in m_conf.get("agent_groups", {}).items():
                            self.add_agent_group(agent_group=agent_group, markets=[market], group_name=group_name)
                else:
                    #single strike
                    market = Option(market_type=m_conf.get("market_type"), name=m_conf.get("name"),
                                   derivatives_config=derivatives_config
                                    , underlying=underlying, repository=self.repository,
                                    short_name=m_conf.get("short_name"), output_dir=self.output_dir)
                    self.market_map[m_key] = market.asset_id
                    self.markets[market.asset_id] = market
                    self.asset_names_map[market.asset_id] = {"name": market.name,
                                                             # "short_name": market.short_name,
                                                             # "ticker": market.ticker,
                                                             }
                    for group_name, agent_group in m_conf.get("agent_groups", {}).items():
                        self.add_agent_group(agent_group=agent_group, markets=[market], group_name=group_name)
            elif instrument_class == "index":
                components = {}
                for comp in m_conf.get("components", []):
                    components[self.markets[self.market_map[comp["security"]]].asset_id] = {
                            "weight": comp["weight"],
                            "security": self.markets[self.market_map[comp["security"]]] }
                market = Index(name=m_conf.get("name"), short_name=m_conf.get("short_name"),
                               repository=self.repository, output_dir=self.output_dir,
                               components=components)
                self.market_map[m_key] = market.asset_id
                self.markets[market.asset_id] = market
                self.asset_names_map[market.asset_id] = { "name": market.name,
                                                         }
            elif instrument_class == "future":
                market = Future(name=m_conf.get("name"), short_name=m_conf.get("short_name"),
                               repository=self.repository, output_dir=self.output_dir,
                                underlying=self.markets[self.market_map[m_conf.get("underlying")]],
                                expiration=m_conf.get("expiration"),
                                multiplier=m_conf.get("multiplier"),
                                margin_rate=m_conf.get("margin_rate"))
                self.market_map[m_key] = market.asset_id
                self.markets[market.asset_id] = market
                self.asset_names_map[market.asset_id] = {"name": market.name,
                                                         }
            elif instrument_class == "stock":
                market = Security(
                    market_type=m_conf.get("market_type"),
                    name=m_conf.get("name"),
                    repository=self.repository,
                    short_name=m_conf.get("short_name"),
                    output_dir=self.output_dir,
                    reference_price=m_conf.get("reference_price"),
                    currency=m_conf.get("currency", "PLN"),
                )
                self.market_map[m_key] = market.asset_id
                self.markets[market.asset_id] = market
                self.asset_names_map[market.asset_id] = {"name": market.name,
                                                         # "short_name": market.short_name,
                                                         # "ticker": market.ticker,
                                                         }
                for group_name, agent_group in m_conf.get("agent_groups", {}).items():
                    self.add_agent_group(agent_group=agent_group, markets=[market], group_name=group_name)
            else:
                raise ValueError(f"Unknown instrument_class: {instrument_class}")


            # TODO: resolve agent dependencies
            for relationship in m_conf.get("agent_dependencies", []):
                if relationship["type"] == "wash_trading_pool":
                    # terminal.write(f"Adding washtrading relationship:")
                    buy_pool = relationship["buy_pool"]
                    sell_pool = relationship["sell_pool"]

                    # TODO: now set the pool hook in the agents...
                    # check all agents with group name "buy_pool" or "sell_pool" ?
                    buy_pool_agents = []
                    sell_pool_agents = []
                    for agent in market.agents.values():
                        # TODO: simplify it. Currently "group_name" is the yaml header of the agent group
                        # try using self.agent_map similarly to market_map
                        if agent.group == buy_pool:
                            # terminal.write(f"\nFound buy agent {agent.agent_id}")
                            buy_pool_agents.append(agent)
                        elif agent.group == sell_pool:
                            # terminal.write(f"\nFound sell agent {agent.agent_id}")
                            sell_pool_agents.append(agent)

                    pool = WashTradingPool(buy_pool=buy_pool_agents, sell_pool=sell_pool_agents)
                    for agent in buy_pool_agents:
                        # terminal.write(f"\nsetting pool for agent  {agent.agent_id}...")
                        agent.set_wt_pool(wt_pool=pool)
                    for agent in sell_pool_agents:
                        # terminal.write(f"\nsetting pool for agent  {agent.agent_id}...")
                        agent.set_wt_pool(wt_pool=pool)

        if agents is not None:
            for a_key, agent_gr_def in agents.items():
                self.create_agents(agent_group=agent_gr_def, group_name=a_key)

        # self.all_markets = self.markets

        return

    #########################  __init__ ends here   ###################



    def add_agent_group(self, *, agent_group:dict, markets: list[Security], group_name: str) -> None:
        for i in range(agent_group["number"]):
            # let's make it in case/ series of ifs to avoid security breach (if used the class name as code directly)
            # ZI agents:
            configuration = agent_group.get("config", {})
            if agent_group["agent_class"] == "ZIAgentNotInformed":
                agent = ZIAgentNotInformed(markets=markets, configuration=configuration,
                                            repository=self.repository, group=group_name)
                self.add_agents([agent])

            # Noise agents:
            if agent_group["agent_class"] == "NoiseAgent":
                agent = NoiseAgent(markets=markets, configuration=configuration,
                                   repository=self.repository, group=group_name,
                                   output_dir=self.output_dir)
                self.add_agents([agent])

            # MMs:
            if agent_group["agent_class"] == "MMZOHAgent":
                agent = MMZOHAgent(markets=markets, configuration=configuration, repository=self.repository,
                                   output_dir=self.output_dir)
                self.add_agents([agent])

            # HBL (Heuristic Belief)
            if agent_group["agent_class"] == "HBLAgent":
                agent = HBLAgent(markets=markets, configuration=configuration, repository=self.repository,
                                   output_dir=self.output_dir)
                self.add_agents([agent])

            # spoofers: (to trick HBL Agents)
            if agent_group["agent_class"] == "SpoofingAgent":
                agent = SpoofingAgent(markets=markets, configuration=configuration, repository=self.repository,
                                   output_dir=self.output_dir)
                self.add_agents([agent])

            # washtrading agents (tricking MMs)
            if agent_group["agent_class"] == "WashTradingAgent":
                agent = WashTradingAgent(markets=markets, group=group_name, configuration=configuration,
                                         repository=self.repository,
                                   output_dir=self.output_dir)
                self.add_agents([agent])
                # those will need the relationship...

            # momentum
            if agent_group["agent_class"] == "MomentumAgent":
                agent = MomentumAgent(markets=markets, configuration=configuration, repository=self.repository,
                                   output_dir=self.output_dir)
                self.add_agents([agent])

            ########## Derivatives agents, complicated ones :)  ###############

            ## MM, simple delta hedger
            if agent_group["agent_class"] == "OptionMMZOHAgent":
                # TODO - what with underlying?
                agent = OptionMMZOHAgent(markets=markets, #market_map=self.market_map, #underlying_market=market.underlying,
                                         configuration=agent_group.get("config", None),
                                         repository=self.repository,
                                        output_dir=self.output_dir)
                self.add_agents([agent])

    def create_agents(self, *, agent_group: dict, group_name: str) -> None:
        # processing the agents defined at the end of yaml
        if agent_group.get("markets", "ALL") == "ALL":
            markets = list(self.markets.values())
        else:
            markets = [self.markets[self.market_map[k]] for k in agent_group["markets"]]

        self.add_agent_group(agent_group=agent_group, markets=markets, group_name=group_name)

    def add_agents(self, agents: list[Agent]) -> None:
        for agent in agents:
            self.logger.info(f"Adding agent {str(agent)} to the simulation")
            agent.fx_rate_provider = self.fx_rate_provider

            config = agent.configuration or {}
            agent.base_currency = config.get(
                "base_currency", "PLN"
            ).upper()

            for currency, amount in config.get(
                    "initial_cash", {}
            ).items():
                agent.credit_cash(
                    currency=currency,
                    amount=Price(amount),
                )

            # Existing registration logic follows
            self.agents[agent.get_id()] = agent

            for asset_id, market in agent.markets.items(): # TODO: this will serve the multimarket agents soon
                market.add_agents([agent]) # TODO: check performance

    def step(self) -> None:
        # TODO: changing the architecture - first agents, then markets
        for agent_id, agent in self.agents.items():
            agent.take_action(current_time=self.current_time)  #
            # now agents decide which markets to enter on their own

        for market_key, market in self.markets.items():
            if market.status != "active":
                self.logger.info(f"Market {market_key} is inactive")
                continue

            # plot the LOB
            if self.current_time > 0 and self.current_time % self.lob_plot_interval == 0:
                if market.instrument_class != "index":
                    # plot the LOB only for tradable intruments
                    market.plot_lob(self.current_time)

            buy_matched_queue_length = 0 if market.order_book is None else len(market.order_book.buy_matched.heap)
            sell_matched_queue_length = 0 if market.order_book is None else len(market.order_book.sell_matched.heap)
            market.logger.info(f"Starting orders execution, matched queues should be empty here: {buy_matched_queue_length}"
                  f" {sell_matched_queue_length}")

            # TODO: we should be asynchronous, redesign this part! No returns!!! :)
            new_orders_matched = market.step(current_time=self.current_time)

            market.logger.info(f"Starting to clear out orders.")
            # initiate market prices instance for the case of no trades: - moved to market.step

            for matched_order in new_orders_matched:
                market.logger.info(f"Matched order {str(matched_order)}")
                market.record_trade(matched_order=matched_order)
            market.logger.info(f'After clearing the market the last traded price is: {market.last_traded_price}')
            market.logger.info(f'And the spread: {market.order_book.buy_unmatched.peek()} {market.order_book.sell_unmatched.peek()}')

        # update value of each agent after each market has been cleared:
        for agent_id, agent in self.agents.items():
            agent.record_valuation(current_time=self.current_time) #, price=market.last_traded_price)

        self.current_time += 1


    def end_sim(self) -> None:
        """ End the simulation and print summary """
        self.logger.info(f"\n\nSimulation ended. time: {self.current_time}")
        terminal.write(f"Preparing agents summary...\n")
        self.last_progress = -1
        self.show_progress_bar(step=0, total=len(self.agents), step_name="Agents")
        agent_steps = 0
        for agent_key, agent in self.agents.items():
            agent.show_summary()

            self.show_progress_bar(step=agent_steps, total=len(self.agents), step_name="Agents")
            agent_steps += 1

        terminal.write(f"\n\nPreparing markets summary...\n")
        self.last_progress = -1
        self.show_progress_bar(step=0, total=len(self.markets), step_name="Markets")
        market_steps = 0
        for market_key, market in self.markets.items():
            market.show_summary()

            self.show_progress_bar(step=market_steps, total=len(self.markets), step_name="Markets")
            market_steps += 1

    def show_progress_bar(self, step: int, total: int | None = None, step_name: str = "Steps") -> None:
        if total is None:
            total = self.sim_time
        progress = (step + 1) / total
        percentage = int(progress * 100)

        if percentage != self.last_progress:
            filled = int(self.bar_length * progress)
            bar = "█" * filled + "░" * (self.bar_length - filled)

            terminal.write(
                f"\r|{bar}| {percentage:3d}%   {step_name} completed: {step + 1}/{total}"
            )
            terminal.flush()

            self.last_progress = percentage

    def sod(self, day: int):
        self.logger.info(f"\nStarting Start-of-Day procedure day: {day} ...")

        self.current_time = 0

        # drop expired securities
        # expired_securities = [m_id for m_id, market in self.markets.items() if market.status == "expired"]
        # or not "active" ?
        # for m_id in expired_securities:
        #     self.markets.pop(m_id)

        for market_key, market in self.markets.items():
            market.sod()

        for agent_id, agent in self.agents.items():
            agent.sod()

        self.logger.info(f"\nStart-of-Day day: {day} procedure completed.")

    def eod(self, day: int):
        self.logger.info(f"\nStarting End-of-Day procedure day: {day} ...")

        for market_key, market in self.markets.items():
            market.eod()

        # markets first, then agents - so that the derivatives are already settled and cleared and exercised
        for agent_id, agent in self.agents.items():
            agent.eod()

        self.logger.info(f"\nEnd-of-Day day: {day} procedure completed.")

    def run(self) -> None:
        terminal.write("\nStarting simulation...\n")

        for day in range(self.days):
            terminal.write(f"\nStarting day {str(day+1)} out of {self.days}\n")
            self.sod(day=day) # start of day

            for step in range(self.sim_time):
                self.logger.info(f"Simulation step start: {step}.", end='')
                self.step()
                self.show_progress_bar(step)

            self.eod(day=day)

        terminal.write("\nSimulation complete.")
        terminal.write("\nPreparing summary...\n")
        self.end_sim()
        terminal.write("\nSummary complete.")
