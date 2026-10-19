from __future__ import annotations

import duckdb
import pandas as pd
from sympy.core import parameters
from pathlib import Path
from typing import TYPE_CHECKING

from marketsim.input import config
from marketsim.loggers.basic import terminal

if TYPE_CHECKING:
    from marketsim.agent import Agent
    from marketsim.market import Security, Index, Future

# TODO: create some base class with abstract methods, use instances depending on the configuration
class Repository:
    def __init__(self, output_dir: Path|str = "tmp"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.localdb = output_dir / "daedalus.duckdb"
        self.connection = duckdb.connect(str(self.localdb))
        self.prepare_tables()


    def prepare_tables(self) -> None:
        ###### static tables - per simulation  ################

        # securities
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS securities (
                    asset_id INTEGER,
                    instrument_class STRING,
                    market_type STRING,
                    name STRING,
                    short_name STRING,
                    eod_status STRING,
                    reference_price DOUBLE
                )
            """)   # TODO: configuration STRING, for options and other derivatives

        # indices
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS indices (
                    index_id INTEGER,
                    name STRING,
                    short_name STRING
                )
            """)

        # components of each index
        # TODO: watch out, this may change through time!
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS index_components (
                    index_id INTEGER,
                    asset_id INTEGER,
                    weight DOUBLE,
                )
            """)

        # future contracts on indices / underlying_type stock/index
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS futures (
                    contract_id INTEGER,
                    underlying_type STRING,
                    underlying_id INTEGER,
                    expiration_day INTEGER,
                    multiplier DOUBLE,
                    margin_rate DOUBLE,
                    name STRING,
                    short_name STRING
                )
            """)

        # derivatives
        self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS options (
                            asset_id INTEGER,
                            underlying_id INTEGER,
                            configuration STRING,
                            expiration_day INTEGER,
                            strike DOUBLE,
                    )
                """)

        # agents - subjects of the simulation
        self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS agents (
                            agent_id INTEGER,
                            name STRING,
                            group_name STRING,
                            configuration STRING,
                    )
                """)

        ##############  dynamic tables - per day mostly   ######################

        # End-of-Tick position (portfolio) of each agent, lowercase columns please!
        self.connection.execute("""
                CREATE TABLE IF NOT EXISTS agent_position_history (
                    day       INTEGER,
                    time_tick INTEGER,
                    agent_id  INTEGER,
                    asset_id  INTEGER,
                    position  INTEGER
                )
            """)
        # deleted column: position_value DOUBLE

        self.connection.execute("""
                CREATE TABLE IF NOT EXISTS agent_futures_position_history (
                    day       INTEGER,
                    time_tick INTEGER,
                    agent_id  INTEGER,
                    asset_id  INTEGER,
                    position  INTEGER
                )
            """)

        self.connection.execute("""
                CREATE TABLE IF NOT EXISTS agent_futures_value_history (
                    day       INTEGER,
                    time_tick INTEGER,
                    agent_id  INTEGER,
                    asset_id  INTEGER,
                    position  INTEGER,
                    avg_initial_price DOUBLE,
                    collateral DOUBLE,
                    unsettled_pnl DOUBLE,
                    settlement_price DOUBLE,
                )
            """)

        # EoD position (portfolio) of each agent, lowercase columns please!
        self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS agent_eod_positions (
                        day       INTEGER,
                        agent_id  INTEGER,
                        asset_id  INTEGER,
                        position  INTEGER
                    )
                """)
        # deleted column: position_value DOUBLE

        self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS agent_eod_futures_position (
                        day       INTEGER,
                        agent_id  INTEGER,
                        asset_id  INTEGER,
                        position  INTEGER
                    )
                """)

        self.connection.execute("""
                CREATE TABLE IF NOT EXISTS agent_eod_futures_value (
                    day       INTEGER,
                    agent_id  INTEGER,
                    asset_id  INTEGER,
                    position  INTEGER,
                    avg_initial_price DOUBLE,
                    collateral DOUBLE,
                    unsettled_pnl DOUBLE,
                    settlement_price DOUBLE,
                )
            """)

        # portfolio value history
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS agent_portfolio_value_history (
                day       INTEGER,
                time_tick INTEGER,
                agent_id  INTEGER,
                portfolio_value DOUBLE
            )
        """)

        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS agent_eod_portfolio_value (
                day       INTEGER,
                agent_id  INTEGER,
                portfolio_value DOUBLE
            )
        """)

        # cash history
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS agent_cash_history (
                day       INTEGER,
                time_tick INTEGER,
                agent_id  INTEGER,
                cash DOUBLE
            )
        """)

        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS agent_eod_cash (
                day       INTEGER,
                agent_id  INTEGER,
                cash DOUBLE
            )
        """)

        # Market situation
        # traded prices # TODO: add volume_by_value (?)
        self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS traded_prices (
                        day       INTEGER,
                        time_tick INTEGER,
                        asset_id  INTEGER,
                        open      DOUBLE,
                        high      DOUBLE,
                        low       DOUBLE,
                        close     DOUBLE,
                        volume    DOUBLE,
                        theoretical DOUBLE,
                    )
                """)

        # TODO: eod_prices / eod_traded_prices
        self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS eod_prices (
                        day       INTEGER,
                        asset_id  INTEGER,
                        open      DOUBLE,
                        high      DOUBLE,
                        low       DOUBLE,
                        close     DOUBLE,
                        volume    DOUBLE,
                        theoretical DOUBLE,
                    )
                """)

        # index values
        self.connection.execute("""
                            CREATE TABLE IF NOT EXISTS index_values (
                                day       INTEGER,
                                time_tick INTEGER,
                                index_id  INTEGER,
                                open      DOUBLE,
                                high      DOUBLE,
                                low       DOUBLE,
                                close     DOUBLE,
                            )
                        """)

        # eod index
        self.connection.execute("""
                            CREATE TABLE IF NOT EXISTS eod_index_values (
                                day       INTEGER,
                                index_id  INTEGER,
                                open      DOUBLE,
                                high      DOUBLE,
                                low       DOUBLE,
                                close     DOUBLE,
                            )
                        """)

        # futures intraday
        self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS future_rates (
                            contract_id INTEGER,
                            day INTEGER,
                            time_tick INTEGER,
                            open DOUBLE,
                            high DOUBLE,
                            low DOUBLE,
                            close DOUBLE,
                            volume DOUBLE,
                            lop INTEGER,
                            theoretical DOUBLE
                        )
                    """)

        # eod future rates
        self.connection.execute("""
                            CREATE TABLE IF NOT EXISTS eod_future_rates (
                                    contract_id INTEGER,
                                    day INTEGER,
                                    open DOUBLE,
                                    high DOUBLE,
                                    low DOUBLE,
                                    close DOUBLE,
                                    volume DOUBLE,
                                    lop INTEGER,
                                    theoretical DOUBLE
                            )
                    """)

        ##### granular tables - may contain significant volumes of data ######################

        # orders
        self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        day INTEGER,
                        price DOUBLE,
                        order_type INTEGER,
                        quantity INTEGER,
                        agent_id INTEGER,
                        time INTEGER,
                        order_id INTEGER,
                        asset_id INTEGER,
                        executed_price DOUBLE,
                        executed_mode STRING,
                        parent_id INTEGER,
                        matched_with INTEGER,
                        valid_until INTEGER,
                    )
                """)


        # trades / matched orders
        self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        day INTEGER,
                        asset_id INTEGER,
                        matched_with INTEGER,
                        executed_mode STRING,
                        executed_time INTEGER,
                        executed_price DOUBLE,
                        order_id INTEGER,
                        order_side INTEGER,
                        executed_volume INTEGER,
                        cash DOUBLE,
                        phase STRING,
                    )
                """)

        self.connection.execute("""
                CREATE TABLE IF NOT EXISTS option_expiration (
                    day INTEGER,
                    asset_id INTEGER,
                    exercise_price DOUBLE,
                    premium DOUBLE)
                """)

    ############# methods for making data persistent ###############################

    def save_security(self, security: Security) -> None:
        self.connection.execute("""
            INSERT INTO securities (
                asset_id,
                instrument_class,
                market_type,
                name,
                eod_status,
                reference_price
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, parameters=[
            security.asset_id,
            security.instrument_class,
            security.market_type,
            security.name,
            security.eod_status,
            security.reference_price,
        ])

    def save_index(self, index: Index) -> None:
        self.connection.execute("""
            INSERT INTO indices (
                index_id,
                name,
                short_name
            ) VALUES (?, ?, ?)
        """, parameters=[index.index_id, index.name, index.short_name])

    def save_index_components(self, components_df: pd.DataFrame) -> None:
        self.connection.register("components_df", components_df)
        self.connection.execute("""
            INSERT INTO index_components (
                index_id, asset_id, weight
            ) 
            SELECT index_id, asset_id, weight
            FROM components_df
        """)
        self.connection.unregister("components_df")

    def save_future(self, future: Future) -> None:
        self.connection.execute(query="""
            INSERT INTO futures (contract_id, underlying_type, underlying_id,
                                expiration_day, multiplier, name, short_name)
            VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, parameters=[future.asset_id,
                                     "index",
                                     future.underlying.asset_id,
                                     future.expiration,
                                     future.multiplier,
                                     future.name,
                                     future.short_name
                                     ])

    ###########################################################################################

    # the players :)
    def save_agent(self, agent: Agent) -> None:
        self.connection.execute("""
            INSERT INTO agents (
                agent_id,
                name,
                group_name,
                configuration
            )
            VALUES (?, ?, ?, ?)
        """, parameters=[agent.agent_id, agent.name, agent.group, agent.configuration])

    def save_position_history(self, position_history_df: pd.DataFrame) -> None:
        self.connection.register("position_history_df", position_history_df)

        self.connection.execute("""
            INSERT INTO agent_position_history (day, time_tick, agent_id, asset_id, position)
            SELECT day, time_tick, agent_id, asset_id, position
            FROM position_history_df
        """)
        # deleted column: position_value

        self.connection.unregister("position_history_df")

    def save_futures_position_history(self, position_history_df: pd.DataFrame) -> None:
        # the same as position_history, but for the futures or derivatives as value calculation differs
        self.connection.register("position_history_df", position_history_df)

        self.connection.execute("""
            INSERT INTO agent_futures_position_history (day, time_tick, agent_id, asset_id, position)
            SELECT day, time_tick, agent_id, asset_id, position
            FROM position_history_df
        """)
        # deleted column: position_value

        self.connection.unregister("position_history_df")

    def save_futures_value_history(self, futures_value_df: pd.DataFrame) -> None:
        self.connection.register("futures_value_df", futures_value_df)

        self.connection.execute("""
            INSERT INTO agent_futures_value_history (day, time_tick, agent_id, asset_id,
                                                    position, avg_initial_price, collateral, unsettled_pnl)
            SELECT day, time_tick, agent_id, asset_id, position, avg_initial_price,
                collateral, unsettled_pnl
            FROM futures_value_df
        """)

        self.connection.unregister("futures_value_df")

    def save_eod_position(self, eod_position_df: pd.DataFrame) -> None:
        self.connection.register("eod_position_df", eod_position_df)

        self.connection.execute("""
            INSERT INTO agent_eod_positions (day, agent_id, asset_id, position)
            SELECT day, agent_id, asset_id, position
            FROM eod_position_df
        """)
        # deleted column: position_value

        self.connection.unregister("eod_position_df")

    def save_eod_futures_position(self, eod_position_df: pd.DataFrame) -> None:
        self.connection.register("eod_position_df", eod_position_df)

        self.connection.execute("""
            INSERT INTO agent_eod_futures_position (day, agent_id, asset_id, position)
            SELECT day, agent_id, asset_id, position
            FROM eod_position_df
        """)
        # deleted column: position_value

        self.connection.unregister("eod_position_df")

    def save_eod_futures_value(self, eod_futures_value_df: pd.DataFrame) -> None:
        self.connection.register("eod_futures_value_df", eod_futures_value_df)

        self.connection.execute("""
            INSERT INTO agent_eod_futures_value (day, agent_id, asset_id,
                                                    position, avg_initial_price, collateral, unsettled_pnl)
            SELECT day, agent_id, asset_id, position, avg_initial_price,
                collateral, unsettled_pnl
            FROM eod_futures_value_df
        """)

        self.connection.unregister("eod_futures_value_df")


    def save_portfolio_value_history(self, portfolio_value_history_df: pd.DataFrame) -> None:
        self.connection.register("portfolio_value_history_df", portfolio_value_history_df)
        self.connection.execute("""
                INSERT INTO agent_portfolio_value_history (day, time_tick, agent_id, portfolio_value)
                SELECT day, time_tick, agent_id, portfolio_value
                FROM portfolio_value_history_df
        """)

        self.connection.unregister("portfolio_value_history_df")

    def save_eod_portfolio_value(self, eod_portfolio_value_df: pd.DataFrame) -> None:
        self.connection.register("eod_portfolio_value_df", eod_portfolio_value_df)
        self.connection.execute("""
                INSERT INTO agent_eod_portfolio_value (day, agent_id, portfolio_value)
                SELECT day, agent_id, portfolio_value
                FROM eod_portfolio_value_df
        """)

        self.connection.unregister("eod_portfolio_value_df")

    def save_cash_history(self, cash_history_df: pd.DataFrame) -> None:
        self.connection.register("cash_history_df", cash_history_df)
        self.connection.execute("""
            INSERT INTO agent_cash_history (day, time_tick, agent_id, cash)
            SELECT day, time_tick, agent_id, cash
            FROM cash_history_df
        """)

        self.connection.unregister("cash_history_df")

    def save_eod_cash(self, eod_cash_df: pd.DataFrame) -> None:
        self.connection.register("eod_cash_df", eod_cash_df)
        self.connection.execute("""
            INSERT INTO agent_eod_cash (day, agent_id, cash)
            SELECT day, agent_id, cash
            FROM eod_cash_df
        """)

        self.connection.unregister("eod_cash_df")

    def save_traded_prices(self, traded_prices_df: pd.DataFrame) -> None:
        # TODO: add version for derivatives, including theoretical - needed?
        self.connection.register("traded_prices_df", traded_prices_df)

        if "theoretical" in traded_prices_df.columns:
            self.connection.execute("""
                            INSERT INTO traded_prices
                            SELECT day, time_tick, asset_id, open, high, low, close, volume, theoretical
                            FROM traded_prices_df      
                        """)
        else:
            self.connection.execute("""
                INSERT INTO traded_prices
                SELECT day, time_tick, asset_id, open, high, low, close, volume, NULL
                FROM traded_prices_df      
            """)

        self.connection.unregister("traded_prices_df")

    def save_eod_prices(self, eod_prices_df: pd.DataFrame) -> None:
        # TODO: add version for derivatives, including theoretical - needed?
        self.connection.register("eod_prices_df", eod_prices_df)

        if "theoretical" in eod_prices_df.columns:
            self.connection.execute("""
                            INSERT INTO eod_prices
                            SELECT day, asset_id, open, high, low, close, volume, theoretical
                            FROM eod_prices_df      
                        """)
        else:
            self.connection.execute("""
                INSERT INTO eod_prices
                SELECT day, asset_id, open, high, low, close, volume, NULL
                FROM eod_prices_df      
            """)

        self.connection.unregister("eod_prices_df")


    def save_orders(self, orders_df: pd.DataFrame) -> None:
        # with duckdb.connect(self.localdb) as conn:
        self.connection.register("orders_df", orders_df)
        self.connection.execute("""
            INSERT INTO orders
            SELECT  day,
                    price,
                    order_type,
                    quantity,
                    agent_id,
                    time,
                    order_id,
                    asset_id,
                    executed_price,
                    executed_mode,
                    parent_id,
                    matched_with,
                    valid_until
            FROM orders_df
        """)
        self.connection.unregister("orders_df")


    def save_trades(self, trades_df: pd.DataFrame) -> None:
        # matched_orders aka trades
        # with duckdb.connect(self.localdb) as conn:
        self.connection.register("trades_df", trades_df)
        # terminal.write(f"Columns: {str(trades_df.columns())}")
        self.connection.execute("""
            INSERT INTO trades (day, asset_id, executed_time, matched_with, executed_mode, executed_price,
                    order_id, order_side, executed_volume, cash, phase)
            SELECT day,
                    asset_id,
                    executed_time,
                    matched_with,
                    executed_mode,
                    executed_price,
                    order_id,
                    order_side,
                    executed_volume,
                    cash,
                    phase,
            FROM trades_df
        """)
        self.connection.unregister("trades_df")

    def save_option_expiration(self, option_expiration_df: pd.DataFrame) -> None:
        self.connection.register("option_expiration_df", option_expiration_df)

        self.connection.execute("""
        INSERT INTO option_expiration (day, asset_id, exercise_price, premium)
        SELECT day, asset_id, exercise_price, premium
        FROM option_expiration_df
        """)

        self.connection.unregister("option_expiration_df")

    ######################################################################################
    #########   methods for data extraction   ############################################
    ######################################################################################

    # for multiday plotting
    def get_eod_positions(self, agent_id: int) -> pd.DataFrame:
        return self.connection.execute(query="""
                SELECT day, asset_id, position
                FROM agent_eod_positions
                WHERE agent_id = ?
            """, parameters=[agent_id]).fetchdf()

    def get_eod_futures_positions(self, agent_id: int) -> pd.DataFrame:
        return self.connection.execute(query="""
                SELECT day, asset_id, position
                FROM agent_eod_futures_position
                WHERE agent_id = ?
            """, parameters=[agent_id]).fetchdf()

    def get_eod_prices(self, asset_id: int) -> pd.DataFrame:
        return self.connection.execute(query="""
                SELECT day, open, high, low, close, volume, theoretical
                FROM eod_prices
                WHERE asset_id = ?
            """, parameters=[asset_id]).fetchdf()

    def get_eod_index_values(self, index_id: int) -> pd.DataFrame:
        return self.connection.execute(query="""
                SELECT day, open, high, low, close
                FROM eod_index_values
                WHERE index_id = ?
            """, parameters=[index_id]).fetchdf()

    def get_eod_future_rates(self, contract_id: int) -> pd.DataFrame:
        return self.connection.execute(query="""
                SELECT day, open, high, low, close, volume, theoretical, lop
                FROM eod_future_rates
                WHERE contract_id = ?
            """, parameters=[contract_id]).fetchdf()

    def get_eod_cash(self, agent_id: int) -> pd.DataFrame:
        return self.connection.execute(query="""
                SELECT day, cash
                FROM agent_eod_cash
                WHERE agent_id = ?
            """, parameters=[agent_id]).fetchdf()

    def get_eod_portfolio_values(self, agent_id: int) -> pd.DataFrame:
        return self.connection.execute(query="""
                SELECT day, portfolio_value
                FROM agent_eod_portfolio_value
                WHERE agent_id = ?
            """, parameters=[agent_id]).fetchdf()

    #################################################################################################
    ##          calculations in the DB
    #################################################################################################

    def calculate_index_values(self, day: int) -> None:
        # calculate intraday index values for all indices
        self.connection.execute(query="""
                INSERT INTO
                    index_values (day, time_tick, index_id, open, high, low, close)
                SELECT tp.day, tp.time_tick, ic.index_id,
                        sum(ic.weight * tp."open" ),
                        sum(ic.weight * tp."close" ),
                        sum(ic.weight * tp."close" ),
                        sum(ic.weight * tp."close" ),
                FROM index_components ic
                LEFT JOIN traded_prices tp
                    ON ic.asset_id = tp.asset_id
                WHERE tp.day = ?
                GROUP BY tp.day, tp.time_tick, ic.index_id
            """, parameters=[day])


    # combine eod_prices into eod_index_values:
    def calculate_eod_index_values(self, day: int) -> pd.DataFrame:
        return self.connection.execute(query="""
                INSERT INTO eod_index_values (day, index_id, open, high, low, close)
                SELECT iv.day, iv.index_id, 
                    FIRST(iv."open" ORDER BY iv.time_tick) AS open,
                    MAX(iv."close" ) as high,
                    MIN(iv."close" ) as low,
                    LAST(iv."close" ORDER BY iv.time_tick) AS close,
                FROM main.index_values iv 
                WHERE iv."day" = ?
                GROUP BY iv.day, iv.index_id""", parameters=[day]).fetchdf()

