from sqlalchemy import (
    Column,
    Float,
    Integer,
    Table,
)

from .base import metadata

position_history = Table(
    "position_history",
    metadata,
    Column("day", Integer),
    Column("time_tick", Integer),
    Column("agent_id", Integer),
    Column("asset_id", Integer),
    Column("position", Integer),
    Column("position_value", Float),
)

portfolio_value_history = Table(
    "portfolio_value_history",
    metadata,
    Column("day", Integer),
    Column("time_tick", Integer),
    Column("agent_id", Integer),
    Column("portfolio_value", Float),
)

cash_history = Table(
    "cash_history",
    metadata,
    Column("day", Integer),
    Column("time_tick", Integer),
    Column("agent_id", Integer),
    Column("cash", Float),
)
