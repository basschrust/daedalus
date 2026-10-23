from sqlalchemy import (
    Column,
    Float,
    Integer,
    String,
    Table,
    Text,
)

from .base import metadata

securities = Table(
    "securities",
    metadata,
    Column("asset_id", Integer, primary_key=True),
    Column("instrument_class", String),
    Column("market_type", String),
    Column("name", String),
    Column("eod_status", String),
    Column("reference_price", Float),
    # Could eventually add a JSON configuration column.
)

options = Table(
    "options",
    metadata,
    Column("asset_id", Integer, primary_key=True),
    Column("underlying_id", Integer),
    Column("configuration", Text),
    Column("expiration_day", Integer),
    Column("strike", Float),
)
