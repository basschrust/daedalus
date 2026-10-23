from sqlalchemy import insert

from marketsim.database.models.securities import securities


class Repository:
    def __init__(self, engine):
        self.engine = engine

    def save_security(self, security) -> None:
        statement = insert(securities).values(
            asset_id=security.asset_id,
            instrument_class=security.instrument_class,
            market_type=security.market_type,
            name=security.name,
            eod_status=security.eod_status,
            reference_price=security.reference_price,
        )

        with self.engine.begin() as conn:
            conn.execute(statement)
