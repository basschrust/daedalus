from pathlib import Path
from typing import Any

from marketsim.database.repositories.interface import RepositoryInterface


def create_repository(
    *,
    driver: str = "duckdb",
    output_dir: Path | str = "tmp",
    **config: Any,
) -> RepositoryInterface:

    if driver == "duckdb":
        from marketsim.database.connectors.duckdb_storage import Repository

        return Repository(output_dir=output_dir)

    if driver == "snowflake":
        from marketsim.database.connectors.snowflake_storage import SnowflakeRepository

        # return SnowflakeRepository(**config)
        return SnowflakeRepository(
            output_dir=output_dir,
            **config,
        )

    raise ValueError(f"Unsupported database driver: {driver}")
