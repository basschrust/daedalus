from marketsim.database.repositories.interface import RepositoryInterface
from marketsim.database.connectors.duckdb_storage import Repository


def test_duckdb_implements_repository_interface(tmp_path):
    repository = Repository(output_dir=tmp_path)

    assert isinstance(repository, RepositoryInterface)
