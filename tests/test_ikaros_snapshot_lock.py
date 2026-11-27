"""Tests for the coordinated snapshot used by Daedalus cache building."""

from __future__ import annotations

from pathlib import Path

import pytest

from marketsim.helpers.ikaros_snapshot import temporary_ikaros_snapshot


def test_snapshot_has_source_data_and_is_removed(*, tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    source = tmp_path / "ikaros.duckdb"
    writer = duckdb.connect(str(source))
    try:
        writer.execute("CREATE TABLE observations (symbol VARCHAR, close DECIMAL(24,8))")
        writer.execute("INSERT INTO observations VALUES ('PZU.WA', 67.46)")
    finally:
        writer.close()

    with temporary_ikaros_snapshot(source_db=source, temp_dir=tmp_path) as snapshot:
        assert snapshot != source
        assert snapshot.is_file()
        reader = duckdb.connect(str(snapshot), read_only=True)
        try:
            assert reader.execute("SELECT symbol FROM observations").fetchall() == [("PZU.WA",)]
        finally:
            reader.close()
    assert source.is_file()
    assert not snapshot.exists()


def test_snapshot_refuses_missing_source(*, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        with temporary_ikaros_snapshot(source_db=tmp_path / "missing.duckdb"):
            pytest.fail("Missing Ikaros DB must never generate an empty snapshot")
