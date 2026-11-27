"""Create a temporary, consistent Ikaros DuckDB copy for Daedalus cache building.

No dependency on Ikaros's Python environment.  The shared lock is released
before the caller starts reading the temporary copy.
"""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import duckdb

from marketsim.helpers.duckdb_file_lock import duckdb_access_lock


@contextmanager
def temporary_ikaros_snapshot(
    *,
    source_db: Path | str,
    temp_dir: Path | str | None = None,
    timeout_seconds: float = 300.0,
) -> Iterator[Path]:
    """Yield a self-contained snapshot that stays available until context exit.

    Never copy an active WAL by hand: open source read-write, CHECKPOINT,
    *close it*, then copy the database file under the shared OS mutex.
    """
    source = Path(source_db).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Ikaros database not found: {source}")
    scratch = Path(temp_dir).expanduser().resolve() if temp_dir is not None else None
    if scratch is not None:
        scratch.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="daedalus_ikaros_", dir=scratch) as directory:
        snapshot = Path(directory) / "ikaros.duckdb"
        with duckdb_access_lock(
            database_path=source,
            timeout_seconds=timeout_seconds,
        ):
            # Cooperating Ikaros connections are all closed now.  External
            # DBeaver instances can still block this: they do not use our mutex.
            conn = duckdb.connect(str(source), read_only=False)
            try:
                conn.execute("CHECKPOINT")
            finally:
                conn.close()
            shutil.copy2(source, snapshot)

        if snapshot.stat().st_size == 0:
            raise ValueError(f"Empty Ikaros database snapshot: {snapshot}")
        # Check that the copied database opens, with no side effects on source.
        conn = duckdb.connect(str(snapshot), read_only=True)
        try:
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()
        yield snapshot
