import re
import sys
from pathlib import Path

# Allow direct execution as ``python tools/last_run_summary.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marketsim.database.inspection import inspect_run_summary as inspect_database


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"

    pattern = re.compile(r"^run_(\d{8}_\d{6})$")

    if not output_dir.is_dir():
        print(f"Output directory not found: {output_dir}")
        sys.exit(1)

    run_dirs = [path for path in output_dir.iterdir() if path.is_dir() and pattern.match(path.name)]

    if not run_dirs:
        print(f"No simulation run directories found in {output_dir}")
        sys.exit(1)

    latest_run = max(run_dirs, key=lambda path: path.name)
    db_path = latest_run / "daedalus.duckdb"

    if not db_path.is_file():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    print(f"Latest run: {latest_run.name}")
    print(f"Database:   {db_path}")

    inspect_database(db_path)


if __name__ == "__main__":
    main()
