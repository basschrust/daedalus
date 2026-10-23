import re
import subprocess
import sys
from pathlib import Path


def main() -> None:
    # Project root: parent of the tools directory
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "marketsim" / "output"

    # Match directories named run_YYYYMMDD_HHMMSS
    pattern = re.compile(r"^run_(\d{8}_\d{6})$")

    run_dirs = [
        path
        for path in output_dir.iterdir()
        if path.is_dir() and pattern.match(path.name)
    ]

    if not run_dirs:
        print(f"No simulation run directories found in {output_dir}")
        sys.exit(1)

    # The timestamp format sorts chronologically as a string
    latest_run = max(run_dirs, key=lambda path: path.name)

    db_path = latest_run / "daedalus.duckdb"

    if not db_path.is_file():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    print(f"Inspecting latest run: {latest_run.name}")
    print(f"Database: {db_path}\n")

    # Reuse the existing inspection script
    inspector = Path(__file__).resolve().parent / "duckdb_inspect.py"

    if not inspector.is_file():
        print(f"Inspector script not found: {inspector}")
        sys.exit(1)

    subprocess.run(
        [sys.executable, str(inspector), str(db_path)],
        check=True,
    )


if __name__ == "__main__":
    main()
