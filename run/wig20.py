"""Prepare a WIG20 scenario from Ikaros Snowflake."""

import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv

from marketsim.scenarios.wig20 import connect, prepare


def main():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-only",
        action="store_true",
    )
    args = parser.parse_args()

    config_path = root / "marketsim/input/wig20.yaml"

    with config_path.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)

    connection = connect()

    try:
        path, cutoff, index_level = prepare(connection, settings, root)
    finally:
        connection.close()

    print(f"Prepared {path} with 21 markets; cutoff={cutoff}; WIG20 close={index_level:.2f}")

    if not args.prepare_only:
        print("Simulation not launched: index normalization and GPW calendar integration pending.")
        print("Review the generated YAML before enabling the full run.")


if __name__ == "__main__":
    main()
