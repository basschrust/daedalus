# config.py
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CONFIG = "market_structure.yaml"


class IncludeLoader(yaml.SafeLoader):
    pass


def include_constructor(loader, node):
    relative_path = loader.construct_scalar(node)

    base_path = Path(loader.name).parent
    file_path = base_path / relative_path

    with open(file_path, "r") as f:
        sub_loader = IncludeLoader(f)
        sub_loader.name = str(file_path)
        return sub_loader.get_single_data()


IncludeLoader.add_constructor("!include", include_constructor)


def load_config(filename: str = DEFAULT_CONFIG) -> dict:
    src_file = Path("marketsim/input") / filename

    with open(src_file, "r") as f:
        loader = IncludeLoader(f)
        loader.name = src_file
        config = loader.get_single_data()

    seed = config.get("seed", 67)

    random.seed(seed)
    np.random.seed(seed)

    return config


def create_run_directory() -> Path:
    output_dir = Path("marketsim/output") / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
