from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.build_dataset_flaggear import build_unique_dataset_flaggear  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.data.paths import ensure_data_dirs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data_ebro_vGearAndFlag.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_data_dirs(config)
    build_unique_dataset_flaggear(config)


if __name__ == "__main__":
    main()
