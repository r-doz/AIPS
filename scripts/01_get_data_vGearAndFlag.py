from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.config import load_config  # noqa: E402
from src.data.copernicus import download_copernicus_data  # noqa: E402
from src.data.gfw_ais_flaggear import download_gfw_ais_flaggear_data  # noqa: E402
from src.data.gfw_sar import download_gfw_sar_data  # noqa: E402
from src.data.paths import ensure_data_dirs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data_ebro_vGearAndFlag.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_data_dirs(config)

    download_copernicus_data(config)
    download_gfw_ais_flaggear_data(config)
    download_gfw_sar_data(config)


if __name__ == "__main__":
    main()
