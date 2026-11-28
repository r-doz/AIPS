import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.data.config import load_config
from src.data.copernicus import download_copernicus_data
from src.data.gfw_ais import download_gfw_ais_data
from src.data.gfw_sar import download_gfw_sar_data
from src.data.paths import ensure_data_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    ensure_data_dirs(config)

    print("======================================")
    print("AIPS pipeline - 01_get_data")
    print("======================================")

    download_copernicus_data(config)
    download_gfw_ais_data(config)
    download_gfw_sar_data(config)

    print("======================================")
    print("Raw data download completed.")
    print("======================================")


if __name__ == "__main__":
    main()
