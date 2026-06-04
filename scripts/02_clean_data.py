from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.data.config import load_config
from src.data.paths import ensure_data_dirs
from src.data.copernicus import clean_copernicus_data
from src.data.gfw_ais import clean_gfw_ais_data
from src.data.gfw_sar import clean_gfw_sar_data


def main() -> None:
    config = load_config()

    ensure_data_dirs(config)

    print("======================================")
    print("AIPS pipeline - 02_clean_data")
    print("======================================")

    clean_copernicus_data(config)
    clean_gfw_ais_data(config)
    clean_gfw_sar_data(config)

    print("======================================")
    print("Data cleaning completed.")
    print("======================================")


if __name__ == "__main__":
    main()
