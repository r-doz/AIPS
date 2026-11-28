import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.data.config import load_config
from src.data.build_dataset import build_unique_dataset
from src.data.paths import ensure_data_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    ensure_data_dirs(config)

    print("======================================")
    print("AIPS pipeline - 03_build_dataset")
    print("======================================")

    build_unique_dataset(config)

    print("======================================")
    print("Final dataset build completed.")
    print("======================================")


if __name__ == "__main__":
    main()
