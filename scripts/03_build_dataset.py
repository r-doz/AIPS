from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.data.config import load_config
from src.data.paths import ensure_data_dirs
from src.data.build_dataset import build_unique_dataset


def main() -> None:
    config = load_config()

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
