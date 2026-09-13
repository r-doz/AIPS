"""Run LGCP training sequentially for experiment configs 1 through 4."""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "11_train_lgcp.py"


def main() -> None:
    # range(x, y): x is inclusive, y is exclusive
    for experiment_number in range(1, 9):
        config = PROJECT_ROOT / "config" / f"exp{experiment_number}.yaml"
        print(f"\nRunning {config.name}...", flush=True)
        subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT), "--config", str(config)],
            check=True,
            cwd=PROJECT_ROOT,
        )


if __name__ == "__main__":
    main()
