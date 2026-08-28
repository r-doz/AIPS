"""Run controlled LGCP experiments over temporal RBF lengthscales.

Every experiment is loaded independently from the same base YAML. Only the
first temporal RBF lengthscale and the run name are changed.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "11_train_lgcp.py"

_spec = importlib.util.spec_from_file_location("train_lgcp", TRAIN_SCRIPT)
train_lgcp = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(train_lgcp)


def parse_lengthscales(raw_values: list[str]) -> list[float]:
    values = [float(value) for value in raw_values]
    if not values or any(value <= 0 for value in values):
        raise ValueError("At least one positive lengthscale is required.")
    if len(values) != len(set(values)):
        raise ValueError("Lengthscales must be unique.")
    return values


def main(base_config: str, lengthscales: list[float]) -> None:
    for index, lengthscale in enumerate(lengthscales, start=1):
        cfg = train_lgcp.load_config(base_config)
        temporal_kernels = cfg["model"]["temporal_kernels"]
        rbf_index = next(
            (
                i
                for i, kernel in enumerate(temporal_kernels)
                if str(kernel.get("type", "")).strip().lower() == "rbf"
            ),
            None,
        )
        if rbf_index is None:
            raise ValueError("The base config has no temporal RBF kernel.")

        temporal_kernels[rbf_index]["hyperparameters"]["lengthscale"] = lengthscale
        # load_config() normalizes kernels before this override, so refresh the
        # model-facing kernel_config after changing the human-facing model block.
        cfg["kernel_config"] = train_lgcp.normalize_kernel_config(cfg)
        cfg["run_name"] = f"{cfg['run_name']}_temporal_rbf_l{lengthscale:g}"

        print("\n" + "=" * 80)
        print(
            f"Temporal-lengthscale experiment {index}/{len(lengthscales)}: "
            f"lengthscale={lengthscale:g}"
        )
        print("=" * 80)
        train_lgcp.main(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="config/train_basic_lgcp_multi_kernel.yaml",
        help="Base LGCP YAML; it is read afresh for every experiment.",
    )
    parser.add_argument(
        "--lengthscales",
        nargs="+",
        default=["0.003", "0.001", "0.15", "0.2", "0.25"],
        help="Positive temporal RBF lengthscales to run.",
    )
    args = parser.parse_args()
    main(args.base_config, parse_lengthscales(args.lengthscales))
