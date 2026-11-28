"""Train one run of the inducing-point and kernel ablation (rebuttal).

Runs scripts/11c_train_hurdle_lgcp.py unchanged on a config written by ik_make_configs.py, with two additions:
  - model.spacetime_composition ('product', as in the paper, or 'sum') is passed to the kernel, since the
    config loader does not forward it (the model then always uses the product);
  - ik_dtype: float64 (near-singular kernels, see config/rebuttal/inducing_kernels.yaml) makes the run use
    double precision: the script's torch.set_default_dtype(float32) is turned into float64.

Usage: python ik_train.py --config <per-run config>   (from the repository root)
"""

from __future__ import annotations

import argparse

import torch

from twostage_hurdle import load_train_module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    mod = load_train_module()
    cfg = mod.base.load_config(args.config)
    cfg["kernel_config"]["spacetime_composition"] = cfg["model"].get("spacetime_composition", "product")
    if cfg.get("ik_dtype", "float32") == "float64":
        set_dtype = torch.set_default_dtype
        torch.set_default_dtype = lambda _: set_dtype(torch.float64)
    mod.main(cfg)


if __name__ == "__main__":
    main()
