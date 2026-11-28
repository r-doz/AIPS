"""Write the per-run configs of the inducing-point and kernel ablation (rebuttal).

For every variant of config/rebuttal/inducing_kernels.yaml and every definitive per-run config of Red-LGCP
(<definitive>/seed_*/configs/<window>.yaml), writes <out>/configs/<variant>/seed_N/<window>.yaml with the
variant's change and report_root <out>/runs/<variant>/seed_N/runs. Also writes <out>/jobs.txt, the configs
ordered by expected cost (largest M first), for the job scheduler.

Usage: python ik_make_configs.py --definitive <dir> --out <dir> [--variants M1000 no_periodic ...]
"""

from __future__ import annotations

import argparse
import copy
import glob
import os

import yaml

from svgp_residual import ROOT


def variants(spec: dict) -> dict:
    out = {}
    for name, M in spec["inducing_points"].items():
        out[name] = {"M_inducing": int(M)}
    for name, k in spec["kernels"].items():
        out[name] = {"model": {key: k[key] for key in ("spatial_kernels", "temporal_kernels", "spacetime_composition")}}
        if k.get("dtype"):
            out[name]["ik_dtype"] = k["dtype"]
    return out


def apply(cfg: dict, change: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    if "M_inducing" in change:
        cfg["M_inducing"] = change["M_inducing"]
    if "model" in change:
        cfg["model"].update(copy.deepcopy(change["model"]))
    if "ik_dtype" in change:
        cfg["ik_dtype"] = change["ik_dtype"]
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--definitive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="*")
    args = ap.parse_args()
    spec = yaml.safe_load(open(ROOT / "config/rebuttal/inducing_kernels.yaml"))
    var = variants(spec)
    names = args.variants or list(var)
    files = sorted(glob.glob(os.path.join(args.definitive, "seed_*/configs/*.yaml")))
    assert len(files) == 80, len(files)
    jobs = []
    for name in names:
        for f in files:
            seed, win = f.split("/")[-3], os.path.basename(f)[:-5]
            c = apply(yaml.safe_load(open(f)), var[name])
            c["report_root"] = os.path.abspath(os.path.join(args.out, "runs", name, seed, "runs"))
            path = os.path.join(args.out, "configs", name, seed, f"{win}.yaml")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            yaml.safe_dump(c, open(path, "w"), sort_keys=False)
            jobs.append((c["M_inducing"], os.path.abspath(path)))
    jobs.sort(key=lambda j: -j[0])  # largest M first, so the slowest runs do not form the tail
    with open(os.path.join(args.out, "jobs.txt"), "w") as fh:
        fh.write("\n".join(p for _, p in jobs) + "\n")
    print(f"{len(jobs)} configs for {len(names)} variants")


if __name__ == "__main__":
    main()
