"""Train the nbinarchx daily-count model and export forecasts for LGCP."""

import argparse
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.daily_count_experiment import run as run_experiment


def run(cfg):
    return run_experiment(cfg, model_kind="nbinarchx")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/daily_count_nbinarchx.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    run(config)
