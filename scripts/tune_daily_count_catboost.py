"""Tune CatBoost on the same eight validation windows as LightGBM."""
import argparse
import yaml
from tune_daily_count_lightgbm import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/tune_daily_count_catboost.yaml')
    parser.add_argument('--n-trials', type=int)
    parser.add_argument('--report-root')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if config.get('model_kind') != 'catboost':
        raise ValueError('This entry point requires model_kind: catboost')
    run(config, args.n_trials, args.report_root)
