"""Regenerate LGCP/GNN spatial maps without training.

Usage: python scripts/replot_spatial.py path/to/experiment [--start-date YYYY-MM-DD]
"""
import argparse
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch
import yaml
from src.models.data_pp_lgcp import prepare_data, compute_meta, make_day_split_masks
from src.visualization.viz_lgcp import plot_daily_interpolated_spatial_maps_separate


def module(filename):
    spec = importlib.util.spec_from_file_location('saved_experiment', ROOT / 'scripts' / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def split(frame, cfg):
    return make_day_split_masks(frame, **{k: cfg[k] for k in (
        'train_fraction', 'data_seed', 'split_strategy', 'test_start_date', 'test_end_date'
    ) if k in cfg and k != 'data_seed'}, random_seed=cfg.get('data_seed', 42))


def standardized(cells, meta):
    return (cells[['longitude', 'latitude']].to_numpy() -
            [meta['lon_mean'], meta['lat_mean']]) / [meta['lon_std'], meta['lat_std']]


def restore(run, cfg):
    if (run / 'spatial_predictions.csv').exists():
        return pd.read_csv(run / 'spatial_predictions.csv', parse_dates=['date'])
    if (run / 'model_state_dict.pt').exists():
        m = module('15_train_gnn.py')
        frame = m.load_parquet_years(cfg['parquet_path'], cfg.get('years'))
        frame['date'] = pd.to_datetime(frame.date)
        frame, features = m.build_node_features(frame, cfg)
        X, Y, dates, cells = m.build_graph_tensors(frame, features, cfg['target_col'])
        train, test = split(pd.DataFrame({'date': dates}), cfg)
        xt = X[test]
        if cfg.get('standardize_features', True):
            scaler = m.StandardScaler().fit(X[train].reshape(-1, X.shape[-1]))
            xt = scaler.transform(xt.reshape(-1, X.shape[-1])).reshape(xt.shape)
        adj = m.normalize_adjacency(m.build_grid_adjacency(
            cells[['longitude', 'latitude']].to_numpy(), connectivity=cfg.get('adjacency', 'queen')))
        model = m.SpatioTemporalGNN(in_dim=X.shape[-1], hidden_dim=int(cfg['hidden_dim']),
                                   num_layers=int(cfg['num_layers']), dropout=float(cfg['dropout']))
        model.load_state_dict(torch.load(run / 'model_state_dict.pt', map_location='cpu', weights_only=True))
        model.eval()
        with torch.no_grad():
            rates = model(torch.tensor(xt, dtype=torch.float32), torch.tensor(adj, dtype=torch.float32)).numpy()
        coords = np.tile(standardized(cells, compute_meta(frame)), (test.sum(), 1))
        dates, observed, rates = np.repeat(dates[test], len(cells)), Y[test].ravel(), np.clip(rates.ravel(), 0, None)
    elif (run / 'model.pt').exists() or (run / 'allocation_predictions.csv').exists():
        prepared = prepare_data(cfg['parquet_path'], train_fraction=cfg['train_fraction'],
            random_seed=cfg['data_seed'], split_strategy=cfg.get('split_strategy', 'random_day'),
            covariate_cols=cfg.get('covariate_cols'), test_start_date=cfg.get('test_start_date'),
            test_end_date=cfg.get('test_end_date'), lag_features=cfg.get('lag_features'), years=cfg.get('years'))
        trc, trv, try_, coords, tecov, observed, _, frame = prepared
        _, test = split(frame, cfg)
        dates = frame.loc[test, 'date'].to_numpy()
        if (run / 'allocation_predictions.csv').exists():
            saved = pd.read_csv(run / 'allocation_predictions.csv')
            if len(saved) != len(dates) or not np.array_equal(pd.to_datetime(saved.date).to_numpy(), dates):
                raise ValueError('Saved allocation rows do not match the reconstructed test data')
            rates = saved['allocated_rate'].to_numpy()
        else:
            if (cfg.get('zero_gate') or {}).get('enabled') or (cfg.get('daily_allocation') or {}).get('enabled'):
                raise ValueError('This older run lacks saved final cell predictions. Its gate/allocation cannot be reproduced without additional saved state; refusing to plot raw predictions as final.')
            m = module('11_train_lgcp.py')
            model = m.SparseLGCP(trc, trv, try_, kernel_config=cfg.get('kernel_config') or m.normalize_kernel_config(cfg),
                                 M_inducing=cfg['M_inducing'], device='cpu', np_seed=cfg.get('np_seed', 0))
            checkpoint = torch.load(run / 'model.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint['model_state'])
            torch.manual_seed(cfg.get('np_seed', 0))
            print('Recomputing Monte Carlo predictions: values may differ slightly from the original run.')
            rates, _, _ = model.predict_rate(coords, tecov, num_samples=cfg['num_pred_samples'])
    else:
        raise ValueError('No spatial_predictions.csv or supported model checkpoint found in this experiment')
    meta = compute_meta(frame)
    # Store the boundary in the same standardized frame used in the original plotting code.
    result = pd.DataFrame({'date': dates, 'x': coords[:, 0], 'y': coords[:, 1],
                           'observed': observed, 'predicted': rates})
    for key in ('lon_mean', 'lon_std', 'lat_mean', 'lat_std'):
        result[key] = meta[key]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--start-date')
    parser.add_argument('--end-date')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--diffusion-strength', type=float)
    parser.add_argument('--radius-cells', type=int)
    parser.add_argument('--display-method', choices=['pchip', 'linear'])
    parser.add_argument('--nx', type=int)
    parser.add_argument('--ny', type=int)
    args = parser.parse_args()
    run = args.experiment.expanduser().resolve()
    output = args.output_dir.expanduser().resolve() if args.output_dir else run / 'plots' / 'spatial_replotted'
    with open(run / 'params.yaml') as f:
        cfg = yaml.safe_load(f)
    os.chdir(ROOT)  # Saved dataset paths are relative to the project root.
    data = restore(run, cfg)
    data['date'] = pd.to_datetime(data.date)
    if args.start_date:
        data = data[data.date >= pd.Timestamp(args.start_date)]
    if args.end_date:
        data = data[data.date <= pd.Timestamp(args.end_date)]
    if data.empty:
        parser.error('No forecast rows in the requested date range')
    settings = dict(cfg.get('spatial_plot') or {})
    for key in ('diffusion_strength', 'radius_cells', 'display_method', 'nx', 'ny'):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    paths = plot_daily_interpolated_spatial_maps_separate(
        data[['x', 'y']].to_numpy(), data.date, data.observed, data.predicted,
        {k: float(data.iloc[0][k]) for k in ('lon_mean', 'lon_std', 'lat_mean', 'lat_std')},
        cfg['gulf_csv_path'], output, **settings)
    print(f'Saved {len(paths)} daily maps to {output}')


if __name__ == '__main__':
    main()
