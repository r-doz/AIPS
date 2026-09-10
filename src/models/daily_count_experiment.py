"""Shared preparation, evaluation, and artifacts for daily count experiments."""

from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.preprocessing import StandardScaler
import yaml

from src.data.multi_year import load_parquet_years, years_label
from src.models.daily_count import NBINGARCH
from src.models.daily_count_weighting import recency_weights
from src.models.data_daily_count import build_daily_design
from src.models.data_pp_lgcp import make_day_split_masks
from src.models.metrics_lgcp import evaluate_metrics


def run(cfg, model_kind="nbinarchx"):
    strategy = cfg.get('split_strategy', 'fixed_test_window')
    if strategy not in ('fixed_test_window', 'chronological'):
        raise ValueError('Use fixed_test_window or chronological; random splits leak temporal history')
    if cfg.get('forecast_mode', 'one_step_ahead') != 'one_step_ahead':
        raise ValueError('Only one_step_ahead evaluation is supported')
    df = load_parquet_years(cfg['parquet_path'], cfg.get('years'))
    # Do not use dates beyond the requested evaluation window, even for schema discovery.
    if strategy == 'fixed_test_window':
        df = df[pd.to_datetime(df['date']).dt.normalize() <= pd.Timestamp(cfg['test_end_date'])]
    X, y, dates, names = build_daily_design(df, cfg)
    train, test = make_day_split_masks(
        pd.DataFrame({'date': dates}), split_strategy=strategy,
        train_fraction=cfg.get('train_fraction', 0.9), random_seed=cfg.get('data_seed', 42),
        test_start_date=cfg.get('test_start_date'), test_end_date=cfg.get('test_end_date'))
    n_train = int(train.sum())
    if n_train == 0 or not test.any() or not np.array_equal(np.flatnonzero(train), np.arange(n_train)):
        raise ValueError('Require a nonempty training prefix followed by test days')
    if strategy == 'fixed_test_window':
        expected = pd.date_range(cfg['test_start_date'], cfg['test_end_date'])
        if not dates[test].equals(expected):
            raise ValueError('Data do not cover the complete requested test window')
    scaler = StandardScaler().fit(X[train])
    # Constant columns (including fixed coordinates) contain no estimable temporal effect.
    active = scaler.var_ > 0
    design = scaler.transform(X)[:, active]
    if model_kind == 'nbinarchx':
        label = 'NB-INGARCH-X'
        model = NBINGARCH(**cfg.get('model', {}))
    elif model_kind == 'lightgbm':
        from lightgbm import LGBMRegressor
        parameters = dict(cfg.get('model', {}))
        if parameters.get('objective', 'poisson') not in ('poisson', 'regression'):
            raise ValueError('The daily LightGBM model requires objective: poisson or regression')
        parameters.setdefault('objective', 'poisson')
        label = 'LightGBM ' + parameters['objective']
        model = LGBMRegressor(**parameters)
        if parameters['objective'] == 'poisson' and not (y[train] > 0).any():
            raise ValueError('LightGBM Poisson requires at least one positive training count')
        if not active.any():
            raise ValueError('LightGBM requires at least one varying training covariate')
    elif model_kind == 'catboost':
        from catboost import CatBoostRegressor
        label = 'CatBoost ' + cfg.get('model', {}).get('loss_function', 'Poisson')
        parameters = dict(cfg.get('model', {}))
        if parameters.get('loss_function', 'Poisson') not in ('Poisson', 'RMSE'):
            raise ValueError('The daily CatBoost model requires loss_function: Poisson or RMSE')
        parameters.setdefault('loss_function', 'Poisson')
        parameters['allow_writing_files'] = False
        model = CatBoostRegressor(**parameters)
        if not active.any() or not (y[train] > 0).any():
            raise ValueError('CatBoost requires varying features and positive training counts')
    else:
        raise ValueError(f'Unknown daily model: {model_kind}')
    print(f'Training {label}: {n_train} days, {active.sum()} varying cell covariates', flush=True)
    half_life = (cfg.get('training') or {}).get('half_life_days')
    if model_kind == 'nbinarchx':
        if half_life is not None:
            raise ValueError('Recency weighting is supported for tree models only')
        model.fit(design[train], y[train])
    else:
        model.fit(design[train], y[train], sample_weight=recency_weights(dates[train], half_life))
    if model_kind == 'nbinarchx':
        predicted = model.predict_one_step(design, y)[test]
        fit_info = model.fit_info_
    elif model_kind == 'catboost':
        predicted = np.maximum(model.predict(design[test], prediction_type=('Exponent' if parameters['loss_function'] == 'Poisson' else 'RawFormulaVal')), 0.0)
        fit_info = {'n_estimators_fitted': int(model.tree_count_)}
    else:
        # No test labels are supplied to fitting, prediction, or early stopping.
        predicted = np.maximum(model.predict(design[test]), 0.0)
        fit_info = {'n_estimators_fitted': int(model.n_estimators_)}
    if not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError('Daily predictions must be finite and nonnegative')
    observed, test_dates = y[test], dates[test]
    common = evaluate_metrics(observed, predicted, test_dates)
    metrics = {key: value for key, value in common.items() if not key.endswith('_obs')}
    if model_kind == 'nbinarchx':
        k = model.dispersion_
        metrics['mean_nb_ll_daily'] = float(nbinom.logpmf(observed, k, k/(k+predicted)).mean())
        metrics['optimizer_converged'] = model.fit_info_['converged']
    metrics['n_train_days'] = n_train
    metrics['n_test_days'] = int(test.sum())
    run_name = f"{years_label(cfg.get('years'))}_{cfg.get('run_name', model_kind)}_{datetime.now():%Y%m%d-%H%M%S-%f}"
    out_dir = Path(cfg.get('report_root', 'reports/experiments')) / cfg.get('model_family', 'daily_count') / run_name
    out_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({'date': test_dates, 'predicted_total': predicted}).to_csv(
        out_dir / 'daily_predictions.csv', index=False)
    pd.DataFrame({'date': test_dates, 'observed_total': observed, 'predicted_total': predicted,
                  'absolute_error': np.abs(observed-predicted),
                  'squared_error': (observed-predicted)**2}).to_csv(out_dir / 'daily_metrics.csv', index=False)
    with (out_dir / 'metrics.yaml').open('w') as f:
        yaml.safe_dump(metrics, f)
    with (out_dir / 'params.yaml').open('w') as f:
        yaml.safe_dump(dict(cfg, fit=fit_info, model_kind=model_kind,
                            **({'dispersion': model.dispersion_} if model_kind == 'nbinarchx' else {}),
                            n_active_features=int(active.sum()), run_directory=str(out_dir)), f)
    joblib.dump({'model': model, 'scaler': scaler, 'active_features': active,
                 'feature_names': names, 'train_dates': dates[train],
                 'train_y': y[train], 'train_design': design[train]}, out_dir / 'model.joblib')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test_dates, observed, 'o-', label='Observed')
    ax.plot(test_dates, predicted, 'o-', label=label)
    ax.set(xlabel='Date', ylabel='Daily total vessel count', title='One-step-ahead daily forecast')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / 'daily_timeseries.png', dpi=160)
    plt.close(fig)
    print(yaml.safe_dump(metrics), flush=True)
    print(f'Results: {out_dir}\nLGCP daily_allocation.totals_csv: {out_dir / "daily_predictions.csv"}')
    return out_dir
