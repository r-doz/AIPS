"""Tune LightGBM on expanding-history, one-step-ahead validation windows."""
import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.multi_year import load_parquet_years
from src.models.data_daily_count import build_daily_design
from src.models.metrics_lgcp import evaluate_metrics
from src.models.daily_count_weighting import recency_weights


def make_candidates(base, search, n_trials, seed):
    """Include the baseline, then sample unique valid parameter combinations."""
    if n_trials < 1:
        raise ValueError('n_trials must be positive (includes baseline)')
    candidates = [dict(base)]
    if n_trials == 1:
        return candidates
    total = int(np.prod([len(v) for v in search.values()]))
    for sample in ParameterSampler(search, n_iter=min(total, n_trials*30), random_state=seed):
        candidate = dict(base, **sample)
        depth = candidate.get('max_depth', -1)
        if depth > 0 and candidate.get('num_leaves', 31) > 2**depth:
            continue
        if 'objective' in candidate and candidate.get('subsample', 1.0) == 1.0:
            candidate['subsample_freq'] = 0
        if candidate not in candidates:
            candidates.append(candidate)
        if len(candidates) == n_trials:
            return candidates
    raise ValueError('Not enough unique valid parameter combinations; reduce n_trials or expand search')


def prepare_windows(df, cfg, windows):
    if cfg.get('forecast_mode', 'one_step_ahead') != 'one_step_ahead':
        raise ValueError('Only one_step_ahead validation is supported')
    prepared, occupied, names = [], set(), set()
    for window in windows:
        name = str(window['name'])
        start, end = pd.Timestamp(window['start']), pd.Timestamp(window['end'])
        expected = pd.date_range(start, end)
        if not len(expected) or name in names or occupied.intersection(expected):
            raise ValueError('Windows need unique names, nonempty ranges, and no overlapping dates')
        names.add(name)
        occupied.update(expected)
        prefix = df[pd.to_datetime(df['date']).dt.normalize() <= end]
        X, y, dates, _ = build_daily_design(prefix, cfg)
        train, test = dates < start, (dates >= start) & (dates <= end)
        if not train.any() or not dates[test].equals(expected):
            raise ValueError(f'{name}: missing training history or validation dates')
        scaler = StandardScaler().fit(X[train])
        active = scaler.var_ > 0
        if not active.any() or not (y[train] > 0).any():
            raise ValueError(f'{name}: need varying features and positive training counts')
        design = scaler.transform(X)[:, active]
        # Explicit feature names avoid sklearn/LightGBM naming warnings.
        design = pd.DataFrame(design, columns=[f'x{i}' for i in range(design.shape[1])])
        prepared.append(dict(name=name, start=str(start.date()), end=str(end.date()),
                             X_train=design.loc[train], y_train=y[train], train_dates=dates[train],
                             X_test=design.loc[test], y_test=y[test], dates=dates[test]))
    if not prepared:
        raise ValueError('At least one validation window is required')
    return prepared


def run(tuning, n_trials=None, report_root=None):
    with open(tuning['base_config']) as f:
        base = yaml.safe_load(f)
    kind = tuning.get('model_kind', 'lightgbm')
    if kind not in ('lightgbm', 'catboost'):
        raise ValueError('model_kind must be lightgbm or catboost')
    objective_key, objective_value = ('objective', 'poisson') if kind == 'lightgbm' else ('loss_function', 'Poisson')
    if base['model'].get(objective_key) not in ([objective_value, 'RMSE'] if kind == 'catboost' else [objective_value]) or objective_key in tuning['search_space']:
        raise ValueError('Keep a supported objective fixed during this search')
    candidate_base = dict(base['model'], half_life_days=(base.get('training') or {}).get('half_life_days'))
    candidates = make_candidates(candidate_base, tuning['search_space'],
                                 n_trials if n_trials is not None else tuning['n_trials'],
                                 tuning.get('seed', 42))
    df = load_parquet_years(base['parquet_path'], base.get('years'))
    windows = prepare_windows(df, base, tuning['windows'])
    out = Path(report_root or tuning.get('report_root', 'reports/tuning/daily_count_lightgbm')) / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'params.yaml').open('w') as f:
        yaml.safe_dump(dict(tuning=tuning, resolved_base_config=base,
                            actual_n_trials=len(candidates), candidates=candidates), f)
    scores, predictions = [], []
    for trial, parameters in enumerate(candidates):
        for window in windows:
            model_params = dict(parameters)
            half_life = model_params.pop('half_life_days')
            if kind == 'catboost':
                from catboost import CatBoostRegressor
                model_params['allow_writing_files'] = False
                model = CatBoostRegressor(**model_params)
            else:
                model = LGBMRegressor(**model_params)
            model.fit(window['X_train'], window['y_train'],
                      sample_weight=recency_weights(window['train_dates'], half_life))
            predicted = (model.predict(window['X_test'], prediction_type=('Exponent' if model_params['loss_function'] == 'Poisson' else 'RawFormulaVal'))
                         if kind == 'catboost' else model.predict(window['X_test']))
            if kind == 'catboost' and model_params['loss_function'] == 'RMSE':
                predicted = np.maximum(predicted, 0.0)
            if not np.isfinite(predicted).all() or (predicted < 0).any():
                raise ValueError(f'Invalid forecasts in trial {trial}, window {window["name"]}')
            metrics = evaluate_metrics(window['y_test'], predicted, window['dates'])
            print(f"  Trial {trial+1}, {window['name']}: MAE={metrics['mae_daily']:.4f}", flush=True)
            scores.append(dict(trial=trial, window=window['name'], start=window['start'],
                               end=window['end'], n_train_days=len(window['y_train']),
                               n_validation_days=len(predicted),
                               **{k: v for k, v in metrics.items() if not k.endswith('_obs')}))
            predictions.extend(dict(trial=trial, window=window['name'], date=str(date.date()),
                                    observed_total=float(obs), predicted_total=float(pred))
                               for date, obs, pred in zip(window['dates'], window['y_test'], predicted))
        frame = pd.DataFrame(scores)
        frame.to_csv(out / 'window_metrics.csv', index=False)
        leaderboard = frame.groupby('trial').agg(
            mean_mae=('mae_daily', 'mean'), mean_rmse=('rmse_daily', 'mean'),
            worst_window_mae=('mae_daily', 'max'), std_window_mae=('mae_daily', 'std'),
            mean_poisson_ll=('mean_ll_daily', 'mean')).reset_index()
        leaderboard = leaderboard.sort_values(['mean_mae', 'mean_rmse', 'trial'])
        leaderboard.to_csv(out / 'leaderboard.csv', index=False)
        pd.DataFrame(predictions).to_csv(out / 'validation_predictions.csv', index=False)
        best_id = int(leaderboard.iloc[0].trial)
        best = deepcopy(base)
        best['model'] = dict(candidates[best_id])
        best['training'] = dict(best.get('training') or {}, half_life_days=best['model'].pop('half_life_days'))
        best['run_name'] = base.get('run_name', 'lightgbm') + '_tuned'
        with (out / 'best_config.yaml').open('w') as f:
            yaml.safe_dump(best, f, sort_keys=False)
        with (out / 'best_parameters.yaml').open('w') as f:
            yaml.safe_dump(dict(trial=best_id, model=best['model'], training=best['training'],
                                mean_validation_mae=float(leaderboard.iloc[0].mean_mae)), f)
        current = frame[frame.trial == trial].mae_daily.mean()
        print(f'Trial {trial+1}/{len(candidates)}: mean MAE={current:.4f}; best={leaderboard.iloc[0].mean_mae:.4f}', flush=True)
    (out / 'README.txt').write_text(
        'Ranking: equal-weight mean window MAE; tie-break by mean window RMSE.\n'
        'Trial 0 is the original configuration. Each window trains on all earlier days.\n'
        'All eight windows are used for model selection, not independent test evaluation.\n'
        'best_config.yaml preserves the base forecast dates: change these to an untouched\n'
        'test window before reporting final performance. The tuner does not refit/deploy a model.\n')
    print(f'Results: {out}\nBest trial: {best_id}', flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/tune_daily_count_lightgbm.yaml')
    parser.add_argument('--n-trials', type=int, help='Override trial count, including baseline')
    parser.add_argument('--report-root', help='Override output root')
    args = parser.parse_args()
    with open(args.config) as f:
        tuning = yaml.safe_load(f)
    run(tuning, args.n_trials, args.report_root)
