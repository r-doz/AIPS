"""Preserve configured cell covariates in a fixed-order daily design matrix."""

import numpy as np
import pandas as pd

from src.models.data_pp_lgcp import add_daily_chl_features, add_lag_features


def build_daily_design(df, cfg):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    target = cfg.get('target_col', 'ais_vessels_count')
    covariates = list(cfg['covariate_cols'])
    if target in covariates:
        raise ValueError('The same-day target cannot be an input covariate')
    if len(set(covariates)) != len(covariates):
        raise ValueError('Duplicate covariate names')
    keys = ['date', 'longitude', 'latitude']
    if df[keys].isna().any().any() or df.duplicated(keys).any():
        raise ValueError('Missing or duplicate date/cell keys')
    # Match the GNN convention for missing cell counts.
    df[target] = pd.to_numeric(df[target], errors='raise').fillna(0).astype(float)
    if not np.isfinite(df[target]).all() or (df[target] < 0).any() or not np.allclose(
            df[target], np.round(df[target]), rtol=0, atol=1e-8):
        raise ValueError('Cell targets must be nonnegative integer counts')
    dates = pd.DatetimeIndex(sorted(df['date'].unique()))
    if len(dates) < 2 or not dates.equals(pd.date_range(dates[0], dates[-1], freq='D')):
        raise ValueError('Daily count models require consecutive calendar days; missing days are not zeros')
    cells = df[['longitude', 'latitude']].drop_duplicates().sort_values(['longitude', 'latitude'])
    if not df.groupby('date').size().eq(len(cells)).all():
        raise ValueError('Unbalanced panel: every day must contain the same cells')
    if 'chl' in df:
        df = add_daily_chl_features(df, chl_col='chl', date_col='date')
    lag_cfg = cfg.get('lag_features') or {}
    if lag_cfg.get('enabled', False):
        if lag_cfg.get('mode', 'observed_past') != 'observed_past':
            raise ValueError('Only observed_past lag features are supported')
        lag_cfg = dict(lag_cfg, target_col=target, date_col='date',
                       cell_cols=['longitude', 'latitude'])
        df, _ = add_lag_features(df, lag_cfg)
    days_in_year = np.where(df['date'].dt.is_leap_year, 366.0, 365.0)
    df['t_norm'] = df['date'].dt.dayofyear / days_in_year
    features = (['longitude', 'latitude'] if cfg.get('include_coords', True) else [])
    features += (['t_norm'] if cfg.get('include_time', True) else [])
    features += covariates
    if len(set(features)) != len(features):
        raise ValueError('Coordinates/time must not also be listed in covariate_cols')
    ordered = df.sort_values(keys)
    values = ordered[features].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Configured covariates contain missing/nonfinite values')
    X = values.reshape(len(dates), len(cells)*len(features))
    y = ordered[target].to_numpy().reshape(len(dates), len(cells)).sum(axis=1)
    names = [f'{feature}[lon={lon},lat={lat}]'
             for lon, lat in cells.itertuples(index=False, name=None) for feature in features]
    return X, y, dates, names
