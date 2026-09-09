Run the daily-total model from the project root:

```bash
python scripts/daily_count_nbinarchx.py --config config/daily_count_nbinarchx.yaml
```

The LightGBM Poisson alternative uses the same daily design, train-only
standardization, chronological splits, metrics, and artifact names:

```bash
python scripts/daily_count_lightgbm.py --config config/daily_count_lightgbm.yaml
```

Install the pinned `lightgbm` dependency from `requirements.txt`. Both scripts
use `src/models/daily_count_experiment.py` for preparation and reporting.
LightGBM receives the configured lag covariates; the NB model's recursive
conditional-mean state is specific to NB-INGARCH-X. LightGBM has no NB dispersion
or NB likelihood metric. Its tree count is configured in advance; test labels
are never used for fitting or early stopping. Tune hyperparameters on earlier
training/validation windows. Poisson objective reference:
https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#objective

The default configuration matches `config/15_gnn.yaml` covariates and its
November 3–9, 2025 test window. Adjust both configurations together for a fair
comparison. The target is the sum of cell vessel counts, not a deduplicated
regional vessel count.

NB-INGARCH-X uses a negative-binomial conditional distribution and a log mean
driven by configured observed-count lags, the previous log conditional mean,
and a regularized linear covariate effect. Positive dynamic coefficients sum
to less than 0.98. The initial mean is the training mean; the first maximum-lag
days initialize the recursion and do not contribute to the fitting likelihood.
The `ridge` coefficient penalizes covariate weights in the average training
negative log likelihood. Fit it on earlier validation windows, not test dates.

Every cell's configured covariates are retained in a fixed spatial order.
Standardization uses training data only; training-constant columns are removed.
There is no spatial averaging or additional feature source. Repeated regional
covariates remain repeated across cells, as in the GNN input. This representation
can have many coefficients relative to training days, hence regularization.

Evaluation is strictly **one step ahead** with frozen fitted parameters: after
predicting a day, its observed count can inform the following days. Same-day
environmental covariates follow the existing GNN convention and must be
available at the intended prediction time. This script does not implement a
whole-week forecast made before any of that week's counts are observed.
Random day splits and missing calendar days are rejected. Missing cell targets
are filled with zero, matching the GNN; incomplete spatial panels are rejected.

Each run creates a unique directory under `reports/experiments/daily_count/`:

- `daily_predictions.csv`: exactly `date,predicted_total`, ready for LGCP.
- `metrics.yaml`: GNN-compatible daily MAE, RMSE, trend scores, and
  `mean_ll_daily` (the same **Poisson** point-forecast score used by the GNN).
  `mean_nb_ll_daily` separately scores this model's fitted NB distribution.
- `daily_metrics.csv`: observed/predicted totals and errors for each test day.
- `daily_timeseries.png`: the single observed-versus-predicted plot.
- `params.yaml` and `model.joblib`: configuration, convergence status, model,
  preprocessing, and training history for reproducibility. Only load trusted
  joblib artifacts.

Use the printed predictions path in the LGCP configuration:

```yaml
daily_allocation:
  enabled: true
  totals_csv: reports/experiments/daily_count/<run>/daily_predictions.csv
  gamma: 1.0
```

Keep the existing classifier/conflict settings when testing the classifier
variant. NB uncertainty is scored separately but is not propagated by the
existing point-forecast LGCP allocation. An optimizer convergence warning is
also recorded in the outputs; inspect it before using a run for comparison.

Tune LightGBM across four May and four November 2025 weekly windows:

```bash
python scripts/tune_daily_count_lightgbm.py --config config/tune_daily_count_lightgbm.yaml
```

The default is 60 configurations including the existing baseline (480 fits).
Use `--n-trials 2` for a short check. All inputs are inherited from `base_config`;
only model parameters change. Each window has its own training-only scaler and
expanding training history. No early stopping uses validation targets.
`leaderboard.csv` ranks by average window MAE, then average window RMSE;
`window_metrics.csv` exposes individual window performance. `params.yaml`
records every candidate and the resolved base configuration, and
`validation_predictions.csv` records forecasts from every trial/window.
`best_parameters.yaml` and runnable `best_config.yaml` export the winner.
These eight windows are validation data after tuning. The exported config keeps
its original forecast dates; select an untouched test period for final scores.
The tuner does not modify the input configuration or launch LGCP.

CatBoost-Poisson uses the same input matrix and artifacts:

```bash
python scripts/daily_count_catboost.py --config config/daily_count_catboost.yaml
python scripts/tune_daily_count_catboost.py --config config/tune_daily_count_catboost.yaml
```

CatBoost's Poisson predictions are explicitly exported on the exponential
(count-mean) scale. Its fit uses only training labels and writes no auxiliary
CatBoost directory. The tuning implementation is shared with LightGBM and
uses the same eight weekly windows.

For either tree model, optional recency weighting is configured separately
from estimator parameters:

```yaml
training:
  half_life_days: 180 # null disables weighting
```

Weights halve for every `half_life_days` of age relative to the last training
observation, and are normalized to mean 1. Training history and input features
remain the same; only the fitting weights change. Both tuning configurations
search uniform, 90-, 180-, and 365-day half-lives. Winners export the selected
half-life under `training` in `best_config.yaml`. Candidate records in the tuning
`params.yaml` include this value alongside estimator parameters for traceability.

CatBoost squared-error experiment:

```bash
python scripts/tune_daily_count_catboost.py --config config/tune_daily_count_catboost_rmse.yaml
python scripts/daily_count_catboost.py --config config/daily_count_catboost_rmse.yaml
```

RMSE uses raw predictions clipped at zero; Poisson continues using exponential
predictions. Both are ranked using the same daily MAE and RMSE. The RMSE search
starts from the recovered best Poisson parameters and runs 20 trials. The tuner
now saves predictions and the best configuration after every completed trial.

The GNN can also supervise daily totals without changing its architecture.
Set `loss.mode` in `config/15_gnn.yaml` to `cell_poisson` (historical default),
`combined`, or `daily_only`, then run `python scripts/15_train_gnn.py`.
Combined loss uses cell Poisson divided by max(training mean cell count, 1)
plus `loss.daily_weight` times daily MAE divided by max(training mean daily
count, 1). Daily-only uses just the latter normalized MAE, independent of
`daily_weight`. Scales are fixed from training targets and written to
`loss_params.yaml`. Default cell Poisson remains unnormalized and unchanged.
Daily-only outputs are supervised only through their sum; individual spatial
predictions should not be interpreted as spatially trained forecasts. The
exported `daily_predictions.csv` still supplies LGCP with the predicted total.
MAE training targets a median, so check RMSE and total bias as well as MAE.
