# Daily predictor comparison

## Current ranking across eight windows

The best daily predictor evaluated so far is **tuned CatBoost with the RMSE
objective**, followed by **tuned LightGBM with the Poisson objective**, then
**GNN with cell Poisson loss**. These results use the seven selected covariates
and the same eight windows in May and November 2025 (days 1–7, 8–14, 15–21,
and 22–28 in each month).

| Model | Mean daily MAE ↓ | Mean window daily RMSE ↓ |
|---|---:|---:|
| **CatBoost tuned — RMSE objective** | **6.732** | **8.727** |
| LightGBM tuned — Poisson objective | 7.172 | 9.056 |
| GNN — cell Poisson loss, seven covariates | 7.388 | 9.865 |
| GNN — daily MSE loss | 7.740 | 9.631 |
| LightGBM — MSE objective | 8.109 | 10.087 |
| Window GLM — Poisson | 9.194 | 12.207 |
| Last available — one-step ahead | 10.964 | 13.981 |
| Global cell mean | 12.402 | 14.413 |

Tuned CatBoost Poisson achieved mean daily MAE **7.197**, worse than CatBoost
RMSE and slightly worse than LightGBM Poisson. The newer MSE runs for GNN and
LightGBM retained their existing hyperparameters; they were not retuned for
the changed objective. The window GLM used `window_size: 7`, adding five lag
predictors beyond the seven listed covariates.

These are validation results: CatBoost and LightGBM were selected on these
same windows. LGCP has not yet been included in this eight-window comparison.
The earlier GNN MAE of 6.881 below used four additional covariates and is not
the seven-covariate baseline used in this ranking.

Sources: [CatBoost RMSE tuning metrics](../tuning/daily_count_catboost_rmse/20260909-185058-174431/window_metrics.csv),
[LightGBM Poisson tuning metrics](daily_count_lightgbm/20260909-181302-023538/window_metrics.csv),
and [completed eight-window batches](../eight_windows/).

## Earlier tuning work

I tried different models to forcast the daily-count. The aim is to train a model stronger than the GNN and passing it to the LGCP+Model. 

I tried and tuned 
- probabilistic version of lighGMB (MAE 7.17 the best configuration)
- probabilistic version of catboost (MAE 7.19 the best configuration)
- deterministic version of lighGMB
- deterministic version of catboost (6.73 the best configuration)


## CatBoost squared-error search

The Poisson search was already stopped when checked. Recovered 44 completed
trials; the best was trial 36 (zero-based), mean validation MAE 7.1970003.
Its configuration is saved in
`reports/tuning/daily_count_catboost/20260909-183018-927256/recovered_best_config.yaml`.

Started a 20-trial RMSE search using the same eight windows and inputs, seeded
with the recovered Poisson parameters. Configuration:
`config/tune_daily_count_catboost_rmse.yaml`. Progress log:
`reports/daily_tuning/catboost_rmse.log`. Results are under
`reports/tuning/daily_count_catboost_rmse/`.

RMSE predictions use the raw count scale and are clipped at zero. Rankings
still use mean window MAE. The best configuration and validation predictions
are now saved after every completed trial. Scores from these windows are
validation scores, not independent test performance.

## Current GNN comparison across eight windows

Completed all eight GNN runs using config/15_gnn.yaml (300 epochs, seed 0).
Mean window MAE: GNN 6.8805, CatBoost RMSE 6.7316.
Mean window RMSE: GNN 9.4466, CatBoost 8.7269.
CatBoost wins MAE on 3/8 windows; GNN wins on 5/8.

The GNN has four extra lag/rolling covariates; CatBoost was tuned on these
windows. This is a descriptive validation comparison, not an identical-input
independent test. Detailed results: [20260909-192336/report.md](gnn_comparison/20260909-192336/report.md).


# GNN loss variants: eight-window evaluation

300 epochs, seed 0, current seven covariates; combined daily_weight=1.0. Training and daily export code are shared.

| Window | Combined MAE | Daily-only MAE |
|---|---:|---:|
| may_1_7 | 6.458 | 5.686 |
| may_8_14 | 8.304 | 11.232 |
| may_15_21 | 7.700 | 9.676 |
| may_22_28 | 10.517 | 7.034 |
| nov_1_7 | 2.054 | 4.888 |
| nov_8_14 | 6.784 | 5.721 |
| nov_15_21 | 8.085 | 8.150 |
| nov_22_28 | 9.815 | 9.739 |
| **Mean** | **7.465** | **7.766** |

These settings are untuned. The earlier original-GNN mean MAE of 6.881 used four additional covariates, so it is not a controlled loss-only baseline. CatBoost RMSE achieved validation MAE 6.732, but was selected on these same eight windows.

Artifacts: [full report](gnn_loss_comparison/20260909-193341/report.md).
