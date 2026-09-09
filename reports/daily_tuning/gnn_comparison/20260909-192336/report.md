# Current GNN versus tuned CatBoost RMSE

Both use expanding training history and one-step-ahead observed lags. GNN keeps config/15_gnn.yaml architecture, inputs, 300 epochs and seed 0; CPU threads were set to 1. Spatial maps were omitted.

CatBoost was selected on these eight windows, so this is a validation comparison, not an independent test.

Input difference: GNN additionally uses cell_roll_mean_7, daily_total_lag_1, daily_total_lag_7, daily_total_roll_mean_7. This is not an identical-feature comparison.

| Window | GNN MAE | CatBoost MAE | GNN RMSE | CatBoost RMSE |
|---|---:|---:|---:|---:|
| may_1_7 | 8.224 | 8.374 | 12.675 | 10.653 |
| may_8_14 | 7.636 | 8.353 | 9.054 | 10.116 |
| may_15_21 | 5.132 | 6.363 | 6.080 | 8.440 |
| may_22_28 | 8.986 | 6.800 | 13.092 | 8.729 |
| nov_1_7 | 1.530 | 2.981 | 1.653 | 3.977 |
| nov_8_14 | 6.689 | 4.606 | 9.191 | 7.663 |
| nov_15_21 | 8.721 | 9.176 | 12.249 | 11.285 |
| nov_22_28 | 8.127 | 7.200 | 11.580 | 8.953 |
| **Mean** | **6.881** | **6.732** | **9.447** | **8.727** |

CatBoost wins on daily MAE in 3/8 windows.
CatBoost source: reports/tuning/daily_count_catboost_rmse/20260909-185058-174431, trial 3.
