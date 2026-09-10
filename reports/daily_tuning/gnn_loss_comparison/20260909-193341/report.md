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
