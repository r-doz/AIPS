# Full-week model comparison

Model results on the full test weeks May 5–11 and November 3–9, 2025. Values verified against the saved `metrics.yaml` files in `may_5_11/1_week` and `november_3_9/1_week`. All metrics are rounded to three decimal places.

Global Cell Mean uses expanding mode; Last Available uses one-step-ahead mode. The LGCP variants initialize a trainable period at 14 days with periodic lengthscale fixed at 0.8.

| Week | Method | LL_obs | MAE_obs | RMSE_obs | Wasserstein | Corr_Δ | Accuracy | Recall | Precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| May 5–11 | Last Available Poisson | -3.390 | 0.388 | 1.136 | **0.032** | 0.049 | 0.872 | 0.545 | 0.612 |
| May 5–11 | Global Cell Mean Poisson | -0.634 | 0.396 | 0.995 | 0.322 | -0.667 | 0.283 | **1.000** | 0.183 |
| May 5–11 | Window Poisson GLM | -0.820 | 0.446 | 1.067 | 0.411 | 0.656 | 0.160 | **1.000** | 0.160 |
| May 5–11 | GNN | -0.888 | 0.469 | 1.067 | 0.385 | 0.917 | 0.160 | **1.000** | 0.160 |
| May 5–11 | ConvLSTM | -0.844 | 0.495 | 1.077 | 0.446 | 0.835 | 0.160 | **1.000** | 0.160 |
| May 5–11 | LGCP | **-0.589** | 0.371 | **0.973** | 0.255 | 0.796 | 0.160 | **1.000** | 0.160 |
| May 5–11 | LGCP + Classifier | -1.733 | **0.307** | 0.977 | 0.188 | **0.925** | **0.892** | 0.764 | **0.636** |
| May 5–11 | LGCP + Classifier (redistribution) | -1.144 | 0.336 | 1.037 | 0.136 | 0.796 | 0.641 | 0.873 | 0.293 |
| Nov 3–9 | Last Available Poisson | -2.473 | 0.318 | 0.788 | **0.009** | 0.295 | 0.866 | 0.547 | 0.569 |
| Nov 3–9 | Global Cell Mean Poisson | -0.791 | 0.433 | 0.876 | 0.269 | -0.243 | 0.257 | **1.000** | 0.172 |
| Nov 3–9 | Window Poisson GLM | -0.628 | 0.413 | 0.778 | 0.391 | 0.879 | 0.155 | **1.000** | 0.155 |
| Nov 3–9 | GNN | -0.947 | 0.465 | 0.883 | 0.324 | 0.836 | 0.155 | **1.000** | 0.155 |
| Nov 3–9 | ConvLSTM | -0.637 | 0.400 | 0.794 | 0.289 | 0.922 | 0.155 | **1.000** | 0.155 |
| Nov 3–9 | LGCP | **-0.570** | 0.334 | 0.699 | 0.242 | **0.938** | 0.155 | **1.000** | 0.155 |
| Nov 3–9 | LGCP + Classifier | -1.797 | 0.250 | 0.683 | 0.182 | 0.824 | **0.892** | 0.660 | **0.648** |
| Nov 3–9 | LGCP + Classifier (redistribution) | -1.613 | **0.239** | **0.604** | 0.115 | **0.938** | 0.618 | 0.698 | 0.243 |

**Bold** indicates the best value within each week, selected before rounding (ties within 1e-12 are highlighted). Lower is better for MAE_obs, RMSE_obs and Wasserstein; higher is better for the other metrics. Corr_Δ is the correlation of daily changes.

## Source metrics

- May 5–11 — [ConvLSTM](may_5_11/1_week/2024+2025_may_5_11_convlstm_20260914-164402/metrics.yaml)
- May 5–11 — [Global Cell Mean Poisson](may_5_11/1_week/2024+2025_may_5_11_global_cell_mean_expanding/metrics.yaml)
- May 5–11 — [GNN](may_5_11/1_week/2024+2025_may_5_11_gnn_20260914-164339/metrics.yaml)
- May 5–11 — [Last Available Poisson](may_5_11/1_week/2024+2025_may_5_11_last_available_one_step_ahead_20260914-164436/metrics.yaml)
- May 5–11 — [LGCP + Classifier](may_5_11/1_week/2024+2025_may_5_11_lgcp_2_spatial_kernels_classifier_hard_period_init_14_days_l0.8_trainable_20260914-153544/metrics.yaml)
- May 5–11 — [LGCP + Classifier (redistribution)](may_5_11/1_week/2024+2025_may_5_11_lgcp_2_spatial_kernels_classifier_hard_redistribute_period_init_14_days_l0.8_trainable_20260914-165334/metrics.yaml)
- May 5–11 — [LGCP](may_5_11/1_week/2024+2025_may_5_11_lgcp_2_spatial_kernels_period_init_14_days_l0.8_trainable_20260914-152821/metrics.yaml)
- May 5–11 — [Window Poisson GLM](may_5_11/1_week/2024+2025_may_5_11_window_poisson_glm_window_7_20260914-164434/metrics.yaml)
- Nov 3–9 — [ConvLSTM](november_3_9/1_week/2024+2025_nov_3_9_convlstm_20260914-163959/metrics.yaml)
- Nov 3–9 — [Global Cell Mean Poisson](november_3_9/1_week/2024+2025_nov_3_9_global_cell_mean_expanding/metrics.yaml)
- Nov 3–9 — [GNN](november_3_9/1_week/2024+2025_nov_3_9_gnn_20260914-163933/metrics.yaml)
- Nov 3–9 — [Last Available Poisson](november_3_9/1_week/2024+2025_nov_3_9_last_available_one_step_ahead_20260914-164045/metrics.yaml)
- Nov 3–9 — [LGCP + Classifier](november_3_9/1_week/2024+2025_nov_3_9_lgcp_2_spatial_kernels_classifier_hard_period_init_14_days_l0.8_trainable_20260914-153202/metrics.yaml)
- Nov 3–9 — [LGCP + Classifier (redistribution)](november_3_9/1_week/2024+2025_nov_3_9_lgcp_2_spatial_kernels_classifier_hard_redistribute_period_init_14_days_l0.8_trainable_20260914-164917/metrics.yaml)
- Nov 3–9 — [LGCP](november_3_9/1_week/2024+2025_nov_3_9_lgcp_2_spatial_kernels_period_init_14_days_l0.8_trainable_20260914-152447/metrics.yaml)
- Nov 3–9 — [Window Poisson GLM](november_3_9/1_week/2024+2025_nov_3_9_window_poisson_glm_window_7_20260914-164041/metrics.yaml)

<!-- Original LaTeX table label: tab:results-altweeks-7day -->
