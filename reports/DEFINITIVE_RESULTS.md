# Definitive results: AISTATS 2027 submission

**Status: frozen.** The folders listed here hold the experiments behind every number in the submitted
paper (Tables 1, 4, 5, 6, Figures 2 and 3, Appendix B.2). Treat them as read-only. New experiments,
including those for the rebuttal, go in new folders (see [Rules for new experiments](#rules-for-new-experiments)).

- Frozen at commit `cbec6b9e`, tagged `aistats2027-submission`.
- Verified on 2026-11-20: running the ten table scripts listed in the README on a copy of the
  repository regenerated every CSV below from the saved runs (largest absolute difference 6e-12, from
  floating-point noise in the GLM refit and ConvLSTM reload). The regenerated tables match Tables 1, 4,
  5 and 6 of the paper at the printed precision.
- Every run folder is indexed in [`definitive_runs.csv`](definitive_runs.csv): one row per run, with
  `status` = `definitive` (504 runs) or `superseded` (248 runs).

## Experiments

Seeds 1-10 for stochastic models and seed 1 for deterministic ones. Each experiment covers the 8 weekly
test windows (days 1-7, 8-14, 15-21 and 22-28 of May and November 2025).

| Paper | Experiment | Results folder | Config | Script |
|---|---|---|---|---|
| Tab. 1, 4, 5 (full, w/o uncert., w/o redistr., w/o gate), 6; Fig. 2, 3 | Red-LGCP | `reports/seeded_monthly_windows_hurdle` | `config/exp4_salinity_hurdle.yaml` | `scripts/11c_train_hurdle_lgcp.py` |
| Tab. 1, 4; Fig. 3 | LAP | `reports/seeded_monthly_windows/13_evaluate_last_available_poisson` | `config/13_evaluate_last_available_poisson.yaml` | `scripts/13_evaluate_last_available_poisson.py` |
| Tab. 1, 4 | GCMP | `reports/seeded_monthly_windows/12b_global_cell_mean` | `config/12b_global_cell_mean.yaml` | `scripts/12b_global_cell_mean.py` |
| Tab. 1, 4 | Poisson GLM | `reports/seeded_monthly_windows_salinity/14_train_window_poisson_glm` | `config/14_window_poisson_glm_salinity.yaml` | `scripts/14_train_window_poisson_glm.py` |
| Tab. 1, 4; Fig. 3 | GNN | `reports/seeded_monthly_windows_salinity_gnn_cellpoisson/15_train_gnn` | `config/15_gnn_salinity_cellpoisson.yaml` | `scripts/15_train_gnn.py` |
| Tab. 1, 4; Fig. 3 | ConvLSTM | `reports/seeded_monthly_windows_salinity/18_train_convlstm` | `config/18_convlstm_salinity.yaml` | `scripts/18_train_convlstm.py` |
| Tab. 5 | Ablation: two-stage | `reports/seeded_monthly_windows_salinity/11_train_lgcp` | `config/exp4_salinity.yaml` | `scripts/11_train_lgcp.py` |
| Tab. 5 | Ablation: w/o classifier | `reports/seeded_monthly_windows_ablation/no_classifier` | `config/exp4_salinity_no_classifier.yaml` | `scripts/11_train_lgcp.py` |
| Tab. 5 | Ablation: w/o kernel | `reports/seeded_monthly_windows_hurdle_ablation/no_kernel` | `config/exp4_salinity_hurdle_no_kernel.yaml` | `scripts/11c_train_hurdle_lgcp.py` |
| App. B.2 | Hyperparameter re-validation | `reports/paper/hparam_revalidation_2weeks_before_salinity` | `*/selected_hyperparameters.yaml` | see README |

The w/o uncertainty, w/o redistribution and w/o gate ablations are not separate runs. They are computed
from the Red-LGCP runs: `metrics_fixed.yaml`, `hurdle_predictions.csv` and `metrics_nogate.yaml`.

## Tables and figures

| Paper | File |
|---|---|
| Table 1 | `reports/statistical_tests/main_results_table.csv` |
| Table 4, MAE to F1 | `reports/statistical_tests/paired_window_tests_hurdle.csv` |
| Table 4, log-likelihood and CRPS | `reports/probabilistic_scores/paired_tests.csv` |
| Table 5 | `reports/statistical_tests/ablation_table.csv` |
| App. B.5 paired tests | `reports/probabilistic_scores/ablation_paired_tests.csv` |
| Table 6 | `reports/interpretability_hurdle/lgcp_parameters_table.csv` |
| Figure 2 | `notebooks/62_plot_map_design_last_model.ipynb` (Red-LGCP, seed 1, 20 November 2025) |
| Figure 3 | `reports/sensitivity_threshold/threshold_sensitivity.pdf` and `summary.csv` |

Per-window values to compare new experiments against:

- Red-LGCP point metrics: `reports/seeded_monthly_windows_hurdle/per_window_per_seed_metrics.csv`
  (gate rule τ + σ).
- Log-likelihood and CRPS of every method: `reports/probabilistic_scores/per_window_per_seed_scores.csv`.
- Each baseline: `per_window_per_seed_metrics.csv` in its results folder.

## Which file in a run folder is definitive

- **Two runs in one window.** The two-stage, w/o classifier, Poisson GLM and ConvLSTM folders hold two
  timestamped runs per window. The definitive run is the newer one, which `seed_N/monthly_metrics.yaml`
  points to. The older runs are superseded:
  - two-stage: they used the classifier gate from before the re-validation (hidden layers `[32, 16]`,
    α = 1e-2, τ = 0.1);
  - Poisson GLM, ConvLSTM and w/o classifier: they match the definitive runs in May and differ in
    November.
- **Red-LGCP and w/o kernel: use `metrics_higher.yaml`.** This file holds the paper's gate rule,
  τ_{j,t} = τ + σ_{j,t}. In these run folders, `metrics.yaml` holds the `lower` rule (τ − σ), because
  the runs were made with `hurdle.primary_variant: lower`. The config now says `higher`, so new runs
  write the paper's rule to `metrics.yaml`.
- The kernel entries in each run's `params.yaml` differ from the config only in spelling (`rbf` vs
  `RBF`) and key order.

## Not part of the paper

- `config/15_gnn_salinity.yaml` (`loss.mode: daily_mse`) is not used for any reported run. It is only
  the base config of the GNN hyperparameter search (`scripts/51c_…`), which always trains with the
  per-cell Poisson loss, so its `loss.mode` is never read. The reported GNN runs use
  `15_gnn_salinity_cellpoisson.yaml`.
- `reports/paper/hparam_revalidation_2weeks_before_salinity/classifier/selected_hyperparameters.yaml`
  (`[32, 16]`, α = 1e-2, τ = 0.1) is the selection used by the superseded two-stage runs. The definitive
  two-stage runs use the Red-LGCP classifier settings (`[32]`, α = 1e-4, τ = 0.3), as App. B.5 states.
- The superseded runs listed in `definitive_runs.csv`.

## Reproducibility across machines

All runs were made on the Demetra cluster (partition `lovelace`, AMD EPYC 7542) with the venv
`~/projects/AIPS/.venv` (torch 2.14.0+cpu, numpy 2.5.3, scikit-learn 1.9.1, pandas 3.0.6), with one
thread per process. Rerunning there reproduces the saved runs (checked on 2026-11-20 for two GNN runs:
seed 1, 1-7 May, MAE 0.2883; seed 7, 22-28 Nov, MAE 0.3910).

Single GNN and ConvLSTM runs are chaotic: a 1e-7 relative perturbation of the inputs changes a run's
MAE in the second decimal. With other library versions or another CPU, single runs can therefore
differ from the saved ones even with the same seed and deterministic algorithms. On a laptop with
torch 2.9.1, numpy 2.3.3 and pandas 2.3.3, the ConvLSTM runs reproduce exactly but the GNN runs do
not (seed 1, 1-7 May: MAE 0.264 instead of 0.288). Retraining all 80 GNN and 80 ConvLSTM test runs on Demetra reproduces every saved run
(largest MAE gap 6e-8; `reports/rebuttal/reviewer1/results/gnn_platform_check.csv`).

## Rules for new experiments

1. Put new configs in `config/rebuttal/`, starting from a copy of the definitive config and changing only
   what the experiment needs.
2. Write new results to `reports/rebuttal/<experiment>/` through `--output-dir`. The runners refuse to
   write into an existing folder, so the definitive results cannot be overwritten by accident. Never
   copy new runs into a definitive folder: the analysis scripts expect exactly one run per window for
   Red-LGCP, w/o kernel and GNN, and fail otherwise.
3. Changes that only affect the gate (τ, γ, ρ, the uncertainty margin) need no retraining. Recompute
   them from the saved `hurdle_predictions.csv`, as
   `reports/sensitivity_threshold/threshold_sensitivity.py` does.
4. Keep the protocol comparable to the paper: the same 8 windows, seeds 1-10, mean and population std
   over 4 windows × 10 seeds per month, and paired exact Wilcoxon tests over the 8 weeks with Holm
   correction.
5. The table scripts rewrite their CSVs in place. Running them on the definitive folders regenerates
   the same files; new analyses should write to `reports/rebuttal/`.
