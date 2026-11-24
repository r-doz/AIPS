# Rebuttal, reviewer 1: baseline fairness

Three experiments requested by reviewer 1. Red-LGCP is the definitive model of the paper, unchanged.
All runs were made on Demetra (partition `lovelace`, venv `~/projects/AIPS/.venv`, job 131450, 59 min)
with `r1_demetra.slurm`; the evaluation (`scripts/rebuttal/r1_evaluate.py`) reads the saved predictions.

1. **Validation-tuned activity threshold for the Poisson baselines.** A cell-day is predicted active
   when P(Y > 0) = 1 − e^(−λ) ≥ τ, instead of λ > 0. τ is chosen on the paper's 4 validation weeks,
   separately for accuracy (τ_acc) and for F1 (τ_F1), on validation forecasts of the same models (seeds 1–10).
   Test forecasts are the definitive ones of Table 1.
2. **Hurdle Poisson GLM**: logistic regression for P(Y > 0) plus a zero-truncated Poisson GLM, with the
   Poisson GLM's features; window and both L2 penalties selected with the paper's validation criterion.
3. **Neural baselines with Red-LGCP's lags (t−1, t−7) and a hurdle head**: GNN-hurdle (the GNN already
   had both lags), ConvLSTM-lag (lags added, Poisson head) and ConvLSTM-lag-hurdle. Hyperparameters
   re-selected with the paper's protocol, seeds 1–10.

Hurdle models (2, 3) use Red-LGCP's gate and redistribution (ρ = γ = 1) at their validation τ_F1, with
σ = 0 (no dropout uncertainty). Tests: seed-averaged weekly values, two-sided exact Wilcoxon over the 8
test weeks, Holm correction within each experiment (5 and 4 comparisons).

## Files

| Path | Content |
|---|---|
| `results/tables.md` | all tables (thresholds, metrics, paired tests) |
| `results/thresholds.csv`, `results/threshold_validation_curves.csv` | selected τ and validation curves |
| `results/exp1_thresholded_baselines.csv` | experiment 1, mean and std per month |
| `results/exp23_hurdle_models.csv` | experiments 2 and 3, mean and std per month |
| `results/exp23_point_forecast_variants.csv` | hurdle models with gate only and without gate |
| `results/paired_tests.csv` | paired tests against Red-LGCP |
| `results/per_window_per_seed.csv` | every (model, seed, week) value |
| `results/gnn_platform_check.csv` | GNN retrained on Demetra vs Table 1 (identical) |
| `hparam/` | grids, scores and selected hyperparameters of the new models |
| `predictions/` | validation and test forecasts of every model (parquet) |
| `logs/` | job and script logs |

## Reproduce

```bash
# on Demetra, from ~/projects/AIPS (writes reports/rebuttal/reviewer1/)
sbatch reports/rebuttal/reviewer1/r1_demetra.slurm
# then, where reports/seeded_monthly_windows_hurdle/per_window_per_seed_metrics.csv and
# reports/probabilistic_scores/per_window_per_seed_scores.csv exist:
cd scripts/rebuttal && python r1_evaluate.py
```
