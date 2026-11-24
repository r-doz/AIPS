# Rebuttal: SVGP residual variance

Question: how does restoring the diagonal SVGP residual variance r_i = k(x_i, x_i) − [A K_uf]_ii (Hensman et
al. 2015) in Red-LGCP's latent field affect CRPS, log-likelihood and the learned GP contribution?

Red-LGCP uses the subset-of-regressors field f = A u, A = K_fu K_uu⁻¹. The SVGP variant samples
f_i = a_iᵀu + √r_i ε_i in the ELBO's Monte Carlo term and in prediction. Since K_fu and A are already computed,
r_i is one row-wise sum per batch (O(BM)), so the cost stays O(BM²). Measured training time on the same node:
median 14.0 min (SoR) vs 14.1 min (SVGP) per run. Nothing else changes: same configs, seeds, hyperparameters,
classifier, gate (τ + σ, ρ = γ = 1) and evaluation.

## Runs

All on Demetra `turing-01` (Intel Xeon Gold 6140), job 131978 (training, 1 h 02 min, 40 CPUs) and job 131994
(scoring and interpretability, after fixing the support of the predictive pmf, see below). `lovelace`, where
the paper's runs were made, was fully booked.

| Variant | Trained with residual | Residual in prediction | Runs |
|---|---|---|---|
| SoR (paper) | no | no | the 80 definitive runs (`reports/seeded_monthly_windows_hurdle`) |
| SoR + residual at prediction | no | yes | the same 80 runs |
| SoR (retrained) | no | no | 80 runs, the definitive per-run configs retrained on turing-01 |
| SVGP (retrained) | yes | yes | 80 runs, same configs and seeds |
| SVGP, no residual at prediction | yes | no | the same 80 runs |

The main comparison is **SVGP (retrained) vs SoR (retrained)**: same machine, configs and seeds. SoR (retrained)
is statistically indistinguishable from the paper's runs (all paired p ≥ 0.19), so the machine change does
not matter for the aggregates.

Scoring follows `reports/probabilistic_scores/compute_scores.py` (300 latent samples, 100 dropout passes,
seed 1000 + seed). Without the residual, the paper's runs reproduce its log-likelihood and CRPS to 6e-6. The
SoR models with the residual added at prediction have heavy-tailed λ samples (up to a few hundred). On those
rows (≤ 4 per run) the support 0..min(400, ·) of compute_scores.py truncated more than 1e-4 of the mass, so
they are recomputed on a support that holds all of it (`svgp_score.hurdle_scores`). The first scoring pass,
which asserted on those rows, is kept in `scores_failed_cap400/`.

## Results (from `results/tables.md`)

Mean over the 8 test weeks and 10 seeds; paired tests over the 8 seed-averaged weeks, exact Wilcoxon,
uncorrected. With Holm over the 7 metrics, no difference between SVGP and SoR (retrained) is significant.

| Metric | SoR (retrained) | SVGP (retrained) | Weeks SVGP better | p |
|---|---|---|---|---|
| Log-lik. ↑ | −0.4485 | −0.4459 | 6/8 | 0.078 |
| CRPS ↓ | 0.2079 | 0.2091 | 2/8 | 0.078 |
| MAE ↓ | 0.284 | 0.286 | 3/8 | 0.195 |
| RMSE ↓ | 0.909 | 0.924 | 0/8 | 0.008 |
| Wasserstein ↓ | 0.093 | 0.095 | 3/8 | 0.195 |
| Accuracy ↑ | 0.905 | 0.906 | 3/8 | 0.844 |
| F1 ↑ | 0.608 | 0.610 | 3/8 | 0.844 |

Adding the residual to the paper's model at prediction only, without retraining, makes CRPS (1/8 weeks better,
p = 0.016), MAE and RMSE (0/8, p = 0.008) worse: r ≈ 0.86 inflates E[λ] by about e^(r/2) ≈ 1.5.

GP contribution (May / November means):

| | SoR (retrained) | SVGP (retrained) |
|---|---|---|
| prior variance k(x, x) | 1.03 / 0.89 | 0.092 / 0.071 |
| share of k(x, x) not captured by the inducing points | 0.84 / 0.86 | 0.63 / 0.68 |
| variance of the posterior-mean field A m (training inputs) | 0.142 / 0.147 | 0.050 / 0.049 |
| latent variance at test (a'Sa, + r for SVGP) | 0.077 / 0.055 | 0.082 / 0.066 |
| static / periodic / short-term share of A m | 0.80 / 0.16 / 0.007, 0.85 / 0.11 / 0.005 | 0.83 / 0.15 / 0.002, 0.88 / 0.09 / 0.002 |
| learned period (days) | 13.96 ± 0.13, 13.97 ± 0.21 | 14.02 ± 0.02, 14.05 ± 0.04 |
| Spearman, static map vs mean positive count | 0.77 / 0.87 | 0.70 / 0.79 |

Covariate effects (multiplicative, per sd) are unchanged within about 0.01, except weekend (0.74 → 0.81 in May,
0.82 → 0.86 in Nov.) and holiday (0.20 → 0.24, 0.22 → 0.25). The seed-to-seed spread of the environmental
effects (chl, thetao, salinity) is roughly halved.

Reading: under SoR the variance that the 300 inducing points cannot represent (mostly the local components,
ℓ = 0.01 in space and ≈ 1.4 days in time) is dropped from the ELBO, so the kernel variances can grow at no cost.
With the residual in the ELBO, the model shrinks all four kernel variances (k(x, x) about 12× smaller) until
total latent variance at test is about what SoR had. Predictive scores therefore barely move. The GP
mean field explains about a third of the variance it did under SoR. Its structure (static ≫ periodic ≫
short-term, a 14-day period, covariate effects) is the same; the period and the environmental effects vary
less across seeds.

## Files

| Path | Content |
|---|---|
| `results/tables.md` | all tables |
| `results/predictive_summary.csv`, `results/paired_tests.csv` | predictive metrics and paired tests |
| `results/gp_contribution.csv` | kernel variances, variance diagnostics, decomposition, covariate effects |
| `results/per_window_per_seed.csv` | every (variant, seed, week) value |
| `scores/all_scores.csv` | raw output of `svgp_score_all.py` (480 rows) |
| `interpretability_sor/`, `interpretability_svgp/` | Table 6 analysis (`interpret_hurdle.py`) of the retrained runs |
| `runs_sor/`, `runs_svgp/`, `configs_sor/`, `configs_svgp/` | retrained runs (without plots) and their configs and logs |
| `logs/` | job logs |

## Reproduce

```bash
# on Demetra, from ~/projects/AIPS, with reports/rebuttal/svgp_residual/definitive/ a copy of
# reports/seeded_monthly_windows_hurdle and, if reports/interpretability_hurdle/interpret_hurdle.py is
# missing, a verbatim copy of it in reports/rebuttal/svgp_residual/
sbatch reports/rebuttal/svgp_residual/svgp_turing.slurm
# then, locally (needs the paper's reports/probabilistic_scores and reports/interpretability_hurdle):
cd scripts/rebuttal && python svgp_summarize.py
```

Code: `scripts/rebuttal/svgp_residual.py` (model and training wrapper around `scripts/11c_train_hurdle_lgcp.py`),
`svgp_score.py`, `svgp_score_all.py`, `svgp_interpret.py`, `svgp_summarize.py`.
