# Rebuttal: what the two-stage ablation measures

A reviewer argued that the two-stage ablation of Table 5 underperforms not for lack of joint gradients but
because it (a) fits an untruncated Poisson LGCP to all cell-days, zeros included, (b) drops MC dropout and
(c) is scored without the hurdle probability. All three are true of the paper's two-stage runs
(`reports/seeded_monthly_windows_salinity/11_train_lgcp`):

- its LGCP is bit-identical to the w/o-classifier LGCP (checked on two runs), a Poisson model trained on
  all cell-days;
- its gate is a scikit-learn MLP ([32], α = 1e-4) without dropout, at a fixed τ = 0.3;
- `reports/probabilistic_scores/ablation_scores.py` scores its log-likelihood and CRPS with the Poisson
  predictive of the w/o-classifier runs, so Table 5 shows the same values for both rows.

In addition, the joint ELBO of Red-LGCP is a sum of a classifier term and a count term (plus the KL), with no
shared parameters. So joint training exchanges no gradients between the two parts. Numerically, on one run:
joint ELBO − (classifier part + count part) = 3e-8 relative, and each part's gradient is zero on the other
part's parameters.

## Ladder

Each step adds one change to the previous one (10 seeds × 8 test weeks, same configs and seeds as the paper):

| Step | Model | Source |
|---|---|---|
| S0 | two-stage model of Table 5 | `reports/probabilistic_scores/ablation_base.csv` |
| S1 | + hurdle probability in LL/CRPS: P(0) = 1 − p, P(k) = p E_q[ZTP(k \| λ)], p from the scikit-learn gate (retrained with the run's settings and seed), λ from the Poisson LGCP; point forecasts unchanged | `twostage_score.py` |
| S2 | + Red-LGCP's MC-dropout classifier (trained alone, see S3), p = mean of 100 passes, gate τ + σ applied to E_q[λ] of the Poisson LGCP | `twostage_score.py` |
| S3 | + zero-truncated LGCP fitted to the positive counts, i.e. Red-LGCP with its classifier and count model trained in two separate stages (fresh Adam each, the joint run's settings); outputs of `11c_train_hurdle_lgcp.py` | `twostage_hurdle.py` |
| joint | Red-LGCP of the paper | `ablation_base.csv` ("full") |

S3 was trained on Demetra `lovelace-01` (AMD EPYC 7542, the machine of the paper's runs), job 132766
(24 min, 80 CPUs). Checks (`results/checks.csv`):
- S0's LL and CRPS are reproduced to 4e-5.
- The retrained scikit-learn gate reproduces 100% of S0's gate decisions.
- S0's point forecasts are reproduced within 0.002 MAE; the residual gap comes from using the exact
  E_q[λ] instead of the run's 300-sample Monte Carlo mean.

## Results (`results/tables.md`)

Means over both months:

| Step | MAE | RMSE | Wass. | Acc. | F1 | Log-lik. | CRPS |
|---|---|---|---|---|---|---|---|
| S0 two-stage (Table 5) | 0.295 | 1.095 | 0.150 | 0.893 | 0.605 | −0.523 | 0.236 |
| S1 + hurdle probability | 0.295 | 1.095 | 0.150 | 0.893 | 0.605 | −0.509 | 0.233 |
| S2 + MC-dropout classifier, τ + σ | 0.293 | 1.111 | 0.147 | 0.906 | 0.610 | −0.512 | 0.234 |
| S3 + zero-truncated LGCP | 0.285 | 0.918 | 0.094 | 0.906 | 0.610 | −0.448 | 0.209 |
| Red-LGCP, joint | 0.284 | 0.911 | 0.093 | 0.906 | 0.609 | −0.448 | 0.208 |

Share of the S0 → joint gap closed by each step, on the four metrics where S0 is significantly worse than
the joint model (App. B.5):

| Step | Wass. | Acc. | Log-lik. | CRPS |
|---|---|---|---|---|
| S1 hurdle probability | 0% | 0% | 18% | 11% |
| S2 MC-dropout classifier | 6% | 103% | −4% | −1% |
| S3 zero-truncated LGCP | 92% | 0% | 87% | 89% |
| remaining (joint − S3) | 2% | −3% | −1% | 2% |

Paired tests over the 8 weeks (exact Wilcoxon, uncorrected):
- S1 vs S0: log-likelihood 6/8 weeks (p = 0.078), CRPS 6/8 (p = 0.38).
- S2 vs S1: accuracy 7/8 (p = 0.016).
- S3 vs S2: Wasserstein 8/8 (p = 0.008), log-likelihood 8/8 (p = 0.008), CRPS 7/8 (p = 0.023).
- S3 vs joint: no difference on any metric (all p ≥ 0.195).

Reading: the reviewer's three factors explain the whole gap. The count model (zero-truncated vs Poisson on
all cell-days) accounts for about 90% of the gap in Wasserstein, log-likelihood and CRPS. The MC-dropout gate
accounts for the accuracy gap. Scoring with the hurdle probability accounts for 10–20% of the LL/CRPS gap.
Training the two parts separately gives the same model as training them jointly.

## Files

| Path | Content |
|---|---|
| `results/tables.md` | all tables and checks |
| `results/summary.csv`, `results/paired_tests.csv`, `results/gap_closed.csv`, `results/checks.csv` | as named |
| `results/per_window_per_seed.csv` | every (step, seed, week) value |
| `scores/all_scores.csv` | raw output of `twostage_score.py` (S1–S3, with the checks) |
| `runs/`, `configs/` | the 80 two-stage hurdle runs (without plots) and their configs and logs |
| `logs/` | job and scoring logs |

## Reproduce

```bash
# on Demetra, from ~/projects/AIPS, with reports/rebuttal/svgp_residual/definitive/ a copy of
# reports/seeded_monthly_windows_hurdle (for the per-run configs)
sbatch reports/rebuttal/twostage/twostage_lovelace.slurm
# then, locally (needs reports/probabilistic_scores/ablation_base.csv):
cd scripts/rebuttal && python twostage_summarize.py
```
